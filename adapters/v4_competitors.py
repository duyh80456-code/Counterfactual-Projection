"""Pinned competitor calls and the NCHW port of dopamine's ReDo recycler."""
from contextlib import contextmanager
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace, MethodType
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'PyCIL': 'f3509b8ca3f20660ce4aa13f19d5283de81b4b35',
    'loss-of-plasticity': 'a6b79580d85f3025bdb601566d3627c5f489f13b',
    'upgd': 'b75e90ad4b09c28971ac9dbb902a8fd86709b28c',
    'C-Flat': 'f57a557523136834bdc28a4902cde9842859b2a8',
    'warm_start': '896176b2cf89f259bcde8e241a8987e6019741d0',
    'dopamine': '5873f5494ee0c2d7c016d0ab2ad530354fec59d0',
}


def source_module(name, path):
    path = ROOT/'third_party'/path
    if not path.is_file():
        raise RuntimeError(f'fetch first: bash scripts/fetch_third_party.sh ({path})')
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def cflat_module():
    path = str(ROOT/'third_party/C-Flat')
    if path not in sys.path: sys.path.append(path)
    return source_module('v4_cflat_icarl', 'C-Flat/models/icarl.py')


def branch_pairs(net):
    return [(name, m) for name, m in net.named_modules()
            if hasattr(m, 'conv_a') and hasattr(m, 'conv_b')]


def dormant_score(activation):
    # dopamine/weight_recyclers.py:estimate_neuron_score, NHWC -> NCHW.
    axes = (0, *range(2, activation.ndim))
    score = activation.abs().mean(dim=axes)
    return score / (score.mean() + 1e-9)


@torch.no_grad()
def recycle(block, activation, optimizer, tau):
    """NeuronRecycler.recycle_dead_neurons, create_masks and reset_momentum.

    Copyright 2023 ReDo authors; Apache-2.0. The upstream ResNet rule only
    recycles conv_a -> conv_b, avoiding channels with residual fan-out.
    """
    mask = dormant_score(activation) <= tau
    first, second = block.conv_a, block.conv_b
    incoming = mask.reshape(-1, 1, 1, 1).expand_as(first.weight)
    outgoing = mask.reshape(1, -1, 1, 1).expand_as(second.weight)
    fresh = torch.empty_like(first.weight)
    nn.init.xavier_uniform_(fresh)
    first.weight.copy_(torch.where(incoming, fresh, first.weight))
    second.weight.copy_(torch.where(outgoing, torch.zeros_like(second.weight), second.weight))
    masks = [(first.weight, incoming), (second.weight, outgoing)]
    if first.bias is not None:
        first.bias.copy_(torch.where(mask, torch.zeros_like(first.bias), first.bias))
        masks.append((first.bias, mask))
    for parameter, reset in masks:
        for value in optimizer.state.get(parameter, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value.mul_(~reset)
    return int(mask.sum())


@torch.no_grad()
def shrink_perturb(net, args, shrink, sigma):
    """warm_start/run.py:126-136; use native initial weights as noise."""
    fresh = type(net)(args, False)
    fresh.update_fc(net.fc.out_features)
    fresh.to(next(net.parameters()))
    for p1, p2 in zip(fresh.parameters(), net.parameters()):
        p1.copy_(shrink*p2 + sigma*p1)
    net.load_state_dict(fresh.state_dict())


class CompetitorHook:
    def __init__(self, method, config):
        self.method, self.config = method, config
        self.features, self.generators = {}, []
        self.epoch = self.last_reset = 0
        self.optimizer = None
        self.steps = 0

    @contextmanager
    def install(self, learner, module):
        net = learner._network
        handles = []
        old_optim = module.optim
        pairs = branch_pairs(net)
        if self.method == 'icarl_snp':
            shrink_perturb(net, learner.args, **self.config)
            if not all(torch.isfinite(p).all() for p in net.parameters()):
                raise FloatingPointError('nonfinite_params after shrink-perturb, step=0')
        if self.method in ('icarl_cbp', 'icarl_redo'):
            for name, block in pairs:
                def capture(m, inputs, name=name):
                    if m.training: self.features[name] = inputs[0].detach()
                handles.append(block.conv_b.register_forward_pre_hook(capture))
        if self.method == 'icarl_cbp':
            from experiments.runners import lop
            lop.upstream_imports()
            from lop.algos.res_gnt import ResGnT
            saved = getattr(learner, '_v4_cbp', None)
            for index, (name, block) in enumerate(pairs):
                first = copy.copy(block.conv_a)
                first._parameters = dict(first._parameters)
                if first.bias is None:
                    first.bias = nn.Parameter(torch.zeros(first.out_channels, device=first.weight.device,
                                                          dtype=first.weight.dtype), requires_grad=False)
                bn = block.bn_a if isinstance(block.bn_a, nn.BatchNorm2d) else nn.BatchNorm2d(
                    first.out_channels).to(first.weight)
                gnt = ResGnT(nn.Sequential(first, bn, block.conv_b), 'relu',
                    replacement_rate=self.config['replacement_rate'], maturity_threshold=100,
                    device=first.weight.device)
                if saved:
                    for key, value in saved[index].items():
                        if isinstance(value,list):
                            value = [v.to(first.weight.device) if isinstance(v,torch.Tensor) else v for v in value]
                        setattr(gnt, key, copy.deepcopy(value))
                self.generators.append((name, gnt))
        original_sgd = old_optim.SGD
        def sgd(params, *args, **kwargs):
            if self.method == 'icarl_upgd':
                upstream = source_module('v4_upgd', 'upgd/core/optim/weight_upgd/first_order.py')
                optimizer = upstream.FirstOrderGlobalUPGD(net.named_parameters(),
                    lr=kwargs['lr'], weight_decay=kwargs.get('weight_decay', 0.), **self.config)
            else:
                optimizer = original_sgd(params, *args, **kwargs)
            self.optimizer = optimizer
            original_step = optimizer.step
            def step(optimizer_self, *a, **kw):
                result = original_step(*a, **kw)
                self.steps += 1
                for name, gnt in self.generators: gnt.gen_and_test([self.features[name]])
                if not all(torch.isfinite(p).all() for p in net.parameters()):
                    raise FloatingPointError(f'nonfinite_params at step={self.steps}')
                return result
            optimizer.step = MethodType(step, optimizer)
            return optimizer
        module.optim = SimpleNamespace(**{**vars(old_optim), 'SGD': sgd})
        try:
            yield
        finally:
            module.optim = old_optim
            for handle in handles: handle.remove()
            if self.generators:
                learner._v4_cbp = [{key: copy.deepcopy(getattr(gnt, key)) for key in
                    ('util', 'ages', 'mean_feature_mag', 'accumulated_num_features_to_replace')}
                    for _, gnt in self.generators]

    def after_epoch(self, learner):
        self.epoch += 1
        if self.method != 'icarl_redo' or self.epoch-self.last_reset < self.config['reset_epochs']:
            return
        for name, block in branch_pairs(learner._network):
            recycle(block, self.features[name], self.optimizer, self.config['tau'])
        self.last_reset = self.epoch
        if not all(torch.isfinite(p).all() for p in learner._network.parameters()):
            raise FloatingPointError(f'nonfinite_params after ReDo, step={self.steps}')
