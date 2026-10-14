"""Temporary virtual units declared by native module paths.

Example: HookSite(model, 'fc1', 'fc2', 'linear', between=('relu',)).
Norms are listed in execution order but act as identity on *new* coordinates,
matching pinned Gromo's extension convention (no inherited BN/LN statistics).
The original path executes its actual norm and activation without modification.
"""
from contextlib import contextmanager, ExitStack
from types import SimpleNamespace
import torch
from torch import nn
from torch.func import functional_call
from methods.vna import (_Products, _make_a_layer, _make_matching_b_layer,
                         _as_a_dict, _resolve_site)
from projection.functional import _preserve_functional_state
from probe.gromo_adapter import _snapshot, _assert_unchanged


class HookSite:
    def __init__(self, model, first: str, second: str, kind, *, between,
                 o_params=None):
        if kind not in ('linear', 'conv'):
            raise ValueError('kind must be linear or conv')
        self.model, self.first, self.second, self.kind = model, first, second, kind
        self.between = tuple(between)
        modules = dict(model.named_modules())
        expected = nn.Linear if kind == 'linear' else nn.Conv2d
        if not isinstance(modules[first], expected) or not isinstance(modules[second], expected):
            raise TypeError('site layers do not match kind')
        for path in self.between:
            modules[path]  # fail immediately on a misspelled declaration
        if kind == 'conv' and (modules[first].groups != 1 or modules[second].groups != 1):
            raise ValueError('grouped convolution sites are not supported')
        self.native_path = first[:-len('.first_layer.layer')] if first.endswith('.first_layer.layer') else None
        if self.native_path:
            _, _, names = _resolve_site(model, self.native_path)
            self.native_name = next((b.name for b in model.growing_blocks()
                                     if model.block(b.name) is modules[self.native_path]), self.native_path)
        else:
            ids = {id(p) for path in (first, second, *self.between)
                   for p in modules[path].parameters()}
            names = tuple(n for n, p in model.named_parameters() if id(p) in ids and p.requires_grad)
        self.parameter_names = tuple(names if o_params is None else o_params)
        if not self.parameter_names or not set(self.parameter_names) <= dict(model.named_parameters()).keys():
            raise ValueError('O scope must name model parameters')
        self.module_name = f'{first}->{second}'
        self.first_layer = SimpleNamespace(layer=modules[first])
        self.second_layer = SimpleNamespace(layer=modules[second])

    def resolve(self, model):
        if model is not self.model:
            raise ValueError('HookSite belongs to a different model')
        return self.module_name, self, self.parameter_names

    def virtual_features(self, x, a):
        value = a(x)
        norms = (nn.modules.batchnorm._BatchNorm, nn.LayerNorm, nn.GroupNorm)
        for path in self.between:
            module = self.model.get_submodule(path)
            if isinstance(module, norms) or module.__class__.__name__.startswith('Growing') and 'Norm' in module.__class__.__name__:
                continue
            if isinstance(module, nn.Sequential):
                for child in module:
                    if not isinstance(child, norms) and 'Norm' not in child.__class__.__name__:
                        value = child(value)
            else:
                value = module(value)
        return value


class HookProducts(_Products):
    """Same JVP/VJP contract, with external B tensors and per-call hooks.

    Native Gromo sites can reuse its exact cached suffix. Generic graphs use
    full native forward replay so branches and residual paths remain exact.
    """
    def __init__(self, model, sites, A_by_site, inputs, names, bn_mode='eval', cache_block_input=False):
        native_cache = len(sites) == 1 and sites[0].native_path and set(names) == set(sites[0].parameter_names)
        super().__init__(model, inputs, names, (), bn_mode=bn_mode,
                         cache_block_input=bool(cache_block_input and native_cache))
        self.virtual = []
        refs = []
        for i, site in enumerate(sites):
            A = _as_a_dict(A_by_site[site])
            weight = A['weight']
            devices = [weight.device.index] if weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                a = _make_a_layer(site.first_layer, A)
                b = _make_matching_b_layer(site.second_layer, len(weight), weight.dtype, weight.device)
            with torch.no_grad():
                for parameter in b.parameters():
                    parameter.zero_()
            b.eval()
            prefix = f'__virtual_{i}.'
            refs.extend((prefix + n, p.detach()) for n, p in b.named_parameters())
            self.virtual.append((site, a, b, prefix))
        self.extension_refs = tuple(refs)

    def _call(self, replacements):
        self.forward_calls += 1
        merged = {n: p for n, p in replacements.items() if not n.startswith('__virtual_')}
        if self.bn_mode == 'train':
            merged.update({n: b.clone() for n, b in self.buffers.items()})
        with ExitStack() as stack:
            for site, a, b, prefix in self.virtual:
                capture = {}
                def first_hook(module, args, capture=capture):
                    capture['input'] = args[0]
                def second_hook(module, args, output, site=site, a=a, b=b, prefix=prefix, capture=capture):
                    feature = site.virtual_features(capture['input'], a)
                    values = {n: replacements.get(prefix + n, p) for n, p in b.named_parameters()}
                    return output + functional_call(b, values, (feature,))
                first_handle = self.model.get_submodule(site.first).register_forward_pre_hook(first_hook)
                stack.callback(first_handle.remove)
                second_handle = self.model.get_submodule(site.second).register_forward_hook(second_hook)
                stack.callback(second_handle.remove)
            if self.suffix is not None:
                return functional_call(self.suffix, {'model.' + n: p for n, p in merged.items()}, (), strict=False).flatten()
            return functional_call(self.model, merged, (self.inputs,), strict=False).flatten()


@contextmanager
def hook_products(model, sites, A_by_site, inputs, names=None, *, bn_mode='eval', cache_block_input=False):
    sites = list(sites)
    for site in sites:
        site.resolve(model)
    names = sites[0].parameter_names if names is None else names
    before = _snapshot(model)
    modes = {m: m.training for m in model.modules()}
    count = sum(p.numel() for p in model.parameters())
    try:
        model.eval()
        if bn_mode == 'train':
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.train()
        with _preserve_functional_state(model), torch.autocast(device_type=inputs.device.type, enabled=False):
            yield HookProducts(model, sites, A_by_site, inputs, names, bn_mode, cache_block_input)
    finally:
        for module, mode in modes.items():
            module.training = mode
        _assert_unchanged(model, before)
        assert count == sum(p.numel() for p in model.parameters()), 'parameter count changed'


def hook_tiny(model, site, batches, rank, *, cotangents, cfg,
              existing_weight_projection, normalize, score_signs, numerical_threshold,
              loss_spec=None, contexts=None):
    """TINY's whitened input/cotangent SVD for native linear/conv sites.

    Norms on new coordinates are identity, as in HookProducts. Incoming
    patches form S; outgoing-kernel cotangents form N. Optional projection
    removes the existing second-layer feature span before the solve.
    """
    from methods.vna_select import normalize_A, _neuron, _concat, score_A
    from methods.vna_ggn import filtered_cotangent
    import torch.nn.functional as F
    batches = list(batches)
    if not batches or rank < 1:
        raise ValueError('need batches and positive rank')
    first, second = site.first_layer.layer, site.second_layer.layer
    d = first.weight[0].numel() + (first.bias is not None)
    seed = {'weight': first.weight.detach()[:1], 'bias': None if first.bias is None else first.bias.detach()[:1]}
    if cotangents is None:
        contexts = [None] * len(batches) if contexts is None else list(contexts)
        if len(contexts) != len(batches):
            raise ValueError('one context required per batch')
        cotangents = [filtered_cotangent(model, site, seed, batch, cfg, loss_spec=loss_spec, ctx=ctx)
                      for batch, ctx in zip(batches, contexts)]
    cotangents = list(cotangents)
    if len(cotangents) != len(batches):
        raise ValueError('one cotangent required per batch')
    before = _snapshot(model)
    modes = {m: m.training for m in model.modules()}
    xs, ys, rs = [], [], []
    try:
        model.eval()
        with _preserve_functional_state(model), torch.enable_grad():
            for (inputs, _), z in zip(batches, cotangents):
                capture = {}
                def save_first(module, args):
                    capture['x'] = args[0].detach()
                def save_second(module, args, output):
                    capture['y'], capture['output'] = args[0].detach(), output
                with ExitStack() as stack:
                    h = first.register_forward_pre_hook(save_first)
                    stack.callback(h.remove)
                    h = second.register_forward_hook(save_second)
                    stack.callback(h.remove)
                    output = model(inputs)
                    residual = torch.autograd.grad(output, capture['output'], z.to(output).reshape_as(output))[0]
                x, y = capture['x'], capture['y']
                if site.kind == 'conv':
                    x = F.unfold(x, first.kernel_size, dilation=first.dilation,
                                 padding=first.padding, stride=first.stride)
                    if first.bias is not None:
                        x = torch.cat((x, torch.ones_like(x[:, :1])), 1)
                    h, w = capture['y'].shape[-2:]
                    x = x.reshape(len(inputs), d, h, w)
                    x = F.unfold(x, second.kernel_size, dilation=second.dilation,
                                 padding=second.padding, stride=second.stride).transpose(1, 2).reshape(-1, d, second.weight[0, 0].numel())
                    y = F.unfold(y, second.kernel_size, dilation=second.dilation,
                                 padding=second.padding, stride=second.stride).transpose(1, 2).reshape(len(x), -1)
                    residual = residual.permute(0, 2, 3, 1).reshape(len(x), -1)
                else:
                    x = x.reshape(-1, first.in_features)
                    if first.bias is not None:
                        x = torch.cat((x, torch.ones_like(x[:, :1])), 1)
                    x = x.unsqueeze(-1)
                    y, residual = y.reshape(len(x), -1), residual.reshape(len(x), -1)
                if second.bias is not None:
                    y = torch.cat((y, torch.ones_like(y[:, :1])), 1)
                xs.append(x.double()); ys.append(y.double()); rs.append(residual.detach().double())
    finally:
        for module, mode in modes.items():
            module.training = mode
        _assert_unchanged(model, before)
    X, Y, R = torch.cat(xs), torch.cat(ys), torch.cat(rs)
    if existing_weight_projection:
        R = R - Y @ (torch.linalg.pinv(Y, rtol=numerical_threshold) @ R)
    n = sum(len(x) for x, _ in batches)
    S = torch.einsum('ndk,nek->de', X, X) / n
    N = torch.einsum('ndk,no->dko', X, R).reshape(d, -1) / n
    es, qs = torch.linalg.eigh(S)
    inv = torch.where(es > numerical_threshold, es.clamp_min(numerical_threshold).rsqrt(), 0.)
    whitening = (qs * inv) @ qs.T
    u, singular, _ = torch.linalg.svd(whitening @ N, full_matrices=False)
    keep = min(rank, max(1, int((singular > numerical_threshold).sum())))
    rows = (whitening @ u[:, :keep]).T.to(first.weight)
    A = {'weight': rows[:, :first.weight[0].numel()].reshape(keep, *first.weight.shape[1:]),
         'bias': rows[:, -1] if first.bias is not None else None}
    if normalize:
        A = normalize_A(model, site, A)
    if score_signs:
        chosen = []
        for i in range(keep):
            plus, minus = _neuron(A, i), _neuron(A, i, -1)
            gains = [sum(score_A(model, site, a, x, z) for (x, _), z in zip(batches, cotangents)) for a in (plus, minus)]
            chosen.append(plus if gains[0] >= gains[1] else minus)
        A = _concat(chosen)
    return A, {'S': S, 'N_z': N, 'singular_values': singular[:keep],
               'generalized_eigenvalues': singular.square(),
               'existing_weight_projection': existing_weight_projection}
