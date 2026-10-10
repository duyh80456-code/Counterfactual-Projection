"""Relaxed Bypass (Jung/Lee IV-A) for the GELUs inside DeiT MLPs."""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from baselines.bypass.optimizer import remove_extension_parameters_

PAPER = 'https://www.donghunlee.com/papers/Jung_Lee_Bypass__IEEE_TNNLS.pdf'

@dataclass(frozen=True)
class BypassConfig:
    opt1_epochs: int = 100
    max_opt2_epochs: int = 50
    contraction_epsilon: float = .002
    gamma_slope: float = 3e-6

    def validate(self, horizon):
        if (self.opt1_epochs < 1 or self.max_opt2_epochs < 1
                or self.opt1_epochs + self.max_opt2_epochs != horizon
                or not math.isfinite(self.contraction_epsilon) or self.contraction_epsilon <= 0
                or not math.isfinite(self.gamma_slope) or self.gamma_slope <= 0):
            raise ValueError('Bypass requires positive opt1/opt2 budgets summing to the shared horizon and finite positive epsilon/gamma')

    def identity(self):
        return {**asdict(self), 'variant': 'relaxed_GELU_diagonal_D', 'paper': PAPER,
            'penalty': 'gamma_slope_times_opt2_step_times_sum_D_l2',
            'projection': 'drop_D_only_when_sum_D_l2_below_epsilon',
            'extension_weight_decay': 0., 'rollback': False, 'scheduler_rebased': False,
            'best_scope': 'original_space_only_after_contraction', 'budget_exhaustion': 'preserve_expanded_state_no_forced_projection'}

class BypassGELU(nn.Module):
    def __init__(self, activation, width, reference):
        super().__init__()
        self.activation = activation
        self.d = nn.Parameter(reference.new_zeros(width))

    def forward(self, inputs):
        return self.activation(inputs) + inputs * self.d


def embed(model, *, expected_sites=12):
    sites = [block.mlp for block in model.blocks]
    if len(sites) != expected_sites or any(not isinstance(site.act, nn.GELU) for site in sites):
        raise ValueError('Bypass requires the original GELU in every DeiT MLP')
    for site in sites:
        site.act = BypassGELU(site.act, site.fc1.out_features, site.fc1.weight)
    return [f'blocks.{i}.mlp.act' for i in range(len(sites))]


def extensions(model):
    return [module for module in model.modules() if isinstance(module, BypassGELU)]


def contraction_norm(model):
    modules = extensions(model)
    return torch.stack([module.d.norm() for module in modules]).sum() if modules else next(model.parameters()).new_zeros(())


def add_to_optimizer(model, optimizer):
    parameters = [module.d for module in extensions(model)]
    groups = [group for group in optimizer.param_groups if group['weight_decay'] == 0.]
    if len(groups) != 1:
        raise ValueError('expected one existing AdamW no-decay group; scheduler groups must stay unchanged')
    known = {id(p) for group in optimizer.param_groups for p in group['params']}
    if not parameters or any(id(p) in known for p in parameters):
        raise ValueError('Bypass extension absent or already in optimizer')
    groups[0]['params'].extend(parameters)


@torch.no_grad()
def contract(model, optimizer, epsilon):
    norm = float(contraction_norm(model))
    if not extensions(model) or not math.isfinite(norm) or norm >= epsilon:
        return False
    parameters = [module.d for module in extensions(model)]
    for block in model.blocks:
        activation = block.mlp.act
        block.mlp.act = activation.activation
    remove_extension_parameters_(optimizer, parameters)
    return True
