"""Commit an exact MLP width expansion, preserving AdamW's old state slices."""
from __future__ import annotations

import torch
from torch import nn
from adapters.deit_mlp_growth import DeitMLPGrowthAdapter
from experiments.run_gromo_pilot import batch_loss


def migrate_parameter(optimizer, old, new, axis):
    groups = [group for group in optimizer.param_groups if any(p is old for p in group['params'])]
    if len(groups) != 1:
        raise ValueError("old parameter must belong to exactly one optimizer group")
    state = optimizer.state.get(old, {})
    migrated = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and value.shape == old.shape:
            expanded = torch.zeros_like(new)
            slices = [slice(None)] * old.ndim
            slices[axis] = slice(0, old.shape[axis])
            expanded[tuple(slices)].copy_(value)
            migrated[key] = expanded
        else:
            migrated[key] = value.clone() if isinstance(value, torch.Tensor) else value
    group = groups[0]
    group['params'] = [new if p is old else p for p in group['params']]
    optimizer.state.pop(old, None)
    if state:
        optimizer.state[new] = migrated


@torch.no_grad()
def commit_growth(model, optimizer, candidate, gamma):
    mlp = DeitMLPGrowthAdapter.resolve_site(model, candidate.module_name)
    rank = candidate.A.shape[0]
    before = sum(p.numel() for p in model.parameters())
    replacements = [(mlp.fc1, 'weight', torch.cat((mlp.fc1.weight, candidate.A), dim=0), 0),
                    (mlp.fc1, 'bias', torch.cat((mlp.fc1.bias, candidate.a), dim=0), 0),
                    (mlp.fc2, 'weight', torch.cat((mlp.fc2.weight, gamma * candidate.B), dim=1), 1)]
    for module, name, value, axis in replacements:
        old = getattr(module, name)
        new = nn.Parameter(value.detach().clone(), requires_grad=old.requires_grad)
        migrate_parameter(optimizer, old, new, axis)
        setattr(module, name, new)
    mlp.fc1.out_features += rank
    mlp.fc2.in_features += rank
    return {"site": candidate.module_name, "effective_rank": rank, "gamma_g": float(gamma),
            "param_count": sum(p.numel() for p in model.parameters()) - before,
            "hidden_width": mlp.fc1.out_features, "optimizer_step_policy": "retain_old_tensor_step",
            "old_moments_preserved": True, "new_moments": "zero", "scheduler_changed": False}


def select_growth_gamma(model, candidate, gate_batch, scales=(0., .025, .05, .1, .2)):
    losses = {}
    for gamma in scales:
        with candidate.virtual_direction(gamma):
            losses[str(gamma)] = batch_loss(model, gate_batch)
    # Prefer zero on a tie; no gain is required to train a zero-output expansion.
    gamma = min(scales, key=lambda value: (losses[str(value)], value))
    return gamma, losses


def restore_growth_geometry(model, state):
    """Instantiate saved widths before model AND optimizer loading on resume."""
    for site in DeitMLPGrowthAdapter.enumerate_sites(model):
        mlp = DeitMLPGrowthAdapter.resolve_site(model, site)
        hidden = state[f'{site}.fc1.weight'].shape[0]
        if hidden != mlp.fc1.out_features:
            if hidden < mlp.fc1.out_features:
                raise ValueError("persistent checkpoint cannot shrink a canonical MLP")
            mlp.fc1 = nn.Linear(mlp.fc1.in_features, hidden).to(mlp.fc1.weight)
            mlp.fc2 = nn.Linear(hidden, mlp.fc2.out_features).to(mlp.fc2.weight)
