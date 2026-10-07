"""Random nulls with explicit norm matching and a private RNG stream."""
from __future__ import annotations

import torch


def matched_gaussian(target, generator):
    noise = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
    norm = target.norm()
    if norm == 0:
        return torch.zeros_like(target)
    return noise * (norm / noise.norm().clamp_min(torch.finfo(noise.dtype).tiny))


def random_parameter_delta(model, reference_delta, site, seed):
    from adapters.deit_mlp_growth import DeitMLPGrowthAdapter
    names = DeitMLPGrowthAdapter.original_mlp_parameters(model, site)
    if set(reference_delta) != set(names):
        raise ValueError("random control requires exactly the four original selected MLP tensors")
    parameters = dict(model.named_parameters())
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    delta = {name: matched_gaussian(reference_delta[name].to(parameters[name]), generator) for name in names}
    return delta, {"seed": seed, "norm_target": {name: float(reference_delta[name].norm()) for name in names},
                   "norm_actual": {name: float(value.norm()) for name, value in delta.items()},
                   "matching": "per_tensor_pre_scale", "scale_policy": "fixed_to_A2_no_gate_selection"}
