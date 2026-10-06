"""One-shot DeiT E-to-O using the existing WHERE and functional projector."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from adapters.deit_mlp_growth import DeitMLPGrowthAdapter
from experiments.run_plateau_comparison import select_by_expansion_gain, finite_projection
from experiments.run_shared_comparison import supervised_functional_descent_direction
from experiments.run_gromo_pilot import (
    actual_update_metrics, batch_loss, cg_diagnostics, eval_logits,
    heldout_metrics, parameter_delta_norm, preview_projected_gain)
from experiments.shared_protocol import rng_state, restore_rng
from probe import CandidateExpansionProbe
from projection import FunctionalProjector


@dataclass(frozen=True)
class CPConfig:
    rank: int = 8
    probe_epsilon: float = .05
    statistics_samples: int = 256
    where_batches: int = 3
    where_samples: int = 32
    projection_samples: int = 64
    gate_samples: int = 32
    cg_iterations: int = 200
    damping: float = 1e-3
    cg_relative_tolerance: float = 1e-2
    cg_preconditioner_probes: int = 8
    scales: tuple = (.025, .05, .1, .2)
    o_only_site: str = "blocks.11.mlp"

    def __post_init__(self):
        if (any(value < 1 for value in (self.rank, self.statistics_samples,
                self.where_batches, self.where_samples, self.projection_samples,
                self.gate_samples, self.cg_iterations)) or
                not 0 < self.probe_epsilon <= 1 or self.damping <= 0 or
                self.cg_relative_tolerance <= 0 or self.cg_preconditioner_probes < 0 or
                not self.scales or any(not 0 < value <= .2 for value in self.scales)):
            raise ValueError("invalid DeiT CP configuration")


def probe_indices(train_indices, seed, config, probe_index=0):
    """Same deterministic batch recipe as the CNN intervention protocol."""
    sizes = [config.statistics_samples] + [config.where_samples] * config.where_batches
    sizes += [config.projection_samples, config.gate_samples]
    if any(size < 1 for size in sizes) or len(train_indices) < sum(sizes):
        raise ValueError("positive probe sizes and enough training samples required")
    generator = torch.Generator().manual_seed(911_731 + seed * 10_000 + probe_index)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    selected = [train_indices[index] for index in order[:sum(sizes)]]
    batches, cursor = [], 0
    for size in sizes:
        batches.append(selected[cursor:cursor + size])
        cursor += size
    return {"statistics": batches[0], "where": batches[1:-2],
            "projection": batches[-2], "gate": batches[-1]}


def reset_adam_moments(optimizer, model, parameter_delta):
    parameters = dict(model.named_parameters())
    reset = []
    for name, delta in parameter_delta.items():
        if not bool(torch.any(delta != 0)):
            continue
        state = optimizer.state.get(parameters[name], {})
        cleared = False
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                state[key].zero_()
                cleared = True
        if cleared:
            reset.append(name)
    return reset  # Adam step counter and all other tensors are retained


def one_shot_intervention(model, optimizer, *, statistics, where_batches,
                          projection_batch, gate_batch, config=CPConfig(),
                          method="ours_e_driven_o"):
    if method not in {"ours_e_driven_o", "o_projection_only"}:
        raise ValueError("unsupported one-shot intervention method")
    rng = rng_state()
    try:
        device = next(model.parameters()).device
        probe = CandidateExpansionProbe()
        selection = {}
        if method == "ours_e_driven_o":
            candidate, selection = select_by_expansion_gain(
                model, statistics, where_batches, config, device)
            if len(model.blocks) == 12 and selection["selection_candidate_count"] != 12:
                raise RuntimeError("DeiT WHERE must scan all 12 MLPs")
            site = candidate.module_name
            fit_signal = probe(model, candidate=candidate, batch=projection_batch, gate=config.probe_epsilon)
            gate_signal = probe(model, candidate=candidate, batch=gate_batch, gate=config.probe_epsilon)
            fit_target, gate_target = fit_signal.delta_logits, gate_signal.delta_logits
            selection["proposal"] = candidate.payload
        else:
            site = config.o_only_site  # declared before seeing E or validation
            fit_target = supervised_functional_descent_direction(model, projection_batch)
            gate_target = supervised_functional_descent_direction(model, gate_batch)
        names = DeitMLPGrowthAdapter.original_mlp_parameters(model, site)
        projector = FunctionalProjector(config.damping, config.cg_iterations,
            tolerance=config.cg_relative_tolerance,
            preconditioner_probes=config.cg_preconditioner_probes)
        projection = projector.project(model, projection_batch[0], fit_target,
                                       block=site, parameter_names=names)
        heldout = projector.evaluate_direction(model, gate_batch[0], gate_target,
                                               projection.parameter_delta)
        scales = tuple(float(value) for value in config.scales)
        if not scales or any(value <= 0 for value in scales):
            raise ValueError("positive line-search scales required")
        gains = {str(scale): preview_projected_gain(model, projection, scale, gate_batch)
                 for scale in scales}
        scale = max(scales, key=lambda value: gains[str(value)])
        loss_before = batch_loss(model, gate_batch)
        baseline = eval_logits(model, gate_batch[0])
        applied = finite_projection(projection) and gains[str(scale)] > 0
        actual, resets = {}, []
        if applied:
            projection.apply_(model, scale)
            resets = reset_adam_moments(optimizer, model, projection.parameter_delta)
            actual = actual_update_metrics(model, gate_batch[0], baseline, gate_target, scale)
        loss_after = batch_loss(model, gate_batch)
        return {**selection, "selected_site": site,
                "uses_structural_E": method == "ours_e_driven_o",
                "source": "native_gromo_linear_tiny" if method == "ours_e_driven_o" else "supervised_projection_only_control",
                "correction_applied": applied,
                "projection_parameter_names": list(names),
                "selected_scale": scale if applied else None,
                "line_search_gains": gains,
                "parameter_delta_norm": float(parameter_delta_norm(projection.parameter_delta)),
                "loss_before": loss_before, "loss_after": loss_after,
                "actual_loss_improvement": loss_before - loss_after,
                "adam_moments_reset_parameters": resets,
                "relative_residual": projection.relative_residual,
                "cosine_alignment": projection.cosine_alignment,
                "heldout_role": "gate_batch_also_used_for_scale_selection",
                **actual, **heldout_metrics(heldout), **cg_diagnostics(projection)}
    finally:
        restore_rng(rng)
