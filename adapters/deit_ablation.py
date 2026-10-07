"""Matched DeiT interventions. Raw reference is read-only and bound to theta_P."""
from __future__ import annotations

from dataclasses import asdict
import torch
from adapters.deit_cp_adapter import CPConfig, reset_adam_moments
from adapters.deit_mlp_growth import MLPExpansionCandidate, DeitMLPGrowthAdapter
from adapters.deit_random_control import matched_gaussian, random_parameter_delta
from adapters.deit_persistent_growth import commit_growth, select_growth_gamma
from adapters.deit_opt_e import OptEConfig, optimize_candidate
from experiments.run_plateau_comparison import select_by_expansion_gain, finite_projection
from experiments.run_shared_comparison import supervised_functional_descent_direction
from experiments.run_gromo_pilot import batch_loss, preview_projected_gain, cg_diagnostics
from projection import FunctionalProjector
from probe import CandidateExpansionProbe

ABLATION_METHODS = ('e_driven_o_raw', 'e_driven_o_normalized', 'random_control_parameter',
                    'random_control_logit', 'persistent_growth', 'opt_e')
SUITE_VERSION = 1


def res18_cp_config():
    return CPConfig(rank=4, projection_samples=32, scales=(.0125, .025, .05))


def select_candidate(model, statistics, where_batches, config, *, normalized=False):
    return select_by_expansion_gain(model, statistics, where_batches, config,
                                   next(model.parameters()).device, normalized=normalized)


def project_target(model, site, fit_target, gate_target, projection_batch, gate_batch, config, *, scales=None):
    names = DeitMLPGrowthAdapter.original_mlp_parameters(model, site)
    projector = FunctionalProjector(config.damping, config.cg_iterations,
        tolerance=config.cg_relative_tolerance, preconditioner_probes=config.cg_preconditioner_probes)
    projection = projector.project(model, projection_batch[0], fit_target, block=site, parameter_names=names)
    diagnostic = projector.evaluate_direction(model, gate_batch[0], gate_target, projection.parameter_delta)
    grid = config.scales if scales is None else scales
    gains = {str(scale): preview_projected_gain(model, projection, scale, gate_batch) for scale in grid}
    scale = max(grid, key=lambda value: gains[str(value)])
    applied = finite_projection(projection) and gains[str(scale)] > 0
    return projection, {'selected_site': site, 'projection_parameter_names': list(names),
        'correction_applied': bool(applied), 'selected_scale': float(scale) if applied else None,
        'line_search_gains': gains, 'relative_residual': projection.relative_residual,
        'cosine_alignment': projection.cosine_alignment, 'gate_relative_residual': diagnostic.relative_residual,
        'gate_cosine_alignment': diagnostic.cosine_alignment, 'gate_fitted_norm_ratio': diagnostic.fitted_norm_ratio,
        'evaluation_role': 'gate_batch_used_for_scale_selection', **cg_diagnostics(projection)}


def build_raw_reference(model, batches, config, fork_hash, *, seed, indices):
    candidate, selection = select_candidate(model, batches['statistics'], batches['where_batches'], config)
    probe = CandidateExpansionProbe()
    fit_target = probe(model, candidate=candidate, batch=batches['projection_batch'], gate=config.probe_epsilon).delta_logits
    gate_target = probe(model, candidate=candidate, batch=batches['gate_batch'], gate=config.probe_epsilon).delta_logits
    projection, record = project_target(model, candidate.module_name, fit_target, gate_target,
        batches['projection_batch'], batches['gate_batch'], config)
    return {'kind': 'deit_raw_intervention_reference', 'suite_version': SUITE_VERSION,
        'fork_hash': fork_hash, 'cp_config': asdict(config), 'seed': seed, 'probe_indices': indices,
        'site': candidate.module_name,
        'candidate': {key: getattr(candidate, key).detach().cpu().clone() for key in ('A', 'a', 'B')},
        'proposal_score': candidate.proposal_score, 'proposal': candidate.payload,
        'fit_target': fit_target.detach().cpu(), 'gate_target': gate_target.detach().cpu(),
        'parameter_delta': {key: value.detach().cpu() for key, value in projection.parameter_delta.items()},
        'record': {**selection, **record}}


def candidate_from_reference(model, reference):
    parameter = next(model.parameters())
    tensors = {key: value.to(parameter) for key, value in reference['candidate'].items()}
    return MLPExpansionCandidate(reference['site'], model, **tensors,
        proposal_score=reference['proposal_score'], payload=reference['proposal'])


def apply_delta(model, optimizer, delta, scale):
    parameters = dict(model.named_parameters())
    before = {name: parameters[name].detach().clone() for name in delta}
    with torch.no_grad():
        for name, value in delta.items():
            parameters[name].add_(value.to(parameters[name]), alpha=float(scale))
    changed = [name for name in before if not torch.equal(parameters[name], before[name])]
    resets = reset_adam_moments(optimizer, model, delta, parameter_before=before)
    return {'projection_changed_parameters': changed, 'adam_moments_reset_parameters': resets,
            'momentum_states_reset': len(resets)}


def execute_intervention(model, optimizer, batches, config, method, reference, *, seed, opt_config=OptEConfig(), growth_fallback=None):
    reference_record = reference['record']
    # Even A1 carries the raw/normalized WHERE tables for offline comparison; they do not select its site.
    record = {key: value for key, value in reference_record.items()
              if key in ('site_evaluations', 'site_functional_evaluations', 'site_scores')}
    record.update(method=method, suite_version=SUITE_VERSION, reference_site=reference['site'],
                  projection_changed_parameters=[], adam_moments_reset_parameters=[], momentum_states_reset=0)
    site = reference['site']
    candidate = candidate_from_reference(model, reference)
    gate_batch = batches['gate_batch']
    before = batch_loss(model, gate_batch)
    delta, scale = None, None
    if method == 'e_driven_o_raw':
        record.update(reference_record)
        delta, scale = reference['parameter_delta'], reference_record['selected_scale']
    elif method == 'random_control_parameter':
        delta, null = random_parameter_delta(model, reference['parameter_delta'], site, seed)
        scale = reference_record['selected_scale']
        record.update(selected_site=site, selected_scale=scale, correction_applied=scale is not None,
                      projection_parameter_names=list(delta), random_control=null)
    elif method == 'persistent_growth':
        gamma, losses = select_growth_gamma(model, candidate, gate_batch)
        persistent = commit_growth(model, optimizer, candidate, 0.)
        if growth_fallback is not None:
            growth_fallback()
        mlp = DeitMLPGrowthAdapter.resolve_site(model, site)
        with torch.no_grad():
            mlp.fc2.weight[:, -candidate.B.shape[1]:].copy_(gamma * candidate.B)
        persistent.update(gamma_g=float(gamma), gamma_losses=losses)
        record.update(selected_site=site, persistent=persistent, correction_applied=True,
                      selected_scale=gamma, projection_parameter_names=[],
                      intervention_kind='persistent_width_expansion_not_projection')
    else:
        scales = None
        if method == 'e_driven_o_normalized':
            candidate, selection = select_candidate(model, batches['statistics'], batches['where_batches'], config, normalized=True)
            site = candidate.module_name
            record.update(selection)
            probe = CandidateExpansionProbe()
            fit_target = probe(model, candidate=candidate, batch=batches['projection_batch'], gate=config.probe_epsilon).delta_logits
            gate_target = probe(model, candidate=candidate, batch=gate_batch, gate=config.probe_epsilon).delta_logits
        elif method == 'o_projection_only':
            site = config.o_only_site
            fit_target = supervised_functional_descent_direction(model, batches['projection_batch'])
            gate_target = supervised_functional_descent_direction(model, gate_batch)
        elif method == 'random_control_logit':
            generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
            fit_original = reference['fit_target'].to(next(model.parameters()))
            gate_original = reference['gate_target'].to(next(model.parameters()))
            fit_target = matched_gaussian(fit_original, generator)
            gate_target = matched_gaussian(gate_original, generator)
            record['random_control'] = {'seed': seed, 'norm_target': float(fit_original.norm()),
                'norm_actual': float(fit_target.norm()), 'scale_policy': 'own_gate_line_search',
                'gate_target_role': 'independent_random_null_not_generalization_diagnostic'}
        elif method == 'opt_e':
            candidate, optimized = optimize_candidate(model, candidate, batches['opt_fit'], batches['opt_val'], config=opt_config, seed=seed)
            record['opt_e'] = optimized
            record['opt_e_protocol'] = asdict(opt_config)
            if optimized['status'] == 'opt_e_failed':
                record.update(selected_site=site, correction_applied=False, selected_scale=None,
                              fallback='vanilla_no_rollback')
                return {**record, 'loss_before': before, 'loss_after': before, 'actual_loss_improvement': 0.}
            probe = CandidateExpansionProbe()
            fit_target = probe(model, candidate=candidate, batch=batches['projection_batch'], gate=1.).delta_logits
            gate_target = probe(model, candidate=candidate, batch=gate_batch, gate=1.).delta_logits
            scales = opt_config.scales
        else:
            raise ValueError(f'unknown ablation arm: {method}')
        projection, projected = project_target(model, site, fit_target, gate_target,
            batches['projection_batch'], gate_batch, config, scales=scales)
        record.update(projected)
        delta, scale = projection.parameter_delta, projected['selected_scale']
    if delta is not None and scale is not None:
        record.update(apply_delta(model, optimizer, delta, scale))
    after = batch_loss(model, gate_batch)
    return {**record, 'loss_before': before, 'loss_after': after, 'actual_loss_improvement': before - after,
            'actual_loss_improvement_role': 'gate_batch_used_for_selection_not_generalization_evidence'}
