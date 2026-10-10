"""Read-only TINY ranking and same-block projection/gate selection."""
from __future__ import annotations

import hashlib
import math
import time
import torch
from adapters.deit_ablation import project_target, apply_delta
from adapters.deit_mlp_growth import DeitMLPGrowthAdapter
from experiments.run_plateau_comparison import select_by_expansion_gain
from experiments.run_shared_comparison import supervised_functional_descent_direction
from experiments.run_gromo_pilot import parameter_delta_norm, cg_diagnostics
from projection import FunctionalProjector
from probe import CandidateExpansionProbe

EPS = 1e-12


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def choose_best_gate(results):
    # Stable tie-break follows the existing TINY rank, not validation or scale size.
    valid = [item for item in results if item['record']['accepted']]
    return max(valid, key=lambda item: item['record']['gate_gain']) if valid else None


def projection_aware_search(model, batches, config, *, diagnostic_all=False,
                            expected_sites=12, comparison_inputs=None):
    started = time.perf_counter()
    before = tensor_hash(model.state_dict())
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        _, where, proposals = select_by_expansion_gain(model, batches['statistics'],
            batches['where_batches'], config, next(model.parameters()).device, return_candidates=True)
        if len(proposals) != expected_sites:
            raise ValueError(f'expected {expected_sites} TINY sites; got {len(proposals)}')
        candidates = proposals if diagnostic_all else proposals[:3]
        results = []
        gradient = supervised_functional_descent_direction(model, batches['projection_batch']) if diagnostic_all else None
        probe = CandidateExpansionProbe()
        for row in candidates:
            candidate, site = row['candidate'], row['site']
            fit = probe(model, candidate=candidate, batch=batches['projection_batch'], gate=config.probe_epsilon).delta_logits
            gate = probe(model, candidate=candidate, batch=batches['gate_batch'], gate=config.probe_epsilon).delta_logits
            names = DeitMLPGrowthAdapter.original_mlp_parameters(model, site)
            try:
                projection, metrics = project_target(model, site, fit, gate,
                    batches['projection_batch'], batches['gate_batch'], config)
            except RuntimeError as error:
                if str(error) != 'all projection attempts produced non-finite solutions':
                    raise
                record = {'block_name': site, 'tiny_rank': row['rank'], 'e_gain': row['mean_e_gain'],
                    'functional_delta_norm': row['mean_delta_f_norm'], 'target_norm': float(fit.norm()),
                    'projection_parameter_names': list(names), 'correction_norm': None,
                    'projection_residual': None, 'projection_cosine': None, 'gate_gain': None,
                    'selected_scale': None, 'accepted': False, 'correction_applied': False,
                    'degenerate_target': float(fit.norm()) <= EPS, 'cg_converged': False,
                    'damping_used': None, 'cg_relative_residual': None, 'solver_failure': str(error)}
                if diagnostic_all:
                    record.update(gradient_residual=None, gradient_damping_used=None,
                        gradient_same_damping_as_expansion=False,
                        gradient_unavailable_reason='no_finite_E_solution_for_matched_damping')
                results.append({'record': record, 'projection': None, 'candidate': candidate})
                if tensor_hash(model.state_dict()) != before:
                    raise AssertionError('failed candidate preview mutated the anchor')
                continue
            if set(projection.parameter_delta) != set(names):
                raise AssertionError('projection escaped the original selected MLP')
            scale_gains = metrics['line_search_gains']
            finite = [(float(scale), gain) for scale, gain in scale_gains.items() if math.isfinite(gain)]
            preview_scale, gain = max(finite, key=lambda pair: pair[1]) if finite else (None, None)
            target_norm = float(fit.norm())
            accepted = bool(metrics['correction_applied'] and gain is not None and gain > 0
                            and target_norm > EPS and metrics['selected_scale'] == preview_scale)
            record = {**metrics, 'block_name': site, 'tiny_rank': row['rank'], 'e_gain': row['mean_e_gain'],
                'functional_delta_norm': row['mean_delta_f_norm'], 'target_norm': target_norm,
                'correction_norm': float(parameter_delta_norm(projection.parameter_delta)),
                'projection_residual': None if target_norm <= EPS else projection.relative_residual,
                'projection_cosine': None if target_norm <= EPS else projection.cosine_alignment,
                'degenerate_target': target_norm <= EPS, 'gate_gain': gain, 'preview_selected_scale': preview_scale,
                'selected_scale': metrics['selected_scale'] if accepted else None, 'accepted': accepted,
                'gate_gain_numerically_tiny': gain is not None and 0 < gain <= 1e-8,
                'cg_converged': projection.cg.converged, 'damping_used': projection.damping_used,
                'cg_relative_residual': projection.cg.relative_residual}
            if diagnostic_all:
                gnorm = float(gradient.norm())
                gprojector = FunctionalProjector(projection.damping_used, config.cg_iterations,
                    tolerance=config.cg_relative_tolerance, max_damping_retries=0,
                    preconditioner_probes=config.cg_preconditioner_probes)
                gp = gprojector.project(model, batches['projection_batch'][0], gradient,
                                        block=site, parameter_names=names)
                record.update(gradient_residual=None if gnorm <= EPS else gp.relative_residual,
                    gradient_target_norm=gnorm, gradient_degenerate_target=gnorm <= EPS,
                    gradient_damping_used=gp.damping_used, gradient_cg=cg_diagnostics(gp),
                    gradient_loss_reduction='sum_CE_logits', gradient_same_damping_as_expansion=True)
            result = {'record': record, 'projection': projection, 'candidate': candidate}
            results.append(result)
            if tensor_hash(model.state_dict()) != before:
                raise AssertionError('candidate preview mutated the anchor')
        chosen = choose_best_gate(results)
        direction = None
        if chosen and comparison_inputs is not None:
            # Fixed inputs across searches; never used to choose a site or scale.
            labels = torch.zeros(len(comparison_inputs), dtype=torch.long, device=comparison_inputs.device)
            direction = probe(model, candidate=chosen['candidate'], batch=(comparison_inputs, labels),
                              gate=config.probe_epsilon).delta_logits.detach().cpu()
        records = [item['record'] for item in results]
        return chosen, {'where': where, 'candidates': records,
            'candidate_blocks': [row['block_name'] for row in records],
            'selected_block': chosen['record']['block_name'] if chosen else None,
            'selected_gate_gain': chosen['record']['gate_gain'] if chosen else None,
            'accepted_count': sum(row['accepted'] for row in records),
            'rejected_count': sum(not row['accepted'] for row in records),
            'no_valid_intervention': chosen is None, 'anchor_hash': before,
            'search_seconds': time.perf_counter() - started,
            'selection_role': 'TINY_WHERE_then_same_anchor_gate_loss_only'}, direction
    finally:
        for module, training in modes.items():
            module.training = training
        if tensor_hash(model.state_dict()) != before:
            raise AssertionError('search mutated model state')


def commit_choice(model, optimizer, chosen):
    if chosen is None:
        return {'correction_applied': False, 'projection_changed_parameters': [],
                'adam_moments_reset_parameters': [], 'momentum_states_reset': 0}
    result = apply_delta(model, optimizer, chosen['projection'].parameter_delta, chosen['record']['selected_scale'])
    if set(result['projection_changed_parameters']) != set(result['adam_moments_reset_parameters']):
        raise AssertionError('Adam reset scope differs from changed tensors')
    return {**result, 'correction_applied': True, 'selected_scale': chosen['record']['selected_scale']}


def diagnostic_summary(records):
    valid = [row for row in records if row['gate_gain'] is not None]
    gate_best = max(valid, key=lambda row: row['gate_gain']) if valid else None
    residuals = [row for row in records if row['projection_residual'] is not None and math.isfinite(row['projection_residual'])]
    residual_best = min(residuals, key=lambda row: row['projection_residual']) if residuals else None
    return {'accepted_blocks': sum(row['accepted'] for row in records),
        'top1_block': records[0]['block_name'], 'best_gate_block': gate_best['block_name'] if gate_best else None,
        'top1_matches_best_gate': gate_best is not None and gate_best['block_name'] == records[0]['block_name'],
        'top3_has_accepted': any(row['accepted'] for row in records if row['tiny_rank'] <= 3),
        'smallest_residual_block': residual_best['block_name'] if residual_best else None,
        'smallest_residual_has_positive_gate_gain': residual_best is not None and residual_best['gate_gain'] is not None and residual_best['gate_gain'] > 0,
        'near_zero_e_gain_accepted_blocks': [row['block_name'] for row in records if abs(row['e_gain']) <= 1e-8 and row['accepted']],
        'near_zero_e_gain_threshold': 1e-8}
