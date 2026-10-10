"""Seed2 D0/A3/A4 only, from a completed historical-best fork; never retrain controls."""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch
from torch.nn.functional import cosine_similarity

from adapters.deit_ablation import res18_cp_config
from adapters.deit_projection_aware import (projection_aware_search, commit_choice, diagnostic_summary, tensor_hash)
from experiments.deit_protocol import (checked_source, historical_best, load_training_context,
    materialize_probe_batches, save_state, evaluate_without_rng)
from experiments.deit_e_rollback import EAccuracyRollback
from experiments.shared_protocol import (sha256_file, atomic_json_save, atomic_torch_save,
    train_epoch, seed_everything, rng_state, restore_rng)
from experiments.deit_logging import emit_event, emit_epoch

METHODS = ('all12_transfer_diagnostic', 'e2o_top3_best_gate', 'e2o_top3_recurrent')


def validate_seed2_fork(fork, recipe):
    if (recipe.seed != 2 or fork['epoch'] != 375
            or abs(fork['historical_best_accuracy'] - .562) > 1e-12
            or recipe.batch_size != 64 or recipe.learning_rate != 5e-4
            or recipe.weight_decay != .05):
        raise ValueError('requires seed2 best epoch375 /56.20%, batch64, LR5e-4, WD.05; no Vanilla retraining')


def write_csv(rows, path):
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    temporary.replace(path)


def export_events(events, path):
    # Rebuild from committed checkpoint events, avoiding duplicate append after resume.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(event) + '\n' for event in events))
    temporary.replace(path)


def identity_for(fork_hash, method, config, horizon, patience):
    return {'version': 1, 'fork_hash': fork_hash, 'method': method,
        'cp_config': json.loads(json.dumps(asdict(config))), 'post_fork_epochs': horizon,
        'algorithm_patience': patience, 'top_k': 3, 'retrigger': method == 'e2o_top3_recurrent',
        'anchor_policy': 'accuracy_then_lower_loss_on_exact_tie',
        'stall_policy': 'controller_anchor', 'selection': 'TINY_rank_then_positive_finite_gate_gain',
        'first_probe_partition': 'original_probe_index0', 'recurrent_probe_partition': 'current_CPU_RNG_draw',
        'functional_comparison_batch': 'fixed_original_projection_inputs',
        'report_best_scope': 'postfork_training_validation_only', 'retrigger_after_final_epoch': False, 'official_test_used': False}


def run(args, fork, recipe, config, *, expected_sites=12):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    fork_hash = sha256_file(args.plateau_checkpoint)
    identity = identity_for(fork_hash, args.method, config, args.post_fork_epochs, args.algorithm_patience)
    diagnostic = args.method == METHODS[0]
    latest = output / 'checkpoint_latest.pt'
    saved = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if saved and (saved.get('kind') != 'deit_projection_aware_latest' or saved.get('run_identity') != identity
                  or saved.get('protocol') != fork['protocol']):
        raise ValueError('projection-aware resume fork/config/method mismatch')
    accuracy, loss, epoch0 = historical_best(fork)
    if saved and (not 0 <= saved['completed_epochs'] <= args.post_fork_epochs
                  or saved['epoch'] != epoch0 + saved['completed_epochs']
                  or len(saved['history']) != saved['completed_epochs'] + 1
                  or saved['search_count'] != len(saved['interventions'])):
        raise ValueError('inconsistent projection-aware resume progress')
    metadata = {'run_identity': identity, 'protocol': fork['protocol'], 'fork_epoch': epoch0,
                'historical_best_accuracy': accuracy, 'historical_best_loss': loss}
    atomic_json_save(metadata, output / 'run_metadata.json')
    if diagnostic and (output / 'result.json').exists():
        result = json.loads((output / 'result.json').read_text())
        if result.get('run_identity') != identity or len(result.get('diagnostics', [])) != expected_sites:
            raise ValueError('D0 existing output belongs to another experiment or is incomplete')
        write_csv(result['diagnostics'], output / 'all12_diagnostics.csv')
        return result
    # Fully completed arms can rebuild outputs on CPU, without model/data/CUDA.
    if saved:
        EAccuracyRollback.from_state(saved['e_controller'], args.algorithm_patience,
            saved['completed_epochs'], stall_on_anchor=True)
    if saved and saved['completed_epochs'] == args.post_fork_epochs and not saved['pending_search']:
        return finish(saved, output)
    seed_everything(recipe.seed)
    device = torch.device(args.device)
    context = load_training_context(args.data_root, recipe, device, saved or fork)
    model, optimizer, scheduler, loader, evaluation_loader, evaluation, train_ids, val_ids, tuning_ids, reserved_ids = context
    model.eval()
    emit_event('deit_run_start', {'method': args.method, 'architecture': fork['protocol']['architecture'],
        'seed': recipe.seed, 'fork_epoch': epoch0, 'historical_best_accuracy': accuracy,
        'resume_offset': saved['completed_epochs'] if saved else 0,
        'post_fork_epochs': 0 if diagnostic else args.post_fork_epochs,
        'algorithm_patience': args.algorithm_patience, 'retrigger': identity['retrigger']}, output)
    live = rng_state()
    comparison_batches, first_indices = materialize_probe_batches(evaluation, train_ids, recipe, config, device)
    restore_rng(live)
    fixed_inputs = comparison_batches['projection_batch'][0]
    if diagnostic:
        model_before = tensor_hash(model.state_dict())
        _, report, _ = projection_aware_search(model, comparison_batches, config,
            diagnostic_all=True, expected_sites=expected_sites)
        assert tensor_hash(model.state_dict()) == model_before
        result = {**metadata, 'method': args.method, 'diagnostics': report['candidates'],
            'summary': diagnostic_summary(report['candidates']), 'search': report,
            'probe_indices': first_indices, 'postfork_training_epochs': 0, 'committed_corrections': 0}
        write_csv(report['candidates'], output / 'all12_diagnostics.csv')
        atomic_json_save(result, output / 'result.json')
        emit_event('all12_transfer_diagnostic', result['summary'], output)
        return result
    before = saved['validation_before'] if saved else evaluate_without_rng(model, evaluation_loader, device)
    controller = (EAccuracyRollback.from_state(saved['e_controller'], args.algorithm_patience,
                    saved['completed_epochs'], stall_on_anchor=True) if saved else
                  EAccuracyRollback(model, optimizer, scheduler, loader, before, epoch0,
                    args.algorithm_patience, stall_on_anchor=True))
    history = saved['history'] if saved else [{'epoch': epoch0, 'post_fork_epoch': 0,
        'validation_accuracy': before['accuracy'], 'validation_loss': before['loss'],
        'validation_role': 'fork_baseline_not_postfork_observation'}]
    events = saved['interventions'] if saved else []
    previous_direction = saved.get('previous_direction') if saved else None
    best_accuracy = saved['report_best_accuracy'] if saved else float('-inf')
    best_loss = saved['report_best_loss'] if saved else float('inf')
    best_epoch = saved['report_best_epoch'] if saved else None
    report_stall = saved['report_stall_counter'] if saved else 0
    completed = saved['completed_epochs'] if saved else 0
    pending = saved['pending_search'] if saved else True

    def payload():
        return dict(run_identity=identity, completed_epochs=completed, interventions=events,
            validation_before=before, historical_best_accuracy=accuracy, historical_best_loss=loss,
            historical_best_epoch=epoch0, report_best_accuracy=best_accuracy, report_best_loss=best_loss,
            report_best_epoch=best_epoch, report_stall_counter=report_stall,
            e_controller=controller.state_dict(), search_count=len(events), pending_search=pending,
            previous_direction=previous_direction)

    def save():
        save_state(latest, model=model, optimizer=optimizer, scheduler=scheduler, loader=loader,
            epoch=epoch0 + completed, history=history, train_indices=train_ids,
            evaluation_indices=val_ids, source_tuning_indices=tuning_ids, trigger_indices=reserved_ids,
            run_protocol=fork['protocol'], kind='deit_projection_aware_latest', **payload())
        atomic_torch_save({**controller.anchor, 'kind': 'deit_projection_aware_anchor',
            'protocol': fork['protocol'], 'run_identity': identity}, output / 'best_checkpoint.pt')
        export_events(events, output / 'interventions.jsonl')
        write_csv(history, output / 'epoch_history.csv')

    def search():
        nonlocal pending, previous_direction
        model.eval()
        round_id = len(events)
        probe_index = 0 if round_id == 0 else int(torch.randint(1, 2**31 - 1, ()).item())
        live_rng = rng_state()
        try:
            batches, indices = (comparison_batches, first_indices) if round_id == 0 else materialize_probe_batches(
                evaluation, train_ids, recipe, config, device, probe_index=probe_index)
            chosen, record, direction = projection_aware_search(model, batches, config,
                expected_sites=expected_sites, comparison_inputs=fixed_inputs)
        finally:
            # Probe/solver randomness does not rewind the live training stream.
            restore_rng(live_rng)
        previous = events[-1] if events else None
        same_batch_cosine = None
        difference_norm = None
        if direction is not None and previous_direction is not None:
            if direction.shape == previous_direction.shape and direction.norm() > 1e-12 and previous_direction.norm() > 1e-12:
                same_batch_cosine = float(cosine_similarity(direction.flatten(), previous_direction.flatten(), dim=0))
                difference_norm = float((direction - previous_direction).norm())
        record.update(search_round=round_id, post_fork_epoch=completed, epoch=epoch0 + completed,
            rollback_id=len(controller.rollback_events), anchor_epoch=controller.anchor['epoch'],
            probe_index=probe_index, probe_indices=indices,
            same_anchor_as_previous=previous is not None and record['anchor_hash'] == previous['anchor_hash'],
            same_block_as_previous=previous is not None and record['selected_block'] is not None and record['selected_block'] == previous['selected_block'],
            functional_delta_cosine_to_previous=same_batch_cosine,
            functional_delta_difference_norm=difference_norm,
            direction_comparison_role='same_fixed_inputs_diagnostic_only',
            new_best_since_last_intervention=False, epochs_until_new_best=None, epochs_until_rollback=None,
            **commit_choice(model, optimizer, chosen))
        immediate = evaluate_without_rng(model, evaluation_loader, device)
        record['validation_immediately_after_projection'] = immediate
        controller.observe(model, optimizer, scheduler, loader, immediate, epoch0 + completed,
                           completed, count_stall=False)
        record['best_anchor_after_commit'] = controller.anchor['validation']
        record['controller_after_search'] = controller.metrics()
        history[-1]['validation_after_search'] = immediate
        history[-1]['controller_after_search'] = controller.metrics()
        if completed > 0:
            history[-1].update(retriggered=True, search_round=round_id,
                intervention_count=sum(event['correction_applied'] for event in events) + int(record['correction_applied']))
        events.append(record)
        previous_direction = direction
        pending = False
        save()  # commit model, optimizer, RNG, round and controller together
        emit_event('projection_aware_intervention', {'method': args.method, **record}, output)

    save()  # pending search is recoverable, including immediately after a rollback
    if pending:
        search()
    for offset in range(completed + 1, args.post_fork_epochs + 1):
        started = time.perf_counter()
        lrs = [group['lr'] for group in optimizer.param_groups]
        training = train_epoch(model, loader, optimizer, device)
        validation = evaluate_without_rng(model, evaluation_loader, device)
        scheduler.step()
        improved = validation['accuracy'] > best_accuracy
        if improved:
            best_accuracy, best_loss, best_epoch = validation['accuracy'], validation['loss'], epoch0 + offset
        report_stall = 0 if improved else report_stall + 1
        row = {'method': args.method, 'phase': 'post_fork', 'seed': recipe.seed,
            'epoch': epoch0 + offset, 'post_fork_epoch': offset, 'post_fork_epochs': args.post_fork_epochs,
            'fork_epoch': epoch0, 'train_loss': training['loss'], 'train_accuracy': training['accuracy'],
            'validation_accuracy': validation['accuracy'], 'validation_loss': validation['loss'],
            'learning_rates': lrs, 'report_best_improved': improved, 'report_best_accuracy': best_accuracy,
            'report_best_loss': best_loss, 'report_best_epoch': best_epoch, 'report_stall_counter': report_stall,
            'scientific_escape': best_accuracy > accuracy, 'delta_vs_historical_best': best_accuracy - accuracy,
            'search_round': len(events) - 1, 'intervention_count': sum(event['correction_applied'] for event in events)}
        anchor_update = controller.observe(model, optimizer, scheduler, loader, validation, epoch0 + offset, offset)
        row.update(anchor_update, controller_stall_counter=controller.accuracy_stall_counter,
            rollback_triggered=anchor_update['rollback_applied'],
            next_learning_rates=[group['lr'] for group in optimizer.param_groups], retriggered=False)
        if anchor_update['controller_accuracy_improved']:
            events[-1]['new_best_since_last_intervention'] = True
            if events[-1]['epochs_until_new_best'] is None:
                events[-1]['epochs_until_new_best'] = offset - events[-1]['post_fork_epoch']
        if anchor_update['rollback_applied']:
            if events[-1]['epochs_until_rollback'] is None:
                events[-1]['epochs_until_rollback'] = offset - events[-1]['post_fork_epoch']
            pending = args.method == 'e2o_top3_recurrent' and offset < args.post_fork_epochs
            emit_event('deit_rollback', {'method': args.method, **controller.rollback_events[-1],
                'retrigger_pending': pending, 'rng_restored': False, 'loader_stream_restored': False}, output)
        completed = offset
        history.append(row)
        save()  # save rollback + pending flag before potentially expensive search
        if pending:
            search()
        emit_epoch(args.method, row, device, started, output)
    return finish(torch.load(latest, map_location='cpu', weights_only=False), output)


def finish(state, output):
    output = Path(output)
    identity = state['run_identity']; events = state['interventions']
    result = {key: state[key] for key in ('protocol', 'run_identity', 'history', 'interventions',
        'historical_best_accuracy', 'historical_best_loss', 'historical_best_epoch',
        'report_best_accuracy', 'report_best_loss', 'report_best_epoch')}
    result.update(method=identity['method'], theta_best_hash=identity['fork_hash'],
        fork_epoch=state['historical_best_epoch'], completed_epochs=state['completed_epochs'],
        post_fork_epochs=identity['post_fork_epochs'], best_postfork_observed={
            'accuracy': state['report_best_accuracy'], 'loss': state['report_best_loss'], 'epoch': state['report_best_epoch']},
        best_anchor_stored={**state['e_controller']['anchor']['validation'], 'epoch': state['e_controller']['anchor']['epoch']},
        delta_vs_historical_best=state['report_best_accuracy'] - state['historical_best_accuracy'],
        scientific_escape=state['report_best_accuracy'] > state['historical_best_accuracy'],
        proposals_tried=sum(len(event['candidates']) for event in events),
        projection_candidates_tried=sum(len(event['candidates']) for event in events),
        where_candidates_evaluated=sum(event['where'].get('selection_candidate_count', 0) for event in events),
        accepted_candidate_count=sum(event['accepted_count'] for event in events),
        corrections_accepted=sum(event['correction_applied'] for event in events),
        search_rounds=len(events), search_seconds=sum(event['search_seconds'] for event in events),
        rollback_events=state['e_controller']['rollback_events'], retrigger=identity['retrigger'])
    export_events(events, output / 'interventions.jsonl')
    write_csv(state['history'], output / 'epoch_history.csv')
    atomic_torch_save({**state['e_controller']['anchor'], 'kind': 'deit_projection_aware_anchor',
        'protocol': state['protocol'], 'run_identity': identity}, output / 'best_checkpoint.pt')
    atomic_json_save(result, output / 'result.json')
    atomic_json_save({'run_identity': identity, 'protocol': state['protocol'],
        'fork_epoch': state['historical_best_epoch'], 'historical_best_accuracy': state['historical_best_accuracy'],
        'historical_best_loss': state['historical_best_loss']}, output / 'run_metadata.json')
    emit_event('deit_arm_complete', {key: result[key] for key in ('method', 'report_best_accuracy',
        'report_best_epoch', 'scientific_escape', 'search_rounds', 'corrections_accepted')}, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--plateau-checkpoint', type=Path, required=True)
    parser.add_argument('--plateau-checkpoint-hash', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--post-fork-epochs', type=int, default=150)
    parser.add_argument('--algorithm-patience', type=int, default=10)
    args = parser.parse_args()
    if sha256_file(args.plateau_checkpoint) != args.plateau_checkpoint_hash:
        raise ValueError('fork SHA256 mismatch')
    fork, recipe = checked_source(args.plateau_checkpoint, {'deit_plateau_fork'})
    validate_seed2_fork(fork, recipe)
    if args.post_fork_epochs != 150 or args.algorithm_patience != 10:
        parser.error('Seed2 protocol fixes horizon150 and patience10')
    run(args, fork, recipe, res18_cp_config())


if __name__ == '__main__':
    main()
