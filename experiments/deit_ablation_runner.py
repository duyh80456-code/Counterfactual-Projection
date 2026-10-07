"""Full-state isolated ablation arms, one intervention and matched controller rules."""
from __future__ import annotations

from dataclasses import asdict
import json
import time
import torch
from adapters.deit_ablation import SUITE_VERSION, execute_intervention
from adapters.deit_opt_e import OptEConfig
from experiments.deit_protocol import (checked_source, load_training_context, historical_best,
    materialize_probe_batches, evaluate_without_rng, save_state)
from experiments.deit_e_rollback import EAccuracyRollback
from experiments.deit_logging import emit_event, emit_epoch
from experiments.shared_protocol import seed_everything, rng_state, restore_rng, atomic_json_save, atomic_torch_save


def arm_identity(config, fork_hash, method, horizon, patience, inner_steps):
    opt_config = OptEConfig(inner_steps=inner_steps)
    identity = {'suite_version': SUITE_VERSION, 'fork_hash': fork_hash, 'method': method,
                'post_fork_epochs': horizon, 'cp_config': json.loads(json.dumps(asdict(config))),
                'controller_policy': 'anchor_rollback_no_retrigger_except_A0_A1_or_failed_opt_e',
                'algorithm_patience': patience, 'opt_e_config': json.loads(json.dumps(asdict(opt_config)))}
    return identity


def completed_arm_result(saved, output):
    """Rebuild a completed non-Vanilla result without data, model or CUDA."""
    identity = saved['run_identity']
    controller = saved.get('e_controller')
    history, interventions = saved['history'], saved['interventions']
    last = history[-1]
    accuracy = saved['historical_best_accuracy']
    result = {key: saved[key] for key in (
        'method', 'protocol', 'run_identity', 'historical_best_accuracy', 'historical_best_loss',
        'historical_best_epoch', 'validation_before', 'validation_immediately_after_projection',
        'report_best_accuracy', 'report_best_loss', 'report_best_epoch', 'history', 'interventions')}
    result.update(run_id=str(output.resolve()), theta_best_hash=identity['fork_hash'],
        fork_epoch=saved['historical_best_epoch'], post_fork_epochs=identity['post_fork_epochs'],
        validation_1_to_5_epochs_after=interventions[0]['validation_1_to_5_epochs_after'],
        delta_vs_historical_best=saved['report_best_accuracy'] - accuracy,
        scientific_escape=saved['report_best_accuracy'] > accuracy,
        report_best_scope='epochs1_to_K_epoch0_separate',
        final_validation_accuracy=last.get('state_validation_accuracy', last['validation_accuracy']),
        final_validation_loss=last.get('state_validation_loss', last['validation_loss']),
        last_observed_validation_accuracy=last['validation_accuracy'],
        last_observed_validation_loss=last['validation_loss'], persistent_growth=saved.get('persistent_growth'),
        controller=controller['protocol'] if controller else None,
        rollback_events=controller['rollback_events'] if controller else [],
        validation_role='report_and_anchor_selection' if controller else 'report_only',
        intervention_count=1, retrigger=False)
    atomic_json_save(result, output / 'result.json')
    return result


def run_arm(args, config, fork, recipe, fork_hash):
    method = 'e_driven_o_raw' if args.method == 'ours_e_driven_o' else args.method
    opt_config = OptEConfig(inner_steps=args.opt_inner_steps)
    identity = arm_identity(config, fork_hash, method, args.post_fork_epochs,
                            args.algorithm_patience, args.opt_inner_steps)
    if method == 'vanilla_continue':
        from experiments.deit_vanilla_reference import export_reused_vanilla_arm
        where_table = None
        if args.raw_reference:
            from experiments.shared_protocol import sha256_file
            identity['raw_reference_hash'] = sha256_file(args.raw_reference)
            raw = torch.load(args.raw_reference, map_location='cpu', weights_only=False)
            if (raw.get('kind') != 'deit_raw_intervention_reference' or raw.get('fork_hash') != fork_hash or
                    raw.get('cp_config') != asdict(config) or raw.get('suite_version') != SUITE_VERSION):
                raise ValueError('Vanilla offline WHERE reference mismatch')
            where_table = raw['record']['site_evaluations']
        path = args.vanilla_reference or args.plateau_checkpoint.parent / 'vanilla_reference.pt'
        reference, _ = checked_source(path, {'deit_vanilla_reference'})
        if args.resume:
            saved, _ = checked_source(args.resume, {'deit_fork_arm_latest'})
            if saved.get('run_identity') != identity:
                raise ValueError('Vanilla suite resume identity mismatch')
        result = export_reused_vanilla_arm(fork, reference, args.output, identity)
        if where_table is not None:
            result.update(site_evaluations=where_table, where_role='offline_only_did_not_select_vanilla')
            atomic_json_save(result, args.output / 'result.json')
        emit_event("vanilla_continue", {"phase": "observed_reference_export", "fork_epoch": result["fork_epoch"],
            "post_fork_epochs": result["post_fork_epochs"], "report_best_accuracy": result["report_best_accuracy"],
            "report_best_loss": result["report_best_loss"], "report_best_epoch": result["report_best_epoch"],
            "scientific_escape": result["scientific_escape"], "additional_training_epochs": 0}, args.output)
        return result
    if args.raw_reference is None or not args.data_root:
        raise ValueError('ablation arms require --raw-reference and --data-root')
    from experiments.shared_protocol import sha256_file
    identity['raw_reference_hash'] = sha256_file(args.raw_reference)
    reference = torch.load(args.raw_reference, map_location='cpu', weights_only=False)
    if (reference.get('kind') != 'deit_raw_intervention_reference' or reference.get('suite_version') != SUITE_VERSION or
            reference['fork_hash'] != fork_hash or reference['cp_config'] != asdict(config) or reference['seed'] != recipe.seed):
        raise ValueError('raw reference fork/config/seed mismatch')
    saved = None
    if args.resume:
        saved, _ = checked_source(args.resume, {'deit_fork_arm_latest'})
        if saved.get('run_identity') != identity or saved['protocol'] != fork['protocol']:
            raise ValueError('ablation resume identity mismatch')
        if (not 0 <= saved['completed_epochs'] <= args.post_fork_epochs or
                len(saved['history']) != saved['completed_epochs'] + 1 or len(saved['interventions']) != 1 or
                saved['epoch'] != fork['epoch'] + saved['completed_epochs']):
            raise ValueError('inconsistent ablation resume state')
    if saved and saved['completed_epochs'] == args.post_fork_epochs:
        result = completed_arm_result(saved, args.output)
        emit_event('deit_resume_complete', {'method': method, 'additional_training_epochs': 0}, args.output)
        return result
    seed_everything(recipe.seed)
    device = torch.device(args.device)
    context = load_training_context(args.data_root, recipe, device, saved or fork)
    model, optimizer, scheduler, loader, eval_loader, evaluation, train_ids, val_ids, tuning_ids, reserved_ids = context
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    emit_event('deit_run_start', {'method': method, 'architecture': fork['protocol']['architecture'],
        'seed': recipe.seed, 'fork_epoch': fork['epoch'], 'resume_offset': saved['completed_epochs'] if saved else 0,
        'post_fork_epochs': args.post_fork_epochs, 'algorithm_patience': args.algorithm_patience, 'retrigger': False}, args.output)
    accuracy, loss, epoch0 = historical_best(fork)
    before = saved['validation_before'] if saved else evaluate_without_rng(model, eval_loader, device)
    controller = None
    controller_enabled = method != 'o_projection_only'
    if saved:
        if saved.get('e_controller'):
            controller = EAccuracyRollback.from_state(saved['e_controller'], args.algorithm_patience,
                                                       saved['completed_epochs'], stall_on_anchor=True)
    elif controller_enabled and method != 'persistent_growth':
        controller = EAccuracyRollback(model, optimizer, scheduler, loader, before, epoch0,
                                       args.algorithm_patience, stall_on_anchor=True)
    history = list(saved['history']) if saved else []
    interventions = list(saved['interventions']) if saved else []
    persistent = saved.get('persistent_growth') if saved else None
    immediate = saved['validation_immediately_after_projection'] if saved else None
    start = saved['completed_epochs'] if saved else 0
    best_accuracy = saved['report_best_accuracy'] if saved else float('-inf')
    best_loss = saved['report_best_loss'] if saved else float('inf')
    best_epoch = saved['report_best_epoch'] if saved else epoch0
    report_stall = saved.get('report_stall_counter', 0) if saved else 0
    def growth_fallback():
        nonlocal controller
        controller = EAccuracyRollback(model, optimizer, scheduler, loader, before, epoch0,
                                       args.algorithm_patience, stall_on_anchor=True)
    if not saved:
        live_rng = rng_state()
        try:
            batches, indices = materialize_probe_batches(evaluation, train_ids, recipe, config, device, include_opt=True)
            if indices != reference['probe_indices']:
                raise ValueError('reference probe partition differs from arm')
            record = execute_intervention(model, optimizer, batches, config, method, reference,
                seed=recipe.seed * 10000 + 4701, opt_config=opt_config, growth_fallback=growth_fallback)
        finally:
            restore_rng(live_rng)
        persistent = record.get('persistent')
        if record.get('fallback') == 'vanilla_no_rollback':
            controller = None
        immediate = evaluate_without_rng(model, eval_loader, device)
        record.update(epoch=epoch0, probe_index=0, probe_indices=indices,
                      validation_before=before, validation_immediately_after_projection=immediate,
                      validation_1_to_5_epochs_after=[])
        interventions.append(record)
        emit_event('e_driven_o_intervention' if method.startswith('e_driven_o') else 'deit_intervention',
                   {'method': method, **record}, args.output)
        row = {'epoch': epoch0, 'post_fork_epoch': 0,
               'validation_accuracy': immediate['accuracy'], 'validation_loss': immediate['loss']}
        if controller:
            row.update(controller.observe(model, optimizer, scheduler, loader, immediate, epoch0, 0, count_stall=False))
            row['controller_stall_counter'] = controller.accuracy_stall_counter
        row['report_stall_counter'] = report_stall
        history.append(row)

    def save(completed):
        save_state(args.output / 'checkpoint_latest.pt', model=model, optimizer=optimizer, scheduler=scheduler,
            loader=loader, epoch=epoch0 + completed, history=history,
            train_indices=train_ids, evaluation_indices=val_ids, source_tuning_indices=tuning_ids,
            trigger_indices=reserved_ids, run_protocol=fork['protocol'], kind='deit_fork_arm_latest',
            run_identity=identity, method=method, completed_epochs=completed, interventions=interventions,
            validation_before=before, validation_immediately_after_projection=immediate,
            historical_best_accuracy=accuracy, historical_best_loss=loss, historical_best_epoch=epoch0,
            report_best_accuracy=best_accuracy, report_best_loss=best_loss, report_best_epoch=best_epoch,
            report_stall_counter=report_stall,
            persistent_growth=persistent, e_controller=controller.state_dict() if controller else None)
        if controller:
            atomic_torch_save({**controller.anchor, 'kind': 'deit_e_controller_best', 'run_identity': identity,
                'protocol': fork['protocol'], 'persistent_growth': persistent}, args.output / 'checkpoint_best.pt')
    save(start)
    from experiments.shared_protocol import train_epoch
    for offset in range(start + 1, args.post_fork_epochs + 1):
        epoch_started = time.perf_counter()
        training_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        train = train_epoch(model, loader, optimizer, device)
        validation = evaluate_without_rng(model, eval_loader, device)
        scheduler.step()
        row = {'epoch': epoch0 + offset, 'post_fork_epoch': offset,
               'train_loss': train['loss'], 'train_accuracy': train['accuracy'],
               'validation_accuracy': validation['accuracy'], 'validation_loss': validation['loss'],
               'learning_rates': training_lrs, 'next_learning_rates': [float(g['lr']) for g in optimizer.param_groups],
               'phase': 'post_fork', 'method': method, 'architecture': fork['protocol']['architecture'], 'seed': recipe.seed,
               'fork_epoch': epoch0, 'post_fork_epochs': args.post_fork_epochs,
               'intervention_count': len(interventions), 'retriggered': False}
        report_improved = validation['accuracy'] > best_accuracy
        if report_improved:
            best_accuracy, best_loss, best_epoch = validation['accuracy'], validation['loss'], epoch0 + offset
        report_stall = 0 if report_improved else report_stall + 1
        row.update(report_best_improved=report_improved, report_best_accuracy=best_accuracy,
                   report_best_loss=best_loss, report_best_epoch=best_epoch, report_stall_counter=report_stall)
        if controller:
            row.update(controller.observe(model, optimizer, scheduler, loader, validation, epoch0 + offset, offset))
            row['controller_stall_counter'] = controller.accuracy_stall_counter
        row['rollback_triggered'] = bool(row.get('rollback_applied', False))
        if not controller:
            row.update(controller_anchor_accuracy=None, controller_anchor_loss=None, controller_anchor_epoch=None,
                controller_anchor_improved=False, controller_anchor_reason='controller_disabled',
                controller_stall_counter=None, rollback_count=0)
        else:
            row['controller_anchor_update_reason'] = row['controller_anchor_reason']
            row['next_learning_rates'] = [float(g['lr']) for g in optimizer.param_groups]
        row['delta_vs_historical_best'] = best_accuracy - accuracy
        row['scientific_escape'] = best_accuracy > accuracy
        if row['rollback_triggered']:
            emit_event('deit_rollback', {'method': method, **controller.rollback_events[-1],
                'rng_restored': False, 'loader_stream_restored': False, 'retriggered': False}, args.output)
        history.append(row)
        interventions[0]['validation_1_to_5_epochs_after'] = [r for r in history if 1 <= r['post_fork_epoch'] <= 5]
        save(offset)
        emit_epoch(method, row, device, epoch_started, args.output)
    last = history[-1]
    result = {'method': method, 'protocol': fork['protocol'], 'run_identity': identity,
        'run_id': str(args.output.resolve()), 'theta_best_hash': fork_hash, 'fork_epoch': epoch0,
        'historical_best_accuracy': accuracy, 'historical_best_loss': loss, 'historical_best_epoch': epoch0,
        'validation_before': before, 'validation_immediately_after_projection': immediate,
        'validation_1_to_5_epochs_after': interventions[0]['validation_1_to_5_epochs_after'],
        'report_best_accuracy': best_accuracy, 'report_best_loss': best_loss, 'report_best_epoch': best_epoch,
        'delta_vs_historical_best': best_accuracy - accuracy, 'scientific_escape': best_accuracy > accuracy,
        'report_best_scope': 'epochs1_to_K_epoch0_separate',
        'history': history, 'interventions': interventions, 'post_fork_epochs': args.post_fork_epochs,
        'final_validation_accuracy': last.get('state_validation_accuracy', last['validation_accuracy']),
        'final_validation_loss': last.get('state_validation_loss', last['validation_loss']),
        'last_observed_validation_accuracy': last['validation_accuracy'],
        'last_observed_validation_loss': last['validation_loss'], 'persistent_growth': persistent,
        'controller': controller.protocol if controller else None,
        'rollback_events': controller.rollback_events if controller else [],
        'validation_role': 'report_and_anchor_selection' if controller else 'report_only',
        'intervention_count': 1, 'retrigger': False}
    atomic_json_save(result, args.output / 'result.json')
    emit_event('deit_arm_complete', {'method': method, 'report_best_accuracy': best_accuracy,
        'report_best_loss': best_loss, 'report_best_epoch': best_epoch,
        'delta_vs_historical_best': best_accuracy - accuracy, 'scientific_escape': best_accuracy > accuracy,
        'rollback_count': len(controller.rollback_events) if controller else 0}, args.output)
    return result
