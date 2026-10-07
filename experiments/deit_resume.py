"""CPU discovery/import of complete, identity-bound DeiT training checkpoints."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil

from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.deit_protocol import protocol, canonical_model_config, checked_source
from experiments.deit_ablation_runner import arm_identity, completed_arm_result
from experiments.shared_protocol import atomic_json_save, sha256_file

KINDS = {'deit_plateau_fork', 'deit_vanilla_latest', 'deit_vanilla_best',
         'deit_vanilla_reference', 'deit_fork_arm_latest', 'deit_raw_intervention_reference'}
FULL_STATE = {'model', 'optimizer', 'scheduler', 'rng', 'train_loader_generator_state',
              'train_indices', 'evaluation_indices', 'source_tuning_indices', 'trigger_indices'}


def unique(items, label):
    distinct = {item['sha256']: item for item in items}
    if len(distinct) > 1:
        raise ValueError(f'Ambiguous {label}: attach one experiment trajectory')
    return next(iter(distinct.values()), None)


def most_advanced(items, key, label):
    if not items:
        return None
    progress = max(item['payload'][key] for item in items)
    return unique([item for item in items if item['payload'][key] == progress], label)


def prepare_resume(roots, output, recipe, config, *, horizon, patience=10, inner_steps=5,
                   arms=()):
    """Import original checkpoint bytes; full state remains on disk, metadata in RAM.

    Hash-bound Phase2 needs original .pt files or an output archive. A repacked
    Kaggle directory has a different hash and cannot silently replace its fork.
    """
    output = Path(output)
    expected = protocol(recipe, canonical_model_config())
    plan = {'phase1': 'start', 'arms': {}, 'rejected': []}
    def compatible(state):
        if state['kind'] == 'deit_raw_intervention_reference':
            return (state.get('suite_version') == 1 and state.get('seed') == recipe.seed
                    and state.get('cp_config') == asdict(config))
        return state.get('protocol') == expected and FULL_STATE.issubset(state)
    def compact(state):
        if state['kind'] == 'deit_raw_intervention_reference':
            return {key: state[key] for key in ('kind', 'suite_version', 'seed', 'cp_config', 'fork_hash')}
        return {key: value for key, value in state.items()
                if key not in FULL_STATE and key != 'e_controller'}
    states = []
    # Include working output, so attachment import never downgrades local progress.
    for index, root in enumerate(dict.fromkeys(map(str, [output, *roots]))):
        if not Path(root).exists():
            continue
        found, rejected = discover_checkpoints(root, output / '_resume_cache' / str(index), kind=KINDS,
            payload_filter=compatible, payload_transform=compact)
        states.extend(found)
        plan['rejected'].extend(rejected)
    def typed(kind):
        return [item for item in states if item['payload']['kind'] == kind]
    def copy(item, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if item['path'].resolve() != path.resolve():
            temporary = path.with_name(path.name + '.importing')
            try:
                shutil.copy2(item['path'], temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return path
    phase1 = output / 'vanilla_stall'
    fork_path = phase1 / 'plateau_checkpoint.pt'
    fork_item = unique(typed('deit_plateau_fork'), 'historical-best fork')
    arm_states = typed('deit_fork_arm_latest')
    if not fork_item:
        if arm_states:
            raise FileNotFoundError('Arm progress exists but matching original plateau_checkpoint.pt is missing')
        latest = most_advanced(typed('deit_vanilla_latest'), 'epoch', 'Phase1 latest')
        if latest:
            meta = latest['payload']
            best = unique([item for item in typed('deit_vanilla_best')
                if item['payload']['epoch'] == meta['historical_best_epoch']
                and item['payload']['history'] == meta['history'][:len(item['payload']['history'])]], 'Phase1 best')
            if not best:
                raise FileNotFoundError('Phase1 resume requires checkpoint_latest.pt and matching full checkpoint_best.pt')
            copy(latest, phase1 / 'checkpoint_latest.pt')
            copy(best, phase1 / 'checkpoint_best.pt')
            plan.update(phase1='resume', phase1_epoch=meta['epoch'])
    else:
        copy(fork_item, fork_path)
        fork, _ = checked_source(fork_path, {'deit_plateau_fork'})
        fork_hash = sha256_file(fork_path)
        plan.update(phase1='reuse_fork', fork_epoch=fork['epoch'], fork_hash=fork_hash)
        # Never restart an existing trajectory when its hash-bound dependencies are missing.
        if any(item['payload']['run_identity']['fork_hash'] != fork_hash for item in arm_states):
            raise ValueError('Attached arm/fork hashes differ; attach original .pt or output .tar.gz, not repacked files')
        vanilla_path = phase1 / 'vanilla_reference.pt'
        reference = unique([item for item in typed('deit_vanilla_reference')
            if item['payload'].get('theta_best_hash') == fork_hash], 'Vanilla reference')
        if reference:
            copy(reference, vanilla_path)
        else:
            terminal = unique([item for item in typed('deit_vanilla_latest')
                if item['payload']['epoch'] == fork['epoch'] + horizon
                and item['payload']['history'] == fork['history'] + fork['vanilla_history']], 'Phase1 terminal')
            if not terminal:
                raise FileNotFoundError('Attach matching vanilla_reference.pt or full Phase1 terminal checkpoint')
            from experiments.deit_vanilla_reference import create_vanilla_reference
            create_vanilla_reference(fork_path, terminal['path'], vanilla_path)
        raw = unique([item for item in typed('deit_raw_intervention_reference')
            if item['payload']['fork_hash'] == fork_hash], 'raw reference')
        raw_path = output / 'raw_intervention_reference.pt'
        if not raw and arm_states:
            raise FileNotFoundError('Arm resume requires the original raw_intervention_reference.pt; do not rebuild it')
        if raw:
            copy(raw, raw_path)
            raw_hash = sha256_file(raw_path)
            for method in arms:
                identity = arm_identity(config, fork_hash, method, horizon, patience, inner_steps)
                identity['raw_reference_hash'] = raw_hash
                candidates = [item for item in arm_states if item['payload']['run_identity'].get('method') == method]
                if any(item['payload']['run_identity'] != identity for item in candidates):
                    raise ValueError(f'{method}: incompatible horizon/config/controller/raw reference; use a separate output')
                selected = most_advanced(candidates, 'completed_epochs', method)
                if not selected:
                    plan['arms'][method] = {'action': 'start', 'completed_epochs': 0}
                    continue
                destination = output / 'arms' / method / 'checkpoint_latest.pt'
                state, _ = checked_source(selected['path'], {'deit_fork_arm_latest'})
                count = state['completed_epochs']
                if (not 0 <= count <= horizon or len(state['history']) != count + 1
                        or state['epoch'] != fork['epoch'] + count):
                    raise ValueError(f'{method}: inconsistent progress/history')
                if method != 'vanilla_continue':
                    if len(state['interventions']) != 1:
                        raise ValueError(f'{method}: missing intervention state')
                    needs_controller = (method != 'o_projection_only'
                        and state['interventions'][0].get('fallback') != 'vanilla_no_rollback')
                    if needs_controller and not state.get('e_controller'):
                        raise ValueError(f'{method}: missing rollback controller state')
                    if state.get('e_controller'):
                        from experiments.deit_e_rollback import EAccuracyRollback
                        controller = state['e_controller']
                        if not {'model', 'optimizer', 'scheduler'}.issubset(controller['anchor']):
                            raise ValueError(f'{method}: incomplete rollback anchor')
                        EAccuracyRollback.from_state(controller, patience, count, stall_on_anchor=True)
                copy(selected, destination)
                action = 'completed' if count == horizon and method != 'vanilla_continue' else 'resume'
                if action == 'completed':
                    completed_arm_result(state, destination.parent)
                plan['arms'][method] = {'action': action, 'completed_epochs': count}
                del state
    atomic_json_save(plan, output / 'resume_plan.json')
    print('Resume plan:', {key: value for key, value in plan.items() if key != 'rejected'}, flush=True)
    return plan
