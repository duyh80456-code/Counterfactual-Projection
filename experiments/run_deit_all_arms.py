"""Confirm a 150-epoch validation-best stall, reuse Vanilla, then run isolated DeiT arms."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
import threading
from pathlib import Path
import subprocess
import sys

import torch
from adapters.deit_ablation import res18_cp_config, build_raw_reference, SUITE_VERSION
from experiments.deit_protocol import (DeitRecipe, checked_source, load_training_context,
    materialize_probe_batches)
from experiments.shared_protocol import atomic_json_save, atomic_torch_save, sha256_file, seed_everything
from experiments.deit_logging import emit_event

ALL_ARMS = ('vanilla_continue', 'o_projection_only', 'e_driven_o_raw', 'e_driven_o_normalized',
            'random_control_parameter', 'random_control_logit', 'persistent_growth', 'opt_e')

SUPPORTED_ARMS = ALL_ARMS + ('vanilla_rollback', 'o_projection_only_rollback')

def run_jobs(commands, manifest_path, *, runner=subprocess.run):
    """A failed arm is recorded; every independent remaining arm still runs."""
    statuses = {}
    for method, command in commands.items():
        emit_event('deit_arm_start', {'method': method}, Path(manifest_path).parent)
        try:
            runner(command, check=True)
            statuses[method] = {'status': 'completed'}
        except Exception as error:
            statuses[method] = {'status': 'failed', 'error': repr(error)}
        emit_event('deit_arm_status', {'method': method, **statuses[method]}, Path(manifest_path).parent)
        atomic_json_save(statuses, manifest_path)
    return statuses



def gpu_slots(spec, device):
    if torch.device(device).type != 'cuda':
        return []
    count = torch.cuda.device_count()
    slots = list(range(min(2, count))) if spec == 'auto' else [int(value) for value in spec.split(',')]
    if not slots or len(set(slots)) != len(slots) or any(slot < 0 or slot >= count for slot in slots):
        raise ValueError('GPU devices must be distinct visible CUDA indices; auto needs at least one GPU')
    return slots


def child_environment(slot, parent_env):
    env = dict(parent_env)
    if slot is None:
        env['CUDA_VISIBLE_DEVICES'] = ''  # A0 reference export is CPU-only.
    else:
        visible = parent_env.get('CUDA_VISIBLE_DEVICES')
        tokens = [value.strip() for value in visible.split(',')] if visible else None
        env['CUDA_VISIBLE_DEVICES'] = tokens[slot] if tokens is not None else str(slot)
    env['PYTHONUNBUFFERED'] = '1'
    return env


def command_on_device(command, slot):
    command = list(command)
    device = 'cpu' if slot is None else 'cuda:0'  # One visible physical GPU in each child.
    if '--device' in command:
        command[command.index('--device') + 1] = device
    else:
        command += ['--device', device]
    return command


def run_jobs_parallel(commands, manifest_path, slots, *, runner=None, parent_env=None):
    """One isolated process per GPU, refill free slots; CPU A0 first, Opt-E last."""
    if not slots or len(slots) != len(set(slots)):
        raise ValueError('parallel scheduler needs distinct GPU slots')
    env = dict(os.environ if parent_env is None else parent_env)
    statuses, print_lock = {}, threading.Lock()
    output = Path(manifest_path).parent
    def launch(method, command, slot):
        child_env = child_environment(slot, env)
        command = command_on_device(command, slot)
        if runner is not None:
            runner(command, check=True, env=child_env)
            return
        process = subprocess.Popen(command, env=child_env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        label = 'CPU' if slot is None else f'GPU{slot}'
        try:
            for line in process.stdout:
                with print_lock:
                    print(f'[{label} {method}] {line.rstrip()}', flush=True)
            code = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    def record(method, status, slot, error=None):
        statuses[method] = {'status': status, 'gpu_slot': slot,
            'child_device': 'cpu' if slot is None else 'cuda:0'}
        if error is not None:
            statuses[method]['error'] = repr(error)
        with print_lock:
            emit_event('deit_arm_start' if status == 'running' else 'deit_arm_status',
                       {'method': method, **statuses[method]}, output)
        atomic_json_save(statuses, manifest_path)
    # CPU export does not consume a GPU worker and never retrains Vanilla.
    if 'vanilla_continue' in commands:
        method = 'vanilla_continue'
        record(method, 'running', None)
        try:
            launch(method, commands[method], None)
            record(method, 'completed', None)
        except Exception as error:
            record(method, 'failed', None, error)
    regular = deque((method, command) for method, command in commands.items()
                    if method not in ('vanilla_continue', 'opt_e'))
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        active, free = {}, deque(slots)
        while regular or active:
            while regular and free:
                slot = free.popleft()
                method, command = regular.popleft()
                record(method, 'running', slot)
                active[executor.submit(launch, method, command, slot)] = (method, slot)
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda future: active[future][1]):
                method, slot = active.pop(future)
                try:
                    future.result()
                    record(method, 'completed', slot)
                except Exception as error:
                    record(method, 'failed', slot, error)
                free.append(slot)
        # Barrier: Opt-E begins only after every other requested arm finished/failed.
        if 'opt_e' in commands:
            method, slot = 'opt_e', slots[0]
            record(method, 'running', slot)
            try:
                executor.submit(launch, method, commands[method], slot).result()
                record(method, 'completed', slot)
            except Exception as error:
                record(method, 'failed', slot, error)
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0', help='Phase1/reference device')
    parser.add_argument('--gpu-devices', default='auto',
        help='Phase2 visible GPU indices, e.g. 0,1; auto uses up to two GPUs')
    parser.add_argument('--plateau-checkpoint', type=Path)
    parser.add_argument('--vanilla-reference', type=Path)
    parser.add_argument('--resume-phase1', type=Path)
    parser.add_argument('--resume-root', type=Path, action='append', default=[],
        help='Attached output directory/archive directory; repeat for multiple mounts')
    parser.add_argument('--schedule-epochs', type=int, default=300)
    parser.add_argument('--max-epoch', type=int, default=800)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--stall-patience', type=int, default=150)
    parser.add_argument('--post-fork-epochs', type=int, default=150)
    parser.add_argument('--algorithm-patience', type=int, default=10)
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--projection-samples', type=int, default=32)
    parser.add_argument('--scales', default='.0125,.025,.05')
    parser.add_argument('--opt-inner-steps', type=int, default=5)
    parser.add_argument('--arms', default=','.join(ALL_ARMS))
    args = parser.parse_args()
    arms = args.arms.split(',')
    if not arms or any(arm not in SUPPORTED_ARMS for arm in arms) or len(set(arms)) != len(arms):
        parser.error('invalid or duplicate arm list')
    config = replace(res18_cp_config(), rank=args.rank, projection_samples=args.projection_samples,
                     scales=tuple(map(float, args.scales.split(','))))
    recipe = DeitRecipe(seed=args.seed, batch_size=args.batch_size, learning_rate=args.learning_rate,
        schedule_epochs=args.schedule_epochs, stall_start_epoch=0,
        stall_patience=args.stall_patience, reference_epochs=args.post_fork_epochs, max_epoch=args.max_epoch)
    recipe.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    phase1 = args.output / 'vanilla_stall'
    # Preserve explicit CLI checkpoint support while validating the local import set.
    if args.plateau_checkpoint:
        declared, declared_recipe = checked_source(args.plateau_checkpoint, {'deit_plateau_fork'})
        if declared_recipe != recipe:
            raise ValueError('explicit fork recipe differs from declared recipe')
        phase1.mkdir(parents=True, exist_ok=True)
        for source, name in ((args.plateau_checkpoint, 'plateau_checkpoint.pt'),
                             (args.vanilla_reference, 'vanilla_reference.pt')):
            if source and source.resolve() != (phase1 / name).resolve():
                destination = phase1 / name
                if destination.exists() and sha256_file(destination) != sha256_file(source):
                    raise ValueError('explicit checkpoint conflicts with local output')
                shutil.copy2(source, destination)
        args.plateau_checkpoint = phase1 / 'plateau_checkpoint.pt'
        if args.vanilla_reference:
            args.vanilla_reference = phase1 / 'vanilla_reference.pt'
        del declared
    from experiments.deit_resume import prepare_resume
    plan = prepare_resume(args.resume_root, args.output, recipe, config,
        horizon=args.post_fork_epochs, patience=args.algorithm_patience,
        inner_steps=args.opt_inner_steps, arms=arms)
    fork_path = args.plateau_checkpoint or (phase1 / 'plateau_checkpoint.pt'
        if (phase1 / 'plateau_checkpoint.pt').exists() else None)
    if fork_path is None:
        command = [sys.executable, '-m', 'experiments.train_deit_plateau', '--data-root', args.data_root,
                   '--output', str(phase1), '--device', args.device]
        for key, value in asdict(recipe).items():
            command += ['--' + key.replace('_', '-'), str(value)]
        resume = args.resume_phase1 or (phase1 / 'checkpoint_latest.pt' if (phase1 / 'checkpoint_latest.pt').exists() else None)
        if resume:
            command += ['--resume', str(resume)]
        subprocess.run(command, check=True)
        fork_path = phase1 / 'plateau_checkpoint.pt'
    fork, actual_recipe = checked_source(fork_path, {'deit_plateau_fork'})
    if actual_recipe != recipe:
        raise ValueError('attached fork recipe does not match the declared all-arm recipe')
    fork_hash = sha256_file(fork_path)
    reference_path = args.output / 'raw_intervention_reference.pt'
    if reference_path.exists():
        reference = torch.load(reference_path, map_location='cpu', weights_only=False)
        if (reference.get('fork_hash') != fork_hash or reference.get('cp_config') != asdict(config) or
                reference.get('suite_version') != SUITE_VERSION or reference.get('seed') != recipe.seed):
            raise ValueError('existing raw reference belongs to a different experiment')
    else:
        seed_everything(recipe.seed)
        device = torch.device(args.device)
        context = load_training_context(args.data_root, recipe, device, fork)
        model = context[0]
        model.eval()
        batches, indices = materialize_probe_batches(context[5], context[6], recipe, config, device, include_opt=True)
        reference = build_raw_reference(model, batches, config, fork_hash, seed=recipe.seed, indices=indices)
        atomic_torch_save(reference, reference_path)
        del context, batches, model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    vanilla_path = args.vanilla_reference or fork_path.parent / 'vanilla_reference.pt'
    commands, completed_statuses = {}, {}
    for method in arms:
        if plan['arms'].get(method, {}).get('action') == 'completed':
            completed_statuses[method] = {'status': 'completed', 'resumed_completed': True}
            emit_event('deit_arm_skip_completed', {'method': method, 'additional_training_epochs': 0}, args.output)
            continue
        output = args.output / 'arms' / method
        command = [sys.executable, '-m', 'experiments.run_deit_fork', '--ablation-suite',
            '--method', method, '--data-root', args.data_root, '--plateau-checkpoint', str(fork_path),
            '--plateau-checkpoint-hash', fork_hash, '--output', str(output), '--device', args.device,
            '--post-fork-epochs', str(args.post_fork_epochs), '--algorithm-patience', str(args.algorithm_patience),
            '--opt-inner-steps', str(args.opt_inner_steps)]
        for key, value in asdict(config).items():
            command += ['--' + key.replace('_', '-'), ','.join(map(str, value)) if key == 'scales' else str(value)]
        command += ['--raw-reference', str(reference_path)]
        if method == 'vanilla_continue':
            command += ['--vanilla-reference', str(vanilla_path)]
        if (output / 'checkpoint_latest.pt').exists():
            command += ['--resume', str(output / 'checkpoint_latest.pt')]
        commands[method] = command
    slots = gpu_slots(args.gpu_devices, args.device) if commands else []
    atomic_json_save({'suite_version': SUITE_VERSION, 'fork_hash': fork_hash, 'recipe': asdict(recipe),
                     'cp_config': asdict(config), 'arms': arms, 'commands': commands,
                     'execution': {'gpu_slots': slots, 'one_visible_gpu_per_arm': bool(slots),
                                   'opt_e_last_barrier': True}}, args.output / 'suite_protocol.json')
    statuses = (run_jobs_parallel(commands, args.output / 'arm_status.json', slots) if slots else
                run_jobs(commands, args.output / 'arm_status.json'))
    statuses.update(completed_statuses)
    atomic_json_save(statuses, args.output / 'arm_status.json')
    summary = {}
    for method, status in statuses.items():
        result_path = args.output / 'arms' / method / 'result.json'
        if status['status'] == 'completed':
            result = json.loads(result_path.read_text())
            if result['theta_best_hash'] != fork_hash:
                raise ValueError('arm used a different fork')
            summary[method] = {key: result[key] for key in ('report_best_accuracy', 'report_best_loss',
                                'delta_vs_historical_best', 'scientific_escape')}
        else:
            summary[method] = status
    atomic_json_save(summary, args.output / 'summary.json')
    print(json.dumps(summary, indent=2))
    if any(row['status'] == 'failed' for row in statuses.values()):
        raise RuntimeError('Some arms failed; see arm_status.json. Other arms were still attempted.')


if __name__ == '__main__':
    main()
