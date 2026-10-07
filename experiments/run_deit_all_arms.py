"""Confirm a 150-epoch validation-best stall, reuse Vanilla, then run isolated DeiT arms."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--plateau-checkpoint', type=Path)
    parser.add_argument('--vanilla-reference', type=Path)
    parser.add_argument('--resume-phase1', type=Path)
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
    if not arms or any(arm not in ALL_ARMS for arm in arms) or len(set(arms)) != len(arms):
        parser.error('invalid or duplicate arm list')
    config = replace(res18_cp_config(), rank=args.rank, projection_samples=args.projection_samples,
                     scales=tuple(map(float, args.scales.split(','))))
    recipe = DeitRecipe(seed=args.seed, batch_size=args.batch_size, learning_rate=args.learning_rate,
        schedule_epochs=args.schedule_epochs, stall_start_epoch=0,
        stall_patience=args.stall_patience, reference_epochs=args.post_fork_epochs, max_epoch=args.max_epoch)
    recipe.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    phase1 = args.output / 'vanilla_stall'
    fork_path = args.plateau_checkpoint
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
    commands = {}
    for method in arms:
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
    atomic_json_save({'suite_version': SUITE_VERSION, 'fork_hash': fork_hash, 'recipe': asdict(recipe),
                     'cp_config': asdict(config), 'arms': arms, 'commands': commands}, args.output / 'suite_protocol.json')
    statuses = run_jobs(commands, args.output / 'arm_status.json')
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
