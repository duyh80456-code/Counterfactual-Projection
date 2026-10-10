"""Run D0 first, then the two new seed2 arms; import old controls for reporting only."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

from experiments.deit_projection_resume import prepare_projection_resume, comparison_table
from experiments.run_deit_all_arms import gpu_slots, run_jobs_parallel, run_jobs
from experiments.shared_protocol import atomic_json_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-root', type=Path, action='append', default=[])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-devices', default='auto')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = prepare_projection_resume(args.resume_root, args.output)
    fork_path = Path(plan['fork_path']); fork_hash = plan['fork_hash']
    commands = {}
    for method in ('all12_transfer_diagnostic', 'e2o_top3_best_gate', 'e2o_top3_recurrent'):
        arm_output = args.output / method
        command = [sys.executable, '-m', 'experiments.run_deit_projection_aware', '--method', method,
            '--plateau-checkpoint', str(fork_path), '--plateau-checkpoint-hash', fork_hash,
            '--data-root', args.data_root, '--output', str(arm_output), '--device', args.device]
        latest = arm_output / 'checkpoint_latest.pt'
        if method != 'all12_transfer_diagnostic' and latest.exists():
            command += ['--resume', str(latest)]
        commands[method] = command
    # D0 is diagnostic only; never enters the epoch/GPU arm scheduler.
    diagnostic_status = run_jobs({'all12_transfer_diagnostic': commands.pop('all12_transfer_diagnostic')},
                                args.output / 'diagnostic_status.json')
    if diagnostic_status['all12_transfer_diagnostic']['status'] != 'completed':
        raise RuntimeError('D0 failed; inspect diagnostic_status.json before running long arms')
    completed = {method: {'status': 'completed', 'resumed_completed': True}
                 for method, row in plan['arms'].items() if row['action'] == 'completed'}
    for method in completed:
        commands.pop(method)
    slots = gpu_slots(args.gpu_devices, args.device) if commands else []
    statuses = (run_jobs_parallel(commands, args.output / 'arm_status.json', slots) if slots else
                run_jobs(commands, args.output / 'arm_status.json'))
    statuses.update(completed)
    atomic_json_save(statuses, args.output / 'arm_status.json')
    comparison_table(args.resume_root, args.output, fork_hash)
    if any(row['status'] == 'failed' for row in statuses.values()):
        raise RuntimeError('A new arm failed; other independent arms were still attempted')


if __name__ == '__main__':
    main()
