"""GroupNorm ResNet-32 pilot under the P1 protocol: baseline iCaRL vs VNA (Gram/power10, no GIR).

Runs main_cil on resnet32_gn (Inc5, superclass, lr 0.05, no warmup, tasks 1-5) and records the
old/new-class test-accuracy curve with the same reader and cadence as the P1 stability-gap replay
(diagnostics/stability_gap.py), plus every VNA intervention.
"""
import argparse
import json
from pathlib import Path

from diagnostics.stability_gap import MAX_TASKS, Tracker, summarize, write


def run(args):
    import numpy as np
    import torch
    from experiments import main_cil as main
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(out, torch, np)
    original_before, original_record = main.MainHook.before_optimizer_step, main.MainHook.record_task
    def before(self, frame):
        tracker.before_step(self)
        return original_before(self, frame)
    def record(self, learner, result):
        tracker.task_end(self, learner)
        return original_record(self, learner, result)
    main.MainHook.before_optimizer_step, main.MainHook.record_task = before, record
    argv = ['--arm', args.arm, '--backbone', 'resnet32_gn', '--class-order', 'superclass', '--seed', str(args.seed),
            '--nobn-lr-scale', '0.5', '--solver', 'gram', '--curvature', 'power10', '--max-tasks', str(MAX_TASKS),
            '--device', args.device, '--data-dir', args.data_dir, '--out', str(out/'run')]
    if args.schedule: argv += ['--schedule', args.schedule]
    if args.fake_data: argv.append('--fake-data')  # CPU smoke check only
    summary = main.run(main.parser().parse_args(argv))
    tasks = json.loads((out/'run'/'tasks.json').read_text())
    events = json.loads((out/'run'/'interventions.json').read_text()) if (out/'run'/'interventions.json').exists() else []
    write(out/'stability_gap'/'report.json', dict(job=f'{args.arm}__gn__s{args.seed}', status='OK',
        run_status=summary.get('status'), backbone='resnet32_gn', groups=8,
        task1_end_cnn_top1=tasks[0]['cnn_top1'] if tasks else None, tasks=summarize(tracker.state, events)))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=('icarl', 'icarl_vna'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--schedule', help='GroupNorm K0 authority (required for icarl_vna)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--fake-data', action='store_true', help='CPU smoke check only')
    return p


if __name__ == '__main__': run(parser().parse_args())
