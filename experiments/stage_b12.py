"""B1 CE / B2 KD: RepDistiller main(), unchanged optimizer and training loops."""
import argparse
from contextlib import nullcontext
import importlib
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import torch
from adapters.repdistiller_hooks import upstream, RepDistillerHook, sites_for, loss_context, PAIRS
from diagnostics.icarl_measure import B3Measurement
from diagnostics.stage_b_smoke import FakeMeasurement
from diagnostics.vna_gate import evaluate_gate, setting_config
from experiments.stage_b5 import working_directory, write_json, write_csv
from experiments.vna_provenance import code_commit
from experiments.shared_protocol import seed_everything
from experiments.vna_resume import Progress, file_hash
from scripts.fetch_teachers import TEACHERS, verify, fetch, teacher_gate, UPSTREAM
from experiments.cifar_pilot import add_arguments, start_memory, pilot_summary

REFERENCE = {'b1': {'resnet8x4': 72.50, 'wrn_16_2': 73.26},
             'b2': {'resnet8x4': 73.33, 'wrn_16_2': 74.92}}
SOURCE = 'https://github.com/HobbitLong/RepDistiller#benchmark-results-on-cifar-100'


def measurement(metadata, mu, *, fake=False, pilot=False):
    def context(ctx, batch): return loss_context(ctx['teacher'], batch)
    if fake and not pilot: return FakeMeasurement(metadata, sites_for, context)
    return B3Measurement(metadata, mu_rel_default=mu, sites_fn=sites_for,
                         context_fn=context, ranks=(4,) if pilot else (4, 16, 64), groups=False,
                         null_repeats=2 if pilot else 8, cg_iters=200 if pilot else 1000)


def run(args, *, measure_override=None, observe=None):
    os.environ.setdefault('VNA_JOB', f'{args.setting}_{args.student}')
    out = Path(args.out or f'outputs/{"pilot" if args.pilot else "stageB"}/{args.setting}_{args.student}').resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_dir = str(Path(args.data_dir).resolve())
    if args.fake_data: torch.set_num_threads(1)
    if args.device.startswith('cuda'): torch.cuda.set_device(torch.device(args.device))
    if args.pilot: start_memory(args.device)
    seed_everything(0)
    np.random.seed(0)
    meta = {'schema_version': 'stage1d-v1', 'setting': args.setting.upper(), 'student': args.student,
            'code_commit': code_commit(), 'seed': 0, 'device': args.device,
            'pilot': args.pilot, 'smoke': args.smoke, 'fake_data': args.fake_data}
    def persist(hook):
        write_csv(out/'measurements.csv', hook.rows)
        write_csv(out/'epochs.csv', [{**meta, **r} for r in hook.epochs])
    if args.pilot and args.setting == 'b2' and not args.fake_data:
        name = PAIRS[args.student]
        teacher_path = Path(args.teacher).resolve() if args.teacher else fetch(name)
        report = teacher_gate(name, teacher_path, data_dir, args.device, max_batches=5,
                              workers=args.workers)
        write_json(out/'teacher_gate.json', {**report, 'status': 'PILOT', 'report_only': True})
    with upstream() as models:
        entry = importlib.import_module('train_teacher' if args.setting == 'b1' else 'train_student')
        argv = ['train_teacher.py', '--model', args.student, '--trial', '0']
        if args.setting == 'b2':
            name = PAIRS[args.student]
            teacher = Path(args.teacher).resolve() if args.teacher else UPSTREAM/'save/models'/f'{name}_vanilla/ckpt_epoch_240.pth'
            if not args.fake_data:
                verify(teacher, *TEACHERS[name][:2])
                if entry.get_teacher_name(str(teacher)) != name: raise ValueError('teacher/student pair is fixed')
            else:
                teacher = out/f'{name}_vanilla/ckpt_epoch_240.pth'
                teacher.parent.mkdir(exist_ok=True)
                torch.save({'model': models[name](num_classes=100).state_dict()}, teacher)
            argv = ['train_student.py', '--model_s', args.student, '--path_t', str(teacher),
                    '--distill', 'kd', '-r', '.1', '-a', '.9', '-b', '0', '--trial', '0']
        identity = {**meta, 'mu_candidates': sorted({args.mu_rel_default, .01}),
                    'teacher_hash': file_hash(teacher) if args.setting == 'b2' else None,
                    'upstream_commit': __import__('subprocess').check_output(
                        ['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
                    'workers': args.workers, 'pilot_batches': args.pilot_batches}
        progress = Progress(out, identity)
        argv += ['--num_workers', str(args.workers)]
        if args.smoke or args.pilot: argv += ['--epochs', '2']
        if args.fake_data: argv += ['--batch_size', '4']
        measure = measure_override if measure_override is not None else measurement(meta, args.mu_rel_default, fake=args.fake_data, pilot=args.pilot)
        hook = RepDistillerHook(measure, persist=persist, observe=observe, smoke=args.smoke, fake=args.fake_data,
                               data_dir=data_dir, batch_size=4 if args.fake_data else 64, pilot=args.pilot, pilot_batches=args.pilot_batches, progress=progress)
        # Upstream chooses CUDA by availability; suppress only for an explicit CPU request.
        cuda = patch('torch.cuda.is_available', return_value=False) if args.device == 'cpu' else nullcontext()
        started = time.monotonic()
        old_argv = sys.argv
        original_load = torch.load
        def checkpoint_load(*a, **kw):
            # PyTorch >=2.6 defaults to weights-only; upstream checkpoints include opt.
            kw.setdefault('weights_only', False)
            kw.setdefault('map_location', 'cpu')
            return original_load(*a, **kw)
        try:
            sys.argv = argv
            with cuda, patch('torch.load', new=checkpoint_load), working_directory(out), hook.install(entry):
                entry.main()
                hook.finish_epoch(hook.opt.epochs, hook.last_frame.f_locals)
                if hook.final_test is None:
                    hook.boundary()
                    hook.final_test = entry.validate(hook.test_loader, hook.model, torch.nn.CrossEntropyLoss(), hook.opt)[0]
                    hook.checkpoint()
                acc = hook.final_test
        finally:
            sys.argv = old_argv
            persist(hook)
    seconds = time.monotonic()-started
    reference = REFERENCE[args.setting][args.student]
    reproduction = {**meta, 'final_test_acc': float(acc), 'reference_top1': reference,
                    'difference_pp': float(acc)-reference, 'flag_gt_2pp': abs(float(acc)-reference) > 2,
                    'reference_source': SOURCE, 'reference_only': True,
                    'deviation': 'upstream 50k -> existing setup_data indices (44872 train, 5000 holdout, 128 tuning excluded)'}
    by_mu = {}
    for mu in (1e-2, 1e-3):
        cfg = setting_config(args.setting.upper(), mu_rel_default=mu)
        cfg['final_train_acc'] = hook.epochs[-1]['train_acc']
        gate = {**meta, **evaluate_gate(hook.rows, cfg)}
        if args.smoke or args.fake_data: gate['status'] = 'INVALID'
        if args.pilot: gate['status'] = 'PILOT'
        by_mu[str(mu)] = gate
        write_json(out/f'gate_mu_{mu}.json', gate)
    official_mu = args.official_mu
    gate = {'by_mu': by_mu, 'official_mu': official_mu,
            'status': by_mu[str(official_mu)]['status'] if official_mu is not None else 'PENDING_STAGE_A'}
    write_json(out/'reproduction.json', reproduction)
    write_json(out/'gate.json', gate)
    diagnostic = sum(p['seconds'] for p in hook.point_seconds)
    write_json(out/'summary.json', {**meta, 'identity': identity, 'mu_rel_default': args.mu_rel_default, 'mu_source': args.mu_source, 'seconds': seconds, 'training_seconds': seconds-diagnostic,
               'epochs_run': len(hook.epochs), 'measurement_point_seconds': hook.point_seconds,
               'final_test_acc': float(acc), 'parameter_count': sum(p.numel() for p in hook.model.parameters()),
               'gate': gate, 'resume_supported': True, 'status': 'COMPLETED',
               **(pilot_summary(hook.epochs, hook.point_seconds, hook.rows, args.device,
                   full_epochs=240, full_points=15, full_batches=hook.full_batches,
                   pilot_batches=hook.train_batches) if args.pilot else {})})
    return hook


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('setting', choices=('b1', 'b2'))
    p.add_argument('--student', choices=PAIRS, required=True)
    p.add_argument('--teacher')
    add_arguments(p)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--fake-data', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--mu-rel-default', type=float, default=.01)
    p.add_argument('--mu-source', default='cli/default')
    p.add_argument('--official-mu', type=float, choices=(1e-2, 1e-3))
    return p


if __name__ == '__main__': run(parser().parse_args())
