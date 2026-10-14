"""Pinned PyCIL iCaRL training, with measure-only Stage B3 interception."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from experiments.cifar_pilot import add_arguments, start_memory, pilot_summary
from adapters.icarl_hooks import ICaRLHook
from diagnostics.icarl_measure import B3Measurement
from diagnostics.vna_gate import evaluate_gate, setting_config
from experiments.stage_b5 import working_directory, write_json, write_csv

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'third_party/PyCIL'
UPSTREAM_PIN = 'f3509b8ca3f20660ce4aa13f19d5283de81b4b35'
REFERENCE_SOURCE = 'PyCIL paper, CIFAR-100 B0 Inc10 iCaRL; third_party/PyCIL/resources/cifar100.png'


def upstream_imports():
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    if not (UPSTREAM/'models/icarl.py').exists(): raise FileNotFoundError('pinned PyCIL absent; no automatic fetch')
    pin = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if pin != UPSTREAM_PIN: raise ValueError('PyCIL commit does not match registered pin')
    if str(UPSTREAM) not in sys.path: sys.path.insert(0, str(UPSTREAM))


def recipe():
    args = json.loads((UPSTREAM/'exps/icarl.json').read_text())
    args['device'] = ['0']
    return args


@contextmanager
def runtime(*, smoke=False, fake=False, workers=4, data_dir=None):
    from models import icarl, base
    from utils import data_manager
    constants = {k: getattr(icarl, k) for k in ('init_epoch', 'epochs', 'num_workers')}
    original_data = data_manager._get_idata
    original_base_loader = base.DataLoader
    def base_loader(*args, **kwargs):
        kwargs['num_workers'] = workers
        return original_base_loader(*args, **kwargs)
    try:
        icarl.num_workers = workers
        base.DataLoader = base_loader
        if smoke: icarl.init_epoch = icarl.epochs = 2
        if fake:
            from torchvision import transforms
            class FakeCIFAR:
                use_path = False
                train_trsf = test_trsf = []
                common_trsf = [transforms.ToTensor()]
                class_order = list(range(20))
                def download_data(self):
                    rng = np.random.RandomState(871)
                    self.train_data = rng.randint(256, size=(160, 32, 32, 3), dtype=np.uint8)
                    self.train_targets = np.repeat(np.arange(20), 8)
                    self.test_data = self.train_data.copy()
                    self.test_targets = self.train_targets.copy()
            data_manager._get_idata = lambda name: FakeCIFAR()
        elif data_dir:
            # PyCIL retains its download_data and transforms; supply its ./data root.
            source = Path(data_dir).resolve()/'cifar-100-python'
            if not source.is_dir(): raise FileNotFoundError(source)
            destination = Path('data/cifar-100-python')
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists(): shutil.copytree(source, destination)
        yield
    finally:
        for key, value in constants.items(): setattr(icarl, key, value)
        data_manager._get_idata = original_data
        base.DataLoader = original_base_loader


def reproduction_gate(tasks, *, smoke=False, fake=False):
    # Published iCaRL uses nearest-mean-of-exemplars inference, logged by PyCIL as NME.
    curve = [r['nme_top1'] for r in tasks if 'nme_top1' in r]
    mean = float(np.mean(curve)) if curve else None
    valid = not smoke and not fake and len(curve) == 10
    return {'status': 'PASS' if valid and abs(mean-64.42) <= 1.5 else 'FAIL' if valid else 'INVALID',
            'metric': 'Average Accuracy (NME)', 'average_accuracy': mean,
            'last_task_accuracy': curve[-1] if curve else None,
            'reference_average_accuracy': 64.42, 'reference_last_task_accuracy': 49.5,
            'reference_source': REFERENCE_SOURCE,
            'reference_paper': 'https://arxiv.org/abs/2112.12533',
            'reference_plot': 'https://github.com/LAMDA-CL/PyCIL/blob/'+UPSTREAM_PIN+'/resources/cifar100.png',
            'tolerance_pp': 1.5, 'tasks_run': len(tasks), 'smoke': smoke, 'fake_data': fake}


def run(args):
    upstream_imports()
    import trainer
    from utils.data_manager import DataManager
    params = recipe()
    if args.device == 'cpu': params['device'] = [-1]
    out = Path(args.out or ('outputs/pilot/b3' if args.pilot else 'outputs/stageB/b3')).resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.fake_data:
        params.update(memory_size=80)
        torch.set_num_threads(1)
    if args.pilot: start_memory(args.device)
    metadata = {'setting': 'B3', 'seed': 1993, 'upstream_commit': UPSTREAM_PIN,
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'pilot': args.pilot, 'smoke': args.smoke, 'fake_data': args.fake_data}
    measure = B3Measurement(metadata, mu_rel_default=args.mu_rel_default,
                            ranks=(4,) if args.pilot else (4, 8, 16),
                            null_repeats=2 if args.pilot else 8, cg_iters=200 if args.pilot else 1000)
    def persist(hook):
        write_csv(out/'measurements.csv', hook.rows)
        write_csv(out/'epochs.csv', hook.epochs)
        write_json(out/'tasks.json', hook.tasks)
    hook = ICaRLHook(measure, persist=persist, pilot=args.pilot,
                     pilot_batches=args.pilot_batches, fake=args.fake_data)
    original_manager = trainer.DataManager
    def manager(*a, **kw):
        result = DataManager(*a, **kw)
        write_json(out/'class_order.json', result._class_order)
        if args.smoke or args.pilot: result._increments = result._increments[:2]
        return result
    write_json(out/'protocol.json', {**metadata, 'args': params, 'workers': args.workers,
               'resume': 'unavailable: a crash restarts training; hourly archives retain completed logs',
               'measurement_buffer_m': hook.buffer.m, 'ranks': [4] if args.pilot else [4, 8, 16],
               'null_repeats': measure.null_repeats, 'cg_iters': measure.cg_iters,
               'pilot_batches': args.pilot_batches if args.pilot else None})
    started = time.monotonic()
    try:
        trainer.DataManager = manager
        with working_directory(out), runtime(smoke=args.smoke or args.pilot, fake=args.fake_data,
                workers=args.workers, data_dir=args.data_dir), hook.install():
            trainer.train(params)
    finally:
        trainer.DataManager = original_manager
        persist(hook)
    seconds = time.monotonic()-started
    baseline = reproduction_gate(hook.tasks, smoke=args.smoke, fake=args.fake_data)
    cfg = setting_config('B3', mu_rel_default=args.mu_rel_default)
    cfg['final_train_acc'] = hook.epochs[-1]['train_acc'] if hook.epochs else None
    gate = evaluate_gate(hook.rows, cfg)
    gate.update(reproduction=baseline, measurements_valid=baseline['status'] == 'PASS')
    if baseline['status'] != 'PASS': gate['status'] = 'INVALID'
    if args.pilot:
        baseline['status'] = gate['status'] = 'PILOT'
        gate['measurements_valid'] = False
    write_json(out/'reproduction_gate.json', baseline)
    write_json(out/'gate.json', gate)
    diagnostic_seconds = sum(p['seconds'] for p in hook.point_seconds)
    write_json(out/'summary.json', {**metadata, 'mu_rel_default': args.mu_rel_default, 'mu_source': args.mu_source, 'seconds': seconds,
               'training_seconds': seconds-diagnostic_seconds, 'epochs_run': len(hook.epochs),
               'tasks_run': len(hook.tasks), 'measurement_point_seconds': hook.point_seconds,
               'resume_supported': False, 'reproduction': baseline, 'gate': gate,
               **(pilot_summary(hook.epochs, hook.point_seconds, hook.rows, args.device,
                   full_epochs=1730, full_points=40,
                   full_batches=sum(c['full'] for c in hook.batch_counts)/len(hook.batch_counts),
                   pilot_batches=sum(c['pilot'] for c in hook.batch_counts)/len(hook.batch_counts))
                   if args.pilot else {})})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--fake-data', action='store_true')
    p.add_argument('--out')
    p.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--data-dir')
    p.add_argument('--mu-rel-default', type=float, default=.01)
    p.add_argument('--mu-source', default='cli/default')
    run(p.parse_args())


if __name__ == '__main__': main()
