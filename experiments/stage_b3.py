"""Pinned PyCIL iCaRL training, with measure-only Stage B3 interception."""
import argparse
from experiments.vna_resume import configure_measurement, measurement_cli
from contextlib import contextmanager
import copy
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
from experiments.vna_resume import Progress
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


def reproduction_gate(tasks, *, smoke=False, fake=False, init_cls=10, increment=10):
    # Published iCaRL uses nearest-mean-of-exemplars inference, logged by PyCIL as NME.
    curve = [r['nme_top1'] for r in tasks if 'nme_top1' in r]
    mean = float(np.mean(curve)) if curve else None
    expected = 1 + int(np.ceil((100-init_cls)/increment)) if init_cls < 100 else 1
    reference = 64.42 if (init_cls, increment) == (10, 10) else None
    valid = not smoke and not fake and len(curve) == expected
    passed = valid and reference is not None and abs(mean-reference) <= 1.5
    return {'status': 'PASS' if passed else 'NO_REFERENCE' if valid and reference is None else 'FAIL' if valid else 'INVALID',
            'metric': 'Average Accuracy (NME)', 'average_accuracy': mean,
            'last_task_accuracy': curve[-1] if curve else None,
            'reference_average_accuracy': reference,
            'reference_last_task_accuracy': 49.5 if reference is not None else None,
            'reference': 'none' if reference is None else reference,
            'reference_source': REFERENCE_SOURCE if reference is not None else 'none published by PyCIL for this protocol',
            'reference_paper': 'https://arxiv.org/abs/2112.12533',
            'reference_plot': 'https://github.com/LAMDA-CL/PyCIL/blob/'+UPSTREAM_PIN+'/resources/cifar100.png',
            'tolerance_pp': 1.5 if reference is not None else None,
            'init_cls': init_cls, 'increment': increment,
            'tasks_run': len(tasks), 'expected_tasks': expected,
            'smoke': smoke, 'fake_data': fake}


def run(args, *, measure_override=None, observe=None, intervention=None):
    if not hasattr(args, 'measure_profile'): args.measure_profile = 'full'
    os.environ.setdefault('VNA_JOB', 'b3')
    upstream_imports()
    import trainer
    from utils.data_manager import DataManager
    params = recipe()
    params.update(seed=[args.seed], init_cls=args.init_cls, increment=args.increment)
    if args.device == 'cpu': params['device'] = [-1]
    out = Path(args.out or ('outputs/pilot/b3' if args.pilot else 'outputs/stageB/b3')).resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.fake_data:
        params.update(memory_size=80)
        torch.set_num_threads(1)
    fingerprint = None
    if not args.fake_data and args.data_dir:
        from experiments.parallel_validity import dataset_fingerprint
        fingerprint = dataset_fingerprint(args.data_dir)
    if args.pilot: start_memory(args.device)
    metadata = {'setting': 'B3', 'seed': args.seed, 'init_cls': args.init_cls,
                'increment': args.increment, 'upstream_commit': UPSTREAM_PIN,
                'dataset_fingerprint': fingerprint,
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'pilot': args.pilot, 'smoke': args.smoke, 'fake_data': args.fake_data,
                'measure_profile': args.measure_profile,
                'arm': getattr(args, 'arm', 'icarl'), 'no_measure': getattr(args, 'no_measure', False)}
    measure = B3Measurement(metadata, mu_rel_default=args.mu_rel_default,
                            ranks=(4,) if args.pilot else (4, 8, 16),
                            null_repeats=2 if args.pilot else 8, cg_iters=200 if args.pilot else 1000,
                            measure_profile=args.measure_profile)
    if measure_override is not None: measure = measure_override
    identity = {**metadata, 'student': params['convnet_type'], 'teacher_hash': None,
                'mu_candidates': [1e-3, 1e-2], 'recipe': params,
                'solver': getattr(args, 'solver', 'auto'), 'output_exact_max': getattr(args, 'output_exact_max', 4096), 'workers': args.workers, 'pilot_batches': args.pilot_batches,
                'intervention': None if intervention is None else {k:v for k,v in intervention.engine.state_dict().items()
                    if k not in ('pools','positions','pool_steps','events','activities','generators')},
                'window': None if intervention is None else intervention.window,
                'calibrate': False if intervention is None else intervention.calibrate,
                'every_epoch': False if intervention is None else intervention.every_epoch,
                'auto_schedule': False if intervention is None else intervention.auto_schedule,
                'init_seed': getattr(args, 'init_seed', 1)}
    identity = copy.deepcopy(identity)
    progress = Progress(out, identity)
    def persist(hook):
        write_csv(out/'measurements.csv', hook.rows)
        write_csv(out/'epochs.csv', hook.epochs)
        write_json(out/'tasks.json', hook.tasks)
        if intervention is not None:
            write_json(out/'interventions.json', intervention.engine.events)
            vna_rows = [event for event in intervention.engine.events
                        if event.get('kind') == 'correction']
            if vna_rows:
                write_csv(out/'vna_log.csv', vna_rows)
            elif not (out/'vna_log.csv').exists():
                (out/'vna_log.csv').write_text('task,step,K,rho,seconds,selection,skipped\n')
            write_json(out/'activities.json', [{**r, 'sites': {site: {k:v for k,v in a.items() if k!='dormant_indices'}
                for site,a in r['sites'].items()}} for r in intervention.engine.activities])
    configure_measurement(measure, args, out)
    hook = ICaRLHook(measure, seed=args.seed, persist=persist, pilot=args.pilot,
                     pilot_batches=args.pilot_batches, fake=args.fake_data, progress=progress, observe=observe, intervention=intervention,
                     measure_profile=args.measure_profile, smoke=args.smoke,
                     save_measurement_checkpoints=getattr(args, 'save_measurement_checkpoints', False))
    summary_path = out/'summary.json'
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        expected_tasks = 2 if args.smoke or args.pilot else 1 + int(np.ceil((100-args.init_cls)/args.increment))
        if previous.get('status') == 'COMPLETED':
            if previous.get('identity') != identity: raise ValueError('completed run identity mismatch')
            if previous.get('tasks_run') == expected_tasks and hook.completed == expected_tasks:
                return hook
    original_manager = trainer.DataManager
    original_random = trainer._set_random
    def initialization():
        original_random()
        torch.manual_seed(args.init_seed)
        torch.cuda.manual_seed_all(args.init_seed)
    def manager(*a, **kw):
        result = DataManager(*a, **kw)
        write_json(out/'class_order.json', result._class_order)
        if args.smoke or args.pilot: result._increments = result._increments[:2]
        return result
    write_json(out/'protocol.json', {**metadata, 'args': params, 'solver': getattr(args, 'solver', 'auto'), 'output_exact_max': getattr(args, 'output_exact_max', 4096), 'workers': args.workers,
               'resume': 'task and epoch prefix replay; full state checkpoints every 25 epochs and after_task',
               'measurement_buffer_m': hook.buffer.m, 'ranks': list(measure.ranks),
               'null_repeats': measure.null_repeats, 'cg_iters': measure.cg_iters,
               'pilot_batches': args.pilot_batches if args.pilot else None})
    started = time.monotonic()
    try:
        trainer.DataManager = manager
        if hasattr(args, 'init_seed'): trainer._set_random = initialization
        with working_directory(out), runtime(smoke=args.smoke or args.pilot, fake=args.fake_data,
                workers=args.workers, data_dir=args.data_dir), hook.install():
            trainer.train(params)
    finally:
        trainer.DataManager = original_manager
        trainer._set_random = original_random
        persist(hook)
    seconds = getattr(hook, 'elapsed_base', 0.) + time.monotonic()-started
    baseline = reproduction_gate(hook.tasks, smoke=args.smoke, fake=args.fake_data,
                                 init_cls=args.init_cls, increment=args.increment)
    by_mu = {}
    for mu in ((1e-2,) if args.measure_profile=='quick' else (1e-2, 1e-3)):
        cfg = setting_config('B3', mu_rel_default=mu)
        if args.measure_profile=='quick': cfg['min_points'] = 0
        cfg['final_train_acc'] = hook.epochs[-1]['train_acc'] if hook.epochs else None
        gate = evaluate_gate(hook.rows, cfg)
        gate.update(reproduction=baseline, measurements_valid=baseline['status'] == 'PASS')
        if baseline['status'] != 'PASS': gate['status'] = 'INVALID'
        if args.pilot:
            baseline['status'] = gate['status'] = 'PILOT'
            gate['measurements_valid'] = False
        by_mu[str(mu)] = gate
        if args.measure_profile=='full': write_json(out/f'gate_mu_{mu}.json', gate)
    official_mu = 1e-2
    gate = {'by_mu': by_mu, 'official_mu': official_mu, 'gate_null_statistic': 'nu',
            'status': by_mu[str(official_mu)]['status'] if official_mu is not None else 'PENDING_STAGE_A'}
    if getattr(args, 'arm', 'icarl') == 'icarl':
        write_json(out/'reproduction_gate.json', baseline)
    if args.measure_profile=='quick':
        gate['status_label'] = 'exploratory'
        write_json(out/'gate_quick.json', gate)
    else: write_json(out/'gate.json', gate)
    from diagnostics.measure_efficiency import exact_cg_summary
    diagnostic_seconds = sum(p['seconds'] for p in hook.point_seconds)
    intervention_seconds = 0. if intervention is None else sum(
        r['seconds'] for r in intervention.engine.events if r['kind'] in ('correction','gir_task'))
    write_json(out/'summary.json', {**metadata, 'identity': identity, 'mu_rel_default': args.mu_rel_default, 'mu_source': args.mu_source, 'seconds': seconds,
               'exact_vs_cg': exact_cg_summary(hook.rows), 'training_seconds': max(0.,seconds-diagnostic_seconds-intervention_seconds),
               'intervention_seconds': intervention_seconds, 'epochs_run': len(hook.epochs),
               'tasks_run': len(hook.tasks), 'measurement_point_seconds': hook.point_seconds,
               'resume_supported': True, 'status': 'COMPLETED', 'reproduction': baseline, 'gate': gate,
               'tasks': hook.tasks,
               'intervention_events': [] if intervention is None else intervention.engine.events,
               'calibration': [] if intervention is None else intervention.calibration,
               'full_batches': [] if intervention is None else intervention.full_batches,
               'iteration_seconds': None if intervention is None or not intervention.iteration_times else
                   float(np.median(intervention.iteration_times)),
               **(pilot_summary(hook.epochs, hook.point_seconds, hook.rows, args.device,
                   full_epochs=1730, full_points=40,
                   full_batches=sum(c['full'] for c in hook.batch_counts)/len(hook.batch_counts),
                   pilot_batches=sum(c['pilot'] for c in hook.batch_counts)/len(hook.batch_counts))
                   if args.pilot else {})})

    return hook

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    measurement_cli(p)
    p.add_argument('--measure-profile', choices=('full','quick'), default='full')
    add_arguments(p)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--fake-data', action='store_true')
    p.add_argument('--out')
    p.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--data-dir')
    p.add_argument('--seed', type=int, default=1993)
    p.add_argument('--init-cls', type=int, choices=(10, 50), default=10)
    p.add_argument('--increment', type=int, choices=(5, 10), default=10)
    p.add_argument('--save-measurement-checkpoints', action='store_true')
    p.add_argument('--mu-rel-default', type=float, default=.01)
    p.add_argument('--mu-source', default='cli/default')
    p.add_argument('--official-mu', type=float, choices=(1e-2, 1e-3))
    return p


def main():
    run(parser().parse_args())


if __name__ == '__main__': main()
