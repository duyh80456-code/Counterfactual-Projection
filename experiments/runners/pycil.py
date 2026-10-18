"""Pinned PyCIL training without the retired measurement stack."""
import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from adapters.icarl_hooks import ICaRLHook
from experiments.vna_resume import Progress
from experiments.io_utils import working_directory, write_json, write_csv

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT/'third_party/PyCIL'
UPSTREAM_PIN = 'f3509b8ca3f20660ce4aa13f19d5283de81b4b35'


class NoMeasurement:
    ranks = (16,)
    null_repeats = 8
    cg_iters = 1000

    def __init__(self, metadata=None, **kwargs):
        self.metadata = metadata or {}

    def __call__(self, *args, **kwargs):
        return []

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
    metadata = {'setting': 'B3', 'seed': args.seed, 'init_cls': args.init_cls,
                'source_digest': os.environ.get('VNA_SOURCE_DIGEST'),
                'increment': args.increment, 'upstream_commit': UPSTREAM_PIN,
                'dataset_fingerprint': fingerprint,
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'pilot': args.pilot, 'smoke': args.smoke, 'fake_data': args.fake_data,
                'measure_profile': args.measure_profile,
                'arm': getattr(args, 'arm', 'icarl'), 'no_measure': getattr(args, 'no_measure', False)}
    measure = measure_override if measure_override is not None else NoMeasurement(metadata)
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
    diagnostic_seconds = sum(p['seconds'] for p in hook.point_seconds)
    intervention_seconds = 0. if intervention is None else sum(
        r['seconds'] for r in intervention.engine.events if r['kind'] in ('correction','gir_task'))
    write_json(out/'summary.json', {**metadata, 'identity': identity, 'mu_rel_default': args.mu_rel_default, 'mu_source': args.mu_source, 'seconds': seconds,
               'training_seconds': max(0.,seconds-diagnostic_seconds-intervention_seconds),
               'intervention_seconds': intervention_seconds, 'epochs_run': len(hook.epochs),
               'tasks_run': len(hook.tasks), 'measurement_point_seconds': hook.point_seconds,
               'resume_supported': True, 'status': 'COMPLETED', 
               'tasks': hook.tasks,
               'intervention_events': [] if intervention is None else intervention.engine.events,
               'calibration': [] if intervention is None else intervention.calibration,
               'full_batches': [] if intervention is None else intervention.full_batches,
               'iteration_seconds': None if intervention is None or not intervention.iteration_times else
                   float(np.median(intervention.iteration_times))})


    return hook

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out')
    p.add_argument('--device', default='cpu')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--data-dir')
    p.add_argument('--seed', type=int, default=1993)
    p.add_argument('--init-cls', type=int, default=10)
    p.add_argument('--increment', type=int, default=10)
    p.add_argument('--fake-data', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.set_defaults(pilot=False, pilot_batches=20, measure_profile='quick',
                   mu_rel_default=.01, mu_source='legacy identity')
    return p
