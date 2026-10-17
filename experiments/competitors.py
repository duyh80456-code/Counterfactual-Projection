"""Unmodified pinned PyCIL and LoP competitors on the registered protocol."""
import argparse
import copy
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import time
import numpy as np
import torch
from experiments import stage_b3 as cil, stage_b5 as lop
from experiments.vna_provenance import code_commit
from experiments.vna_resume import Progress, Status, rng_state, restore_rng, log, pause_if_due

CIL_METHODS = ('finetune', 'wa', 'podnet', 'bic', 'foster', 'der', 'memo', 'tagfex', 'joint')
EXPANDING = ('der', 'memo', 'tagfex')


def cil_recipe(method, setting, seed=1993, device='cuda:0'):
    source = 'finetune' if method == 'joint' else method
    original = json.loads((cil.UPSTREAM/f'exps/{source}.json').read_text())
    params = copy.deepcopy(original)
    init, increment = (100, 0) if method == 'joint' else ((50, 10) if setting == 'b50inc10' else (10, 10))
    backbone = {'podnet': 'cosine_resnet32', 'memo': 'memo_resnet32'}.get(method, 'resnet32')
    params.update(dataset='cifar100_aa' if method == 'tagfex' else 'cifar100', memory_size=2000, seed=[seed], convnet_type=backbone,
                  init_cls=init, increment=increment, device=[-1] if device == 'cpu' else ['0'])
    overrides = {key: {'upstream': original.get(key), 'value': value}
                 for key, value in params.items() if original.get(key) != value}
    return params, f'exps/{source}.json', overrides


@contextmanager
def cil_runtime(method, *, fake=False, data_dir=None, boundary=lambda: None):
    """Wrap I/O and deadlines; keep the method's optimizer and training code."""
    from utils import data_manager
    from models import base
    from torchvision import transforms
    module = importlib.import_module('models.'+('finetune' if method == 'joint' else method))
    saved = []
    def replace(obj, name, value):
        saved.append((obj, name, getattr(obj, name))); setattr(obj, name, value)
    original_loader = torch.utils.data.DataLoader
    class Loader(original_loader):
        def __init__(self, *args, **kwargs):
            if fake: kwargs['num_workers'] = 0
            super().__init__(*args, **kwargs)
        def __iter__(self):
            for batch in super().__iter__():
                boundary()
                yield batch
    try:
        replace(module, 'DataLoader', Loader)
        replace(base, 'DataLoader', Loader)
        if fake:
            class FakeCIFAR:
                use_path = False
                train_trsf = test_trsf = []
                common_trsf = [transforms.ToTensor()]
                class_order = list(range(100))
                def download_data(self):
                    rng = np.random.RandomState(871)
                    self.train_data = rng.randint(256, size=(800,32,32,3), dtype=np.uint8)
                    self.train_targets = np.repeat(np.arange(100), 8)
                    self.test_data, self.test_targets = self.train_data.copy(), self.train_targets.copy()
            replace(data_manager, '_get_idata', lambda name: FakeCIFAR())
            for key in ('init_epoch', 'epochs', 'ft_epochs'):
                if hasattr(module, key): replace(module, key, 1)
        elif data_dir:
            source = Path(data_dir).resolve()/'cifar-100-python'
            if not source.is_dir(): raise FileNotFoundError(source)
            destination = Path('data/cifar-100-python')
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists(): destination.symlink_to(source, target_is_directory=True)
            elif destination.resolve() != source:
                from experiments.parallel_validity import dataset_fingerprint
                if dataset_fingerprint(destination.parent) != dataset_fingerprint(source.parent):
                    raise ValueError('staged CIFAR-100 differs from fingerprinted input')
        yield module
    finally:
        for obj, name, value in reversed(saved): setattr(obj, name, value)


def task_accuracy(tasks, result, task):
    row = {'task': task}
    for label, accuracy in zip(('cnn', 'nme'), result):
        if accuracy is None: continue
        row[label+'_top1'] = float(accuracy['top1'])
        row[label+'_grouped'] = accuracy['grouped']
        row[label+'_average_accuracy'] = float(np.mean([r[label+'_top1'] for r in tasks]+[accuracy['top1']]))
        keys = sorted(k for k in accuracy['grouped'] if '-' in k)
        drops = [max([accuracy['grouped'][k]] + [r[label+'_grouped'].get(k, accuracy['grouped'][k])
                 for r in tasks])-accuracy['grouped'][k] for k in keys[:-1]]
        row[label+'_forgetting'] = float(np.mean(drops)) if drops else 0.
    return row


def run_cil(args):
    if not args.fake_data and not args.data_dir:
        raise ValueError('--data-dir is required to fingerprint real CIFAR-100 inputs')
    cil.upstream_imports()
    import trainer
    from utils.data_manager import DataManager
    from utils import factory
    params, source, overrides = cil_recipe(args.method, args.setting, args.seed, args.device)
    full_params = copy.deepcopy(params)
    if args.fake_data:
        torch.set_num_threads(1)
        params.update(memory_size=40, memory_per_class=4)
        for key in ('init_epoch', 'init_epochs', 'epochs', 'boosting_epochs', 'compression_epochs'):
            if key in params: params[key] = 1
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    identity = dict(benchmark='cil_competitor', arm=args.method, seed=args.seed,
        init_cls=params['init_cls'], increment=params['increment'], args=copy.deepcopy(params),
        cfg_source=source, config_overrides=overrides, fake_data=args.fake_data,
        fixture_overrides={k: {'upstream':full_params.get(k), 'value':v}
                           for k,v in params.items() if full_params.get(k) != v},
        dataset_family='cifar100', native_augmentation_retained=True,
        code_commit=code_commit(), source_digest=os.environ.get('VNA_SOURCE_DIGEST'), upstream_commit=cil.UPSTREAM_PIN,
        dataset_fingerprint=None if args.fake_data else __import__(
            'experiments.parallel_validity', fromlist=['dataset_fingerprint']).dataset_fingerprint(args.data_dir),
        model_group='expanding' if args.method in EXPANDING else 'fixed',
        resume='exact task-boundary state; interrupted task restarts from its saved initial state')
    progress = Progress(out, identity)
    if progress.state and not args.resume: raise ValueError('existing checkpoint requires --resume')
    lop.write_json(out/'protocol.json', identity)
    if args.resume and (out/'summary.json').exists():
        previous = json.loads((out/'summary.json').read_text())
        if previous.get('status') == 'COMPLETED':
            if previous.get('identity') != identity: raise ValueError('completed identity mismatch')
            return previous
    status = Status(out, args.method)
    started = time.monotonic(); base_seconds = 0.; tasks = []
    def boundary():
        def paused():
            if progress.path.exists():
                state = torch.load(progress.path, map_location='cpu', weights_only=False)
                state['elapsed_seconds'] = base_seconds+time.monotonic()-started
                progress.save(state)
            lop.write_json(out/'summary.json', dict(status='PAUSED', identity=identity, resume_supported=True))
        pause_if_due(paused)
    try:
        with lop.working_directory(out), cil_runtime(args.method, fake=args.fake_data,
                data_dir=args.data_dir, boundary=boundary):
            trainer._set_random()
            trainer._set_device(params)
            manager = DataManager(params['dataset'], params['shuffle'], args.seed,
                                  params['init_cls'], params['increment'], params.get('aug', 1))
            lop.write_json(out/'class_order.json', manager._class_order)
            if args.fake_data: manager._increments = manager._increments[:1]
            model = factory.get_model(params['model_name'], params)
            if progress.state:
                model = progress.state['model']; tasks = progress.state['tasks']
                for key, value in vars(model).items():
                    if isinstance(value, torch.nn.Module): value.to(model._device)
                    elif isinstance(value, torch.Tensor): setattr(model, key, value.to(model._device))
                base_seconds = progress.state['elapsed_seconds']
                restore_rng(progress.state['rng'])
            def checkpoint():
                state = copy.deepcopy(model)
                for key in ('train_loader', 'test_loader', 'val_loader'):
                    vars(state).pop(key, None)
                progress.save(dict(model=state, tasks=tasks, rng=rng_state(),
                    elapsed_seconds=base_seconds+time.monotonic()-started))
                lop.write_json(out/'tasks.json', tasks)
            if progress.state is None: checkpoint()
            for task in range(len(tasks), manager.nb_tasks):
                boundary(); status.update(task=task+1, phase='upstream training')
                model.incremental_train(manager)
                result = model.eval_task()
                model.after_task()
                tasks.append(task_accuracy(tasks, result, task+1))
                checkpoint()
                log(f'{args.method} task={task+1} {tasks[-1]}')
            seconds = base_seconds+time.monotonic()-started
            evaluated_parameters = sum(p.numel() for p in model._network.parameters())
            compressed = getattr(model, '_snet', None)
            parameters = sum(p.numel() for p in compressed.parameters()) if compressed is not None else evaluated_parameters
        gate = dict(status='INVALID' if args.fake_data else 'NO_REFERENCE', reference='none',
                    reference_source='no audited numeric reference for this exact protocol',
                    tolerance_pp=1.5, metric='Average Accuracy (CNN / NME as available)')
        summary = dict(identity=identity, arm=args.method, seed=args.seed, status='COMPLETED',
            benchmark='cil_competitor', fake_data=args.fake_data, tasks=tasks, tasks_run=len(tasks),
            init_cls=params['init_cls'], increment=params['increment'], seconds=seconds,
            parameter_count=parameters, gpu_hours=seconds/3600 if args.device != 'cpu' else 0.,
            evaluated_parameter_count=evaluated_parameters,
            parameter_count_source='compressed student' if compressed is not None else 'evaluated network',
            model_group=identity['model_group'], reproduction=gate, resume_supported=True)
        lop.write_json(out/'reproduction_gate.json', gate)
        lop.write_json(out/'summary.json', summary)
        return summary
    finally: status.close()


class ReDoHook:
    """Call upstream GnTredo after supervised SGD, preserving its own counter."""
    def before(self, hook, learner, x, target):
        if not hasattr(learner, 'gnt'):
            from lop.algos.gntRedo import GnTredo
            learner.gnt = GnTredo(learner.net.layers, learner.net.act_type)
    def after(self, hook, learner):
        learner.gnt.gen_and_test(torch.stack(learner.previous_features, dim=1))
    def state_dict(self): return {}
    def load_state_dict(self, state): pass

    @contextmanager
    def install(self):
        from lop.algos.bp import Backprop
        original = Backprop.__init__
        def initialize(learner, *args, **kwargs):
            original(learner, *args, **kwargs)
            self.before(None, learner, None, None)
        Backprop.__init__ = initialize
        try: yield
        finally: Backprop.__init__ = original


def b5_recipe(arm, value=None):
    name = {'l2':'relu_wd', 'snp':'relu_snp', 'redo':'relu'}[arm]
    source = f'lop/slowly_changing_regression/cfg/sgd/bp/{name}.json'
    params = json.loads((lop.UPSTREAM/source).read_text())
    if arm == 'redo': params['step_size'] = .01
    key = {'l2':'weight_decay', 'snp':'perturb_scale'}.get(arm)
    grid = params[key] if key else [None]
    if key:
        if value not in grid: raise ValueError('unregistered competitor hyperparameter')
        params[key] = value
    return params, source, grid


def run_b5_seed(args, seed, value, out):
    from adapters.lop_hooks import LearnHook, learner_state
    lop.upstream_imports()
    params, source, grid = b5_recipe(args.arm, value)
    if args.fake_data: params.update(num_data_points=1000, flip_after=500)
    params.update(run_idx=0, use_gpu=0, data_file=str(out/'upstream_in_progress.tmp'),
                  env_file=str(out/'data/regression.pkl'))
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(benchmark='b5_competitor', arm=args.arm, seed=seed, params=params,
        cfg_source=source, hyperparameter=value, fake_data=args.fake_data,
        code_commit=code_commit(), source_digest=os.environ.get('VNA_SOURCE_DIGEST'), upstream_commit=lop.UPSTREAM_PIN,
        redo={'threshold':.01, 'reset_period':1000} if args.arm == 'redo' else None)
    progress = Progress(out, identity, name='deadline.pt')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != identity:
        raise ValueError('B5 competitor resume identity mismatch')
    lop.write_json(out/'protocol.json', identity)
    if args.resume and (out/'summary.json').exists():
        previous = json.loads((out/'summary.json').read_text())
        if previous.get('status') == 'COMPLETED': return previous
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    lop.prepare_data('regression', params, out, fake=args.fake_data)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    path = out/'resume.pkl'
    resumed = None
    if path.exists():
        if not args.resume: raise ValueError('existing checkpoint requires --resume')
        with path.open('rb') as stream: resumed = pickle.load(stream)
    def checkpoint(state):
        lop.atomic_pickle(path, state)
        return path.stat().st_size
    status = Status(out, f'b5_{args.arm} seed={seed}')
    intervention = ReDoHook() if args.arm == 'redo' else None
    hook = LearnHook('regression', 'bp', params, seed=seed, resume=resumed,
        checkpoint=checkpoint, progress=progress, status=status, intervention=intervention)
    hook.task_log_only = True
    started = time.monotonic()
    try:
        from contextlib import nullcontext
        with intervention.install() if intervention else nullcontext():
            lop.run_native('regression', params, out, hook)
        hook._checkpoint(hook.last_learner, hook.last_frame)
        tasks = list(hook.results.tasks())
        seconds = hook.elapsed_base+time.monotonic()-started
        summary = dict(identity=identity, benchmark='b5_competitor', arm=args.arm, seed=seed,
            hyperparameter=value, status='COMPLETED', tasks=tasks, steps=hook.step,
            mean_last50=float(np.mean([r['online_error'] for r in tasks][-50:])),
            seconds=seconds, parameter_count=sum(p.numel() for p in hook.last_learner.net.parameters()),
            gpu_hours=0., resume_supported=True)
        lop.atomic_pickle(out/'final_state.pkl', dict(learner=learner_state(hook.last_learner), rng=rng_state()))
        lop.write_json(out/'tasks.json', tasks); lop.write_json(out/'summary.json', summary)
        return summary
    finally: status.close()


def run_b5(args):
    lop.upstream_imports()
    if args.fake_data: torch.set_num_threads(1)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    raw = json.loads((lop.UPSTREAM/('lop/slowly_changing_regression/cfg/sgd/bp/'+
                    {'l2':'relu_wd', 'snp':'relu_snp', 'redo':'relu'}[args.arm]+'.json')).read_text())
    key = {'l2':'weight_decay', 'snp':'perturb_scale'}.get(args.arm)
    grid = raw[key] if key else [None]
    candidates = [run_b5_seed(args, 0, value, out/'scan'/str(value)/'seed0') for value in grid]
    chosen = min(candidates, key=lambda r:r['mean_last50'])
    value = chosen['hyperparameter']
    protocol = dict(benchmark='b5_competitor', arm=args.arm, code_commit=code_commit(), source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
        upstream_commit=lop.UPSTREAM_PIN, hyperparameter_name=key, grid=grid, selected=value,
        selection='minimum seed0 mean online error over final 50 tasks; ties use cfg order',
        seeds=[0,1,2], fake_data=args.fake_data)
    lop.write_json(out/'protocol.json', protocol)
    variants = [chosen]+[run_b5_seed(args, seed, value, out/f'seed{seed}') for seed in (1,2)]
    summary = dict(benchmark='b5_competitor', arm=args.arm, status='COMPLETED', identity=protocol,
        variants=variants, selected=value, mean_last50=float(np.mean([r['mean_last50'] for r in variants])),
        std_last50=float(np.std([r['mean_last50'] for r in variants], ddof=1)),
        seconds=sum(r['seconds'] for r in candidates)+sum(r['seconds'] for r in variants[1:]))
    lop.write_json(out/'summary.json', summary)
    lop.write_json(out/'tasks.json', {str(r['seed']):r['tasks'] for r in variants})
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='benchmark', required=True)
    c = sub.add_parser('cil'); c.add_argument('--method', choices=CIL_METHODS, required=True)
    c.add_argument('--setting', choices=('inc10', 'b50inc10'), default='inc10')
    c.add_argument('--seed', type=int, default=1993)
    c.add_argument('--device', choices=('cpu','cuda:0'), default='cuda:0')
    c.add_argument('--data-dir')
    b = sub.add_parser('b5'); b.add_argument('--arm', choices=('l2','snp','redo'), required=True)
    for child in (c,b):
        child.add_argument('--out', required=True)
        child.add_argument('--resume', action='store_true')
        child.add_argument('--fake-data', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    (run_cil if args.benchmark == 'cil' else run_b5)(args)
