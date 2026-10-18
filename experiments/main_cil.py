"""Main B0 Inc5 iCaRL runs using pinned PyCIL training and exact resume hooks."""
import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from adapters.cil_setup import class_order, resnet32_nobn
from adapters.icarl_hooks import ICaRLHook, Prediction, sites_for, loss_context
from diagnostics.vna_align import _global_state
from methods import vna_v3
from experiments import stage_b3 as cil
from experiments.competitors import cil_recipe, cil_runtime
from experiments.stage_b5 import working_directory, write_json, write_csv
from experiments.vna_provenance import code_commit
from experiments.vna_resume import Progress, Status

DATA_ROOT = '/kaggle/input/datasets/nlnk1607/cifar100'
ARMS = ('icarl', 'icarl_full')
RESULT_FILES = ('summary.json', 'tasks.json', 'continual_eval.csv', 'diagnostics.json',
                'protocol.json', 'status.json')


def recipe(seed=0, device='cuda:0', backbone='resnet32_nobn'):
    params, _, _ = cil_recipe('icarl', 'b0inc5', seed, device)
    params.update(init_cls=5, increment=5, fixed_memory=False, convnet_type=backbone)
    return params


@contextmanager
def runtime(args, order):
    from models import icarl
    from utils import data_manager, inc_net
    saved = []
    def replace(obj, name, value):
        saved.append((obj, name, getattr(obj, name))); setattr(obj, name, value)
    original_data = data_manager._get_idata
    original_convnet = inc_net.get_convnet
    def data(name):
        value = original_data(name)
        value.class_order = order
        return value
    def convnet(params, pretrained=False):
        return resnet32_nobn() if params['convnet_type'] == 'resnet32_nobn' else original_convnet(params, pretrained)
    try:
        replace(data_manager, '_get_idata', data)
        replace(inc_net, 'get_convnet', convnet)
        for key, value in dict(init_epoch=1 if args.fake_data else 100,
                epochs=1 if args.fake_data else 50, init_milestones=[30,60,85],
                milestones=[round(80*50/170), round(120*50/170)], num_workers=args.workers,
                init_lr=.1*args.nobn_lr_scale if args.backbone.endswith('nobn') else .1,
                lrate=.1*args.nobn_lr_scale if args.backbone.endswith('nobn') else .1).items():
            replace(icarl, key, value)
        yield
    finally:
        for obj, name, value in reversed(saved): setattr(obj, name, value)


class Intervention:
    """Geometric task schedule; interact with the engine only through its public API."""
    def __init__(self, engine, K0):
        self.engine, self.K0 = engine, K0
        self.task, self.last_iteration, self.interval = None, None, K0
        self.iteration_times, self.events = [], []

    def state_dict(self):
        return {**{k: copy.deepcopy(v) for k, v in vars(self).items() if k != 'engine'},
                'engine': self.engine.state_dict()}

    def load_state_dict(self, state):
        self.engine.load_state_dict(state['engine'])
        for k, value in state.items():
            if k != 'engine': setattr(self, k, copy.deepcopy(value))

    def start_task(self, learner):
        self.task = learner._cur_task+1
        self.last_iteration, self.interval = None, self.K0
        state = self.engine.state_dict()
        state.update(task=self.task, last_iteration=None, K_next=1, applications=0,
                     last_refresh=-10, warm={}, A={})
        self.engine.load_state_dict(state)

    def after_step(self, hook, learner, optimizer):
        if self.last_iteration is not None and hook.steps-self.last_iteration < self.interval: return
        batch = tuple(t.to(learner._device) for t in hook.current_batch)
        spec, ctx = loss_context(learner, batch)
        ctx.update(classes=learner._total_classes, task=self.task)
        model = Prediction(learner._network)
        event = self.engine.apply(model, sites_for(model), batch, spec, ctx, optimizer,
                                  hook.steps, force=True)
        if self.last_iteration is not None: self.interval *= 2
        self.last_iteration = hook.steps
        if event is not None: self.events.append(event)
        if not all(bool(torch.isfinite(p).all()) for p in learner._network.parameters()):
            self.events.append(dict(kind='nonfinite_params', nonfinite_params=True,
                                    task=self.task, step=hook.steps))
            raise Diverged('nonfinite parameters after intervention')


class Diverged(RuntimeError): pass


def accuracy_metrics(tasks, predictions, targets, known, label):
    correct = predictions[:, 0] == targets
    current = {str(c): float(100*correct[targets == c].mean()) for c in np.unique(targets)}
    old = targets < known
    total = float(100*correct.mean())
    forgetting = []
    for c, value in current.items():
        past = [t[label+'_per_class'][c] for t in tasks if c in t[label+'_per_class']]
        forgetting.append(max(past)-value if past else 0.)
    return {label+'_top1': total, label+'_old_top1': float(100*correct[old].mean()) if old.any() else None,
            label+'_new_top1': float(100*correct[~old].mean()), label+'_learning_accuracy': float(100*correct[~old].mean()),
            label+'_per_class': current, label+'_forgetting': float(np.mean(forgetting)),
            label+'_average_accuracy': float(np.mean([t[label+'_top1'] for t in tasks]+[total])),
            label+'_last_accuracy': total}


def feature_accuracy(learner, manager, boundary=lambda: None):
    """Multinomial linear probe, all seen training features, deterministic zero start."""
    from torch.utils.data import DataLoader
    def features(source):
        dataset = manager.get_dataset(np.arange(learner._total_classes), source=source, mode='test')
        xs, ys = [], []
        for _, x, y in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0):
            boundary()
            with torch.no_grad(): xs.append(learner._network.extract_vector(x.to(learner._device)).detach())
            ys.append(y.to(learner._device))
        return torch.cat(xs).double(), torch.cat(ys)
    with _global_state():
        modes = {m: m.training for m in learner._network.modules()}
        try:
            learner._network.eval()
            x, y = features('train'); xt, yt = features('test')
            weight = torch.zeros(x.shape[1], learner._total_classes, device=x.device, dtype=x.dtype, requires_grad=True)
            bias = torch.zeros(learner._total_classes, device=x.device, dtype=x.dtype, requires_grad=True)
            optimizer = torch.optim.LBFGS([weight, bias], max_iter=100, line_search_fn='strong_wolfe')
            def closure():
                boundary(); optimizer.zero_grad()
                loss = nn.functional.cross_entropy(x@weight+bias, y)
                loss.backward(); return loss
            optimizer.step(closure)
            with torch.no_grad(): return float(100*((xt@weight+bias).argmax(1) == yt).double().mean())
        finally:
            for module, mode in modes.items(): module.training = mode


class MainHook(ICaRLHook):
    def __init__(self, *a, manager, status, **kw):
        super().__init__(*a, **kw)
        self.manager, self.status = manager, status

    def after_epoch(self, learner, optimizer, frame):
        super().after_epoch(learner, optimizer, frame)
        self.status.update(task=learner._cur_task+1, epoch=self.current_epoch, step=self.steps, phase='training')
        if not np.isfinite(self.epochs[-1]['train_loss']): raise Diverged('nonfinite training loss')

    def measurement_boundary(self):
        # Reuse the last training checkpoint; a partially evaluated task is replayed.
        if self.progress: self.progress.boundary(lambda: None)

    def record_task(self, learner, result):
        row = dict(task=learner._cur_task+1, parameter_count=sum(p.numel() for p in learner._network.parameters()))
        with _global_state():
            self.measurement_boundary()
            for label, predictions in (('cnn', learner._eval_cnn(learner.test_loader)),
                    ('nme', learner._eval_nme(learner.test_loader, learner._class_means))):
                row.update(accuracy_metrics(self.tasks, *predictions, learner._known_classes, label))
            dataset = learner.test_loader.dataset
            indices = np.random.RandomState(0).permutation(len(dataset))[:256]
            samples = [dataset[int(i)] for i in indices]
            batch = (torch.stack([s[1] for s in samples]).to(learner._device),
                     torch.tensor([int(s[2]) for s in samples], device=learner._device))
            model = Prediction(learner._network)
            self.rows.append(dict(task=row['task'], probe_seed=0, probe_indices=indices.tolist(),
                                  **vna_v3.diagnose(model, batch)))
            if row['task'] in (5,10,15,20):
                row['feature_accuracy'] = feature_accuracy(learner, self.manager, self.measurement_boundary)
        self.tasks.append(row)
        if self.persist: self.persist(self)


def run(args):
    cil.upstream_imports()
    import trainer
    from utils.data_manager import DataManager
    from methods.vna_v3 import V3Engine
    if not callable(getattr(vna_v3, 'diagnose', None)):
        raise RuntimeError('Merge Phase 2 first: methods.vna_v3.diagnose(model, batch) is required')
    if args.backbone == 'resnet32' and args.nobn_lr_scale != 1.:
        raise ValueError('nobn_lr_scale applies only to noBN')
    order = class_order(args.seed, args.class_order, None if args.fake_data else args.data_dir)
    schedule = json.loads(Path(args.schedule).read_text()) if args.schedule else None
    if args.arm == 'icarl_full' and args.coefficient != 0 and not args.fake_data and not schedule:
        raise ValueError('icarl_full requires the backbone authority --schedule (K0 preflight)')
    K0 = int(schedule['K0']) if schedule else 1
    if K0 < 1: raise ValueError('K0 must be positive')
    params = recipe(args.seed, args.device, args.backbone)
    params['shuffle'] = False
    if args.fake_data:
        torch.set_num_threads(1)
        params.update(memory_size=20, memory_per_class=2)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    from experiments.parallel_validity import dataset_fingerprint
    protocol = dict(setting='icarl', algorithm_version='v4', arm=args.arm, seed=args.seed,
        code_commit=code_commit(), source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
        upstream_commit=cil.UPSTREAM_PIN, fake_data=args.fake_data, expected_tasks=2 if args.fake_data else 20,
        dataset_fingerprint=None if args.fake_data else dataset_fingerprint(args.data_dir),
        data_dir=args.data_dir, class_order=order, class_order_kind=args.class_order,
        backbone=args.backbone, nobn_lr_scale=args.nobn_lr_scale, coefficient=args.coefficient,
        r=16, live_candidates=4, compute_budget=1., schedule=schedule, K0=K0, cfg=copy.deepcopy(params),
        probe=dict(size=256, seed=0, source='seen test classes'),
        feature_probe=dict(tasks=[5,10,15,20], optimizer='LBFGS', max_iter=100,
                           initialization='zeros', data='all seen train, test transforms', regularization=0),
        training=dict(first_epochs=1 if args.fake_data else 100, later_epochs=1 if args.fake_data else 50,
                      first_milestones=[30,60,85], later_milestones=[24,35]),
        resume='native exact epoch checkpoint; task prefix replay',
        model_config=dict(backbone=args.backbone, class_order=args.class_order),
        comparison_protocol=dict(init_cls=5, increment=5, memory_size=params['memory_size'], fixed_memory=False,
            backbone=args.backbone, class_order=args.class_order, nobn_lr_scale=args.nobn_lr_scale,
            first_epochs=1 if args.fake_data else 100, later_epochs=1 if args.fake_data else 50))
    progress = Progress(out, protocol)
    if (out/'summary.json').exists():
        previous = json.loads((out/'summary.json').read_text())
        if previous.get('identity') != protocol: raise ValueError('completed identity mismatch')
        if previous['status'] == 'COMPLETED': return previous
    status = Status(out, args.arm)
    intervention = Intervention(V3Engine('vna_gir', seed=args.seed, coefficient=args.coefficient, K_base=K0), K0) if args.arm == 'icarl_full' else None
    def persist(hook):
        write_json(out/'tasks.json', hook.tasks)
        write_csv(out/'continual_eval.csv', [{k:v for k,v in row.items() if not k.endswith('_per_class')} for row in hook.tasks])
        if not (out/'continual_eval.csv').exists(): (out/'continual_eval.csv').write_text('task,cnn_top1,nme_top1\n')
        write_json(out/'diagnostics.json', hook.rows)
        if intervention is not None: write_json(out/'interventions.json', intervention.events)
        write_json(out/'protocol.json', {**protocol, 'parameter_count': hook.tasks[-1]['parameter_count'] if hook.tasks else None,
                   'mu0': intervention.engine.state_dict().get('mu0') if intervention else None})
    original_manager, original_random = trainer.DataManager, trainer._set_random
    def initialize():
        import random
        original_random()
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    def manager(*a, **kw):
        value = DataManager(*a, **kw)
        if args.fake_data: value._increments = value._increments[:2]
        hook.manager = value
        return value
    hook = MainHook(seed=args.seed, fake=args.fake_data, persist=persist, progress=progress,
                    intervention=intervention, manager=None, status=status)
    started = time.monotonic()
    run_status, reason = 'COMPLETED', None
    try:
        persist(hook)
        trainer.DataManager, trainer._set_random = manager, initialize
        # cil_runtime supplies upstream I/O; runtime applies only registered setup overrides.
        with working_directory(out), cil_runtime('icarl', fake=args.fake_data, data_dir=args.data_dir), runtime(args, order), hook.install():
            trainer.train(copy.deepcopy(params))
    except Diverged as exc:
        run_status, reason = 'DIVERGED', str(exc)
    except SystemExit as exc:
        if exc.code == 75:
            status.update(status='PAUSED', phase='paused')
        raise
    finally:
        trainer.DataManager, trainer._set_random = original_manager, original_random
        persist(hook)
        status.close()
    seconds = hook.elapsed_base+time.monotonic()-started
    summary = dict(identity=protocol, setting='icarl', algorithm_version='v4', arm=args.arm, seed=args.seed,
        init_cls=5, increment=5,
        status=run_status, reason=reason, tasks=hook.tasks, tasks_run=len(hook.tasks),
        parameter_count=sum(p.numel() for p in hook.learner._network.parameters()),
        seconds=seconds, gpu_hours=seconds/3600 if args.device != 'cpu' else 0.,
        resume_supported=True, nobn_lr_halving_required=run_status=='DIVERGED' and
            args.backbone.endswith('nobn') and hook.learner._cur_task==0 and
            bool(hook.epochs) and bool(np.isnan(hook.epochs[-1]['train_loss'])))
    for label in ('cnn','nme'):
        if hook.tasks:
            summary.update({label+'_average_accuracy': hook.tasks[-1][label+'_average_accuracy'],
                            label+'_last_accuracy': hook.tasks[-1][label+'_top1'],
                            label+'_forgetting': hook.tasks[-1][label+'_forgetting'],
                            label+'_learning_accuracy': float(np.mean([r[label+'_learning_accuracy'] for r in hook.tasks]))})
    write_json(out/'summary.json', summary)
    status.update(status=run_status, phase='finished', reason=reason)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, default='icarl')
    p.add_argument('--backbone', choices=('resnet32', 'resnet32_nobn'), default='resnet32_nobn')
    p.add_argument('--class-order', choices=('superclass','random'), default='superclass')
    p.add_argument('--nobn-lr-scale', type=float, choices=(1.,.5), default=1.)
    p.add_argument('--coefficient', type=float, default=1.)
    p.add_argument('--schedule')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=('cpu','cuda:0'), default='cuda:0')
    p.add_argument('--data-dir', default=DATA_ROOT)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--fake-data', action='store_true')
    p.add_argument('--out', required=True)
    return p


if __name__ == '__main__': run(parser().parse_args())
