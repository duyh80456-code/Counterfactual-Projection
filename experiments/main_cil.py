"""Main B0 Inc5 iCaRL runs using pinned PyCIL training and exact resume hooks."""
import argparse
import copy
import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from adapters.cil_setup import class_order, resnet32_nobn, resnet32_gn
from adapters.icarl_hooks import ICaRLHook, Prediction, sites_for, loss_context
from diagnostics.vna_align import _global_state
from methods import vna_v3
from experiments.runners import pycil as cil
from experiments.competitors import cil_recipe, cil_runtime
from experiments.runners.lop import working_directory, write_json, write_csv
from experiments.vna_provenance import code_commit
from experiments.vna_resume import Progress, Status, log
from experiments.grad_clip import GradClip, TrainingDiverged, positive_tau, step_details, check_parameters
from experiments.lr_warmup import nonnegative_epochs

DATA_ROOT = '/kaggle/input/datasets/nlnk1607/cifar100'
# Engine arm per CIL arm; ablations (§4 round 3) differ from icarl_full in one component only.
ENGINE_ARMS = {
    'icarl_full': 'vna_gir', 'icarl_gir': 'gir', 'icarl_vna': 'vna',
    'icarl_full_beta0': 'vna_gir_beta0', 'icarl_full_Arand': 'vna_gir_rand',
    'icarl_kernel_self': 'kernel_self_vna_gir', 'icarl_gir_random': 'vna_gir_girinit-random',
    'icarl_gir_gradmax': 'vna_gir_girinit-gradmax', 'icarl_gir_tiny': 'vna_gir_girinit-tiny'}
ARMS = ('icarl',)+tuple(ENGINE_ARMS)
# noBN lr = 0.1 * scale; the registered scale is 0.5 for every noBN arm (PREREGISTRATION_CHANGES).
NOBN_LR_SCALES = (.5,)
RESULT_FILES = ('summary.json', 'tasks.json', 'continual_eval.csv', 'diagnostics.json',
                'protocol.json', 'status.json')


def recipe(seed=0, device='cuda:0', backbone='resnet32_nobn', increment=5):
    params, _, _ = cil_recipe('icarl', 'b0inc5', seed, device)
    params.update(init_cls=increment, increment=increment, fixed_memory=False, convnet_type=backbone)
    return params


def training_settings(fake=False, reproduction=False, nobn_lr_scale=1.):
    first, later = (200,170) if reproduction else (100,50)
    return dict(init_epoch=1 if fake else first, epochs=1 if fake else later,
                init_milestones=[round(v*first/200) for v in (60,120,170)],
                milestones=[round(v*later/170) for v in (80,120)],
                init_lr=.1*nobn_lr_scale, lrate=.1*nobn_lr_scale)


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
        net = (resnet32_nobn() if params['convnet_type'] == 'resnet32_nobn' else resnet32_gn()
               if params['convnet_type'] == 'resnet32_gn' else original_convnet(params, pretrained))
        if hasattr(net, 'fc'): del net.fc
        return net
    try:
        replace(data_manager, '_get_idata', data)
        replace(inc_net, 'get_convnet', convnet)
        for key, value in dict(training_settings(args.fake_data or getattr(args, 'preflight_only', False), nobn_lr_scale=
                args.nobn_lr_scale if args.backbone in ('resnet32_nobn', 'resnet32_gn') else 1.),
                num_workers=args.workers).items():
            replace(icarl, key, value)
        yield
    finally:
        for obj, name, value in reversed(saved): setattr(obj, name, value)


class Intervention:
    """Geometric task schedule; interact with the engine only through its public API."""
    def __init__(self, engine, K0, preflight=False):
        self.engine, self.K0 = engine, K0
        self.preflight = preflight
        self.counts = {}
        self.task, self.last_iteration, self.interval = None, None, K0
        self.iteration_times, self.events, self.followups = [], [], []

    def state_dict(self):
        return {**{k: copy.deepcopy(v) for k, v in vars(self).items() if k != 'engine'},
                'engine': self.engine.state_dict()}

    def load_state_dict(self, state):
        self.engine.load_state_dict(state['engine'])
        for k, value in state.items():
            if k != 'engine': setattr(self, k, copy.deepcopy(value))

    def start_task(self, learner):
        for row in self.followups:
            if row['status']=='RUNNING':row['status']='TASK_ENDED'
        self.task = learner._cur_task+1
        self.last_iteration, self.interval = None, self.K0
        self.engine.begin_task(self.task)

    def observe_loss(self,hook,loss):
        for row in self.followups:
            if row['status']!='RUNNING' or self.task!=row['task'] or hook.steps<=row['iteration']:continue
            if row['samples'] and hook.steps<=row['samples'][-1]['step']:continue
            value=float(loss.detach())
            row['samples'].append(dict(step=hook.steps,relative_step=hook.steps-row['iteration'],
                loss=value if np.isfinite(value) else None,finite=bool(np.isfinite(value))))
            if len(row['samples'])==20:row['status']='COMPLETED'

    def after_step(self, hook, learner, optimizer):
        if self.engine.coefficient == 0 or self.task<2: return
        if self.preflight and self.counts.get(self.task, 0) >= 3: return
        if not self.preflight and self.last_iteration is not None and hook.steps-self.last_iteration < self.interval: return
        if self.preflight and learner._device.type == 'cuda': torch.cuda.synchronize(learner._device)
        iteration_seconds = max(time.monotonic()-hook.iteration_started, 1e-9) if self.preflight else None
        batch = tuple(t.to(learner._device) for t in hook.current_batch)
        spec, ctx = loss_context(learner, batch)
        ctx.update(classes=learner._total_classes, task=self.task)
        model = Prediction(learner._network)
        probe=fixed_probe(learner,256)[0] if hasattr(learner,'test_loader') else None
        if self.preflight:
            epochs = training_settings()['init_epoch' if self.task == 1 else 'epochs']
            self.engine.protocol.update(baseline_iteration_seconds=iteration_seconds,
                iterations_per_epoch=len(learner.train_loader),
                task_iterations=len(learner.train_loader)*epochs)
            self.engine.preflight(model, sites_for(model), batch, spec, ctx, optimizer, hook.steps,probe_batch=probe)
            self.counts[self.task] = self.counts.get(self.task, 0)+1
            if self.task == 2 and self.counts[self.task] == 3: raise PreflightComplete()
            return
        event = self.engine.apply(model, sites_for(model), batch, spec, ctx, optimizer,
                                  hook.steps, force=True,probe_batch=probe)
        if self.last_iteration is not None: self.interval *= 2
        # Compute-budget guard (§2.5): over budget -> the next gap doubles once more.
        if event is not None and event.get('budget_guard'): self.interval *= 2
        self.last_iteration = hook.steps
        if event is not None:
            self.events.append(event)
            if not event.get('skipped') and (event.get('s',0)>0 or event.get('swaps',0)>0):
                self.followups.append(dict(task=self.task,iteration=hook.steps,status='RUNNING',samples=[]))



class Diverged(RuntimeError): pass
class PreflightComplete(Exception): pass
class BaselineReused(Exception): pass


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


@contextmanager
def read_only(model):
    modes = {m: m.training for m in model.modules()}
    buffers = {n: b.clone() for n,b in model.named_buffers()}
    with _global_state():
        try:
            model.eval()
            yield
        finally:
            with torch.no_grad():
                for n,b in model.named_buffers(): b.copy_(buffers[n])
            for m,mode in modes.items(): m.training = mode


def fixed_probe(learner,size):
    with _global_state():
        dataset=learner.test_loader.dataset
        indices=np.random.RandomState(0).permutation(len(dataset))[:size]
        samples=[dataset[int(i)] for i in indices]
        batch=(torch.stack([s[1] for s in samples]).to(learner._device),
               torch.tensor([int(s[2]) for s in samples],device=learner._device))
    return batch,indices


def measure_task(learner, manager, history, boundary=lambda: None, probe_size=256, mu=None):
    row = dict(task=learner._cur_task+1, parameter_count=sum(p.numel() for p in learner._network.parameters()))
    with read_only(learner._network):
        boundary()
        for label, predictions in (('cnn', learner._eval_cnn(learner.test_loader)),
                ('nme', learner._eval_nme(learner.test_loader, learner._class_means))):
            row.update(accuracy_metrics(history, *predictions, learner._known_classes, label))
        batch,indices=fixed_probe(learner,probe_size)
        model = Prediction(learner._network)
        diagnostic = dict(task=row['task'], probe_seed=0, probe_indices=indices.tolist(),
                              **vna_v3.diagnose(model, batch, sites=sites_for(model), mu=mu,solver='cg_ir',b_probe_size=min(128,len(batch[0]))))
        if row['task'] in (5,10,15,20):
            row['feature_accuracy'] = feature_accuracy(learner, manager, boundary)
    fmt = lambda v: 'n/a' if v is None else f'{v:.2f}'
    log(f"TASK {row['task']}: CNN top1 {fmt(row.get('cnn_top1'))} (old {fmt(row.get('cnn_old_top1'))}, "
        f"new {fmt(row.get('cnn_new_top1'))}) avg {fmt(row.get('cnn_average_accuracy'))} "
        f"forget {fmt(row.get('cnn_forgetting'))} learn {fmt(row.get('cnn_learning_accuracy'))} | "
        f"NME top1 {fmt(row.get('nme_top1'))} avg {fmt(row.get('nme_average_accuracy'))} | "
        f"B {fmt(diagnostic.get('B'))} mu_diag {diagnostic.get('mu_diag')} feature_acc {fmt(row.get('feature_accuracy'))}")
    return row, diagnostic


class MainHook(ICaRLHook):
    def __init__(self, *a, manager, status, **kw):
        self.fake = kw.get('fake', False)
        super().__init__(*a, **kw)
        self.manager, self.status = manager, status
        self.diverged = None
        self.clipper=None
        self.step_observer=self.epoch_observer=None

    def checkpoint(self,**kwargs):
        if self.clipper is not None:self.grad_clip_state=self.clipper.state_dict()
        return super().checkpoint(**kwargs)

    def before_training_epochs(self):
        self.checkpoint()

    def before_optimizer_step(self, frame):
        loss = frame.f_locals.get('loss')
        intervention=getattr(self,'intervention',None)
        if isinstance(loss,torch.Tensor) and intervention is not None and not intervention.preflight:
            intervention.observe_loss(self,loss)
        if self.clipper is not None:
            epoch=int(frame.f_locals.get('epoch',self.current_epoch))+1
            self.pending_clip=self.clipper.apply((p.grad for p in self.learner._network.parameters()),
                task=self.learner._cur_task+1,step=self.steps,epoch=epoch)
            if self.step_observer:
                self.pending_clip.update(step_details(self.learner,self.optimizer,frame))
                self.clipper.steps.append(self.pending_clip)
        if isinstance(loss,torch.Tensor) and not bool(torch.isfinite(loss).all()):
            self.diverged=dict(task=self.learner._cur_task+1,epoch=int(frame.f_locals.get('epoch',self.current_epoch))+1,
                               step=self.steps,tensor='loss')
            raise Diverged('nonfinite training loss')
        if self.clipper is not None and self.clipper.tau is not None and self.pending_clip['nonfinite_norm']:
            raise TrainingDiverged(dict(task=self.learner._cur_task+1,epoch=self.pending_clip['epoch'],step=self.steps,tensor='gradient'))

    def after_optimizer_step(self,frame):
        if self.clipper is None:return
        row=self.pending_clip
        # None is a measuring control: retain the native failure boundary.
        if self.clipper.tau is not None:check_parameters(self.learner,task=row['task'],epoch=row['epoch'],step=self.steps)
        if self.step_observer:self.step_observer(row,self.learner)

    def after_epoch(self, learner, optimizer, frame):
        if not np.isfinite(frame.f_locals['losses']):
            self.diverged = dict(task=learner._cur_task+1, epoch=int(frame.f_locals['epoch'])+1,
                                 step=self.steps, tensor='loss')
            raise Diverged('nonfinite training loss')
        super().after_epoch(learner, optimizer, frame)
        if self.epoch_observer:self.epoch_observer(self.epochs[-1])
        self.status.update(task=learner._cur_task+1, epoch=self.current_epoch, step=self.steps, phase='training')
        self.checkpoint()

    def measurement_boundary(self):
        # Reuse the last training checkpoint; a partially evaluated task is replayed.
        if self.progress: self.progress.boundary(lambda: None)

    def record_task(self, learner, result):
        if self.intervention is not None and self.intervention.preflight: return
        engine = self.intervention.engine if self.intervention is not None else None
        mu = engine.mu_star if engine is not None and engine.mu_star_task == learner._cur_task+1 else None
        row, diagnostic = measure_task(learner, self.manager, self.tasks, self.measurement_boundary,
                                       4 if self.fake else 256, mu=mu)
        self.rows.append(diagnostic)
        self.tasks.append(row)
        if self.persist: self.persist(self)
        if row['task']==2 and getattr(self,'baseline_reference',None) is not None:
            from experiments.main_cil_baseline import compare
            result=compare(self.baseline_reference,self.epochs)
            write_json(self.progress.out/'baseline_loss_check.json',result)
            log(f"BASELINE epoch-loss check: {result['status']} ({result['epochs_compared']} epochs, exact equality)")
            if result['status']=='PASS':raise BaselineReused()


def run(args):
    cil.upstream_imports()
    import trainer
    from utils.data_manager import DataManager
    from methods.vna_v3 import V3Engine
    if not callable(getattr(vna_v3, 'diagnose', None)):
        raise RuntimeError('Merge Phase 2 first: methods.vna_v3.diagnose(model, batch) is required')
    if args.grad_clip is not None and args.backbone!='resnet32_nobn':raise ValueError('grad-clip requires noBN')
    if args.backbone == 'resnet32':args.nobn_lr_scale=1.
    elif args.nobn_lr_scale!=.5:raise ValueError('noBN requires nobn_lr_scale=0.5')
    if not 1 <= args.max_tasks <= 20: raise ValueError('max_tasks must be between 1 and 20')
    increment=getattr(args,'increment',5)
    warmup_epochs=getattr(args,'warmup_epochs',0)
    gir_rule=getattr(args,'gir_rule','schur_dead')
    if gir_rule=='certswap' and args.backbone!='resnet32_nobn':
        raise ValueError('CertSwap suffix cache requires the noBN SkipInit backbone')
    policy_version='certswap_t32_v1' if gir_rule=='certswap' else vna_v3.POLICY_VERSION
    setup={}
    if increment!=5:setup.update(init_cls=increment,increment=increment)
    if warmup_epochs:setup['warmup_epochs']=warmup_epochs
    if gir_rule!='schur_dead':
        setup.update(gir_rule=gir_rule,certswap_topk=args.certswap_topk,
            certswap_lm_budget=args.certswap_lm_budget,certswap_log_schur=args.certswap_log_schur,
            certswap_candidate_rule='tiny_V_unchanged_no_median_no_sign')
    tasks = min(args.max_tasks, 2 if args.fake_data else 100//increment)
    order = class_order(args.seed, args.class_order, None if args.fake_data else args.data_dir)
    schedule = json.loads(Path(args.schedule).read_text()) if args.schedule else None
    if args.arm in ENGINE_ARMS and args.coefficient != 0 and not args.fake_data and not args.preflight_only and not schedule:
        raise ValueError(f'{args.arm} requires the backbone authority --schedule (K0 preflight)')
    if args.preflight_only and not args.schedule_output: raise ValueError('preflight requires --schedule-output')
    if args.preflight_only and args.arm != 'icarl_full': raise ValueError('preflight requires icarl_full')
    if schedule:
        if (schedule.get('increment',5)!=increment or schedule.get('warmup_epochs',0)!=warmup_epochs or
                schedule.get('gir_rule','schur_dead')!=gir_rule):
            raise ValueError('authority schedule incremental/warmup/GIR setup mismatch')
        expected = dict(authoritative=True, code_commit=code_commit(), source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
                        backbone=args.backbone, nobn_lr_scale=args.nobn_lr_scale, curvature=args.curvature,solver=args.solver,policy_version=policy_version,gir_live=args.gir_live,**setup)
        if any(schedule.get(k) != v for k,v in expected.items()): raise ValueError('authority schedule identity mismatch')
        if schedule.get('synthetic',False) and not args.fake_data: raise ValueError('synthetic schedule')
    K0 = int(schedule['K0']) if schedule else 1
    if K0 < 1: raise ValueError('K0 must be positive')
    params = recipe(args.seed, args.device, args.backbone, increment)
    params['shuffle'] = False
    if args.fake_data:
        torch.set_num_threads(1)
        params.update(memory_size=20, memory_per_class=2)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    from experiments.parallel_validity import dataset_fingerprint
    protocol = dict(setting='icarl', algorithm_version='v4', solver=args.solver,
        policy_version=policy_version,gir_live=args.gir_live,lm_acceptance_rho=.25,intervention_start_task=2, arm=args.arm, seed=args.seed,
        code_commit=code_commit(), source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
        upstream_commit=cil.UPSTREAM_PIN, fake_data=args.fake_data, expected_tasks=tasks,
        dataset_fingerprint=None if args.fake_data else dataset_fingerprint(args.data_dir),
        data_dir=args.data_dir, class_order=order, class_order_kind=args.class_order,
        backbone=args.backbone, nobn_lr_scale=args.nobn_lr_scale, coefficient=args.coefficient, curvature=args.curvature,
        r=16, live_candidates=4, compute_budget=1., preflight_only=args.preflight_only, schedule=schedule, K0=K0, cfg=copy.deepcopy(params),
        probe=dict(size=256, b_size=128, b_solver='cg_ir',seed=0, source='seen test classes'),
        feature_probe=dict(tasks=[5,10,15,20], optimizer='LBFGS', max_iter=100,
                           initialization='zeros', data='all seen train, test transforms', regularization=0),
        training=dict(first_epochs=1 if args.fake_data else 100, later_epochs=1 if args.fake_data else 50,
                      first_milestones=[30,60,85], later_milestones=[24,35]),
        resume='native exact epoch checkpoint; task prefix replay',
        model_config=dict(backbone=args.backbone, class_order=args.class_order),
        comparison_protocol=dict(init_cls=increment, increment=increment, memory_size=params['memory_size'], fixed_memory=False,
            backbone=args.backbone, class_order=args.class_order, nobn_lr_scale=args.nobn_lr_scale,
            first_epochs=1 if args.fake_data else 100, later_epochs=1 if args.fake_data else 50))
    if setup:
        protocol.update(setup,class_order_sha256=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest())
        if warmup_epochs:protocol['training']['warmup_epochs']=warmup_epochs
        if increment==10 and warmup_epochs==1:protocol['training_protocol']='P2'
        protocol['comparison_protocol'].update(setup)
    if args.max_tasks != 20: protocol['comparison_protocol']['max_tasks']=args.max_tasks
    if args.grad_clip is not None:protocol['grad_clip']=args.grad_clip
    baseline_reference=None
    if args.baseline_reference:
        if args.arm!='icarl' or args.max_tasks!=20 or args.fake_data or args.preflight_only or setup:
            raise ValueError('baseline reference requires real 20-task icarl')
        from experiments.main_cil_baseline import validate
        data=Path(args.baseline_reference).read_bytes()
        baseline_reference=json.loads(data);validate(baseline_reference,protocol)
        protocol['baseline_reference_sha256']=hashlib.sha256(data).hexdigest()
    progress = Progress(out, protocol)
    if (out/'summary.json').exists():
        previous = json.loads((out/'summary.json').read_text())
        if previous.get('identity') != protocol: raise ValueError('completed identity mismatch')
        if previous['status'] in ('COMPLETED','REUSED_BASELINE'): return previous
    status = Status(out, args.arm)
    engine_options={k:setup[k] for k in ('gir_rule','certswap_topk','certswap_lm_budget','certswap_log_schur') if k in setup}
    intervention = Intervention(V3Engine(ENGINE_ARMS[args.arm], seed=args.seed, coefficient=args.coefficient, K0=K0,
        curvature=args.curvature,solver=args.solver,gir_live=args.gir_live,**engine_options), K0, args.preflight_only) if args.arm in ENGINE_ARMS else None
    if intervention is not None and schedule: intervention.engine.configure_schedule(schedule)
    def persist(hook):
        write_json(out/'tasks.json', hook.tasks)
        write_json(out/'epochs.json', hook.epochs)
        if hook.clipper is not None:hook.clipper.save(out/'grad_clip.json')
        write_csv(out/'continual_eval.csv', [{k:v for k,v in row.items() if not k.endswith('_per_class')} for row in hook.tasks])
        if not (out/'continual_eval.csv').exists(): (out/'continual_eval.csv').write_text('task,cnn_top1,nme_top1\n')
        write_json(out/'diagnostics.json', hook.rows)
        if intervention is not None:
            write_json(out/'interventions.json', intervention.events)
            write_json(out/'stability.json',intervention.followups)
        write_json(out/'protocol.json', {**protocol, 'parameter_count': hook.tasks[-1]['parameter_count'] if hook.tasks else None,
                   'mu0': intervention.engine.protocol.get('mu0') if intervention else None})
    original_manager, original_random = trainer.DataManager, trainer._set_random
    def initialize():
        import random
        original_random()
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    def manager(*a, **kw):
        value = DataManager(*a, **kw)
        value._increments = value._increments[:tasks]
        hook.manager = value
        return value
    hook = MainHook(seed=args.seed, fake=args.fake_data, persist=persist, progress=progress,
                    intervention=intervention, manager=None, status=status)
    if warmup_epochs:hook.warmup_epochs=warmup_epochs
    hook.clipper=GradClip(args.grad_clip) if args.backbone=='resnet32_nobn' else None
    if hook.clipper is not None and hasattr(hook,'grad_clip_state'):hook.clipper.load_state_dict(hook.grad_clip_state)
    hook.step_observer=getattr(args,'_step_observer',None)
    hook.epoch_observer=getattr(args,'_epoch_observer',None)
    hook.baseline_reference=baseline_reference
    started = time.monotonic()
    run_status, reason = 'COMPLETED', None
    try:
        persist(hook)
        trainer.DataManager, trainer._set_random = manager, initialize
        # cil_runtime supplies upstream I/O; runtime applies only registered setup overrides.
        with working_directory(out), cil_runtime('icarl', fake=args.fake_data, data_dir=args.data_dir), runtime(args, order), hook.install():
            trainer.train(copy.deepcopy(params))
    except PreflightComplete:
        points = intervention.engine.protocol['preflight_points']
        schedule = vna_v3.schedule_from_points(points, len(hook.learner.train_loader))
        schedule.update(authoritative=True, setting='icarl', backbone=args.backbone,
            nobn_lr_scale=args.nobn_lr_scale, code_commit=code_commit(), curvature=args.curvature,solver=args.solver,policy_version=policy_version,gir_live=args.gir_live,
            source_digest=os.environ.get('VNA_SOURCE_DIGEST'), synthetic=args.fake_data,
            schedule_source=args.schedule_source, r=16,**setup)
        write_json(Path(args.schedule_output), schedule)
        run_status = 'PREFLIGHT'
    except BaselineReused:
        run_status='REUSED_BASELINE'
        write_json(out/'reused_baseline.json',dict(reference=baseline_reference,
            verification=json.loads((out/'baseline_loss_check.json').read_text())))
    except (Diverged,FloatingPointError,torch.linalg.LinAlgError) as exc:
        if isinstance(exc,TrainingDiverged):hook.diverged=exc.detail
        elif hook.diverged is None:hook.diverged=dict(task=hook.learner._cur_task+1,epoch=hook.current_epoch+1,step=hook.steps,tensor='training')
        run_status, reason = 'FAILED_DIVERGED', str(exc)
        status.update(status=run_status, phase='diverged', diverged=hook.diverged)
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
        init_cls=increment, increment=increment,
        status=run_status, reason=reason, diverged=hook.diverged,
        safe_checkpoint=str(progress.path) if progress.path.exists() else None, tasks=hook.tasks, tasks_run=len(hook.tasks),
        parameter_count=sum(p.numel() for p in hook.learner._network.parameters()),
        seconds=seconds, gpu_hours=seconds/3600 if args.device != 'cpu' else 0.,
        resume_supported=True)
    for label in ('cnn','nme'):
        if hook.tasks and run_status!='REUSED_BASELINE':
            summary.update({label+'_average_accuracy': hook.tasks[-1][label+'_average_accuracy'],
                            label+'_last_accuracy': hook.tasks[-1][label+'_top1'],
                            label+'_forgetting': hook.tasks[-1][label+'_forgetting'],
                            label+'_learning_accuracy': float(np.mean([r[label+'_learning_accuracy'] for r in hook.tasks]))})
    write_json(out/'summary.json', summary)
    if run_status == 'FAILED_DIVERGED':
        log(f"FAILED_DIVERGED {hook.diverged}; safe checkpoint: {summary['safe_checkpoint']}")
    status.update(status=run_status, phase='finished', reason=reason)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, default='icarl')
    p.add_argument('--backbone', choices=('resnet32', 'resnet32_nobn', 'resnet32_gn'), default='resnet32_nobn')
    p.add_argument('--class-order', choices=('superclass','random'), default='superclass')
    p.add_argument('--nobn-lr-scale', type=float, choices=(1.,*NOBN_LR_SCALES), default=.5)
    p.add_argument('--coefficient', type=float, default=1.)
    p.add_argument('--grad-clip',type=positive_tau,default=None)
    p.add_argument('--increment',type=int,choices=(5,10),default=5)
    p.add_argument('--warmup-epochs',type=nonnegative_epochs,default=0)
    p.add_argument('--gir-rule',choices=('schur_dead','certswap'),default='schur_dead')
    p.add_argument('--certswap-topk',type=int,default=8)
    p.add_argument('--certswap-lm-budget',type=int,default=20)
    p.add_argument('--certswap-log-schur',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--gir-live',action='store_true',help='Ablation: allow live-slot GIR')
    p.add_argument('--solver',choices=('gram','cg_ir'),default='cg_ir')
    p.add_argument('--baseline-reference',help='Completed old baseline + 150 full-precision epoch losses')
    p.add_argument('--curvature', choices=('power10','exact'), default='exact')
    p.add_argument('--max-tasks', type=int, default=20)
    p.add_argument('--schedule')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--schedule-output')
    p.add_argument('--schedule-source', default='authority', choices=('authority','self'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=('cpu','cuda:0'), default='cuda:0')
    p.add_argument('--data-dir', default=DATA_ROOT)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--fake-data', action='store_true')
    p.add_argument('--out', required=True)
    return p


if __name__ == '__main__': run(parser().parse_args())
