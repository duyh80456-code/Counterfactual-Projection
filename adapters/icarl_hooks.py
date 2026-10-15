"""Read-only interception of the pinned PyCIL loader and epoch scheduler."""
from contextlib import contextmanager
import copy
import inspect
from itertools import islice
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from diagnostics.vna_align import MeasurementBuffer, MeasurementHook, _global_state
from methods.vna_losses import CE, ICaRLPyCIL
from methods.vna_sites import HookSite
from experiments.vna_resume import (log, hook_state, restore_hook, rng_state,
                                    restore_rng, projected, set_locals)


class Prediction(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        # Upstream uses F.relu; this stateless module acts on virtual channels only.
        self.virtual_relu = nn.ReLU()

    def forward(self, x):
        return self.native(x)['logits']


def sites_for(model):
    from convs.cifar_resnet import ResNetBasicblock
    return [HookSite(model, f'{path}.conv_a', f'{path}.conv_b', 'conv',
                     between=(f'{path}.bn_a', 'virtual_relu'))
            for path, block in model.named_modules() if isinstance(block, ResNetBasicblock)]


def loss_context(learner, batch):
    from models.icarl import _KD_loss
    known = learner._known_classes
    ctx = {'targets': batch[1], 'known_classes': known}
    if known:
        teacher = learner._old_network
        modes = {m: m.training for m in teacher.modules()}
        with _global_state():
            try:
                teacher.eval()
                with torch.no_grad(): ctx['teacher_logits'] = teacher(batch[0])['logits'].detach()
            finally:
                for module, mode in modes.items(): module.training = mode
    def upstream_loss(z, c):
        loss = F.cross_entropy(z, c['targets'])
        return loss + _KD_loss(z[:, :known], c['teacher_logits'], 2) if known else loss
    ctx['loss_fn'] = upstream_loss
    return (ICaRLPyCIL(T=2, known_classes=known) if known else CE()), ctx


class ICaRLHook:
    def __init__(self, measure=None, *, seed=1993, persist=None, observe=None, pilot=False, pilot_batches=20, fake=False, progress=None):
        self.buffer = MeasurementBuffer(4 if pilot and fake else 512)
        self.pilot, self.pilot_batches = pilot, pilot_batches
        self.batch_counts = []
        self.measurement = MeasurementHook(measure, seed=seed) if measure else None
        self.persist, self.observe = persist, observe
        self.rows, self.epochs, self.tasks, self.point_seconds = [], [], [], []
        self.learner = None
        self.steps = 0
        self.progress = progress
        self.resume = progress.state if progress else None
        self.eval_results = []
        self.completed = 0
        self.replay_task = -1
        self.pending_measure = None
        self.optimizer = self.scheduler = None
        self.task_start = None
        self.current_epoch = 0
        self.trainer_frame = None
        self.loop_info = None
        if self.resume:
            restore_hook(self, self.resume['hook'])
            self.completed = self.resume['completed']
            self.eval_results = copy.deepcopy(self.resume['eval_results'])
            self.pending_measure = self.resume.get('pending_measure')
            progress.restore_artifacts()

    @staticmethod
    def learner_state(learner):
        # Networks are objects: include grown heads, frozen teacher and all memory.
        return copy.deepcopy({k: v for k, v in vars(learner).items()
                              if k not in ('train_loader', 'test_loader')})

    def checkpoint(self, *, mid=True, frame=None):
        if self.progress is None or self.learner is None: return
        frame = frame or self.trainer_frame
        curves = {k: copy.deepcopy(frame.f_locals[k]) for k in
                  ('cnn_curve', 'nme_curve', 'cnn_matrix', 'nme_matrix')
                  if frame is not None and k in frame.f_locals}
        if not mid and self.eval_results and frame is not None and frame.f_locals.get('task') == self.completed-1:
            for label, result in zip(('cnn', 'nme'), self.eval_results[-1]):
                if result is not None and label+'_curve' in curves:
                    for metric in ('top1', 'top5'): curves[label+'_curve'][metric].append(result[metric])
                    curves[label+'_matrix'].append([result['grouped'][k] for k in sorted(result['grouped']) if '-' in k])
        self.progress.save({'completed': self.completed, 'mid': mid,
            'learner': self.learner_state(self.learner), 'task_start': self.task_start,
            'epoch': self.current_epoch, 'loop_info': self.loop_info,
            'optimizer': None if self.optimizer is None else self.optimizer.state_dict(),
            'scheduler': None if self.scheduler is None else self.scheduler.state_dict(),
            'rng': rng_state(), 'hook': hook_state(self), 'curves': curves,
            'eval_results': self.eval_results, 'pending_measure': self.pending_measure})

    def boundary(self):
        if self.progress:
            self.progress.boundary(self.checkpoint,
                self.epochs[-1]['seconds'] if self.epochs else 0.,
                self.point_seconds[-1]['seconds'] if self.point_seconds else 0.)


    def after_epoch(self, learner, optimizer, frame):
        from models import icarl
        values = frame.f_locals
        epoch = int(values['epoch']) + 1
        self.current_epoch = epoch
        self.loop_info = values.get('info')
        last = icarl.init_epoch if learner._cur_task == 0 else icarl.epochs
        train_acc = round(float(values['correct']) * 100 / values['total'], 2) / 100
        self.epochs.append({'task': learner._cur_task+1, 'epoch': epoch,
                            'train_loss': values['losses']/len(learner.train_loader),
                            'train_acc': train_acc, 'eval_acc': float(values['test_acc'])/100 if epoch % 5 == 1 else None,
                            'eval_loss': None, 'lr': self.epoch_lr, 'seconds': time.monotonic()-self.epoch_started})
        row = self.epochs[-1]
        log(f"task={row['task']} epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"train_acc={train_acc:.6f} eval_loss={row['eval_loss']} eval_acc={row['eval_acc']} "
            f"lr={row['lr']} seconds={row['seconds']:.3f}")
        if self.observe: self.observe(learner, optimizer)
        point = learner._cur_task == 1 and epoch == 2 if self.pilot else measurement_epoch(learner._cur_task+1, epoch, last)
        if self.measurement and getattr(self.measurement.measure, 'metadata', {}).get('smoke') and any(p.get('task') == learner._cur_task+1 for p in self.point_seconds):
            point = False
        if self.measurement and point:
            self.pending_measure = epoch
            self.measure_point(learner, epoch, train_acc, last)
            self.pending_measure = None
            if any(r.get('status') == 'timeout' for r in self.rows): self.checkpoint()
        if self.persist: self.persist(self)
        if epoch % 25 == 0: self.checkpoint()
        self.boundary()

    def measure_point(self, learner, epoch, train_acc, last):
        self.boundary()
        device = next(learner._network.parameters()).device
        batch = tuple(t.to(device) for t in self.buffer.batch())
        model = Prediction(learner._network)
        spec, loss_ctx = loss_context(learner, batch)
        ctx = {**loss_ctx, 'learner': learner, 'task': learner._cur_task+1,
               'epoch': epoch, 'step': self.steps, 'end_of_task': epoch == last,
               'train_acc': train_acc,
               'first_measurement_point': learner._cur_task == 0 and epoch == max(1, last//2)}
        started = time.monotonic()
        self.rows.extend(self.measurement(model, batch, spec, ctx) or [])
        self.point_seconds.append({'task': ctx['task'], 'epoch': epoch,
                                   'batch_size': len(batch[0]), 'seconds': time.monotonic()-started})
        projected(self, 1730, 13)

    def record_task(self, learner, result):
        row = {'task': learner._cur_task+1}
        for label, accuracy in zip(('cnn', 'nme'), result):
            if accuracy is None: continue
            row[f'{label}_top1'] = float(accuracy['top1'])
            row[f'{label}_grouped'] = accuracy['grouped']
            history = [r[f'{label}_top1'] for r in self.tasks] + [row[f'{label}_top1']]
            row[f'{label}_average_accuracy'] = float(np.mean(history))
            # PyCIL's grouped class bins are ten classes in the real protocol.
            keys = sorted(k for k in accuracy['grouped'] if '-' in k)
            drops = [max(r[f'{label}_grouped'].get(k, accuracy['grouped'][k])
                         for r in self.tasks) - accuracy['grouped'][k] for k in keys[:-1]]
            row[f'{label}_forgetting'] = float(np.mean(drops)) if drops else 0.
        row.update({k: self.epochs[-1][k] for k in ('train_loss', 'train_acc')})
        row['memory_size'] = len(learner._data_memory)
        row['memory_per_class'] = learner.samples_per_class
        self.tasks.append(row)
        if self.persist: self.persist(self)

    @contextmanager
    def install(self):
        from models import icarl
        original_loader = icarl.DataLoader
        original_scheduler = torch.optim.lr_scheduler.MultiStepLR
        original_train, original_eval = icarl.iCaRL._train, icarl.iCaRL.eval_task
        original_incremental, original_after = icarl.iCaRL.incremental_train, icarl.iCaRL.after_task
        original_tqdm = icarl.tqdm
        hook = self
        class BufferedLoader(original_loader):
            limit = None
            def __len__(self):
                length = super().__len__()
                return min(length, self.limit) if self.limit else length
            def __iter__(self):
                if hook.learner is not None and self is hook.learner.train_loader:
                    hook.epoch_started = time.monotonic()
                    hook.epoch_lr = hook.optimizer.param_groups[0]['lr']
                iterator = super().__iter__()
                for item in islice(iterator, self.limit) if self.limit else iterator:
                    if hook.learner is not None and self is hook.learner.train_loader:
                        hook.buffer.update((item[1], item[2])); hook.steps += 1
                    yield item
        class EpochScheduler(original_scheduler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                hook.optimizer, hook.scheduler = self.optimizer, self
                if hook.resume is not None and hook.resume['mid']:
                    self.optimizer.load_state_dict(hook.resume['optimizer'])
                    self.load_state_dict(hook.resume['scheduler'])
                    restore_rng(hook.resume['rng'])
        class EpochProgress:
            def __init__(self, iterable):
                start = hook.resume['epoch'] if hook.resume is not None and hook.resume['mid'] else 0
                self.bar = original_tqdm(list(iterable)[start:])
            def set_description(self, *args, **kwargs): return self.bar.set_description(*args, **kwargs)
            def __iter__(self):
                if hook.resume is not None and hook.resume['mid']:
                    # At a final-epoch pause the upstream loop has no live iteration,
                    # but its trailing logging.info(info) still reads this local.
                    hook.loop_info = hook.resume.get('loop_info')
                    set_locals(inspect.currentframe().f_back, {'info': hook.loop_info})
                    if hook.pending_measure is not None:
                        epoch = hook.pending_measure
                        last = icarl.init_epoch if hook.learner._cur_task == 0 else icarl.epochs
                        hook.measure_point(hook.learner, epoch, hook.epochs[-1]['train_acc'], last)
                        hook.pending_measure = None
                    hook.resume = None
                for epoch in self.bar:
                    hook.boundary()
                    yield epoch
                    hook.after_epoch(hook.learner, hook.optimizer, inspect.currentframe().f_back)
        def incremental(learner, *args, **kwargs):
            hook.trainer_frame = inspect.currentframe().f_back
            hook.replay_task += 1
            hook.learner = learner
            if hook.replay_task < hook.completed: return
            if hook.resume is not None:
                state = hook.resume['task_start'] if hook.resume['mid'] else hook.resume['learner']
                vars(learner).update(copy.deepcopy(state))
                if not hook.resume['mid']:
                    restore_rng(hook.resume['rng']); hook.resume = None
            hook.task_start = hook.learner_state(learner)
            if hook.progress:
                hook.progress.boundary(lambda: hook.checkpoint(mid=False),
                    hook.epochs[-1]['seconds'] if hook.epochs else 0.,
                    hook.point_seconds[-1]['seconds'] if hook.point_seconds else 0.)
            return original_incremental(learner, *args, **kwargs)
        def after(learner, *args, **kwargs):
            if hook.replay_task < hook.completed:
                if hook.resume is not None and not hook.resume['mid'] and hook.replay_task == hook.completed-1:
                    vars(learner).update(copy.deepcopy(hook.resume['learner']))
                    restore_rng(hook.resume['rng']); hook.resume = None
                return
            result = original_after(learner, *args, **kwargs)
            hook.completed = hook.replay_task+1
            # Curves currently exclude this task; replayed eval results rebuild the
            # exact trainer-side curves including its just-completed task.
            hook.checkpoint(mid=False, frame=inspect.currentframe().f_back)
            return result
        def train(learner, *args, **kwargs):
            hook.learner = learner
            hook.current_epoch = hook.resume['epoch'] if hook.resume is not None and hook.resume['mid'] else 0
            if hook.resume is not None and hook.resume['mid']:
                vars(learner).update(copy.deepcopy(hook.resume['learner']))
            else:
                hook.buffer.samples.clear()
            if hook.pilot:
                full = len(learner.train_loader)
                learner.train_loader.limit = hook.pilot_batches
                learner.test_loader.limit = 5
                if hook.resume is None or not hook.resume['mid']:
                    hook.batch_counts.append({'full': full, 'pilot': len(learner.train_loader)})
            return original_train(learner, *args, **kwargs)
        def evaluate(learner, *args, **kwargs):
            if hook.replay_task < hook.completed:
                return copy.deepcopy(hook.eval_results[hook.replay_task])
            result = original_eval(learner, *args, **kwargs)
            hook.eval_results.append(copy.deepcopy(result))
            hook.record_task(learner, result)
            return result
        try:
            icarl.tqdm = EpochProgress
            icarl.iCaRL.incremental_train, icarl.iCaRL.after_task = incremental, after
            icarl.DataLoader = BufferedLoader
            torch.optim.lr_scheduler.MultiStepLR = EpochScheduler
            icarl.iCaRL._train, icarl.iCaRL.eval_task = train, evaluate
            yield self
        finally:
            icarl.tqdm = original_tqdm
            icarl.iCaRL.incremental_train, icarl.iCaRL.after_task = original_incremental, original_after
            icarl.DataLoader = original_loader
            torch.optim.lr_scheduler.MultiStepLR = original_scheduler
            icarl.iCaRL._train, icarl.iCaRL.eval_task = original_train, original_eval


def measurement_epoch(task, epoch, last):
    """Ten midpoint observations plus reference endpoints at tasks 2, 5, 10."""
    return epoch == max(1, last//2) or (epoch == last and task in {2, 5, 10})
