"""Read-only interception of the pinned PyCIL loader and epoch scheduler."""
from contextlib import contextmanager
import inspect
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from diagnostics.vna_align import MeasurementBuffer, MeasurementHook, _global_state
from methods.vna_losses import CE, ICaRLPyCIL
from methods.vna_sites import HookSite


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
    def __init__(self, measure=None, *, seed=1993, persist=None, observe=None):
        self.buffer = MeasurementBuffer(128)
        self.measurement = MeasurementHook(measure, seed=seed) if measure else None
        self.persist, self.observe = persist, observe
        self.rows, self.epochs, self.tasks, self.point_seconds = [], [], [], []
        self.learner = None
        self.steps = 0

    def after_epoch(self, learner, optimizer, frame):
        from models import icarl
        values = frame.f_locals
        epoch = int(values['epoch']) + 1
        last = icarl.init_epoch if learner._cur_task == 0 else icarl.epochs
        train_acc = round(float(values['correct']) * 100 / values['total'], 2) / 100
        self.epochs.append({'task': learner._cur_task+1, 'epoch': epoch,
                            'train_loss': values['losses']/len(learner.train_loader),
                            'train_acc': train_acc})
        if self.observe: self.observe(learner, optimizer)
        if self.measurement and epoch in {1, 10, max(1, last//2), last}:
            device = next(learner._network.parameters()).device
            batch = tuple(t.to(device) for t in self.buffer.batch())
            model = Prediction(learner._network)
            spec, loss_ctx = loss_context(learner, batch)
            ctx = {**loss_ctx, 'learner': learner, 'task': learner._cur_task+1,
                   'epoch': epoch, 'step': self.steps, 'end_of_task': epoch == last,
                   'train_acc': train_acc}
            started = time.monotonic()
            self.rows.extend(self.measurement(model, batch, spec, ctx) or [])
            self.point_seconds.append({'task': ctx['task'], 'epoch': epoch,
                                       'batch_size': len(batch[0]), 'seconds': time.monotonic()-started})
        if self.persist: self.persist(self)

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
        hook = self
        class BufferedLoader(original_loader):
            def __iter__(self):
                for item in super().__iter__():
                    if hook.learner is not None and self is hook.learner.train_loader:
                        hook.buffer.update((item[1], item[2])); hook.steps += 1
                    yield item
        class EpochScheduler(original_scheduler):
            def __init__(self, *args, **kwargs):
                self.ready = False
                super().__init__(*args, **kwargs)
                self.ready = True
            def step(self, *args, **kwargs):
                result = super().step(*args, **kwargs)
                if self.ready and hook.learner is not None:
                    hook.after_epoch(hook.learner, self.optimizer, inspect.currentframe().f_back)
                return result
        def train(learner, *args, **kwargs):
            hook.learner = learner
            hook.buffer.samples.clear()
            try: return original_train(learner, *args, **kwargs)
            finally: hook.learner = None
        def evaluate(learner, *args, **kwargs):
            result = original_eval(learner, *args, **kwargs)
            hook.record_task(learner, result)
            return result
        try:
            icarl.DataLoader = BufferedLoader
            torch.optim.lr_scheduler.MultiStepLR = EpochScheduler
            icarl.iCaRL._train, icarl.iCaRL.eval_task = train, evaluate
            yield self
        finally:
            icarl.DataLoader = original_loader
            torch.optim.lr_scheduler.MultiStepLR = original_scheduler
            icarl.iCaRL._train, icarl.iCaRL.eval_task = original_train, original_eval
