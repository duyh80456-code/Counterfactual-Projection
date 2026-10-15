"""Read-only adapters for the vendored RepDistiller (BSD-2-Clause) entry points."""
from contextlib import contextmanager
import copy
import importlib
import inspect
import os
from pathlib import Path
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from diagnostics.vna_align import MeasurementBuffer, MeasurementHook, _global_state
from experiments.cifar_pilot import LimitedLoader
from methods.vna_losses import CE, KDRepDistiller
from methods.vna_sites import HookSite
from experiments.vna_resume import (log, set_locals, hook_state, restore_hook,
                                    rng_state, restore_rng, projected)

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT/'third_party/RepDistiller'
PAIRS = {'resnet8x4': 'resnet32x4', 'wrn_16_2': 'wrn_40_2'}
POINTS = set(range(20, 241, 20)) | {148, 178, 208}


@contextmanager
def upstream():
    """Keep upstream's unqualified imports separate from PyCIL's models/utils."""
    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    roots = {'models', 'dataset', 'helper', 'distiller_zoo', 'crd', 'train_student', 'train_teacher'}
    def owned(name): return name.split('.')[0] in roots
    saved = {n: m for n, m in sys.modules.items() if owned(n)}
    path = sys.path[:]
    for name in saved: del sys.modules[name]
    sys.path.insert(0, str(UPSTREAM))
    try:
        yield importlib.import_module('models').model_dict
    finally:
        for name in list(sys.modules):
            if owned(name): del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = path


def sites_for(model):
    sites = []
    for path, block in model.named_modules():
        if block.__class__.__name__ != 'BasicBlock': continue
        if not all(hasattr(block, n) for n in ('conv1', 'conv2')): continue
        middle = ('bn2', 'relu2') if hasattr(block, 'relu2') else ('bn1', 'relu')
        sites.append(HookSite(model, path+'.conv1', path+'.conv2', 'conv',
                              between=tuple(path+'.'+n for n in middle)))
    return sites


def loss_context(teacher, batch):
    ctx = {'targets': batch[1]}
    ce = nn.CrossEntropyLoss()
    if teacher is None:
        ctx['loss_fn'] = lambda z, c: ce(z, c['targets'])
        return CE(), ctx
    from distiller_zoo.KD import DistillKL
    kd = DistillKL(4)
    modes = {m: m.training for m in teacher.modules()}
    with _global_state():
        try:
            teacher.eval()
            with torch.no_grad(): ctx['teacher_logits'] = teacher(batch[0]).detach().double()
        finally:
            for m, mode in modes.items(): m.training = mode
    ctx['loss_fn'] = lambda z, c: .1*ce(z, c['targets']) + .9*kd(z, c['teacher_logits'])
    return KDRepDistiller(alpha=.1, beta=.9, T=4), ctx


class BufferedLoader(LimitedLoader):
    def __init__(self, loader, hook, limit=None):
        super().__init__(loader, limit)
        self.hook = hook
    def __iter__(self):
        for batch in super().__iter__():
            self.hook.buffer.update(batch[:2]); self.hook.steps += 1
            yield batch


class RepDistillerHook:
    def __init__(self, measure=None, *, persist=None, observe=None, smoke=False, fake=False,
                 data_dir=None, batch_size=64, pilot=False, pilot_batches=20, progress=None):
        self.buffer = MeasurementBuffer(4 if pilot and fake else 512)
        self.pilot, self.pilot_batches = pilot, pilot_batches
        self.measurement = MeasurementHook(measure, seed=0) if measure else None
        self.persist, self.observe = persist, observe
        self.smoke, self.fake, self.data_dir, self.batch_size = smoke, fake, data_dir, batch_size
        self.rows, self.epochs, self.point_seconds = [], [], []
        self.steps = 0
        self.progress = progress
        self.resume = progress.state if progress else None
        self.replay_epoch = 0
        self.returns = {}
        self.final_test = None
        self.pending_measure = None
        if self.resume:
            restore_hook(self, self.resume["hook"])
            self.returns = copy.deepcopy(self.resume["returns"])
            self.final_test = self.resume.get("final_test")
            self.pending_measure = self.resume.get("pending_measure")
            progress.restore_artifacts()
        self.model = self.teacher = self.optimizer = self.opt = self.test_loader = None

    def loaders(self, original, **kwargs):
        instance = kwargs.get('is_instance', False)
        if self.fake:
            g = torch.Generator().manual_seed(871)
            x = torch.randn(8, 3, 32, 32, generator=g)
            y = torch.arange(8)%100
            train = TensorDataset(x, y, torch.arange(8)) if instance else TensorDataset(x, y)
            evaluation = TensorDataset(x[:4], y[:4])
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            val = DataLoader(evaluation, batch_size=self.batch_size)
            self.test_loader = val
        else:
            from experiments.vna_support import setup_data
            result = original(**kwargs)
            native, self.test_loader = result[:2]
            _, _, indices, trigger, evaluation, _ = setup_data(self.data_dir or 'data', False, 0)
            # Instance indices are needed by train_distill, but validate expects pairs.
            from torchvision.datasets import CIFAR100
            holdout = CIFAR100(root=native.dataset.root, train=True,
                               transform=self.test_loader.dataset.transform, download=False)
            train_loader = DataLoader(Subset(native.dataset, indices), batch_size=native.batch_size,
                                      shuffle=True, num_workers=native.num_workers)
            val = DataLoader(Subset(holdout, trigger+evaluation), batch_size=self.test_loader.batch_size,
                             shuffle=False, num_workers=self.test_loader.num_workers)
        self.full_batches = len(train_loader)
        self.train_batches = min(len(train_loader), self.pilot_batches) if self.pilot else len(train_loader)
        if self.pilot:
            val = LimitedLoader(val, 5)
            self.test_loader = LimitedLoader(self.test_loader, 5)
        result = (BufferedLoader(train_loader, self, self.pilot_batches if self.pilot else None), val)
        return (*result, len(train_loader.dataset)) if instance else result

    def measure(self, epoch):
        if not self.measurement: return
        if self.smoke and self.point_seconds: return
        batch = tuple(t.to(next(self.model.parameters()).device) for t in self.buffer.batch())
        spec, ctx = loss_context(self.teacher, batch)
        ctx.update(teacher=self.teacher, epoch=epoch, step=self.steps, task=0, end_of_task=False,
                   known_classes=0, train_acc=self.epochs[-1]['train_acc'])
        self.boundary()
        started = time.monotonic()
        self.rows.extend(self.measurement(self.model, batch, spec, ctx) or [])
        self.point_seconds.append({'epoch': epoch, 'seconds': time.monotonic()-started})
        projected(self, self.opt.epochs, 1 if self.pilot or self.smoke else len(POINTS))
        if self.persist: self.persist(self)

    def checkpoint(self):
        if self.progress is None or self.model is None: return
        values = self.last_frame.f_locals
        self.progress.save({'epoch': len(self.epochs), 'model': self.model.state_dict(),
            'teacher': None if self.teacher is None else self.teacher.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'rng': rng_state(),
            'best_acc': values.get('best_acc', 0), 'hook': hook_state(self),
            'pending_measure': self.pending_measure, 'returns': self.returns, 'teacher_eval': getattr(self, 'teacher_eval', self.progress.state.get('teacher_eval') if self.progress.state else None), 'final_test': self.final_test})

    def boundary(self):
        if self.progress:
            self.progress.boundary(self.checkpoint, self.epochs[-1]['seconds'] if self.epochs else 0.,
                self.point_seconds[-1]['seconds'] if self.point_seconds else 0.)

    def restore(self, frame):
        if self.resume is None: return
        state = self.resume
        self.model.load_state_dict(state['model'])
        if self.teacher is not None: self.teacher.load_state_dict(state['teacher'])
        self.optimizer.load_state_dict(state['optimizer'])
        set_locals(frame, {'best_acc': state['best_acc']})
        restore_rng(state['rng'])
        self.resume = None

    @contextmanager
    def install(self, entry):
        original_lr, original_loader = entry.adjust_learning_rate, entry.get_cifar100_dataloaders
        original_train, original_validate, original_save = entry.train, entry.validate, torch.save
        from dataset import cifar100
        from helper import loops
        from unittest.mock import patch
        import builtins
        def epoch_print(*args, **kwargs):
            if args and str(args[0]).startswith(('Epoch: [', 'Test: [')):
                return
            return builtins.print(*args, **kwargs)
        folder = cifar100.get_data_folder
        def scheduler(epoch, opt, optimizer):
            frame = inspect.currentframe().f_back
            self.last_frame = frame
            values = frame.f_locals
            self.model = values.get('model_s', values.get('model'))
            self.teacher = values.get('model_t')
            self.optimizer, self.opt = optimizer, opt
            self.replay_epoch = epoch
            if self.resume is not None and epoch <= self.resume["epoch"]:
                return original_lr(epoch, opt, optimizer)
            self.restore(frame)
            if epoch > 1: self.finish_epoch(epoch-1, frame.f_locals)
            self.boundary()
            return original_lr(epoch, opt, optimizer)
        def train(epoch, *args, **kwargs):
            if self.resume is not None and epoch <= self.resume['epoch']:
                return self.returns[epoch]['train']
            result = original_train(epoch, *args, **kwargs)
            self.returns.setdefault(epoch, {})['train'] = copy.deepcopy(result)
            return result
        def validate(*args, **kwargs):
            frame = inspect.currentframe().f_back
            epoch = frame.f_locals.get('epoch') if frame.f_code is entry.main.__code__ else None
            if epoch is None and self.resume is not None and 'teacher_eval' in self.resume:
                return self.resume['teacher_eval']
            if epoch is not None and self.resume is not None and epoch <= self.resume['epoch']:
                return self.returns[epoch]['eval']
            result = original_validate(*args, **kwargs)
            if epoch is not None: self.returns.setdefault(epoch, {})['eval'] = copy.deepcopy(result)
            else: self.teacher_eval = copy.deepcopy(result)
            return result
        def save(*args, **kwargs):
            # Upstream replay must never overwrite best/save_freq snapshots.
            frame = inspect.currentframe().f_back
            if frame.f_code is entry.main.__code__ and self.resume is not None:
                return
            return original_save(*args, **kwargs)
        try:
            entry.train, entry.validate, torch.save = train, validate, save
            entry.adjust_learning_rate = scheduler
            entry.get_cifar100_dataloaders = lambda **kw: self.loaders(original_loader, **kw)
            if self.data_dir: cifar100.get_data_folder = lambda: str(Path(self.data_dir).resolve())
            with patch.object(loops, 'print', epoch_print, create=True):
                yield self
        finally:
            entry.adjust_learning_rate, entry.get_cifar100_dataloaders = original_lr, original_loader
            cifar100.get_data_folder = folder
            entry.train, entry.validate, torch.save = original_train, original_validate, original_save

    def finish_epoch(self, epoch, values):
        self.restore(self.last_frame) if self.resume is not None and epoch >= self.resume["epoch"] else None
        if epoch <= len(self.epochs):
            if self.pending_measure == epoch:
                self.measure(epoch); self.pending_measure = None; self.checkpoint()
            return
        self.epochs.append({'epoch': epoch, 'train_acc': float(values['train_acc'])/100,
                            'train_loss': float(values['train_loss']), 'eval_acc': float(values['test_acc'])/100,
                            'eval_loss': float(values['test_loss']), 'lr': self.optimizer.param_groups[0]['lr'],
                            'seconds': values['time2']-values['time1']})
        log(f"epoch={epoch} train_loss={float(values['train_loss']):.6f} train_acc={float(values['train_acc']):.6f} "
            f"eval_loss={float(values['test_loss']):.6f} eval_acc={float(values['test_acc']):.6f} "
            f"lr={self.optimizer.param_groups[0]['lr']} seconds={self.epochs[-1]['seconds']:.3f}")
        if self.observe: self.observe(self.model, self.optimizer)
        if (epoch == 2 if self.pilot else epoch in POINTS or self.smoke and epoch == 2):
            self.pending_measure = epoch
            self.measure(epoch)
            self.pending_measure = None
        if self.persist: self.persist(self)
        if epoch % 10 == 0 or epoch in POINTS or epoch == self.opt.epochs or self.smoke and epoch == 2:
            self.checkpoint()
        self.boundary()
