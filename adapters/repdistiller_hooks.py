"""Read-only adapters for the vendored RepDistiller (BSD-2-Clause) entry points."""
from contextlib import contextmanager
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
                 data_dir=None, batch_size=64, pilot=False, pilot_batches=20):
        self.buffer = MeasurementBuffer(4 if pilot and fake else 128)
        self.pilot, self.pilot_batches = pilot, pilot_batches
        self.measurement = MeasurementHook(measure, seed=0) if measure else None
        self.persist, self.observe = persist, observe
        self.smoke, self.fake, self.data_dir, self.batch_size = smoke, fake, data_dir, batch_size
        self.rows, self.epochs, self.point_seconds = [], [], []
        self.steps = 0
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
        batch = tuple(t.to(next(self.model.parameters()).device) for t in self.buffer.batch())
        spec, ctx = loss_context(self.teacher, batch)
        ctx.update(teacher=self.teacher, epoch=epoch, step=self.steps, task=0, end_of_task=False,
                   known_classes=0, train_acc=self.epochs[-1]['train_acc'])
        started = time.monotonic()
        self.rows.extend(self.measurement(self.model, batch, spec, ctx) or [])
        self.point_seconds.append({'epoch': epoch, 'seconds': time.monotonic()-started})
        if self.persist: self.persist(self)

    @contextmanager
    def install(self, entry):
        original_lr, original_loader = entry.adjust_learning_rate, entry.get_cifar100_dataloaders
        from dataset import cifar100
        folder = cifar100.get_data_folder
        def scheduler(epoch, opt, optimizer):
            frame = inspect.currentframe().f_back
            self.last_frame = frame
            values = frame.f_locals
            self.model = values.get('model_s', values.get('model'))
            self.teacher = values.get('model_t')
            self.optimizer, self.opt = optimizer, opt
            if epoch > 1: self.finish_epoch(epoch-1, values)
            return original_lr(epoch, opt, optimizer)
        try:
            entry.adjust_learning_rate = scheduler
            entry.get_cifar100_dataloaders = lambda **kw: self.loaders(original_loader, **kw)
            if self.data_dir: cifar100.get_data_folder = lambda: str(Path(self.data_dir).resolve())
            yield self
        finally:
            entry.adjust_learning_rate, entry.get_cifar100_dataloaders = original_lr, original_loader
            cifar100.get_data_folder = folder

    def finish_epoch(self, epoch, values):
        self.epochs.append({'epoch': epoch, 'train_acc': float(values['train_acc'])/100,
                            'train_loss': float(values['train_loss']), 'eval_acc': float(values['test_acc'])/100,
                            'eval_loss': float(values['test_loss']), 'lr': self.optimizer.param_groups[0]['lr'],
                            'seconds': values['time2']-values['time1']})
        if self.observe: self.observe(self.model, self.optimizer)
        if (epoch == 2 if self.pilot else epoch in POINTS or self.smoke and epoch == 2): self.measure(epoch)
        if self.persist: self.persist(self)
