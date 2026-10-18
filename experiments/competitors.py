"""Unmodified pinned PyCIL and LoP competitors on the registered protocol."""
import copy
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.runners import pycil as cil, lop

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
