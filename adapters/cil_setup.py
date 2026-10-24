"""CIFAR-100 superclass order and the PyCIL SkipInit backbone."""
import pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn


# CIFAR-100 python meta order, with fine labels in their official numeric order.
COARSE_NAMES = ('aquatic_mammals fish flowers food_containers fruit_and_vegetables '
                'household_electrical_devices household_furniture insects large_carnivores '
                'large_man-made_outdoor_things large_natural_outdoor_scenes large_omnivores_and_herbivores '
                'medium_mammals non-insect_invertebrates people reptiles small_mammals trees '
                'vehicles_1 vehicles_2').split()
COARSE_GROUPS = ((4,30,55,72,95), (1,32,67,73,91), (54,62,70,82,92),
                 (9,10,16,28,61), (0,51,53,57,83), (22,39,40,86,87),
                 (5,20,25,84,94), (6,7,14,18,24), (3,42,43,88,97),
                 (12,17,37,68,76), (23,33,49,60,71), (15,19,21,31,38),
                 (34,63,64,66,75), (26,45,77,79,99), (2,11,35,46,98),
                 (27,29,44,78,93), (36,50,65,74,80), (47,52,56,59,96),
                 (8,13,48,58,90), (41,69,81,85,89))
FINE_TO_COARSE = tuple(next(c for c, group in enumerate(COARSE_GROUPS) if f in group)
                       for f in range(100))


def class_order(seed, order='superclass', data_dir=None):
    rng = np.random.RandomState(seed)
    if order == 'random': return rng.permutation(100).tolist()
    if order != 'superclass': raise ValueError(order)
    mapping = np.array(FINE_TO_COARSE)
    if data_dir:
        root = Path(data_dir)
        if (root/'cifar-100-python').is_dir(): root = root/'cifar-100-python'
        with (root/'train').open('rb') as stream: train = pickle.load(stream, encoding='latin1')
        fine, coarse = train.get('fine_labels'), train.get('coarse_labels')
        if coarse is not None:
            fine, coarse = np.asarray(fine), np.asarray(coarse)
            if len(fine) != len(coarse): raise ValueError('fine/coarse label length mismatch')
            values = [np.unique(coarse[fine == f]) for f in range(100)]
            if any(len(v) != 1 for v in values): raise ValueError('invalid fine/coarse partition')
            mapping = np.array([v[0] for v in values])
            if sorted(np.bincount(mapping, minlength=20).tolist()) != [5]*20:
                raise ValueError('CIFAR-100 must have 20 groups of five')
        if (root/'meta').exists():
            with (root/'meta').open('rb') as stream: meta = pickle.load(stream, encoding='latin1')
            names = meta.get('coarse_label_names')
            if names is not None:
                names = [n.decode() if isinstance(n, bytes) else n for n in names]
                if names != COARSE_NAMES: raise ValueError('unexpected CIFAR-100 coarse label names')
    return [int(f) for c in rng.permutation(20) for f in rng.permutation(np.flatnonzero(mapping == c))]


class SkipScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, x): return self.alpha*x


def skipinit(model):
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, SkipScale().to(child.weight) if name == 'bn_b' else nn.Identity())
            elif isinstance(child, nn.Conv2d) and child.bias is None:
                child.bias = nn.Parameter(child.weight.new_zeros(child.out_channels))
    return model


def resnet32_nobn():
    from convs.cifar_resnet import resnet32
    return skipinit(resnet32())


GN_GROUPS = 8  # registered GroupNorm variant (V_O/h check only): 8 groups, affine weight 1, bias 0


def resnet32_gn():
    """PyCIL resnet32 with every BatchNorm2d replaced by GroupNorm(8, C); convolutions unchanged."""
    from convs.cifar_resnet import resnet32
    model = resnet32()
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                norm = nn.GroupNorm(GN_GROUPS, child.num_features).to(child.weight)
                # Zero-init gamma of each residual block's last norm (Goyal et al. 2017; Wu & He 2018).
                if name == 'bn_b': nn.init.zeros_(norm.weight)
                setattr(module, name, norm)
    return model


class FixtureBackbone(nn.Module):
    """Two-channel residual fixture; the production backbone is never reduced."""
    def __init__(self, block):
        super().__init__()
        self.conv = nn.Conv2d(3,2,3,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.branch = block(2,2)
        self.out_dim = 2
    def forward(self,x):
        x = nn.functional.adaptive_avg_pool2d(x, 4)
        x = self.branch(torch.relu(self.bn(self.conv(x))))
        return dict(features=x.mean((2,3)),fmaps=[x,x,x])
