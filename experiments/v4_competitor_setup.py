"""Shared v4 class order, SkipInit conversion and task evaluation."""
from contextlib import contextmanager
from pathlib import Path
import pickle
import numpy as np
import torch
from torch import nn
from experiments.vna_resume import rng_state, restore_rng

# CIFAR-100's official fine labels in each coarse class (alphabetical order).
SUPERCLASSES = (
    (4,30,55,72,95), (1,32,67,73,91), (54,62,70,82,92), (9,10,16,28,61),
    (0,51,53,57,83), (22,39,40,86,87), (5,20,25,84,94), (6,7,14,18,24),
    (3,42,43,88,97), (12,17,37,68,76), (23,33,49,60,71), (15,19,21,31,38),
    (34,63,64,66,75), (26,45,77,79,99), (2,11,35,46,98), (27,29,44,78,93),
    (36,50,65,74,80), (47,52,56,59,96), (8,13,48,58,90), (41,69,81,85,89))

COARSE_NAMES = ('aquatic_mammals','fish','flowers','food_containers','fruit_and_vegetables',
    'household_electrical_devices','household_furniture','insects','large_carnivores',
    'large_man-made_outdoor_things','large_natural_outdoor_scenes',
    'large_omnivores_and_herbivores','medium_mammals','non-insect_invertebrates',
    'people','reptiles','small_mammals','trees','vehicles_1','vehicles_2')


def class_order(seed, order, data_dir=None):
    if order == 'random': return np.random.RandomState(seed).permutation(100).tolist()
    groups = SUPERCLASSES
    if data_dir:
        path = Path(data_dir)/'cifar-100-python/train'
        with path.open('rb') as stream: data = pickle.load(stream, encoding='latin1')
        if 'coarse_labels' in data:
            fine, coarse = np.array(data['fine_labels']), np.array(data['coarse_labels'])
            groups = tuple(tuple(np.unique(fine[coarse == i])) for i in range(20))
            if groups != SUPERCLASSES: raise ValueError('CIFAR fine/coarse mapping mismatch')
        with (path.parent/'meta').open('rb') as stream: meta = pickle.load(stream, encoding='latin1')
        if 'coarse_label_names' in meta:
            names = tuple(n.decode() if isinstance(n,bytes) else n for n in meta['coarse_label_names'])
            if names != COARSE_NAMES: raise ValueError('CIFAR coarse label names mismatch')
    rng = np.random.RandomState(seed)
    return [int(c) for i in rng.permutation(20) for c in rng.permutation(groups[i])]


class SkipScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(()))
    def forward(self, x): return self.alpha*x


def skipinit(net):
    for name, module in list(net.named_children()):
        if isinstance(module, nn.Conv2d) and module.bias is None:
            module.bias = nn.Parameter(torch.zeros(module.out_channels, device=module.weight.device,
                                                   dtype=module.weight.dtype))
        elif isinstance(module, nn.BatchNorm2d):
            setattr(net, name, SkipScale().to(module.weight) if name == 'bn_b' else nn.Identity())
        else: skipinit(module)
    return net


@contextmanager
def read_only(model):
    rng = rng_state()
    modes = {m: m.training for m in model.modules()}
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    try:
        model.eval()
        yield
    finally:
        with torch.no_grad():
            for n, b in model.named_buffers(): b.copy_(buffers[n])
        for m, mode in modes.items(): m.training = mode
        restore_rng(rng)


def task_metrics(learner, manager, history, task):
    with read_only(learner._network):
        cnn, truth = learner._eval_cnn(learner.test_loader)
        nme = None
        if hasattr(learner, '_class_means') and learner._class_means is not None:
            nme, _ = learner._eval_nme(learner.test_loader, learner._class_means)
    row = dict(task=task, parameter_count=sum(p.numel() for p in learner._network.parameters()))
    for label, prediction in (('cnn', cnn), ('nme', nme)):
        if prediction is None: continue
        correct = prediction[:, 0] == truth
        per_class = [float(correct[truth == i].mean()*100) for i in range(learner._total_classes)]
        old = truth < learner._known_classes
        row.update({label+'_top1': float(correct.mean()*100), label+'_per_class': per_class,
            label+'_old_accuracy': float(correct[old].mean()*100) if old.any() else None,
            label+'_new_accuracy': float(correct[~old].mean()*100)})
        drops = [max([r[label+'_per_class'][i] for r in history if i < len(r.get(label+'_per_class', []))]
                     + [per_class[i]])-per_class[i] for i in range(learner._known_classes)]
        row[label+'_forgetting'] = float(np.mean(drops)) if drops else 0.
        row[label+'_average_accuracy'] = float(np.mean([r[label+'_top1'] for r in history]+[row[label+'_top1']]))
    row['learning_accuracy'] = row['cnn_new_accuracy']
    if task in (5, 10, 15, 20): row['feature_accuracy'] = feature_accuracy(learner, manager)
    return row


def feature_accuracy(learner, manager):
    from torch.utils.data import DataLoader
    net = learner._network
    with read_only(net), torch.enable_grad():
        vectors = []
        for source in ('train', 'test'):
            ds = manager.get_dataset(np.arange(learner._total_classes), source=source, mode='test')
            loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
            with torch.no_grad(): x, y = learner._extract_vectors(loader)
            vectors.append((torch.tensor(x, device=learner._device), torch.tensor(y, device=learner._device)))
        x, y = vectors[0]
        head = nn.Linear(x.shape[1], learner._total_classes).to(x)
        optimizer = torch.optim.LBFGS(head.parameters(), max_iter=100, line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(head(x), y)
            loss.backward()
            return loss
        optimizer.step(closure)
        with torch.no_grad():
            return float((head(vectors[1][0]).argmax(1) == vectors[1][1]).float().mean()*100)


class FixtureBackbone(nn.Module):
    """Two-channel residual fixture; the production backbone is never reduced."""
    def __init__(self, block):
        super().__init__()
        self.conv = nn.Conv2d(3,2,3,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.branch = block(2,2)
        self.out_dim = 2
    def forward(self,x):
        x = self.branch(torch.relu(self.bn(self.conv(x))))
        return dict(features=x.mean((2,3)),fmaps=[x,x,x])
