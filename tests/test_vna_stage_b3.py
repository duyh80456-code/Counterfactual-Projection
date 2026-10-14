"""Only the three tiny CPU checks requested by TASK_STAGE_B3 §4."""
import copy
from contextlib import nullcontext
import numpy as np
import torch
from torchvision import transforms
from experiments.stage_b3 import upstream_imports, recipe
from adapters.icarl_hooks import ICaRLHook, Prediction, sites_for, loss_context
from diagnostics.vna_align import _assert_bitwise, _global_state
from methods.vna_joint import joint_products
from methods.vna_losses import ICaRLPyCIL


def fake_manager(monkeypatch, *, classes=4, per_class=1000, increment=2):
    from utils import data_manager as dm
    class FakeData:
        use_path = False
        train_trsf = test_trsf = []
        common_trsf = [transforms.ToTensor()]
        class_order = list(range(classes))
        def download_data(self):
            rng = np.random.RandomState(19)
            self.train_data = rng.randint(256, size=(classes*per_class, 32, 32, 3), dtype=np.uint8)
            self.train_targets = np.repeat(np.arange(classes), per_class)
            self.test_data = self.train_data[::per_class].copy()
            self.test_targets = np.arange(classes)
    monkeypatch.setattr(dm, '_get_idata', lambda name: FakeData())
    return dm.DataManager('cifar100', True, 1993, increment, increment)


def tiny_run(monkeypatch, enabled):
    from models import icarl, base
    dm = fake_manager(monkeypatch)
    # Keep training to batch 8 and two classes/task, while supplying the full
    # 1000/class herding pool to test the actual 2000/seen-class memory rule.
    original_dataset = dm.get_dataset
    def dataset(*args, **kwargs):
        result = original_dataset(*args, **kwargs)
        if kwargs.get('mode') == 'train':
            chosen = np.concatenate([np.flatnonzero(result.labels == k)[:4] for k in np.unique(result.labels)])
            result.images, result.labels = result.images[chosen], result.labels[chosen]
        return result
    dm.get_dataset = dataset
    for key, value in {'init_epoch': 1, 'epochs': 1, 'batch_size': 8, 'num_workers': 0}.items():
        monkeypatch.setattr(icarl, key, value)
    original_loader = base.DataLoader
    monkeypatch.setattr(base, 'DataLoader', lambda *a, **kw: original_loader(*a, **{**kw, 'num_workers': 0}))
    # Herding itself is unmodified. Deterministic cheap feature vectors avoid
    # thousands of ResNet inference calls in this deliberately tiny test.
    def vectors(learner, loader):
        images = loader.dataset.images
        values = images[:, 0, 0, :].astype(np.float64)/255
        features = np.tile(np.concatenate([values, np.ones((len(values), 1))], 1), (1, 16))
        return features, loader.dataset.labels
    monkeypatch.setattr(icarl.iCaRL, '_extract_vectors', vectors)
    params = recipe(); params.update(init_cls=2, increment=2, device=[torch.device('cpu')])
    learner = icarl.iCaRL(params); learner.topk = 2
    optimizers, memory, snapshots = [], [], []
    optimizer_objects = []
    original_sgd = icarl.optim.SGD
    def sgd(*a, **kw):
        optimizer = original_sgd(*a, **kw); optimizer_objects.append(optimizer)
        return optimizer
    monkeypatch.setattr(icarl.optim, 'SGD', sgd)
    def measure(model, batch, spec, ctx, *, generator):
        assert len(batch[0]) <= 128
        assert (spec.known_classes == 2) if ctx['task'] == 2 else spec.name == 'CE'
        assert len(sites_for(model)) == 15
        model.eval()
        site = sites_for(model)[0]
        A = {'weight': site.first_layer.layer.weight[:1].detach().clone(), 'bias': None}
        with joint_products(model, [site], {site: A}, batch[0][:1]) as products:
            assert torch.equal(products._call({}), model(batch[0][:1]).flatten())
        # Exercise the isolation boundary: modes, BN buffers, gradients and RNG.
        torch.rand(1); np.random.rand()
        with torch.no_grad(): next(model.parameters()).add_(1)
        return []
    hook = ICaRLHook(measure)
    with hook.install() if enabled else nullcontext():
        for _ in range(2):
            learner.incremental_train(dm)
            optimizers.append(copy.deepcopy(optimizer_objects[-1].state_dict()))
            memory.append((learner.samples_per_class, np.bincount(learner._targets_memory.astype(int)), len(learner._data_memory)))
            learner.eval_task()
            learner.after_task()
            snapshots.append(copy.deepcopy(learner._old_network.state_dict()))
    if enabled: assert len(hook.epochs) == len(hook.tasks) == len(hook.point_seconds) == 2
    return {'model': learner._network.state_dict(), 'optimizer': optimizers,
            'old_network': snapshots, 'memory': memory}, dm


def test_t_align_and_exemplar_memory(monkeypatch):
    upstream_imports()
    previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with _global_state():
            runs = []
            for enabled in (False, True):
                torch.manual_seed(1); np.random.seed(1)
                with monkeypatch.context() as local:
                    runs.append(tiny_run(local, enabled)[0])
            _assert_bitwise(*runs)
            for task, (per_class, counts, size) in enumerate(runs[1]['memory'], 1):
                assert per_class == 2000//(2*task)
                assert size == 2000 and np.all(counts == per_class)
    finally: torch.set_num_threads(previous_threads)


def test_spec12_class_order_and_upstream_loss(monkeypatch):
    upstream_imports()
    from models.icarl import _KD_loss
    dm = fake_manager(monkeypatch, classes=100, per_class=1, increment=10)
    assert dm._class_order == np.random.RandomState(1993).permutation(100).tolist()
    z = torch.randn(8, 4, dtype=torch.float64)
    teacher = torch.randn(8, 2, dtype=torch.float64)
    targets = torch.arange(8)%4
    ctx = {'targets': targets, 'teacher_logits': teacher}
    spec = ICaRLPyCIL(T=2, known_classes=2)
    expected = torch.nn.functional.cross_entropy(z, targets) + _KD_loss(z[:, :2], teacher, 2)
    assert torch.equal(spec.loss(z, ctx), expected)
    # The adapter's authoritative callable must use that imported loss too.
    class Teacher(torch.nn.Module):
        def forward(self, x): return {'logits': teacher}
    class Learner:
        _known_classes = 2
        _old_network = Teacher()
    actual_spec, actual_ctx = loss_context(Learner(), (torch.zeros(8, 3, 32, 32), targets))
    assert torch.equal(actual_ctx['loss_fn'](z, actual_ctx), expected)
    assert torch.equal(actual_spec.loss(z, actual_ctx), expected)


def test_resnet32_zero_b_and_parameter_count():
    upstream_imports()
    from models.icarl import iCaRL
    previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        params = recipe(); params['device'] = [torch.device('cpu')]
        model = iCaRL(params)._network; model.update_fc(4)
        model = Prediction(model).eval()
        batch = torch.zeros(1, 3, 32, 32)
        before = copy.deepcopy(model.state_dict())
        count = sum(p.numel() for p in model.parameters())
        sites = sites_for(model); assert len(sites) == 15
        proposals = {site: {'weight': site.first_layer.layer.weight[:4].detach().clone(), 'bias': None} for site in sites}
        with torch.no_grad(): expected = model(batch)
        with joint_products(model, sites, proposals, batch) as products:
            assert torch.equal(products._call({}), expected.flatten())
            assert sum(p.numel() for p in model.parameters()) == count
        _assert_bitwise(before, model.state_dict())
        assert sum(p.numel() for p in model.parameters()) == count
    finally: torch.set_num_threads(previous_threads)
