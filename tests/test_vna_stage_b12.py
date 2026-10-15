"""Tiny CPU tests for Kaggle only; TASK_STAGE_B_REST forbids local pytest."""
import copy
import hashlib
import numpy as np
import pytest
import torch
from adapters.repdistiller_hooks import upstream, sites_for, loss_context
from diagnostics.vna_align import _assert_bitwise, _global_state
from experiments.stage_b12 import parser, run
from methods.vna_joint import joint_products
from scripts.fetch_teachers import verify


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def check_site(model, site, x):
    saved = copy.deepcopy(model.state_dict())
    modes = [m.training for m in model.modules()]
    A = {'weight': site.first_layer.layer.weight[:1].detach().clone(), 'bias': None}
    with _global_state():
        model.eval()
        expected = model(x).detach().flatten()
        with joint_products(model, [site], {site: A}, x) as p:
            assert torch.equal(p._call({}), expected)
            w = torch.randn(sum(t.numel() for _, t in p.extension_refs), dtype=x.dtype)
            u = torch.randn(expected.numel(), dtype=x.dtype)
            torch.testing.assert_close(u @ p.JEv(w), w @ p.JET(u), rtol=1e-8, atol=1e-10)
            pieces = w.split([t.numel() for _, t in p.extension_refs])
            def perturbed(sign, eps):
                return p._call({n: t+sign*eps*v.reshape_as(t) for (n, t), v in zip(p.extension_refs, pieces)})
            # A fixed 1e-5 crosses downstream ReLU kinks; shrink by branch masks, never by derivative agreement.
            from torch.utils._python_dispatch import TorchDispatchMode
            class ReLUMasks(TorchDispatchMode):
                def __init__(self):
                    super().__init__()
                    self.masks = []
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    out = func(*args, **(kwargs or {}))
                    if func in (torch.ops.aten.relu.default, torch.ops.aten.relu_.default):
                        self.masks.append((out.detach() > 0).clone())
                    return out
            # Capture functional ReLUs as well as modules (RepDistiller residual outputs use F.relu).
            with ReLUMasks() as capture:
                p._call({})
            base_masks = capture.masks
            assert base_masks
            eps = 1e-5
            for _ in range(12):
                with ReLUMasks() as capture_plus:
                    plus = perturbed(1, eps)
                with ReLUMasks() as capture_minus:
                    minus = perturbed(-1, eps)
                assert len(base_masks) == len(capture_plus.masks) == len(capture_minus.masks)
                if all(torch.equal(a, b) and torch.equal(a, c)
                       for a, b, c in zip(base_masks, capture_plus.masks, capture_minus.masks)):
                    break
                eps /= 2
            else:
                pytest.fail('could not find a smooth finite-difference neighborhood')
            torch.testing.assert_close(p.JEv(w), (plus-minus)/(2*eps), rtol=1e-5, atol=1e-7)
        for m, mode in zip(model.modules(), modes): m.training = mode
    _assert_bitwise(saved, model.state_dict())
    assert modes == [m.training for m in model.modules()]


@pytest.mark.parametrize('name', ['resnet8x4', 'resnet32x4', 'wrn_16_2', 'wrn_40_2', 'resnet32'])
def test_architecture_counts(name):
    with upstream() as models:
        from models.resnet import resnet8x4, resnet32x4, resnet32
        from models.wrn import wrn_16_2, wrn_40_2
        direct = dict(resnet8x4=resnet8x4, resnet32x4=resnet32x4, resnet32=resnet32,
                      wrn_16_2=wrn_16_2, wrn_40_2=wrn_40_2)
        actual = sum(p.numel() for p in models[name](num_classes=100).parameters())
        assert actual == sum(p.numel() for p in direct[name](num_classes=100).parameters())


@pytest.mark.parametrize('seed', [0, 1])
@pytest.mark.parametrize('name', ['resnet8x4', 'wrn_16_2', 'resnet32'])
def test_sites_zero_b_order_adjoint_and_buffers(name, seed):
    torch.manual_seed(seed)
    with upstream() as models:
        model = models[name](num_classes=100).double()
        sites = sites_for(model)
        assert len(sites) == {'resnet8x4': 3, 'wrn_16_2': 6, 'resnet32': 15}[name]
        for site in sites:
            prefix = site.first.rsplit('.', 1)[0]
            assert site.between == tuple(prefix+'.'+s for s in (('bn2', 'relu2') if name.startswith('wrn') else ('bn1', 'relu')))
        # All sites preserve output/BN; adjoint/FD is also checked for each stage.
        for site in sites[::max(1, len(sites)//3)]:
            check_site(model, site, torch.randn(1, 3, 32, 32, dtype=torch.float64))
        model.eval()
        x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
        A = {s: {'weight': s.first_layer.layer.weight[:1].detach(), 'bias': None} for s in sites}
        with joint_products(model, sites, A, x) as p:
            assert torch.equal(p._call({}), model(x).flatten())


@pytest.mark.parametrize('setting', ['b1', 'b2'])
def test_t_align_two_epochs(monkeypatch, tmp_path, setting):
    from adapters import repdistiller_hooks
    monkeypatch.setattr(repdistiller_hooks, 'POINTS', {1, 2})
    def measure(model, batch, spec, ctx, *, generator):
        model.eval()
        site = sites_for(model)[0]
        A = {'weight': site.first_layer.layer.weight[:1].detach(), 'bias': None}
        with joint_products(model, [site], {site: A}, batch[0][:1]) as products:
            assert torch.equal(products._call({}), model(batch[0][:1]).flatten())
        small = tuple(t[:2] for t in batch)
        actual, context = loss_context(ctx['teacher'], small)
        assert actual.name == ('CE' if setting == 'b1' else 'KDRepDistiller')
        residual = actual.residual(model(small[0]), context)
        assert torch.isfinite(residual).all()
        # Isolation must also restore modes, buffers, gradients and global RNG.
        torch.rand(1); np.random.rand()
        with torch.no_grad(): next(model.parameters()).add_(1)
        return []
    snapshots = []
    with _global_state():
        for enabled in (False, True):
            args = parser().parse_args([setting, '--student', 'resnet8x4', '--smoke', '--fake-data',
                                       '--device', 'cpu', '--workers', '0', '--out', str(tmp_path/str(enabled))])
            observed = []
            def observe(model, optimizer):
                observed.append(copy.deepcopy((model.state_dict(), optimizer.state_dict())))
            hook = run(args, measure_override=measure if enabled else False, observe=observe)
            assert len(hook.epochs) == 2
            snapshots.append(observed)
    _assert_bitwise(*snapshots)


def test_kd_residual_and_hessian_imported_loss():
    with upstream():
        from distiller_zoo.KD import DistillKL
        from methods.vna_losses import KDRepDistiller
        z = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn_like(z); targets = torch.tensor([0, 1])
        ctx = {'targets': targets, 'teacher_logits': teacher}
        ctx['loss_fn'] = lambda logits, c: .1*torch.nn.CrossEntropyLoss()(logits, c['targets']) + .9*DistillKL(4)(logits, c['teacher_logits'])
        spec = KDRepDistiller(alpha=.1, beta=.9, T=4)
        expected = torch.autograd.grad(ctx['loss_fn'](z, ctx), z, create_graph=True)[0]
        torch.testing.assert_close(spec.residual(z, ctx), expected, rtol=1e-12, atol=1e-12)
        v = torch.randn_like(z)
        hessian = torch.autograd.grad(expected, z, v)[0]
        torch.testing.assert_close(spec.metric_apply(z, v, ctx), hessian, rtol=1e-12, atol=1e-12)
        eps = 1e-5
        fd = (spec.residual(z+eps*v, ctx)-spec.residual(z-eps*v, ctx))/(2*eps)
        torch.testing.assert_close(spec.metric_apply(z, v, ctx), fd, rtol=1e-5, atol=1e-8)


def test_teacher_hash_fake_file(tmp_path):
    path = tmp_path/'teacher.pth'; path.write_bytes(b'verified teacher fixture')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify(path, len(path.read_bytes()), digest) == digest
    with pytest.raises(ValueError): verify(path, len(path.read_bytes())+1, digest)
    with pytest.raises(ValueError): verify(path, len(path.read_bytes()), '0'*64)
