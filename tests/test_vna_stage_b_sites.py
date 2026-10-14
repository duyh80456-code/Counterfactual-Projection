"""Hook extensions: native equivalence, adjoints, FD and state preservation."""
import copy
import pytest
import torch
from torch import nn
from methods.vna import block_products, dedicated_generator, vna_block_correction, VNAConfig, _module_path
from methods.vna_sites import HookSite
from methods.vna_select import tiny_V, tiny_U_from_cotangent, normalize_A
from methods.vna_joint import joint_correction
from methods.vna_ggn import GGNConfig, GGNProducts
from experiments.vna_support import build_model, isolated_rng


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def assert_state(model, saved, modes, count, modules, rng):
    assert list(model.named_modules()) == modules
    assert count == sum(p.numel() for p in model.parameters())
    assert modes == [m.training for m in model.modules()]
    assert all(torch.equal(v, saved[n]) for n, v in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('kind', ['linear', 'conv', 'wrn'])
@pytest.mark.parametrize('bn_mode', ['eval', 'train'])
def test_generic_extension_finite_difference_adjoint_and_native_alignment(kind, bn_mode):
    gen = dedicated_generator(20, kind)
    with isolated_rng(gen):
        if kind == 'linear':
            model = nn.Sequential(nn.Linear(7, 5), nn.ReLU(), nn.Linear(5, 3)).double()
            inputs = torch.randn(3, 7, generator=gen, dtype=torch.float64)
            site = HookSite(model, '0', '2', 'linear', between=('1',))
        else:
            # WRN pre-activation: BN-ReLU-conv1-BN-ReLU-conv2.
            model = nn.Sequential(nn.BatchNorm2d(3), nn.ReLU(), nn.Conv2d(3, 4, 3, padding=1),
                                  nn.BatchNorm2d(4), nn.ReLU(), nn.Conv2d(4, 3, 3, padding=1),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten()).double()
            if kind == 'conv':
                model[0] = nn.Identity()
            inputs = torch.randn(3, 3, 5, 5, generator=gen, dtype=torch.float64)
            site = HookSite(model, '2', '5', 'conv', between=('3', '4'))
    model.train()
    model[-1].eval()  # preserve mixed modes
    saved, modes = copy.deepcopy(model.state_dict()), [m.training for m in model.modules()]
    count, modules, rng = sum(p.numel() for p in model.parameters()), list(model.named_modules()), torch.get_rng_state().clone()
    A = {'weight': site.first_layer.layer.weight[:2].detach(),
         'bias': site.first_layer.layer.bias[:2].detach()}
    oracle = copy.deepcopy(model)
    oracle.eval()
    if bn_mode == 'train':
        for m in oracle.modules():
            if isinstance(m, nn.BatchNorm2d): m.train()
    native = oracle(inputs).flatten().detach()
    with block_products(model, site, A, inputs, bn_mode) as p:
        assert count == sum(v.numel() for v in model.parameters())
        torch.testing.assert_close(p._call({}), native, rtol=0, atol=0)
        w = torch.randn(sum(v.numel() for _, v in p.extension_refs), generator=gen, dtype=torch.float64)
        v = torch.randn(sum(v.numel() for _, v in p.parameter_refs), generator=gen, dtype=torch.float64)
        u = torch.randn(native.numel(), generator=gen, dtype=torch.float64)
        for refs, tangent, action, transpose in ((p.extension_refs, w, p.JEv, p.JET), (p.parameter_refs, v, p.Jv, p.JT)):
            torch.testing.assert_close(u @ action(tangent), tangent @ transpose(u), rtol=1e-10, atol=1e-10)
            pieces = tangent.split([r.numel() for _, r in refs])
            eps = 1e-5
            plus = p._call({n: r + eps*t.reshape_as(r) for (n, r), t in zip(refs, pieces)})
            minus = p._call({n: r - eps*t.reshape_as(r) for (n, r), t in zip(refs, pieces)})
            torch.testing.assert_close(action(tangent), (plus-minus)/(2*eps), rtol=1e-5, atol=1e-7)
        op = GGNProducts(p, torch.tensor([0, 1, 2]))
        z = op.g.reshape(3, -1)
    assert_state(model, saved, modes, count, modules, rng)
    # Every required selection boundary accepts the generic site.
    normalized = normalize_A(model, site, A)
    assert normalized['weight'].shape == A['weight'].shape
    selected, capture = tiny_V(model, site, [(inputs, torch.tensor([0, 1, 2]))], 2,
                               cotangents=[z], score_signs=False)
    assert selected['weight'].shape[0] <= 2 and torch.isfinite(capture['singular_values']).all()
    incoming, eigen = tiny_U_from_cotangent(model, site, inputs, z, 2)
    assert incoming['weight'].shape[0] <= 2 and torch.isfinite(eigen).all()
    assert_state(model, saved, modes, count, modules, rng)
    with pytest.raises(RuntimeError, match='injected'):
        with block_products(model, site, A, inputs) as p:
            p._call({})
            raise RuntimeError('injected')
    assert_state(model, saved, modes, count, modules, rng)


@pytest.mark.parametrize('index', range(8))
def test_phase0_resnet18_each_block_hook_equivalence(index):
    gen = dedicated_generator(29, 'phase0-hook')
    with isolated_rng(gen):
        model = build_model('cpu', fake=False).double()
    ref = model.growing_blocks()[index]
    block = model.block(ref.name)
    path = _module_path(model, block)
    site = HookSite(model, path+'.first_layer.layer', path+'.second_layer.layer', 'conv',
                    between=(path+'.first_layer.post_layer_function',))
    inputs = torch.randn(2, 3, 8, 8, generator=gen, dtype=torch.float64)
    batch = (inputs, torch.tensor([0, 1]))
    A = {'weight': block.first_layer.layer.weight[:1].detach().clone(), 'bias': None}
    saved, modes = copy.deepcopy(model.state_dict()), [m.training for m in model.modules()]
    count, modules, rng = sum(p.numel() for p in model.parameters()), list(model.named_modules()), torch.get_rng_state().clone()
    results = []
    for resolver in (ref.name, site):
        with block_products(model, resolver, A, inputs, cache_block_input=True) as p:
            logits = p._call({})
            vector = torch.randn(logits.numel(), generator=dedicated_generator(3, 'shared'), dtype=torch.float64)
            results.append((logits, p.K(vector), p.KE(vector)))
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, rtol=1e-8, atol=1e-12)
    cfg = VNAConfig(cg_iters=4, cg_tol=1e-10, hutchinson_probes=1, cache_block_input=True)
    corrections = [vna_block_correction(model, resolver, A, batch, cfg, apply_gain_guard=False,
                                       cg_breakdown_flags=True, rhs_normalized_cg=True) for resolver in (ref.name, site)]
    for name in corrections[0][0]:
        torch.testing.assert_close(corrections[0][0][name], corrections[1][0][name], rtol=1e-8, atol=1e-12)
    for key in ('V', 'rho'):
        assert corrections[0][1][key] == pytest.approx(corrections[1][1][key], rel=1e-8, abs=1e-14)
    assert_state(model, saved, modes, count, modules, rng)


def test_hook_joint_single_and_multiple_sites():
    gen = dedicated_generator(14, 'joint-hook')
    with isolated_rng(gen):
        model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3)).double()
    sites = [HookSite(model, '0', '2', 'linear', between=('1',)), HookSite(model, '2', '4', 'linear', between=('3',))]
    A = {s: {'weight': s.first_layer.layer.weight[:2].detach(), 'bias': s.first_layer.layer.bias[:2].detach()} for s in sites}
    batch = (torch.randn(4, 4, generator=gen, dtype=torch.float64), torch.tensor([0, 1, 2, 0]))
    cfg = GGNConfig(cg_iters=100, cg_tol=1e-10, damping_norm='outputs')
    from methods.vna_joint import joint_products
    with joint_products(model, sites, A, batch[0]) as joint:
        vector = torch.randn(12, generator=gen, dtype=torch.float64)
        image = joint.KE(vector)
    images = []
    for s in sites:
        with block_products(model, s, A[s], batch[0]) as p:
            images.append(p.KE(vector))
    torch.testing.assert_close(image, sum(images), rtol=1e-12, atol=1e-12)
    d, V, rho, record = joint_correction(model, sites, A, batch, cfg, mu=.1)
    assert d and V >= 0 and rho >= 0 and record['cg_converged']


@pytest.mark.parametrize('widths', [(20, 5, 1), (784, 2000, 2000, 2000, 10)])
def test_lop_fix_ltu_net_and_deep_ffnn_shapes_zero_B(widths):
    # LoP regression (5 hidden units), online MNIST (3 x 2000 hidden units).
    gen = dedicated_generator(15, 'lop-shapes')
    with isolated_rng(gen):
        layers = []
        for i, (n, m) in enumerate(zip(widths, widths[1:])):
            if i: layers.append(nn.ReLU())
            layers.append(nn.Linear(n, m))
        model = nn.Sequential(*layers).double().eval()
    x = torch.randn(1, widths[0], generator=gen, dtype=torch.float64)
    expected = model(x).detach().flatten()
    for i in range(0, len(layers)-2, 2):
        site = HookSite(model, str(i), str(i+2), 'linear', between=(str(i+1),))
        A = {'weight': model[i].weight[:1].detach(), 'bias': model[i].bias[:1].detach()}
        with block_products(model, site, A, x) as p:
            torch.testing.assert_close(p._call({}), expected, rtol=0, atol=0)
            w = torch.randn(sum(t.numel() for _, t in p.extension_refs), generator=gen, dtype=torch.float64)
            u = torch.randn(len(expected), generator=gen, dtype=torch.float64)
            torch.testing.assert_close(u @ p.JEv(w), w @ p.JET(u), rtol=1e-10, atol=1e-10)
            eps = 1e-5
            pieces = w.split([t.numel() for _, t in p.extension_refs])
            plus = p._call({n: t+eps*v.reshape_as(t) for (n, t), v in zip(p.extension_refs, pieces)})
            minus = p._call({n: t-eps*v.reshape_as(t) for (n, t), v in zip(p.extension_refs, pieces)})
            torch.testing.assert_close(p.JEv(w), (plus-minus)/(2*eps), rtol=1e-5, atol=1e-7)
