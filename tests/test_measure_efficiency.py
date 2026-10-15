"""TASK_MEASURE_EFFICIENCY 5.1–5.4: numerical oracles and isolation."""
import importlib.util
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch import nn
from diagnostics.icarl_measure import B3Measurement
from diagnostics.measure_efficiency import SolveCache, virtual_energy
from diagnostics.stage_a import null_residual
from methods.vna import random_A_like
from methods.vna_ggn import GGNConfig
from methods.vna_joint import joint_correction
from methods.vna_losses import CE
from methods.vna_sites import HookSite


@pytest.fixture(autouse=True)
def cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    from diagnostics.vna_align import _global_state
    with _global_state():
        torch.manual_seed(45)
        yield
    torch.set_num_threads(old)


class SmallResNet(nn.Module):
    """CIFAR-shaped residual model with two independently virtualizable sites."""
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(16)
        self.stem = nn.Conv2d(3, 2, 1)
        self.a = nn.Conv2d(2, 2, 1)
        self.relu = nn.ReLU()
        self.b = nn.Conv2d(2, 2, 1)
        self.c = nn.Conv2d(2, 2, 1)
        self.relu2 = nn.ReLU()
        self.d = nn.Conv2d(2, 2, 1)
        self.head = nn.Linear(2, 3)
    def forward(self, x):
        x = self.stem(self.pool(x))
        x = x + self.b(self.relu(self.a(x)))
        x = x + self.d(self.relu2(self.c(x)))
        return self.head(x.mean((2, 3)))


def sites(model):
    return [HookSite(model, 'a', 'b', 'conv', between=('relu',)),
            HookSite(model, 'c', 'd', 'conv', between=('relu2',))]


def context(ctx, batch): return CE(), {'targets': batch[1]}


def legacy(tmp_path, path):
    """Load baseline into a temporary module, never touching the checkout."""
    file = tmp_path/PathName(path)
    file.write_bytes(subprocess.check_output(['git', 'show', f'688b61d:{path}']))
    spec = importlib.util.spec_from_file_location(file.stem, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def PathName(path): return path.replace('/', '_')


def explicit_nulls():
    # Test-only hook: same explicit residuals at every site/rank in old/new code.
    draws = {}
    counters = {}
    def draw(g, kind, generator, **kwargs):
        index = counters.get(kind, 0) % 16
        counters[kind] = index+1
        key = kind, index
        if key not in draws:
            draws[key] = null_residual(g, kind, torch.Generator().manual_seed(700+index), **kwargs)
        return draws[key].clone()
    return draw


def test_51_old_measurement_numerical_equivalence(tmp_path):
    old = legacy(tmp_path, 'diagnostics/icarl_measure.py')
    old_joint = legacy(tmp_path, 'methods/vna_joint.py')
    model = SmallResNet().double().eval()
    batch = torch.randn(32, 3, 32, 32, dtype=torch.float64), torch.arange(32)%3
    ctx = dict(task=2, epoch=1, step=1, end_of_task=False, train_acc=.5, known_classes=1)
    meta = dict(seed=45, setting='B3')
    new = B3Measurement(meta, sites_fn=sites, context_fn=context, ranks=(1,), groups=False)
    baseline = old.B3Measurement(meta, sites_fn=sites, context_fn=context, ranks=(1,), groups=False)
    baseline.mus = new.mus
    # Baseline's redundant joint call inferred damping on eval. Explicitly pin
    # the stat-derived mu to compare the identity under exactly the same mu.
    pinned = {}
    def correction(*args, **kwargs):
        mu_rel = args[4].damping_rel
        if kwargs.get('mu') is not None: pinned[mu_rel] = kwargs['mu']
        kwargs['mu'] = pinned[mu_rel]
        return old_joint.joint_correction(*args, **kwargs)
    def random(a, **kwargs): return random_A_like(a, generator=torch.Generator().manual_seed(19))
    with patch.object(old, 'null_residual', explicit_nulls()), patch.object(old, 'random_A_like', random), patch.object(old, 'joint_correction', correction):
        expected = baseline(model, batch, CE(), ctx, generator=torch.Generator().manual_seed(83))
    with patch('diagnostics.measure_efficiency.null_residual', explicit_nulls()), patch('diagnostics.icarl_measure.random_A_like', random):
        actual = new(model, batch, CE(), ctx, generator=torch.Generator().manual_seed(83))
    assert len(expected) == len(actual)
    for a, b in zip(actual, expected):
        for key in ('V', 'U', 'rho', 'eta', 'V_tiny_V', 'V_random_frob', 'V_tiny_U', 'rho_joint'):
            assert a[key] == pytest.approx(b[key], rel=1e-6, abs=1e-25), key
        for kind in ('N1', 'N2'):
            for key in (f'null_rho_{kind}', f'null_lambda_{kind}'):
                torch.testing.assert_close(torch.tensor(json.loads(a[key])), torch.tensor(json.loads(b[key])), rtol=1e-6, atol=1e-25)
        assert a['V_consistency'] <= 1e-6
        assert a['stat_m'] == a['eval_m'] == 16


def test_52_joint_sum_identity(tmp_path):
    old = legacy(tmp_path, 'methods/vna_joint.py')
    model = SmallResNet().double().eval()
    data = torch.randn(32, 3, 32, 32, dtype=torch.float64), torch.arange(32)%3
    scope = sites(model)
    cfg = GGNConfig(cg_iters=1000, cg_tol=1e-6, project_bn=False)
    cache = SolveCache(model, scope[0], data, CE(), {'targets': data[1]}, cfg)
    e = CE().residual(model(data[0]), {'targets': data[1]})
    z = cache.solve('real', e, data, {'targets': data[1]})
    A = {s: {'weight': s.first_layer.layer.weight[:1].detach(), 'bias': s.first_layer.layer.bias[:1].detach()} for s in scope}
    energies = [virtual_energy(model, [s], {s: A[s]}, data, z, e) for s in scope]
    reference = old.joint_correction(model, scope, A, data, cfg, mu=cache.mu)[3]
    assert sum(v['rho'] for v in energies) == pytest.approx(reference['rho'], rel=1e-6)
    assert sum(v['V'] for v in energies) == pytest.approx(reference['V'], rel=1e-6)


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 3))
    def predict(self, x):
        feature = self.layers[:2](x)
        return self.layers[2](feature), [feature]


@pytest.mark.parametrize('discrepancy', [0., 1e-3])
def test_53_b5_selection_only_reads_stat(monkeypatch, discrepancy):
    from adapters.lop_hooks import Prediction
    from diagnostics import lop_measure as mod
    model = Prediction(Native()).double().eval()
    x = torch.randn(256, 3, dtype=torch.float64)
    x[:, 0] = torch.arange(256) # sample identities survive slicing
    batch = x, torch.arange(256)%3
    generator = torch.Generator().manual_seed(9)
    order = torch.randperm(256, generator=torch.Generator().manual_seed(9))
    expected_stat = set(order[:128].tolist())
    counts = {'tiny_V': 0, 'tiny_U': 0, 'swaps': 0}
    original_v, original_u = mod.tiny_V, mod.tiny_U_from_cotangent
    def check_v(model, site, batches, *args, **kwargs):
        counts['tiny_V'] += 1
        for data in batches: assert set(data[0][:, 0].int().tolist()) <= expected_stat
        return original_v(model, site, batches, *args, **kwargs)
    def check_u(model, site, inputs, *args, **kwargs):
        counts['tiny_U'] += 1
        assert set(inputs[:, 0].int().tolist()) == expected_stat
        return original_u(model, site, inputs, *args, **kwargs)
    monkeypatch.setattr(mod, 'tiny_V', check_v)
    monkeypatch.setattr(mod, 'tiny_U_from_cotangent', check_u)
    original_correction = mod.joint_correction
    def correction(*args, **kwargs):
        result = original_correction(*args, **kwargs)
        result[3]['cg_converged'] = True
        if discrepancy:
            result[3]['V'] /= 1 + discrepancy
        return result
    monkeypatch.setattr(mod, 'joint_correction', correction)
    monkeypatch.setattr(mod.SolveCache, 'converged', lambda self, **kwargs: True)
    monkeypatch.setattr(mod, 'spectrum', lambda *a, **k: {})
    original_swap = mod.swap_directions
    def swap(*args):
        counts['swaps'] += 1
        return original_swap(*args)
    monkeypatch.setattr(mod, 'swap_directions', swap)
    measure = mod.B5Measurement('mnist', {'seed': 3}, ranks=[1, 2])
    shadow = SimpleNamespace(bounds=[.1])
    ctx = dict(task=1, step=256, point='first', within_task_step=256, buffer_task_purity=1.,
               buffer_m=256, loss_fn=lambda z,c: nn.functional.cross_entropy(z,c['targets']),
               shadow=shadow, cbp_candidates=[('cbp', shadow, [torch.tensor([0])])], init_row_norms=[1.])
    rows = measure(model, batch, CE(), ctx, generator=generator)
    assert len(rows) == 4 and counts['tiny_U'] == 4
    assert counts['swaps'] == 4 # CBP and ReDo, one site, two mu; independent of rank
    assert all(r['stat_m'] == r['eval_m'] == 128 for r in rows)
    assert rows[0]['null_draw_id'] == rows[1]['null_draw_id']
    if discrepancy:
        assert all(r['V_consistency'] == pytest.approx(1e-3) for r in rows)
        assert all(not r['cg_converged_real'] and not r['cg_converged'] for r in rows)
    else:
        assert all(r['cg_converged_real'] and r['cg_converged'] for r in rows)


def test_54_gate():
    from diagnostics.vna_gate import evaluate_gate, setting_config
    rows = [dict(task=2, epoch=i, step=i, end_of_task=False, rank=16, mu_rel=.01,
                 cg_converged=True, rho_joint=.4, significant_N1=False, significant_rho_N1=True,
                 V_tiny_V=2., V_random_frob=1., train_acc=1.) for i in range(10)]
    cfg = setting_config('B3')
    result = evaluate_gate(rows, cfg)
    assert result['status'] == 'PASS' and result['conditions']['train_acc'] == 'n/a'
    assert result['gate_null_statistic'] == 'rho'
    rows[0].update(cg_converged=False, status='timeout')
    assert evaluate_gate(rows, cfg)['excluded_fraction'] == .1
    assert evaluate_gate(rows, cfg)['status'] == 'PASS'
    rows[1]['cg_converged'] = False
    result = evaluate_gate(rows, cfg)
    assert result['status'] == 'INVALID' and result['excluded_points'] == 2
    # A duplicate site must neither dilute nor inflate point exclusions.
    assert evaluate_gate(rows+[dict(rows[0], site='second')], cfg)['excluded_fraction'] == .2


def test_runtime_timeout_heartbeat_solves_and_restoration(tmp_path):
    from diagnostics.vna_align import MeasurementHook
    from experiments.vna_resume import MeasurementMonitor
    from methods.vna_joint import _inverse
    from projection.cg import conjugate_gradient
    callbacks = []
    result = conjugate_gradient(lambda v: 2*v, torch.ones(3), callback=callbacks.append)
    assert callbacks == [0, 1] and result.converged
    with MeasurementMonitor({'setting': 'B1'}, {'epoch': 1}, out=tmp_path, traceback_s=0) as monitor:
        _inverse(lambda v: 2*v, torch.ones(3, dtype=torch.float64), GGNConfig())
    import csv
    rows = list(csv.DictReader((tmp_path/'solves.csv').open()))
    assert len(rows) == 1 and int(rows[0]['cg_iters_used']) == 1
    assert float(rows[0]['seconds']) >= 0
    heartbeat = json.loads((tmp_path/'heartbeat.json').read_text())
    assert heartbeat['cg_iteration'] == 1 and heartbeat['epoch'] == 1
    model = SmallResNet().double().train()
    measure = B3Measurement({'seed': 45, 'setting': 'B1'}, sites_fn=sites, context_fn=context,
                            ranks=(1,), groups=False)
    measure.runtime = dict(out=tmp_path, timeout_s=-1, traceback_s=0)
    before = {k: v.clone() for k,v in model.state_dict().items()}
    batch = torch.randn(4, 3, 32, 32, dtype=torch.float64), torch.arange(4)%3
    ctx = dict(task=0, epoch=1, step=1, end_of_task=False, train_acc=.2)
    timed = MeasurementHook(measure)(model, batch, CE(), ctx)
    assert len(timed) == 2 and all(r['status'] == 'timeout' for r in timed)
    assert model.training and all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_51_old_lop_arms_explicit_residuals(tmp_path):
    old = legacy(tmp_path, 'diagnostics/lop_measure.py')
    original_joint = legacy(tmp_path, 'methods/vna_joint.py')
    model = SmallResNet().double().eval()
    data = torch.randn(32, 3, 32, 32, dtype=torch.float64), torch.arange(32)%3
    site = sites(model)[0]
    cfg = GGNConfig(cg_iters=1000, cg_tol=1e-6, project_bn=False)
    cache = SolveCache(model, site, data, CE(), {'targets': data[1]}, cfg)
    e = CE().residual(model(data[0]), {'targets': data[1]})
    for index, residual in enumerate((e, null_residual(e, 'N1', torch.Generator().manual_seed(2)),
                                      null_residual(e, 'N2', torch.Generator().manual_seed(3)))):
        z = cache.solve(str(index), residual, data, {'targets': data[1]})
        A, cap = old.tiny_V(model, site, [data], 1, cotangents=[z], cfg=cfg)
        incoming, _ = old.tiny_U_from_cotangent(model, site, data[0], residual, 1, cfg)
        for arm in (A, incoming, random_A_like(A, generator=torch.Generator().manual_seed(4))):
            expected = original_joint.joint_correction(model, [site], {site: arm}, data, cfg,
                mu=cache.mu, cotangent=residual)[3]
            actual = virtual_energy(model, [site], {site: arm}, data, z, residual)
            for key in ('V', 'U', 'eta', 'rho'):
                assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-25)
        from methods.vna_select import tiny_V
        _, new_cap = tiny_V(model, site, [data], 1, cotangents=[z], cfg=cfg)
        torch.testing.assert_close(cap['generalized_eigenvalues'], new_cap['generalized_eigenvalues'], rtol=1e-6, atol=0)


def test_verify_energy_returns_discrepancy():
    from diagnostics.measure_efficiency import verify_energy
    assert verify_energy({'V': 1.001}, {'V': 1.}) == pytest.approx(1e-3)
    assert verify_energy({'V': 0.}, {'V': 0.}) == 0.
