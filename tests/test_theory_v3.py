"""Fast reference checks and production push-through regression oracles."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from methods.vna_losses import ICaRLPyCIL
from methods.vna_v3 import V3Engine


@pytest.fixture(autouse=True)
def numerical_state():
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    with torch.random.fork_rng():
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(1)
        yield
    torch.set_default_dtype(dtype)
    torch.set_num_threads(threads)


def reference(name):
    path = Path(__file__).resolve().parents[1]/'docs/vna'/name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('function', ['T0', 'T1', 'T6', 'T9', 'T11', 'T14', 'T17', 'T18'])
def test_v3_reference(function):
    module = reference('final_check_v3.py')
    getattr(module, function)()
    assert module.RES and all(ok for _, ok in module.RES), module.RES


def test_unified_reference_with_negative_controls():
    module = reference('check_unified_gn.py')
    module.math_checks()
    assert len(module.RES) == 15
    assert all(ok for _, ok in module.RES), module.RES


def test_push_through_production_saturation_and_no_range_requirement():
    generator = torch.Generator().manual_seed(2026)
    worst_old = 0.
    for scale in (1., 5., 10., 20., 30.):
        logits = scale*torch.randn(4, 5, generator=generator)
        spec = ICaRLPyCIL(known_classes=3)
        ctx = dict(targets=torch.arange(4), known_classes=3,
                   teacher_logits=torch.randn(4, 3, generator=generator))
        e = spec.residual(logits, ctx).flatten()
        J = torch.randn(20, 13, generator=generator)
        JE = torch.randn(20, 3, generator=generator)
        products = SimpleNamespace(K=lambda r: J @ (J.T @ r.flatten()),
                                   JT=lambda r: J.T @ r.flatten(), JEv=lambda w: JE @ w)
        op = SimpleNamespace(logits=logits, loss_spec=spec, ctx=ctx, p=products)
        eye = torch.eye(20)
        H = torch.stack([spec.metric_apply(logits, v.reshape_as(logits), ctx).flatten()
                         for v in eye], 1)
        vals, Q = torch.linalg.eigh(H)
        R = (Q*vals.clamp_min(0).sqrt()) @ Q.T
        mu = max(.01*float(torch.trace(J.T @ H @ J))/20, 1e-12)
        A = J.T @ H @ J + mu*torch.eye(13)
        # Dense parameter solves are independent of the output factorization.
        expected = H @ J @ torch.linalg.solve(A, J.T @ e)
        op.F = lambda v: J.T @ H @ J @ v
        engine = V3Engine(); engine.tol = 1e-12
        solve = engine._solve(op, mu, lambda v: torch.zeros_like(v), [], warm={})
        z = H @ J @ solve('z', J.T @ e)
        torch.testing.assert_close(z, expected, rtol=1e-8, atol=1e-11)
        outside = e + torch.ones_like(e)
        actual = H @ J @ solve('outside', J.T @ outside)
        oracle = H @ J @ torch.linalg.solve(A, J.T @ outside)
        torch.testing.assert_close(actual, oracle, rtol=1e-8, atol=1e-11)
        assert torch.isfinite(actual).all()
        w = JE.T @ z
        stacked = torch.cat((R @ J, mu**.5*torch.eye(13)))
        target = torch.cat((R @ JE @ w, torch.zeros(13)))
        expected_d = torch.linalg.lstsq(stacked, target, driver='gelsd').solution
        torch.testing.assert_close(solve('d', J.T @ H @ JE @ w), expected_d, rtol=1e-8, atol=1e-10)
        vals, Q = torch.linalg.eigh(H)
        keep = vals > vals.max()*1e-10
        pinv_R = (Q[:, keep]*vals[keep].rsqrt()) @ Q[:, keep].T
        M = R @ J @ J.T @ R
        old_z = R @ M @ torch.linalg.solve(M + mu*eye, pinv_R @ e)
        if scale == 20:
            worst_old = max(worst_old, float((old_z-expected).norm()/expected.norm()))
    assert worst_old > .01


def test_schur_deflation_with_stale_complement_negative_control():
    module = reference('final_check_v3.py')
    J = torch.tensor([[1.], [0.]])
    H = torch.diag(torch.tensor([2., 1.]))
    r = torch.ones(2); mu = .1
    G = torch.linalg.inv(J.T@H@J+mu*torch.eye(1))
    Hp = H-H@J@G@J.T@H
    rp = r-H@J@G@J.T@r
    first = torch.ones(2, 1)
    S = first.T@Hp@first+mu*torch.eye(1)
    updated = rp-Hp@first@torch.linalg.solve(S, first.T@rp)
    J1 = torch.cat((J, first), 1)
    G1 = torch.linalg.inv(J1.T@H@J1+mu*torch.eye(2))
    expected = r-H@J1@G1@J1.T@r
    torch.testing.assert_close(updated, expected, rtol=1e-12, atol=1e-12)
    second = torch.tensor([[0.], [1.]])
    Hp1 = H-H@J1@G1@J1.T@H
    S2 = second.T@Hp1@second+mu*torch.eye(1)
    gain = module.qstar(J1,H,r,mu)-module.qstar(torch.cat((J1,second),1),H,r,mu)
    correct = .5*float((second.T@updated)@torch.linalg.solve(S2,second.T@updated))
    stale = .5*float((second.T@rp)@torch.linalg.solve(S2,second.T@rp))
    assert abs(correct-gain)/gain < 1e-10
    assert abs(stale-gain)/gain > .01
