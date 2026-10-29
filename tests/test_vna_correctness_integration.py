"""Regression tests for conservative corrections to the existing H1--H4 pipeline."""
import pytest
import torch
from methods.cl_projection import ProjectionOperator, ConstraintSolver, Uncertified
from methods.cl_replay import CorrectionConfig, CurrentTaskLoss
from methods.cl_h3 import TangentReconfiguration
from methods.vna_sites import HookSite
from torch import nn


def _operator():
    return ProjectionOperator(lambda x: x, lambda x: x, lambda x: x, 1.,
                              tolerance=1e-9, preconditioner_probes=4)


def test_already_feasible_metric_projection_skips_unneeded_inverse_solves(monkeypatch):
    op = _operator()
    candidate = torch.tensor([2., 1.], dtype=torch.float64)
    constraints = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    def forbidden(*args, **kwargs):
        raise AssertionError('an inverse solve is unnecessary for feasible d0')
    monkeypatch.setattr(op, 'solve', forbidden)
    result = ConstraintSolver(op).solve(candidate, constraints)
    torch.testing.assert_close(result.direction, candidate, rtol=0, atol=0)
    assert result.status == 'CERTIFIED'
    assert result.objective == 0 and result.stationarity == 0
    assert not bool((result.slack < 0).any())
    assert op.audit and op.audit['passed']


def test_early_feasible_projection_never_bypasses_operator_audit():
    op = ProjectionOperator(lambda x: x, lambda x: 2*x, lambda x: x, 1.)
    with pytest.raises(Uncertified, match='operator audit'):
        ConstraintSolver(op).solve(torch.tensor([1., 1.], dtype=torch.float64),
                                   torch.eye(2, dtype=torch.float64))


def test_zero_rhs_returns_exact_solution_without_jacobi(monkeypatch):
    op = _operator()
    def forbidden(*args, **kwargs):
        raise AssertionError('Jacobi should not be constructed for zero RHS')
    monkeypatch.setattr(op, '_jacobi', forbidden)
    result = op.solve(torch.zeros(3, dtype=torch.float64))
    assert torch.equal(result, torch.zeros_like(result))
    assert op.solves[-1]['status'] == 'CERTIFIED'
    assert op.solves[-1]['iterations'] == 0
    assert op.solves[-1]['residual'] == 0


def test_nonzero_infeasible_candidate_still_uses_original_kkt_projection():
    op = _operator()
    candidate = torch.tensor([-1., 1.], dtype=torch.float64)
    constraints = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
    result = ConstraintSolver(op).solve(candidate, constraints)
    torch.testing.assert_close(result.direction, torch.tensor([0., 1.], dtype=torch.float64),
                               atol=1e-9, rtol=0)
    assert len(op.solves) == 2


def test_h3_source_score_interval_contains_exact_dense_reference():
    A = torch.tensor([[3., .4], [.4, 2.]], dtype=torch.float64)
    g = torch.tensor([1., -.3], dtype=torch.float64)
    xhat = torch.tensor([.30, -.21], dtype=torch.float64)
    r = g - A @ xhat
    m = torch.linalg.eigvalsh(A).min().item()
    observed = 2.5 - .5 * (g @ xhat).item()
    bound = .5 * g.norm().item() * r.norm().item() / m
    exact = 2.5 - .5 * (g @ torch.linalg.solve(A, g)).item()
    assert observed - bound <= exact <= observed + bound


def test_h3_engine_exports_surrogate_intervals_for_a_real_swap():
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2)).double()
    with torch.no_grad():
        net[0].weight.fill_(1)
        net[0].bias.copy_(torch.tensor([-10., 1.], dtype=torch.float64))
    opt = torch.optim.SGD(net.parameters(), lr=.01, momentum=.9)
    x = torch.tensor([[0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
    losses = CurrentTaskLoss(torch.tensor([0, 1, 0]))
    result = TangentReconfiguration(CorrectionConfig(damping_relative=1.,
        preconditioner_probes=0), max_candidates=2).step(
        net, opt, x, losses, [HookSite(net, '0', '2', 'linear', between=('1',))])
    assert result['certified']
    records = [result['identity']] + [row for row in result['trials'] if 'net_surrogate_interval' in row]
    assert records
    for row in records:
        lo, hi = row['net_surrogate_interval']
        assert lo <= row['net_surrogate'] <= hi
        assert row['net_surrogate_error_bound'] >= 0
    if result['choice'] != 'identity':
        assert isinstance(result['surrogate_wins_identity_certified'], bool)


def test_large_finite_gradient_cannot_disappear_during_normalization():
    op = _operator()
    candidate = torch.tensor([-1., 1.], dtype=torch.float64)
    constraints = torch.tensor([[1e200, 0.], [0., 1.]], dtype=torch.float64)
    result = ConstraintSolver(op).solve(candidate, constraints)
    assert result.status == 'CERTIFIED'
    assert torch.all(constraints @ result.direction >= 0)
    assert result.direction[0].abs() < 1e-8
    assert result.objective > 0


def test_tiny_finite_gradient_cannot_disappear_during_normalization():
    op = _operator()
    candidate = torch.tensor([-1., 1.], dtype=torch.float64)
    constraints = torch.tensor([[1e-200, 0.], [0., 1.]], dtype=torch.float64)
    result = ConstraintSolver(op).solve(candidate, constraints)
    assert result.status == 'CERTIFIED'
    assert torch.all(constraints @ result.direction >= 0)
    assert result.direction[0].abs() < 1e-8


def test_cached_audit_does_not_skip_deadline_on_fast_paths():
    op = _operator()
    op._audit(torch.ones(2, dtype=torch.float64))
    def expired():
        raise TimeoutError('expired')
    op.boundary = expired
    with pytest.raises(TimeoutError, match='expired'):
        op.solve(torch.zeros(2, dtype=torch.float64))
    with pytest.raises(TimeoutError, match='expired'):
        ConstraintSolver(op).solve(torch.ones(2, dtype=torch.float64),
                                    torch.eye(2, dtype=torch.float64))


def test_boundary_feasible_candidate_uses_kkt_instead_of_roundoff_shortcut():
    op = _operator()
    candidate = torch.tensor([1., 1.], dtype=torch.float64)
    constraints = torch.tensor([[1., -1.]], dtype=torch.float64)
    result = ConstraintSolver(op).solve(candidate, constraints)
    assert result.status == 'CERTIFIED'
    assert len(op.solves) == 1
    torch.testing.assert_close(result.direction, candidate, atol=0, rtol=0)


def test_nonfinite_rhs_is_rejected_before_operator_evaluation(monkeypatch):
    op = _operator()
    def forbidden(*args):
        raise AssertionError('invalid RHS must not reach operator audit')
    monkeypatch.setattr(op, '_audit', forbidden)
    with pytest.raises(Uncertified, match='non-finite linear solve RHS'):
        op.solve(torch.tensor([float('nan')], dtype=torch.float64))
