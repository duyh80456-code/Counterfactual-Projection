"""Numerical acceptance checks for H1/H2/H4 shared correction solver."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from methods.cl_projection import ProjectionOperator, Uncertified
from methods.cl_replay import CandidateUpdate, CorrectionConfig, LossComponent, ReplayCorrection
from methods.cl_state import State, vector
from diagnostics.vna_align import _assert_bitwise


def test_jacobi_recovers_ill_scaled_spd_without_global_rng_consumption():
    J = torch.diag(torch.tensor([1e-3, 1., 1e3], dtype=torch.double))
    g = torch.tensor([1., -2., 3.], dtype=torch.double)
    def operator(probes):
        return ProjectionOperator(lambda x: J @ x, lambda x: J.T @ x,
                                  lambda x: x, 1e-6, tolerance=1e-9,
                                  max_iterations=2, preconditioner_probes=probes)
    with pytest.raises(Uncertified):
        operator(0).solve(g)
    before = torch.get_rng_state().clone()
    A = operator(4)
    x = A.solve(g)
    assert A.solves[-1]['status'] == 'CERTIFIED'
    assert A.solves[-1]['relative_residual'] <= 1e-9
    torch.testing.assert_close(x, torch.linalg.solve(J.T @ J + 1e-6 * torch.eye(3), g), rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(torch.get_rng_state(), before)


def test_randomized_jacobi_for_large_diagonal_operator_is_deterministic():
    values=torch.logspace(-2, 2, 24, dtype=torch.double)
    rhs=torch.linspace(.2, 2., 24, dtype=torch.double)
    A=ProjectionOperator(lambda x:values*x,lambda x:values*x,lambda x:x,
                         1e-5,tolerance=1e-9,max_iterations=5,
                         preconditioner_probes=3,preconditioner_seed=7)
    before=torch.get_rng_state().clone()
    result=A.solve(rhs)
    torch.testing.assert_close(result,rhs/(values.square()+1e-5),rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(torch.get_rng_state(),before)
    assert A.solves[-1]['preconditioner_probes']==3


def toy():
    net = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():net.weight.zero_()
    opt = torch.optim.SGD(net.parameters(), lr=.1, momentum=.9)
    x = torch.ones(4, 1, dtype=torch.double)
    component = LossComponent('square', lambda z:(z - 2).square().mean())
    return net, opt, x, SimpleNamespace(components=(component, component))


def test_failed_cg_retries_entire_proposal_and_commits_only_certified_direction(monkeypatch):
    net, opt, x, losses = toy()
    engine = ReplayCorrection(CorrectionConfig(damping_retries=3, damping_multiplier=10., max_trials=2))
    calls = []
    def proposal(*args, damping=None, **kwargs):
        calls.append(damping)
        mu = .01 if damping is None else damping
        if mu < 1. - 1e-12:
            raise Uncertified('linear solve failed true-residual certificate',
                              [dict(mu=mu, relative_residual=.1)])
        return CandidateUpdate(torch.tensor([-1.], dtype=torch.double),
                               torch.tensor([[-1.],[-1.]], dtype=torch.double),
                               None, dict(mu=mu, first_order_gains=[1., 1.], curvature=1.))
    monkeypatch.setattr(engine, 'propose', proposal)
    event = engine.step(net, opt, x, losses, [], .1)
    assert calls == [None, .1, 1.]
    assert event['status'] == 'ACCEPTED' and event['accepted']
    assert event['step_norm'] > 0 and net.weight.item() > 0
    assert [row['status'] for row in event['damping_attempts']] == ['UNCERTIFIED','UNCERTIFIED','CERTIFIED']
    assert event['mu'] == 1.


def test_locked_damping_is_not_silently_changed_and_all_failed_solvers_rollback(monkeypatch):
    net, opt, x, losses = toy()
    engine = ReplayCorrection(CorrectionConfig(damping_retries=4))
    initial = State(net, opt)
    calls=[]
    def fail(*args, damping=None, **kwargs):
        calls.append(damping)
        with torch.no_grad():net.weight.add_(3)
        raise Uncertified('linear solve failed true-residual certificate',
                          [dict(mu=.1 if damping is None else damping, relative_residual=.1)])
    monkeypatch.setattr(engine, 'propose', fail)
    event = engine.step(net, opt, x, losses, [], .1, damping=.1)
    assert calls == [.1] and event['status'] == 'UNCERTIFIED'
    assert len(event['damping_attempts']) == 1
    _assert_bitwise(net.state_dict(), initial.values, 'locked_damping_rollback')
    _assert_bitwise(opt.state_dict(), initial.opt, 'locked_damping_optimizer')
    # Without an explicit lock the entire proposal is retried, but still rolled back.
    calls.clear()
    event = engine.step(net, opt, x, losses, [], .1)
    assert len(calls) == 5 and all(row['status'] == 'UNCERTIFIED' for row in event['damping_attempts'])
    _assert_bitwise(net.state_dict(), initial.values, 'all_retries_failed_rollback')


def test_no_op_cannot_be_reported_as_accepted(monkeypatch):
    net, opt, x, losses = toy()
    engine=ReplayCorrection(CorrectionConfig())
    monkeypatch.setattr(engine, 'propose', lambda *a,**kw:CandidateUpdate(
        torch.zeros_like(vector(net)),torch.zeros(2,vector(net).numel(),dtype=torch.double),None,{}))
    row=engine.step(net,opt,x,losses,[],.1)
    assert not row['accepted'] and row['status']=='REJECTED' and row['step_norm']==0.
