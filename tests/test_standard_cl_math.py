"""Specified mathematical checks. Written only; no results are asserted."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from methods.cl_projection import ProjectionOperator,ConstraintSolver,Uncertified,schur_gain
from methods.cl_replay import ReplayLoss,ReplayCorrection,CorrectionConfig,CandidateUpdate,LossComponent
from methods.cl_state import State,evaluate,vector
from methods.cl_h3 import vna_column_derivative
from diagnostics.vna_align import _assert_bitwise,_global_state


@pytest.fixture(autouse=True)
def random_isolation():
    with _global_state():
        torch.manual_seed(123)
        yield


def operator(J,H,mu=.7,R=None,penalty=None,tolerance=1e-9):
    J,H=J.double(),H.double();R=torch.eye(J.shape[1],dtype=J.dtype) if R is None else R.double()
    return ProjectionOperator(lambda x:J@x,lambda v:J.T@v,lambda v:H@v,mu,
        regularizer=lambda d:R@d,regularizer_floor=float(torch.linalg.eigvalsh(R).min()),
        penalty=(lambda d:torch.zeros_like(d)) if penalty is None else penalty,
        tolerance=tolerance,max_iterations=100)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_projection_direct_generalized_metric_and_feature_penalty(dtype):
    J=torch.randn(6,4,dtype=dtype).double();Q=torch.randn(6,6,dtype=dtype).double();M=Q@Q.T
    S=torch.randn(4,4,dtype=dtype).double();R=S@S.T+torch.eye(4,dtype=torch.double)
    E=torch.randn(6,3,dtype=dtype).double();K=E@E.T;phi=torch.randn(3,4,dtype=dtype).double()
    P=.3*phi.T@phi;A=operator(J,M,R=R,penalty=lambda d:P@d)
    g=torch.randn(4,dtype=torch.double);d,z=A.project(g,lambda v:K@v)
    dense=J.T@M@J+A.mu*R+P
    expected=torch.linalg.solve(dense,J.T@M@K@M@J@torch.linalg.solve(dense,g))
    torch.testing.assert_close(d,expected,rtol=1e-7,atol=1e-9)
    torch.testing.assert_close(g@d,z@K@z,rtol=1e-7,atol=1e-9)
    assert float(g@d)>=-1e-9


def test_virtual_can_be_better_or_worse_than_native_at_common_step():
    I=torch.eye(2,dtype=torch.double);g=torch.tensor([1.,.2],dtype=torch.double);A=operator(I,I)
    native,_=A.project(g,lambda z:z)
    better,_=A.project(g,lambda z:4*z);worse,_=A.project(g,lambda z:.1*z)
    assert float(g@better)>float(g@native)>float(g@worse)
    assert not torch.any(A.project(torch.zeros_like(g),lambda z:4*z)[0])


def test_shared_metric_split_has_no_unconstrained_candidate_novelty():
    J=torch.randn(5,3,dtype=torch.double);H=torch.eye(5,dtype=torch.double)
    E=torch.randn(5,2,dtype=torch.double);K=E@E.T;A=operator(J,H)
    gn=torch.randn(3,dtype=torch.double);gr=torch.randn(3,dtype=torch.double)
    dn,zn=A.project(gn,lambda z:K@z);dr,zr=A.project(gr,lambda z:K@z)
    total,z=A.project(gn+gr,lambda z:K@z)
    torch.testing.assert_close(total,dn+dr,rtol=1e-7,atol=1e-9)
    torch.testing.assert_close(z,zn+zr,rtol=1e-7,atol=1e-9)
    Q=torch.stack((zn,zr))@K@torch.stack((zn,zr)).T
    assert float(torch.linalg.eigvalsh(Q).min())>=-1e-9


@pytest.mark.parametrize('gradients',[[[1.,0.],[0.,1.]],[[1.,0.],[-1.,0.]],[[1.,0.],[2.,0.]],[[0.,0.],[1.,1.]],[[0.,0.],[0.,0.]]])
@pytest.mark.parametrize('candidate',[[-2.,-1.],[1.,2.]])
def test_two_halfspace_kkt_and_numerical_qp_reference(gradients,candidate):
    scipy=pytest.importorskip('scipy.optimize')
    metric=torch.tensor([[2.,.2],[.2,1.]],dtype=torch.double)
    A=operator(torch.eye(2),torch.zeros(2,2),mu=1.,R=metric)
    d0=torch.tensor(candidate,dtype=torch.double);C=torch.tensor(gradients,dtype=torch.double)
    result=ConstraintSolver(A,1e-8).solve(d0,C)
    assert float(result.slack.min())>=-1e-8 and float(result.multipliers.min())>=-1e-8
    assert result.stationarity<1e-7 and result.complementarity<1e-7
    assert float((A(result.direction-d0)-C.T@result.multipliers).norm())<1e-7
    nonzero=np.linalg.norm(C.numpy(),axis=1)>0
    reference=scipy.minimize(lambda d:.5*(d-d0.numpy())@metric.numpy()@(d-d0.numpy()),np.zeros(2),
        jac=lambda d:metric.numpy()@(d-d0.numpy()),method='SLSQP',
        constraints=[dict(type='ineq',fun=lambda d:C.numpy()[nonzero]@d)] if nonzero.any() else [],
        options=dict(ftol=1e-11,maxiter=100))
    assert reference.success
    np.testing.assert_allclose(result.direction.numpy(),reference.x,atol=1e-6)


def test_failed_cg_does_not_certify():
    J=torch.diag(torch.tensor([1.,10.,100.],dtype=torch.double));A=operator(J,torch.eye(3),tolerance=1e-12)
    A.max_iterations=1
    with pytest.raises(Uncertified):A.solve(torch.ones(3,dtype=torch.double))
    assert A.solves[-1]['status']=='UNCERTIFIED'


@pytest.mark.parametrize('kind,reference',[('er','replay_ce_kd'),('icarl','replay_ce_kd'),('icarl','kd')])
def test_loss_partition_preserves_native_mean_ce_and_kd(kind,reference):
    z=torch.randn(5,4,dtype=torch.double);targets=torch.tensor([0,1,2,3,3]);teacher=torch.randn(5,2,dtype=torch.double)
    loss=ReplayLoss(targets,targets<2,kind=kind,known=2,teacher=teacher,reference=reference)
    torch.testing.assert_close(sum(c(z) for c in loss.components),loss.total(z),rtol=1e-12,atol=1e-12)
    v=torch.randn_like(z)
    hessian=torch.autograd.functional.hessian(loss.total,z).reshape(z.numel(),z.numel())
    torch.testing.assert_close(loss.spec.metric_apply(z,v,loss.ctx).flatten(),hessian@v.flatten(),rtol=1e-10,atol=1e-10)


def test_old_kd_shift_invariance_does_not_protect_global_classifier():
    z=torch.tensor([[0.,0.,1.]],dtype=torch.double);y=torch.tensor([2])
    loss=ReplayLoss(y,torch.tensor([False]),kind='icarl',known=2,teacher=torch.zeros(1,2,dtype=torch.double))
    shifted=z.clone();shifted[:,:2]+=10
    torch.testing.assert_close(loss.kd(z),loss.kd(shifted))
    assert z.argmax(1).item()==2 and shifted.argmax(1).item()!=2


def test_float32_teacher_uses_same_fixed_weights_in_residual_and_metric():
    z=torch.randn(5,4,dtype=torch.double);teacher=torch.randn(5,3,dtype=torch.float32)
    y=torch.tensor([0,1,2,3,3]);loss=ReplayLoss(y,y<3,kind='icarl',known=3,teacher=teacher)
    gradient=torch.autograd.functional.jacobian(loss.total,z)
    parts=sum(torch.autograd.functional.jacobian(c,z) for c in loss.components)
    torch.testing.assert_close(parts,gradient,rtol=1e-12,atol=1e-12)
    v=torch.randn_like(z);H=torch.autograd.functional.hessian(loss.total,z).reshape(z.numel(),z.numel())
    torch.testing.assert_close(loss.spec.metric_apply(z,v,loss.ctx).flatten(),H@v.flatten(),rtol=1e-11,atol=1e-12)


def test_bn_diagnostic_and_rejected_step_restore_everything(monkeypatch):
    net=nn.Sequential(nn.Linear(2,3),nn.BatchNorm1d(3),nn.ReLU(),nn.Linear(3,2)).float().train()
    opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9);x=torch.randn(4,2)
    losses=ReplayLoss(torch.tensor([0,1,0,1]),torch.tensor([False,False,True,True]))
    state=State(net,opt)
    evaluate(net,x,losses.components)
    _assert_bitwise(net.state_dict(),state.values,'bn.forward')
    engine=ReplayCorrection(CorrectionConfig(max_trials=2))
    def fail(*a,**kw):raise Uncertified('forced test failure')
    monkeypatch.setattr(engine,'propose',fail)
    row=engine.step(net,opt,x,losses,[],.1)
    assert row['status']=='UNCERTIFIED'
    _assert_bitwise(net.state_dict(),state.values,'rejected.model')
    _assert_bitwise(opt.state_dict(),state.opt,'rejected.optimizer')
    from experiments.vna_resume import rng_state
    _assert_bitwise(rng_state(),state.rng,'rejected.rng')


def test_finite_loss_rejection_rolls_back_real_parameter_step(monkeypatch):
    net=nn.Linear(1,1,bias=False).double();net.weight.data.zero_();x=torch.ones(3,1,dtype=torch.double)
    opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9)
    c=LossComponent('square',lambda z:z.square().mean())
    losses=SimpleNamespace(components=(c,c));engine=ReplayCorrection(CorrectionConfig(constrained=False,max_trials=2))
    monkeypatch.setattr(engine,'propose',lambda *a,**kw:CandidateUpdate(torch.ones(1,dtype=torch.double),torch.zeros(2,1,dtype=torch.double),None,{}))
    before=State(net,opt);event=engine.step(net,opt,x,losses,[],.1)
    assert not event['accepted'] and event['step_norm']==0 and event['actual_loss_change']==[0.,0.]
    _assert_bitwise(net.state_dict(),before.values,'finite.reject')


def test_schur_added_column_gain_and_noncommuting_vna_derivative():
    J=torch.randn(4,2,dtype=torch.double);V=torch.randn(4,1,dtype=torch.double)
    Q=torch.randn(4,4,dtype=torch.double);H=Q@Q.T;K=J@J.T;e=torch.randn(4,dtype=torch.double);mu=.6
    def minimum(X):
        b=X.T@e;A=X.T@H@X+mu*torch.eye(X.shape[1],dtype=X.dtype)
        return -.5*b@torch.linalg.solve(A,b)
    torch.testing.assert_close(schur_gain(J,V,H,e,mu),minimum(J)-minimum(torch.cat((J,V),1)))
    E=torch.randn(4,3,dtype=torch.double);KE=E@E.T;v=V[:,0]
    def value(alpha):
        z=e-mu*torch.linalg.solve(H@(K+alpha*torch.outer(v,v))+mu*torch.eye(4,dtype=H.dtype),e)
        return z@KE@z
    epsilon=1e-5
    torch.testing.assert_close(vna_column_derivative(K,H,KE,e,v,mu,.2),
        (value(.2+epsilon)-value(.2-epsilon))/(2*epsilon),rtol=1e-5,atol=1e-6)
