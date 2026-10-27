"""Focused CPU audit: shared protocols, BN transactions and H1/H2 integration."""
from copy import deepcopy
from types import SimpleNamespace
import random
import numpy as np
import pytest
import torch
from torch import nn
from torch.func import functional_call
from diagnostics.vna_align import _global_state,_assert_bitwise
from diagnostics.cl_mechanism import compare_local
from methods.cl_projection import ProjectionOperator,ConstraintSolver
from methods.cl_replay import ReplayCorrection,ReplayLoss,CorrectionConfig,CandidateUpdate,LossComponent,ComputeBudgetExhausted
from methods.cl_state import State,vector
from methods.cl_h2 import GeometryCorrection
from methods.cl_h3 import choose_swap,SwapCandidate
from methods.cl_h4 import DomainMNIST,DomainNetwork
from methods.vna_sites import HookSite
from methods.vna_v3 import training_products
from methods.vna_ggn import GGNProducts


@pytest.fixture(autouse=True)
def isolated_random_state():
    threads=torch.get_num_threads();torch.set_num_threads(1)
    with _global_state():
        torch.manual_seed(96)
        yield
    torch.set_num_threads(threads)


class TinyBN(nn.Module):
    def __init__(self):
        super().__init__();self.hidden=nn.Linear(2,3);self.bn=nn.BatchNorm1d(3)
        self.relu=nn.ReLU();self.head=nn.Linear(3,3)
    def forward(self,x):return self.head(self.relu(self.bn(self.hidden(x))))
    def sites(self):return [HookSite(self,'hidden','head','linear',between=('bn','relu'))]


def problem(dtype=torch.float64):
    model=TinyBN().to(dtype).train();optimizer=torch.optim.SGD(model.parameters(),lr=.02,momentum=.9)
    x=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[-1.,0.],[0.,-1.],[-1.,1.]],dtype=dtype)
    y=torch.tensor([0,1,2,0,1,2]);teacher=torch.randn(6,2,dtype=dtype)
    loss=ReplayLoss(y,y<2,kind='icarl',known=2,teacher=teacher)
    return model,optimizer,x,loss


def assert_state(snapshot,model,optimizer):
    current=State(model,optimizer)
    for name in ('values','buffers','requires_grad','opt','rng','modes','gradients'):
        _assert_bitwise(getattr(snapshot,name),getattr(current,name),'audit.'+name)


def test_state_includes_nonpersistent_buffers_dtype_and_freeze_flags():
    model,optimizer,x,loss=problem()
    model.register_buffer('scratch',torch.tensor([3.,4.],dtype=torch.double),persistent=False)
    optimizer.zero_grad();loss.total(model(x)).backward();optimizer.step()
    snapshot=State(model,optimizer)
    model.float();model.scratch.add_(10);model.hidden.weight.requires_grad_(False);model.eval()
    model.register_buffer('scratch',model.scratch,persistent=True)
    snapshot.restore()
    assert_state(snapshot,model,optimizer)
    assert 'scratch' not in model.state_dict() and model.hidden.weight.requires_grad


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_parameter_gradient_decomposition_on_same_bn_forward(dtype):
    model,optimizer,x,loss=problem(dtype);snapshot=State(model,optimizer)
    with training_products(model,[],{},x) as p:
        op=GGNProducts(p,loss.targets,loss.spec,loss.ctx)
        parts=[torch.autograd.functional.jacobian(c,op.logits).flatten() for c in loss.components]
        summed=sum(p.JT(e).double() for e in parts)
        z=functional_call(model,{n:b.clone() for n,b in model.named_buffers()},(x,))
        direct=torch.cat([v.flatten() for v in torch.autograd.grad(loss.total(z),tuple(model.parameters()))]).double()
        tolerance=2e-5 if dtype==torch.float32 else 1e-10
        torch.testing.assert_close(summed,direct,rtol=tolerance,atol=tolerance)
        torch.testing.assert_close(summed,op.gradient,rtol=tolerance,atol=tolerance)
    assert_state(snapshot,model,optimizer)


def test_replay_teacher_logits_are_constant_even_if_caller_passes_a_graph():
    teacher=torch.randn(4,2,dtype=torch.double,requires_grad=True)
    student=torch.randn(4,3,dtype=torch.double,requires_grad=True);y=torch.tensor([0,1,2,2])
    loss=ReplayLoss(y,y<2,kind='icarl',known=2,teacher=teacher)
    frozen=loss.teacher.clone()
    with torch.no_grad():teacher.add_(1.)
    assert torch.equal(loss.teacher,frozen) and loss.teacher.data_ptr()!=teacher.data_ptr()
    loss.total(student).backward()
    assert teacher.grad is None and not loss.teacher.requires_grad and student.grad is not None


def test_qp_tensor_and_sequence_interfaces_have_identical_kkt_solution():
    A=ProjectionOperator(lambda x:x,lambda x:x,lambda x:x,.5,tolerance=1e-10)
    C=torch.tensor([[1.,0.],[-1.,1.]],dtype=torch.double);d0=torch.tensor([-2.,-.3],dtype=torch.double)
    first=ConstraintSolver(A).solve(d0,C);second=ConstraintSolver(A).solve(d0,list(C))
    torch.testing.assert_close(first.direction,second.direction)
    assert first.slack.min()>=-1e-9 and first.multipliers.min()>=0
    assert first.stationarity<1e-9 and first.complementarity<1e-9


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_sgd_then_bn_intervention_and_matched_kernel_protocol(dtype):
    model,optimizer,x,loss=problem(dtype)
    original=vector(model);optimizer.zero_grad();loss.total(model(x)).backward();optimizer.step()
    assert not torch.equal(original,vector(model))
    assert optimizer.state and model.bn.num_batches_tracked.item()==1
    snapshot=State(model,optimizer)
    cfg=CorrectionConfig(constrained=True,damping_relative=.5,cg_tolerance=1e-5 if dtype==torch.float32 else 1e-8)
    rows=compare_local(model,optimizer,x,loss,model.sites(),.02,random_replicates=2,
        config=cfg,evaluation=(x,loss.targets,2),update_norm=.02,compute_seconds=60.)
    corrections=[r for r in rows if r['comparison_group']!='native_optimizer_reference']
    assert len(corrections)==5
    assert len({r['protocol']['checkpoint_sha256'] for r in corrections})==1
    assert len({r['protocol']['window_sha256'] for r in corrections})==1
    assert len({r['mu'] for r in corrections})==1
    assert all(r['damping_source']=='shared_diagnostic_lock' for r in corrections)
    for row in corrections:
        assert row['status'] in ('ACCEPTED','REJECTED'),row
        assert row['matched_update_budget'] and row['matched_compute_budget']
        assert row['step_norm']<=.02 and row['bn_buffer_drift']==0.
        assert row['acceptance']=='componentwise_nonincrease'
        assert row['seconds_total']>=row['seconds']>=0.
        assert all(t['physical_norm']<=.02 for t in row['trials'] if t['in_update_budget'])
        assert all(a<=b for a,b in zip(row['after'],row['before']))
    normal=next(r for r in rows if r['kernel']=='native_optimizer')
    frozen=next(r for r in rows if r['kernel']=='native_optimizer_frozen_buffers')
    assert normal['bn_buffer_drift']>0 and frozen['bn_buffer_drift']==0
    assert normal['step_norm']==frozen['step_norm']
    assert all(not r['eligible_for_kernel_attribution'] and r['protocol']['acceptance']=='none' for r in (normal,frozen))
    assert_state(snapshot,model,optimizer)


def test_rejected_unified_candidate_restores_bn_momentum_rng_modes_and_grads(monkeypatch):
    model,optimizer,x,loss=problem();model.head.eval()
    optimizer.zero_grad();loss.total(model(x)).backward();optimizer.step()
    snapshot=State(model,optimizer);engine=ReplayCorrection(CorrectionConfig(mode='unified',constrained=False,max_trials=2))
    def bad_candidate(*args,**kwargs):
        torch.rand(3);np.random.rand();random.random()
        return CandidateUpdate(torch.full_like(vector(model),1e6),torch.zeros(2,vector(model).numel(),dtype=torch.double),None,{})
    monkeypatch.setattr(engine,'propose',bad_candidate)
    event=engine.step(model,optimizer,x,loss,model.sites(),.02)
    assert event['status']=='REJECTED' and event['actual_loss_change']==[0.,0.]
    assert len(event['trials'])==2
    assert_state(snapshot,model,optimizer)


def test_compute_budget_censors_and_restores_without_poisoning_next_call(monkeypatch):
    model,optimizer,x,loss=problem();snapshot=State(model,optimizer)
    engine=ReplayCorrection(CorrectionConfig())
    event=engine.step(model,optimizer,x,loss,model.sites(),.02,compute_seconds=1e-12)
    assert event['status']=='SKIPPED_RESOURCE_BUDGET' and not event['accepted']
    assert event['compute_overshoot_seconds']>0
    assert_state(snapshot,model,optimizer)
    monkeypatch.setattr(engine,'propose',lambda *a,**kw:CandidateUpdate(torch.zeros_like(vector(model)),
        torch.zeros(2,vector(model).numel(),dtype=torch.double),None,{}))
    assert engine.step(model,optimizer,x,loss,model.sites(),.02)['status']=='ACCEPTED'
    assert_state(snapshot,model,optimizer)


def test_comparison_without_step_budget_refuses_kernel_attribution():
    model,optimizer,x,loss=problem()
    rows=compare_local(model,optimizer,x,loss,model.sites(),.02,random_replicates=2,
        config=CorrectionConfig(damping_relative=.5,cg_tolerance=1e-8))
    assert all(not r['matched_update_budget'] and not r['eligible_for_kernel_attribution'] for r in rows)


def test_compute_overrun_after_candidate_application_rolls_back(monkeypatch):
    model=nn.Linear(1,1,bias=False).double();model.weight.data.fill_(1.)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9);x=torch.ones(2,1,dtype=torch.double)
    component=LossComponent('square',lambda z:z.square().mean());loss=SimpleNamespace(components=(component,component))
    optimizer.zero_grad();component(model(x)).backward();optimizer.step();snapshot=State(model,optimizer)
    engine=ReplayCorrection(CorrectionConfig(constrained=False));calls=[]
    def boundary():
        calls.append(1)
        if len(calls)==3:raise ComputeBudgetExhausted('test: final diagnostics exceeded allowance')
    monkeypatch.setattr(engine,'boundary',boundary)
    monkeypatch.setattr(engine,'propose',lambda *a,**kw:CandidateUpdate(torch.ones(1,dtype=torch.double),
        torch.ones(2,1,dtype=torch.double),None,dict(first_order_gains=[1.,1.],curvature=1.)))
    event=engine.step(model,optimizer,x,loss,[],.1,compute_seconds=10.)
    assert event['status']=='SKIPPED_RESOURCE_BUDGET' and not event['accepted']
    assert event['trials'][0]['after'][0]<event['trials'][0]['before'][0]
    assert event['after']==event['before'] and event['step_norm']==0. and event['actual_loss_change']==[0.,0.]
    assert_state(snapshot,model,optimizer)


def test_h2_feature_penalty_is_shared_by_split_and_final_solve(monkeypatch):
    class FeatureBN(TinyBN):
        def forward(self,x):
            f=self.relu(self.bn(self.hidden(x)));return dict(features=f,logits=self.head(f))
    class Logits(nn.Module):
        def __init__(self,net):super().__init__();self.native=net
        def forward(self,x):return self.native(x)['logits']
    _,_,x,loss=problem();native=FeatureBN().double();model=Logits(native)
    optimizer=torch.optim.SGD(model.parameters(),lr=.02,momentum=.9)
    engine=GeometryCorrection(CorrectionConfig(kernel='native',damping_relative=.5,cg_tolerance=1e-8))
    seen=[];original=ProjectionOperator.solve
    def capture(self,rhs):
        seen.append((id(self.penalty),float(self.penalty(torch.ones_like(rhs)).norm())))
        return original(self,rhs)
    monkeypatch.setattr(ProjectionOperator,'solve',capture)
    event=engine.step(model,optimizer,x,loss,[],.02,replay_inputs=x[:4],geometry_weight=.3)
    assert event['status'] in ('ACCEPTED','REJECTED'),event
    assert len(seen)>=3 and len({identity for identity,_ in seen})==1
    assert all(norm>0 for _,norm in seen)
    assert event['cotangent_recomputed'] and event['branch']=='H2'


def test_h3_commits_real_inplace_candidate_but_has_no_auto_generator():
    model=nn.Linear(1,1,bias=False).double();model.weight.data.zero_();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    x=torch.ones(3,1,dtype=torch.double);loss=lambda z:(z-1).square().mean()
    def apply(net):
        with torch.no_grad():net.weight.fill_(1.)
    row=choose_swap(model,optimizer,x,(loss,loss),[SwapCandidate('fit','live',apply,lambda *a:None)])
    assert row['choice']=='fit' and model.weight.item()==1.
    assert row['search_scope']=='provided finite candidates only' and row['final_losses']==(0.,0.)


def test_h3_commits_validated_state_without_reexecuting_candidate_callback():
    model=nn.Linear(1,1,bias=False).double();model.weight.data.zero_();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    x=torch.ones(3,1,dtype=torch.double);loss=lambda z:(z-1).square().mean();calls=[]
    def changing_candidate(net):
        calls.append(1)
        with torch.no_grad():net.weight.fill_(1. if len(calls)==1 else 10.)
    row=choose_swap(model,optimizer,x,(loss,loss),[SwapCandidate('changing','live',changing_candidate,lambda *a:None)])
    assert len(calls)==1 and model.weight.item()==1. and row['final_losses']==(0.,0.)


def test_domain_transform_is_shared_between_train_test_without_domain_head():
    image=torch.arange(784,dtype=torch.float32).reshape(1,28,28)/784
    base=[(image,7)]
    train=DomainMNIST(base,2,'permuted',seed=8);test=DomainMNIST(base,2,'permuted',seed=8)
    other=DomainMNIST(base,3,'permuted',seed=8)
    assert torch.equal(train[0][0],test[0][0]) and not torch.equal(train[0][0],other[0][0])
    assert train[0][1]==test[0][1]==other[0][1]==7
    model=DomainNetwork().eval();count=sum(p.numel() for p in model.parameters())
    assert model(torch.stack((train[0][0],other[0][0]))).shape==(2,10)
    assert sum(p.numel() for p in model.parameters())==count
