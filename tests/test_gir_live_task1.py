"""Historical task-1 GIR, candidate Gram, isolation and unchanged defaults."""
from copy import deepcopy
from types import ModuleType, SimpleNamespace
import subprocess
import pytest
import torch
from torch import nn
from diagnostics.vna_align import _assert_bitwise, _global_state
from experiments.vna_resume import rng_state
from experiments.main_cil import Intervention, parser
from methods import vna_v3 as vna
from methods.vna_sites import HookSite
from methods.vna_losses import CE
from methods.vna_ggn import GGNProducts


@pytest.fixture(autouse=True)
def isolated():
    threads = torch.get_num_threads()
    with _global_state():
        torch.manual_seed(731); torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def problem(dead=False):
    model = nn.Sequential(nn.Linear(1,1), nn.ReLU(), nn.Linear(1,2)).double()
    with torch.no_grad():
        model[0].weight.fill_(1); model[0].bias.zero_()
        model[2].weight.copy_(torch.tensor([[0.],[0. if dead else 1.]])); model[2].bias.zero_()
    site = HookSite(model,'0','2','linear',between=('1',))
    batch = (torch.tensor([[1.],[2.]],dtype=torch.float64),torch.tensor([0,0]))
    opt = torch.optim.SGD(model.parameters(),lr=.05,momentum=.9)
    for p in model.parameters(): opt.state[p]['momentum_buffer']=torch.ones_like(p)
    engine = vna.V3Engine('vna_gir',solver='gram',curvature='power10',gir_live_task1=True,K0=512)
    engine.begin_task(1); engine.mu=.2
    proposal = dict(A=dict(weight=torch.tensor([[3.]],dtype=torch.float64),bias=torch.zeros(1,dtype=torch.float64)),
        direction=torch.ones(2,dtype=torch.float64),score=.1,bias_q=None)
    return engine, model, site, batch, opt, proposal


def metric(model,batch):
    ctx=dict(targets=batch[1],known_classes=0)
    with vna.training_products(model,[],{},batch[0]) as p:
        op=GGNProducts(p,batch[1],CE(),ctx)
        blocks=CE().metric_blocks(op.logits,ctx)
        J=torch.stack([p.Jv(torch.eye(op.size,dtype=torch.float64)[i]) for i in range(op.size)],1)
    return ctx,op.g,blocks,J


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('dead',[False,True])
def test_real_power10_gir_proposals_with_native_model_dtype(dtype,dead):
    engine,model,site,batch,_,_=problem(dead)
    model.to(dtype=dtype)
    batch=(batch[0].to(dtype=dtype),batch[1])
    opt=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9)
    before_rng=rng_state()
    row=engine.apply(model,[site],batch,CE(),dict(targets=batch[1],known_classes=0),opt,1,force=True)
    assert row['gir_live_task1'] and not row['vna_applied'] and row['s']==0
    assert row['solver']=='gram' and row['L_after']<=row['L_before']
    assert all(p.dtype==dtype and bool(torch.isfinite(p).all()) for p in model.parameters())
    if dead: assert row['swaps_dead']==1
    _assert_bitwise(before_rng,rng_state())


@pytest.mark.parametrize('dead',[False,True])
def test_live_gamma_matches_dense_parameter_solve_and_dead_shortcut(monkeypatch,dead):
    engine,model,site,batch,opt,proposal=problem(dead)
    ctx,e,blocks,J=metric(model,batch);H=torch.block_diag(*blocks)
    snapshot=engine._training_state(model,opt);ages=deepcopy(engine.install_ages)
    fu=engine._slot_contribution(model,site,batch,CE(),ctx,0)
    def q(j,c):
        rhs=j.T@c
        return -.5*float(rhs@torch.linalg.solve(j.T@H@j+.2*torch.eye(j.shape[1],dtype=j.dtype),rhs))
    engine.install_gir(model,site,[proposal],[0],opt)
    _,_,_,Jp=metric(model,batch)
    expected=float(e@fu)-.5*float(fu@H@fu)+q(J,e)-q(Jp,e-H@fu)
    engine._restore_training_state(model,opt,snapshot);engine.install_ages=ages
    monkeypatch.setattr(engine,'prepare_gir',lambda *a,**kw:[deepcopy(proposal)])
    trials=[];before_rng=rng_state()
    records=engine._run_gir_task1(model,[site],batch,CE(),ctx,opt,1,e,e,blocks,.2,trials)
    assert len(records)==1 and records[0]['slot_kind']==('dead' if dead else 'live')
    if dead:
        assert records[0]['Gamma'] is None and records[0]['gain_lower_bound']>0
    else:
        assert records[0]['Gamma']==pytest.approx(expected,abs=1e-10) and expected>0
        assert records[0]['j_L']<=0 and records[0]['j_KD'] is None
        assert trials[0]['solver']=='gram'
    assert model[2].weight.eq(0).all()
    assert opt.state[model[0].weight]['momentum_buffer'].eq(0).all()
    _assert_bitwise(before_rng,rng_state())


def test_ce_veto_and_failed_candidate_restore_everything(monkeypatch):
    engine,model,site,batch,opt,proposal=problem()
    ctx,e,blocks,J=metric(model,batch)
    monkeypatch.setattr(engine,'prepare_gir',lambda *a,**kw:[proposal])
    original_jump=engine._slot_jump
    monkeypatch.setattr(engine,'_slot_jump',lambda *a:(1.,None))
    before=engine._training_state(model,opt)
    assert engine._run_gir_task1(model,[site],batch,CE(),ctx,opt,1,e,e,blocks,.2,[])==[]
    _assert_bitwise(before,engine._training_state(model,opt))
    monkeypatch.setattr(engine,'_slot_jump',original_jump)
    original_gram=vna.gram;calls=[]
    def failed_candidate(p):
        calls.append(1)
        if len(calls)>1: raise torch.linalg.LinAlgError('candidate failure')
        return original_gram(p)
    monkeypatch.setattr(vna,'gram',failed_candidate)
    trials=[]
    assert engine._run_gir_task1(model,[site],batch,CE(),ctx,opt,1,e,e,blocks,.2,trials)==[]
    assert trials[0]['reason']=='chol_fail' and not engine.install_ages
    _assert_bitwise(before,engine._training_state(model,opt))


def test_dead_then_four_smallest_live_slots(monkeypatch):
    engine,_,_,batch,_,proposal=problem()
    model=nn.Sequential(nn.Linear(1,6),nn.ReLU(),nn.Linear(6,2)).double()
    site=HookSite(model,'0','2','linear',between=('1',))
    opt=torch.optim.SGD(model.parameters(),lr=.05)
    ctx,e,blocks,J=metric(model,batch)
    order=[]
    monkeypatch.setattr(engine,'prepare_gir',lambda *a,**kw:[proposal])
    monkeypatch.setattr(engine,'_slot_contribution',lambda *a:torch.tensor([float(a[-1]),0.,0.,0.],dtype=torch.float64))
    def veto(*a):
        order.append(a[-1]);return 1.,None
    monkeypatch.setattr(engine,'_slot_jump',veto)
    records=engine._run_gir_task1(model,[site],batch,CE(),ctx,opt,1,e,e,blocks,.2,[])
    assert [r['unit'] for r in records]==[0]
    assert order==[1,2,3,4]


@pytest.mark.parametrize('with_swap',[False,True])
def test_task1_is_gir_only_and_preserves_read_only_state(monkeypatch,with_swap):
    engine,model,site,batch,opt,proposal=problem()
    if not with_swap: engine.mu=None
    def forbidden(*a,**kw): pytest.fail('Task 1 must not run a VNA step or LM search')
    monkeypatch.setattr(engine,'_apply_current',forbidden)
    monkeypatch.setattr(engine,'_lm',forbidden)
    monkeypatch.setattr(engine,'prepare_gir',lambda *a,**kw:[proposal] if with_swap else [])
    monkeypatch.setattr(engine,'_stability',lambda *a,**kw:dict(lambda_max_ggn=1.))
    before=engine._training_state(model,opt);state=engine.state_dict()
    engine.apply(model,[site],batch,CE(),dict(targets=batch[1]),opt,1,force=True,apply=False)
    _assert_bitwise(before,engine._training_state(model,opt));_assert_bitwise(state,engine.state_dict())
    row=engine.apply(model,[site],batch,CE(),dict(targets=batch[1]),opt,1,force=True)
    assert row['skipped']==(not with_swap) and row['swaps_live']==int(with_swap) and row['swaps_dead']==0
    assert row['s']==0 and not row['vna_applied'] and row['vna_skip_reason']=='task1_gir_only'
    assert engine.A==state['A']
    if with_swap: assert engine.mu==state['mu']
    else:
        assert engine.mu>0 and engine.protocol['mu0']==engine.mu
        _assert_bitwise(before,engine._training_state(model,opt))


@pytest.mark.parametrize('task,enabled',[(1,False),(2,False),(2,True)])
def test_default_and_task2_match_original_6be_bitwise(monkeypatch,task,enabled):
    source=subprocess.check_output(['git','show','6be654310c2e4123391fd7d8d31716d25c89c6fb:methods/vna_v3.py'],text=True)
    reference=ModuleType('reference_6be');exec(compile(source,'reference_6be','exec'),reference.__dict__)
    _,model,site,batch,opt,_=problem()
    base=reference.V3Engine('vna_gir',solver='gram',curvature='power10',K0=512)
    engine=vna.V3Engine('vna_gir',solver='gram',curvature='power10',K0=512,gir_live_task1=enabled)
    base.begin_task(task);engine.begin_task(task)
    snapshot=engine._training_state(model,opt)
    monkeypatch.setattr(vna.time,'perf_counter',lambda:0.)
    expected=base.apply(model,[site],batch,CE(),dict(targets=batch[1]),opt,1,force=True)
    expected_state=engine._training_state(model,opt)
    engine._restore_training_state(model,opt,snapshot)
    actual=engine.apply(model,[site],batch,CE(),dict(targets=batch[1]),opt,1,force=True)
    _assert_bitwise(expected,actual);_assert_bitwise(expected_state,engine._training_state(model,opt))
    state=engine.state_dict()
    if enabled:
        state.pop('gir_live_task1');state.pop('task1_gir_policy')
    _assert_bitwise(base.state_dict(),state)


def test_flag_resume_identity_and_default_task1_hook():
    from experiments.main_cil_results import configuration_signature
    protocol=dict(arm='icarl_full',K0=512)
    assert configuration_signature(protocol)!=configuration_signature(dict(protocol,gir_live_task1=True))
    default=vna.V3Engine('vna_gir',solver='gram',curvature='power10')
    variant=vna.V3Engine('vna_gir',solver='gram',curvature='power10',gir_live_task1=True)
    with pytest.raises(ValueError,match='task-1'):default.load_state_dict(variant.state_dict())
    with pytest.raises(ValueError,match='task-1'):variant.load_state_dict(default.state_dict())
    legacy=variant.state_dict();legacy.pop('task1_gir_policy')
    with pytest.raises(ValueError,match='task-1 GIR policy'):variant.load_state_dict(legacy)
    assert not parser().parse_args(['--out','unused']).gir_live_task1
    hook=Intervention(default,512);hook.task=1
    hook.after_step(SimpleNamespace(steps=1),None,None)
    assert not hook.events


def test_flag_enables_task1_hook_with_measured_budget_and_geometric_guard(monkeypatch):
    import time
    from experiments import main_cil
    engine,model,site,batch,opt,_=problem()
    intervention=Intervention(engine,512);intervention.task=1
    learner=SimpleNamespace(_device=torch.device('cpu'),_network=model,_total_classes=2)
    hook=SimpleNamespace(steps=1,iteration_started=time.monotonic()-.01,current_batch=batch)
    monkeypatch.setattr(main_cil,'loss_context',lambda *a:(CE(),dict(targets=batch[1],known_classes=0)))
    monkeypatch.setattr(main_cil,'Prediction',lambda m:m)
    monkeypatch.setattr(main_cil,'sites_for',lambda m:[site])
    calls=[]
    def applied(*a,**kw):
        calls.append(1)
        return dict(skipped=True,swaps=1,budget_guard=True)
    monkeypatch.setattr(engine,'apply',applied)
    intervention.after_step(hook,learner,opt)
    assert calls==[1] and engine.protocol['baseline_iteration_seconds']>0
    assert intervention.interval==1024 and intervention.followups
    hook.steps=1024
    intervention.after_step(hook,learner,opt)
    assert calls==[1]
    hook.steps=1025
    intervention.after_step(hook,learner,opt)
    assert calls==[1,1] and intervention.interval==4096
