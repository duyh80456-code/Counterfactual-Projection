"""Dead-only Schur gain, LM acceptance and observational stability logs."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from methods.vna_exact import OutputCG,OutputSolve,CGSwap
from methods.vna_v3 import V3Engine
from methods.vna_losses import MSE
from methods.vna_sites import HookSite
from experiments.main_cil import Intervention
from experiments.vna_resume import rng_state
from diagnostics.vna_align import _assert_bitwise
from tests.test_vna_cg_output import operator


@pytest.fixture(autouse=True)
def isolated():
    threads=torch.get_num_threads()
    with torch.random.fork_rng():
        torch.manual_seed(731);torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize('backend',['gram','cg_ir'])
def test_direct_dead_gain_matches_explicit_schur_and_residual_bound(backend):
    J=torch.randn(6,8,dtype=torch.float64);J[:,:2]=0
    V=torch.randn(6,2,dtype=torch.float64);e=torch.randn(6,dtype=torch.float64);mu=.7
    blocks=torch.eye(3,dtype=torch.float64).repeat(2,1,1)/3
    H=torch.block_diag(*blocks);A=J.T@H@J+mu*torch.eye(8,dtype=J.dtype)
    cross=J.T@H@V;S=V.T@H@V+mu*torch.eye(2,dtype=J.dtype)-cross.T@torch.linalg.solve(A,cross)
    s=V.T@(e-H@J@torch.linalg.solve(A,J.T@e))
    expected=.5*float(s@torch.linalg.solve(S,s))
    if backend=='gram':
        base=OutputSolve(J@J.T,blocks,mu);swap=base.swapped(J[:,:0],V)
    else:
        base=OutputCG(operator(J.float()),blocks,mu)
        # The high operator has the exact intended double Jacobian, not a cast low result.
        base.G.high=operator(J)
        swap=base.swapped(None,V.float().double(),native_ids=torch.arange(2),outgoing_ids=torch.arange(2),high_add=V)
    gain=swap.dead_gain(e)
    assert gain==pytest.approx(expected,rel=1e-9,abs=1e-12)
    assert abs(gain-expected)<=swap.error_bound+1e-13
    assert swap.gain_record['gain_method']=='dead_schur' and swap.error_bound>=0
    assert gain>swap.error_bound


def test_lm_rejects_small_positive_rho_and_accepts_quarter():
    engine=V3Engine();engine.begin_task(2)
    spec=SimpleNamespace(metric_blocks=lambda *a:torch.ones(1,1,1,dtype=torch.float64))
    op=SimpleNamespace(g=torch.ones(1,dtype=torch.float64),H=lambda v:v,loss_spec=spec,
        logits=torch.zeros(1,1,dtype=torch.float64),ctx={})
    p=SimpleNamespace(parameter_refs=(),JT=lambda y:y)
    rhos=iter((.046,.249,.25))
    def ratio(*a):
        y=a[-1];pred=float(y-.5*y.square())
        return (1.,None),(1.-next(rhos)*pred,None)
    engine._measure_ratio=ratio
    row=engine._lm(None,None,spec,{},p,op,torch.eye(1,dtype=torch.float64),.1)
    assert row['mu_trajectory']==pytest.approx([.1,.2,.8])
    assert [t['outcome'] for t in row['trials']]==['model_reject','model_reject','accepted']
    assert row['rho']>=.25


def dead_problem():
    model=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,1)).double()
    with torch.no_grad():model[0].weight[0]=0;model[0].bias[0]=-1;model[2].weight[:,0]=0
    batch=torch.randn(5,2,dtype=torch.float64),torch.randn(5,1,dtype=torch.float64)
    site=HookSite(model,'0','2','linear',between=('1',))
    return model,site,batch,torch.optim.SGD(model.parameters(),lr=.01)


def test_positive_gain_below_error_bound_never_swaps(monkeypatch):
    model,site,batch,opt=dead_problem()
    def uncertain(swap,e):
        swap.error_bound=1.;swap.gain_record={};return .1
    monkeypatch.setattr(CGSwap,'dead_gain',uncertain)
    engine=V3Engine('gir');engine.begin_task(2);engine.mu=.3
    row=engine.apply(model,[site],batch,MSE(),dict(targets=batch[1]),opt,1,force=True)
    assert row['gir_trials'] and row['swaps']==0
    assert all(not r['gain_accepted'] and r['slot_kind']=='dead' for r in row['gir_trials'])


def test_task1_hook_does_not_apply_or_preflight():
    class Forbidden:
        coefficient=1.
        def begin_task(self,t):pass
        def apply(self,*a,**kw):pytest.fail('task1 application')
        def preflight(self,*a,**kw):pytest.fail('task1 preflight')
    for preflight in (False,True):
        intervention=Intervention(Forbidden(),1,preflight)
        learner=SimpleNamespace(_cur_task=0)
        intervention.start_task(learner)
        intervention.after_step(SimpleNamespace(steps=1),learner,None)
        assert not intervention.events and not intervention.counts


def test_followup_twenty_losses_survive_resume_and_do_not_duplicate():
    a=Intervention(V3Engine(),100);a.task=2
    a.followups=[dict(task=2,iteration=10,status='RUNNING',samples=[])]
    hook=SimpleNamespace(steps=0)
    for step in range(11,21):hook.steps=step;a.observe_loss(hook,torch.tensor(float(step)))
    state=deepcopy(a.state_dict());b=Intervention(V3Engine(),100);b.load_state_dict(state)
    for current in (a,b):
        for step in range(21,31):
            hook.steps=step;current.observe_loss(hook,torch.tensor(float(step)))
            current.observe_loss(hook,torch.tensor(float(step)))
    assert a.followups==b.followups and len(a.followups[0]['samples'])==20
    assert a.followups[0]['status']=='COMPLETED'


def test_dead_probe_disagreement_is_readonly_and_observational():
    model=nn.Sequential(nn.Linear(1,1),nn.ReLU(),nn.Linear(1,1)).double()
    with torch.no_grad():model[0].weight.fill_(1);model[0].bias.zero_()
    x=-torch.ones(3,1,dtype=torch.float64);batch=x,model(x).detach()
    probe=torch.ones(256,1,dtype=torch.float64),torch.zeros(256,1,dtype=torch.float64)
    site=HookSite(model,'0','2','linear',between=('1',));opt=torch.optim.SGD(model.parameters(),lr=.01)
    engine=V3Engine('vna');engine.begin_task(2)
    before=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),rng_state()
    row=engine.apply(model,[site],batch,MSE(),dict(targets=batch[1]),opt,1,force=True,apply=False,probe_batch=probe)
    comparison=row['stability_before']['dead_comparison'][site.module_name]
    assert comparison['window_only']==[0] and comparison['probe_dead']==[]
    assert row['stability_before']['probe_size']==256 and row['lambda_max_ggn_before'] is not None
    assert not engine.events
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state()))


def test_dead_gain_bound_covers_perturbed_base_and_augmented_solutions():
    from methods.vna_exact import schur_gain
    J=torch.randn(5,4,dtype=torch.float64);V=torch.randn(5,2,dtype=torch.float64)
    H=torch.diag(torch.rand(5,dtype=torch.float64));e=torch.randn(5,dtype=torch.float64);mu=.3
    A=J.T@H@J+mu*torch.eye(4,dtype=J.dtype)
    aug=torch.cat((J,V),1);AA=aug.T@H@aug+mu*torch.eye(6,dtype=J.dtype)
    x=torch.linalg.solve(A,J.T@e);z=torch.linalg.solve(AA,aug.T@e)
    s=V.T@(e-H@J@x);expected=.5*float(s@z[-2:])
    for scale in (1e-3,1e-7):
        xx=x+scale*torch.randn_like(x);zz=z+scale*torch.randn_like(z)
        ss=V.T@(e-H@J@xx)
        gain,bound,_=schur_gain(ss,zz[-2:],float(torch.trace(V.T@H@V)),mu,
            float((J.T@e-A@xx).norm()),float((aug.T@e-AA@zz).norm()))
        assert abs(gain-expected)<=bound and bound>0


def test_live_slots_are_only_tried_with_ablation(monkeypatch):
    from tests.test_vna_v4_fix2 import gir_problem,run_gir
    engine,model,site,batch,spec,ctx,opt=gir_problem()
    with torch.no_grad():model[0].weight[2].zero_();model[0].bias[2]=1.
    calls=[]
    monkeypatch.setattr(engine,'_slot_jump',lambda *a:(calls.append(a[-1]) or 1.,None))
    assert not run_gir(engine,model,site,batch,spec,ctx,opt,units=(2,)) and not calls
    engine.gir_live=True
    assert not run_gir(engine,model,site,batch,spec,ctx,opt,units=(2,)) and calls==[2]
    with pytest.raises(ValueError,match='ablation'):V3Engine().load_state_dict(engine.state_dict())
    with pytest.raises(ValueError,match='ablation'):V3Engine().configure_schedule(dict(K0=1,gir_live=True,policy_version=engine.policy_version))
    with pytest.raises(ValueError,match='policy'):V3Engine().configure_schedule(dict(K0=1))
    old=V3Engine().state_dict();old.pop('policy_version')
    with pytest.raises(ValueError,match='policy'):V3Engine().load_state_dict(old)


def test_float32_dead_slots_are_checked_on_the_double_model():
    model,site,batch,opt=dead_problem()
    model.float();batch=tuple(t.float() for t in batch)
    engine=V3Engine('gir');engine.begin_task(2);engine.mu=.3
    row=engine.apply(model,[site],batch,MSE(),dict(targets=batch[1]),opt,1,force=True)
    assert row['gir_trials'] and row['swaps']>0
    assert all(r['slot_kind']=='dead' and r['gain_method']=='dead_schur' for r in row['gir_trials'])


@pytest.mark.parametrize('backend',['gram','cg_ir'])
def test_live_ablation_rejects_positive_gamma_below_bound(backend,monkeypatch):
    from methods.vna_exact import WoodburySwap
    model,site,batch,opt=dead_problem()
    with torch.no_grad():model[0].weight.zero_();model[0].bias.fill_(1.)
    def uncertain(swap,e,fu):
        swap.error_bound=1.;swap.gain_record={};return .1
    monkeypatch.setattr(CGSwap if backend=='cg_ir' else WoodburySwap,'gamma',uncertain)
    engine=V3Engine('gir',solver=backend,gir_live=True);engine.begin_task(2);engine.mu=.3
    row=engine.apply(model,[site],batch,MSE(),dict(targets=batch[1]),opt,1,force=True)
    assert row['gir_trials'] and row['swaps']==0
    assert all(r['slot_kind']=='live' and not r['gain_accepted'] and
               r['gain']<r['linear_solve_error_bound'] for r in row['gir_trials'])


def test_live_gram_gamma_bound_covers_perturbed_solves(monkeypatch):
    J=torch.randn(6,8,dtype=torch.float64);ids=torch.arange(2)
    V=torch.randn(6,2,dtype=J.dtype);Jp=J.clone();Jp[:,ids]=V
    H=torch.eye(6,dtype=J.dtype)/3;blocks=torch.eye(3,dtype=J.dtype).repeat(2,1,1)/3
    e=torch.randn(6,dtype=J.dtype);fu=torch.randn_like(e)*.1;mu=.7
    ep=e-H@fu
    def exact_q(J,c):
        b=J.T@c;A=J.T@H@J+mu*torch.eye(J.shape[1],dtype=J.dtype)
        return -.5*float(b@torch.linalg.solve(A,b))
    jump=-float(e@fu)+.5*float(fu@H@fu)
    expected=-jump+exact_q(J,e)-exact_q(Jp,ep)
    base=OutputSolve(J@J.T,blocks,mu);swap=base.swapped(J[:,ids],V)
    native,extended=base.solve,swap.solve
    monkeypatch.setattr(base,'solve',lambda c:native(c)+1e-5*torch.ones_like(c))
    monkeypatch.setattr(swap,'solve',lambda c:extended(c)-2e-5*torch.ones_like(c))
    gamma=swap.gamma(e,fu)
    assert abs(gamma-expected)<=swap.error_bound and swap.error_bound>0
    assert swap.error_bound==swap.gain_record['base_q_error_bound']+swap.gain_record['swapped_q_error_bound']
