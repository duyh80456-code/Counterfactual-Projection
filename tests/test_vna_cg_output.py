"""Matrix-free T29/T31 equivalence and explicit convergence failure."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from methods.vna_exact import FrozenGram, OutputCG, OutputSolve, CGConvergenceError
from methods.vna_v3 import V3Engine, training_products, seed_proposals
from methods.vna_ggn import GGNProducts
from diagnostics.vna_align import _assert_bitwise
from experiments.vna_resume import rng_state
from tests.test_vna_exact import problem, isolated


def operator(J):
    p=SimpleNamespace(parameter_refs=(('weights',torch.zeros(J.shape[1],dtype=J.dtype)),),
        Jv=lambda v:J@v.to(J),JT=lambda v:J.T@v.to(J))
    return FrozenGram(p,high=operator(J.double()) if J.dtype!=torch.float64 else None)


@pytest.mark.parametrize('conv,bn',[(False,False),(False,True),(True,False),(True,True)])
def test_cg_matches_gram_on_native_products(conv,bn):
    model,site,batch,spec,ctx,opt=problem(conv,bn)
    before=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),rng_state()
    with training_products(model,[site],seed_proposals([site]),batch[0]) as p:
        from methods.vna_exact import gram
        op=GGNProducts(p,batch[1],spec,ctx);G,_=gram(p)
        blocks=spec.metric_blocks(op.logits,ctx)+.5*V3Engine.old_blocks(op)
        cg=OutputCG(FrozenGram(p),blocks,.2)
        y=cg.solve(op.g);expected=OutputSolve(G,blocks,.2).solve(op.g)
        torch.testing.assert_close(y,p.JT(expected),rtol=1e-6,atol=1e-9)
        torch.testing.assert_close(y,p.JT(expected),rtol=1e-6,atol=1e-9)
        assert cg.records[-1]['relative_residual']<=1e-8
        assert cg.records[-1]['parameter_relative_residual']<=1e-8
        assert not any(t.ndim==2 and t.shape==G.shape for t in vars(cg).values() if isinstance(t,torch.Tensor))
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state()))


def test_sequential_cg_gamma_matches_frozen_gram():
    torch.manual_seed(4)
    J=torch.randn(9,14,dtype=torch.float64)
    blocks=torch.eye(3,dtype=J.dtype).repeat(3,1,1)/3
    c=torch.randn(9,dtype=J.dtype);mu=.4
    cg=OutputCG(operator(J),blocks,mu);dense=OutputSolve(J@J.T,blocks,mu)
    for i in range(3):
        ids=torch.arange(2*i,2*i+2);remove=J[:,ids];add=torch.randn(9,2,dtype=J.dtype);fu=torch.randn(9,dtype=J.dtype)*.02
        a,b=cg.swapped(remove,add,native_ids=ids,outgoing_ids=ids),dense.swapped(remove,add)
        assert a.gamma(c,fu)==pytest.approx(b.gamma(c,fu),rel=1e-6,abs=1e-9)
        a.commit();b.commit();J=J.clone();J[:,ids]=add;c=c-cg.metric(fu)
        torch.testing.assert_close(cg.G.Jv(cg.solve(c)),dense.G@dense.solve(c),rtol=1e-6,atol=1e-9)
    assert cg.G.version==3


def test_cg_nonconvergence_is_not_returned_as_a_solution():
    torch.manual_seed(3)
    J=torch.randn(8,12,dtype=torch.float64);blocks=torch.eye(4,dtype=J.dtype).repeat(2,1,1)
    factor=OutputCG(operator(J),blocks,.01,max_iter=1)
    with pytest.raises(CGConvergenceError) as error:factor.solve(torch.randn(8,dtype=J.dtype))
    assert not error.value.record['converged'] and error.value.record['relative_residual']>1e-8


def test_lm_nonconverged_trial_increases_mu_and_restores_training_state(monkeypatch):
    model,site,batch,spec,ctx,opt=problem()
    engine=V3Engine();engine.begin_task(1)
    before=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),rng_state()
    original=OutputCG.solve;attempts=[]
    def fail_first(self,rhs,**kw):
        attempts.append(self.mu)
        if len(attempts)==1:raise CGConvergenceError(dict(converged=False,relative_residual=1.,solver='cg'))
        return original(self,rhs,**kw)
    monkeypatch.setattr(OutputCG,'solve',fail_first)
    with training_products(model,[site],seed_proposals([site]),batch[0]) as p:
        op=GGNProducts(p,batch[1],spec,ctx)
        lm=engine._lm(model,batch,spec,ctx,p,op,FrozenGram(p),.2)
    assert attempts[:2]==[.2,.4] and lm['rho_trajectory'][0]==float('-inf')
    assert lm['reason'] is None and not lm['records'][0]['converged']
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state()))


def test_default_exact_and_power10_reference_mode():
    assert V3Engine().curvature=='exact'
    assert V3Engine(curvature='power10').curvature=='power10'


def test_frozen_operator_matches_overlapping_column_replacements():
    torch.manual_seed(11)
    J=torch.randn(7,10,dtype=torch.float64);G=operator(J);current=J.clone()
    for native,outgoing in (([0,1,2],[1,2]),([2,3,4],[3]),([1,6],[6])):
        ids=torch.tensor(native);out=torch.tensor(outgoing);add=torch.randn(7,len(out),dtype=J.dtype)
        G=G.swapped(ids,out,add)
        current[:,ids]=0.;current[:,out]=add
        v=torch.randn(10,dtype=J.dtype);w=torch.randn(7,dtype=J.dtype)
        torch.testing.assert_close(G.Jv(v),current@v,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(G.JT(w),current.T@w,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(G@w,current@current.T@w,rtol=1e-12,atol=1e-12)


def test_cg_application_never_builds_dense_gram_or_live_native_columns(monkeypatch):
    from methods import vna_v3 as vna
    from torch import nn
    from methods.vna_sites import HookSite
    from methods.vna_losses import MSE
    model=nn.Sequential(nn.Linear(3,4),nn.ReLU(),nn.Linear(4,1)).double()
    with torch.no_grad():model[0].weight[:2]=0;model[0].bias[:2]=-1;model[2].weight[:,:2]=0
    batch=torch.randn(6,3,dtype=torch.float64),torch.randn(6,1,dtype=torch.float64)
    site=HookSite(model,'0','2','linear',between=('1',));opt=torch.optim.SGD(model.parameters(),lr=.01)
    engine=V3Engine('vna_gir');engine.begin_task(1);engine.mu=.2
    monkeypatch.setattr(vna,'gram',lambda *a,**k:pytest.fail('dense Gram on CG path'))
    original=engine._slot_columns;seen=[]
    def ids_only(*args,**kwargs):
        assert kwargs['materialize'] is False
        seen.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(engine,'_slot_columns',ids_only)
    row=engine.apply(model,[site],batch,MSE(),dict(targets=batch[1]),opt,1,force=True)
    assert not row['skipped'] and row['swaps']>0 and seen
    assert row['gram_bytes']==0 and all(r['n_jvp_evaluated']==0 for r in row['gir_trials'])
    diagnostic=vna.diagnose(model,batch,sites=[site],spec=MSE(),ctx=dict(targets=batch[1]),mu=.2)
    assert diagnostic['diagnostic_flag'] is None and diagnostic['gram_bytes']==0


def test_native_float32_products_keep_double_iterates_and_verify_caller_residual():
    torch.manual_seed(7)
    J=torch.randn(8,12,dtype=torch.float32)*.01
    factor=OutputCG(operator(J),torch.eye(4,dtype=torch.float64).repeat(2,1,1),10.)
    x=factor.solve(torch.randn(8,dtype=torch.float64))
    assert x.dtype==torch.float64
    assert factor.records[-1]['parameter_dtype']=='torch.float32'
    assert factor.records[-1]['relative_residual']<=1e-8


def test_residual_and_bilinear_q_error_bounds():
    torch.manual_seed(8)
    J=torch.randn(5,8,dtype=torch.float64);mu=.2
    A=J.T@J+mu*torch.eye(8,dtype=J.dtype);b=torch.randn(8,dtype=J.dtype)
    exact=torch.linalg.solve(A,b);x=exact+torch.randn(8,dtype=J.dtype)*1e-6
    r=b-A@x
    assert float((x-exact).norm())<=float(r.norm())/mu
    assert abs(float(-.5*b@x+.5*b@exact))<=float(b.norm()*r.norm())/(2*mu)
