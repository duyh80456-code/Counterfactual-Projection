"""True double IR residuals and the shared downstream chain rule."""
from copy import deepcopy
import pytest
import torch
from torch import nn
from methods.vna_downstream import Downstream
from methods.vna_exact import FrozenGram, OutputCG, CGConvergenceError, columns
from methods.vna_v3 import V3Engine, training_products, seed_proposals, rows_proposal
from methods.vna_sites import HookSite
from methods.vna_ggn import GGNProducts
from diagnostics.vna_align import _assert_bitwise
from experiments.vna_resume import rng_state
from tests.test_vna_exact import problem, isolated


@pytest.mark.parametrize('conv',[False,True])
@pytest.mark.parametrize('block_samples',[1,2,8])
def test_shared_downstream_equals_full_jacobian_gradient_and_exact_curvature(conv,block_samples,monkeypatch):
    monkeypatch.setattr(Downstream,'block_samples',block_samples)
    model,site,batch,spec,ctx,opt=problem(conv)
    if conv:
        model[3]=nn.Conv2d(2,3,3,padding=1,dilation=1).double()
        site=HookSite(model,'0','3','conv',between=('1','2'))
    A=rows_proposal(site,torch.randn(1,model[0].weight[0].numel()+int(model[0].bias is not None),dtype=torch.float64))
    before=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),rng_state()
    ds=Downstream(model,site,batch[0])
    import weakref
    original=ds._block;blocks=[]
    def one_live_block(inputs,check=False):
        assert all(ref() is None for ref in blocks)
        result=original(inputs,check=check)
        if not check:blocks.append(weakref.ref(result[0]))
        return result
    monkeypatch.setattr(ds,'_block',one_live_block)
    with training_products(model,[site],{site.module_name:A},batch[0]) as p:
        op=GGNProducts(p,batch[1],spec,ctx);size=sum(v.numel() for _,v in p.extension_refs)
        ref=p.extension_refs[0][1]
        direct=columns(p.JEv,torch.arange(size),size,ref,chunk=1)
        J=ds.columns(A)
        torch.testing.assert_close(J,direct,rtol=1e-10,atol=1e-12)
        H=spec.metric_blocks(op.logits,ctx)+2.*V3Engine.old_blocks(op)
        metric=lambda v:torch.einsum('nij,nj->ni',H,v.reshape(H.shape[:2])).flatten()
        (s,a,active,count),=ds.score([dict(A=A)],op.g,metric)
        torch.testing.assert_close(s,p.JET(op.g),rtol=1e-10,atol=1e-12)
        b=direct.T@torch.vmap(metric)(direct.T).T
        torch.testing.assert_close(torch.linalg.eigvalsh(a),torch.linalg.eigvalsh(b),rtol=1e-9,atol=1e-12)
    assert ds.vjp_calls==1+6*((len(batch[0])+block_samples-1)//block_samples)
    assert ds.dtype==torch.float64 and not hasattr(ds,'D')
    output_elements=3*batch[0].shape[-1]**2 if conv else 3
    assert ds.bytes==min(block_samples,len(batch[0]))*3*output_elements*8
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state()))


def test_downstream_includes_residual_branch_and_skip_scale():
    class Residual(nn.Module):
        def __init__(self):
            super().__init__();self.conv_a=nn.Conv2d(2,2,1);self.relu=nn.ReLU()
            self.conv_b=nn.Conv2d(2,2,3,padding=1);self.fc=nn.Linear(2,3)
        def forward(self,x):return self.fc(torch.relu(x+.3*self.conv_b(self.relu(self.conv_a(x)))).mean((2,3)))
    model=Residual().double();site=HookSite(model,'conv_a','conv_b','conv',between=('relu',))
    x=torch.randn(2,2,3,3,dtype=torch.float64);A=dict(weight=torch.randn(1,2,1,1,dtype=torch.float64),bias=torch.zeros(1,dtype=torch.float64))
    ds=Downstream(model,site,x)
    with training_products(model,[site],{site.module_name:A},x) as p:
        size=sum(v.numel() for _,v in p.extension_refs);ref=p.extension_refs[0][1]
        J=columns(p.JEv,torch.arange(size),size,ref)
    torch.testing.assert_close(ds.columns(A),J,rtol=1e-10,atol=1e-12)


def test_no_bn_alone_does_not_justify_batch_separability():
    class Coupled(nn.Module):
        def __init__(self):super().__init__();self.a=nn.Linear(2,3);self.b=nn.Linear(3,2)
        def forward(self,x):
            y=self.b(torch.relu(self.a(x)))
            return y+y.mean(0,keepdim=True)
    model=Coupled().double();site=HookSite(model,'a','b','linear',between=())
    with pytest.raises(ValueError,match='cross-sample'):
        Downstream(model,site,torch.randn(3,2,dtype=torch.float64)).measure()


def test_ir_targets_actual_double_model_not_float32_residual():
    model,site,batch,spec,ctx,opt=problem();model.float();batch=(batch[0].float(),batch[1])
    before=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),rng_state()
    with training_products(model,[site],seed_proposals([site]),batch[0]) as p:
        op=GGNProducts(p,batch[1],spec,ctx);G=FrozenGram(p)
        blocks=spec.metric_blocks(op.logits,ctx);factor=OutputCG(G,blocks,.3)
        x=factor.solve(op.g);hp=G.high.products;P=sum(v.numel() for _,v in hp.parameter_refs)
        J=torch.stack([hp.Jv(v) for v in torch.eye(P,dtype=torch.float64)],1)
        A=J.T@torch.block_diag(*blocks)@J+.3*torch.eye(P,dtype=torch.float64);b=J.T@op.g
        expected=torch.linalg.solve(A,b)
        torch.testing.assert_close(x,expected,rtol=1e-9,atol=1e-11)
        assert float((b-A@x).norm()/b.norm())<=1e-10
        row=factor.records[-1]
        assert row['inner_tolerance']==1e-4 and row['residual_dtype']=='torch.float64'
        assert len(row['residual_history'])>=2 and row['converged']
        assert next(G.high.products.model.parameters()).dtype==torch.float64
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state()))


def test_warm_and_cached_solutions_are_verified_and_local_to_one_operator():
    model,site,batch,spec,ctx,_=problem()
    with training_products(model,[site],seed_proposals([site]),batch[0]) as p:
        op=GGNProducts(p,batch[1],spec,ctx);G=FrozenGram(p);records=[]
        solve=V3Engine._output_solver(G,spec.metric_blocks(op.logits,ctx),V3Engine.old_blocks(op),.3,records)
        solve('d_N',op.g,0.);warm=solve('d_N',op.g,.1)
        assert records[-1]['warm_start']
        cold=OutputCG(G,spec.metric_blocks(op.logits,ctx)+.1*V3Engine.old_blocks(op),.3).solve(op.g)
        torch.testing.assert_close(warm,cold,rtol=1e-8,atol=1e-10)
        assert not FrozenGram(p).warm
        factor=OutputCG(G,spec.metric_blocks(op.logits,ctx),.3)
        a=factor.solve(op.g);b=factor.solve(op.g)
        assert factor.records[-1]['cached'] and factor.solves==1
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not any('warm' in key for key in vars(V3Engine()))


def test_bn_uses_existing_power10_path(monkeypatch):
    model,site,batch,spec,ctx,opt=problem(bn=True)
    monkeypatch.setattr('methods.vna_v3.Downstream',lambda *a:pytest.fail('D_site with BN'))
    engine=V3Engine();engine.mu=.3
    result=engine.prepare_gir(model,site,batch,spec,ctx,opt,[0],1,
        residual=torch.randn(9,dtype=torch.float64),H=lambda v:v,mu=.3)
    assert result and all(r['curvature_method']=='power10' for r in result)


def test_exact_prepare_reuses_one_double_downstream_per_site():
    model,site,batch,spec,ctx,opt=problem();model.float();batch=(batch[0].float(),batch[1])
    engine=V3Engine();engine.mu=.3;cache={};r=torch.randn(9,dtype=torch.float64)
    before=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),rng_state()
    rows=engine.prepare_gir(model,site,batch,spec,ctx,opt,[0],1,residual=r,H=lambda v:v,mu=.3,cache=cache)
    ds=cache['downstream']
    other=engine.prepare_gir(model,site,batch,spec,ctx,opt,[0],1,residual=2*r,H=lambda v:v,mu=.3,cache=cache)
    assert ds is cache['downstream'] and rows and other
    for row in rows:
        assert row['curvature_method']=='eigvalsh_downstream'
        assert '_Jv64' not in row and '_Jv' not in row
        high=ds.model;fixed=HookSite(high,site.first,site.second,site.kind,between=site.between)
        A={k:None if v is None else v.double() for k,v in row['A'].items()}
        with training_products(high,[fixed],{site.module_name:A},batch[0].double()) as p:
            size=sum(v.numel() for _,v in p.extension_refs);ref=p.extension_refs[0][1]
            J=columns(p.JEv,torch.arange(size),size,ref)
        torch.testing.assert_close(ds.columns(row['A']),J,rtol=1e-10,atol=1e-12)
        assert row['curvature']==pytest.approx(float(torch.linalg.eigvalsh(J.T@J)[-1]),rel=1e-9,abs=1e-12)
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state()))


def test_native_float32_full_gir_preflight_restores_all_state(monkeypatch):
    from methods.vna_losses import MSE
    model=nn.Sequential(nn.Linear(3,4),nn.ReLU(),nn.Linear(4,1)).float()
    with torch.no_grad():model[0].weight[:2]=0;model[0].bias[:2]=-1;model[2].weight[:,:2]=0
    site=HookSite(model,'0','2','linear',between=('1',))
    batch=torch.randn(6,3),torch.randn(6,1)
    opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    for p in model.parameters():opt.state[p]['momentum_buffer']=torch.ones_like(p)
    engine=V3Engine('vna_gir');engine.begin_task(1);engine.mu=.3
    before=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),rng_state()
    original=Downstream.columns;dense_calls=[]
    def only_at_gamma(ds,A):
        import inspect
        assert inspect.currentframe().f_back.f_code.co_name=='_run_gir'
        dense_calls.append(1)
        return original(ds,A)
    monkeypatch.setattr(Downstream,'columns',only_at_gamma)
    monkeypatch.setattr(Downstream,'block_samples',2)
    point=engine.preflight(model,[site],batch,MSE(),dict(targets=batch[1]),opt)['trials'][0]
    assert 0<len(dense_calls)<=len(point['gir_trials'])
    assert point['downstream'][0]['block_samples']==2 and not point['downstream'][0]['retains_full_D']
    assert not point['skipped'] and point['swaps']>0 and point['downstream']
    assert all(r['relative_residual']<=1e-10 for r in point['solves'] if r.get('converged'))
    assert all(r['candidate_double_jvp_evaluated']==0 for r in point['gir_trials'])
    assert engine.applications==0 and not engine.events
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state()))


def test_double_replacements_survive_overlapping_native_swaps():
    from tests.test_vna_cg_output import operator
    J=torch.randn(6,8,dtype=torch.float64)
    G=operator(J.float());target=J.float().double()
    for native,outgoing in (([0,1,2],[1,2]),([2,3,4],[3,4])):
        ids=torch.tensor(native);out=torch.tensor(outgoing)
        add=torch.randn(6,len(out),dtype=torch.float64)
        G=G.swapped(ids,out,add.float().double(),add)
        target[:,ids]=0.;target[:,out]=add
        rhs=torch.randn(6,dtype=torch.float64)
        factor=OutputCG(G,torch.eye(3,dtype=torch.float64).repeat(2,1,1),2.)
        x=factor.solve(rhs)
        expected=torch.linalg.solve(target.T@target+2.*torch.eye(8,dtype=torch.float64),target.T@rhs)
        torch.testing.assert_close(x,expected,rtol=1e-9,atol=1e-11)


def test_downstream_handles_inplace_activation_after_output():
    model=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,2),nn.ReLU(inplace=True)).double()
    site=HookSite(model,'0','2','linear',between=('1',))
    x=torch.randn(3,2,dtype=torch.float64)
    A=dict(weight=torch.randn(1,2,dtype=torch.float64),bias=torch.zeros(1,dtype=torch.float64))
    ds=Downstream(model,site,x)
    with training_products(model,[site],{site.module_name:A},x) as p:
        size=sum(v.numel() for _,v in p.extension_refs);ref=p.extension_refs[0][1]
        J=columns(p.JEv,torch.arange(size),size,ref)
    torch.testing.assert_close(ds.columns(A),J,rtol=1e-10,atol=1e-12)


def test_ir_failure_cannot_accept_an_inaccurate_native_operator():
    from tests.test_vna_cg_output import operator
    G=operator(torch.eye(2,dtype=torch.float32)*100.)
    G.high=operator(torch.eye(2,dtype=torch.float64))
    factor=OutputCG(G,torch.eye(2,dtype=torch.float64)[None],.1)
    with pytest.raises(CGConvergenceError) as error:factor.solve(torch.ones(2,dtype=torch.float64))
    assert error.value.record['failure_reason']=='ir_stagnation'
    assert not error.value.record['converged']
