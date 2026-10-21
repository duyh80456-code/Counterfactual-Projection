"""Prompt section 8: small float64 T29–T31 checks, no training smoke runs."""
import inspect
from copy import deepcopy
import pytest
import torch
from torch import nn
from diagnostics.vna_align import _assert_bitwise, _global_state
from experiments.vna_resume import rng_state
from methods.vna_exact import gram, columns, OutputSolve
from methods.vna_losses import ICaRLPyCIL, MSE
from methods.vna_sites import HookSite
from methods.vna_ggn import GGNProducts
from methods.vna_v3 import V3Engine, training_products, seed_proposals, net_gain


@pytest.fixture(autouse=True)
def isolated():
    threads = torch.get_num_threads()
    with _global_state():
        torch.manual_seed(40); torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def problem(conv=False, bn=False):
    if conv:
        model = nn.Sequential(nn.Conv2d(1,2,1), nn.BatchNorm2d(2) if bn else nn.Identity(),
                              nn.ReLU(), nn.Conv2d(2,3,1), nn.AdaptiveAvgPool2d(1), nn.Flatten()).double()
        x = torch.randn(3,1,2,2,dtype=torch.float64)
    else:
        model = nn.Sequential(nn.Linear(2,3), nn.BatchNorm1d(3) if bn else nn.Identity(),
                              nn.ReLU(), nn.Linear(3,3)).double()
        x = torch.randn(3,2,dtype=torch.float64)
    site = HookSite(model,'0','3','conv' if conv else 'linear',between=('1','2'))
    y = torch.tensor([0,1,2]); spec=ICaRLPyCIL(known_classes=2)
    ctx=dict(targets=y,known_classes=2,teacher_logits=torch.randn(3,2,dtype=torch.float64))
    return model,site,(x,y),spec,ctx,torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)


@pytest.mark.parametrize('conv',[False,True])
@pytest.mark.parametrize('bn',[False,True])
def test_t29_dense(conv,bn):
    model,site,batch,spec,ctx,_=problem(conv,bn)
    with training_products(model,[site],seed_proposals([site]),batch[0]) as p:
        op=GGNProducts(p,batch[1],spec,ctx)
        G,_=gram(p,chunk=3)
        J=torch.stack([p.Jv(v) for v in torch.eye(op.size,dtype=torch.float64)],1)
        torch.testing.assert_close(G,J@J.T,rtol=1e-10,atol=1e-10)
        blocks=spec.metric_blocks(op.logits,ctx); old=V3Engine.old_blocks(op)
        H=torch.block_diag(*blocks)
        for beta in (0.,.5,5.):
            M=torch.block_diag(*(blocks+beta*old)); mu=.2
            solver=OutputSolve(G,blocks+beta*old,mu)
            e=op.g; y=solver.solve(e); image=G@y; x=p.JT(y)
            A=J.T@M@J+mu*torch.eye(op.size,dtype=torch.float64)
            exact=torch.linalg.solve(A,J.T@e)
            quantities=[(x,exact),(M@image,M@J@exact),((y@G@y).sqrt(),exact.norm()),
                        (image@H@image,exact@(J.T@H@J)@exact),(e@image,(J.T@e)@exact),
                        (e-M@image,e-M@J@exact)]
            u=torch.randn(len(e),dtype=e.dtype)
            quantities.append((p.JT(solver.solve(H@u)),torch.linalg.solve(A,J.T@H@u)))
            for actual,expected in quantities: torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-10)


def test_dead_install_gram():
    model,site,batch,spec,ctx,opt=problem()
    with torch.no_grad(): model[0].weight[0]=0; model[0].bias[0]=-1
    a=dict(weight=torch.tensor([[[1.]]],dtype=torch.float64).reshape(1,1).expand(1,2).clone(),
           bias=torch.tensor([.2],dtype=torch.float64))
    with training_products(model,[site],{site.module_name:a},batch[0]) as p:
        logits=p._call({}); G,_=gram(p)
        ne=sum(v.numel() for _,v in p.extension_refs)
        Jv=columns(p.JEv,torch.arange(ne),ne,p.extension_refs[0][1])
    V3Engine().install_gir(model,site,[dict(A=a)],[0],opt)
    with training_products(model,[site],seed_proposals([site]),batch[0]) as p:
        new,_=gram(p)
        torch.testing.assert_close(p._call({}),logits,atol=1e-10,rtol=0)
    torch.testing.assert_close(new,G+Jv@Jv.T,atol=1e-10,rtol=0)


def test_woodbury_sequential_dense():
    n,c,P=2,3,11; J=torch.randn(n*c,P,dtype=torch.float64)
    blocks=torch.eye(c,dtype=torch.float64).repeat(n,1,1)/n
    H=torch.block_diag(*blocks); e=torch.randn(n*c,dtype=torch.float64); mu=.3
    solver=OutputSolve(J@J.T,blocks,mu)
    for i in range(2):
        remove=J[:,i:i+1]; add=torch.randn(n*c,2,dtype=torch.float64); fu=torch.randn(n*c,dtype=torch.float64)*.01
        Jnew=torch.cat((J[:,:i],J[:,i+1:],add),1)
        swap=solver.swapped(remove,add)
        assert swap.gamma(e,fu)==pytest.approx(net_gain(e,H,J,Jnew,fu,mu),rel=1e-8,abs=1e-10)
        swap.commit(); e=e-H@fu; J=Jnew
        direct=OutputSolve(J@J.T,blocks,mu)
        torch.testing.assert_close(solver.solve(e),direct.solve(e),rtol=1e-8,atol=1e-10)


def test_lm_reject_nan_and_restore(monkeypatch):
    model,site,batch,spec,ctx,opt=problem(bn=True)
    engine=V3Engine(); engine.begin_task(1)
    before=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),rng_state()
    original=engine._loss_parts; calls=[0]
    def loss(*args):
        calls[0]+=1
        value=original(*args)
        return (float('nan'),value[1]) if calls[0]==1 else value
    monkeypatch.setattr(engine,'_loss_parts',loss)
    with training_products(model,[site],seed_proposals([site]),batch[0]) as p:
        op=GGNProducts(p,batch[1],spec,ctx); G,_=gram(p)
        lm=engine._lm(model,batch,spec,ctx,p,op,G,1e-10)
    assert lm['reason'] is None and lm['rho']>0 and torch.isfinite(torch.tensor(lm['rho']))
    assert lm['rho_trajectory'][0]==float('-inf') and len(lm['mu_trajectory'])>1
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state()))


def test_nan_gate_no_gir(monkeypatch):
    model,site,batch,spec,ctx,opt=problem()
    engine=V3Engine('vna_gir'); engine.begin_task(1)
    monkeypatch.setattr(engine,'_measure_ratio',lambda *a: ((1.,None),(float('nan'),None)))
    monkeypatch.setattr(engine,'_run_gir',lambda *a: pytest.fail('GIR after failed LM'))
    row=engine.apply(model,[site],batch,spec,ctx,opt,1,force=True)
    assert row['skipped'] and row['skip_reason']=='lm_budget'
    assert 'conjugate_gradient' not in inspect.getsource(__import__('methods.vna_v3',fromlist=['']))


def test_read_only_application_and_nonfinite_optimizer_rollback(monkeypatch):
    model,site,batch,spec,ctx,opt=problem(bn=True)
    engine=V3Engine('kernel_self');engine.begin_task(1)
    before=deepcopy(model.state_dict()),deepcopy(opt.state_dict()),deepcopy(engine.state_dict()),rng_state()
    engine.apply(model,[site],batch,spec,ctx,opt,1,force=True,apply=False)
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),engine.state_dict(),rng_state()))
    def broken(*a):
        opt.state[next(model.parameters())]['nested']={'bad':torch.tensor(float('nan'))}
        return dict(skipped=True)
    monkeypatch.setattr(engine,'_apply',broken)
    row=engine.apply(model,[site],batch,spec,ctx,opt,1,force=True)
    assert row['nonfinite_after_apply'] and row['skipped']
    _assert_bitwise(before[:2],(model.state_dict(),opt.state_dict()))


def test_t31_two_dead_swaps_use_frozen_theta_and_updated_residual(monkeypatch):
    model=nn.Sequential(nn.Linear(3,4),nn.ReLU(),nn.Linear(4,1)).double()
    with torch.no_grad():model[0].weight[:2]=0;model[0].bias[:2]=-1;model[2].weight[:,:2]=0
    site=HookSite(model,'0','2','linear',between=('1',))
    x=torch.randn(6,3,dtype=torch.float64);targets=torch.randn(6,1,dtype=torch.float64)
    batch=(x,targets);spec=MSE();ctx=dict(targets=targets)
    opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    frozen=deepcopy(model);fixed=HookSite(frozen,'0','2','linear',between=('1',))
    original=deepcopy(frozen.state_dict());engine=V3Engine('vna_gir');engine.begin_task(1)
    with training_products(frozen,[fixed],seed_proposals([fixed]),x) as p:
        op=GGNProducts(p,targets,spec,ctx);G,_=gram(p)
        factor=OutputSolve(G,spec.metric_blocks(op.logits,ctx),.2)
        values=dict(e=op.g,e_beta=op.g,factor=factor)
    residuals=[];prepare=engine.prepare_gir
    def checked(m,*a,**kw):
        assert m is frozen
        _assert_bitwise(original,frozen.state_dict())
        residuals.append(kw['residual'].clone())
        return prepare(m,*a,**kw)
    monkeypatch.setattr(engine,'prepare_gir',checked)
    rows=engine._run_gir(model,[site],frozen,[fixed],batch,spec,ctx,opt,
        {site.module_name:[0,1]},1,values,.2)
    assert len(rows)==2 and all(row['Gamma_woodbury']>0 for row in rows)
    assert not torch.equal(residuals[0],residuals[1])
    _assert_bitwise(original,frozen.state_dict())


def test_divergence_guard_stops_before_sgd_and_records_location():
    from types import SimpleNamespace
    from experiments.main_cil import MainHook, Diverged
    hook=object.__new__(MainHook);hook.learner=SimpleNamespace(_cur_task=1)
    hook.current_epoch=2;hook.steps=2051;hook.diverged=None
    def native_frame():
        loss=torch.tensor(float('nan'))
        hook.before_optimizer_step(inspect.currentframe())
    with pytest.raises(Diverged):native_frame()
    assert hook.diverged==dict(task=2,epoch=3,step=2051,tensor='loss')


def test_exact_engine_resume_one_native_step_is_bitwise():
    from experiments.vna_resume import restore_rng
    model,site,batch,spec,ctx,opt=problem()
    engine=V3Engine('kernel_self');engine.begin_task(1)
    engine.apply(model,[site],batch,spec,ctx,opt,1,force=True)
    state=deepcopy((model.state_dict(),opt.state_dict(),engine.state_dict(),rng_state()))
    def advance():
        opt.zero_grad();spec.loss(model(batch[0]),ctx).backward();opt.step()
        engine.apply(model,[site],batch,spec,ctx,opt,2,force=True)
        torch.rand(3)
        params=deepcopy(model.state_dict());optim=deepcopy(opt.state_dict())
        # Wall-clock fields are observations rather than algorithm state.
        es=deepcopy(engine.state_dict())
        es.pop('algorithm_seconds')
        for event in es['events']:
            for key in tuple(event):
                if key.endswith('seconds'):event.pop(key)
            for record in event.get('solves',[]):record.pop('seconds',None)
        return params,optim,es,rng_state()
    expected=advance()
    model.load_state_dict(state[0]);opt.load_state_dict(state[1]);engine.load_state_dict(state[2]);restore_rng(state[3])
    _assert_bitwise(expected,advance())


def test_t31_adjacent_sites_remove_current_shared_columns():
    model=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,3),nn.ReLU(),nn.Linear(3,1)).double()
    first=HookSite(model,'0','2','linear',between=('1',))
    second=HookSite(model,'2','4','linear',between=('3',))
    x=torch.randn(3,2,dtype=torch.float64)
    with training_products(model,[],{},x) as p:
        _, ids1, outgoing=V3Engine._slot_columns(p,first,0)
        current={i:None for i in ids1.tolist()}
        added=torch.randn(3,len(outgoing),dtype=torch.float64)
        current.update(zip(outgoing.tolist(),added.T))
        Ju,ids2,_=V3Engine._slot_columns(p,second,1,current)
        native=torch.stack([p.Jv(nn.functional.one_hot(i,sum(v.numel() for _,v in p.parameter_refs)).double()) for i in ids2],1)
        for j,i in enumerate(ids2.tolist()):
            if i in current:native[:,j]=0. if current[i] is None else current[i]
        assert set(ids2.tolist()) & set(outgoing.tolist())
        torch.testing.assert_close(Ju,native,atol=1e-12,rtol=0)


def test_derivative_chunks_bound_vmap_width_and_preserve_exact_products(monkeypatch):
    from types import SimpleNamespace
    J=torch.randn(11,7,dtype=torch.float64)
    widths=[]; original=torch.vmap
    def tracked(action):
        mapped=original(action)
        def call(batch):
            widths.append(len(batch))
            return mapped(batch)
        return call
    monkeypatch.setattr(torch,'vmap',tracked)
    jv=lambda v:J@v
    jt=lambda v:J.T@v
    p=SimpleNamespace(_call=lambda _:torch.zeros(11,dtype=J.dtype),
                      parameter_refs=(('weights',torch.zeros(7,dtype=J.dtype)),),Jv=jv,JT=jt)
    G,info=gram(p)
    torch.testing.assert_close(G,J@J.T,rtol=1e-12,atol=1e-12)
    assert info['gram_chunk']==4 and max(widths)<=4
    widths.clear()
    result=columns(jv,torch.arange(7),7,p.parameter_refs[0][1])
    torch.testing.assert_close(result,J,rtol=1e-12,atol=1e-12)
    assert max(widths)<=4
