"""T32 functional certificates, native slot transactions and suffix equivalence."""
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.func import functional_call

from diagnostics.vna_align import _assert_bitwise
from experiments.vna_resume import rng_state, restore_rng
from methods import vna_certswap as certswap
from methods.vna_losses import CE, ICaRLPyCIL, MSE
from methods.vna_sites import HookSite
from methods.vna_v3 import POLICY_VERSION, V3Engine


@pytest.fixture(autouse=True)
def isolated():
    threads=torch.get_num_threads()
    with torch.random.fork_rng():
        torch.manual_seed(32024)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        from adapters.cil_setup import resnet32_nobn
        self.convnet=resnet32_nobn()
        del self.convnet.fc
        self.fc=nn.Linear(self.convnet.out_dim,3)

    def forward(self,x):
        return {'logits':self.fc(self.convnet(x)['features'])}


def problem(device='cpu',dtype=torch.float64):
    from experiments.runners.pycil import upstream_imports
    from adapters.icarl_hooks import Prediction,sites_for
    from adapters.cil_setup import SkipScale
    upstream_imports()
    model=Prediction(Native()).to(device=device,dtype=dtype).train()
    assert len(sites_for(model))==15
    assert sum(isinstance(module,nn.Conv2d) for module in model.modules())==31
    assert sum(p.numel() for p in model.parameters())>460000
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module,nn.Conv2d):
                module.weight.zero_();module.bias.zero_()
            elif isinstance(module,SkipScale):module.alpha.fill_(1.)
        model.native.convnet.conv_1_3x3.weight[0,0,1,1]=1.
        for site in sites_for(model):
            site.first_layer.layer.bias[0]=-1.
            site.first_layer.layer.bias[1:]=1.
        model.native.fc.weight.zero_()
        model.native.fc.weight[:,0]=torch.tensor([1.,-1.,0.],device=device,dtype=dtype)
        model.native.fc.bias.zero_()
    sites=[site for site in sites_for(model) if '.4.conv_a' in site.first]
    values=torch.tensor([.25,.75,1.25,1.75],device=device,dtype=dtype)
    inputs=values[:,None,None,None].expand(-1,3,32,32).clone()
    targets=torch.ones(4,device=device,dtype=torch.long)
    ctx=dict(targets=targets,known_classes=2,teacher_logits=torch.zeros(4,2,device=device,dtype=dtype))
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    for parameter in model.parameters():
        optimizer.state[parameter].update(momentum_buffer=torch.ones_like(parameter),
            exp_avg=torch.full_like(parameter,2.),step=torch.tensor(7.))
    proposals={}
    for site in sites:
        weight=torch.zeros_like(site.first_layer.layer.weight[:1]);weight[0,0,1,1]=1.
        proposals[site.module_name]=dict(weight=weight,bias=weight.new_tensor([-.5]))
    proposals[sites[1].module_name]['weight'].neg_()
    proposals[sites[1].module_name]['bias'].fill_(.9)
    return model,sites,(inputs,targets),ICaRLPyCIL(T=2,known_classes=2),ctx,optimizer,proposals


def slot_forward(model,site,inputs,proposal):
    cache=certswap.PrefixCache(model,[site],inputs)
    first,second=site.first_layer.layer,site.second_layer.layer
    incoming=first.weight.detach().clone();incoming[0]=proposal['weight'][0]
    bias=first.bias.detach().clone()
    bias[0]=proposal['bias'][0]
    outgoing=second.weight.detach().clone();outgoing[:,0]=0
    index=torch.tensor([0],device=outgoing.device)
    def forward(vector):
        replacement=outgoing.index_copy(1,index,vector.reshape(outgoing.shape[0],1,*outgoing.shape[2:]))
        return cache.forward(site,{'conv_a.weight':incoming,'conv_a.bias':bias,'conv_b.weight':replacement})
    return cache,forward,outgoing


def test_t32_dead_slot_exact_zero_jump_and_functional_trials_do_not_mutate():
    model,sites,batch,spec,ctx,opt,proposals=problem()
    cache,forward,reference=slot_forward(model,sites[-1],batch[0],proposals[sites[-1].module_name])
    state=deepcopy((model.state_dict(),opt.state_dict(),rng_state()))
    size=reference[:,0].numel()
    swapped=forward(reference.new_zeros(size))
    assert torch.equal(swapped,cache.logits)
    before=certswap.losses(cache.forward(sites[-1],None),spec,ctx)
    assert certswap.losses(swapped,spec,ctx)==before
    result,record=certswap.certify(forward,size,reference,before,spec,ctx,mu0=10.,budget=20)
    assert record['jL']==record['jKD']==0.
    assert result is not None and record['rho']>=.25 and record['pred_B']>0
    assert len(record['lm_trials'])<=20
    _assert_bitwise(state,(model.state_dict(),opt.state_dict(),rng_state()))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cached_suffix_matches_full_forward_after_deep_then_shallow_edit():
    model,sites,batch,_,_,_,_=problem()
    from adapters.icarl_hooks import sites_for
    sites=sites_for(model)
    cache=certswap.PrefixCache(model,sites,batch[0])
    for site in sites:
        assert torch.equal(cache.forward(site),model(batch[0]))
    shallow,deep=sites[0],sites[-1]
    block_path=deep.first.rsplit('.',1)[0]
    replacement=deep.second_layer.layer.weight.detach().clone();replacement[:,1]=.125
    predicted=cache.forward(deep,{'conv_b.weight':replacement})
    full=functional_call(model,{f'{block_path}.conv_b.weight':replacement},(batch[0],),strict=False)
    assert torch.equal(predicted,full)
    with torch.no_grad():deep.second_layer.layer.weight.copy_(replacement)
    captured=[]
    handle=model.get_submodule(shallow.first.rsplit('.',1)[0]).register_forward_pre_hook(
        lambda _m,args:captured.append(args[0].detach().clone()))
    try:current=model(batch[0])
    finally:handle.remove()
    assert torch.equal(captured[0],cache.inputs[shallow.module_name])
    assert torch.equal(cache.forward(shallow),current)


def test_t32_certified_sequence_measures_current_ce_and_preswap_kd():
    model,sites,batch,spec,ctx,opt,proposals=problem()
    engine=V3Engine('vna_gir',gir_rule='certswap',certswap_topk=1,certswap_log_schur=False)
    engine.begin_task(2)
    initial=certswap.losses(model(batch[0]),spec,ctx)
    state=deepcopy(opt.state_dict())
    report=certswap.run(engine,model,sites,proposals,batch,spec,ctx,opt,11,{})
    assert len(report['accepted'])>=2
    indices={site.module_name:i for i,site in enumerate(sites)}
    assert [indices[r['site']] for r in report['sites']]==[2,1,0]
    for trial in report['trials']:
        assert trial['activation_fraction']==(.5 if indices[trial['site']]==1 else .75)
    current=initial
    masks={parameter:torch.zeros_like(parameter,dtype=torch.bool) for parameter in model.parameters()}
    for row in report['accepted']:
        assert row['jL']<=0 and row['jKD']<=0 and row['pred_B']>0 and row['rho']>=.25
        assert row['L_before']==pytest.approx(current[0],abs=1e-13)
        assert row['KD_before']==pytest.approx(current[1],abs=1e-13)
        assert row['L_after']<=row['L_before']-.25*row['pred_B']+1e-13
        assert row['L_after']<row['L_before'] and row['KD_after']<=row['KD_before']
        assert row['schur_gain'] is None
        current=row['L_after'],row['KD_after']
        site=next(s for s in sites if s.module_name==row['site']);unit=row['unit']
        assert torch.equal(site.first_layer.layer.weight[unit],proposals[site.module_name]['weight'][0])
        assert torch.equal(site.first_layer.layer.bias[unit],proposals[site.module_name]['bias'][0])
        assert row['activation_fraction']==(.5 if site is sites[1] else .75)
        masks[site.first_layer.layer.weight][unit]=True
        masks[site.first_layer.layer.bias][unit]=True
        masks[site.second_layer.layer.weight][:,unit]=True
    assert certswap.losses(model(batch[0]),spec,ctx)==pytest.approx(current,abs=1e-13)
    for index,parameter in enumerate(model.parameters()):
        for key in ('momentum_buffer','exp_avg'):
            value=opt.state[parameter][key];old=state['state'][index][key]
            assert torch.count_nonzero(value[masks[parameter]])==0
            assert torch.equal(value[~masks[parameter]],old[~masks[parameter]])
        assert torch.equal(opt.state[parameter]['step'],state['state'][index]['step'])


def test_t32_rejected_live_swap_preserves_model_optimizer_and_rng(monkeypatch):
    model,sites,(inputs,_),_,_,opt,proposals=problem()
    site=sites[-1]
    with torch.no_grad():
        site.first_layer.layer.bias.fill_(1.)
        site.second_layer.layer.weight[:,0]=1.
    targets=torch.zeros(len(inputs),dtype=torch.long);ctx=dict(targets=targets)
    engine=V3Engine('vna_gir',gir_rule='certswap',certswap_topk=1,certswap_log_schur=False)
    engine.begin_task(2)
    before=deepcopy((model.state_dict(),opt.state_dict(),rng_state()))
    def forbidden(*args,**kwargs):raise AssertionError('curvature computed after positive measured jump')
    monkeypatch.setattr(torch.linalg,'eigh',forbidden)
    report=certswap.run(engine,model,[site],proposals,(inputs,targets),CE(),ctx,opt,1,{})
    assert not report['accepted'] and report['rejections']['jL_pos']==1
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state()))
    assert not engine.install_ages


def test_t32_negative_control_positive_ce_prediction_can_increase_kd():
    logits=torch.tensor([[1.,-1.,0.]],dtype=torch.float64)
    direction=torch.tensor([[1.,-1.,0.]],dtype=torch.float64)
    ctx=dict(targets=torch.tensor([2]),known_classes=2,teacher_logits=logits[:,:2].clone())
    spec=ICaRLPyCIL(T=2,known_classes=2)
    def forward(vector):return logits+direction*vector[0]
    zero=torch.zeros(1,dtype=torch.float64,requires_grad=True)
    gradient=torch.autograd.grad(nn.functional.cross_entropy(forward(zero),ctx['targets']),zero)[0]
    metric=CE().metric_blocks(logits,ctx)[0]
    F=(direction@metric@direction.T).reshape(1,1)
    step=torch.linalg.solve(F+.1*torch.eye(1,dtype=torch.float64),gradient)
    pred=float(gradient@step-.5*step@F@step)
    before=certswap.losses(logits,spec,ctx)
    predicted_only=certswap.losses(forward(-step),spec,ctx)
    assert pred>0 and predicted_only[0]<before[0] and predicted_only[1]>before[1]
    result,record=certswap.certify(forward,1,logits,before,spec,ctx,mu0=.1,budget=1)
    assert result is None and record['reason']=='lm_budget' and record['kd_fail']


def test_t32_zero_gradient_rejects_before_curvature_and_eigen_matches_solve(monkeypatch):
    logits=torch.zeros(2,3,dtype=torch.float64)
    ctx=dict(targets=torch.tensor([0,1]))
    with monkeypatch.context() as patch:
        patch.setattr(torch.linalg,'eigh',lambda *a,**k:pytest.fail('zero gradient computed eigendecomposition'))
        result,record=certswap.certify(lambda v:logits+0*v.sum(),1,logits,
            certswap.losses(logits,CE(),ctx),CE(),ctx)
    assert result is None and record['reason']=='g_zero'
    matrix=torch.randn(7,7,dtype=torch.float64)
    F=matrix.T@matrix
    eigenvalues,eigenvectors=torch.linalg.eigh(F)
    gradient=torch.randn(7,dtype=torch.float64)
    for mu in (.01,.1,1.,100.):
        actual=certswap.eigen_step(eigenvalues,eigenvectors,gradient,mu)
        expected=torch.linalg.solve(F+mu*torch.eye(7,dtype=torch.float64),gradient)
        torch.testing.assert_close(actual,expected,rtol=0,atol=1e-10)


def test_old_default_engine_has_identical_state_and_trajectory(monkeypatch):
    monkeypatch.setattr('time.perf_counter',lambda:0.)
    monkeypatch.setattr('time.monotonic',lambda:0.)
    monkeypatch.setattr(certswap,'run',lambda *a,**k:pytest.fail('default policy entered CertSwap'))
    left=V3Engine('vna_gir',solver='gram',curvature='exact')
    right=V3Engine('vna_gir',solver='gram',curvature='exact',gir_rule='schur_dead')
    assert left.policy_version==right.policy_version==POLICY_VERSION
    assert not any(k.startswith('certswap') or k=='gir_rule' for k in left.state_dict())
    _assert_bitwise(left.state_dict(),right.state_dict())
    with pytest.raises(ValueError,match='policy'):
        V3Engine('vna_gir',gir_rule='certswap').load_state_dict(left.state_dict())
    model=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,1)).double()
    with torch.no_grad():model[0].weight[0]=0;model[0].bias[0]=-1;model[2].weight[:,0]=0
    batch=torch.randn(4,2,dtype=torch.float64),torch.randn(4,1,dtype=torch.float64)
    initial=deepcopy(model.state_dict());rng=rng_state();outcomes=[]
    for engine in (left,right):
        model.load_state_dict(initial);restore_rng(rng)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        history=[]
        for _ in range(2):
            optimizer.zero_grad();loss=nn.functional.mse_loss(model(batch[0]),batch[1])
            history.append(float(loss.detach()));loss.backward();optimizer.step()
        site=HookSite(model,'0','2','linear',between=('1',))
        engine.begin_task(2)
        row=engine.apply(model,[site],batch,MSE(),dict(targets=batch[1]),optimizer,2,force=True)
        assert 'certswap' not in row
        outcomes.append(deepcopy((model.state_dict(),optimizer.state_dict(),engine.state_dict(),history,row,rng_state())))
    _assert_bitwise(*outcomes)


def test_optin_engine_calls_certswap_on_post_vna_network(monkeypatch):
    model=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,1)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float64))
        model[0].bias.fill_(1.)
        model[2].weight.copy_(torch.tensor([[.2,.3,.5]],dtype=torch.float64));model[2].bias.fill_(.1)
    inputs=torch.tensor([[.1,.2],[.2,.4],[.3,.6],[.4,.8]],dtype=torch.float64)
    targets=torch.tensor([[-.2],[-.4],[-.6],[-.8]],dtype=torch.float64)
    batch=inputs,targets;spec=MSE();ctx=dict(targets=targets)
    site=HookSite(model,'0','2','linear',between=('1',))
    optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    engine=V3Engine('vna_gir',solver='gram',curvature='exact',gir_rule='certswap')
    engine.begin_task(2)
    observed=[]
    class MeasuredCache:
        def __init__(self,current_model,current_sites,current_inputs,measure=None):
            assert current_model is model and measure is not None
            self.losses=measure()
    def measured(current_engine,current_model,sites,A,current_batch,current_spec,current_ctx,opt,iteration,values):
        assert current_engine is engine and current_model is model and opt is optimizer
        assert current_batch is batch
        observed.append(float(spec.loss(model(inputs),ctx).detach()))
        return dict(sites=[],accepted=[],rejections={},seconds=0.)
    monkeypatch.setattr(certswap,'PrefixCache',MeasuredCache)
    monkeypatch.setattr(certswap,'run',measured)
    monkeypatch.setattr(engine,'_run_gir',lambda *a,**k:pytest.fail('CertSwap entered the old GIR decision path'))
    row=engine.apply(model,[site],batch,spec,ctx,optimizer,1,force=True)
    assert not row['skipped'] and observed==pytest.approx([row['L_after']],abs=1e-13)
    assert row['certswap']['accepted']==[] and row['certswap_seconds']==0.
    assert 'vna_seconds' in row and engine.policy_version=='certswap_t32_v1'


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires a CUDA GPU; CPU is not a substitute')
def test_cuda_real_resnet32_dead_jump_and_suffix_are_bitwise(dtype):
    model,sites,batch,spec,ctx,opt,proposals=problem('cuda:0',dtype)
    from adapters.icarl_hooks import sites_for
    assert len(sites_for(model))==15
    previous=torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    try:
        cache=certswap.PrefixCache(model,sites,batch[0])
        with torch.no_grad():
            full=model(batch[0]).double()
            for site in sites:
                before=cache.forward(site,None)
                assert torch.equal(before,full)
                _,forward,reference=slot_forward(model,site,batch[0],proposals[site.module_name])
                swapped=forward(reference.new_zeros(reference[:,0].numel()))
                assert torch.equal(before,swapped)
                old=certswap.losses(before,spec,ctx);new=certswap.losses(swapped,spec,ctx)
                assert new[0]-old[0]==0. and new[1]-old[1]==0.
    finally:
        torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark=previous


def test_each_site_measures_before_on_its_own_suffix_even_with_cached_full_logits(monkeypatch):
    model,sites,batch,spec,ctx,opt,proposals=problem()
    cache=certswap.PrefixCache(model,sites,batch[0])
    cache.logits=torch.full_like(cache.logits,123.)
    engine=V3Engine('vna_gir',gir_rule='certswap',certswap_topk=1,certswap_log_schur=False)
    engine.begin_task(2)
    seen=[]
    def observe(forward,size,reference,before,spec,ctx,**kwargs):
        site=sites[len(sites)-1-len(seen)]
        with torch.no_grad():expected=certswap.losses(cache.forward(site,None),spec,ctx)
        assert before==expected
        seen.append(site.module_name)
        return None,dict(reason='g_zero',lm_trials=[])
    monkeypatch.setattr(certswap,'certify',observe)
    certswap.run(engine,model,sites,proposals,batch,spec,ctx,opt,1,{'certswap_cache':cache})
    assert seen==[site.module_name for site in reversed(sites)]
