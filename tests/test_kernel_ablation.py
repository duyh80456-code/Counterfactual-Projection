"""Fixed-window kernel algebra, isolation, provenance and replay comparisons."""
from copy import deepcopy
from types import ModuleType, SimpleNamespace
import json
import subprocess
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from diagnostics.vna_align import _assert_bitwise,_global_state
from diagnostics import kernel_ablation as a
from methods import vna_v3 as vna
from methods.vna_losses import ICaRLPyCIL
from methods.vna_ggn import GGNProducts
from methods.vna_sites import HookSite


@pytest.fixture(autouse=True)
def isolated():
    threads=torch.get_num_threads()
    with _global_state():
        torch.manual_seed(271);torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


class Net(nn.Module):
    def __init__(self,classes=3):
        super().__init__();self.layers=nn.Sequential(nn.Linear(2,4),nn.ReLU(),nn.Linear(4,classes))
    def forward(self,x):return dict(logits=self.layers(x))


class Prediction(nn.Module):
    def __init__(self,net):super().__init__();self.net=net
    def forward(self,x):return self.net(x)['logits']


def sites(model):return [HookSite(model,'net.layers.0','net.layers.2','linear',between=('net.layers.1',))]


def problem(dtype=torch.float64):
    net=Net().to(dtype=dtype);teacher=Net(2).to(dtype=dtype);teacher.eval()
    x=torch.randn(4,2,dtype=dtype);y=torch.tensor([0,1,2,2])
    opt=torch.optim.SGD(net.parameters(),lr=.05,momentum=.9)
    for p in net.parameters():p.grad=torch.ones_like(p);opt.state[p]['momentum_buffer']=torch.randn_like(p)
    learner=SimpleNamespace(_network=net,_old_network=teacher,_known_classes=2,_device=torch.device('cpu'),
        test_loader=DataLoader(TensorDataset(torch.arange(4),x,y),batch_size=2))
    def context(l,batch):
        return ICaRLPyCIL(known_classes=2),dict(targets=batch[1],known_classes=2,
            teacher_logits=l._old_network(batch[0])['logits'].detach())
    main=SimpleNamespace(Prediction=Prediction,sites_for=sites,loss_context=context)
    batch=(x,y);spec,ctx=context(learner,batch)
    return learner,opt,batch,spec,ctx,main


def test_native_kernel_replaces_only_KE_and_restores_products():
    learner,opt,batch,spec,ctx,main=problem();model=Prediction(learner._network)
    engine=a.KernelEngine('vna',kernel='kernel_self',solver='gram',curvature='power10');engine.begin_task(2)
    control=a.BaseEngine('vna',solver='gram',curvature='power10');control.begin_task(2)
    with vna.training_products(model,sites(model),vna.seed_proposals(sites(model)),batch[0]) as p:
        op=GGNProducts(p,batch[1],spec,ctx);G,_=vna.gram(p)
        original=p.KE
        p.KE=lambda z:G@z
        expected=control._direction(p,op,.3,G=G)
        del p.KE
        actual=engine._direction(p,op,.3,G=G)
        assert 'KE' not in vars(p)
        for key in ('dy','V','V_O','beta','g_dot_d'):
            if key not in actual:continue
            if isinstance(actual[key],torch.Tensor):torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
            else:assert actual[key]==expected[key]
        assert original.__func__==p.KE.__func__


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_random_kernel_has_reference_trace_per_site_and_rank(dtype):
    learner,opt,batch,spec,ctx,main=problem(dtype);model=Prediction(learner._network);ss=sites(model)
    engine=a.KernelEngine('vna',kernel='vna_rand',solver='gram',curvature='power10');engine.begin_task(2)
    with vna.training_products(model,ss,vna.seed_proposals(ss),batch[0]) as p:
        op=GGNProducts(p,batch[1],spec,ctx);values=engine._direction(p,op,.3,split_only=True)
    rng=a.cpu(a.rng_state());before=a.training_state(learner._network,opt)
    reference=super(a.KernelEngine,engine)._select(model,ss,batch,values)
    result=engine._select(model,ss,batch,values)
    assert result
    for site in ss:
        name=site.module_name
        assert result[name]['weight'].shape==reference[name]['weight'].shape
        assert result[name]['weight'].shape[0]<=16
        with vna.training_products(model,[site],{name:reference[name]},batch[0]) as p:target=engine._trace_extension(p)
        with vna.training_products(model,[site],{name:result[name]},batch[0]) as p:actual=engine._trace_extension(p)
        assert actual==pytest.approx(target,rel=1e-6 if dtype==torch.float32 else 1e-10,abs=1e-12)
    _assert_bitwise(rng,a.cpu(a.rng_state()));_assert_bitwise(before,a.training_state(learner._network,opt))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_vna_kernel_matches_original_6be_and_first_step_restore(monkeypatch,dtype):
    source=subprocess.check_output(['git','show',a.REFERENCE+':methods/vna_v3.py'],text=True)
    reference=ModuleType('reference_6be');exec(compile(source,'reference_6be','exec'),reference.__dict__)
    learner,opt,batch,spec,ctx,main=problem(dtype);model=Prediction(learner._network);ss=sites(model)
    engine=a.KernelEngine('vna',kernel='vna',solver='gram',curvature='power10');engine.begin_task(2)
    control=reference.V3Engine('vna',solver='gram',curvature='power10');control.begin_task(2)
    before=a.training_state(learner._network,opt)
    monkeypatch.setattr(vna.time,'perf_counter',lambda:0.)
    expected=control.apply(model,ss,batch,spec,ctx,opt,2001,force=True)
    expected_state=a.training_state(learner._network,opt)
    a.restore(learner._network,opt,None,before)
    got=engine.apply(model,ss,batch,spec,ctx,opt,2001,force=True)
    _assert_bitwise(a.cpu({k:got[k] for k in expected}),a.cpu(expected))
    _assert_bitwise(expected_state,a.training_state(learner._network,opt))
    first=dict(a.training_state(learner._network,opt),engine=a.cpu(engine.state_dict()))
    for _ in range(2):engine.apply(model,ss,batch,spec,ctx,opt,2001,force=True)
    a.restore(learner._network,opt,engine,first)
    _assert_bitwise(first,dict(a.training_state(learner._network,opt),engine=a.cpu(engine.state_dict())))


@pytest.mark.parametrize('kernel',a.ARMS)
def test_measurement_preserves_network_teacher_optimizer_engine_and_rng(kernel):
    learner,opt,batch,spec,ctx,main=problem()
    engine=a.KernelEngine('vna',kernel=kernel,solver='gram',curvature='power10');engine.begin_task(2)
    before=a.training_state(learner._network,opt);teacher=a.cpu(learner._old_network);state=a.cpu(engine.state_dict())
    held=(batch[0]*1.2,batch[1])
    result=a.observe(engine,learner,opt,batch,spec,ctx,held,main)
    assert result['lambda_max_ggn']>=0 and result['KD_independent'] is not None
    assert set(result['accuracy'])=={'old','new','total'}
    _assert_bitwise(before,a.training_state(learner._network,opt));_assert_bitwise(teacher,a.cpu(learner._old_network))
    _assert_bitwise(state,a.cpu(engine.state_dict()))


def source_state():
    protocol=dict(seed=0,arm='icarl_full',backbone='resnet32_nobn',class_order_kind='superclass',nobn_lr_scale=.5,
        policy_version=vna.POLICY_VERSION,gir_live=False,intervention_start_task=2,dataset_fingerprint={'train':'a','test':'b','meta':'c'},
        class_order=list(range(100)),code_commit=a.REFERENCE,cfg=dict(init_cls=5,increment=5,memory_size=2000,fixed_memory=False),
        training=dict(first_epochs=100,later_epochs=50,first_milestones=[30,60,85],later_milestones=[24,35]))
    identity=dict(protocol,arm='icarl')
    state=dict(identity=identity,completed=1,mid=False,epoch=100,hook=dict(steps=2000),
        learner={'_network':nn.Linear(2,3),'_data_memory':torch.zeros(2)},optimizer={},scheduler={},rng=a.rng_state())
    return state,protocol


def test_source_checkpoint_proof_and_amended_source_rejection():
    state,p=source_state();assert a.validate_source(state,p,0)['task1_baseline_bitwise']
    full=deepcopy(state);full['identity']['arm']='icarl_full'
    with pytest.raises(ValueError,match='bitwise proof'):a.validate_source(full,p,0)
    a.validate_source(full,p,0,state)
    full['learner']['_network'].weight.data.add_(1.)
    with pytest.raises(AssertionError):a.validate_source(full,p,0,state)
    state['identity']['gir_live_task1']=True
    with pytest.raises(ValueError,match='amended'):a.validate_source(state,p,0)


@pytest.mark.parametrize('key,value',[('epoch',99),('completed',2)])
def test_wrong_checkpoint_coordinates_rejected(key,value):
    state,p=source_state();state[key]=value
    with pytest.raises(ValueError,match='step2000'):a.validate_source(state,p,0)


def test_replay_compares_algorithmic_events_and_all_missing_rows():
    original=[dict(task=2,iteration=2001,s=1.,mu=.3,swaps=1,
        swap_records=[dict(site='a',unit=2,Gamma=.2,gamma_seconds=1.)])]
    same=deepcopy(original);same[0]['swap_records'][0]['gamma_seconds']=123.
    assert a.compare_events(original,same)==[]
    same[0]['mu']=.4
    assert a.compare_events(original,same)[0]['original']['mu']==.3
    assert a.compare_events(original,[])[0]['replay'] is None


def test_kernel_resume_identity():
    engine=a.KernelEngine(kernel='vna')
    with pytest.raises(ValueError,match='kernel resume'):a.KernelEngine(kernel='vna_rand').load_state_dict(engine.state_dict())


def test_native_kernel_is_not_disabled_by_empty_tiny_V(monkeypatch):
    learner,opt,batch,spec,ctx,main=problem();model=Prediction(learner._network)
    monkeypatch.setattr(a.BaseEngine,'_select',lambda *a,**kw:{})
    engine=a.KernelEngine(kernel='kernel_self')
    result=engine._select(model,sites(model),batch,{})
    assert result and all(p['weight'].shape[0]==16 for p in result.values())
    assert a.KernelEngine(kernel='vna')._select(model,sites(model),batch,{})=={}


def test_zero_random_trace_cannot_silently_break_normalization(monkeypatch):
    monkeypatch.setattr(a.BaseEngine,'_select',lambda *a,**kw:{'site':{}})
    engine=a.KernelEngine(kernel='vna_rand')
    monkeypatch.setattr(engine,'_randomize',lambda *a:{'site':dict(trace_target=1.,trace_before=0.)})
    with pytest.raises(ValueError,match='zero random kernel trace'):engine._select(None,None,None,None)


def test_discovery_excludes_own_five_task_continuations(tmp_path):
    state,p=source_state();p.update(expected_tasks=20,solver='gram',curvature='power10')
    original=tmp_path/'original';original.mkdir()
    (original/'protocol.json').write_text(json.dumps(p))
    (original/'interventions.json').write_text('[]');(original/'epochs.json').write_text('[]')
    checkpoint=tmp_path/'baseline'/'end_task1'/'latest.pt';checkpoint.parent.mkdir(parents=True)
    torch.save(state,checkpoint)
    own=tmp_path/'own';own.mkdir()
    (own/'protocol.json').write_text(json.dumps(dict(p,expected_tasks=5)))
    (own/'interventions.json').write_text('[]');(own/'epochs.json').write_text('[]')
    found=a.discover_inputs([tmp_path],0)
    assert found['original_run']==str(original) and found['checkpoint']==str(checkpoint)


def test_terminal_results_need_kernel_identity_and_complete_horizon():
    identity=dict(kernel='vna',seed=0)
    result=dict(status='COMPLETED',ablation_identity=identity,chain_completed=True,tasks=[dict(task=t) for t in range(1,6)])
    assert a.terminal_result(result,identity)
    with pytest.raises(ValueError,match='identity'):a.terminal_result(result,dict(identity,kernel='vna_rand'))
    result['chain_completed']=False
    with pytest.raises(ValueError,match='eight-step'):a.terminal_result(result,identity)
    result['chain_completed']=True
    result['tasks'].pop()
    with pytest.raises(ValueError,match='five-task'):a.terminal_result(result,identity)
    result.update(status='FAILED_DIVERGED',diverged=dict(task=4,epoch=1,step=5601,tensor='loss'))
    assert a.terminal_result(result,identity)


def test_explicit_legacy_import_preserves_solver_state_and_strict_resume():
    old=a.BaseEngine('vna',solver='gram',curvature='power10',K0=512)
    old.mu=.4;old.nu=4.;old.A={'site':dict(weight=torch.ones(1,2))}
    state=old.state_dict()
    engine=a.KernelEngine('vna',kernel='kernel_self',solver='gram',curvature='power10',K0=512)
    with pytest.raises(ValueError,match='kernel resume'):engine.load_state_dict(state)
    engine.import_legacy_state(state)
    imported=engine.state_dict();imported.pop('kernel_variant')
    _assert_bitwise(a.cpu(imported),a.cpu(state))
    with pytest.raises(ValueError,match='strict resume'):engine.import_legacy_state(engine.state_dict())


@pytest.mark.parametrize('norm,radius,valid,exceeds',[(.2,.3,True,False),(.4,.3,False,True),(0.,None,True,None),(.1,None,False,None)])
def test_trust_region_uses_actual_step_not_large_scalar(norm,radius,valid,exceeds):
    got=a.trust_audit(norm,radius)
    assert got['valid']==valid and got['exceeds']==exceeds


def test_norm_audit_excludes_subsequent_gir_swaps(monkeypatch):
    model=nn.Linear(1,1,bias=False).double()
    with torch.no_grad():model.weight.zero_()
    engine=a.KernelEngine('vna_gir',kernel='vna')
    def gir(self,model,*args,**kwargs):
        with torch.no_grad():model.weight.add_(10.)
        return []
    def apply(self,model,*args,**kwargs):
        frozen=deepcopy(model)
        with torch.no_grad():model.weight.add_(.1)
        self._run_gir(model,[],frozen)
        return dict(Delta=.2,s=1.)
    monkeypatch.setattr(a.BaseEngine,'_run_gir',gir);monkeypatch.setattr(a.BaseEngine,'apply',apply)
    event=engine.apply(model,[],None,None,{},torch.optim.SGD(model.parameters(),lr=.05),2001)
    assert model.weight.item()==pytest.approx(10.1)
    assert event['trust_region_audit']['step_norm_actual']==pytest.approx(.1)
    assert event['trust_region_audit']['valid'] and '_audit_step_norm' not in vars(engine)


def test_kernel_diagnostics_use_correct_operator_and_do_not_certify_power10():
    event=dict(V=6.,V_O=2.,B=3.,lambda_E=2.,ratio=1.,lambda_max_ggn_before=4.,lambda_max_ggn_after=8.)
    native=a.kernel_diagnostics(event,'kernel_self',True)
    virtual=a.kernel_diagnostics(event,'vna',False)
    assert native['lambda_kernel_pre']==4. and native['ratio_estimate']==.5
    assert native['legacy_lambda_E']==2. and native['after_stage']=='after_VNA_and_GIR'
    assert not native['legacy_ratio_matches_kernel'] and virtual['legacy_ratio_matches_kernel']
    assert virtual['lambda_kernel_pre']==2. and virtual['ratio_estimate']==1.
    assert not native['bound_certified'] and not virtual['bound_certified']
    assert not native['rank_matched'] and native['V_pre']==event['V']


def test_read_only_apply_leaves_no_audit_or_training_mutation():
    learner,opt,batch,spec,ctx,main=problem();model=Prediction(learner._network)
    engine=a.KernelEngine('vna',solver='gram',curvature='power10');engine.begin_task(2)
    before=a.training_state(learner._network,opt);state=a.cpu(engine.state_dict())
    engine.apply(model,sites(model),batch,spec,ctx,opt,2001,force=True,apply=False)
    _assert_bitwise(before,a.training_state(learner._network,opt));_assert_bitwise(state,a.cpu(engine.state_dict()))
    assert '_audit_step_norm' not in vars(engine) and '_audit_seconds' not in vars(engine)


def test_norm_audit_overhead_is_excluded_from_budget_guard():
    engine=a.KernelEngine();engine._audit_seconds=3.
    engine.protocol.update(baseline_iteration_seconds=1.,budget=1.)
    row=dict(seconds=12.,gir_seconds=5.,K_next=16)
    assert engine.budget_guard(row,10)==16
    assert row['seconds']==9. and row['gir_seconds']==2. and not row['budget_guard']
