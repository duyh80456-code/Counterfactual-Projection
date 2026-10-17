"""Native v3 lifecycle: no-op identities, exact resume, and windows."""
import copy
from contextlib import nullcontext
import numpy as np
import pytest
import torch
from adapters.lop_hooks import LearnHook, learner_state
from adapters.vna_v3_hooks import LoPV3, ICaRLV3
from diagnostics.vna_align import MeasurementBuffer, _global_state, _assert_bitwise
from experiments.stage_b5 import upstream_imports, run_native
from experiments.vna_resume import rng_state
from methods.vna_v3 import V3Engine
from tests.test_vna_stage_b5 import fixture, reset


@pytest.fixture(autouse=True)
def single_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    with _global_state():yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('base',['bp','cbp'])
def test_zero_coefficient_native_bitwise(base,tmp_path):
    upstream_imports();params=fixture('regression',base,tmp_path)
    if base=='cbp':params['replacement_rate']=.1
    states=[]
    for enabled in (False,True):
        reset()
        intervention=LoPV3(V3Engine(arm='cbp_vna_gir' if base=='cbp' else 'vna_gir',coefficient=0.))
        hook=LearnHook('regression',base,params,intervention=intervention if enabled else None)
        hook.buffer=MeasurementBuffer(128)
        with intervention.install() if enabled else nullcontext():
            logs=run_native('regression',params,tmp_path,hook)
        states.append(dict(logs=logs,learner=learner_state(hook.last_learner),rng=rng_state()))
    _assert_bitwise(*states)


def test_active_resume_preserves_fifo_and_solver_state(tmp_path):
    upstream_imports();params=fixture('regression','bp',tmp_path)
    def make(resume=None,checkpoint=None,observe=None):
        intervention=LoPV3(V3Engine(arm='vna',seed=13,K_base=100))
        hook=LearnHook('regression','bp',params,intervention=intervention,resume=resume,
                       checkpoint=checkpoint,observe=observe)
        hook.buffer=MeasurementBuffer(128)
        return hook,intervention
    reset();full,fp=make()
    with fp.install():expected=run_native('regression',params,tmp_path,full)
    state={}
    class Pause(Exception):pass
    def checkpoint(value):
        if value['step']==200:
            state.update(copy.deepcopy(value));raise Pause
        return 0
    def observe(learner,x,y,task,step):
        if step==200:partial.last_checkpoint=float('-inf')
    reset();partial,pp=make(checkpoint=checkpoint,observe=observe)
    with pp.install(),pytest.raises(Pause):run_native('regression',params,tmp_path,partial)
    reset();resumed,rp=make(resume=state)
    with rp.install():actual=run_native('regression',params,tmp_path,resumed)
    _assert_bitwise(expected,actual)
    _assert_bitwise(learner_state(full.last_learner),learner_state(resumed.last_learner))
    _assert_bitwise(list(full.buffer.samples),list(resumed.buffer.samples))
    for key in ('warm','A','K_next','last_iteration','applications'):
        _assert_bitwise(getattr(fp.engine,key),getattr(rp.engine,key))
    assert len(fp.task_metrics)==2


def test_icarl_zero_coefficient_native_bitwise(monkeypatch):
    from experiments.stage_b3 import upstream_imports
    from adapters.icarl_hooks import ICaRLHook
    from tests import test_vna_stage_b3 as native
    upstream_imports();results=[]
    for enabled in (False,True):
        torch.manual_seed(0);np.random.seed(0)
        with monkeypatch.context() as patch:
            if enabled:
                patch.setattr(native,'ICaRLHook',lambda *a,**kw:ICaRLHook(*a,**kw,
                    intervention=ICaRLV3(V3Engine(arm='vna',coefficient=0.))))
            result,_=native.tiny_run(patch,True)
            results.append(dict(state=result,rng=rng_state()))
    _assert_bitwise(*results)


def test_mnist_horizon_architecture_and_fixed_cbp_rate():
    from experiments.pilot_v3 import parser,lop_recipe
    args=parser().parse_args(['--setting','mnist','--arm','pm_bp','--out','unused'])
    _,params,_=lop_recipe(args)
    assert (params['num_tasks'],params['num_hidden_layers'],params['num_features'])==(800,3,10)
    args=parser().parse_args(['--setting','regression','--arm','b5_cbp','--out','unused'])
    _,params,_=lop_recipe(args)
    assert params['replacement_rate']==1e-3


def test_cbp_selection_timing_and_native_rng_preserved(monkeypatch):
    upstream_imports()
    from lop.algos.cbp import ContinualBackprop
    from lop.nets.ffnn import FFNN
    from types import SimpleNamespace
    selections=[]
    for arm in ('baseline','cbp_gir','cbp_gradmax'):
        reset()
        learner=ContinualBackprop(FFNN(input_size=3,num_features=3).double(),
            replacement_rate=1.,maturity_threshold=0,device='cpu')
        engine=V3Engine(arm=arm)
        intervention=LoPV3(engine)
        hook=SimpleNamespace(step=0,task_steps=10,buffer=MeasurementBuffer(128))
        intervention.before(hook,learner,torch.ones(4,3,dtype=torch.float64),torch.zeros(4,1,dtype=torch.float64))
        chosen=[]
        original=learner.gnt.test_features
        def choose(features):
            result=original(features)
            chosen.append([u.clone() for u in result[0]])
            return result
        monkeypatch.setattr(learner.gnt,'test_features',choose)
        def proposal(model,site,batch,spec,ctx,optimizer,units,iteration,gradmax=False):
            assert iteration==1
            assert gradmax==(arm=='cbp_gradmax')
            return [{'A':{'weight':torch.ones_like(site.first_layer.layer.weight[:1]),
                          'bias':torch.zeros_like(site.first_layer.layer.bias[:1])},'score':1.} for _ in units]
        monkeypatch.setattr(engine,'prepare_gir',proposal)
        with intervention.install():
            learner.gnt.gen_and_test([torch.ones(4,3,dtype=torch.float64)])
        selections.append(dict(selected=chosen,rng=rng_state()))
    _assert_bitwise(selections[0],selections[1]);_assert_bitwise(selections[0],selections[2])


def test_online_ace_residual_and_curvature_match_native_loss():
    from experiments.pilot_v3_online import OnlineCE
    spec=OnlineCE()
    z=torch.randn(5,4,dtype=torch.float64,requires_grad=True)
    target=torch.tensor([0,1,2,3,0])
    ctx=dict(targets=target,stream_size=2,
             stream_mask=torch.tensor([[True,True,False,False],[True,True,False,False]]))
    native=torch.nn.functional.cross_entropy(z[:2].masked_fill(~ctx['stream_mask'],-1e9),target[:2])+torch.nn.functional.cross_entropy(z[2:],target[2:])
    gradient=torch.autograd.grad(native,z,create_graph=True)[0]
    vector=torch.randn_like(z)
    hessian=torch.autograd.grad(gradient,z,vector)[0]
    torch.testing.assert_close(spec.residual(z,ctx),gradient)
    torch.testing.assert_close(spec.metric_apply(z,vector,ctx),hessian)


def test_current_cil_minibatch_and_cross_task_fifo(monkeypatch):
    from types import SimpleNamespace
    import adapters.vna_v3_hooks as hooks
    from experiments.stage_b3 import upstream_imports as cil_imports
    cil_imports()
    class Network(torch.nn.Linear):
        def forward(self,x):return {'logits':super().forward(x)}
    engine=V3Engine(arm='vna');engine.begin_task(1)
    adapter=ICaRLV3(engine)
    batch=(torch.randn(7,3),torch.arange(7)%2)
    learner=SimpleNamespace(_network=Network(3,2),_known_classes=0,_total_classes=2,
        _cur_task=0,train_loader=range(10))
    hook=SimpleNamespace(current_batch=batch,steps=1,current_epoch=0,epoch_step=1)
    captured=[]
    monkeypatch.setattr(hooks,'cil_sites',lambda model:[])
    monkeypatch.setattr(engine,'apply',lambda model,sites,actual,*a,**kw:captured.append(actual))
    adapter.after_step(hook,learner,torch.optim.SGD(learner._network.parameters(),lr=.1))
    assert captured[0][0].data_ptr()==batch[0].data_ptr()
    assert captured[0][1].data_ptr()==batch[1].data_ptr()
    online=LoPV3(V3Engine(coefficient=0.))
    fifo_hook=SimpleNamespace(step=0,task_steps=128,buffer=MeasurementBuffer(128))
    online.before(fifo_hook,None,torch.ones(128,3),torch.ones(128,1))
    fifo_hook.step=128
    online.before(fifo_hook,None,torch.zeros(1,3),torch.zeros(1,1))
    x,y=fifo_hook.buffer.batch()
    assert len(x)==128 and torch.equal(x[0],torch.ones(3)) and torch.equal(x[-1],torch.zeros(3))
    assert online.engine.task==2


def test_preflight_selects_common_mu_and_selected_trial_time(tmp_path):
    from types import SimpleNamespace
    from experiments.pilot_v3 import parser,finalize_calibration
    args=parser().parse_args(['--setting','icarl','--arm','icarl_vna','--out',str(tmp_path),'--fake-data'])
    points=[dict(task=task,classes=10*task,vna_seconds=99.,iteration=i,gir_seconds=0.)
            for task in (1,2) for i in (1,2,3)]
    reports=[dict(trials=[dict(mu_rel=mu,Delta_hat=1. if mu==.001 else 0.,
        g_dot_d=1.,gradient_norm=1.,seconds=seconds) for mu,seconds in ((.001,30.),(.01,2.),(.1,1.))]) for _ in points]
    adapter=SimpleNamespace(calibration=points,mu_trials=reports,iteration_times=[.1]*200,full_batches=[40,55])
    engine=V3Engine()
    schedule=finalize_calibration(args,engine,adapter,tmp_path)
    assert engine.mu_rel==.01 and schedule['mu_condition_met']
    assert schedule['t_vna_task2']==[2.]
    assert schedule['projected_vna_seconds']==10.
    assert schedule['K_base']==100 and schedule['iteration_samples']==200
    assert schedule['calibration'][0]['actual_application_seconds']==99.
