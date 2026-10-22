"""Native baseline identity and exact B5 resume after runner extraction."""
import copy
import random
from contextlib import nullcontext
import numpy as np
import pytest
import torch
from adapters.lop_hooks import LearnHook, learner_state
from adapters.vna_v3_hooks import LoPV3
from diagnostics.vna_align import MeasurementBuffer, _global_state, _assert_bitwise
from experiments.runners.lop import upstream_imports, run_native, recipe, prepare_data
from experiments.vna_resume import rng_state
from methods.vna_v3 import V3Engine

def fixture(benchmark, arm, path, tasks=2):
    params, meta = recipe(benchmark, arm, 1e-3 if benchmark == 'regression' and arm == 'cbp' else None)
    params['use_gpu'] = 0
    path.mkdir(parents=True, exist_ok=True)
    params['data_file'] = str(path/'logs.pkl'); params['run_idx'] = 0
    params['num_features'] = 5 if benchmark == 'regression' else 4
    if benchmark == 'regression': params.update(flip_after=160, num_data_points=tasks*160)
    elif benchmark == 'mnist': params.update(num_tasks=tasks, change_after=160, fixture_rank_padding=True)
    else: params.update(num_tasks=tasks, num_showings=1, mini_batch_size=100,
                        fake_train_images_per_class=100, fake_test_images_per_class=50, examples_per_epoch=200)
    prepare_data(benchmark, params, path, fake=True, smoke=True)
    return params

def reset(seed=13):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

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
        intervention=LoPV3(V3Engine(arm='vna',seed=13,K0=100))
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
    assert not hasattr(fp.engine,'warm') and not hasattr(rp.engine,'warm')
    for key in ('mu','mu_star','nu','A','K_next','last_iteration','applications','policy_version','gir_live'):
        _assert_bitwise(getattr(fp.engine,key),getattr(rp.engine,key))
    assert len(fp.task_metrics)==2

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

def test_cbp_short_buffer_keeps_native_initialization(monkeypatch):
    upstream_imports()
    from lop.algos.cbp import ContinualBackprop
    from lop.nets.ffnn import FFNN
    from types import SimpleNamespace
    learner=ContinualBackprop(FFNN(input_size=3,num_features=3).double(),
        replacement_rate=1.,maturity_threshold=0,device='cpu')
    engine=V3Engine(arm='cbp_gir')
    intervention=LoPV3(engine)
    hook=SimpleNamespace(step=0,task_steps=10,buffer=MeasurementBuffer(31))
    intervention.before(hook,learner,torch.ones(4,3,dtype=torch.float64),torch.zeros(4,1,dtype=torch.float64))
    monkeypatch.setattr(engine,'prepare_gir',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('prepare')))
    monkeypatch.setattr(engine,'install_gir',lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('install')))
    with intervention.install():
        learner.gnt.gen_and_test([torch.ones(4,3,dtype=torch.float64)])
    event=engine.events[-1]
    assert event['gir_fallback'] and event['swaps']==0
