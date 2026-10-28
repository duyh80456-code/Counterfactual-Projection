"""Focused benchmark wiring and restart checks; no real datasets or GPU."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from diagnostics.vna_align import _global_state,_assert_bitwise


@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    import importlib.util,sys
    if sys.version_info>=(3,11) and importlib.util.find_spec('avalanche'):
        # Local Py3.12 cannot import typing4.4; emulate only its no-op Protocol constructors.
        from avalanche.training.templates.strategy_mixin_protocol import SupervisedStrategyProtocol,SGDStrategyProtocol
        for protocol in (SupervisedStrategyProtocol,SGDStrategyProtocol):
            monkeypatch.setattr(protocol,'__init__',lambda self:None,raising=False)
    threads=torch.get_num_threads();torch.set_num_threads(1)
    with _global_state():yield
    torch.set_num_threads(threads)


def test_protocols_match_upstream_and_domain_table():
    from experiments.runners import lop
    h2=json.loads(Path('configs/standard_cl/h2_online_cifar100.json').read_text())
    assert (h2['tasks'],h2['train_epochs'],h2['train_mb_size'],h2['memory_size'])==(20,3,10,2000)
    assert (h2['lr'],h2['momentum'],h2['weight_decay'],h2['validation_fraction'])==(.1,0,0,.05)
    cfg=json.loads(Path('configs/standard_cl/h3_imagenet.json').read_text())
    params=json.loads((lop.UPSTREAM/cfg['upstream_config']).read_text())
    assert (params['num_tasks'],params['num_showings'],params['num_classes'],params['mini_batch_size'])==(5000,250,2,100)
    assert (params['step_size'],params['momentum'],params['replacement_rate'])==(.01,.9,3e-4)
    from methods.cl_h4 import DomainMNIST
    from torch.utils.data import TensorDataset
    h4=json.loads(Path('configs/standard_cl/h4_rotated_mnist.json').read_text())
    base=TensorDataset(torch.zeros(1,1,28,28),torch.zeros(1,dtype=torch.long))
    assert [DomainMNIST(base,d,'rotated',angle_step=h4['angle_step']).identity()['angle_degrees']
            for d in range(h4['domains'])]==list(range(0,300,15))


def test_imagenet_split_uses_original_arrays_without_normalizing(tmp_path):
    from experiments.standard_imagenet import load_task
    for c in (7,3):np.save(tmp_path/f'{c}.npy',np.full((700,3,32,32),c,dtype=np.uint8))
    x,y,tx,ty=load_task(tmp_path,[7,3])
    assert x.shape==(1200,3,32,32) and tx.shape==(200,3,32,32)
    assert x.dtype==torch.float32 and float(x[0].mean())==7 and float(tx[-1].mean())==3
    assert y.tolist()==[0]*600+[1]*600 and ty.tolist()==[0]*100+[1]*100
    np.save(tmp_path/'3.npy',np.zeros((699,3,32,32),dtype=np.uint8))
    with pytest.raises(ValueError,match='unexpected shape'):load_task(tmp_path,[7,3])


@pytest.mark.parametrize('arm',['cbp','bp','vna','gir','vna_gir'])
def test_imagenet_native_epochs_resume_and_full_hook_order(tmp_path,monkeypatch,arm):
    from experiments import standard_imagenet as runner
    from experiments.runners import lop
    from methods.cl_replay import ReplayCorrection
    from methods.cl_h3 import TangentReconfiguration
    lop.upstream_imports()
    upstream=tmp_path/'native';(upstream/'lop/imagenet/cfg').mkdir(parents=True)
    params=json.loads((lop.UPSTREAM/'lop/imagenet/cfg/cbp.json').read_text())
    params.update(mini_batch_size=2)
    (upstream/'lop/imagenet/cfg/cbp.json').write_text(json.dumps(params))
    with (upstream/'lop/imagenet/class_order').open('wb') as stream:pickle.dump([np.arange(1000)],stream)
    monkeypatch.setattr(lop,'UPSTREAM',upstream);monkeypatch.setattr(lop,'upstream_imports',lambda:None)
    monkeypatch.setattr(runner,'data_identity',lambda *args:{'fixture':'local synthetic only'})
    data=tmp_path/'classes';data.mkdir()
    for c in range(4):np.save(data/f'{c}.npy',np.random.RandomState(c).normal(size=(4,3,32,32)).astype(np.float32))
    original=runner.load_task
    monkeypatch.setattr(runner,'load_task',lambda root,classes:original(root,classes,2,2))
    cfg=json.loads(Path('configs/standard_cl/h3_imagenet_small.json').read_text());cfg.update(num_tasks=2,num_showings=2,intervention_every=2)
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    order=[]
    def vna(self,model,opt,x,losses,sites,lr):
        assert len(losses.components)==1 and losses.components[0].name=='current_task_ce'
        order.append('vna');self.applications+=1
        with torch.no_grad():next(model.parameters()).add_(torch.rand(())*.00001)
        event=dict(seconds=0.,order='vna');self.events.append(event);return event
    def gir(self,model,opt,x,losses,sites):
        order.append('gir');self.applications+=1
        event=dict(seconds=0.,order='gir');self.events.append(event);return event
    monkeypatch.setattr(ReplayCorrection,'step',vna);monkeypatch.setattr(TangentReconfiguration,'step',gir)
    def args(out):return runner.parser().parse_args(['--arm',arm,'--data-dir',str(data),'--device','cpu',
        '--config',str(config),'--out',str(out)])
    saved=[];save_original=runner.Progress.save
    def observe_save(self,state):
        saved.append((state['task'],state['epoch'],state['vna']['applications'],state['gir']['applications']))
        return save_original(self,state)
    monkeypatch.setattr(runner.Progress,'save',observe_save)
    runner.run(args(tmp_path/'complete'))
    if arm=='vna_gir':assert (0,1,1,1) in saved
    assert order=={'cbp':[],'bp':[],'vna':['vna']*4,'gir':['gir']*4,'vna_gir':['vna','gir']*4}[arm]
    calls=0
    def deadline():
        nonlocal calls
        calls+=1;return calls==5
    monkeypatch.setattr(runner,'deadline_due',deadline)
    with pytest.raises(SystemExit) as exc:runner.run(args(tmp_path/'paused'))
    assert exc.value.code==75
    monkeypatch.setattr(runner,'deadline_due',lambda:False);runner.run(args(tmp_path/'paused'))
    a=torch.load(tmp_path/'complete/latest.pt',weights_only=False);b=torch.load(tmp_path/'paused/latest.pt',weights_only=False)
    for key in ('learner','rng','rows','epochs','steps','last_event','window'):_assert_bitwise(a[key],b[key],key)
    if arm=='vna_gir':
        interrupted=False
        def interrupt_gir(*args,**kwargs):
            nonlocal interrupted
            event=gir(*args,**kwargs)
            if not interrupted:interrupted=True;raise SystemExit(75)
            return event
        monkeypatch.setattr(TangentReconfiguration,'step',interrupt_gir)
        with pytest.raises(SystemExit):runner.run(args(tmp_path/'during_gir'))
        monkeypatch.setattr(TangentReconfiguration,'step',gir);runner.run(args(tmp_path/'during_gir'))
        c=torch.load(tmp_path/'during_gir/latest.pt',weights_only=False)
        for key in ('learner','rng','rows','epochs','steps','last_event','window','vna','gir'):_assert_bitwise(a[key],c[key],'episode.'+key)
        for filename in ('interventions.json','vna_interventions.json'):
            events=json.loads((tmp_path/'during_gir'/filename).read_text())
            points=[(r['task'],r['step']) for r in events]
            assert len(points)==len(set(points))==4
    from scripts.collect_standard_cl import collect
    assert collect([tmp_path/'complete'])[0]['outcomes']=={'COMPLETED':1}


class ToyOnlineNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        from avalanche.models import IncrementalClassifier
        self.hidden=nn.Linear(3,3);self.relu=nn.ReLU();self.linear=IncrementalClassifier(3,1)
    def standard_features(self,x):return self.relu(self.hidden(x))
    def forward(self,x):return self.linear(self.standard_features(x))
    def sites(self):
        from methods.vna_sites import HookSite
        return [HookSite(self,'hidden','linear.classifier','linear',between=('relu',))]


def test_native_slim_features_and_sites_match_original_forward(tmp_path,monkeypatch):
    pytest.importorskip('avalanche')
    import avalanche.benchmarks as benchmarks
    import avalanche.benchmarks.classic as classic
    from experiments.standard_online import native_setup
    raw=tmp_path/'cifar-100-python';raw.mkdir()
    for name in ('train','test','meta'):(raw/name).touch()
    from avalanche.benchmarks.generators import dataset_benchmark
    from avalanche.benchmarks.utils import make_classification_dataset
    from torch.utils.data import TensorDataset
    data=make_classification_dataset(TensorDataset(torch.randn(20,3,32,32),torch.tensor([0,1]*10)),task_labels=0)
    original=dataset_benchmark([data],[data]);original.classes_order_original_ids=[37,12]
    monkeypatch.setattr(classic,'SplitCIFAR100',lambda *a,**kw:original)
    cfg=json.loads(Path('configs/standard_cl/h2_online_cifar100.json').read_text())
    scenario,model=native_setup(cfg,str(tmp_path),0)
    assert scenario is not original
    assert scenario.classes_order_original_ids==[37,12]
    assert len(scenario.train_stream[0].dataset)==19
    assert len(scenario.valid_stream[0].dataset)==1
    model.eval();x=torch.randn(2,3,32,32);capture=[]
    handle=model.net.linear.register_forward_pre_hook(lambda module,args:capture.append(args[0].detach().clone()))
    with torch.no_grad():model(x);features=model.standard_features(x)
    handle.remove();_assert_bitwise(features,capture[0],'pooled_native_features')
    assert features.shape==(2,160) and len(model.sites())==8


@pytest.mark.parametrize('arm',['lpr','h2','h2_tuned','h2_fallback','h2_locked'])
def test_avalanche_online_native_callbacks_and_resume(tmp_path,monkeypatch,arm):
    pytest.importorskip('avalanche')
    from avalanche.benchmarks.generators import dataset_benchmark
    from avalanche.benchmarks.utils import make_classification_dataset
    from torch.utils.data import TensorDataset
    from experiments import standard_online as runner
    cfg=json.loads(Path('configs/standard_cl/h2_online_cifar100.json').read_text())
    cfg.update(tasks=2,train_mb_size=2,memory_size=8,intervention_every=2)
    if arm in ('h2_tuned','h2_fallback'):
        cfg['geometry_selection']=dict(weights=[0.,1.],first_task=2,validation_per_task=4)
        cfg['correction'].update(kernel='native',constrained=False)
    elif arm!='lpr':cfg.pop('geometry_selection',None)
    if arm=='h2_locked':cfg['geometry_authority']=dict(status='LOCKED',selected_weight=.1,sha256='fixture')
    cfg['correction'].update(damping_relative=1.,cg_iterations=30,trace_probes=1,cg_tolerance=1e-4,max_trials=3)
    config=tmp_path/'online.json';config.write_text(json.dumps(cfg))
    def setup(*args):
        gen=torch.Generator().manual_seed(81)
        datasets=[make_classification_dataset(TensorDataset(torch.randn(12,3,generator=gen),torch.tensor([t*2,t*2+1]*6)),task_labels=0) for t in range(2)]
        scenario=dataset_benchmark(datasets,datasets);scenario.classes_order_original_ids=list(range(4))
        validation=[SimpleNamespace(dataset=TensorDataset(torch.randn(4,3,generator=gen),torch.tensor([t*2,t*2+1]*2))) for t in range(2)]
        scenario.valid_stream=validation
        return scenario,ToyOnlineNetwork()
    monkeypatch.setattr(runner,'native_setup',setup)
    if arm=='h2_fallback':monkeypatch.setattr(runner.GeometryPlugin,'select_geometry',lambda *a:dict(selected_weight=None,status='NO_CERTIFIED_NONZERO_CANDIDATE',seconds=0.,trials=[]))
    def args(out):return runner.parser().parse_args(['--arm','h2' if arm.startswith('h2_') else arm,'--data-dir',str(tmp_path),'--device','cpu',
        '--config',str(config),'--out',str(out)])
    runner.run(args(tmp_path/'complete'))
    a=torch.load(tmp_path/'complete/latest.pt',weights_only=False)
    assert a['clock']['train_iterations']==36 and a['clock']['train_exp_counter']==12
    if arm=='lpr':
        assert a['geometry_selection'] is None and a['geometry_selection_attempts']==[]
        assert a['lpr_inverses'] and a['lpr_every_iter']==30
        from avalanche.benchmarks.scenarios import OnlineCLScenario
        from avalanche.training.plugins import ReplayPlugin
        from avalanche.training.storage_policy import ClassBalancedBuffer
        from avalanche.training.supervised import Naive
        import random
        random.seed(0);np.random.seed(0);torch.manual_seed(0)
        scenario,model=setup();storage=ClassBalancedBuffer(8,adaptive_size=True)
        replay=ReplayPlugin(mem_size=8,batch_size_mem=2,storage_policy=storage)
        kwargs=cfg['lpr']
        algorithm=runner.official_module().LPRPlugin({k:SimpleNamespace(**v) for k,v in kwargs.items()},storage)
        native=Naive(model,torch.optim.SGD(model.parameters(),lr=.1),torch.nn.CrossEntropyLoss(),
            train_mb_size=2,train_epochs=3,eval_mb_size=128,device='cpu',evaluator=None,eval_every=-1,
            plugins=[replay,runner.LPRCallbacks(algorithm)])
        for t,exp in enumerate(scenario.train_stream):
            stream=OnlineCLScenario(scenario.streams.values(),experiences=exp,experience_size=2,access_task_boundaries=False)
            native.train(stream.train_stream,eval_streams=[],num_workers=0,drop_last=True)
            runner.evaluate(model,scenario.test_stream[:t+1],'cpu')
        _assert_bitwise(a['model'].state_dict(),model.state_dict(),'unmodified_native_online')
    else:assert a['engine']['applications']>0
    if arm=='h2_locked':
        assert a['geometry_selection_attempts']==[]
        assert a['geometry_selection']['selected_weight']==.1
        assert all(e['effective_geometry_weight']==.1 for e in a['engine']['events'])
    if arm=='h2_fallback':
        assert a['geometry_selection']['status']=='FALLBACK_ER'
        summary=json.loads((tmp_path/'complete/summary.json').read_text())
        assert summary['status']=='COMPLETED' and summary['geometry_evidence']=='VNA_PREFIX_THEN_ER_FALLBACK'
        assert not any(e.get('step',0)>a['geometry_selection']['step'] for e in a['engine']['events'])
    calls=0
    def deadline():
        nonlocal calls
        calls+=1;return calls==(12 if arm=='lpr' else 3)
    monkeypatch.setattr(runner,'deadline_due',deadline)
    with pytest.raises(SystemExit):runner.run(args(tmp_path/'paused'))
    if arm=='lpr':assert torch.load(tmp_path/'paused/latest.pt',weights_only=False)['lpr_inverses']
    monkeypatch.setattr(runner,'deadline_due',lambda:False);runner.run(args(tmp_path/'paused'))
    b=torch.load(tmp_path/'paused/latest.pt',weights_only=False)
    for key in ('optimizer','clock','rng','rows','epochs','position','last_event'):_assert_bitwise(a[key],b[key],key)
    _assert_bitwise(a['model'].state_dict(),b['model'].state_dict(),'model')
    assert a['lpr_inverses'].keys()==b['lpr_inverses'].keys()
    for name in a['lpr_inverses']:_assert_bitwise(a['lpr_inverses'][name].matrix,b['lpr_inverses'][name].matrix,'lpr.'+name)
    if arm=='h2_fallback':
        original_callback=runner.GeometryPlugin.after_update;interrupted=False
        def interrupt_fallback(self,strategy,**kwargs):
            nonlocal interrupted
            original_callback(self,strategy,**kwargs)
            if self.selection and self.selection.get('status')=='FALLBACK_ER' and not interrupted:
                interrupted=True;raise SystemExit(75)
        monkeypatch.setattr(runner.GeometryPlugin,'after_update',interrupt_fallback)
        with pytest.raises(SystemExit):runner.run(args(tmp_path/'during_fallback'))
        monkeypatch.setattr(runner.GeometryPlugin,'after_update',original_callback);runner.run(args(tmp_path/'during_fallback'))
        c=torch.load(tmp_path/'during_fallback/latest.pt',weights_only=False)
        for key in ('optimizer','clock','rng','rows','epochs','geometry_selection','geometry_selection_attempts'):_assert_bitwise(a[key],c[key],'fallback.'+key)
        _assert_bitwise(a['model'].state_dict(),c['model'].state_dict(),'fallback.model')
    if arm in ('h2','h2_tuned'):
        if arm=='h2_tuned':
            assert a['geometry_selection']['status']=='SELECTED'
            assert a['geometry_selection']['selected_weight']==b['geometry_selection']['selected_weight']
            assert a['geometry_selection']['validation_sha256']==b['geometry_selection']['validation_sha256']
            assert 4<a['geometry_selection']['validation_samples']<=8
            assert set(a['geometry_selection']['seen_classes'])<=set(a['geometry_seen_classes'])
            chosen=a['geometry_selection']['selected_weight'];selected_step=a['geometry_selection']['step']
            later=[e for e in a['engine']['events'] if e.get('step',0)>selected_step]
            assert later and all(e.get('effective_geometry_weight')==chosen for e in later)
        original=runner.GeometryCorrection.step;interrupted=False
        def interrupt(self,model,opt,x,losses,*positional,**kwargs):
            nonlocal interrupted
            row=original(self,model,opt,x,losses,*positional,**kwargs)
            if not interrupted and int(losses.targets.max())>=2:
                interrupted=True;raise SystemExit(75)
            return row
        monkeypatch.setattr(runner.GeometryCorrection,'step',interrupt)
        with pytest.raises(SystemExit):runner.run(args(tmp_path/'during_correction'))
        monkeypatch.setattr(runner.GeometryCorrection,'step',original);runner.run(args(tmp_path/'during_correction'))
        c=torch.load(tmp_path/'during_correction/latest.pt',weights_only=False)
        for key in ('optimizer','clock','rng','rows','epochs'):_assert_bitwise(a[key],c[key],'transaction.'+key)
        _assert_bitwise(a['model'].state_dict(),c['model'].state_dict(),'transaction.model')
