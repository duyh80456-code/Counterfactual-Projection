import ast
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from adapters.cil_setup import class_order,FINE_TO_COARSE
from diagnostics.vna_align import _assert_bitwise
from experiments import main_cil as main
from experiments import p1_compare as study
from methods.vna_v3 import V3Engine,training_products,FULL_POLICY_VERSION,POLICY_VERSION
from methods.vna_sites import HookSite
from scripts.build_kaggle_p1_compare import build,CASES


@pytest.mark.parametrize('increment',[10,20])
@pytest.mark.parametrize('seed',[0,1])
def test_superclass_tasks_cover_all_classes(increment,seed):
    order=class_order(seed)
    tasks=[order[i:i+increment] for i in range(0,100,increment)]
    assert len(tasks)==100//increment
    assert sorted(c for task in tasks for c in task)==list(range(100))
    for task in tasks:
        coarse=[FINE_TO_COARSE[c] for c in task]
        assert len(set(coarse))==increment//5
        assert all(coarse.count(c)==5 for c in set(coarse))


@pytest.mark.parametrize('bn',[False,True])
def test_t29e_dead_slot_zero_jacobian_and_function_preservation(bn):
    torch.set_num_threads(1);torch.manual_seed(13)
    model=nn.Sequential(nn.Linear(3,3),nn.BatchNorm1d(3) if bn else nn.Identity(),nn.ReLU(),nn.Linear(3,2)).double()
    with torch.no_grad():
        model[0].weight[0]=0;model[0].bias[0]=-1
        if bn:model[1].bias[0]=-1
    site=HookSite(model,'0','3','linear',between=('1','2'))
    x=torch.randn(4,3,dtype=torch.float64)
    opt=torch.optim.SGD(model.parameters(),lr=.05,momentum=.9)
    for p in model.parameters():opt.state[p]['momentum_buffer']=torch.ones_like(p)
    engine=V3Engine('vna_gir',solver='gram',curvature='power10');engine.begin_task(2)
    assert 0 in engine.dead_slots(model,[site],(x,torch.zeros(4,dtype=torch.long)))[site.module_name]
    with training_products(model,[],{},x) as products:
        J,ids,outgoing=engine._slot_columns(products,site,0,dead=False)
        assert J.eq(0).all()
        before=products._call({}).detach().clone()
    proposal={'weight':torch.randn(1,3,dtype=torch.float64),'bias':torch.ones(1,dtype=torch.float64),
              'bn_weight':torch.ones(1,dtype=torch.float64),'bn_bias':torch.zeros(1,dtype=torch.float64)}
    assert engine.install_gir(model,site,[dict(A=proposal)],[0],opt)==1
    with training_products(model,[],{},x) as products:
        torch.testing.assert_close(products._call({}),before,rtol=0,atol=1e-12)
    assert model[3].weight[:,0].eq(0).all()
    assert opt.state[model[0].weight]['momentum_buffer'][0].eq(0).all()


def test_curvature_copy_preserves_live_training_state():
    main.cil.upstream_imports();torch.set_num_threads(1)
    class Net(nn.Linear):
        def forward(self,x):return {'logits':super().forward(x)}
    net=Net(3,2).double();optimizer=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9)
    for p in net.parameters():p.grad=torch.ones_like(p);optimizer.state[p]['momentum_buffer']=torch.ones_like(p)
    learner=SimpleNamespace(_network=net,_old_network=None,_known_classes=0,_cur_task=0)
    hook=SimpleNamespace(learner=learner,optimizer=optimizer,intervention=None,steps=1,epoch_step=1)
    batch=(torch.randn(4,3,dtype=torch.float64),torch.tensor([0,1,0,1]))
    before=study.live_state(hook)
    value=study.curvature(hook,batch,0,0)
    assert value['finite'] and value['lambda_max_ggn']>0 and value['state_bitwise_unchanged']
    _assert_bitwise(before,study.live_state(hook))


@pytest.mark.parametrize('case',CASES)
def test_four_configuration_cpu_smoke(case,tmp_path):
    backbone,increment=CASES[case]
    source=tmp_path/'cflat_bn_protocol.json'
    source.write_text(json.dumps(dict(arm='icarl_cflat',backbone='bn',fake_data=False,
                                     training=dict(init_lr=.1,lrate=.1))))
    results=[]
    for arm in ('icarl','icarl_full'):
        cli=['--out',str(tmp_path/arm),'--arm',arm,'--backbone',backbone,
             '--increment',str(increment),'--max-tasks',str(100//increment),'--workers','0',
             '--solver','gram','--curvature','power10','--warmup-epochs','0','--gir-rule','schur_dead',
             '--fake-data','--device','cpu']
        if case=='M1s2':cli+=['--seed','2']
        if backbone=='resnet32':cli+=['--bn-lr-protocol',str(source)]
        result=study.run(main.parser().parse_args(cli));results.append(result)
        assert result['status']=='COMPLETED' and result['tasks_run']==2
        p=result['identity'];assert p['cfg']['increment']==increment and p.get('warmup_epochs',0)==0
        if arm=='icarl':
            saved=json.loads((tmp_path/arm/'run'/'protocol.json').read_text())
            assert saved=={**p,'parameter_count':result['tasks'][-1]['parameter_count'],'mu0':None}
        assert p['policy_version']==(FULL_POLICY_VERSION if arm=='icarl_full' else POLICY_VERSION) and p.get('grad_clip') is None
        trace=json.loads((tmp_path/arm/'stability_gap'/'measurements.json').read_text())
        assert set(trace['tasks'])=={'1','2'}
        assert all(any(point['r']==0 for point in entry['points']) for entry in trace['tasks'].values())
        assert all(row['state_bitwise_unchanged'] for row in json.loads((tmp_path/arm/'ggn_power10.json').read_text()))
        if arm=='icarl_full':
            ggn={(row['task'],row['iteration']):row for row in json.loads((tmp_path/arm/'ggn_power10.json').read_text()) if row['r']>0}
            events=json.loads((tmp_path/arm/'run'/'interventions.json').read_text())
            matched=[(ggn[(e['task'],e['iteration'])],e) for e in events if (e['task'],e['iteration']) in ggn]
            assert matched
            for row,event in matched:
                assert row['lambda_max_ggn']==pytest.approx(event['lambda_max_ggn_before'],rel=1e-7,abs=1e-10)
    assert results[0]['tasks'][0]==results[1]['tasks'][0]


def test_notebook_lanes_and_preflight():
    for case in CASES:
        book=build('a'*40,case,'b'*64)
        for cell in book['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        code=''.join(book['cells'][-1]['source'])
        assert ("for arm in ('icarl_full',):" if case in ('NB1','M1s2') else "for arm in ('icarl','icarl_full')") in code
        if case=='NB1':
            assert 'validate_nb1_baseline' in code and 'BASELINE_ROOT' in code
            assert "_logfix_v3" in ''.join(book['cells'][1]['source'])
        assert 'fcntl.LOCK_EX' in code and 'K0 =' in code and "lane['total_hours_range'][1]>11" in code
        if case=='M1s2':
            assert 'pool.submit(lane,0,2)' in code and 'pool.submit(lane,1,1)' not in code
            assert 'AUTHORITY_ROOT' in code and 'calibration_seed=0' in code
            assert 'gram_power10.json' not in code
        else:assert 'pool.submit(lane,0,0)' in code and 'pool.submit(lane,1,1)' in code


def test_diagnostics_resume_preserves_model_optimizer_rng(tmp_path,monkeypatch):
    def args(name):return main.parser().parse_args(['--out',str(tmp_path/name),'--arm','icarl',
        '--increment','10','--max-tasks','10','--fake-data','--device','cpu','--workers','0',
        '--solver','gram','--curvature','power10'])
    complete=study.run(args('complete'))
    original=main.ICaRLHook.checkpoint
    def pause(hook,**kw):
        original(hook,**kw)
        if not kw.get('mid',True) and hook.completed==1:raise SystemExit(75)
    with monkeypatch.context() as patch:
        patch.setattr(main.ICaRLHook,'checkpoint',pause)
        with pytest.raises(SystemExit) as error:study.run(args('resumed'))
        assert error.value.code==75
    resumed=study.run(args('resumed'))
    assert resumed['tasks']==complete['tasks']
    left,right=(torch.load(tmp_path/name/'run/latest.pt',weights_only=False,map_location='cpu') for name in ('complete','resumed'))
    for key in ('optimizer','rng'):_assert_bitwise(left[key],right[key])
    _assert_bitwise(left['learner']['_network'].state_dict(),right['learner']['_network'].state_dict())


def test_bn_lr_protocol_required_fields(tmp_path):
    source=tmp_path/'protocol.json';source.write_text(json.dumps(dict(arm='icarl_cflat',backbone='nobn')))
    args=main.parser().parse_args(['--out',str(tmp_path/'run'),'--backbone','resnet32',
                                 '--bn-lr-protocol',str(source),'--fake-data','--device','cpu'])
    with pytest.raises(ValueError,match='C-Flat BN'):main.run(args)


@pytest.mark.parametrize('bn',[False,True])
def test_curvature_equals_engine_lambda_before_at_intervention(bn):
    main.cil.upstream_imports();torch.set_num_threads(1);torch.manual_seed(41)
    class Native(nn.Sequential):
        def forward(self,x):return {'logits':super().forward(x)}
    net=Native(nn.Linear(3,3),nn.BatchNorm1d(3) if bn else nn.Identity(),nn.ReLU(),nn.Linear(3,4)).double()
    teacher=Native(nn.Linear(3,2)).double().eval()
    learner=SimpleNamespace(_network=net,_old_network=teacher,_known_classes=2,_cur_task=1)
    optimizer=torch.optim.SGD(net.parameters(),lr=.05,momentum=.9)
    batch=torch.randn(4,3,dtype=torch.float64),torch.tensor([0,1,2,3])
    spec,ctx=main.loss_context(learner,batch)
    spec.loss(net(batch[0])['logits'],ctx).backward();optimizer.step()
    engine=V3Engine('vna_gir',seed=7,solver='gram',curvature='power10');engine.begin_task(2)
    hook=SimpleNamespace(learner=learner,optimizer=optimizer,steps=31,epoch_step=1,
                         intervention=main.Intervention(engine,1))
    before=study.live_state(hook)
    row=study.curvature(hook,batch,1,7)
    _assert_bitwise(before,study.live_state(hook))
    model=main.Prediction(net)
    site=HookSite(model,'native.0','native.3','linear',between=('native.1','native.2'))
    spec,ctx=main.loss_context(learner,batch)
    event=engine.apply(model,[site],batch,spec,ctx,optimizer,31,force=True)
    assert row['old_samples']==2 and row['known_classes']==2
    assert row['generator_seed']==event['stability_before']['generator_seed']
    assert row['lambda_max_ggn']==pytest.approx(event['lambda_max_ggn_before'],rel=1e-12,abs=1e-12)


def test_bn_notebook_missing_protocol_raises_and_logs_source(tmp_path):
    from scripts.build_kaggle_p1_compare import RUNNER
    from pathlib import Path
    scope=dict(BACKBONE='resnet32',Path=lambda p:tmp_path if str(p)=='/kaggle/input' else Path(p))
    prefix=RUNNER.split('SCHEDULE=')[0]
    with pytest.raises(FileNotFoundError,match='C-Flat BN protocol'):exec(prefix,scope)
    assert "'C-FLAT BN LR SOURCE:',BN_PROTOCOL,'LR:'" in prefix


def test_class_scaled_costs_and_session_estimate():
    schedule=dict(K0=512,calibration=[dict(task=2,vna_seconds=60.,stability_seconds=5.)]*3)
    for inc in (10,20):
        result=study.estimate_lane(inc,3600.,schedule,300.)
        last=result['by_task'][-1]
        assert last['classes']==100
        assert last['seconds_per_application_range']==pytest.approx([65*(100/(2*inc))**p for p in (2,3)])
        assert last['application_steps'][:3]==[1,513,1537]
        assert result['total_hours_range'][0]==pytest.approx((7500+result['full_intervention_seconds_range'][0])/3600)
        assert result['sessions_11h_range'][1]>=result['sessions_11h_range'][0]>=1


def test_source_digest_is_propagated_to_all_run_protocols(tmp_path,monkeypatch):
    monkeypatch.setenv('VNA_SOURCE_DIGEST','c'*64)
    args=main.parser().parse_args(['--out',str(tmp_path),'--arm','icarl',
         '--increment','10','--fake-data','--device','cpu','--workers','0'])
    study.run(args)
    files=list(tmp_path.rglob('*protocol.json'))
    assert files
    assert all(json.loads(p.read_text())['source_digest']=='c'*64 for p in files)
    monkeypatch.delenv('VNA_SOURCE_DIGEST')
    args.fake_data=False
    with pytest.raises(ValueError,match='source_digest'):study.run(args)
