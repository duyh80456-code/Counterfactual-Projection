import ast
from copy import deepcopy
import json
import pickle
import sys
import threading
from pathlib import Path
import pytest
from adapters.cil_setup import COARSE_NAMES,FINE_TO_COARSE
from experiments import main_cil as main
from experiments.p1_compare import validate_p1_baseline
from experiments.parallel_validity import dataset_fingerprint
from methods.vna_v3 import FULL_POLICY_VERSION
from scripts.build_kaggle_p1_compare import build


@pytest.fixture
def legacy(tmp_path):
    data=tmp_path/'data'/'cifar-100-python';data.mkdir(parents=True)
    (data/'train').write_bytes(pickle.dumps(dict(fine_labels=list(range(100)),coarse_labels=FINE_TO_COARSE)))
    (data/'test').write_bytes(b'test fixture')
    (data/'meta').write_bytes(pickle.dumps(dict(coarse_label_names=COARSE_NAMES)))
    root=tmp_path/'baseline'/'icarl__s2';root.mkdir(parents=True)
    cfg=main.recipe(2,'cuda:0','resnet32_nobn',5);cfg['shuffle']=False
    p=dict(arm='icarl',seed=2,backbone='resnet32_nobn',class_order_kind='superclass',
        code_commit='84f833db7b152264003c26b236c5c628c64b8773',source_digest=None,
        upstream_commit=main.cil.UPSTREAM_PIN,policy_version='schur_gamma_bound_rho25_v2',algorithm_version='v4',
        fake_data=False,expected_tasks=20,preflight_only=False,nobn_lr_scale=.5,coefficient=1.,
        solver='gram',curvature='power10',gir_live=False,cfg=cfg,
        dataset_fingerprint=dataset_fingerprint(data.parent),class_order=main.class_order(2,'superclass',data.parent),
        training=dict(first_epochs=100,later_epochs=50,first_milestones=[30,60,85],later_milestones=[24,35]))
    tasks=[dict(task=t,cnn_top1=50.,nme_top1=51.,parameter_count=460000+5*t) for t in range(1,21)]
    summary=dict(status='COMPLETED',identity=p,tasks=tasks,seconds=3600.)
    (root/'summary.json').write_text(json.dumps(summary))
    (root/'protocol.json').write_text(json.dumps({**p,'parameter_count':460100,'mu0':None}))
    (root/'tasks.json').write_text(json.dumps(tasks))
    return root,data.parent


@pytest.mark.parametrize('field',[None,'code_commit','source_digest','seed','fake_data','dataset','epochs','cfg','tasks','protocol'])
def test_original_inc5_baseline_provenance_and_bytes(legacy,field):
    root,data=legacy
    s=json.loads((root/'summary.json').read_text());p=s['identity']
    if field=='dataset':p['dataset_fingerprint']['test']='a'*64
    elif field=='epochs':p['training']['later_epochs']=49
    elif field=='cfg':p['cfg']['increment']=10
    elif field=='tasks':s['tasks'].pop()
    elif field=='protocol':
        q=json.loads((root/'protocol.json').read_text());q['mu0']=.1
        (root/'protocol.json').write_text(json.dumps(q))
    elif field:
        p[field]={'code_commit':'a'*40,'source_digest':'a'*64,'seed':1,'fake_data':True}[field]
    (root/'summary.json').write_text(json.dumps(s))
    before={str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()}
    if field:
        with pytest.raises(ValueError):validate_p1_baseline(root,2,data,5)
    else:
        result=validate_p1_baseline(root,2,data,5)
        assert result['code_commit']==p['code_commit'] and result['source_digest'] is None
        assert result['baseline_measurements']=='not_recorded_in_legacy_run'
    assert before=={str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()}


@pytest.mark.parametrize('corruption',[None,'lr_scale','epochs','memory','warmup','clip',
                                      'dataset','class_order','coordinates','step','tensor'])
def test_diverged_seed2_baseline_requires_matching_protocol_and_recorded_failure(legacy,corruption):
    root,data=legacy
    summary=json.loads((root/'summary.json').read_text())
    p=summary['identity']
    summary.update(status='FAILED_DIVERGED',tasks=summary['tasks'][:1],
        diverged=dict(task=2,epoch=1,step=8,tensor='loss'),cnn_last_accuracy=float('nan'))
    if corruption=='lr_scale':p['nobn_lr_scale']=1.
    if corruption=='epochs':p['training']['later_epochs']=49
    if corruption=='memory':p['cfg']['memory_size']=1999
    if corruption=='warmup':p['warmup_epochs']=1
    if corruption=='clip':p['grad_clip']=1.
    if corruption=='dataset':p['dataset_fingerprint']['train']='a'*64
    if corruption=='class_order':p['class_order'].reverse()
    if corruption=='coordinates':summary.pop('diverged')
    if corruption=='step':summary['diverged']['step']=9
    if corruption=='tensor':summary['diverged']['tensor']=''
    (root/'summary.json').write_text(json.dumps(summary))
    (root/'tasks.json').write_text(json.dumps(summary['tasks']))
    (root/'protocol.json').write_text(json.dumps({**p,
        'parameter_count':summary['tasks'][-1]['parameter_count'],'mu0':None}))
    before={str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()}
    if corruption:
        with pytest.raises(ValueError):validate_p1_baseline(root,2,data,5)
    else:
        reference=validate_p1_baseline(root,2,data,5)
        assert reference['status']=='VALIDATED_ORIGINAL_BASELINE'
        assert reference['outcome']=='FAILED_DIVERGED'
        assert reference['diverged']==dict(task=2,epoch=1,step=8,tensor='loss')
        assert reference['code_commit']=='84f833db7b152264003c26b236c5c628c64b8773'
        assert reference['source_digest'] is None
    assert before=={str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()}


def test_nb1_still_rejects_failed_baseline(legacy):
    from experiments.p1_compare import validate_nb1_baseline
    root,data=legacy
    summary=json.loads((root/'summary.json').read_text())
    summary.update(status='FAILED_DIVERGED',diverged=dict(task=2,epoch=1,step=8,tensor='loss'))
    (root/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='NB1 baseline not completed'):
        validate_nb1_baseline(root,0,data)


def test_seed2_notebook_uses_fresh_locked_seed0_authority(legacy,tmp_path):
    root,data=legacy
    book=build('c'*40,'M1s2','a'*64)
    for cell in book['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    runtime=''.join(book['cells'][2]['source'])
    runner=''.join(book['cells'][3]['source'])
    assert 'outputs/p1_inc5_full_v3_authority' in runtime
    assert "'icarl__nobn__superclass__s2'" in runtime
    assert "for arm in ('icarl_full',):" in runner
    assert 'jobs=[pool.submit(lane,0,2)]' in runner
    assert 'gram_power10.json' not in runner
    assert "command('icarl_full',0,AUTHORITY_ROOT/'_preflight_seed0')" in runner
    new=tmp_path/'new';new.mkdir()
    authority=tmp_path/'authority';authority.mkdir()
    calls=[]
    def launch(cmd,gpu,name,log):
        calls.append((cmd,gpu,name))
        assert name=='preflight' and cmd[cmd.index('--seed')+1]=='0'
        assert cmd[cmd.index('--increment')+1]=='5' and '--preflight-only' in cmd
        schedule=Path(cmd[cmd.index('--schedule-output')+1])
        schedule.write_text(json.dumps(dict(authoritative=True,backbone='resnet32_nobn',init_cls=5,increment=5,
            code_commit='c'*40,source_digest='a'*64,solver='gram',curvature='power10',
            policy_version=FULL_POLICY_VERSION,gir_live=False,calibration_seed=0,K0=100,
            nobn_lr_scale=.5,class_order_kind='superclass',dataset_fingerprint=dataset_fingerprint(data))))
        return True
    scope=dict(ROOT=new,BASELINE_ROOT=root.parent,AUTHORITY_ROOT=authority,DATA_ROOT=data,
        BACKBONE='resnet32_nobn',INCREMENT=5,NOTEBOOK='M1s2',REPO_COMMIT='c'*40,SOURCE_DIGEST='a'*64,
        RUN_ROOT=tmp_path,Path=Path,sys=sys,threading=threading,json=json,launch=launch,
        archive_outputs=lambda:None)
    exec(runner.split('\ntry:\n    estimate()')[0],scope)
    assert scope['schedule'](0) and scope['schedule'](0) and len(calls)==1
    schedule=scope['SCHEDULE']
    assert schedule.parent==authority and schedule.name=='resnet32_nobn_inc5_seed0.json'
    saved=json.loads(schedule.read_text())
    for field,value in (('calibration_seed',2),('policy_version','schur_gamma_bound_rho25_v2'),
                        ('source_digest','b'*64),('increment',10),('dataset_fingerprint',{})):
        schedule.write_text(json.dumps({**saved,field:value}))
        with pytest.raises(ValueError,match='authority mismatch'):scope['schedule'](0)
