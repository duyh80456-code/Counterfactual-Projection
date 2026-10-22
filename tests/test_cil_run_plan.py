"""Registered LR selection, native resume authority and main result aggregation."""
import ast
import csv
import json
import sys
from pathlib import Path
import pytest
from adapters.cil_setup import class_order
from experiments import main_cil as main,v4_competitors as competitors
from scripts import build_kaggle_vna_parallel as builder
from scripts.collect_cil_main import collect,export


def completed(root,seed,solver='gram',commit='a'*40):
    p=dict(setting='icarl',algorithm_version='v4',arm='icarl_full',seed=seed,code_commit=commit,
        upstream_commit=main.cil.UPSTREAM_PIN,expected_tasks=20,r=16,backbone='resnet32_nobn',
        class_order_kind='superclass',class_order=class_order(seed),nobn_lr_scale=.5,solver=solver,curvature='power10',
        policy_version='schur_gamma_bound_rho25_v2',schedule=dict(K0=40,calibration_code_commit='c'*40),
        cfg=dict(init_cls=5,increment=5,memory_size=2000,fixed_memory=False),
        training=dict(first_epochs=100,later_epochs=50,first_milestones=[30,60,85],later_milestones=[24,35]),
        dataset_fingerprint={k:'b'*64 for k in ('train','test','meta')})
    tasks=[]
    for i in range(1,21):
        row=dict(task=i,feature_accuracy=70.)
        for head in ('cnn','nme'):
            row.update({head+'_'+k:60.+seed for k in ('top1','old_top1','new_top1','learning_accuracy','forgetting','average_accuracy','last_accuracy')})
            row[head+'_per_class']={str(c):60. for c in range(5*i)}
        tasks.append(row)
    summary=dict(identity=p,status='COMPLETED',tasks=tasks,gpu_hours=1.)
    summary.update({h+'_'+k:60.+seed for h in ('cnn','nme') for k in ('average_accuracy','last_accuracy')})
    root.mkdir(parents=True)
    values={'protocol.json':p,'summary.json':summary,'tasks.json':tasks,'status.json':dict(status='COMPLETED'),
        'diagnostics.json':[dict(task=i,B=1.,mu_diag=.1,norm_w2=2.) for i in range(1,21)],'interventions.json':[]}
    for name,value in values.items():(root/name).write_text(json.dumps(value))
    (root/'continual_eval.csv').write_text('task\n')
    return p


def competitor_result(root,method='gem',scale=.5):
    cfg,source,overrides=competitors.recipe(method,'nobn',0,'cpu',False,scale)
    p=dict(benchmark='v4_competitor',arm=method,effective_method=method,backbone='nobn',order='superclass',seed=0,
        code_commit='a'*40,cfg=cfg,cfg_source=source,config_overrides=overrides,upstream_commit=main.cil.UPSTREAM_PIN,
        competitor_pins=competitors.PINS,method_config=competitors.grid(method)[0],
        implementation=competitors.SOURCES.get(method,dict(source=f'PyCIL/models/{method}.py',reimplemented=False)),
        fake_data=False,disabled=False,reproduction=False,epochs='100/50',expected_tasks=20,nobn_lr_scale=scale,
        training=main.training_settings(False,False,scale),class_order=class_order(0),
        dataset_fingerprint={k:'b'*64 for k in ('train','test','meta')})
    tasks=[dict(task=i,cnn_per_class=[10.]*(5*i),cnn_top1=10.,cnn_new_top1=10.,cnn_average_accuracy=10.,
                cnn_last_accuracy=10.,cnn_forgetting=0.,cnn_learning_accuracy=10.) for i in range(1,21)]
    s=dict(status='COMPLETED',identity=p,tasks=tasks,parameter_count=100,gpu_hours=1.,average_incremental_accuracy=10.,
           cnn_average_accuracy=10.,cnn_last_accuracy=10.)
    root.mkdir(parents=True)
    for name,value in {'summary.json':s,'protocol.json':p,'tasks.json':tasks,'status.json':dict(phase='COMPLETED'),
                       'diagnostics.json':[dict(task=i) for i in range(1,21)]}.items():(root/name).write_text(json.dumps(value))
    (root/'continual_eval.csv').write_text('task\n'+''.join(f'{i}\n' for i in range(1,21)))
    return s


def test_notebook_jobs_and_real_commands(tmp_path):
    def jobs(preset):return sum((list(lane) for lane in builder.PRESETS[preset]),[])
    assert len(jobs('M1'))==4 and all(j.endswith(('__s1','__s2')) for j in jobs('M1'))
    assert len(jobs('C2'))==12 and all('__bn__' not in j and not j.endswith('__s0') for j in jobs('C2'))
    assert len(jobs('C3'))==8
    nodes=[n for n in ast.parse(builder.RUNNER).body if isinstance(n,ast.FunctionDef) and n.name in ('schedule_name','command')]
    scope=dict(ROOT=tmp_path,STAGE_B=tmp_path,PRESET='M1',sys=sys,DATA_ROOT=tmp_path,NOBN_LR_SCALE=.5)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'commands','exec'),scope)
    for job in jobs('M1'):
        command=scope['command'](job)
        assert command[command.index('--solver')+1]=='gram'
        assert command[command.index('--curvature')+1]=='power10'
    scope['PRESET']='C2'
    command=scope['command']('gem__sweep')
    assert '--reuse-root' in command
    command=scope['command']('wa__nobn__superclass__s1')
    assert command[command.index('--best-config')+1]==str(tmp_path/'wa__sweep'/'best_config.json')
    nb=builder.build('a'*40,'M1')
    source='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert 'MAIN RUN HELD' not in source
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))


def test_main_export_partial_then_three_seeds_and_rejects_solver_mix(tmp_path):
    completed(tmp_path/'s0',0)
    out=tmp_path/'reports'
    rows=export([tmp_path],out)
    assert rows[0]['coverage']=='PARTIAL_SEEDS' and rows[0]['cnn_A_std'] is None
    completed(tmp_path/'s1',1,commit='d'*40);completed(tmp_path/'s2',2,commit='e'*40)
    rows=export([tmp_path],out)
    assert rows[0]['coverage']=='COMPLETE_3_SEEDS'
    assert rows[0]['cnn_A_mean']==61. and rows[0]['cnn_A_std']==1.
    assert len(list(csv.DictReader((out/'average_accuracy_by_task.csv').open())))==40
    assert len(list(out.glob('*.svg')))==2 and len(list(out.glob('*.png')))==2
    path=tmp_path/'s1'/'protocol.json';p=json.loads(path.read_text());p['solver']='cg_ir';path.write_text(json.dumps(p))
    summary=tmp_path/'s1'/'summary.json';s=json.loads(summary.read_text());s['identity']=p;summary.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='mixed solver'):collect([tmp_path])


def test_gem_reuse_is_exact_rate_validated_and_cacheable(tmp_path,monkeypatch):
    s=competitor_result(tmp_path/'old')
    monkeypatch.setattr('experiments.parallel_validity.dataset_fingerprint',lambda root:s['identity']['dataset_fingerprint'])
    monkeypatch.setattr(competitors,'class_order',lambda *a:s['identity']['class_order'])
    trial=competitors.parser().parse_args(['--method','gem','--out',str(tmp_path/'new'),'--reuse-root',str(tmp_path),
                                         '--config','{}','--nobn-lr-scale','.5'])
    # GEM's registered config, rather than an arbitrary user override.
    trial.config=json.dumps(competitors.grid('gem')[0])
    result=competitors.reuse_gem_candidate(trial)
    assert result==s and json.loads((tmp_path/'new'/'protocol.json').read_text())['code_commit']=='a'*40
    assert competitors.reuse_gem_candidate(trial)==s
    trial.nobn_lr_scale=.1
    assert competitors.reuse_gem_candidate(trial) is None
    trial.nobn_lr_scale=.5
    (tmp_path/'new'/'status.json').write_text('{}')
    # Remove the source so the cache itself must pass validation.
    import shutil
    shutil.rmtree(tmp_path/'old')
    with pytest.raises(ValueError,match='incomplete'):competitors.reuse_gem_candidate(trial)


def test_sweep_collection_keeps_only_registered_winner(tmp_path):
    root=tmp_path/'gem__sweep'
    winner=competitor_result(root/'scan'/'lr0'/'0')
    competitor_result(root/'scan'/'lr1'/'0',scale=1.)
    p=winner['identity']
    best=dict(method='gem',config=p['method_config'],nobn_lr_scale=.5,lr_scales=list(competitors.NOBN_LR_SCALES),
        selected_trial='scan/lr0/0',score=10.,code_commit='a'*40,dataset_fingerprint=p['dataset_fingerprint'],
        seed=0,backbone='nobn',order='superclass',metric='cnn_average_accuracy',synthetic=False)
    (root/'best_config.json').write_text(json.dumps(best))
    groups,skipped=collect([tmp_path]);assert len(groups)==1 and len(next(iter(groups.values())))==1
    competitor_result(tmp_path/'legacy',scale=1.)
    groups,skipped=collect([tmp_path]);assert len(next(iter(groups.values())))==1
    assert skipped[0]['reason']=='non-winning seed0 LR/config'


def test_c2_sweep_barrier_and_resume_skip(tmp_path):
    import threading
    from types import SimpleNamespace
    calls=[];lock=threading.Lock();lanes=builder.PRESETS['C2'];sweeps={j for lane in lanes for j in lane if j.endswith('__sweep')}
    def launch(cmd,gpu,job,log):
        with lock:
            if not job.endswith('__sweep'):
                assert sweeps<={name for _,name in calls} or all((scope['ROOT']/j/'best_config.json').exists() for j in sweeps)
            calls.append((gpu,job))
        return True
    scope=dict(PRESET='C2',STAGE_B=tmp_path,REPO_COMMIT='a'*40,RUN_TAG='fixture',ENV={},
        JOBS_GPU0=lanes[0],JOBS_GPU1=lanes[1],JOBS_CPU=[],SOURCE_DIGEST=None,
        Path=Path,json=json,sys=sys,DATA_ROOT=tmp_path,launch=launch,pause_due=lambda:False,
        archive_outputs=lambda:None,RUN_ROOT=tmp_path,archive_stop=threading.Event(),archiver=SimpleNamespace(join=lambda **k:None))
    exec(builder.RUNNER,scope)
    assert len(calls)==12 and {g for g,j in calls}=={0,1}
    # Native completed sweeps are skipped by the same runner on a later session.
    root=scope['ROOT']
    for job in sweeps:
        p=root/job;p.mkdir(parents=True)
        (p/'best_config.json').write_text(json.dumps(dict(code_commit='a'*40,source_digest=None,synthetic=False)))
    calls.clear()
    scope['JOBS_GPU0'],scope['JOBS_GPU1']=lanes[:2]
    exec(builder.RUNNER,scope)
    assert len(calls)==8 and all(not j.endswith('__sweep') for g,j in calls)
