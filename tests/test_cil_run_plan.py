"""Registered LR selection, native resume authority and main result aggregation."""
import ast
import csv
import json
import subprocess
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
    assert 'C1' not in builder.PRESETS
    assert len(jobs('C3'))==12 and any(j.endswith('__s0') for j in jobs('C3'))
    nodes=[n for n in ast.parse(builder.RUNNER).body if isinstance(n,ast.FunctionDef) and n.name in ('schedule_name','command')]
    scope=dict(ROOT=tmp_path,STAGE_B=tmp_path,PRESET='M1',sys=sys,DATA_ROOT=tmp_path,NOBN_LR_SCALE=.5,GRAD_CLIP=None)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'commands','exec'),scope)
    for job in jobs('M1'):
        command=scope['command'](job)
        assert command[command.index('--solver')+1]=='gram'
        assert command[command.index('--curvature')+1]=='power10'
    scope['PRESET']='C3'
    assert '--best-config' not in scope['command']('icarl_cbp__nobn__superclass__s0')
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


def clip_package(root,baseline=False):
    from scripts.collect_cil_main import canonical_hash
    from experiments.clip_study import GRID,A_CASES
    import hashlib
    native=root/'_sources'/'native'
    p=completed(native,0);p['source_digest']='1'*64
    if baseline:p['arm']='icarl'
    summary=json.loads((native/'summary.json').read_text());summary['identity']=p
    summary['parameter_count']=100
    (native/'protocol.json').write_text(json.dumps({**p,'parameter_count':100,'mu0':None if baseline else .01}))
    raw=json.dumps(summary);(native/'summary.json').write_text(raw)
    digest=hashlib.sha256(raw.encode()).hexdigest()
    source={f.name:json.loads(f.read_text()) if f.suffix=='.json' else f.read_text() for f in native.iterdir()}
    reference=dict(identity=p,summary=summary,summary_sha256=digest,safe=dict(sha256='3'*64))
    epochs=1050 if baseline else 250
    loss=dict(status='PASS',epochs_compared=epochs,expected_epochs=epochs,
              absolute_tolerance=0.,relative_tolerance=0.,differences=[])
    study=dict(mode='B' if baseline else 'prefix',arm=p['arm'],seed=0,tau=None if baseline else 2.,
        status='COMPLETED' if baseline else 'PREFIX_COMPLETED',tasks=20 if baseline else 4,epoch=None,
        code_commit='c'*40,reference_sha256=digest,loss_check=loss,diagnostic_only=True,
        equivalent_under_clip=True,checkpoint_sha256=None,
        grad_clip=dict(run=dict(n_steps=2500,n_clipped=0,nonfinite_steps=0,max_preclip_norm=1.)))
    proof=dict(status='EQUIVALENT_PREFIX',tau=2.,loss_check=loss,n_clipped=0,n_steps_checked=2500,
        scope='all 20 tasks; full norm replay in B' if baseline else dict(task=4,epoch=None),
        checkpoint_sha256='3'*64,reference=reference)
    selection=dict(status='SELECTED',tau=2.,G_max=1.,grid=list(GRID),code_commit='c'*40,
        checks=[dict(tau=t,baseline_bound=t>=2.,all_four_survive_60=True) for t in GRID],
        source_reports={**{f'A/{a}__s{s}/{label}/study.json':'4'*64 for a,s in A_CASES
            for label in ('none',*(f'tau{t:g}' for t in GRID))},**{f'B/icarl__s{s}/study.json':'4'*64 for s in (0,1)}})
    package=dict(reference=reference,verification=proof,source_artifacts=source,
        source_artifact_sha256={k:canonical_hash(v) for k,v in source.items()},source_summary_text=raw,
        source_summary_sha256=digest,selection=selection,selection_sha256=canonical_hash(selection),
        study=study,consumer_commit='c'*40,consumer_source_digest='2'*64)
    destination=root/'C'/'reused';destination.mkdir(parents=True)
    (destination/'clip_reuse.json').write_text(json.dumps(package))
    return package,destination


@pytest.mark.parametrize('baseline',[False,True])
def test_collector_clip_reuse_preserves_producer_and_ignores_diagnostics(tmp_path,baseline):
    package,destination=clip_package(tmp_path,baseline)
    diagnostic=tmp_path/'C'/'other'/'prefix'
    completed(diagnostic,0,solver='deliberately_not_a_result')
    (diagnostic/'study.json').write_text(json.dumps(dict(diagnostic_only=True,mode='prefix')))
    groups,skipped=collect([tmp_path]);runs=next(iter(groups.values()))
    assert len(runs)==1 and runs[0]['summary']['identity']==package['reference']['identity']
    assert runs[0]['protocol']['parameter_count']==100
    assert runs[0]['protocol'].get('grad_clip') is None
    assert runs[0]['comparison_protocol']['grad_clip']==2.
    assert runs[0]['comparison_protocol']['source_digest']=='2'*64
    assert runs[0]['clip_reuse']['epochs_compared']==(1050 if baseline else 250)
    assert any(s['reason']=='diagnostic-only clipping study' for s in skipped)
    if not baseline:
        p=completed(tmp_path/'C'/'trained',1,commit='c'*40)
        p.update(grad_clip=2.,source_digest='2'*64)
        (tmp_path/'C'/'trained'/'protocol.json').write_text(json.dumps(p))
        path=tmp_path/'C'/'trained'/'summary.json';s=json.loads(path.read_text());s['identity']=p;path.write_text(json.dumps(s))
        groups,_=collect([tmp_path]);assert len(next(iter(groups.values())))==2
    package['study']['grad_clip']['run']['n_clipped']=1
    (destination/'clip_reuse.json').write_text(json.dumps(package))
    with pytest.raises(ValueError,match='loss/norm evidence'):collect([tmp_path])


def test_collector_rejects_tampered_clipping_reuse_source(tmp_path):
    package,destination=clip_package(tmp_path)
    package['source_artifacts']['tasks.json'][0]['cnn_top1']=99.
    (destination/'clip_reuse.json').write_text(json.dumps(package))
    with pytest.raises(ValueError,match='artifact hash mismatch'):collect([tmp_path])


@pytest.mark.parametrize('baseline',[False,True])
def test_collector_user_fixed_tau50_keeps_original_infeasible_evidence(tmp_path,baseline):
    import copy
    from scripts.collect_cil_main import canonical_hash
    from experiments.clip_study import FIXED_SELECTION_POLICY,FIXED_SELECTION_RULE
    package,destination=clip_package(tmp_path,baseline)
    original=copy.deepcopy(package['selection'])
    original.update(status='NO_FEASIBLE_TAU',tau=None,G_max=42.947,
                    code_commit='3a6638846b612b29dcc8c4010ced759a46b151a2',source_digest=None)
    for check in original['checks']:check['baseline_bound']=False
    selection={**copy.deepcopy(original),'status':'SELECTED','tau':50.,'selection_policy':FIXED_SELECTION_POLICY,
        'code_commit':package['consumer_commit'],'source_digest':package['consumer_source_digest'],
        'evidence_code_commit':original['code_commit'],'original_selection':original,
        'original_selection_sha256':canonical_hash(original),'original_rule_satisfied':False,
        'rule':FIXED_SELECTION_RULE}
    package.update(selection=selection,selection_sha256=canonical_hash(selection))
    package['verification']['tau']=50.
    package['study'].update(tau=None if baseline else 50.,
        code_commit=original['code_commit'] if baseline else package['consumer_commit'],
        source_digest=original['source_digest'] if baseline else package['consumer_source_digest'])
    (destination/'clip_reuse.json').write_text(json.dumps(package))
    groups,_=collect([tmp_path]);run=next(iter(groups.values()))[0]
    assert run['comparison_protocol']['grad_clip']==50.
    assert run['protocol'].get('grad_clip') is None
    assert run['clip_reuse']['selection_policy']==FIXED_SELECTION_POLICY
    assert run['clip_reuse']['evidence_code_commit']==original['code_commit']
    assert run['summary']['identity']==package['reference']['identity']
    package['study']['code_commit']='f'*40
    (destination/'clip_reuse.json').write_text(json.dumps(package))
    with pytest.raises(ValueError,match='replay scope'):collect([tmp_path])
    package['study']['code_commit']=original['code_commit'] if baseline else package['consumer_commit']
    package['selection']['original_selection']['G_max']=1.
    package['selection_sha256']=canonical_hash(package['selection'])
    (destination/'clip_reuse.json').write_text(json.dumps(package))
    with pytest.raises(ValueError):collect([tmp_path])


def test_clipping_certificate_upgrades_identical_attached_source_only(tmp_path):
    package,_=clip_package(tmp_path/'clip')
    native=tmp_path/'original';native.mkdir()
    for name,value in package['source_artifacts'].items():
        (native/name).write_text(json.dumps(value) if name.endswith('.json') else value)
    for inputs in ([native,tmp_path/'clip'],[tmp_path/'clip',native]):
        groups,_=collect(inputs);records=next(iter(groups.values()))
        assert len(records)==1 and records[0]['clip_reuse']['tau']==2.
    summary=json.loads((native/'summary.json').read_text());summary['gpu_hours']=2.
    (native/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='Conflicting duplicate'):collect([native,tmp_path/'clip'])


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
        archive_outputs=lambda:None,RUN_ROOT=tmp_path,archive_stop=threading.Event(),archiver=SimpleNamespace(join=lambda **k:None),
        GRAD_CLIP=None,subprocess=subprocess)
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
