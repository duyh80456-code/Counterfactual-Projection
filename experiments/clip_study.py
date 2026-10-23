"""Registered A/B/C clipping study: native replay, exact-loss gates and bounded tau selection."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import threading
import time
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import torch
from experiments.io_utils import atomic_json_save,atomic_torch_save
from experiments.vna_resume import launch_logged
from experiments.vna_provenance import code_commit
from scripts.collect_parallel import files

GRID=(1.,2.,5.,10.,20.,50.)
A_CASES=(('icarl',2),('icarl_full',2),('icarl_cbp',1),('icarl_redo',1))
C_CASES=tuple((arm,seed) for arm in ('icarl','icarl_full','icarl_cbp','icarl_redo','icarl_snp','icarl_upgd','gem','wa','podnet','icarl_cflat') for seed in range(3))
ARTIFACTS={'summary.json','protocol.json','epochs.json','tasks.json','diagnostics.json','continual_eval.csv',
    'interventions.json','status.json','latest.pt','latest.previous.pt','reused_baseline.json'}


def key(arm,seed):return f'{arm}__s{seed}'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def import_inputs(input_root,target):
    """Namespace each input source; never merge distinct runs with identical old paths."""
    input_root=Path(input_root);target=Path(target);target.mkdir(parents=True,exist_ok=True)
    sources=[input_root] if input_root.is_dir() else []
    sources+=sorted(p for p in input_root.rglob('*') if p.is_file() and p.name.endswith(('.tar.gz','.tgz','.zip','.tar'))) if input_root.is_dir() else [input_root]
    for source in sources:
        if source.is_file() and source.name.startswith('vna4_clip_'):continue
        label=hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
        for path,data in files(source,ARTIFACTS):
            if '..' in path.parts or path.is_absolute():raise ValueError('unsafe input artifact path')
            if 'outputs' not in path.parts or 'clip_study' in path.parts:continue
            destination=target/label/str(path)
            if destination.exists():
                if destination.read_bytes()!=data:raise ValueError('attached input changed under existing source namespace')
                continue
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
    return target


def inventory(root,out):
    from experiments.runners.pycil import upstream_imports
    from experiments.v4_competitors import DEFAULT_CONFIGS
    upstream_imports();candidates={};rejected=[]
    for path in Path(root).rglob('summary.json'):
        value=json.loads(path.read_text());p=value.get('identity',{});arm,seed=p.get('arm'),p.get('seed')
        if arm not in {a for a,s in C_CASES} or seed not in (0,1,2):continue
        if p.get('backbone') not in ('nobn','resnet32_nobn') or p.get('nobn_lr_scale')!=.5 or p.get('grad_clip') is not None:continue
        if p.get('class_order_kind',p.get('order'))!='superclass' or p.get('fake_data') or p.get('preflight_only'):continue
        if value.get('status')=='REUSED_BASELINE':continue
        if arm=='icarl_full' and (p.get('solver'),p.get('curvature'),p.get('policy_version'),p.get('gir_live'))!=('gram','power10','schur_gamma_bound_rho25_v2',False):continue
        if arm in DEFAULT_CONFIGS and (p.get('method_config')!=DEFAULT_CONFIGS[arm] or p.get('parameter_policy')!='original_defaults_no_search_v1'):continue
        if arm=='icarl_cflat':
            from adapters.v4_competitors import cflat_module
            cflat_module()
        checkpoints=[];epochs=None
        for checkpoint in (path.with_name('latest.pt'),path.with_name('latest.previous.pt')):
            if not checkpoint.exists():continue
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state.get('identity')!=p:raise ValueError('source summary/checkpoint identity mismatch: '+str(checkpoint))
            if 'learner' in state:
                completed=state['completed'];mid=state['mid'];epoch=state['epoch'] if mid else None
                steps=state['hook']['steps'];rows=state['hook']['epochs']
            else:
                completed=len(state['tasks']);mid=False;epoch=None
                rows=getattr(state['model'],'_v4_epochs',[]);steps=completed
            checkpoints.append(dict(path=str(checkpoint),sha256=sha(checkpoint),completed=completed,mid=mid,epoch=epoch,steps=steps))
            if epochs is None or len(rows)>len(epochs):epochs=rows
        if not checkpoints:
            rejected.append(dict(path=str(path),reason='no native checkpoint'));continue
        epochs_path=path.with_name('epochs.json')
        if epochs_path.exists():epochs=json.loads(epochs_path.read_text())
        safe=max(checkpoints,key=lambda c:(c['steps'],not c['mid']))
        task1=next((c for c in checkpoints if c['completed']==1 and not c['mid']),None)
        try:date=int(subprocess.check_output(['git','show','-s','--format=%ct',p['code_commit']],text=True,stderr=subprocess.DEVNULL).strip())
        except subprocess.CalledProcessError:date=0
        descriptor=dict(identity=p,summary=value,summary_path=str(path),summary_sha256=sha(path),epochs=epochs,
            checkpoints=checkpoints,safe=safe,task1=task1)
        score=(value.get('status')=='COMPLETED',len(value.get('tasks',[])),date)
        name=key(arm,seed)
        if name not in candidates or score>candidates[name][0]:candidates[name]=(score,descriptor)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);references={}
    for name,(score,descriptor) in candidates.items():
        destination=out/(name+'.json');atomic_json_save(descriptor,destination);references[name]=str(destination)
    atomic_json_save(dict(references=references,rejected=rejected,missing=[key(a,s) for a,s in C_CASES if key(a,s) not in references]),out/'inventory.json')
    return references


def read_study(path):
    path=Path(path)/'study.json'
    return json.loads(path.read_text()) if path.exists() else None


def choose_tau(root,commit):
    root=Path(root);B={};missing=[]
    def binding(report,mode,arm,seed,tau):
        if report and any(report.get(k)!=v for k,v in dict(mode=mode,arm=arm,seed=seed,tau=tau,code_commit=commit,source_digest=os.environ.get('VNA_SOURCE_DIGEST')).items()):
            raise ValueError('study report identity mismatch')
    for seed in (0,1):
        report=read_study(root/'B'/key('icarl',seed))
        binding(report,'B','icarl',seed,None)
        if not report or report.get('status')!='COMPLETED' or report.get('equivalent_under_clip') is not True:
            missing.append('B/'+key('icarl',seed));continue
        stats=report['grad_clip']['run']
        if (stats['nonfinite_steps'] or not stats['n_steps'] or stats['n_clipped'] or
            stats['max_preclip_norm'] is None or not math.isfinite(stats['max_preclip_norm'])):
            raise ValueError('stable baseline must have nonempty finite unclipped gradient norms')
        loss=report.get('loss_check',{})
        if loss.get('status')!='PASS' or loss.get('epochs_compared')!=1050 or loss.get('expected_epochs')!=1050:
            raise ValueError('Phase B requires all 1050 exact epoch losses')
        B[seed]=report
    A={}
    for arm,seed in A_CASES:
        control=read_study(root/'A'/key(arm,seed)/'none')
        binding(control,'A',arm,seed,None)
        if not control or control.get('status') not in ('REPLAY_COMPLETED','DIVERGED','FAILED_DIVERGED'):
            missing.append('A/'+key(arm,seed)+'/none')
        for tau in GRID:
            report=read_study(root/'A'/key(arm,seed)/f'tau{tau:g}')
            binding(report,'A',arm,seed,tau)
            if not report or report.get('status') not in ('REPLAY_COMPLETED','DIVERGED','FAILED_DIVERGED'):
                missing.append(f'A/{key(arm,seed)}/tau{tau:g}');continue
            A[arm,seed,tau]=report
    if missing:return dict(status='INCOMPLETE',missing=missing,grid=list(GRID),code_commit=commit)
    G=max(r['grad_clip']['run']['max_preclip_norm'] for r in B.values())
    eligible=[];checks=[]
    def consecutive(rows):
        steps=[r.get('step') for r in rows]
        return (len(steps)==60 and all(isinstance(s,int) for s in steps) and
            steps==list(range(steps[0],steps[0]+60)))
    for tau in GRID:
        survived=all(A[a,s,tau]['status']=='REPLAY_COMPLETED' and consecutive(A[a,s,tau]['rows']) and
            all(not r.get('nonfinite_norm') and isinstance(r.get('preclip_norm'),(int,float)) and
                math.isfinite(r['preclip_norm']) and r.get('loss') is not None and math.isfinite(r['loss'])
                for r in A[a,s,tau]['rows']) for a,s in A_CASES)
        checks.append(dict(tau=tau,baseline_bound=tau>=2*G,all_four_survive_60=survived))
        if tau>=2*G and survived:eligible.append(tau)
    hashes={str(p.relative_to(root)):sha(p) for p in root.glob('[AB]/**/study.json')}
    return dict(status='SELECTED' if eligible else 'NO_FEASIBLE_TAU',tau=min(eligible) if eligible else None,
        G_max=G,grid=list(GRID),checks=checks,code_commit=commit,source_digest=os.environ.get('VNA_SOURCE_DIGEST'),source_reports=hashes,
        rule='min grid tau such that tau >= 2*G_max and all four A runs survive 60 steps; no grid expansion')


def load_selection(path):
    path=Path(path);value=json.loads(path.read_text())
    if value.get('status')!='SELECTED' or value.get('grid')!=list(GRID) or value.get('tau') not in GRID:
        raise ValueError('No preregistered feasible gradient clipping threshold')
    checks=value.get('checks',[]);valid=[c['tau'] for c in checks if c.get('baseline_bound') and c.get('all_four_survive_60')]
    if not valid or min(valid)!=value['tau'] or value['tau']<2*value['G_max']:raise ValueError('clipping selection rule mismatch')
    if choose_tau(path.parent,value['code_commit'])!=value:raise ValueError('clipping study evidence changed or is incomplete')
    return value


def prepare_repository(repo,commit):
    repo=Path(repo)
    if subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==commit:return repo
    target=repo.with_name('clip_reference_'+commit[:12])
    if not target.exists():
        subprocess.run(['git','clone','--shared','--no-checkout',str(repo),str(target)],check=True)
        remote=subprocess.check_output(['git','-C',str(repo),'remote','get-url','origin'],text=True).strip()
        subprocess.run(['git','-C',str(target),'fetch',remote,commit],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',commit],check=True)
    if subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()!=commit:raise ValueError('reference checkout pin mismatch')
    for upstream in (repo/'third_party').iterdir():
        destination=target/'third_party'/upstream.name
        if upstream.is_dir() and not destination.exists():destination.symlink_to(upstream.resolve(),target_is_directory=True)
    return target


def run(args):
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True);repo=Path(__file__).resolve().parents[1]
    source=import_inputs(args.input_root,root/'_sources')
    refs=inventory(source,root/'_references')
    paused=threading.Event();lanes={};deadline=float(os.environ.get('VNA_DEADLINE',time.time()+11.5*3600))
    worker=repo/'diagnostics/grad_clip_study.py'
    def execute(gpu,mode,arm,seed,destination,tau=None,reference=None,checkpoint=None,tasks=20,epoch=None,
                reference_epochs=None,reference_repo=None,proof=None):
        existing=read_study(destination)
        if existing and existing['status'] not in ('PAUSED','ERROR','RUNNING'):
            if (existing.get('code_commit')!=subprocess.check_output(['git','-C',str(reference_repo or repo),'rev-parse','HEAD'],text=True).strip() or existing.get('source_digest')!=os.environ.get('VNA_SOURCE_DIGEST') or existing.get('tau')!=tau or existing.get('mode')!=mode or existing.get('tasks')!=tasks or existing.get('epoch')!=epoch or (reference and existing.get('reference_sha256')!=json.loads(Path(reference).read_text())['summary_sha256']) or (checkpoint and existing.get('checkpoint_sha256')!=sha(checkpoint))):
                raise ValueError('cached study identity mismatch')
            return existing
        if paused.is_set():return None
        destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,'-u',str(worker),'--repo',str(reference_repo or repo),'--support-repo',str(repo),
            '--mode',mode,'--arm',arm,'--seed',str(seed),'--out',str(destination),'--data-dir',args.data_dir,'--tasks',str(tasks)]
        for flag,value in (('--tau',tau),('--reference',reference),('--checkpoint',checkpoint),('--epoch',epoch),
                           ('--reference-epochs',reference_epochs),('--proof',proof)):
            if value is not None:cmd.extend([flag,str(value)])
        launch_logged(cmd,cwd=repo,env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONUNBUFFERED':'1',
            'VNA_LANE':str(gpu),'VNA_JOB':f'clip_{mode}_{arm}_s{seed}','PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION':'python'},
            log_path=destination/'run.log',lane=gpu,job=f'clip_{mode}_{arm}_s{seed}',lanes=lanes,paused=paused,deadline=deadline)
        return read_study(destination)
    def reference(arm,seed):
        name=key(arm,seed)
        if name not in refs:return None,None
        path=Path(refs[name]);return path,json.loads(path.read_text())
    failures=[];failure_lock=threading.Lock()
    def guarded(call):
        try:call()
        except Exception as exc:
            with failure_lock:failures.append(f'{type(exc).__name__}: {exc}')
            print('STUDY JOB ERROR:',failures[-1],flush=True)
    if args.phase=='AB':
        def phase_A():
            for arm,seed in A_CASES:
                path,ref=reference(arm,seed)
                if not ref or not ref['task1']:
                    print('MISSING TASK1 CHECKPOINT:',key(arm,seed),flush=True);continue
                for tau in (None,*GRID):
                    if paused.is_set():return
                    destination=root/'A'/key(arm,seed)/('none' if tau is None else f'tau{tau:g}')
                    guarded(lambda:execute(0,'A',arm,seed,destination,tau=tau,reference=path,checkpoint=ref['task1']['path']))
        def phase_B():
            for seed in (0,1):
                path,ref=reference('icarl',seed)
                if not ref or ref['summary']['status']!='COMPLETED' or len(ref.get('epochs') or [])!=1050:
                    print('MISSING COMPLETE BASELINE EPOCHS:',seed,flush=True);continue
                guarded(lambda:execute(1,'B','icarl',seed,root/'B'/key('icarl',seed),reference=path))
        with ThreadPoolExecutor(max_workers=2) as pool:
            a,b=pool.submit(phase_A),pool.submit(phase_B);a.result();b.result()
        selection=choose_tau(root,code_commit());atomic_json_save(selection,root/'grad_clip_selection.json')
        if selection['status']=='SELECTED':
            text=(repo/'docs/vna/PREREGISTRATION_CHANGES.md').read_text()
            text+=f"\n## Clipping selection before Phase C\nG_max={selection['G_max']!r}; tau={selection['tau']!r}.\nRule: {selection['rule']}.\nEvidence SHA256: {json.dumps(selection['source_reports'],sort_keys=True)}\n"
            (root/'PREREGISTRATION_CHANGES.md').write_text(text)
        print(json.dumps(selection,indent=2));write_report(root)
    else:
        selection=load_selection(args.selection or root/'grad_clip_selection.json');tau=selection['tau']
        prereg=Path(args.selection or root/'grad_clip_selection.json').with_name('PREREGISTRATION_CHANGES.md')
        if not prereg.exists():raise ValueError('Phase C requires the written preregistration generated with selection')
        if selection['code_commit']!=code_commit():raise ValueError('Phase C must use the A/B source commit')
        records={};lock=threading.Lock()
        def case(gpu,arm,seed):
            if paused.is_set():return
            destination=root/'C'/key(arm,seed);destination.mkdir(parents=True,exist_ok=True)
            path,ref=reference(arm,seed);status=None;proof=None
            if arm=='icarl' and seed in (0,1):
                b=read_study(root/'B'/key(arm,seed))
                if not b or not b.get('equivalent_under_clip') or b['grad_clip']['run']['max_preclip_norm']>tau:
                    raise ValueError('missing full baseline equivalence from B')
                status='REUSED';equivalence=dict(status='EQUIVALENT_PREFIX',tau=tau,loss_check=b['loss_check'],
                    n_clipped=0,n_steps_checked=b['grad_clip']['run']['n_steps'],scope='all 20 tasks; full norm replay in B',reference=ref)
                atomic_json_save(equivalence,destination/'equivalence.json')
            elif arm in ('wa','podnet','icarl_cflat') or arm=='gem' and seed in (1,2):
                status='TRAIN'
            elif ref is None:
                if arm=='icarl_upgd' and seed==2:status='TRAIN'
                else:raise ValueError('missing source '+key(arm,seed))
            else:
                complete=ref['summary']['status']=='COMPLETED'
                safe=ref['safe'];tasks=4 if complete else safe['completed']+(1 if safe['mid'] and safe['epoch'] else 0)
                epoch=None if complete or not safe['mid'] or not safe['epoch'] else safe['epoch']
                historical_epochs=path
                if not ref.get('epochs') and tasks:
                    old=prepare_repository(repo,ref['identity']['code_commit'])
                    old_result=execute(gpu,'prefix',arm,seed,destination/'reference',reference=path,tasks=tasks,epoch=epoch,reference_repo=old)
                    if not old_result or old_result['status'] not in ('PREFIX_COMPLETED','COMPLETED'):raise ValueError('historical prefix did not complete')
                    historical_epochs=destination/'reference'/'observed_epochs.json'
                elif ref.get('epochs'):
                    historical_epochs=destination/'reference_epochs.json';atomic_json_save(ref['epochs'],historical_epochs)
                if tasks:
                    result=execute(gpu,'prefix',arm,seed,destination/'prefix',tau=tau,reference=path,tasks=tasks,epoch=epoch,
                        reference_epochs=historical_epochs)
                    if result is None:return
                    equivalent=result.get('equivalent_under_clip') is True
                else:
                    equivalent=True;result=dict(loss_check=dict(status='PASS',epochs_compared=0),grad_clip=dict(run=dict(n_clipped=0)))
                    from experiments.grad_clip import GradClip
                    (destination/'prefix').mkdir(exist_ok=True);atomic_torch_save(GradClip(tau).state_dict(),destination/'prefix'/'grad_clip_state.pt')
                if equivalent:
                    proof=destination/'prefix'/'equivalence.json'
                    atomic_json_save(dict(status='EQUIVALENT_PREFIX',tau=tau,loss_check=result['loss_check'],n_clipped=0,
                        scope=dict(task=tasks,epoch=epoch),n_steps_checked=result['grad_clip']['run'].get('n_steps',0),checkpoint_sha256=safe['sha256'],reference=ref),proof)
                    status='REUSED' if complete else 'RESUME'
                else:status='TRAIN'
            if status=='REUSED':
                verification=json.loads(Path(proof or destination/'equivalence.json').read_text())
                parent=Path(ref['summary_path']).parent
                artifacts={}
                for name in ARTIFACTS-{'latest.pt','latest.previous.pt'}:
                    file=parent/name
                    if file.exists():artifacts[name]=json.loads(file.read_text()) if name.endswith('.json') else file.read_text()
                approving=read_study(root/'B'/key(arm,seed)) if arm=='icarl' and seed in (0,1) else read_study(destination/'prefix')
                atomic_json_save(dict(reference=ref,verification=verification,source_artifacts=artifacts,
                    source_summary_sha256=sha(parent/'summary.json'),source_summary_text=(parent/'summary.json').read_text(),
                    source_artifact_sha256={name:canonical_sha(value) for name,value in artifacts.items()},
                    selection=selection,selection_sha256=canonical_sha(selection),study=approving,
                    consumer_commit=code_commit(),consumer_source_digest=os.environ.get('VNA_SOURCE_DIGEST')),destination/'clip_reuse.json')
            result=None
            if status in ('TRAIN','RESUME'):
                result=execute(gpu,'train',arm,seed,destination/'run',tau=tau,reference=path,
                    checkpoint=ref['safe']['path'] if status=='RESUME' else None,proof=proof if status=='RESUME' else None)
                if result is None:return
                final_status=result['status'];metrics=result.get('grad_clip',{}).get('run',{})
            else:
                final_status='REUSED';metrics=dict(n_clipped=0,n_steps=None)
            row=dict(arm=arm,seed=seed,tau=tau,status=final_status,action=status,equivalent_under_clip=status in ('REUSED','RESUME'),
                n_clipped=metrics.get('n_clipped'),n_steps=metrics.get('n_steps'),
                clip_rate=metrics['n_clipped']/metrics['n_steps'] if metrics.get('n_steps') else 0. if status=='REUSED' else None,
                proof=str(proof) if proof else str(destination/'equivalence.json') if status=='REUSED' else None)
            atomic_json_save(row,destination/'result.json')
            with lock:records[key(arm,seed)]=row;atomic_json_save(list(records.values()),root/'phase_C.json')
        pending=Queue()
        for item in C_CASES:pending.put(item)
        def lane(gpu):
            while not paused.is_set():
                try:arm,seed=pending.get_nowait()
                except Empty:return
                try:case(gpu,arm,seed)
                except Exception as exc:
                    row=dict(arm=arm,seed=seed,tau=tau,status='ERROR',error=f'{type(exc).__name__}: {exc}')
                    destination=root/'C'/key(arm,seed);destination.mkdir(parents=True,exist_ok=True)
                    atomic_json_save(row,destination/'result.json')
                    with lock:records[key(arm,seed)]=row;atomic_json_save(list(records.values()),root/'phase_C.json')
                    with failure_lock:failures.append(row['error'])
                    print('STUDY JOB ERROR:',row['error'],flush=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            a=pool.submit(lane,0);b=pool.submit(lane,1);a.result();b.result()
        write_report(root)
    atomic_json_save(dict(errors=failures,paused=paused.is_set()),root/'study_status.json')
    if paused.is_set():raise SystemExit(75)
    if failures:raise RuntimeError('Clipping study has infrastructure errors; see study_status.json')


def write_report(root):
    root=Path(root);lines=['# Gradient clipping study','', '## Phase A','',
        '| Run | tau | Status | Failed step | Max preclip norm | Clipped |','|---|---|---|---|---|---|']
    for path in sorted(root.glob('A/**/study.json')):
        r=json.loads(path.read_text());rows=r.get('rows',[])
        norms=[p['preclip_norm'] for p in rows if p.get('preclip_norm') is not None]
        failed=r.get('outcome',{}).get('diverged') if r.get('outcome') else None
        lines.append(f"| {key(r['arm'],r['seed'])} | {r['tau']} | {r['status']} | {failed} | {max(norms,default=None)} | {sum(p['clipped'] for p in rows)} |")
    lines+=['','## Phase B and selection','']
    for path in sorted(root.glob('B/**/study.json')):
        r=json.loads(path.read_text());lines.append(f"- {key(r['arm'],r['seed'])}: {r['status']}; exact loss {r['loss_check']['status']}; grad stats {r['grad_clip']['run'] if r.get('grad_clip') else None}")
    if (root/'grad_clip_selection.json').exists():lines+=['','```json',(root/'grad_clip_selection.json').read_text(),'```']
    lines+=['','## Phase C','']
    for path in sorted(root.glob('C/*/result.json')):lines.append('- '+path.read_text())
    (root/'CLIP_STUDY_REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('AB','C'),required=True);p.add_argument('--input-root',default='/kaggle/input')
    p.add_argument('--out',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--selection')
    run(p.parse_args())
