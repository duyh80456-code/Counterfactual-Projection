"""One isolated native training/replay process for the registered clipping study."""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


class ReplayComplete(Exception):pass


def import_support(root):
    path=Path(root)/'experiments/grad_clip.py'
    spec=importlib.util.spec_from_file_location('clip_study_support',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def compare_epochs(reference,current,task,epoch=None):
    def take(rows):return [r for r in rows if r['task']<=task and
        (epoch is None or r['task']<task or r['epoch']<=epoch)]
    old=take(reference);new=take(current)
    expected={(r['task'],r['epoch']):r['train_loss'] for r in old}
    observed={(r['task'],r['epoch']):r['train_loss'] for r in new}
    differences=[dict(task=k[0],epoch=k[1],reference=expected.get(k),current=observed.get(k))
        for k in sorted(expected.keys()|observed.keys()) if k not in expected or k not in observed or expected[k]!=observed[k]]
    count=100+50*(task-1) if epoch is None else (100+50*(task-2)+epoch if task>1 else epoch)
    coverage={(t,e) for t in range(1,task+1) for e in range(1,
        (epoch if t==task and epoch is not None else 100 if t==1 else 50)+1)}
    passed=(len(old)==len(new)==count and set(expected)==set(observed)==coverage and not differences and
        all(isinstance(v,(int,float)) and math.isfinite(v) for v in (*expected.values(),*observed.values())))
    return dict(status='PASS' if passed else 'FAIL',epochs_compared=len(new),expected_epochs=count,
        absolute_tolerance=0.,relative_tolerance=0.,differences=differences)


def mathematical_identity(identity):
    excluded={'code_commit','source_digest','data_dir','resume','baseline_reference_sha256','grad_clip',
              'parameter_count','mu0','schedule','probe','feature_probe'}
    result={k:v for k,v in identity.items() if k not in excluded and v is not None}
    result['cfg']={k:v for k,v in result.get('cfg',{}).items() if k!='device'}
    schedule=identity.get('schedule')
    if schedule:result['schedule']={k:v for k,v in schedule.items() if k not in ('code_commit','source_digest')}
    return result


def load_state(torch,source):
    return torch.load(source,map_location='cpu',weights_only=False)


def run(args):
    repo=Path(args.repo).resolve();sys.path.insert(0,str(repo))
    import torch
    support=import_support(args.support_repo)
    from experiments.io_utils import atomic_json_save,atomic_torch_save
    from experiments.runners.pycil import upstream_imports
    upstream_imports()
    from experiments import main_cil as main,v4_competitors as competitors
    from experiments.vna_provenance import code_commit
    from experiments.vna_resume import Progress
    descriptor=json.loads(Path(args.reference).read_text()) if args.reference else None
    identity=descriptor['identity'] if descriptor else None
    if identity and (identity['arm'],identity['seed'])!=(args.arm,args.seed):raise ValueError('reference arm/seed mismatch')
    is_main=args.arm in ('icarl','icarl_full')
    native=main if is_main else competitors
    historical=repo!=Path(args.support_repo).resolve()
    if historical and identity and code_commit()!=identity['code_commit']:raise ValueError('historical reference commit mismatch')
    if historical and args.tau is not None:raise ValueError('historical producer is a no-clip reference only')
    cli=['--out',str(Path(args.out).resolve()),'--device',args.device,'--seed',str(args.seed),
         '--data-dir',args.data_dir,'--nobn-lr-scale','.5', '--arm' if is_main else '--method',args.arm]
    if is_main:cli+=['--backbone','resnet32_nobn','--class-order','superclass','--solver','gram','--curvature','power10']
    else:cli+=['--backbone','nobn','--order','superclass','--resume']
    if not historical and args.tau is not None:cli+=['--grad-clip',str(args.tau)]
    parsed=native.parser().parse_args(cli)
    out=Path(parsed.out);out.mkdir(parents=True,exist_ok=True)
    schedule=identity.get('schedule') if identity else None
    if is_main and args.arm=='icarl_full':
        if not schedule:raise ValueError('full replay requires the original frozen Gram K0 authority')
        if (identity.get('solver'),identity.get('curvature'),identity.get('gir_live'))!=('gram','power10',False):
            raise ValueError('full replay requires registered Gram/power10/dead-only identity')
        schedule=copy.deepcopy(schedule);schedule['code_commit']=code_commit();schedule['source_digest']=os.environ.get('VNA_SOURCE_DIGEST')
        atomic_json_save(schedule,out/'schedule.json');parsed.schedule=str(out/'schedule.json')
    if not is_main and identity:parsed.config=json.dumps(identity['method_config'])
    reference_epochs=descriptor.get('epochs') if descriptor else None
    if args.reference_epochs:reference_epochs=json.loads(Path(args.reference_epochs).read_text())
    if args.mode=='B' and not reference_epochs:raise ValueError('Phase B requires full-precision original epoch losses')
    rows=[];epochs=[];holder={};original_init=None
    observer_clip=support.GradClip(None) if historical else None
    def step_observer(row,learner):
        with torch.no_grad():
            row['parameter_norm_after']=support.scalar(torch.linalg.vector_norm(torch.stack([p.detach().norm(2) for p in learner._network.parameters()]),2))
        rows.append(copy.deepcopy(row))
        if args.mode=='A' and row['task']==2 and sum(r['task']==2 for r in rows)>=60:raise ReplayComplete()
    def epoch_observer(row):
        epochs.append(copy.deepcopy(row))
        if args.mode=='prefix' and row['task']==args.tasks and (args.epoch is None or row['epoch']==args.epoch):
            limit=100 if args.tasks==1 else 50
            if args.epoch is not None or row['epoch']==limit:raise ReplayComplete()
    parsed._step_observer=step_observer if args.mode=='A' else None;parsed._epoch_observer=epoch_observer
    if is_main:
        original_init=main.MainHook.__init__
        def init(hook,*a,**kw):
            original_init(hook,*a,**kw);holder['hook']=hook
        main.MainHook.__init__=init
    original_progress=native.Progress
    if args.checkpoint:
        if args.mode not in ('A','train'):raise ValueError('only A or authorized continuation can import a checkpoint')
        state=load_state(torch,args.checkpoint)
        source=state['identity']
        if source.get('grad_clip') is not None:raise ValueError('source must be the unclipped safe checkpoint')
        if args.mode=='A':
            if is_main:
                if state.get('completed')!=1 or state.get('mid'):raise ValueError('Phase A needs a completed task1 boundary')
            elif len(state['tasks'])!=1 or state['model']._cur_task!=0:raise ValueError('Phase A needs completed task1')
        else:
            proof=json.loads(Path(args.proof).read_text()) if args.proof else {}
            if (proof.get('status')!='EQUIVALENT_PREFIX' or proof.get('tau')!=args.tau or
                proof.get('checkpoint_sha256')!=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()):
                raise ValueError('checkpoint continuation requires its exact no-clip prefix proof')
        class ImportedProgress(Progress):
            def __init__(self,directory,current,*a,**kw):
                directory=Path(directory)
                if not (directory/'latest.pt').exists():
                    if mathematical_identity(source)!=mathematical_identity(current):raise ValueError('checkpoint mathematical identity mismatch')
                    converted=copy.deepcopy(state);converted['identity']=current
                    if args.mode=='A':
                        converted.get('hook',{}).pop('grad_clip_state',None)
                        converted.pop('grad_clip',None)
                    if is_main:
                        converted['artifacts']={k:v for k,v in converted.get('artifacts',{}).items() if k not in ('protocol.json','grad_clip.json')}
                        if args.mode=='train':converted['hook']['grad_clip_state']=load_state(torch,Path(args.proof).with_name('grad_clip_state.pt'))
                    elif args.mode=='train':converted['grad_clip']=load_state(torch,Path(args.proof).with_name('grad_clip_state.pt'))
                    atomic_torch_save(converted,directory/'latest.pt')
                    atomic_json_save(dict(source=str(args.checkpoint),source_commit=source['code_commit'],
                        source_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),mode=args.mode,
                        proof=args.proof),directory/'checkpoint_import.json')
                super().__init__(directory,current,*a,**kw)
        native.Progress=ImportedProgress
    original_runtime=None;old_before=old_epoch=None
    if historical:
        if is_main:
            old_before,old_epoch=main.MainHook.before_optimizer_step,main.MainHook.after_epoch
            def before(hook,frame):
                old_before(hook,frame)
                observer_clip.apply((p.grad for p in hook.learner._network.parameters()),task=hook.learner._cur_task+1,
                    step=hook.steps,epoch=int(frame.f_locals.get('epoch',0))+1)
            def after(hook,learner,optimizer,frame):
                old_epoch(hook,learner,optimizer,frame);epoch_observer(hook.epochs[-1])
            main.MainHook.before_optimizer_step,main.MainHook.after_epoch=before,after
        else:
            original_runtime=competitors.runtime
            def observe_optimizer(opt,frame,gradients=None):
                learner=frame.f_locals['self']
                observer_clip.apply(gradients if gradients is not None else
                    (p.grad for group in opt.param_groups for p in group['params']),task=learner._cur_task+1,
                    step=sum(len(v) for v in observer_clip.tasks.values())+1,epoch=int(frame.f_locals['epoch'])+1)
            if args.arm=='icarl_upgd':
                upstream=competitors.source_module('v4_upgd','upgd/core/optim/weight_upgd/first_order.py')
                upstream.FirstOrderGlobalUPGD.step=lambda opt:support.upgd_step(opt,observe_optimizer)
            @contextmanager
            def runtime(*a,**kw):
                a=list(a);old_done=a[4]
                def done():
                    old_done();frame=support.training_frame();learner=frame.f_locals['self']
                    epoch_observer(dict(task=learner._cur_task+1,epoch=int(frame.f_locals['epoch'])+1,
                        train_loss=frame.f_locals['losses']/len(learner.train_loader)))
                a[4]=done
                with original_runtime(*a,**kw) as pair,support.optimizer_clipping(pair[0],observe_optimizer):yield pair
            competitors.runtime=runtime
    outcome=None;status='RUNNING';error=None
    try:
        outcome=native.run(parsed)
        status=outcome['status']
    except ReplayComplete:status='REPLAY_COMPLETED' if args.mode=='A' else 'PREFIX_COMPLETED'
    except SystemExit as exc:
        if exc.code!=75:raise
        status='PAUSED'
    except Exception as exc:
        status='ERROR';error=f'{type(exc).__name__}: {exc}'
    finally:
        if original_init:main.MainHook.__init__=original_init
        native.Progress=original_progress
        if original_runtime:competitors.runtime=original_runtime
        if old_before:main.MainHook.before_optimizer_step,main.MainHook.after_epoch=old_before,old_epoch
    clip=observer_clip
    if is_main and 'hook' in holder:epochs=copy.deepcopy(holder['hook'].epochs)
    if not historical:
        if is_main and 'hook' in holder:clip=holder['hook'].clipper
        elif (out/'grad_clip.json').exists():
            # v4 saves the complete state at its safe checkpoint; completed prefix may be later.
            clip=getattr(parsed,'_clipper',None)
    if clip is not None:
        clip.save(out/'grad_clip.json');atomic_torch_save(clip.state_dict(),out/'grad_clip_state.pt')
        if not rows:rows=copy.deepcopy(clip.steps)
    if (out/'epochs.json').exists():
        saved=json.loads((out/'epochs.json').read_text())
        if len(saved)>len(epochs):epochs=saved
    atomic_json_save(epochs,out/'observed_epochs.json')
    clip_report=json.loads((out/'grad_clip.json').read_text()) if (out/'grad_clip.json').exists() else None
    if args.mode=='A':
        selected=[r for r in rows if r['task']==2]
        # Include a failed pre-step row, which never reaches the post-update observer.
        if clip is not None:
            existing={r['step'] for r in selected}
            selected+=copy.deepcopy([r for r in clip.steps if r['task']==2 and r['step'] not in existing])
        rows=sorted(selected,key=lambda r:r['step'])
    check=compare_epochs(reference_epochs,epochs,args.tasks,args.epoch) if reference_epochs else dict(status='MISSING_REFERENCE')
    passed=status in ('COMPLETED','PREFIX_COMPLETED') and check['status']=='PASS' and clip_report is not None and clip_report['run']['n_clipped']==0
    report=dict(mode=args.mode,arm=args.arm,seed=args.seed,tau=args.tau,status=status,error=error,tasks=args.tasks,epoch=args.epoch,
        code_commit=code_commit(),source_digest=os.environ.get('VNA_SOURCE_DIGEST'),reference_commit=identity.get('code_commit') if identity else None,
        reference_sha256=descriptor.get('summary_sha256') if descriptor else None,rows=rows,
        loss_check=check,grad_clip=clip_report,diagnostic_only=args.mode!='train',
        equivalent_under_clip=passed if args.mode in ('B','prefix') else None,
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
        outcome=outcome)
    atomic_json_save(report,out/'study.json')
    print(json.dumps({k:report[k] for k in ('mode','arm','seed','tau','status','error','equivalent_under_clip')}),flush=True)
    if status=='PAUSED':raise SystemExit(75)
    if status=='ERROR':raise RuntimeError(error)
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--support-repo',required=True)
    p.add_argument('--mode',choices=('A','B','prefix','train'),required=True)
    p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--tau',type=float);p.add_argument('--tasks',type=int,default=20);p.add_argument('--epoch',type=int)
    p.add_argument('--reference');p.add_argument('--reference-epochs');p.add_argument('--checkpoint');p.add_argument('--proof')
    p.add_argument('--out',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--device',default='cuda:0')
    return p


if __name__=='__main__':run(parser().parse_args())
