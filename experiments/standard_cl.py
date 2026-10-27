"""BN CIFAR-100 H1 pilot; upstream head, herding and NME, isolated new loop."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from adapters.icarl_hooks import Prediction,sites_for,ICaRLHook
from diagnostics.vna_align import _global_state
from experiments.io_utils import atomic_json_save
from experiments.parallel_validity import dataset_fingerprint
from experiments.runners import pycil
from experiments.competitors import cil_runtime
from experiments.io_utils import working_directory
from experiments.vna_provenance import code_commit
from experiments.vna_resume import Progress,rng_state,restore_rng,deadline_due,log
from methods.cl_state import isolated
from methods.cl_replay import CorrectionConfig,ReplayCorrection,ReplayLoss

ARMS=dict(baseline=None,native='native',gn='gn',random='random',vna='tiny_V',h1='tiny_V',gradient='gradient')


class ExperimentLogger:
    def __init__(self,out):self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
    def save(self,name,value):atomic_json_save(value,self.out/(name+'.json'))


def cnn_metrics(learner,manager):
    net=learner._network;seen=learner._total_classes;known=learner._known_classes
    dataset=manager.get_dataset(np.arange(seen),source='test',mode='test')
    loader=DataLoader(dataset,batch_size=128,shuffle=False,num_workers=0)
    correct=torch.zeros(seen,dtype=torch.float64);counts=correct.clone();old_only=0;old_count=0
    with isolated(net),torch.no_grad():
        net.eval()
        for _,x,y in loader:
            z=net(x.to(learner._device))['logits'].cpu();pred=z.argmax(1)
            if not bool(torch.isfinite(z).all()):raise RuntimeError('nonfinite CNN evaluation logits')
            counts+=torch.bincount(y,minlength=seen)
            correct+=torch.bincount(y[pred==y],minlength=seen)
            mask=y<known
            if known:old_only+=int((z[mask,:known].argmax(1)==y[mask]).sum());old_count+=int(mask.sum())
    per_class=100*correct/counts.clamp_min(1);bounds=np.cumsum([0]+list(manager._increments))
    row=dict(global_accuracy=float(100*correct.sum()/counts.sum()),per_class=per_class.tolist(),
        old_accuracy=float(100*correct[:known].sum()/counts[:known].sum()) if known else None,
        new_accuracy=float(100*correct[known:].sum()/counts[known:].sum()),
        old_only_accuracy=100*old_only/old_count if old_count else None,
        task_accuracy=[float(per_class[bounds[t]:bounds[t+1]].mean()) for t in range(learner._cur_task+1)])
    return row


def matrix_metrics(matrix):
    if not matrix:return dict(average_incremental_accuracy=None,final_accuracy=None,forgetting=None,backward_transfer=None)
    last=matrix[-1];old=len(last)-1
    return dict(average_incremental_accuracy=float(np.mean([np.mean(row) for row in matrix])),
        final_accuracy=float(np.mean(last)),
        forgetting=float(np.mean([max(row[t] for row in matrix[t:])-last[t] for t in range(old)])) if old else 0.,
        backward_transfer=float(np.mean([last[t]-matrix[t][t] for t in range(old)])) if old else 0.)


def run(args):
    cfg=json.loads(Path(args.config).read_text());out=Path(args.out).resolve();logger=ExperimentLogger(out)
    args.data_dir=str(Path(args.data_dir).resolve())
    pycil.upstream_imports()
    from utils.data_manager import DataManager
    from models.icarl import iCaRL
    from models import icarl
    if cfg['backbone']!='resnet32' or cfg['memory_size']!=2000:raise ValueError('this pilot locks BN ResNet32 and 2000 herded exemplars')
    if args.arm not in ARMS or args.baseline not in ('er','icarl'):raise ValueError('unknown arm/baseline')
    if args.baseline=='er' and args.reference=='kd':raise ValueError('ER has no KD reference')
    if not 1<=cfg['init_cls']<=100 or cfg['increment']<1:raise ValueError('invalid class increments')
    if min(cfg['first_epochs'],cfg['later_epochs'],cfg['batch_size'])<1 or cfg['workers']<0:
        raise ValueError('invalid epoch/batch/worker configuration')
    if cfg['intervention_every']<1 or cfg['compute_budget_ratio']<=0:raise ValueError('positive cadence and resource budget required')
    if not 1<=args.tasks<=len(range(cfg['init_cls'],101,cfg['increment'])):raise ValueError('invalid task horizon')
    if args.mode=='unified' and args.arm!='baseline' and cfg['intervention_every']!=1:
        raise ValueError('unified certificates require every optimizer step; set intervention_every=1 explicitly')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if args.device!='cpu':torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    raw=json.loads((pycil.UPSTREAM/'exps/icarl.json').read_text())
    params=dict(raw,device=[torch.device(args.device)],seed=[args.seed],init_cls=cfg['init_cls'],increment=cfg['increment'],
        convnet_type='resnet32',memory_size=2000,fixed_memory=False)
    fingerprint=dataset_fingerprint(args.data_dir)
    correction_cfg=CorrectionConfig(kernel=ARMS[args.arm] or 'tiny_V',constrained=args.arm not in ('vna',),
        mode=args.mode,seed=args.seed,**cfg['correction'])
    from diagnostics.kernel_ablation import flags
    protocol=dict(benchmark='standard_cl_h1',branch='H1',code_commit=code_commit(),upstream_commit=pycil.UPSTREAM_PIN,
        source_digest=None,baseline=args.baseline,baseline_implementation='PyCIL iCaRL-style' if args.baseline=='icarl' else 'ER with PyCIL herding',
        arm=args.arm,seed=args.seed,mode=args.mode,reference=args.reference,config=cfg,correction=asdict(correction_cfg),tasks=args.tasks,
        runtime_flags=flags(),
        dataset_fingerprint=fingerprint,raw_upstream_config=raw,evaluation='final epoch; CNN primary; NME separate',
        acceptance='same train-mode window; parameter certificate only; BN buffer effects reported separately',
        evidence_status='INCONCLUSIVE',test_gates='NOT_RUN; user requested code only',
        budget_comparison=args.budget,compute_budget_ratio=cfg['compute_budget_ratio'])
    manifest=out/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('experiment protocol mismatch')
    logger.save('protocol',protocol)
    if (out/'summary.json').exists():
        previous=json.loads((out/'summary.json').read_text())
        if previous.get('protocol')!=protocol:raise ValueError('summary protocol mismatch')
        if previous.get('status')=='COMPLETED':
            if len(previous.get('tasks',[]))!=args.tasks:raise ValueError('incomplete completed summary')
            return previous
    progress=Progress(out,protocol);resume=progress.state
    if resume:progress.restore_artifacts()
    learner=None;manager=None;engine=ReplayCorrection(correction_cfg);failure_coordinates={}
    rows=[] if not resume else deepcopy(resume['rows']);epochs=[] if not resume else deepcopy(resume['epochs'])
    trajectory=[] if not resume else deepcopy(resume['trajectory']);steps=0 if not resume else resume['steps']
    last_event=0 if not resume else resume['last_event']
    task_origins={} if not resume else deepcopy(resume['task_origins'])
    training_seconds=0. if not resume else resume['training_seconds'];correction_seconds=0. if not resume else resume['correction_seconds']
    elapsed_base=0. if not resume else resume['elapsed_seconds'];started=time.monotonic()
    if resume:engine.load_state_dict(resume['engine'])
    def save(phase,epoch,optimizer=None,scheduler=None):
        logger.save('epochs',epochs);logger.save('interventions',engine.events);logger.save('trajectory',trajectory);logger.save('tasks',rows)
        progress.save(dict(phase=phase,epoch=epoch,completed=len(rows),learner=ICaRLHook.learner_state(learner),
            optimizer=None if optimizer is None else optimizer.state_dict(),scheduler=None if scheduler is None else scheduler.state_dict(),
            rng=rng_state(),engine=engine.state_dict(),rows=rows,epochs=epochs,trajectory=trajectory,steps=steps,last_event=last_event,
            task_origins=task_origins,training_seconds=training_seconds,correction_seconds=correction_seconds,
            elapsed_seconds=elapsed_base+time.monotonic()-started))

    class ContinualLearner(iCaRL):
        def _train(self,train_loader,test_loader):
            nonlocal resume,steps,last_event,training_seconds,correction_seconds
            self._network.to(self._device)
            if self._old_network is not None:self._old_network.to(self._device);self._old_network.eval()
            first=self._cur_task==0;count=cfg['first_epochs'] if first else cfg['later_epochs']
            optimizer=torch.optim.SGD(self._network.parameters(),lr=cfg['init_lr'] if first else cfg['lr'],
                momentum=cfg['momentum'],weight_decay=cfg['init_weight_decay'] if first else cfg['weight_decay'])
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['first_milestones'] if first else cfg['later_milestones'],gamma=cfg['lr_decay'])
            start=0
            if resume and resume['phase']=='train':
                vars(self).update(deepcopy(resume['learner']));self._network.to(self._device)
                if self._old_network is not None:self._old_network.to(self._device);self._old_network.eval()
                optimizer=torch.optim.SGD(self._network.parameters(),lr=cfg['init_lr'] if first else cfg['lr'],momentum=cfg['momentum'],
                    weight_decay=cfg['init_weight_decay'] if first else cfg['weight_decay'])
                scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg['first_milestones'] if first else cfg['later_milestones'],gamma=cfg['lr_decay'])
                optimizer.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler'])
                restore_rng(resume['rng']);start=resume['epoch'];resume=None
            task=self._cur_task+1;task_start=task_origins.setdefault(task,steps)
            if not any(r['task']==task for r in trajectory):trajectory.append(dict(task=task,step=0,**cnn_metrics(self,manager)))
            for epoch in range(start,count):
                if deadline_due():save('train',epoch,optimizer,scheduler);raise SystemExit(75)
                epoch_rng=rng_state();epoch_engine=engine.state_dict();epoch_steps=steps;epoch_last=last_event
                epoch_training=training_seconds;epoch_correction=correction_seconds
                epoch_model=ICaRLHook.learner_state(self);epoch_opt=deepcopy(optimizer.state_dict());epoch_sched=deepcopy(scheduler.state_dict())
                trajectory_length=len(trajectory);self._network.train();loss_sum=0.;correct=total=0;stamp=time.monotonic()
                try:
                    for _,x,y in train_loader:
                        batch_started=time.monotonic();x,y=x.to(self._device),y.to(self._device)
                        native_buffers={n:b.detach().clone() for n,b in self._network.named_buffers()}
                        teacher=None
                        if args.baseline=='icarl' and self._known_classes:
                            with torch.no_grad():teacher=self._old_network(x)['logits'].detach()
                        losses=ReplayLoss(y,y<self._known_classes,kind=args.baseline,known=self._known_classes,
                            teacher=teacher,reference=args.reference)
                        optimizer.zero_grad();z=self._network(x)['logits'];loss=losses.total(z)
                        if not bool(torch.isfinite(loss)):
                            failure_coordinates.update(task=task,epoch=epoch+1,step=steps+1,tensor='loss')
                            raise FloatingPointError(f'nonfinite loss {failure_coordinates}')
                        loss.backward();steps+=1
                        if args.mode=='intervention' or args.arm=='baseline':optimizer.step()
                        training_seconds+=time.monotonic()-batch_started
                        due=args.arm!='baseline' and steps-last_event>=cfg['intervention_every']
                        permitted=args.budget!='matched_compute' or correction_seconds<=cfg['compute_budget_ratio']*training_seconds
                        if due and permitted:
                            last_event=steps;model=Prediction(self._network)
                            event=engine.step(model,optimizer,x,losses,sites_for(model),optimizer.param_groups[0]['lr'])
                            event.update(task=task,epoch=epoch+1,step=steps,task_step=steps-task_start)
                            event['native_bn_buffer_drift']=float(sum((b.double()-native_buffers[n].double()).square().sum()
                                for n,b in self._network.named_buffers()).sqrt()) if native_buffers else 0.
                            correction_seconds+=event['seconds'];logger.save('interventions',engine.events)
                        elif due:
                            last_event=steps
                            engine.events.append(dict(task=task,epoch=epoch+1,step=steps,status='SKIPPED_RESOURCE_BUDGET',accepted=False,
                                compute_budget_ratio=cfg['compute_budget_ratio']))
                        loss_sum+=float(loss.detach());correct+=int((z.argmax(1)==y).sum());total+=len(y)
                        r=steps-task_start
                        if r in cfg['trajectory_steps']:trajectory.append(dict(task=task,step=r,**cnn_metrics(self,manager)))
                    scheduler.step()
                    epochs.append(dict(task=task,epoch=epoch+1,train_loss=loss_sum/len(train_loader),train_acc=100*correct/total,
                        lr=scheduler.get_last_lr(),seconds=time.monotonic()-stamp))
                    trajectory.append(dict(task=task,step=steps-task_start,epoch=epoch+1,**cnn_metrics(self,manager)))
                    save('train',epoch+1,optimizer,scheduler)
                    log(f'H1 {args.arm} task={task} epoch={epoch+1}/{count} loss={epochs[-1]["train_loss"]:.6f}')
                except SystemExit as exc:
                    if exc.code!=75:raise
                    vars(self).update(epoch_model);self._network.to(self._device)
                    optimizer=torch.optim.SGD(self._network.parameters(),lr=cfg['init_lr'] if first else cfg['lr'])
                    optimizer.load_state_dict(epoch_opt)
                    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[],gamma=cfg['lr_decay']);scheduler.load_state_dict(epoch_sched)
                    engine.load_state_dict(epoch_engine);steps=epoch_steps;last_event=epoch_last
                    training_seconds=epoch_training;correction_seconds=epoch_correction;del trajectory[trajectory_length:]
                    restore_rng(epoch_rng);save('train',epoch,optimizer,scheduler);raise

    status='COMPLETED';failure=None;native_loader_settings=(icarl.batch_size,icarl.num_workers)
    try:
        with working_directory(out),cil_runtime('icarl',data_dir=args.data_dir):
            manager=DataManager('cifar100',True,args.seed,cfg['init_cls'],cfg['increment'])
            logger.save('class_order',list(map(int,manager._class_order)))
            learner=ContinualLearner(params)
            if resume:
                vars(learner).update(deepcopy(resume['learner']));restore_rng(resume['rng'])
                if resume['phase']=='train':learner._cur_task-=1;learner._total_classes=learner._known_classes
                else:resume=None
            icarl.batch_size,icarl.num_workers=cfg['batch_size'],cfg['workers']
            for task in range(len(rows),args.tasks):
                learner.incremental_train(manager)
                cnn,nme=learner.eval_task();metrics=cnn_metrics(learner,manager)
                points=[r for r in trajectory if r['task']==task+1 and r['old_accuracy'] is not None]
                pre=points[0]['old_accuracy'] if points else None
                row=dict(task=task+1,**metrics,nme_accuracy=None if nme is None else float(nme['top1']),
                    stability_gap=None if pre is None else pre-min(r['old_accuracy'] for r in points),
                    recovery_step=next((r['step'] for r in points if r['step']>0 and r['old_accuracy']>=pre),None) if pre is not None else None)
                rows.append(row);learner.after_task();save('boundary',0)
    except SystemExit as exc:
        status='PAUSED' if exc.code==75 else 'INTERRUPTED';raise
    except KeyboardInterrupt as exc:status='INTERRUPTED';failure=repr(exc);raise
    except FloatingPointError as exc:status='FAILED_DIVERGED';failure=str(exc)
    except torch.cuda.OutOfMemoryError as exc:status='FAILED_RESOURCE_OOM';failure=str(exc)
    except Exception as exc:
        status='FAILED_ERROR';failure=repr(exc);raise
    finally:
        icarl.batch_size,icarl.num_workers=native_loader_settings
        elapsed=elapsed_base+time.monotonic()-started
        metrics=matrix_metrics([row['task_accuracy'] for row in rows])
        if rows:metrics.update(average_incremental_accuracy=float(np.mean([r['global_accuracy'] for r in rows])),final_accuracy=rows[-1]['global_accuracy'])
        logger.save('summary',dict(status=status,failure=failure,diverged=failure_coordinates or None,protocol=protocol,tasks=rows,
            **metrics,training_seconds=training_seconds,correction_seconds=correction_seconds,
            wall_seconds=elapsed,gpu_hours=elapsed/3600 if args.device!='cpu' else 0.,evidence_status='INCONCLUSIVE',
            native_steps=steps,nonzero_corrections=sum(e.get('step_norm',0)>0 for e in engine.events),
            parameter_count=sum(p.numel() for p in learner._network.parameters()) if learner is not None else None,
            peak_vram_bytes=torch.cuda.max_memory_allocated(torch.device(args.device)) if args.device!='cpu' else 0,
            nme_average_incremental_accuracy=float(np.mean([r['nme_accuracy'] for r in rows])) if rows and all(r['nme_accuracy'] is not None for r in rows) else None,
            nme_final_accuracy=rows[-1]['nme_accuracy'] if rows else None,
            extra_updates_certified_only_at_intervention=args.mode=='intervention'))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/standard_cl/h1_cifar100_bn.json');p.add_argument('--data-dir',required=True)
    p.add_argument('--out',required=True);p.add_argument('--arm',choices=ARMS,default='h1')
    p.add_argument('--baseline',choices=('er','icarl'),default='icarl');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--reference',choices=('replay_ce_kd','kd'),default='replay_ce_kd')
    p.add_argument('--mode',choices=('intervention','unified'),default='intervention')
    p.add_argument('--budget',choices=('matched_update','matched_compute'),default='matched_update')
    p.add_argument('--tasks',type=int,default=3);p.add_argument('--device',default='cuda:0')
    return p


if __name__=='__main__':run(parser().parse_args())
