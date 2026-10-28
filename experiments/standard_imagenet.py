"""H3: pinned CBP Continual ImageNet32 binary stream, versus VNA -> GIR."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import random
import time
import numpy as np
import torch
from torch import nn
from adapters.lop_hooks import sites_for,learner_state,restore_learner
from experiments.runners import lop
from experiments.standard_cl import ExperimentLogger
from experiments.vna_resume import Progress,rng_state,restore_rng,deadline_due,log
from experiments.vna_provenance import code_commit
from methods.cl_replay import CorrectionConfig,ReplayCorrection,ReplayLoss
from methods.cl_h3 import TangentReconfiguration
from methods.cl_state import isolated


class ImageNetwork(nn.Module):
    def __init__(self,native):super().__init__();self.native=native
    @property
    def layers(self):return self.native.layers
    @property
    def act_type(self):return self.native.act_type
    @property
    def last_filter_output(self):return self.native.last_filter_output
    def predict(self,x):return self.native.predict(x)
    def forward(self,x):return self.predict(x)[0]


def load_task(root,classes,train_count=600,test_count=100):
    train=[];test=[]
    for label in classes:
        values=np.load(Path(root)/f'{int(label)}.npy',allow_pickle=False)
        if values.shape!=(train_count+test_count,3,32,32):raise ValueError(f'ImageNet32 class {label}: unexpected shape {values.shape}')
        train.append(values[:train_count]);test.append(values[train_count:])
    return (torch.tensor(np.concatenate(train),dtype=torch.float32),torch.arange(len(classes)).repeat_interleave(train_count),
            torch.tensor(np.concatenate(test),dtype=torch.float32),torch.arange(len(classes)).repeat_interleave(test_count))


def data_identity(root,order):
    result={}
    for label in sorted(set(map(int,order))):
        path=Path(root)/f'{label}.npy';digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
        result[str(label)]=digest.hexdigest()
        if len(result)%100==0:log(f'ImageNet dataset fingerprint: {len(result)}/1000 classes')
    return result


def run(args):
    lop.upstream_imports()
    from lop.nets.conv_net import ConvNet
    from lop.algos.bp import Backprop
    from lop.algos.convCBP import ConvCBP
    from diagnostics.kernel_ablation import flags
    cfg=json.loads(Path(args.config).read_text())
    source=lop.UPSTREAM/cfg['upstream_config'];params=json.loads(source.read_text())
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    with (lop.UPSTREAM/'lop/imagenet/class_order').open('rb') as stream:orders=pickle.load(stream)
    order=np.asarray(orders[args.seed],dtype=int)
    if set(order)!=set(range(1000)):raise ValueError('upstream class order must be a permutation of 1000 classes')
    identity=data_identity(args.data_dir,order)
    if args.device!='cpu':torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    with torch.device(args.device):
        model=ImageNetwork(ConvNet(num_classes=params['num_classes']))
        common=dict(net=model,step_size=params['step_size'],momentum=params['momentum'],loss='nll',opt=params['opt'],device=args.device)
        learner=ConvCBP(**common,init='default',replacement_rate=params['replacement_rate'],decay_rate=params['decay_rate'],
            util_type=params['util_type'],maturity_threshold=params['maturity_threshold']) if args.arm=='cbp' else Backprop(**common)
    correction=CorrectionConfig(seed=args.seed,**cfg['correction'])
    vna=ReplayCorrection(correction);gir=TangentReconfiguration(correction)
    # Existing training statistics require equal spatial grids; pooling sites are measurement-only.
    all_sites=[site for site in sites_for(model) if site.kind=='linear']
    gir_sites=all_sites
    protocol=dict(branch='H3',arm=args.arm,seed=args.seed,code_commit=code_commit(),upstream_commit=lop.UPSTREAM_PIN,
        config=cfg,params=params,tasks=params['num_tasks'],upstream_recipe=str(source.relative_to(lop.UPSTREAM)),
        correction=asdict(correction),dataset='Continual ImageNet32',data_sha256=identity,class_order=order.tolist(),
        split=dict(train_per_class=600,test_per_class=100),normalization='upstream raw float conversion; no extra normalization',
        head_reset='zero final weight/bias at every task; retain optimizer momentum, as upstream',runtime_flags=flags(),
        vna_sites=[s.module_name for s in all_sites],gir_sites=[s.module_name for s in gir_sites],device=args.device,evidence_status='INCONCLUSIVE')
    logger=ExperimentLogger(args.out);out=Path(args.out)
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('ImageNet protocol mismatch')
    logger.save('protocol',protocol);progress=Progress(out,protocol)
    if (out/'summary.json').exists():
        previous=json.loads((out/'summary.json').read_text())
        if previous.get('protocol')!=protocol:raise ValueError('ImageNet summary protocol mismatch')
        if previous.get('status')=='COMPLETED':
            if len(previous.get('tasks',[]))!=params['num_tasks']:raise ValueError('incomplete ImageNet summary')
            return previous
    task=epoch=steps=last_event=0;rows=[];epochs=[];elapsed=0.;window=[];task_rng=None
    if progress.state:
        s=progress.state;progress.restore_artifacts();restore_learner(learner,s['learner'])
        if args.arm=='cbp':
            for name,value in vars(learner.gnt).items():
                if isinstance(value,list):setattr(learner.gnt,name,[v.to(args.device) if torch.is_tensor(v) else v for v in value])
        vna.load_state_dict(s['vna']);gir.load_state_dict(s['gir']);restore_rng(s['rng'])
        task,epoch,steps,last_event=s['task'],s['epoch'],s['steps'],s['last_event']
        rows,epochs,elapsed,window,task_rng=s['rows'],s['epochs'],s['elapsed_seconds'],s['window'],s['task_rng']
    started=time.monotonic();status='COMPLETED';failure=None;coordinates=None
    def save():
        logger.save('tasks',rows);logger.save('epochs',epochs);logger.save('interventions',gir.events);logger.save('vna_interventions',vna.events)
        progress.save(dict(learner=learner_state(learner),vna=vna.state_dict(),gir=gir.state_dict(),rng=rng_state(),
            task=task,epoch=epoch,steps=steps,last_event=last_event,rows=rows,epochs=epochs,window=window,task_rng=task_rng,
            elapsed_seconds=elapsed+time.monotonic()-started))
    try:
        while task<params['num_tasks']:
            labels=np.tile(order,params['num_classes']*params['num_tasks']//1000+1)[task*2:(task+1)*2]
            x,y,test_x,test_y=load_task(args.data_dir,labels)
            x,y,test_x,test_y=(v.to(args.device) for v in (x,y,test_x,test_y))
            if task_rng is None:
                task_rng=rng_state()
                with torch.no_grad():model.layers[-1].weight.zero_();model.layers[-1].bias.zero_()
                window=[]
            # Reconstruct upstream's cumulative in-place epoch permutations without consuming resumed RNG.
            if epoch:
                current_rng=rng_state();restore_rng(task_rng)
                for _ in range(epoch):
                    indices=np.random.permutation(len(y));x,y=x[indices],y[indices]
                restore_rng(current_rng)
            while epoch<params['num_showings']:
                if deadline_due():save();raise SystemExit(75)
                snapshot=learner_state(learner);epoch_rng=rng_state();old_window=deepcopy(window)
                old_vna,old_gir=vna.state_dict(),gir.state_dict();old_steps,old_last=steps,last_event
                old_x,old_y=x,y
                try:
                    indices=np.random.permutation(len(y));x,y=x[indices],y[indices]
                    train_acc=[];losses_epoch=[]
                    for start in range(0,len(y),params['mini_batch_size']):
                        bx,by=x[start:start+params['mini_batch_size']],y[start:start+params['mini_batch_size']]
                        with torch.device(args.device):loss,z=learner.learn(bx,by)
                        if not bool(torch.isfinite(loss)):
                            coordinates=dict(task=task+1,epoch=epoch+1,step=steps+1,tensor='loss')
                            raise FloatingPointError(str(coordinates))
                        train_acc.append(float((z.argmax(1)==by).float().mean())*100);losses_epoch.append(float(loss));steps+=1
                        window.append((bx.detach().cpu().clone(),by.detach().cpu().clone()))
                        while sum(len(v[1]) for v in window)>cfg['window']:
                            extra=sum(len(v[1]) for v in window)-cfg['window']
                            if extra>=len(window[0][1]):window.pop(0)
                            else:window[0]=(window[0][0][extra:],window[0][1][extra:])
                        if args.arm=='vna_gir' and steps-last_event>=cfg['intervention_every']:
                            wx=torch.cat([v[0] for v in window]).to(args.device);wy=torch.cat([v[1] for v in window]).to(args.device)
                            losses=ReplayLoss(wy,torch.zeros_like(wy,dtype=torch.bool),kind='er')
                            event=vna.step(model,learner.opt,wx,losses,all_sites,params['step_size'])
                            event.update(task=task+1,epoch=epoch+1,step=steps)
                            event=gir.step(model,learner.opt,wx,losses,gir_sites);event.update(task=task+1,epoch=epoch+1,step=steps)
                            last_event=steps
                        if deadline_due():raise SystemExit(75)
                    with isolated(model),torch.no_grad():
                        model.eval();correct=0
                        for start in range(0,len(test_y),params['mini_batch_size']):
                            z=model(test_x[start:start+params['mini_batch_size']])
                            if not bool(torch.isfinite(z).all()):raise FloatingPointError('nonfinite ImageNet evaluation logits')
                            correct+=int((z.argmax(1)==test_y[start:start+params['mini_batch_size']]).sum())
                    epochs.append(dict(task=task+1,epoch=epoch+1,train_accuracy=float(np.mean(train_acc)),
                        test_accuracy=100*correct/len(test_y),loss=float(np.mean(losses_epoch))))
                except SystemExit as exc:
                    if exc.code!=75:raise
                    restore_learner(learner,snapshot);restore_rng(epoch_rng);window=old_window
                    vna.load_state_dict(old_vna);gir.load_state_dict(old_gir);steps,last_event=old_steps,old_last;x,y=old_x,old_y
                    save();raise
                epoch+=1
                if epoch%25==0:save()
                log(f'H3 {args.arm} task={task+1}/{params["num_tasks"]} epoch={epoch}/{params["num_showings"]} acc={epochs[-1]["test_accuracy"]:.2f}')
            selected=epochs[-params['num_showings']:]
            rows.append(dict(task=task+1,classes=labels.tolist(),final_accuracy=selected[-1]['test_accuracy'],
                mean_accuracy=float(np.mean([r['test_accuracy'] for r in selected])),mean_error=float(np.mean([100-r['test_accuracy'] for r in selected])),
                early_accuracy=selected[0]['test_accuracy'],late_accuracy=selected[-1]['test_accuracy']))
            task+=1;epoch=0;task_rng=None;save()
    except SystemExit as exc:status='PAUSED' if exc.code==75 else 'INTERRUPTED';raise
    except FloatingPointError as exc:status='FAILED_DIVERGED';failure=str(exc)
    except BaseException as exc:status='FAILED_ERROR';failure=repr(exc);raise
    finally:
        seconds=elapsed+time.monotonic()-started
        logger.save('summary',dict(protocol=protocol,status=status,failure=failure,diverged=coordinates,tasks=rows,
            last_50_task_error=float(np.mean([r['mean_error'] for r in rows[-50:]])) if rows else None,
            plasticity_decay=rows[0]['late_accuracy']-rows[-1]['late_accuracy'] if rows else None,
            final_accuracy=rows[-1]['final_accuracy'] if rows else None,wall_seconds=seconds,
            correction_seconds=sum(r['seconds'] for r in vna.events)+sum(r['seconds'] for r in gir.events),
            parameter_count=sum(p.numel() for p in model.parameters()),gpu_hours=seconds/3600 if args.device!='cpu' else 0.,evidence_status='INCONCLUSIVE'))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/standard_cl/h3_imagenet.json');p.add_argument('--out',required=True)
    p.add_argument('--arm',choices=('cbp','vna_gir'),required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--data-dir',required=True);p.add_argument('--device',default='cuda:0')
    return p


if __name__=='__main__':run(parser().parse_args())
