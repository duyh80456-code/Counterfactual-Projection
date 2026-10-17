"""Post-optimizer v3 hooks on official CCL-DC ER/ER-ACE, without copying training."""
import copy
import inspect
import json
import random
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from adapters.vna_v3_hooks import Calibration, evaluation
from experiments import stage_b5 as io
from experiments.vna_resume import Progress, Status, rng_state, restore_rng
from methods.vna_losses import CE, _cov
from methods.vna_sites import HookSite

UPSTREAM = Path(__file__).resolve().parents[1]/'third_party/CCL-DC'
UPSTREAM_PIN = '1d2cf9bdb6ae0e82e0eb05a1a240271a02dfafe9'
UPSTREAM_URL = 'https://github.com/maorong-wang/CCL-DC'


def upstream_imports():
    if not (UPSTREAM/'src/learners/baseline/er.py').exists():
        raise FileNotFoundError('pinned official CCL-DC absent; run the notebook dependency preparation')
    actual = subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    if actual != UPSTREAM_PIN: raise ValueError('CCL-DC commit mismatch')
    if str(UPSTREAM) not in sys.path: sys.path.insert(0,str(UPSTREAM))


class Prediction(nn.Module):
    def __init__(self,native):
        super().__init__(); self.native=native; self.virtual_relu=nn.ReLU()
    def forward(self,x): return self.native.logits(x)


def sites_for(model):
    from src.models.resnet import BasicBlock
    return [HookSite(model,f'{name}.conv1',f'{name}.conv2','conv',
                     between=(f'{name}.bn1','virtual_relu'))
            for name,module in model.named_modules() if isinstance(module,BasicBlock)]


class OnlineCE(CE):
    """Exact ER-ACE masked stream CE plus separately averaged replay CE."""
    def loss(self,z,ctx):
        split=ctx['stream_size']; mask=ctx['stream_mask']
        value=F.cross_entropy(z[:split].masked_fill(~mask,-1e9),ctx['targets'][:split])
        if split<len(z): value=value+F.cross_entropy(z[split:],ctx['targets'][split:])
        return value
    def metric_apply(self,z,v,ctx):
        split=ctx['stream_size']; mask=ctx['stream_mask']
        out=torch.zeros_like(z,dtype=torch.float64)
        probabilities=z[:split].double().masked_fill(~mask,-1e9).softmax(-1)
        out[:split]=_cov(probabilities,v[:split].double())/split
        if split<len(z): out[split:]=_cov(z[split:].double().softmax(-1),v[split:].double())/(len(z)-split)
        return out


def native_params(args):
    upstream_imports()
    from config.parser import Parser
    name='ERACE' if args.arm.startswith('erace') else 'ER'
    path=UPSTREAM/f'config/CVPR24/cifar100/{name},c100,m2000.yaml'
    saved=sys.argv
    try:
        sys.argv=[saved[0]]
        params=Parser().parse(['--config',str(path),'--no-wandb'])
    finally: sys.argv=saved
    params.seed=args.seed; params.run_id=args.seed; params.no_wandb=True
    params.data_root_dir=args.data_dir or './data'
    # Worker zero makes loader-prefix replay independent of worker prefetch RNG.
    params.num_workers=0
    if getattr(args,'online_nf',64)!=64:
        params.nf=args.online_nf; params.dim_in=8*params.nf
    if getattr(args,'online_tasks',10)!=10: params.n_tasks=args.online_tasks
    if getattr(args,'online_memory',2000)!=2000: params.mem_size=args.online_memory
    return params,path


class NativeStepHook(Calibration):
    def __init__(self,engine,learner,*,preflight=False):
        super().__init__(preflight)
        self.engine,self.learner=engine,learner
        self.iteration=0; self.task=0; self.task_losses={}; self.task_accuracies={}
        self.last_training_seconds=0.

    def state_dict(self):
        return {key:copy.deepcopy(value) for key,value in vars(self).items()
                if key not in ('engine','learner')}|{'engine':self.engine.state_dict()}

    def load_state_dict(self,state):
        self.engine.load_state_dict(state['engine'])
        for key,value in state.items():
            if key!='engine': setattr(self,key,value)

    def optimizer_step(self,original,*args,**kwargs):
        frame=inspect.currentframe().f_back
        while frame and not (frame.f_code.co_name=='train' and frame.f_locals.get('self') is self.learner):
            frame=frame.f_back
        if frame is None: return original(*args,**kwargs)
        values=frame.f_locals
        result=original(*args,**kwargs)
        self.iteration+=1
        self.task=values['task_id']+1
        self.engine.begin_task(self.task)
        if 'combined_x' in values:
            batch=values['combined_x'],values['combined_y'].long()
            spec=CE(); ctx={'targets':batch[1]}
            logits=values['logits']
        else:
            xs,ys=values['aug_xs'],values['batch_y'].to(self.learner.device)
            if values['mem_x'].size(0):
                batch=torch.cat((xs,values['aug_xm'])),torch.cat((ys,values['mem_y'].to(ys.device)))
            else: batch=xs,ys
            spec=OnlineCE();ctx={'targets':batch[1],'stream_size':len(xs),'stream_mask':values['mask'].bool()}
            logits=values['logits_stream']
        model=Prediction(self.learner.model)
        ctx['classes']=self.task*(self.learner.params.n_classes//self.learner.params.n_tasks)
        count=len(values['dataloader'])
        self.full_batches=[count]
        correction_started=time.monotonic()
        self.apply(self.engine,model,sites_for(model),batch,spec,ctx,self.learner.optim,
                   self.iteration,max(0,count-getattr(values['dataloader'],'skip',0)-values['j']-1))
        self.last_training_seconds=max(0.,time.monotonic()-self.iteration_started-(time.monotonic()-correction_started))
        if len(self.iteration_times)<200: self.iteration_times.append(self.last_training_seconds)
        self.task_losses.setdefault(self.task,[]).append(float(values['loss'].detach()))
        stream_size=len(values['batch_y'])
        observed_logits=logits[-stream_size:] if 'combined_x' in values else logits
        correct=float((observed_logits.detach().argmax(1)==values['batch_y'].to(logits.device)).float().mean())
        self.task_accuracies.setdefault(self.task,[]).append(correct)
        return result


class ResumeLoader:
    def __init__(self,native,hook,*,skip=0,after_prefix=None,after_batch=None):
        self.native,self.hook,self.skip=native,hook,skip
        self.after_prefix,self.after_batch=after_prefix,after_batch
    def __len__(self): return len(self.native)
    def __iter__(self):
        iterator=iter(self.native)
        for _ in range(self.skip): next(iterator)
        if self.after_prefix: self.after_prefix()
        for index,batch in enumerate(iterator,start=self.skip):
            self.hook.iteration_started=time.monotonic()
            yield batch
            if self.after_batch: self.after_batch(index+1)


def run_online(args):
    from experiments.pilot_v3 import make_engine, protocol, finalize_calibration, write_events, online_model_config
    params,cfg_path=native_params(args)
    if args.device.startswith('cuda'): torch.cuda.set_device(args.device)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    if params.learner=='ER':
        from src.learners.baseline.er import ERLearner as Learner
    else:
        from src.learners.baseline.er_ace import ER_ACELearner as Learner
    learner=Learner(params)
    if args.fake_data:
        from torch.utils.data import DataLoader, TensorDataset
        generator=torch.Generator().manual_seed(871)
        if params.labels_order is None:
            params.labels_order=np.random.permutation(params.n_classes).tolist()
        loaders={}
        classes_per_task=params.n_classes//params.n_tasks
        for task in range(params.n_tasks):
            labels=torch.tensor(params.labels_order[task*classes_per_task:(task+1)*classes_per_task]).repeat(5)
            images=torch.rand(len(labels),3,32,32,generator=generator)
            for split in ('train','test'):
                loaders[f'{split}{task}']=DataLoader(TensorDataset(images,labels),batch_size=10,shuffle=True)
    else:
        from src.utils.data import get_loaders
        loaders=get_loaders(params)
    engine=make_engine(args)
    hook=NativeStepHook(engine,learner,preflight=args.preflight_only)
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    info=protocol(args,engine,vars(params),UPSTREAM_PIN)
    from experiments.parallel_validity import dataset_fingerprint
    info.update(model_config=online_model_config(args),
                dataset_fingerprint=None if args.fake_data else dataset_fingerprint(params.data_root_dir),
                expected_tasks=2 if args.fake_data or args.preflight_only else params.n_tasks)
    info.update(upstream_url=UPSTREAM_URL,cfg_source=str(cfg_path.relative_to(UPSTREAM)),
                reduced_backbone_source='config/parser.py -nf: nf=20 reduced ResNet18',
                resume='native loader prefix replay; model, optimizer, reservoir, RNG, VNA state after stream batch',
                actual_window='same augmented native combined minibatch; ER-ACE separate CE means and stream mask',
                first_stream_batch='official ER skips optimization with empty memory; no algorithm override')
    if args.resume and (out/'summary.json').exists():
        previous=json.loads((out/'summary.json').read_text())
        if previous.get('status')=='COMPLETED':
            if json.loads((out/'protocol.json').read_text()) != info:
                raise ValueError('completed online run identity mismatch')
            return previous
    progress=Progress(out,info)
    if not args.resume and progress.state: raise ValueError('existing online checkpoint requires --resume')
    resume=progress.state
    tasks=copy.deepcopy(resume['tasks']) if resume else []
    completed=resume['completed'] if resume else 0
    elapsed=resume.get('elapsed_seconds',0.) if resume else 0.
    started=time.monotonic();last_save=started
    current_task=completed;within=0;task_rng=None
    if resume:
        if resume['mid']:
            restore_rng(resume['task_rng'])
        else:
            learner.model.load_state_dict(resume['model']);learner.optim.load_state_dict(resume['optimizer'])
            learner.buffer=resume['buffer'];hook.load_state_dict(resume['hook']);restore_rng(resume['rng'])
            for key,value in resume['learner_values'].items():setattr(learner,key,value)
    def save(mid=True):
        values={key:copy.deepcopy(getattr(learner,key)) for key in
                ('stream_idx','iter','results','results_forgetting','classes_seen_so_far') if hasattr(learner,key)}
        progress.save(dict(completed=current_task,mid=mid,within=within,task_rng=task_rng,
            model=learner.model.state_dict(),optimizer=learner.optim.state_dict(),buffer=learner.buffer,
            learner_values=values,hook=hook.state_dict(),rng=rng_state(),tasks=tasks,
            elapsed_seconds=elapsed+time.monotonic()-started))
        write_events(out,engine);io.write_json(out/'tasks.json',tasks)
    status=Status(out,f'v3 online {args.arm}')
    original=learner.optim.step
    learner.optim.step=lambda *a,**kw:hook.optimizer_step(original,*a,**kw)
    try:
        last_task=2 if args.preflight_only or args.fake_data else params.n_tasks
        for task in range(completed,last_task):
            current_task=task;within=0
            task_rng=rng_state()
            pending=resume if resume and resume['mid'] and task==completed else None
            if pending: task_rng=pending['task_rng'];within=pending['within']
            def prefix():
                if not pending:return
                learner.model.load_state_dict(pending['model']);learner.optim.load_state_dict(pending['optimizer'])
                learner.buffer=pending['buffer'];hook.load_state_dict(pending['hook'])
                for key,value in pending['learner_values'].items():setattr(learner,key,value)
                restore_rng(pending['rng'])
            def after_batch(count):
                nonlocal within,last_save
                within=count
                status.update(task=task+1,step=hook.iteration,phase='native online training')
                progress.boundary(save,point_seconds=max([e.get('seconds',0) for e in engine.events[-3:]],default=0))
                if time.monotonic()-last_save>=900:save();last_save=time.monotonic()
            stream=ResumeLoader(loaders[f'train{task}'],hook,skip=within,after_prefix=prefix,after_batch=after_batch)
            learner.train(stream,task_name=f'train{task}',task_id=task,dataloaders=loaders)
            learner.before_eval();accuracy,forgetting=learner.evaluate(loaders,task);learner.after_eval()
            losses=np.array(hook.task_losses.get(task+1,[]));accuracies=np.array(hook.task_accuracies.get(task+1,[]))
            half=np.flatnonzero(losses<=losses[0]*.5) if len(losses) else []
            row=dict(task=task+1,accuracy=float(accuracy)*100,forgetting=float(forgetting)*100,
                learning_accuracy=float(learner.results[-1][task])*100,
                anytime_accuracy_auc=float(np.trapezoid(accuracies)/max(1,len(accuracies)-1))*100,
                anytime_loss_auc=float(np.trapezoid(losses)/max(1,len(losses)-1)),
                iterations_to_half_loss=int(half[0])+1 if len(half) else None)
            tasks.append(row);current_task=task+1;within=0;save(mid=False);resume=None
        seconds=elapsed+time.monotonic()-started
        summary=dict(algorithm_version='v3',setting='online',arm=args.arm,seed=args.seed,status='COMPLETED',
            tasks=tasks,tasks_run=len(tasks),seconds=seconds,gpu_hours=seconds/3600,
            final_accuracy=tasks[-1]['accuracy'],mean_accuracy=float(np.mean([r['accuracy'] for r in tasks])),
            learning_accuracy=float(learner.learning_accuracy())*100,
            forgetting=tasks[-1]['forgetting'],anytime_accuracy_auc=float(np.mean([r['anytime_accuracy_auc'] for r in tasks])),
            intervention_seconds=sum(e.get('seconds',0) for e in engine.events))
        if args.preflight_only:info['preflight']=finalize_calibration(args,engine,hook,out)
        info.update(mu_rel=engine.mu_rel,K_base=engine.K_base)
        io.write_json(out/'protocol.json',info);io.write_json(out/'summary.json',summary)
        io.write_json(out/'tasks.json',tasks)
        io.write_csv(out/'continual_eval.csv', [dict(task=row['task'], accuracy=row['accuracy'],
            new_accuracy=row['learning_accuracy'],
            old_accuracy=float(np.nanmean(learner.results[i][:i]))*100 if i else None)
            for i,row in enumerate(tasks)])
        write_events(out,engine)
        return summary
    finally:
        learner.optim.step=original;status.close()
