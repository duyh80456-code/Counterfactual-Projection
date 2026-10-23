"""No-op-preserving clipping of the vector consumed by the training optimizer."""
from contextlib import contextmanager
import inspect
import math
import copy
import numpy as np
import torch
from experiments.io_utils import atomic_json_save


class TrainingDiverged(FloatingPointError):
    def __init__(self,detail):
        self.detail=detail
        super().__init__('FAILED_DIVERGED '+str(detail))


def positive_tau(value):
    value=float(value)
    if not math.isfinite(value) or value<=0:raise ValueError('grad-clip must be finite and positive')
    return value


class GradClip:
    def __init__(self,tau=None):
        self.tau=None if tau is None else positive_tau(tau)
        self.tasks={};self.steps=[]

    def state_dict(self):return copy.deepcopy(dict(tau=self.tau,tasks=self.tasks,steps=self.steps))

    def load_state_dict(self,state):
        if state['tau']!=self.tau:raise ValueError('gradient clipping resume threshold mismatch')
        self.tasks=copy.deepcopy(state['tasks']);self.steps=copy.deepcopy(state.get('steps',[]))

    @torch.no_grad()
    def apply(self,gradients,*,task,step,epoch=None):
        gradients=[g for g in gradients if g is not None]
        norm=torch.linalg.vector_norm(torch.stack([g.detach().norm(2) for g in gradients]),2) if gradients else torch.tensor(0.)
        value=float(norm);clipped=False
        if self.tau is not None and norm>self.tau:
            scale=self.tau/(norm+1e-6)
            for gradient in gradients:gradient.mul_(scale)
            clipped=True
        row=dict(task=int(task),step=int(step),task_step=len(self.tasks.get(str(task),[]))+1,epoch=epoch,preclip_norm=value if math.isfinite(value) else None,
                 nonfinite_norm=not math.isfinite(value),clipped=clipped)
        self.tasks.setdefault(str(task),[]).append(row)
        return row

    def report(self):
        def summary(rows):
            norms=[r['preclip_norm'] for r in rows if r['preclip_norm'] is not None]
            nonfinite=sum(r['nonfinite_norm'] for r in rows)
            return dict(n_steps=len(rows),n_clipped=sum(r['clipped'] for r in rows),
                max_preclip_norm=None if nonfinite else max(norms,default=None),max_finite_preclip_norm=max(norms,default=None),
                preclip_norm=dict(p50=float(np.quantile(norms,.5)) if norms else None,
                    p99=float(np.quantile(norms,.99)) if norms else None,max=None if nonfinite else max(norms,default=None),finite_quantiles_only=bool(nonfinite)),
                nonfinite_steps=nonfinite,
                clipped_steps=[dict(step=r['step'],norm=r['preclip_norm']) for r in rows if r['clipped']])
        rows=[r for values in self.tasks.values() for r in values]
        return dict(tau=self.tau,run=summary(rows),tasks={k:summary(v) for k,v in self.tasks.items()},
                    scope='training optimizer vector only; excludes VNA/GIR and diagnostic optimizers')

    def save(self,path):atomic_json_save(self.report(),path)


def training_frame():
    frame=inspect.currentframe().f_back
    while frame:
        if frame.f_code.co_name in ('_init_train','_update_representation','_run') and 'epoch' in frame.f_locals:return frame
        frame=frame.f_back
    return None


def scalar(value):
    if isinstance(value,torch.Tensor):value=float(value.detach())
    if isinstance(value,(int,float)):return float(value) if math.isfinite(value) else None
    return None


def step_details(learner,optimizer,frame):
    values=frame.f_locals if frame else {}
    logits=values.get('logits');loss=values.get('loss')
    losses=values.get('loss_list')
    if losses is not None:loss=sum(losses)
    known=learner._known_classes
    row=dict(loss=scalar(loss),ce=None,kd=None,lr=optimizer.param_groups[0]['lr'],
             old_max_logit=None,new_max_logit=None,parameter_norm=None,resets=0)
    with torch.no_grad():
        if isinstance(logits,torch.Tensor):
            row['old_max_logit']=scalar(logits[:,:known].abs().max()) if known else None
            row['new_max_logit']=scalar(logits[:,known:].abs().max()) if logits.shape[1]>known else None
            targets=values.get('targets')
            if targets is not None:
                row['ce']=scalar(torch.nn.functional.cross_entropy(logits,targets))
        if 'loss_clf' in values:
            row['ce']=scalar(values['loss_clf']);row['kd']=scalar(values.get('loss_kd',0.))
        elif losses is not None:
            row['ce']=scalar(losses[0]);row['kd']=scalar(sum(losses[1:]))
        elif row['loss'] is not None and row['ce'] is not None:row['kd']=row['loss']-row['ce'] if known else 0.
        row['parameter_norm']=scalar(torch.linalg.vector_norm(torch.stack([p.detach().norm(2) for p in learner._network.parameters()]),2))
    return row


def check_parameters(learner,*,task,epoch,step):
    for name,p in learner._network.named_parameters():
        if not bool(torch.isfinite(p).all()):raise TrainingDiverged(dict(task=task,epoch=epoch,step=step,tensor=name))


@contextmanager
def optimizer_clipping(module,callback):
    """Instrument SGD instances, including GEM's projected and C-Flat's final base step."""
    from types import SimpleNamespace
    old=module.optim
    def sgd(*args,**kwargs):
        optimizer=old.SGD(*args,**kwargs)
        optimizer.register_step_pre_hook(lambda opt,a,kw:callback(opt,training_frame()))
        return optimizer
    module.optim=SimpleNamespace(**{**vars(old),'SGD':sgd})
    try:yield
    finally:module.optim=old


@torch.no_grad()
def upgd_step(optimizer,callback):
    """Pinned FirstOrderGlobalUPGD arithmetic; stage effective gradients before its adds."""
    maximum=torch.tensor(-torch.inf)
    for group in optimizer.param_groups:
        for name,p in zip(group['names'],group['params']):
            if 'gate' in name:continue
            state=optimizer.state[p]
            if not state:state.update(step=0,avg_utility=torch.zeros_like(p.data))
            state['step']+=1
            utility=state['avg_utility']
            utility.mul_(group['beta_utility']).add_(-p.grad.data*p.data,alpha=1-group['beta_utility'])
            current=utility.max()
            if current>maximum:maximum=current
    updates=[]
    for group in optimizer.param_groups:
        for name,p in zip(group['names'],group['params']):
            if 'gate' in name:continue
            state=optimizer.state[p]
            correction=1-group['beta_utility']**state['step']
            noise=torch.randn_like(p.grad)*group['sigma']
            utility=torch.sigmoid_((state['avg_utility']/correction)/maximum)
            effective=(p.grad.data+noise)*(1-utility)
            updates.append((p,group,effective))
    callback(optimizer,training_frame(),[g for p,group,g in updates])
    for p,group,gradient in updates:
        p.data.mul_(1-group['lr']*group['weight_decay']).add_(gradient,alpha=-group['lr'])
