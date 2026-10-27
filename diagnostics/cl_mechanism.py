"""H0 local fixed-state comparator, without training or changing the input state."""
from dataclasses import replace
import time
import torch
from torch.func import functional_call
from methods.cl_replay import CorrectionConfig,ReplayCorrection
from methods.cl_state import State,isolated,vector,evaluate,predict
from methods.vna_v3 import training_products
from methods.vna_ggn import GGNProducts


def compare_local(model,optimizer,inputs,losses,sites,step_size,*,random_replicates=5,config=CorrectionConfig(),evaluation=None):
    if random_replicates<2:raise ValueError('at least two random draws are needed; no statistical-superiority claim is automatic')
    rows=[]
    def local_accuracy():
        if evaluation is None:return None
        x,y,known=evaluation;z=predict(model,x);pred=z.argmax(1);old=y<known
        return dict(total=100*float((pred==y).float().mean()),
            old=100*float((pred[old]==y[old]).float().mean()) if bool(old.any()) else None,
            new=100*float((pred[~old]==y[~old]).float().mean()) if bool((~old).any()) else None)
    for kernel,draws in (('tiny_V',1),('native',1),('random',random_replicates),('gn',1)):
        for draw in range(draws):
            with isolated(model,optimizer):
                accuracy_before=local_accuracy()
                engine=ReplayCorrection(replace(config,kernel=kernel,constrained=False,mode='intervention'))
                engine.applications=draw
                event=engine.step(model,optimizer,inputs,losses,sites,step_size)
                rows.append(dict(event,random_draw=draw if kernel=='random' else None,
                    accuracy_before=accuracy_before,accuracy_after=local_accuracy(),
                    metric_seed=config.seed,evidence_status='INCONCLUSIVE',comparison='local fixed checkpoint only'))
    with isolated(model,optimizer),torch.enable_grad():
        started=time.perf_counter();before=evaluate(model,inputs,losses.components);weights=vector(model);accuracy_before=local_accuracy()
        model.train();optimizer.zero_grad()
        z=functional_call(model,{n:b.clone() for n,b in model.named_buffers()},(inputs,))
        if isinstance(z,dict):z=z['logits']
        losses.total(z).backward();initial=State(model,optimizer);optimizer.step()
        after=evaluate(model,inputs,losses.components)
        delta=weights-vector(model);accuracy_after=local_accuracy();initial.restore()
        with training_products(model,[],{},inputs) as products:
            op=GGNProducts(products,losses.targets,losses.spec,losses.ctx)
            first_order=float(op.gradient@delta);quadratic=first_order-float(delta@op.F(delta))/2
        rows.append(dict(kernel='native_optimizer',before=before,after=after,seconds=time.perf_counter()-started,
            step_norm=float(delta.norm()),first_order_predicted_gain=first_order,quadratic_predicted_gain=quadratic,
            accuracy_before=accuracy_before,accuracy_after=accuracy_after,evidence_status='INCONCLUSIVE',
            comparison='actual optimizer parameter step; no correction acceptance filter'))
    return rows
