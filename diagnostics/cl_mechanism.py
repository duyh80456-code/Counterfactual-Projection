"""H0 matched kernel comparisons and separately labelled native BN references."""
from dataclasses import replace
from copy import deepcopy
import hashlib
import math
import time
import torch
from torch.func import functional_call
from diagnostics.vna_align import _assert_bitwise
from methods.cl_replay import CorrectionConfig,ReplayCorrection
from methods.cl_state import State,isolated,vector,evaluate,predict
from methods.vna_v3 import training_products
from methods.vna_ggn import GGNProducts


def _fingerprint(tensors):
    result=hashlib.sha256()
    for name,value in tensors.items():
        result.update(name.encode())
        if torch.is_tensor(value):
            result.update(str((tuple(value.shape),value.dtype)).encode())
            result.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        else:result.update(repr(value).encode())
    return result.hexdigest()


def compare_local(model,optimizer,inputs,losses,sites,step_size,*,random_replicates=5,
                  config=CorrectionConfig(constrained=False),evaluation=None,update_norm=None,compute_seconds=None):
    if random_replicates<2:raise ValueError('at least two random draws are required; no automatic superiority claim')
    for name,value in (('update_norm',update_norm),('compute_seconds',compute_seconds)):
        if value is not None and (not math.isfinite(value) or value<=0):raise ValueError(f'{name} must be positive finite')
    rows=[];device=next(model.parameters()).device
    def synchronize():
        if device.type=='cuda':torch.cuda.synchronize(device)
    def local_accuracy():
        if evaluation is None:return None
        x,y,known=evaluation;z=predict(model,x);pred=z.argmax(1);old=y<known
        return dict(total=100*float((pred==y).float().mean()),
            old=100*float((pred[old]==y[old]).float().mean()) if bool(old.any()) else None,
            new=100*float((pred[~old]==y[~old]).float().mean()) if bool((~old).any()) else None)
    synchronize();shared_started=time.perf_counter();checkpoint=State(model,optimizer)
    def window_values():
        return dict(inputs=inputs,targets=losses.targets,replay=losses.replay,teacher=losses.teacher,
            kind=losses.kind,known=losses.known,reference=losses.reference,temperature=losses.temperature,reduction=losses.reduction)
    window=deepcopy(window_values())
    def verify_checkpoint():
        current=State(model,optimizer)
        for key in ('values','buffers','requires_grad','opt','rng','modes','gradients'):
            _assert_bitwise(getattr(checkpoint,key),getattr(current,key),'comparison.'+key)
        _assert_bitwise(window,window_values(),'comparison.window')
    with isolated(model,optimizer),training_products(model,[],{},inputs) as products:
        op=GGNProducts(products,losses.targets,losses.spec,losses.ctx)
        mu=ReplayCorrection(config)._operator(products,op).mu
    synchronize()
    checkpoint_values={**checkpoint.values,**{'buffer:'+n:b for n,b in checkpoint.buffers.items()},
        'modes':checkpoint.modes,'requires_grad':checkpoint.requires_grad}
    common=dict(checkpoint_sha256=_fingerprint(checkpoint_values),
        window_sha256=_fingerprint(window),
        shared_preparation_seconds=time.perf_counter()-shared_started,locked_mu=mu,
        requested_update_norm=update_norm,requested_compute_seconds=compute_seconds,
        cost_scope='seconds: correction core including its snapshots/diagnostics; seconds_total: full arm including accuracy and outer isolation',
        compute_budget_policy='shared wall limit checked at operation boundaries; overshoot recorded and censored',
        acceptance='componentwise_nonincrease' if config.constrained else 'total_nonincrease')
    arms=[(kernel,draws,config.constrained) for kernel,draws in (('tiny_V',1),('native',1),('random',random_replicates),('gn',1))]
    if not config.constrained:arms.extend((kernel,1,True) for kernel in ('tiny_V','native'))
    for kernel,draws,constrained in arms:
        for draw in range(draws):
            verify_checkpoint();synchronize();started=time.perf_counter()
            with isolated(model,optimizer):
                accuracy_before=local_accuracy()
                engine=ReplayCorrection(replace(config,kernel=kernel,constrained=constrained,mode='intervention'));engine.applications=draw
                event=engine.step(model,optimizer,inputs,losses,sites,step_size,
                    damping=mu,update_norm=update_norm,compute_seconds=compute_seconds)
                accuracy_after=local_accuracy()
            synchronize()
            completed=event['status'] in ('ACCEPTED','REJECTED')
            within_compute=completed and (compute_seconds is None or event['seconds']<=compute_seconds)
            group='constrained_kernel' if constrained else 'kernel'
            rows.append(dict(event,protocol=dict(common,acceptance='componentwise_nonincrease' if constrained else 'total_nonincrease'),random_draw=draw if kernel=='random' else None,
                comparison_group=group if kernel!='gn' else 'gn_candidate_control',
                checkpoint_verified=True,matched_update_budget=update_norm is not None,
                matched_compute_budget=compute_seconds is not None,completed_within_compute_budget=within_compute,
                actual_update_norms_matched=False,
                eligible_for_kernel_attribution=kernel!='gn' and update_norm is not None and within_compute,
                accuracy_before=accuracy_before,accuracy_after=accuracy_after,
                seconds_total=time.perf_counter()-started,metric_seed=config.seed,evidence_status='INCONCLUSIVE'))
    for buffers in ('standard_bn','frozen_buffers'):
        verify_checkpoint();synchronize();started=time.perf_counter()
        with isolated(model,optimizer),torch.enable_grad():
            before=evaluate(model,inputs,losses.components);weights=vector(model);initial=State(model,optimizer)
            accuracy_before=local_accuracy();synchronize();update_started=time.perf_counter()
            model.train();optimizer.zero_grad()
            z=model(inputs) if buffers=='standard_bn' else functional_call(model,{n:b.clone() for n,b in model.named_buffers()},(inputs,))
            if isinstance(z,dict):z=z['logits']
            losses.total(z).backward();optimizer.step();synchronize();update_seconds=time.perf_counter()-update_started
            after=evaluate(model,inputs,losses.components);delta=weights-vector(model);accuracy_after=local_accuracy()
            buffer_drift=float(sum((b.double()-initial.buffers[n].double()).square().sum() for n,b in model.named_buffers()).sqrt()) if any(True for _ in model.buffers()) else 0.
            initial.restore()
            with training_products(model,[],{},inputs) as products:
                op=GGNProducts(products,losses.targets,losses.spec,losses.ctx)
                first_order=float(op.gradient@delta);quadratic=first_order-float(delta@op.F(delta))/2
        synchronize();total=time.perf_counter()-started
        rows.append(dict(kernel='native_optimizer' if buffers=='standard_bn' else 'native_optimizer_frozen_buffers',
            protocol=dict(common,acceptance='none',cost_scope='seconds/seconds_total: full reference arm; update_seconds: native forward/backward/step'),
            comparison_group='native_optimizer_reference',
            bn_behavior=buffers,bn_buffer_drift=buffer_drift,checkpoint_verified=True,
            eligible_for_kernel_attribution=False,matched_update_budget=False,matched_compute_budget=False,
            before=before,after=after,seconds=total,seconds_total=total,update_seconds=update_seconds,
            step_norm=float(delta.norm()),first_order_predicted_gain=first_order,quadratic_predicted_gain=quadratic,
            accuracy_before=accuracy_before,accuracy_after=accuracy_after,evidence_status='INCONCLUSIVE',
            comparison='unfiltered native reference; not a matched kernel/correction arm'))
    verify_checkpoint()
    return rows


if __name__=='__main__':
    import argparse
    from pathlib import Path
    from experiments.io_utils import atomic_json_save
    from methods.cl_replay import ReplayLoss
    from methods.vna_sites import HookSite
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint',help='trusted local torch payload: model, optimizer, inputs, loss_args, site_specs')
    source.add_argument('--fake-data',action='store_true')
    p.add_argument('--out',required=True);p.add_argument('--update-norm',type=float,default=.01)
    p.add_argument('--compute-seconds',type=float);p.add_argument('--random-replicates',type=int,default=5)
    args=p.parse_args();torch.set_num_threads(1)
    if args.fake_data:
        torch.manual_seed(21)
        model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4),torch.nn.ReLU(),torch.nn.Linear(4,3)).double().train()
        optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.0002)
        payload=dict(model=model,optimizer=optimizer,inputs=torch.randn(6,3,dtype=torch.double),
            loss_args=dict(targets=torch.tensor([0,1,2,0,1,2]),replay=torch.tensor([True,False,False,True,False,False])),
            site_specs=[dict(first='0',second='3',kind='linear',between=('1','2'))])
    else:payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model,optimizer=payload['model'],payload['optimizer'];losses=ReplayLoss(**payload['loss_args'])
    sites=[HookSite(model,**spec) for spec in payload['site_specs']]
    rows=compare_local(model,optimizer,payload['inputs'],losses,sites,optimizer.param_groups[0]['lr'],
        random_replicates=args.random_replicates,update_norm=args.update_norm,compute_seconds=args.compute_seconds)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    atomic_json_save(dict(branch='H0',synthetic_smoke=args.fake_data,rows=rows,evidence_status='INCONCLUSIVE'),target)
