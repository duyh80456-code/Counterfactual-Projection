"""Stage 2a: final-model CIFAR-100 SGD and solver-agnostic VNA arms."""
from __future__ import annotations
import argparse
import copy
from itertools import islice
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from experiments.shared_protocol import (atomic_torch_save, atomic_json_save, make_train_loader,
    make_eval_loader, evaluate, seed_everything, rng_state, restore_rng, validate_resume_protocol)
from experiments.vna_support import setup_data, build_model, DeviceCIFAR, DeviceTrainLoader, augmented_batch, isolated_rng
from experiments.vna_training_support import TrainingVNA, sync
from methods.vna import add_vna_grad_


def official_test(args, evaluation):
    if args.fake_data:
        from torch.utils.data import TensorDataset
        g=torch.Generator().manual_seed(987654)
        return TensorDataset(torch.randn(128,3,8,8,generator=g),torch.randint(3,(128,),generator=g))
    from torchvision.datasets import CIFAR100
    return CIFAR100(args.data_root,train=False,transform=evaluation.transform,download=False)


def write_jsonl(path, rows):
    temporary=path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    temporary.replace(path)


def clean_json(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:clean_json(v) for k,v in value.items()}
    if isinstance(value,list): return [clean_json(v) for v in value]
    return value


def training_protocol(args):
    scale=args.epochs_scale if args.arm=='sgd_cm' else 1.
    protocol={k:v for k,v in vars(args).items() if k not in ('output','resume','max_epochs','device','data_root','budget_root','disk_budget')}
    protocol.update(total_epochs=round(args.epochs*scale),milestones=[round(100*scale),round(150*scale)],
                    architecture='tiny' if args.fake_data else 'resnet18',format_version=2)
    return protocol


def run(args):
    device=torch.device(args.device)
    if args.fake_data: torch.set_num_threads(1)
    if args.amp and device.type!='cuda': raise ValueError('--amp requires CUDA')
    for name in ('batch_size','epochs','vna_every','refresh_epochs','vna_batch','stat_batch','stat_batches','rank','max_sites','gate_boot','factor_refresh_epochs','vna_start_epoch'):
        if getattr(args,name)<1: raise ValueError(name+' must be positive')
    if args.stat_batch < args.gate_boot or not 0<args.dorfler_theta<=1 or args.epochs_scale<=0:
        raise ValueError('invalid refresh/scale configuration')
    torch.backends.cudnn.benchmark=args.cudnn_benchmark
    seed_everything(args.seed)
    np.random.seed(args.seed)
    train,evaluation,indices,trigger,eval_indices,tuning=setup_data(args.data_root,args.fake_data,args.seed)
    model=build_model(device,args.fake_data)
    if args.channels_last: model.to(memory_format=torch.channels_last)
    params=sum(p.numel() for p in model.parameters()); initial_params=params
    scale=args.epochs_scale if args.arm=='sgd_cm' else 1.
    epochs=round(args.epochs*scale); milestones=[round(100*scale),round(150*scale)]
    optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=5e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones,gamma=.1)
    scaler=torch.cuda.amp.GradScaler(enabled=args.amp)
    engine=TrainingVNA(args,device)
    protocol=training_protocol(args)
    epoch=step=correction_count=0; history=[]; corrections=[]; loader_state=None; final=None
    if args.resume:
        cp=torch.load(args.resume,map_location='cpu',weights_only=False)
        if cp.get('kind')!='vna_training': raise ValueError('not a Stage 2a training checkpoint')
        validate_resume_protocol(cp['protocol'],protocol)
        if cp['splits']!=[indices,trigger,eval_indices,tuning]: raise ValueError('resume split mismatch')
        model.load_state_dict(cp['model']); optimizer.load_state_dict(cp['optimizer']); scheduler.load_state_dict(cp['scheduler'])
        scaler.load_state_dict(cp['scaler']); engine.load_state_dict(cp['vna'])
        restore_rng(cp['rng']); np.random.set_state(cp['numpy_rng'])
        epoch,step,history=cp['epoch'],cp['step'],cp['history']
        correction_count=cp['correction_count']
        final=cp['final']; loader_state=cp['train_loader_generator_state']
    loader=make_train_loader(train,indices,args.batch_size,args.workers,loader_state,args.seed+1)
    statistics_data=train
    if args.gpu_data:
        if args.fake_data: raise ValueError('--gpu-data requires CIFAR')
        statistics_data=DeviceCIFAR(train,device)
        loader=DeviceTrainLoader(statistics_data,indices,args.batch_size,args.seed+1,loader_state)
    eval_loader=make_eval_loader(evaluation,eval_indices,args.batch_size,args.workers)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    active=args.arm in ('vna','vna_random','ngd')
    def save():
        payload={'kind':'vna_training','epoch':epoch,'step':step,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'rng':rng_state(),'numpy_rng':np.random.get_state(),'train_loader_generator_state':loader.generator.get_state(),
            'splits':[indices,trigger,eval_indices,tuning],'protocol':protocol,'history':history,
            'correction_count':correction_count,'vna':engine.state_dict(),'final':final}
        # Bound checkpoint replacement and raw + old + new archives before writing.
        from experiments.vna_training_kaggle import check_training_budget
        check_training_budget(Path(args.budget_root) if args.budget_root else output,
                              output/'latest.pt',payload,args.disk_budget)
        atomic_torch_save(payload,output/'latest.pt')
        write_jsonl(output/'epochs.jsonl',history)
        with (output/'corrections.jsonl').open('a') as log:
            for row in corrections: log.write(json.dumps(row,allow_nan=False)+'\n')
        corrections.clear()
        atomic_json_save(history,output/'history.json')
    for current in range(epoch+1,min(epochs,args.max_epochs)+1):
        sync(device); start=time.perf_counter(); components={}; applied=skipped=0; diagnostics=[]
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        vna_active=active and current>=args.vna_start_epoch
        if vna_active and (not engine.cache or (current-args.vna_start_epoch)%args.refresh_epochs==0):
            components=engine.refresh(model,statistics_data,indices,current)
        elif vna_active and (current-args.vna_start_epoch)%args.factor_refresh_epochs==0:
            components=engine.refresh_factors(model,statistics_data,indices,current)
        model.train(); loss_sum=torch.zeros((),device=device,dtype=torch.float64); correct=torch.zeros((),device=device,dtype=torch.long); count=0
        batches=islice(loader,args.limit_batches) if args.limit_batches else loader
        for inputs,targets in batches:
            inputs=inputs.to(device,non_blocking=True); targets=targets.to(device,non_blocking=True)
            if args.channels_last: inputs=inputs.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=args.amp):
                logits=model(inputs); loss=torch.nn.functional.cross_entropy(logits.float(),targets)
            if args.amp:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            else: loss.backward()
            step+=1
            if vna_active and step%args.vna_every==0:
                sync(device); t=time.perf_counter()
                batch=augmented_batch(statistics_data,indices,args.vna_batch,engine.generators['correction'],device)
                with isolated_rng(engine.generators['construction']):
                    for site in engine.cache:
                        delta,diag=engine.correction(model,site,batch)
                        if diag['skipped']: skipped+=1
                        else:
                            # Zero coefficient deliberately performs diagnostics but no gradient write.
                            coefficient=engine.coefficient
                            refs=dict(model.named_parameters())
                            grad_norm=sum(float(refs[n].grad.detach().double().square().sum()) for n in delta if refs[n].grad is not None)**.5
                            diag['applied_delta_ratio']=abs(coefficient)*diag['correction_norm']/max(grad_norm,1e-30)
                            if coefficient!=0: add_vna_grad_(model,delta,coefficient)
                            applied+=1
                        correction_count+=1
                        diag.setdefault('applied_delta_ratio',0.)
                        diagnostics.append(diag); corrections.append(clean_json(dict(diag,epoch=current,step=step,arm=args.arm)))
                sync(device); components['correction']=components.get('correction',0.)+time.perf_counter()-t
            if args.amp: scaler.step(optimizer); scaler.update()
            else: optimizer.step()
            loss_sum+=loss.detach().double()*targets.numel(); correct+=(logits.argmax(1)==targets).sum(); count+=targets.numel()
        if count==0: raise ValueError('empty training epoch')
        loss_sum,correct=torch.stack((loss_sum,correct.double())).tolist()
        sync(device); train_seconds=time.perf_counter()-start
        eval_metrics=evaluate(model,eval_loader,device); sync(device)
        lr=optimizer.param_groups[0]['lr']; scheduler.step(); epoch=current
        # G3 uses measured wall time, so it is log-only, outside numerical state.
        rate=(history[-1]['train']['loss']-loss_sum/count)/max(train_seconds,1e-12) if history else None
        past_rates=[r.get('loss_decrease_per_second') for r in history
                    if r['epoch']>=engine.controller['refresh_epoch']]
        past_rates=[r for r in past_rates if r is not None]
        reference_rate=past_rates[0] if past_rates else rate
        g3=rate<.5*reference_rate if rate is not None and reference_rate is not None else None
        def mean(key):
            values=[d[key] for d in diagnostics if math.isfinite(d[key])]
            return sum(values)/len(values) if values else None
        history.append(clean_json({'arm':args.arm,'solver':args.solver,'selection':args.selection,'epoch':epoch,
            'train':{'loss':loss_sum/count,'accuracy':correct/count},'eval':eval_metrics,'lr':lr,
            'epoch_seconds':time.perf_counter()-start,'train_seconds':train_seconds,'vna_seconds':components,
            'corrections_applied':applied,'corrections_skipped':skipped,'mean_rho':mean('rho'),'mean_V':mean('V'),
            'mean_applied_delta_ratio':mean('applied_delta_ratio'),'max_applied_delta_ratio':max((d['applied_delta_ratio'] for d in diagnostics),default=None),
            'mean_raw_gain':mean('raw_gain'),'mean_realized_gain':mean('realized_gain'),'loss_decrease_per_second':rate,
            'gate_statistics':{'G1':any(e['G1'] for e in engine.cache.values()),'G2':(mean('rho') or 0.)>.3,'G3':g3},'selected_sites':{s:e['rank'] for s,e in engine.cache.items()},
            'controller':copy.deepcopy(engine.controller),'weight_norms':{n:float(p.detach().double().norm()) for n,p in model.named_parameters() if n.endswith('weight')},
            'peak_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}))
        if sum(p.numel() for p in model.parameters())!=initial_params: raise RuntimeError('parameter count changed')
        save(); print(json.dumps({'epoch':epoch,'arm':args.arm,'train':history[-1]['train'],'eval':eval_metrics}),flush=True)
    if epoch==epochs and final is None:
        # The final metric is saved in latest; completed resume never evaluates again.
        sync(device); test_start=time.perf_counter()
        test=official_test(args,evaluation)
        test_metrics=evaluate(model,make_eval_loader(test,list(range(len(test))),args.batch_size,args.workers),device)
        sync(device)
        test_seconds=time.perf_counter()-test_start
        seconds=sum(row['epoch_seconds'] for row in history)+test_seconds
        final={'arm':args.arm,'solver':args.solver,'selection':args.selection,'seed':args.seed,
            'test':test_metrics,'params':params,'sgd_params':initial_params,'gpu_hours':seconds/3600 if device.type=='cuda' else 0.,
            'wall_hours':seconds/3600,'peak_memory_bytes':max(row['peak_memory_bytes'] for row in history),
            'vna_seconds':sum(sum(v for k,v in row['vna_seconds'].items() if k in ('refresh','factor_refresh','correction')) for row in history),
            'test_evaluations':1,'test_seconds':test_seconds,'epoch':epoch}
        save()
    if final is not None: atomic_json_save(final,output/'final.json')
    return output/'latest.pt'


def parser():
    from experiments.train_recipe_checkpoints import parser as recipe_parser
    p=recipe_parser(); p.description=__doc__; p.set_defaults(output='outputs/vna_training',max_epochs=10**9)
    p.add_argument('--budget-root',help='shared notebook output root for disk accounting')
    p.add_argument('--disk-budget',type=int,default=5_000_000_000)
    p.add_argument('--arm',choices=['sgd','sgd_cm','vna','vna_random','ngd'],default='sgd')
    p.add_argument('--solver',choices=['exact','kfac','lite'],default='exact')
    p.add_argument('--epochs',type=int,default=200,help='recipe length; shorter runs for CPU tests')
    p.add_argument('--epochs-scale',type=float,default=1.3)
    from experiments.vna_training_config import add_vna_arguments
    add_vna_arguments(p)
    return p


if __name__=='__main__': run(parser().parse_args())
