"""Real Phase0 only: paired checkpoint experiments; never launches later phases."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import torch
from torch import nn
from experiments.io_utils import atomic_torch_save,atomic_json_save,write_csv
from ett.core.units import discover
from ett.core.ett import ETT,measured_loss
from ett.baselines import cbp


def network(cfg):
    layers=[];sizes=[cfg['inputs'],*cfg['hidden'],cfg['outputs']]
    for i,(a,b) in enumerate(zip(sizes,sizes[1:])):
        layer=nn.Linear(a,b);nn.init.kaiming_uniform_(layer.weight,nonlinearity='linear' if i==len(sizes)-2 else 'relu');nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i<len(sizes)-2:layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def sites_for(model):
    sites,skipped=discover(model)
    for site in sites:site.init_bound=math.sqrt(6/site.first.in_features)
    return sites


def train(model,opt,batches,loss_fn):
    started=time.perf_counter()
    for x,y in batches:
        opt.zero_grad(set_to_none=True);loss=loss_fn(model(x),y)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite Phase0 training loss')
        loss.backward();opt.step()
    return time.perf_counter()-started


def evaluate(model,batches,loss_fn):
    return sum(measured_loss(model,b,loss_fn) for b in batches)/len(batches)


def sign_test(values):
    pos=sum(v>0 for v in values);n=sum(v!=0 for v in values)
    return sum(math.comb(n,j) for j in range(pos,n+1))/2**n if n else 1.


def summarize(rows):
    tables=[]
    for setting in sorted({r['setting'] for r in rows}):
        for h in (1,10,100):
            chosen=[r for r in rows if r['setting']==setting and r['h']==h]
            if not chosen:continue
            row=dict(setting=setting,h=h,checkpoints=len(chosen))
            for metric in ('GE','delta','net','cbp_minus_compact','gm_minus_compact','virtual_kept_fraction','g_score','g_back','dead_fraction'):
                values=torch.tensor([r[metric] for r in chosen],dtype=torch.double)
                q=torch.quantile(values,torch.tensor([.25,.5,.75],dtype=torch.double)).tolist()
                row[metric+'_median']=q[1];row[metric+'_iqr']=q[2]-q[0]
            net=[r['net'] for r in chosen];row.update(positive_fraction=sum(v>0 for v in net)/len(net),sign_p=sign_test(net))
            if h==1:
                values=[r['first_order_ratio'] for r in chosen if r['first_order_ratio'] is not None]
                row['first_order_ratio_median']=float(torch.tensor(values).median()) if values else None
            row['gate']=len(chosen)>=20 and h in (10,100) and row['net_median']>0 and row['sign_p']<.05 and row['cbp_minus_compact_median']>0
            tables.append(row)
    return tables


def paired(model,opt,gnt,cfg,batches,eval_batches,step,h,loss_fn,cycle):
    anchor_rng=torch.get_rng_state();start=time.perf_counter()
    native,nopt=deepcopy((model,opt));torch.set_rng_state(anchor_rng);native_seconds=train(native,nopt,batches,loss_fn)
    LN=evaluate(native,eval_batches,loss_fn)
    expanded,eopt=deepcopy((model,opt));torch.set_rng_state(anchor_rng)
    engine=ETT(expanded,eopt,sites_for(expanded),rho=cfg['rho'],h=h,T=cfg['T'],maturity=cfg['maturity'],
        seed=cfg['ett_seed']+step,window=cfg['window'])
    for age in engine.ages.values():age.fill_(step)
    engine.cycle=cycle
    engine.begin(batches[0],loss_fn);energy=engine.active['gradient_energy'];training=0.
    for batch in batches:
        seconds=train(expanded,eopt,[batch],loss_fn);training+=seconds;engine.after_step(batch,seconds)
    LE=evaluate(expanded,eval_batches,loss_fn)
    records=engine.contract(eval_batches[0],loss_fn,acceptance=False)
    LC=evaluate(expanded,eval_batches,loss_fn)
    reset_losses={};reset_counts={}
    for name,gradmax in (('cbp',False),('gm',True)):
        reset,ropt=deepcopy((model,opt));torch.set_rng_state(anchor_rng)
        rgnt=cbp.make(reset,ropt,cfg);cbp.restore(rgnt,cbp.history(gnt))
        reset_counts[name]=cbp.single_event(rgnt,sites_for(reset),batches[0],loss_fn,torch.Generator().manual_seed(cfg['ett_seed']+step),gradmax)
        train(reset,ropt,batches,loss_fn);reset_losses[name]=evaluate(reset,eval_batches,loss_fn)
    torch.set_rng_state(anchor_rng)
    k=sum(r['k'] for r in records);total_units=sum(s.width for s in engine.sites)+k
    return dict(setting=cfg['setting'],step=step,h=h,LN=LN,LE=LE,LC=LC,LCBP=reset_losses['cbp'],LGM=reset_losses['gm'],
        GE=LN-LE,delta=LC-LE,net=LN-LC,cbp_minus_compact=reset_losses['cbp']-LC,gm_minus_compact=reset_losses['gm']-LC,
        first_order_prediction=cfg['lr']*energy,first_order_ratio=(LN-LE)/(cfg['lr']*energy) if h==1 and energy>0 else None,
        virtual_kept_fraction=sum(r['virtual_kept'] for r in records)/max(k,1),
        dead_fraction=sum(r['dead_units'] for r in records)/max(total_units,1),
        g_score=sum(r['g_score'] for r in records),g_back=sum(r['g_back'] for r in records),
        reset_counts=reset_counts,cycle_logs=records,native_train_seconds=native_seconds,expanded_train_seconds=training,
        seconds=time.perf_counter()-start)


def run(args):
    cfg=json.loads(Path(args.config).read_text());out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    protocol=dict(commit=commit,config=cfg,phase=0,device='cpu',dtype='float32',batch_size=1,
                  config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest())
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Phase0 resume provenance mismatch')
    atomic_json_save(protocol,out/'protocol.json');torch.set_num_threads(1)
    if cfg['setting']=='scr':
        from ett.envs.scr import data
        X,Y=data(cfg,args.data_dir);task_steps=cfg['flip_after']
        batch=lambda index:(X[index:index+1],Y[index:index+1]);loss_fn=nn.functional.mse_loss
        data_fingerprint=hashlib.sha256(X.numpy().tobytes()+Y.numpy().tobytes()).hexdigest()
    else:
        from ett.envs.pmnist import PMNIST
        stream=PMNIST(cfg,args.data_dir,args.download);task_steps=stream.task_steps;batch=stream.batch
        data_fingerprint=stream.fingerprint;loss_fn=nn.functional.cross_entropy
        if cfg['steps']!=cfg['tasks']*task_steps:raise ValueError('PM run must include the full registered task sequence')
    torch.manual_seed(cfg['init_seed']);model=network(cfg);opt=torch.optim.SGD(model.parameters(),lr=cfg['lr'])
    gnt=cbp.make(model,opt,cfg);step=0;rows=[];elapsed=0.;native_seconds=0.;last_checkpoint=0
    reserve=max(cfg['horizons'])+cfg['evaluation_batches']
    anchors=sorted(set(min(int(t), (int(t)//task_steps+1)*task_steps-reserve-1)
        for t in torch.linspace(0,cfg['steps']-reserve-1,cfg['checkpoints'],dtype=torch.double).tolist()))
    if len(anchors)<20:raise ValueError('Phase0 needs at least twenty distinct anchors')
    if (out/'latest.pt').exists():
        state=torch.load(out/'latest.pt',weights_only=False)
        if state['protocol']!=protocol or state['data_fingerprint']!=data_fingerprint:raise ValueError('Phase0 checkpoint identity mismatch')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);cbp.restore(gnt,state['gnt'])
        step=state['step'];rows=state['rows'];elapsed=state['elapsed'];native_seconds=state['native_seconds'];torch.set_rng_state(state['rng']);last_checkpoint=step
    start=time.perf_counter();last_heartbeat=start
    def save(status):
        seconds=elapsed+time.perf_counter()-start
        atomic_torch_save(dict(protocol=protocol,data_fingerprint=data_fingerprint,step=step,model=model.state_dict(),optimizer=opt.state_dict(),gnt=cbp.history(gnt),rows=rows,
            elapsed=seconds,native_seconds=native_seconds,rng=torch.get_rng_state()),out/'latest.pt')
        atomic_json_save(dict(status=status,step=step,total_steps=cfg['steps'],commit=commit,wall_seconds=seconds),out/'status.json')
        atomic_json_save(summarize(rows),out/'summary.json')
        (out/'measurements.jsonl').write_text(''.join(json.dumps(dict(r,commit=commit,config=cfg))+'\n' for r in rows))
        flat=[{k:v for k,v in r.items() if k not in ('cycle_logs','reset_counts')} for r in rows]
        write_csv(out/'measurements.csv',flat)
    try:
        while step<cfg['steps']:
            if args.deadline_seconds and time.perf_counter()-start>=args.deadline_seconds:
                save('PAUSED');return 75
            if step in anchors:
                for h in cfg['horizons']:
                    if any(r['step']==step and r['h']==h for r in rows):continue
                    sequence=[batch(step+j) for j in range(h)]
                    evaluation=[batch(step+max(cfg['horizons'])+j) for j in range(cfg['evaluation_batches'])]
                    atomic_torch_save(dict(protocol=protocol,step=step,model=model.state_dict(),optimizer=opt.state_dict(),gnt=cbp.history(gnt),rng=torch.get_rng_state()),out/'checkpoints'/f'{step}.pt')
                    row=paired(model,opt,gnt,cfg,sequence,evaluation,step,h,loss_fn,anchors.index(step));rows.append(row);save('RUNNING')
                    print(f"{cfg['setting']} anchor={step} h={h} GE={row['GE']:.6g} delta={row['delta']:.6g} net={row['net']:.6g}",flush=True)
            b=batch(step);started=time.perf_counter();opt.zero_grad(set_to_none=True)
            prediction,features=cbp.capture_features(model,b[0]);loss=loss_fn(prediction,b[1])
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite native BP loss')
            loss.backward();opt.step();cbp.observe(gnt,features);native_seconds+=time.perf_counter()-started;step+=1
            if step-last_checkpoint>=cfg['checkpoint_every']:
                save('RUNNING');last_checkpoint=step
            if time.perf_counter()-last_heartbeat>=30:
                print(f"{cfg['setting']} progress={step}/{cfg['steps']} elapsed={elapsed+time.perf_counter()-start:.1f}s",flush=True);last_heartbeat=time.perf_counter()
        save('COMPLETED')
        report=dict(protocol=protocol,tables=summarize(rows),gate_pass=any(t['gate'] for t in summarize(rows)),native_seconds=native_seconds,
                    note='Phase0 diagnostic branches, not a full-run overhead estimate. Stop and await approval before Phase1.')
        atomic_json_save(report,out/'phase0_report.json');return 0
    except BaseException:
        save('FAILED');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--data-dir',default='/tmp/ett_data');parser.add_argument('--download',action='store_true');parser.add_argument('--deadline-seconds',type=float,default=None)
    raise SystemExit(run(parser.parse_args()))
