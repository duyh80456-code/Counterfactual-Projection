"""Real Phase0 only: paired checkpoint experiments; never launches later phases."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import time
import torch
from torch import nn
from experiments.io_utils import atomic_torch_save,atomic_json_save,write_csv
from ett.core.units import discover
from ett.core.ett import ETT,measured_loss
from ett.core.stats import measurement
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


def combine(batches):
    return torch.cat([b[0] for b in batches]),torch.cat([b[1] for b in batches])


def sign_test(values):
    pos=sum(v>0 for v in values);n=sum(v!=0 for v in values)
    return sum(math.comb(n,j) for j in range(pos,n+1))/2**n if n else 1.


def summarize(rows):
    tables=[]
    for setting in sorted({r['setting'] for r in rows}):
        for h in (1,10,100):
            chosen=[r for r in rows if r['setting']==setting and r['h']==h]
            if any(r.get('window_protocol')!='trained_1000_80_20_v1' for r in chosen):raise ValueError('invalid window measurements cannot enter the gate')
            if not chosen:continue
            if len({r['step'] for r in chosen})!=len(chosen):raise ValueError('duplicate anchor in Phase0 gate')
            row=dict(setting=setting,h=h,checkpoints=len(chosen))
            for metric in ('GE','delta','net','cbp_minus_compact','gm_minus_compact','virtual_kept_fraction','g_score','g_back','dead_fraction','net_post','g_actual_data','buffer_check_delta'):
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


def paired(model,opt,gnt,cfg,batches,eval_batches,post_batches,post_eval,history_batches,step,h,loss_fn,cycle):
    anchor_rng=torch.get_rng_state();start=time.perf_counter()
    native,nopt=deepcopy((model,opt));torch.set_rng_state(anchor_rng);native_seconds=train(native,nopt,batches,loss_fn)
    LN=evaluate(native,eval_batches,loss_fn)
    expanded,eopt=deepcopy((model,opt));torch.set_rng_state(anchor_rng)
    engine=ETT(expanded,eopt,sites_for(expanded),rho=cfg['rho'],h=h,T=cfg['T'],maturity=cfg['maturity'],
        seed=cfg['ett_seed']+step)
    if len(history_batches)!=engine.buffer.capacity:raise ValueError('wrong pre-anchor trained buffer size')
    engine.buffer.seen=step-len(history_batches)
    for batch in history_batches:engine.buffer.append(batch)
    pool_batch,_,_=engine.buffer.split()
    for age in engine.ages.values():age.fill_(step)
    engine.cycle=cycle;engine.steps=step;engine.begin(loss_fn)
    pool_energy=engine.active['gradient_energy'];training=0.;energy=None
    if h==1:
        with measurement(expanded),torch.enable_grad():
            objective=loss_fn(expanded(batches[0][0]),batches[0][1])
            gradients=torch.autograd.grad(objective,[site.last.weight for site in engine.sites])
            energy=sum(float(g[:,engine.active['original'][site.incoming]:].square().sum()) for site,g in zip(engine.sites,gradients))
    for batch in batches:
        seconds=train(expanded,eopt,[batch],loss_fn);training+=seconds;engine.after_step(batch,seconds)
    LE=evaluate(expanded,eval_batches,loss_fn)
    GE_train=measured_loss(native,batches[0],loss_fn)-measured_loss(expanded,batches[0],loss_fn) if h==1 else None
    records=engine.contract(loss_fn,acceptance=False)
    LC=evaluate(expanded,eval_batches,loss_fn)
    reset_losses={};reset_counts={};branches={'N':(native,nopt),'C':(expanded,eopt)}
    for name,gradmax in (('CBP',False),('GM',True)):
        reset,ropt=deepcopy((model,opt));torch.set_rng_state(anchor_rng)
        rgnt=cbp.make(reset,ropt,cfg);cbp.restore(rgnt,cbp.history(gnt))
        reset_counts[name]=cbp.single_event(rgnt,sites_for(reset),batches[0],loss_fn,torch.Generator().manual_seed(cfg['ett_seed']+step),gradmax,pool_batch=pool_batch)
        train(reset,ropt,batches,loss_fn);reset_losses[name]=evaluate(reset,eval_batches,loss_fn);branches[name]=(reset,ropt)
    post_losses={};post_seconds={}
    for name,(branch,bopt) in branches.items():
        torch.set_rng_state(anchor_rng);post_seconds[name]=train(branch,bopt,post_batches,loss_fn)
        post_losses[name]=evaluate(branch,post_eval,loss_fn)
    torch.set_rng_state(anchor_rng)
    k=sum(r['k'] for r in records);total_units=sum(s.width for s in engine.sites)+k
    return dict(window_protocol='trained_1000_80_20_v1',setting=cfg['setting'],step=step,h=h,LN=LN,LE=LE,LC=LC,LCBP=reset_losses['CBP'],LGM=reset_losses['GM'],
        GE=LN-LE,delta=LC-LE,net=LN-LC,cbp_minus_compact=reset_losses['CBP']-LC,gm_minus_compact=reset_losses['GM']-LC,
        LN_post=post_losses['N'],LC_post=post_losses['C'],LCBP_post=post_losses['CBP'],LGM_post=post_losses['GM'],
        net_post=post_losses['N']-post_losses['C'],post_train_seconds=post_seconds,
        GE_train=GE_train,first_order_prediction=cfg['lr']*energy if energy is not None else None,
        first_order_ratio=GE_train/(cfg['lr']*energy) if energy is not None and energy>0 else None,
        pool_gradient_energy=pool_energy,buffer_samples=engine.buffer.capacity,
        virtual_kept_fraction=sum(r['virtual_kept'] for r in records)/max(k,1),
        dead_fraction=sum(r['dead_units'] for r in records)/max(total_units,1),
        g_score=sum(r['g_score'] for r in records),g_back=sum(r['g_back'] for r in records),
        g_actual_data=sum(r['g_actual_data'] for r in records),buffer_check_delta=sum(r['delta'] for r in records),
        reset_counts=reset_counts,cycle_logs=records,native_train_seconds=native_seconds,expanded_train_seconds=training,
        seconds=time.perf_counter()-start)


def state_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str((tensor.dtype,tuple(tensor.shape))).encode());digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def check_storage(path):
    existing=Path(path).resolve()
    while not existing.exists():existing=existing.parent
    free=shutil.disk_usage(existing).free;required=2*1024**3
    if free<required:raise OSError(f'Phase0 requires at least 2 GiB free before running; available={free} bytes, required={required}')
    return free


def reusable_anchors(cfg,batch_fingerprint):
    root=Path(cfg['reuse_native_from']);latest=torch.load(root/'latest.pt',weights_only=False)
    if latest['data_fingerprint']!=batch_fingerprint:raise ValueError('native reuse dataset fingerprint mismatch')
    native_keys=('setting','seed','data_seed','init_seed','steps','inputs','hidden','outputs','lr','flip_after','target_features','flipping_bits','beta','tasks','cbp_maturity','rho')
    result={}
    for path in sorted((root/'checkpoints').glob('*.pt')):
        saved=torch.load(path,weights_only=False);old=saved['protocol']['config']
        if any(old.get(k)!=cfg.get(k) for k in native_keys):raise ValueError('native checkpoint protocol mismatch: '+str(path))
        if saved['protocol']['commit']!=cfg['reuse_source']:raise ValueError('unregistered native source')
        step=saved['step']
        if step!=int(path.stem):raise ValueError('native checkpoint coordinate mismatch')
        result[step]=dict(path=str(path),source=saved['protocol']['commit'],state_dict_sha256=state_hash(saved['model']),
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def run(args):
    free_bytes=check_storage(args.out)
    print(f'Phase0 storage preflight free_bytes={free_bytes} required_bytes={2*1024**3}',flush=True)
    cfg=json.loads(Path(args.config).read_text());out=Path(args.out)
    if (out/'VALIDITY.json').exists() and json.loads((out/'VALIDITY.json').read_text()).get('status')=='INVALID_WINDOW':
        raise ValueError('Choose a fresh output tag; INVALID_WINDOW measurements must remain unchanged')
    out.mkdir(parents=True,exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    protocol=dict(commit=commit,config=cfg,phase=0,window_protocol='trained_1000_80_20_v1',device='cpu',dtype='float32',batch_size=1,
                  config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                  required_free_bytes=2*1024**3,evaluation_samples=cfg['evaluation_samples'],post_steps=cfg['post_steps'],
                  first_order_scope='same b0 after one native/expanded SGD step; not future evaluation loss')
    torch.set_num_threads(1)
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
    reuse=reusable_anchors(cfg,data_fingerprint)
    reserve=max(cfg['horizons'])+2*cfg['evaluation_samples']
    first=cfg['buffer_samples']
    preferred=[t for t in reuse if first<=t<cfg['steps']-reserve and t%task_steps+reserve<task_steps]
    remaining=cfg['checkpoints']-len(preferred)
    anchors=sorted(set(preferred+[min(int(t),(int(t)//task_steps+1)*task_steps-reserve-1)
        for t in torch.linspace(first,cfg['steps']-reserve-1,remaining,dtype=torch.double).tolist()]))
    anchors=[t for t in anchors if t>=first]
    reuse={step:record for step,record in reuse.items() if step in anchors}
    protocol.update(native_anchor_reuse=reuse,anchor_steps=anchors)
    # JSON object keys must be stable across checkpoint and on-disk protocol validation.
    protocol=json.loads(json.dumps(protocol))
    if (out/'protocol.json').exists():
        previous=json.loads((out/'protocol.json').read_text())
        if previous!=protocol:raise ValueError('Phase0 reuse manifest mismatch')
    atomic_json_save(protocol,out/'protocol.json')
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
        flat=[{k:v for k,v in r.items() if k not in ('cycle_logs','reset_counts','post_train_seconds')} for r in rows]
        write_csv(out/'measurements.csv',flat)
    try:
        while step<cfg['steps']:
            if step-last_checkpoint>=cfg['checkpoint_every'] and shutil.disk_usage(out).free<2*1024**3:
                save('PAUSED_STORAGE');return 75
            if args.deadline_seconds and time.perf_counter()-start>=args.deadline_seconds:
                save('PAUSED');return 75
            future=[t for t in anchors if t>step]
            if step not in anchors and future and min(future) in reuse:
                target=min(future);record=reuse[target];saved=torch.load(record['path'],weights_only=False)
                if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['file_sha256']:raise ValueError('native checkpoint file changed')
                if state_hash(saved['model'])!=record['state_dict_sha256']:raise ValueError('native source changed')
                model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);cbp.restore(gnt,saved['gnt'])
                if state_hash(model.state_dict())!=record['state_dict_sha256']:raise ValueError('native loaded hash mismatch')
                torch.set_rng_state(saved['rng']);step=target
                print(f"REUSE native anchor={step} hash={record['state_dict_sha256']}",flush=True)
            if step in anchors:
                for h in cfg['horizons']:
                    if any(r['step']==step and r['h']==h for r in rows):continue
                    sequence=[batch(step+j) for j in range(h)]
                    evaluation=[combine([batch(step+max(cfg['horizons'])+j) for j in range(cfg['evaluation_samples'])])]
                    post=[batch(step+h+j) for j in range(cfg['post_steps'])]
                    post_eval=[combine([batch(step+max(cfg['horizons'])+cfg['evaluation_samples']+j) for j in range(cfg['evaluation_samples'])])]
                    history_batches=[batch(j) for j in range(step-cfg['buffer_samples'],step)]
                    atomic_torch_save(dict(protocol=protocol,step=step,model=model.state_dict(),optimizer=opt.state_dict(),gnt=cbp.history(gnt)),out/'checkpoints'/f'{step}.pt')
                    row=paired(model,opt,gnt,cfg,sequence,evaluation,post,post_eval,history_batches,step,h,loss_fn,anchors.index(step))
                    row.update(native_state_dict_sha256=state_hash(model.state_dict()),native_reused=step in reuse)
                    rows.append(row);save('RUNNING')
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
