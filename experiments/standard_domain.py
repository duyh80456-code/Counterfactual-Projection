"""H4 Rotated/Permuted MNIST ER controls and fixed-head replay correction."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.standard_cl import ExperimentLogger,matrix_metrics,ARMS
from experiments.vna_resume import Progress,rng_state,restore_rng,deadline_due,log
from experiments.vna_provenance import code_commit
from methods.cl_h4 import DomainMNIST,DomainNetwork,Reservoir,domain_losses
from methods.cl_replay import CorrectionConfig,ReplayCorrection
from methods.cl_state import State,isolated,evaluate


def accuracies(model,datasets,device):
    result=[]
    with isolated(model),torch.no_grad():
        model.eval()
        for dataset in datasets:
            correct=total=0
            for x,y in DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0):
                logits=model(x.to(device))
                if not bool(torch.isfinite(logits).all()):raise RuntimeError('nonfinite domain evaluation logits')
                correct+=int((logits.argmax(1).cpu()==y).sum());total+=len(y)
            result.append(100*correct/total)
    return result


def run(args):
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    logger=ExperimentLogger(args.out);out=Path(args.out)
    if args.normalization=='groupnorm' and ARMS[args.arm] in ('tiny_V','random'):
        raise ValueError('GroupNorm virtual-neuron construction is not implemented; use native/GN/baseline control')
    if args.batch_size<2 or args.epochs<1 or args.domains<1:raise ValueError('positive domains/epochs and batch size>=2 required')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=DomainNetwork(args.normalization).to(args.device)
    if args.device!='cpu':torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.)
    memory=Reservoir(args.memory_size,args.seed)
    config=CorrectionConfig(kernel=ARMS[args.arm] or 'tiny_V',constrained=args.arm!='vna',mode=args.mode,seed=args.seed)
    engine=ReplayCorrection(config)
    base_train=MNIST(args.data_dir,train=True,download=False,transform=ToTensor())
    base_test=MNIST(args.data_dir,train=False,download=False,transform=ToTensor())
    import hashlib
    hashes={str(path.relative_to(Path(args.data_dir))):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((Path(args.data_dir)/'MNIST/raw').glob('*')) if path.is_file()}
    from diagnostics.kernel_ablation import flags
    protocol=dict(branch='H4',code_commit=code_commit(),config={k:v for k,v in vars(args).items() if k!='out'},runtime_flags=flags(),
        dataset_hashes=hashes,correction=asdict(config),classifier_classes=10,head_expansion=False,
        memory='reservoir of previously observed transformed samples, including current-domain prefix',
        evidence_status='INCONCLUSIVE',test_gates='NOT_RUN; code-only request')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('domain protocol mismatch')
    logger.save('protocol',protocol)
    if (out/'summary.json').exists():
        previous=json.loads((out/'summary.json').read_text())
        if previous.get('protocol')!=protocol:raise ValueError('domain summary protocol mismatch')
        if previous.get('status')=='COMPLETED':
            if len(previous.get('tasks',[]))!=args.domains:raise ValueError('incomplete domain summary')
            return previous
    progress=Progress(out,protocol);resume=progress.state
    rows=[];trajectory=[];steps=0;start_domain=start_epoch=0;elapsed_base=0.
    if resume:
        progress.restore_artifacts();model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
        memory.load_state_dict(resume['memory']);engine.load_state_dict(resume['engine']);restore_rng(resume['rng'])
        rows,trajectory,steps=resume['rows'],resume['trajectory'],resume['steps']
        start_domain,start_epoch=resume['domain'],resume['epoch'];elapsed_base=resume['elapsed_seconds']
    started=time.monotonic();status='COMPLETED';failure=None;failure_coordinates={}
    tests=[DomainMNIST(base_test,d,args.kind,args.seed,args.angle_step) for d in range(args.domains)]
    def save(domain,epoch):
        logger.save('tasks',rows);logger.save('interventions',engine.events);logger.save('trajectory',trajectory)
        progress.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),memory=memory.state_dict(),engine=engine.state_dict(),
            rng=rng_state(),rows=rows,trajectory=trajectory,steps=steps,domain=domain,epoch=epoch,
            elapsed_seconds=elapsed_base+time.monotonic()-started))
    try:
        for domain in range(start_domain,args.domains):
            dataset=DomainMNIST(base_train,domain,args.kind,args.seed,args.angle_step)
            loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,drop_last=True)
            for epoch in range(start_epoch if domain==start_domain else 0,args.epochs):
                if deadline_due():save(domain,epoch);raise SystemExit(75)
                snapshot=State(model,optimizer);old_memory=deepcopy(memory.state_dict());old_engine=engine.state_dict()
                epoch_steps=steps;points=len(trajectory)
                try:
                    for x,y in loader:
                        native_buffers={n:b.detach().clone() for n,b in model.named_buffers()}
                        model.train();x,y=x.to(args.device),y.to(args.device);incoming=x.detach().clone();incoming_y=y.detach().clone()
                        replay=memory.sample(args.batch_size,args.device)
                        mask=torch.zeros(len(y),dtype=torch.bool,device=args.device)
                        if replay is not None:
                            old_x,old_y,_=replay;x=torch.cat((x,old_x));y=torch.cat((y,old_y))
                            mask=torch.cat((mask,torch.ones(len(old_y),dtype=torch.bool,device=args.device)))
                        losses=domain_losses(y,mask);optimizer.zero_grad();z=model(x);loss=losses.total(z)
                        if not bool(torch.isfinite(loss)):
                            failure_coordinates.update(domain=domain,epoch=epoch+1,step=steps+1,tensor='loss')
                            raise FloatingPointError(f'nonfinite loss {failure_coordinates}')
                        loss.backward();steps+=1
                        if args.mode=='intervention' or args.arm=='baseline':optimizer.step()
                        if args.arm!='baseline':
                            row=engine.step(model,optimizer,x,losses,model.sites(),args.lr)
                            row.update(domain=domain,epoch=epoch+1,step=steps)
                            row['native_bn_buffer_drift']=float(sum((b.double()-native_buffers[n].double()).square().sum()
                                for n,b in model.named_buffers()).sqrt()) if native_buffers else 0.
                            row['frozen_bn_window_loss']=evaluate(model,x,losses.components,'frozen_bn')
                        memory.add(incoming,incoming_y,domain)
                        if deadline_due():raise SystemExit(75)
                    trajectory.append(dict(domain=domain,epoch=epoch+1,domain_accuracies=accuracies(model,tests[:domain+1],args.device)))
                    save(domain,epoch+1);log(f'H4 {args.arm} domain={domain} epoch={epoch+1}/{args.epochs}')
                except SystemExit as exc:
                    if exc.code!=75:raise
                    snapshot.restore();memory.load_state_dict(old_memory);engine.load_state_dict(old_engine)
                    steps=epoch_steps;del trajectory[points:];save(domain,epoch);raise
            row=dict(domain=domain,domain_accuracies=accuracies(model,tests[:domain+1],args.device))
            rows.append(row);save(domain+1,0)
    except SystemExit as exc:
        status='PAUSED' if exc.code==75 else 'INTERRUPTED';raise
    except KeyboardInterrupt as exc:status='INTERRUPTED';failure=repr(exc);raise
    except FloatingPointError as exc:status='FAILED_DIVERGED';failure=str(exc)
    except torch.cuda.OutOfMemoryError as exc:status='FAILED_RESOURCE_OOM';failure=str(exc)
    except Exception as exc:status='FAILED_ERROR';failure=repr(exc);raise
    finally:
        seconds=elapsed_base+time.monotonic()-started
        logger.save('summary',dict(status=status,failure=failure,diverged=failure_coordinates or None,protocol=protocol,tasks=rows,
            **matrix_metrics([r['domain_accuracies'] for r in rows]),wall_seconds=seconds,
            parameter_count=sum(p.numel() for p in model.parameters()),
            peak_vram_bytes=torch.cuda.max_memory_allocated(torch.device(args.device)) if args.device!='cpu' else 0,
            gpu_hours=seconds/3600 if args.device!='cpu' else 0.,evidence_status='INCONCLUSIVE'))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--kind',choices=('rotated','permuted'),default='rotated');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--arm',choices=ARMS,default='h1');p.add_argument('--normalization',choices=('bn','groupnorm'),default='bn')
    p.add_argument('--mode',choices=('intervention','unified'),default='intervention')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--domains',type=int,default=5);p.add_argument('--angle-step',type=float,default=15.)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--memory-size',type=int,default=2000)
    p.add_argument('--lr',type=float,default=.01);p.add_argument('--device',default='cuda:0')
    return p


if __name__=='__main__':run(parser().parse_args())
