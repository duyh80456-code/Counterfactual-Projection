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
from experiments.vna_resume import Progress,rng_state,restore_rng,deadline_due,log,reset_cuda_peak_memory_stats
from experiments.vna_provenance import code_commit,source_hashes
from methods.cl_h4 import DomainMNIST,DomainNetwork,Reservoir,domain_losses
from methods.cl_replay import CorrectionConfig,ReplayCorrection
from methods.cl_state import State,isolated,evaluate,predict


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
    if args.arm in ('h2','lpr'):raise ValueError('H2/LPR are independent CIL branches')
    if args.normalization=='groupnorm' and ARMS[args.arm] in ('tiny_V','random'):
        raise ValueError('GroupNorm virtual-neuron construction is not implemented; use native/GN/baseline control')
    if args.batch_size<2 or args.epochs<1 or args.domains<1:raise ValueError('positive domains/epochs and batch size>=2 required')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=DomainNetwork(args.normalization,width=16 if args.fake_data else 256).to(args.device)
    reset_cuda_peak_memory_stats(args.device)
    optimizer=torch.optim.SGD(model.parameters(),lr=args.lr,momentum=0.,weight_decay=args.weight_decay)
    memory=Reservoir(args.memory_size,args.seed)
    numerical=dict(damping_relative=1.,cg_tolerance=1e-4,cg_iterations=100,trace_probes=1,max_trials=3) if args.fake_data else {}
    if args.solver=='cg':numerical.update(preconditioner_probes=0,damping_retries=0)
    numerical['eval_acceptance']=args.eval_acceptance
    config=CorrectionConfig(kernel=ARMS[args.arm] or 'tiny_V',constrained=args.arm!='vna',mode=args.mode,seed=args.seed,
        candidate_strategy=args.candidate_strategy,physical_update_norm=args.physical_update_norm,
        physical_budget_policy=args.physical_budget_policy,**numerical)
    engine=ReplayCorrection(config)
    if args.fake_data:
        from torch.utils.data import TensorDataset
        generator=torch.Generator().manual_seed(981)
        base_train=TensorDataset(torch.rand(8,1,28,28,generator=generator),torch.arange(8)%10)
        base_test=TensorDataset(torch.rand(4,1,28,28,generator=generator),torch.arange(4)%10)
    else:
        base_train=MNIST(args.data_dir,train=True,download=False,transform=ToTensor())
        base_test=MNIST(args.data_dir,train=False,download=False,transform=ToTensor())
    import hashlib
    hashes={str(path.relative_to(Path(args.data_dir))):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(args.data_dir)/'MNIST/raw'/name for name in
                ('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')) if path.is_file()}
    from diagnostics.kernel_ablation import flags
    tests=[DomainMNIST(base_test,d,args.kind,args.seed,args.angle_step) for d in range(args.domains)]
    protocol=dict(branch='H4',code_commit=code_commit(),source_hashes=source_hashes(),config={k:v for k,v in vars(args).items() if k!='out'},runtime_flags=flags(),
        synthetic_smoke=args.fake_data,split_sizes=dict(train=len(base_train),test=len(base_test)),
        transformations=[dataset.identity() for dataset in tests],
        dataset_hashes=hashes,correction=asdict(config),classifier_classes=10,head_expansion=False,
        domain_reference=args.domain_reference,
        domain_mechanism='training-only domain IDs partition current/past CE constraints; shared fixed head and backbone; no shift estimator or domain-conditioned features',acceptance='train and eval componentwise nonincrease within dtype-scaled roundoff',
        memory='reservoir of previously observed transformed samples, including current-domain prefix',
        evidence_status='INCONCLUSIVE',test_gates='prelaunch CPU mathematical/synthetic validation; benchmark and CUDA validation pending')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('domain protocol mismatch')
    logger.save('protocol',protocol)
    if (out/'summary.json').exists():
        previous=json.loads((out/'summary.json').read_text())
        if previous.get('protocol')!=protocol:raise ValueError('domain summary protocol mismatch')
        if previous.get('status')=='COMPLETED':
            if len(previous.get('tasks',[]))!=args.domains:raise ValueError('incomplete domain summary')
            return previous
    progress=Progress(out,protocol);resume=progress.state
    rows=[];trajectory=[];normalization=[];steps=0;start_domain=start_epoch=0;elapsed_base=0.
    if resume:
        progress.restore_artifacts();model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
        memory.load_state_dict(resume['memory']);engine.load_state_dict(resume['engine']);restore_rng(resume['rng'])
        rows,trajectory,steps=resume['rows'],resume['trajectory'],resume['steps']
        normalization=resume['normalization']
        start_domain,start_epoch=resume['domain'],resume['epoch'];elapsed_base=resume['elapsed_seconds']
    started=time.monotonic();status='COMPLETED';failure=None;failure_coordinates={}
    def save(domain,epoch):
        logger.save('tasks',rows);logger.save('accuracy_matrix',[r['domain_accuracies'] for r in rows]);logger.save('interventions',engine.events);logger.save('trajectory',trajectory);logger.save('normalization',normalization)
        progress.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),memory=memory.state_dict(),engine=engine.state_dict(),
            rng=rng_state(),rows=rows,trajectory=trajectory,normalization=normalization,steps=steps,domain=domain,epoch=epoch,
            elapsed_seconds=elapsed_base+time.monotonic()-started))
    try:
        for domain in range(start_domain,args.domains):
            dataset=DomainMNIST(base_train,domain,args.kind,args.seed,args.angle_step)
            loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,drop_last=True)
            for epoch in range(start_epoch if domain==start_domain else 0,args.epochs):
                if deadline_due():save(domain,epoch);raise SystemExit(75)
                snapshot=State(model,optimizer);old_memory=deepcopy(memory.state_dict());old_engine=engine.state_dict()
                epoch_steps=steps;points=len(trajectory);normalization_points=len(normalization)
                try:
                    for x,y in loader:
                        native_buffers={n:b.detach().clone() for n,b in model.named_buffers()}
                        model.train();x,y=x.to(args.device),y.to(args.device);incoming=x.detach().clone();incoming_y=y.detach().clone()
                        replay=memory.sample(args.batch_size,args.device)
                        mask=torch.zeros(len(y),dtype=torch.bool,device=args.device)
                        if replay is not None:
                            old_x,old_y,_=replay;x=torch.cat((x,old_x));y=torch.cat((y,old_y))
                            mask=torch.cat((mask,torch.ones(len(old_y),dtype=torch.bool,device=args.device)))
                        eval_before=predict(model,x)
                        losses=domain_losses(y,mask,current_domain=domain if args.domain_reference=='previous' else None,
                            replay_domains=None if replay is None else replay[2]) if replay is not None else domain_losses(y,mask);optimizer.zero_grad();z=model(x);loss=losses.total(z)
                        if not bool(torch.isfinite(loss)):
                            failure_coordinates.update(domain=domain,epoch=epoch+1,step=steps+1,tensor='loss')
                            raise FloatingPointError(f'nonfinite loss {failure_coordinates}')
                        loss.backward();steps+=1
                        if args.mode=='intervention' or args.arm=='baseline':optimizer.step()
                        if args.arm!='baseline':
                            row=engine.step(model,optimizer,x,losses,model.sites(),args.lr)
                            row.update(domain=domain,epoch=epoch+1,step=steps,
                                current_domain_samples=int((~losses.replay).sum()),previous_domain_samples=int(losses.replay.sum()),
                                reference_policy=args.domain_reference)
                            row['native_bn_buffer_drift']=float(sum((b.double()-native_buffers[n].double()).square().sum()
                                for n,b in model.named_buffers()).sqrt()) if native_buffers else 0.
                            row['frozen_bn_window_loss']=evaluate(model,x,losses.components,'frozen_bn')
                        eval_after=predict(model,x)
                        with isolated(model),torch.no_grad():
                            for n,b in model.named_buffers():b.copy_(native_buffers[n])
                            eval_old_buffers=predict(model,x)
                        normalization.append(dict(step=steps,domain=domain,
                            eval_total_drift=float((eval_after.double()-eval_before.double()).norm()),
                            eval_weight_only_drift=float((eval_old_buffers.double()-eval_before.double()).norm()),
                            eval_buffer_only_drift=float((eval_after.double()-eval_old_buffers.double()).norm()),
                            scope='exact finite-change decomposition; norms need not add; no eval protection theorem'))
                        memory.add(incoming,incoming_y,domain)
                        if deadline_due():raise SystemExit(75)
                    trajectory.append(dict(domain=domain,epoch=epoch+1,domain_accuracies=accuracies(model,tests[:domain+1],args.device)))
                    save(domain,epoch+1);log(f'H4 {args.arm} domain={domain} epoch={epoch+1}/{args.epochs} accuracy={trajectory[-1]["domain_accuracies"]}')
                except SystemExit as exc:
                    if exc.code!=75:raise
                    snapshot.restore();memory.load_state_dict(old_memory);engine.load_state_dict(old_engine)
                    steps=epoch_steps;del trajectory[points:];del normalization[normalization_points:];save(domain,epoch);raise
            row=dict(domain=domain,domain_accuracies=accuracies(model,tests[:domain+1],args.device))
            row.update(new_task_accuracy=row['domain_accuracies'][-1],old_task_accuracy=float(np.mean(row['domain_accuracies'][:-1])) if domain else None)
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
            correction_seconds=sum(row['seconds'] for row in engine.events),
            gpu_hours=seconds/3600 if args.device!='cpu' else 0.,evidence_status='INCONCLUSIVE'))
    return json.loads((out/'summary.json').read_text())


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--kind',choices=('rotated','permuted'),default='rotated');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--arm',choices=tuple(a for a in ARMS if a not in ('h2','lpr')),default='h1');p.add_argument('--normalization',choices=('bn','groupnorm'),default='bn')
    p.add_argument('--solver',choices=('pcg','cg'),default='pcg')
    p.add_argument('--candidate-strategy',choices=('original','pareto_minimax'),default='original')
    p.add_argument('--physical-update-norm',type=float,default=None)
    p.add_argument('--physical-budget-policy',choices=('cap','matched'),default='cap')
    p.add_argument('--eval-acceptance',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--domain-reference',choices=('previous','mixed'),default='previous')
    p.add_argument('--mode',choices=('intervention','unified'),default='intervention')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--domains',type=int,default=5);p.add_argument('--angle-step',type=float,default=15.)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--memory-size',type=int,default=2000)
    p.add_argument('--lr',type=float,default=.01);p.add_argument('--device',default='cuda:0')
    p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--config',default='configs/standard_cl/h4_rotated_mnist.json')
    p.add_argument('--fake-data',action='store_true')
    return p


def parse_args(argv=None):
    p=parser();known,_=p.parse_known_args(argv)
    defaults=json.loads(Path(known.config).read_text())
    if set(defaults)-{a.dest for a in p._actions}:raise ValueError('unknown Domain-IL config field')
    p.set_defaults(**defaults)
    return p.parse_args(argv)


if __name__=='__main__':run(parse_args())
