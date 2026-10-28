"""H2: upstream LPR online Split-CIFAR100, with a Geometry-VNA comparison."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from experiments.standard_cl import ExperimentLogger,matrix_metrics
from experiments.vna_resume import Progress,rng_state,restore_rng,deadline_due,log,reset_cuda_peak_memory_stats
from experiments.vna_provenance import code_commit,source_hashes
from methods.cl_h2 import GeometryCorrection
from methods.cl_lpr import ROOT,PIN,SOURCE_SHA256,official_module
from methods.cl_replay import CorrectionConfig,ReplayLoss
from methods.cl_state import isolated,State
from methods.vna_sites import HookSite

if importlib.util.find_spec('avalanche'):
    from avalanche.training.plugins import SupervisedPlugin,ReplayPlugin
else:
    SupervisedPlugin=ReplayPlugin=object


class OriginDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,replay):self.dataset,self.replay=dataset,bool(replay);self.targets=dataset.targets
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        item=self.dataset[index]
        if len(item)!=3:raise ValueError('expected Avalanche classification sample (x,y,task)')
        return item[0],item[1],self.replay


def with_origin(dataset,replay):
    from avalanche.benchmarks.utils import make_classification_dataset
    return make_classification_dataset(OriginDataset(dataset,replay),task_labels=dataset.targets_task_labels)


class OriginReplayPlugin(ReplayPlugin):
    """Tag H2 replay explicitly; keep native LPR's ReplayPlugin untouched."""
    def before_training_exp(self,strategy,num_workers=0,shuffle=True,drop_last=False,**kwargs):
        self.has_replay=len(self.storage_policy.buffer)>0
        if not self.has_replay:return
        from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
        strategy.dataloader=ReplayDataLoader(with_origin(strategy.adapted_dataset,False),with_origin(self.storage_policy.buffer,True),
            oversample_small_tasks=True,batch_size=self.batch_size or strategy.train_mb_size,
            batch_size_mem=self.batch_size_mem or strategy.train_mb_size,task_balanced_dataloader=self.task_balanced_dataloader,
            num_workers=num_workers,shuffle=shuffle,drop_last=drop_last)
    def before_training_iteration(self,strategy,**kwargs):
        if not self.has_replay:
            strategy.mbatch.insert(-1,torch.zeros_like(strategy.mb_y,dtype=torch.bool))
        replay_mask(strategy)


def replay_mask(strategy):
    if len(strategy.mbatch)!=4:raise ValueError('missing explicit H2 sample origin')
    mask=strategy.mbatch[-2]
    if mask.dtype!=torch.bool or mask.shape!=strategy.mb_y.shape:raise ValueError('invalid H2 sample origin')
    return mask


class OnlineNetwork(nn.Module):
    def __init__(self,net):
        super().__init__();self.net=net;self.virtual_relu=nn.ReLU()
    def forward(self,x):return self.net(x)
    def standard_features(self,x):
        n=self.net
        x=torch.relu(n.bn1(n.conv1(x)))
        for layer in (n.layer1,n.layer2,n.layer3,n.layer4):x=layer(x)
        return torch.nn.functional.avg_pool2d(x,4).flatten(1)
    def sites(self):
        return [HookSite(self,name+'.conv1',name+'.conv2','conv',between=(name+'.bn1','virtual_relu'))
                for name,module in self.named_modules() if hasattr(module,'conv1') and hasattr(module,'conv2')]


class GeometrySelectionFailed(RuntimeError):
    pass


class GeometryPlugin(SupervisedPlugin):
    def __init__(self,cfg,seed):
        self.cfg=cfg;self.engine=GeometryCorrection(CorrectionConfig(seed=seed,**cfg['correction']));self.last_event=0;self.selection=None;self.selection_attempts=[];self.seen_classes=set();self.validation_stream=None;self.selection_path=None
        policy=cfg.get('geometry_selection')
        if policy and (not isinstance(policy.get('first_task'),int) or policy['first_task']<2 or not isinstance(policy.get('max_windows',3),int) or policy.get('max_windows',3)<1):raise ValueError('Geometry pilot must start with old classes and have a positive window budget')
    def select_geometry(self,strategy,x,losses,mask):
        import hashlib
        from time import perf_counter
        from torch.utils.data import DataLoader,Subset
        from experiments.io_utils import atomic_json_save
        policy=self.cfg['geometry_selection'];model,opt=strategy.model,strategy.optimizer
        if self.validation_stream is None:raise ValueError('Geometry selection requires the held-out validation stream')
        weights=policy['weights']
        if not weights or len(set(weights))!=len(weights) or any(not np.isfinite(w) or w<0 for w in weights):raise ValueError('finite nonnegative geometry grid required')
        limit=policy['validation_per_task']
        if not isinstance(limit,int) or limit<1:raise ValueError('positive validation sample budget required')
        snapshot=State(model,opt);started=perf_counter();trials=[];validation=[]
        try:
            for experience in self.validation_stream:
                dataset=experience.dataset.eval() if hasattr(experience.dataset,'eval') else experience.dataset
                subset=Subset(dataset,list(range(min(limit,len(dataset)))))
                for batch in DataLoader(subset,batch_size=128,shuffle=False,num_workers=0):
                    vx,vy=batch[0].to(x.device),batch[1].to(x.device)
                    known=torch.isin(vy,torch.tensor(sorted(self.seen_classes),device=vy.device,dtype=vy.dtype))
                    if bool(known.any()):validation.append((vx[known],vy[known]))
            if not validation:raise ValueError('Empty held-out validation stream')
            fingerprint=hashlib.sha256()
            for vx,vy in validation:
                fingerprint.update(vx.detach().cpu().numpy().tobytes());fingerprint.update(vy.detach().cpu().numpy().tobytes())
            def validation_loss():
                with isolated(model,opt),torch.no_grad():
                    model.eval();total=count=0
                    for vx,vy in validation:
                        total+=float(torch.nn.functional.cross_entropy(model(vx),vy,reduction='sum'));count+=len(vy)
                    return total/count
            snapshot.restore();baseline=validation_loss()
            for weight in sorted(weights):
                log(f'H2 geometry pilot task={self.task+1} window={len(self.selection_attempts)+1} weight={weight} starting')
                snapshot.restore();engine=GeometryCorrection(self.engine.config);engine.applications=self.engine.applications
                event=engine.step(model,opt,x,losses,model.sites(),self.cfg['lr'],replay_inputs=x[mask],geometry_weight=weight)
                score=validation_loss() if event.get('accepted') and event.get('step_norm',0)>0 else None
                trials.append(dict(weight=weight,validation_loss=score,event=event))
                log(f'H2 geometry pilot weight={weight} status={event["status"]} validation_loss={score}')
            eligible=[r for r in trials if r['validation_loss'] is not None and np.isfinite(r['validation_loss'])]
            best=min(eligible,key=lambda r:(r['validation_loss'],r['weight'])) if eligible else None
            report=dict(status='SELECTED' if best else 'NO_CERTIFIED_NONZERO_CANDIDATE',
                selected_weight=best['weight'] if best else None,task=self.task+1,
                source='held-out seen-task validation only; never test data',validation_sha256=fingerprint.hexdigest(),
                validation_samples=sum(len(vy) for _,vy in validation),seen_classes=sorted(self.seen_classes),baseline_validation_loss=baseline,
                selected_validation_loss=best['validation_loss'] if best else None,
                validation_improved=best['validation_loss']<baseline if best else False,
                trials=trials,seconds=perf_counter()-started)
            if self.selection_path:atomic_json_save(report,self.selection_path)
            return report
        finally:snapshot.restore()

    def after_update(self,strategy,**kwargs):
        if self.cfg.get('geometry_selection'):self.seen_classes.update(strategy.mb_y.detach().cpu().tolist())
        step=strategy.clock.train_iterations+1
        if step-self.last_event<self.cfg['intervention_every']:return
        x,y=strategy.mb_x,strategy.mb_y
        mask=replay_mask(strategy)
        if not bool(mask.any()):return
        losses=ReplayLoss(y,mask,kind='er')
        selection=None
        if self.cfg.get('geometry_selection') and self.task+1>=self.cfg['geometry_selection']['first_task'] and self.selection is None:
            selection=self.select_geometry(strategy,x,losses,mask)
            selection.update(step=step,window=len(self.selection_attempts)+1);self.selection_attempts.append(selection)
            if self.selection_path:
                from experiments.io_utils import atomic_json_save
                atomic_json_save(dict(selection,windows=self.selection_attempts),self.selection_path)
            if selection['selected_weight'] is None:
                limit=self.cfg['geometry_selection'].get('max_windows',3)
                self.last_event=step
                self.engine.events.append(dict(status='SKIPPED_GEOMETRY_SELECTION',accepted=False,
                    application=self.engine.applications,task=self.task+1,step=step,seconds=selection['seconds'],
                    geometry_selection=selection,
                    certified=any(t['event'].get('certified',False) for t in selection['trials']),
                    candidate_generated=any(t['event'].get('candidate_generated',False) for t in selection['trials']),
                    jvp=sum(t['event'].get('jvp',0) for t in selection['trials']),
                    vjp=sum(t['event'].get('vjp',0) for t in selection['trials']),
                    rejection_reason='GEOMETRY_SELECTION_NO_QUALIFIED_CANDIDATE',
                    reason='No qualified candidate in this window; continue bounded pilot'))
                self.engine.applications+=1
                if len(self.selection_attempts)>=limit:raise GeometrySelectionFailed('No qualified geometry correction within the registered validation pilot budget')
                return
            self.selection=selection
            log(f'H2 frozen geometry weight={selection["selected_weight"]} validation_loss={selection["selected_validation_loss"]} baseline_validation_loss={selection["baseline_validation_loss"]}')
        weight=self.selection['selected_weight'] if self.selection else self.cfg['geometry_weight']
        row=self.engine.step(strategy.model,strategy.optimizer,x,losses,strategy.model.sites(),self.cfg['lr'],
            replay_inputs=x[mask],geometry_weight=weight)
        row.update(task=self.task+1,step=step,effective_geometry_weight=weight)
        if selection is not None:
            row.update(geometry_selection=selection,geometry_selection_seconds=selection['seconds'])
            row['seconds']+=selection['seconds']
            for key in ('jvp','vjp'):row[key]=row.get(key,0)+sum(t['event'].get(key,0) for t in selection['trials'])
        row.update(incoming_samples=int((~mask).sum()),replay_samples=int(mask.sum()),origin_tracking='explicit sample field; order independent')
        self.last_event=step


class LPRCallbacks(SupervisedPlugin):
    def __init__(self,algorithm):self.algorithm=algorithm
    def before_backward(self,strategy,**kwargs):self.algorithm.before_backward(strategy,**kwargs)
    def after_backward(self,strategy,**kwargs):self.algorithm.after_backward(strategy,**kwargs)


class FiniteLoss(SupervisedPlugin):
    def before_training_exp(self,strategy,**kwargs):
        self.last_loss=None;self.last_accuracy=None
        active={id(p) for group in strategy.optimizer.param_groups for p in group['params']}
        # Avalanche0.4 retains empty entries for replaced heads; torch2.2 cannot serialize those orphans.
        for parameter,state in list(strategy.optimizer.state.items()):
            if id(parameter) not in active:
                if any(v is not None for v in state.values()):raise ValueError('refuse to discard nonempty orphan optimizer state')
                del strategy.optimizer.state[parameter]
    def before_backward(self,strategy,**kwargs):
        if not bool(torch.isfinite(strategy.loss)):
            raise FloatingPointError(f'nonfinite loss at online experience={strategy.clock.train_exp_counter}, step={strategy.clock.train_iterations+1}, tensor=loss')
        self.last_loss=float(strategy.loss.detach())
        if strategy.clock.train_exp_iterations==0:
            mask=~replay_mask(strategy) if len(strategy.mbatch)==4 else torch.arange(len(strategy.mb_y),device=strategy.mb_y.device)<strategy.train_mb_size
            self.last_accuracy=float((strategy.mb_output.detach().argmax(1)[mask]==strategy.mb_y[mask]).float().mean())*100


def native_setup(cfg,data_dir,seed):
    import avalanche
    if avalanche.__version__!='0.4.0':raise ValueError('H2 requires the upstream avalanche-lib==0.4.0')
    from avalanche.benchmarks.classic import SplitCIFAR100
    from avalanche.benchmarks import benchmark_with_validation_stream
    from avalanche.models import IncrementalClassifier
    from torchvision import transforms
    if not all((Path(data_dir)/'cifar-100-python'/name).is_file() for name in ('train','test','meta')):
        raise FileNotFoundError('Attach original CIFAR100 cifar-100-python/{train,test,meta}')
    official_module()
    path=ROOT/'src/toolkit/slim_resnet18.py'
    spec=importlib.util.spec_from_file_location('_standard_lpr_slim',path)
    import sys
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    normalize=transforms.Normalize((.5071,.4865,.4409),(.2673,.2564,.2762))
    train=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])
    test=transforms.Compose([transforms.ToTensor(),normalize])
    # Upstream seeds global RNG and passes seed=None to SplitCIFAR100.
    scenario=SplitCIFAR100(cfg['tasks'],seed=None,return_task_id=False,shuffle=True,
        class_ids_from_zero_in_each_exp=False,class_ids_from_zero_from_first_exp=True,
        train_transform=train,eval_transform=test,dataset_root=data_dir)
    class_order=list(scenario.classes_order_original_ids)
    scenario=benchmark_with_validation_stream(scenario,validation_size=cfg['validation_fraction'],shuffle=True)
    # Avalanche's validation wrapper returns a generic scenario without NC metadata.
    scenario.classes_order_original_ids=class_order
    net=module.SlimResNet18(1,input_size=(32,32,3))
    net.linear=IncrementalClassifier(net.linear.in_features,1)
    return scenario,OnlineNetwork(net)


def evaluate(model,stream,device):
    from torch.utils.data import DataLoader
    result=[]
    with isolated(model),torch.no_grad():
        model.eval()
        for experience in stream:
            correct=total=0
            for batch in DataLoader(experience.dataset,batch_size=128,shuffle=False,num_workers=0):
                x,y=batch[:2];z=model(x.to(device))
                if not bool(torch.isfinite(z).all()):raise FloatingPointError('nonfinite evaluation logits')
                correct+=int((z.argmax(1).cpu()==y).sum());total+=len(y)
            result.append(100*correct/total)
    return result


def run(args):
    from avalanche.benchmarks.scenarios import OnlineCLScenario
    from avalanche.training.plugins import ReplayPlugin
    from avalanche.training.storage_policy import ClassBalancedBuffer
    from avalanche.training.supervised import Naive
    from types import SimpleNamespace
    from diagnostics.kernel_ablation import flags
    cfg=json.loads(Path(args.config).read_text())
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    scenario,model=native_setup(cfg,args.data_dir,args.seed)
    model=model.to(args.device)
    storage=ClassBalancedBuffer(max_size=cfg['memory_size'],adaptive_size=True)
    replay=(ReplayPlugin if args.arm in ('lpr','er') else OriginReplayPlugin)(mem_size=cfg['memory_size'],batch_size_mem=cfg['train_mb_size'],storage_policy=storage)
    lpr_cfg=cfg['lpr']
    lpr=official_module().LPRPlugin({k:SimpleNamespace(**v) for k,v in lpr_cfg.items()},storage)
    geometry=GeometryPlugin(cfg,args.seed)
    optimizer=torch.optim.SGD(model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
    finite=FiniteLoss()
    def make_strategy(net,opt):
        return Naive(net,opt,torch.nn.CrossEntropyLoss(),train_mb_size=cfg['train_mb_size'],train_epochs=cfg['train_epochs'],
            eval_mb_size=128,device=args.device,evaluator=None,eval_every=-1,
            plugins=[replay,finite]+([LPRCallbacks(lpr)] if args.arm=='lpr' else [geometry] if args.arm=='h2' else []))
    strategy=make_strategy(model,optimizer)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(args.data_dir)/'cifar-100-python'/name for name in ('train','test','meta')) if p.is_file()}
    protocol=dict(branch='H2',arm=args.arm,seed=args.seed,code_commit=code_commit(),source_hashes=source_hashes(),config=cfg,tasks=cfg['tasks'],
        upstream_commit=PIN,lpr_source_sha256=SOURCE_SHA256,avalanche='0.4.0',backbone='SlimResNet18 nf20, IncrementalClassifier',dataset_fingerprint=hashes,
        class_order=list(map(int,scenario.classes_order_original_ids)),runtime_flags=flags(),correction=asdict(geometry.engine.config),
        acceptance=('componentwise' if geometry.engine.config.constrained else 'total')+' incoming/replay loss gates within dtype-scaled roundoff; geometry penalty supplements constraints',
        online=True,task_boundaries=False,
        geometry_calibration='external seen-experience validation prefix from task2; calibration uses benchmark boundaries; online updates do not receive task IDs',validation_fraction=cfg['validation_fraction'],memory='upstream ClassBalancedBuffer/ReplayPlugin',
        lpr_configuration_source=f'LPR@{PIN}/README.md: Split-CIFAR100 example omega_0=4 beta=2 every_iter=30',
        replay_origin='native loader for LPR; explicit boolean field for Geometry-VNA, preserved under permutations',
        framework_compatibility='typing4.4 Protocol semantics; prune only empty/None orphan optimizer entries after head adaptation for torch2.2 serialization',
        arm_implementation='official LPR callbacks' if args.arm=='lpr' else 'ER' if args.arm=='er' else 'ER + Geometry-Aware VNA',
        evidence_status='INCONCLUSIVE')
    logger=ExperimentLogger(args.out);out=Path(args.out);geometry.selection_path=out/'geometry_selection.json'
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('online protocol mismatch')
    logger.save('protocol',protocol);progress=Progress(out,protocol)
    if (out/'summary.json').exists():
        previous=json.loads((out/'summary.json').read_text())
        if previous.get('protocol')!=protocol:raise ValueError('online summary protocol mismatch')
        if previous.get('status')=='COMPLETED':
            if len(previous.get('tasks',[]))!=cfg['tasks']:raise ValueError('incomplete online summary')
            return previous
    rows=[];epochs=[];online_rows=[];task=position=0;elapsed=0.;stream_rng=None
    reset_cuda_peak_memory_stats(args.device)
    if progress.state:
        s=progress.state;progress.restore_artifacts();strategy.model=s['model'].to(args.device)
        strategy.optimizer=torch.optim.SGD(strategy.model.parameters(),lr=cfg['lr'],momentum=cfg['momentum'],weight_decay=cfg['weight_decay'])
        strategy.optimizer.load_state_dict(s['optimizer']);vars(strategy.clock).update(s['clock'])
        replay.storage_policy=s['storage'];lpr.storage_policy=replay.storage_policy
        lpr._preconditioner_invs={k:official_module().PreconditionerInv(v.matrix.to(args.device),v.has_bias,v.parallel)
                                  for k,v in s['lpr_inverses'].items()};lpr.update_config.every_iter=s['lpr_every_iter']
        clock=s['clock'];strategy=make_strategy(strategy.model,strategy.optimizer);vars(strategy.clock).update(clock)
        strategy.optimized_param_id=dict(strategy.model.named_parameters())
        geometry.engine.load_state_dict(s['engine']);geometry.last_event=s['last_event'];geometry.selection=s.get('geometry_selection');geometry.selection_attempts=s.get('geometry_selection_attempts',[]);geometry.seen_classes=set(s.get('geometry_seen_classes',[]))
        rows,epochs,task,position,elapsed=s['rows'],s['epochs'],s['task'],s['position'],s['elapsed_seconds']
        stream_rng=s['stream_rng'];restore_rng(s['rng'])
        online_rows=s.get('online_rows',[])
    started=time.monotonic();status='COMPLETED';failure=None;coordinates=None
    def save():
        logger.save('tasks',rows);logger.save('accuracy_matrix',[r['domain_accuracies'] for r in rows]);logger.save('epochs',epochs);logger.save('interventions',geometry.engine.events);logger.save('online',online_rows)
        progress.save(dict(model=strategy.model,optimizer=strategy.optimizer.state_dict(),clock=vars(strategy.clock),storage=replay.storage_policy,
            lpr_inverses=lpr._preconditioner_invs,lpr_every_iter=lpr.update_config.every_iter,engine=geometry.engine.state_dict(),last_event=geometry.last_event,
            rng=rng_state(),rows=rows,epochs=epochs,online_rows=online_rows,geometry_selection=geometry.selection,geometry_selection_attempts=geometry.selection_attempts,geometry_seen_classes=sorted(geometry.seen_classes),task=task,position=position,stream_rng=stream_rng,elapsed_seconds=elapsed+time.monotonic()-started))
    try:
        while task<cfg['tasks']:
            fresh=stream_rng is None
            if fresh:stream_rng=rng_state()
            current_rng=rng_state();restore_rng(stream_rng)
            online=OnlineCLScenario(original_streams=scenario.streams.values(),experiences=scenario.train_stream[task],
                experience_size=cfg['train_mb_size'],access_task_boundaries=False)
            experiences=list(online.train_stream)
            if not fresh:restore_rng(current_rng)
            geometry.task=task
            if cfg.get('geometry_selection') and args.arm=='h2':geometry.validation_stream=scenario.valid_stream[:task+1]
            for index,experience in enumerate(experiences):
                if index<position:continue
                if deadline_due():save();raise SystemExit(75)
                snapshot=State(strategy.model,strategy.optimizer)
                old=dict(storage=deepcopy(replay.storage_policy),clock=deepcopy(vars(strategy.clock)),optimized=strategy.optimized_param_id is not None,
                    param_groups=[list(group['params']) for group in strategy.optimizer.param_groups],
                    inverses=deepcopy(lpr._preconditioner_invs),every=lpr.update_config.every_iter,
                    engine=geometry.engine.state_dict(),last=geometry.last_event,selection=deepcopy(geometry.selection),selection_attempts=deepcopy(geometry.selection_attempts),seen_classes=set(geometry.seen_classes))
                try:
                    strategy.train(experience,eval_streams=[],num_workers=0,drop_last=True)
                    value=finite.last_loss
                except SystemExit as exc:
                    if exc.code!=75:raise
                    for group,parameters in zip(strategy.optimizer.param_groups,old['param_groups']):group['params']=parameters
                    snapshot.restore();replay.storage_policy=old['storage'];lpr.storage_policy=replay.storage_policy
                    strategy.optimized_param_id=dict(strategy.model.named_parameters()) if old['optimized'] else None
                    vars(strategy.clock).clear();vars(strategy.clock).update(old['clock']);lpr._preconditioner_invs=old['inverses']
                    lpr.update_config.every_iter=old['every'];geometry.engine.load_state_dict(old['engine']);geometry.last_event=old['last'];geometry.selection=old['selection'];geometry.selection_attempts=old['selection_attempts'];geometry.seen_classes=old['seen_classes']
                    save();raise
                position=index+1
                online_rows.append(dict(task=task+1,online_batch=position,loss=value,accuracy=finite.last_accuracy,
                    optimizer_steps=strategy.clock.train_iterations))
                if position%100==0:
                    log(f'H2 {args.arm} task={task+1} online_batch={position} loss={value}');save()
            accuracy=evaluate(strategy.model,scenario.test_stream[:task+1],args.device)
            rows.append(dict(task=task+1,domain_accuracies=accuracy,total=float(np.mean(accuracy)),
                new_task_accuracy=accuracy[-1],old_task_accuracy=float(np.mean(accuracy[:-1])) if task else None))
            log(f'H2 {args.arm} task={task+1} test_accuracy={accuracy} mean={np.mean(accuracy):.2f}')
            epochs.append(dict(task=task+1,online_minibatches=position,repeats=cfg['train_epochs']))
            task+=1;position=0;stream_rng=None;save()
    except SystemExit as exc:status='PAUSED' if exc.code==75 else 'INTERRUPTED';raise
    except GeometrySelectionFailed as exc:status='FAILED_PREFLIGHT';failure=str(exc);raise
    except FloatingPointError as exc:
        status='FAILED_DIVERGED';failure=str(exc)
        coordinates=coordinates or dict(task=task+1,online_batch=position,step=strategy.clock.train_iterations+1,tensor='loss')
    except BaseException as exc:status='FAILED_ERROR';failure=repr(exc);raise
    finally:
        logger.save('interventions',geometry.engine.events)
        seconds=elapsed+time.monotonic()-started
        logger.save('summary',dict(protocol=protocol,status=status,failure=failure,diverged=coordinates,tasks=rows,
            **matrix_metrics([r['domain_accuracies'] for r in rows]),wall_seconds=seconds,gpu_hours=seconds/3600 if args.device!='cpu' else 0.,
            correction_seconds=sum(e['seconds'] for e in geometry.engine.events),parameter_count=sum(p.numel() for p in strategy.model.parameters()),
            online_accuracy=float(np.mean([r['accuracy'] for r in online_rows if r['accuracy'] is not None])) if any(r['accuracy'] is not None for r in online_rows) else None,
            geometry_selection=geometry.selection,geometry_selection_attempts=geometry.selection_attempts,online_accuracy_scope='incoming samples before first update of each online experience; no test-set use',
            peak_vram_bytes=torch.cuda.max_memory_allocated(torch.device(args.device)) if args.device!='cpu' else 0,
            representation_drift_mean=float(np.mean([r['replay_feature_drift'] for r in geometry.engine.events if 'replay_feature_drift' in r])) if any('replay_feature_drift' in r for r in geometry.engine.events) else None,
            replay_diagnostics=dict(origin_tracking='explicit fields in Geometry-VNA; native ReplayDataLoader in LPR',
                measured_interventions=sum('replay_samples' in r for r in geometry.engine.events)),
            evidence_status='INCONCLUSIVE'))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/standard_cl/h2_online_cifar100.json');p.add_argument('--out',required=True)
    p.add_argument('--data-dir',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--arm',choices=('lpr','er','h2'),required=True)
    return p


if __name__=='__main__':run(parser().parse_args())
