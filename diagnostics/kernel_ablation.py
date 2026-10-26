"""Fixed P1 it2001 kernel chains, native continuation and an explicit s1 replay gate."""
import argparse
from copy import copy, deepcopy
import hashlib
import json
import os
from pathlib import Path
import torch
from torch import nn
from diagnostics.vna_align import _assert_bitwise, _global_state
from experiments.io_utils import atomic_json_save, atomic_torch_save
from experiments.vna_resume import Progress, rng_state, restore_rng, deadline_due, log
from methods import vna_v3 as vna
from methods.vna_ggn import GGNProducts

ARMS=('vna','kernel_self','vna_rand')
REFERENCE='6be654310c2e4123391fd7d8d31716d25c89c6fb'
P1_SETUP=dict(backbone='resnet32_nobn',class_order_kind='superclass',nobn_lr_scale=.5)
P1_CFG=dict(init_cls=5,increment=5,memory_size=2000,fixed_memory=False)
BaseEngine=vna.V3Engine


class KernelEngine(BaseEngine):
    def __init__(self,*a,kernel='vna',**kw):
        super().__init__(*a,**kw)
        if kernel not in ARMS: raise ValueError('unknown projection kernel')
        self.kernel_variant=kernel

    def load_state_dict(self,state):
        if state.get('kernel_variant')!=self.kernel_variant: raise ValueError('kernel resume mismatch')
        super().load_state_dict(state)

    def _select(self,model,sites,batch,values,**kw):
        reference=super()._select(model,sites,batch,values,**kw)
        if self.kernel_variant=='kernel_self' and not reference:
            # Native K remains defined when tiny_V has no positive direction.
            reference={name:{k:None if t is None else t.repeat((16,)+(1,)*(t.ndim-1))
                             for k,t in proposal.items()} for name,proposal in vna.seed_proposals(sites).items()}
        if self.kernel_variant=='vna_rand':
            result=self._randomize(model,sites,batch,reference)
            if any(p['trace_target']>0 and p['trace_before']<=0 for p in result.values()):
                raise ValueError('zero random kernel trace cannot match positive tiny_V trace')
            return result
        return reference

    def _direction(self,p,op,mu,*,G=None,**kw):
        if self.kernel_variant!='kernel_self': return super()._direction(p,op,mu,G=G,**kw)
        native=vna.gram(p)[0] if G is None else G
        previous=p.KE;owned='KE' in vars(p)
        try:
            p.KE=lambda z:vna.solve_image(native,z)
            return super()._direction(p,op,mu,G=native,**kw)
        finally:
            if owned:p.KE=previous
            else:del p.KE


def cpu(value):
    if isinstance(value,nn.Module):
        return dict(type=type(value).__qualname__,weights=cpu(value.state_dict()),
                    requires_grad={n:p.requires_grad for n,p in value.named_parameters()},
                    modes={n:m.training for n,m in value.named_modules()})
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return deepcopy(value)


def training_state(network,opt):
    return dict(network=cpu(network.state_dict()),optimizer=cpu(opt.state_dict()),rng=cpu(rng_state()),
        modes={n:m.training for n,m in network.named_modules()},
        requires_grad={n:p.requires_grad for n,p in network.named_parameters()},
        gradients={n:cpu(p.grad) for n,p in network.named_parameters()})


def restore(network,opt,engine,state):
    network.load_state_dict(state['network']);opt.load_state_dict(state['optimizer'])
    if engine is not None:engine.load_state_dict(state['engine'])
    for n,m in network.named_modules():m.training=state['modes'][n]
    for n,p in network.named_parameters():
        p.requires_grad_(state['requires_grad'][n])
        p.grad=None if state['gradients'][n] is None else state['gradients'][n].to(p)
    restore_rng(state['rng'])


def flags():
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),torch_version=torch.__version__,
        cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')


def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover_inputs(roots,seed,requested=None):
    result=dict(requested or {});runs={};checkpoints={}
    for root in map(Path,roots):
        for path in root.rglob('protocol.json'):
            try:p=read(path)
            except (OSError,ValueError):continue
            if (p.get('seed')==seed and p.get('arm')=='icarl_full' and all(p.get(k)==v for k,v in P1_SETUP.items())
                    and all(p.get('cfg',{}).get(k)==v for k,v in P1_CFG.items()) and p.get('policy_version')==vna.POLICY_VERSION
                    and p.get('grad_clip') is None and not p.get('warmup_epochs',0) and not p.get('gir_live')
                    and p.get('solver')=='gram' and p.get('curvature')=='power10'
                    and p.get('expected_tasks')==20 and not p.get('gir_live_task1') and (path.parent/'interventions.json').exists()
                    and (path.parent/'epochs.json').exists()):
                signature=hashlib.sha256(path.read_bytes()+(path.parent/'interventions.json').read_bytes()+
                    (path.parent/'epochs.json').read_bytes()).hexdigest()
                runs[signature]=str(path.parent)
        for path in root.rglob('*.pt'):
            if 'end_task1' not in path.parts and path.name not in ('latest.pt','latest.previous.pt','task1.pt'):continue
            try:state=load(path)
            except Exception:continue
            p=state.get('identity',{})
            if (p.get('seed')==seed and p.get('arm')=='icarl' and all(p.get(k)==v for k,v in P1_SETUP.items())
                    and all(p.get('cfg',{}).get(k)==v for k,v in P1_CFG.items()) and state.get('completed')==1
                    and state.get('hook',{}).get('steps')==2000 and
                    ((not state.get('mid') and state.get('epoch')==100) or (state.get('mid') and state.get('epoch')==0))):
                checkpoints[digest(path)]=str(path)
    for key,choices in (('original_run',runs),('checkpoint',checkpoints)):
        if result.get(key):continue
        if len(choices)!=1:raise ValueError(f'seed{seed}: set INPUTS[{seed}][{key!r}]; {len(choices)} unique compatible sources: {list(choices.values())}')
        result[key]=next(iter(choices.values()))
    return result


def validate_source(state,protocol,seed,baseline=None):
    p=state['identity']
    expected=dict(seed=seed,**P1_SETUP)
    for source in (p,protocol):
        if any(source.get(k)!=v for k,v in expected.items()):raise ValueError('P1 checkpoint setup mismatch')
        if source.get('gir_live_task1') or source.get('grad_clip') is not None:raise ValueError('amended task-1 source')
        if source.get('warmup_epochs',0)!=0:raise ValueError('warmup source')
        if source.get('fake_data') or source.get('preflight_only'):raise ValueError('synthetic source')
        if any(source['cfg'].get(k)!=v for k,v in P1_CFG.items()):
            raise ValueError('requires P1 Inc5 memory 2000')
        if source.get('training')!=dict(first_epochs=100,later_epochs=50,first_milestones=[30,60,85],later_milestones=[24,35]):
            raise ValueError('training schedule mismatch')
    for key in ('dataset_fingerprint','class_order'):
        if not p.get(key) or p[key]!=protocol.get(key):raise ValueError('source '+key+' mismatch')
    if sorted(p['class_order'])!=list(range(100)):raise ValueError('incomplete class order')
    if not (state['completed']==1 and state['hook']['steps']==2000 and
            ((not state['mid'] and state['epoch']==100) or (state['mid'] and state['epoch']==0))):
        raise ValueError('need native end-task1 or task2 epoch0 checkpoint at step2000')
    if p['arm']!='icarl':
        if baseline is None:raise ValueError('full task-1 source needs matching baseline checkpoint for bitwise proof')
        validate_source(baseline,protocol,seed)
        for key in ('learner','optimizer','scheduler','rng'):
            _assert_bitwise(cpu(state[key]),cpu(baseline[key]),'task1_baseline.'+key)
    if protocol.get('policy_version')!=vna.POLICY_VERSION or protocol.get('gir_live'):
        raise ValueError('requires original 6be6543 dead-slot policy')
    return dict(task1_baseline_bitwise=True,source_commit=p['code_commit'])


def loss_batch(dataset,limit=128):
    groups={}
    for i in range(len(dataset)):
        _,_,y=dataset[i];groups.setdefault(int(y),[]).append(i)
    ids=[]
    for offset in range(max(map(len,groups.values()))):
        for label in sorted(groups):
            if offset<len(groups[label]):ids.append(groups[label][offset])
            if len(ids)==limit:break
        if len(ids)==limit:break
    samples=[dataset[i] for i in ids]
    return (torch.stack([s[1] for s in samples]),torch.tensor([int(s[2]) for s in samples])),ids


def accuracy(learner):
    correct=[];targets=[];net=learner._network
    with _global_state(),torch.no_grad():
        modes={m:m.training for m in net.modules()}
        try:
            net.eval()
            for _,x,y in learner.test_loader:
                pred=net(x.to(learner._device))['logits'].argmax(1).cpu()
                correct.append(pred.eq(y.cpu()));targets.append(y.cpu())
        finally:
            for m,mode in modes.items():m.training=mode
    hits=torch.cat(correct).double();y=torch.cat(targets);old=y<learner._known_classes
    return dict(old=float(100*hits[old].mean()) if old.any() else None,
        new=float(100*hits[~old].mean()) if (~old).any() else None,total=float(100*hits.mean()))


def observe(engine,learner,opt,batch,spec,ctx,heldout,main):
    net=learner._network;model=main.Prediction(net);sites=main.sites_for(model)
    saved=training_state(net,opt);engine_state=engine.state_dict();teacher=cpu(learner._old_network)
    with _global_state():
        L,KD=engine._evaluate_current(model,batch,spec,ctx)
        independent=tuple(t.to(learner._device) for t in heldout)
        hs,hc=main.loss_context(learner,independent)
        hL,hKD=engine._evaluate_current(model,independent,hs,hc)
        with vna.training_products(model,[],{},batch[0]) as products:
            op=GGNProducts(products,batch[1],spec,ctx)
            lam=vna.power_lambda_max(op.F,op.size,vna.dedicated_generator(0,'kernel-ablation-ggn',batch[0].device),
                iterations=10,device=batch[0].device)
        result=dict(L_window=L,KD_window=KD,L_independent=hL,KD_independent=hKD,lambda_max_ggn=lam,
                    accuracy=accuracy(learner))
    _assert_bitwise(saved,training_state(net,opt),'measurement.training')
    _assert_bitwise(cpu(engine_state),cpu(engine.state_dict()),'measurement.engine')
    _assert_bitwise(teacher,cpu(learner._old_network),'measurement.teacher')
    return result


def event_signature(row):
    return {k:row.get(k) for k in ('task','iteration','s','mu','mu_star','skipped','skip_reason','rejected','swaps','swap_records')}


def compare_events(original,replayed):
    differences=[]
    for i in range(max(len(original),len(replayed))):
        a=event_signature(original[i]) if i<len(original) else None
        b=event_signature(replayed[i]) if i<len(replayed) else None
        # Wall-clock fields in swap records are not algorithmic decisions.
        def strip(v):
            if isinstance(v,dict):return {k:strip(x) for k,x in v.items() if 'seconds' not in k and k not in ('contribution_norm',)}
            if isinstance(v,list):return [strip(x) for x in v]
            return v
        if strip(a)!=strip(b):
            left,right=strip(a),strip(b)
            fields=sorted(k for k in set(left or {})|set(right or {}) if (left or {}).get(k)!=(right or {}).get(k))
            differences.append(dict(index=i,fields=fields,original=left,replay=right))
    return differences


def run(args):
    from experiments import main_cil as main
    from experiments.parallel_validity import dataset_fingerprint
    main.cil.upstream_imports()
    root=Path(args.original_run);root=root/'run' if (root/'run'/'protocol.json').exists() else root
    protocol=read(root/'protocol.json');source=load(args.checkpoint)
    proof=validate_source(source,protocol,args.seed,load(args.baseline_checkpoint) if args.baseline_checkpoint else None)
    if protocol['dataset_fingerprint']!=dataset_fingerprint(args.data_dir):raise ValueError('dataset fingerprint mismatch')
    if protocol['class_order']!=main.class_order(args.seed,'superclass',args.data_dir):raise ValueError('actual superclass order mismatch')
    if protocol.get('upstream_commit')!=main.cil.UPSTREAM_PIN:raise ValueError('upstream PyCIL pin mismatch')
    original_epochs=read(root/'epochs.json');original_events=read(root/'interventions.json')
    history={(e['task'],e['epoch']):e for e in original_epochs}
    if sorted(e['epoch'] for e in source['hook']['epochs'] if e['task']==1)!=list(range(1,101)):
        raise ValueError('incomplete task-1 epoch trace in checkpoint')
    for e in source['hook']['epochs']:
        if e['task']!=1:continue
        ref=history.get((1,e['epoch']))
        if ref is None or any(ref.get(k)!=e.get(k) for k in ('train_loss','train_acc','lr')):
            raise ValueError('task-1 checkpoint trace does not match original full run')
    proof['task1_epoch_trace_verified']=True
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    replay_path=out.parent/'s1_replay'/'replay.json'
    if args.seed==1 and args.mode=='ablation':
        if not replay_path.exists() or read(replay_path).get('status') not in ('REPLAY_MATCHED','REPLAY_MISMATCH'):
            raise ValueError('s1 requires its original-run replay audit before ablation')
    identity=dict(experiment='fixed_kernel_it2001',seed=args.seed,kernel=args.arm,mode=args.mode,
        code_commit=main.code_commit(),policy_reference=REFERENCE,checkpoint_sha256=digest(args.checkpoint),
        original_protocol=protocol,proof=proof,max_tasks=5,chain_steps=8,
        reused_prefix_seconds=source.get('elapsed_seconds'),
        replay_audit_sha256=digest(replay_path) if args.seed==1 and args.mode=='ablation' else None,
        measurement=dict(rank=16,L='CE+KD',KD='temperature2 old-class distillation',
            lambda_max='native GGN power10 on intervention window',independent_loss='balanced official-test batch',
            trust_region='original Gram quadratic; actual norm is measured only'))
    manifest=out/'protocol.json'
    if manifest.exists() and read(manifest)!=identity:raise ValueError('ablation identity mismatch')
    atomic_json_save(identity,manifest)
    if (out/'summary.json').exists() and read(out/'summary.json').get('status') in ('COMPLETED','FAILED_DIVERGED'):
        return
    if args.mode=='replay' and (out/'replay.json').exists() and read(out/'replay.json')['status'] in ('REPLAY_MATCHED','REPLAY_MISMATCH'):
        return
    full_schedule=deepcopy(protocol.get('schedule'))
    if not full_schedule:raise ValueError('original full-arm K0 authority schedule is required')
    full_schedule.update(calibration_code_commit=protocol['code_commit'],code_commit=main.code_commit(),source_digest=None)
    atomic_json_save(full_schedule,out/'schedule.json')
    cli=main.parser().parse_args(['--arm','icarl_full','--seed',str(args.seed),'--backbone','resnet32_nobn',
        '--class-order','superclass','--nobn-lr-scale','.5','--coefficient',str(protocol.get('coefficient',1.)),
        '--solver',protocol['solver'],'--curvature',protocol['curvature'],
        '--max-tasks','5','--schedule',str(out/'schedule.json'),'--data-dir',args.data_dir,'--device',args.device,
        '--out',str(out/'run')])
    if protocol['solver']!='gram' or protocol['curvature']!='power10':raise ValueError('requires original Gram/power10 run')
    class SelectedEngine(KernelEngine):
        def __init__(self,*a,**kw):super().__init__(*a,kernel=args.arm,**kw)
    selected=BaseEngine if args.mode=='replay' else SelectedEngine
    vna.V3Engine=selected
    class BorrowProgress(Progress):
        def __init__(self,directory,current,name='latest.pt'):
            super().__init__(directory,current,name)
            if self.state is not None and self.state.get('ablation_identity')!=identity:
                raise ValueError('native continuation ablation identity mismatch')
            if self.state is None:
                self.state=deepcopy(source);self.state['identity']=deepcopy(current)
                self.state['elapsed_seconds']=0.
                engine=selected('vna_gir',seed=args.seed,K0=full_schedule['K0'],solver='gram',curvature='power10')
                engine.configure_schedule(full_schedule)
                adapter=main.Intervention(engine,full_schedule['K0'])
                if self.state['mid']:
                    adapter.task=2;engine.begin_task(2)
                self.state['hook']['intervention_state']=adapter.state_dict()
                self.state['artifacts']={}
        def save(self,state):super().save(dict(state,ablation_identity=identity))
    main.Progress=BorrowProgress
    original_after=main.Intervention.after_step
    original_epoch=main.MainHook.after_epoch
    original_record=main.MainHook.record_task
    reference_events=original_events
    reference_epochs=original_epochs
    audit=read(out/'replay.json') if (out/'replay.json').exists() else dict(status='RUNNING',differences=[])
    rows=[]
    class ReplayDone(Exception):pass
    def after_step(adapter,hook,learner,opt):
        if args.mode=='replay':
            original_after(adapter,hook,learner,opt)
            return
        if hook.steps!=2001:
            if hook.steps<2001:raise ValueError('unexpected step before it2001')
            return original_after(adapter,hook,learner,opt)
        if adapter.task!=2 or hook.current_epoch!=0:raise ValueError('anchor is not task2 it2001 epoch1')
        device=learner._device;batch=tuple(t.to(device) for t in hook.current_batch)
        spec0,ctx0=main.loss_context(learner,batch)
        expected=next((e for e in original_events if e.get('task')==2 and e.get('iteration')==2001),None)
        losses=adapter.engine._evaluate_current(main.Prediction(learner._network),batch,spec0,ctx0)
        if expected is None or losses!=(expected.get('L_before'),expected.get('KD_before')):
            raise ValueError('reconstructed it2001 window losses differ from original; exact original checkpoint/window is required')
        anchor=dict(training=training_state(learner._network,opt),teacher=cpu(learner._old_network),
            memory=cpu({k:v for k,v in vars(learner).items() if 'memory' in k}),window=cpu(batch),
            scheduler=cpu(hook.scheduler.state_dict()),engine=cpu(adapter.engine.state_dict()),flags=flags())
        # Kernel metadata differs across arms; the underlying pre-intervention engine must not.
        anchor['engine'].pop('kernel_variant',None)
        common=Path(args.anchor_dir)/f's{args.seed}.pt';common.parent.mkdir(parents=True,exist_ok=True)
        if common.exists():_assert_bitwise(load(common),anchor,'common_anchor')
        else:atomic_torch_save(anchor,common)
        native=training_state(learner._network,opt)
        native_learner,native_opt=learner,opt
        learner=copy(learner);learner._network=deepcopy(native_learner._network)
        opt=torch.optim.SGD(learner._network.parameters(),lr=native_opt.param_groups[0]['lr'])
        restore(learner._network,opt,None,native)
        chain_engine=deepcopy(adapter.engine);chain_engine.arm='vna'
        spec,ctx=main.loss_context(learner,batch);ctx.update(task=2,classes=learner._total_classes)
        with _global_state():heldout,heldout_ids=loss_batch(learner.test_loader.dataset)
        atomic_json_save(dict(source='official_test_disjoint_from_train_window',indices=heldout_ids),out/'independent_batch.json')
        cache_path=out/'chain_state.pt'
        first=None;start=1
        if cache_path.exists():
            cache=load(cache_path)
            if cache['identity']!=identity:raise ValueError('chain resume identity mismatch')
            rows[:]=cache['rows'];first=cache['first'];restore(learner._network,opt,chain_engine,cache['last']);start=cache['k']+1
        else:rows.append(dict(k=0,**observe(chain_engine,learner,opt,batch,spec,ctx,heldout,main)))
        model=main.Prediction(learner._network);sites=main.sites_for(model)
        for k in range(start,9):
            if deadline_due():raise SystemExit(75)
            log(f'kernel={args.arm} seed={args.seed} k={k}/8 START')
            before=[p.detach().double().clone() for p in learner._network.parameters()]
            event=chain_engine.apply(model,sites,batch,spec,ctx,opt,2001,force=True)
            norm=float(sum((p.detach().double()-q).square().sum() for p,q in zip(learner._network.parameters(),before)).sqrt())
            row=dict(k=k,s=event.get('s'),step_norm=norm,Delta=event.get('Delta'),V_O=event.get('V_O'),
                trust_region_exceeds=norm>event['Delta'] if event.get('Delta') is not None else None,
                event=event,**observe(chain_engine,learner,opt,batch,spec,ctx,heldout,main))
            rows.append(row)
            last=dict(training_state(learner._network,opt),engine=cpu(chain_engine.state_dict()))
            if k==1:first=deepcopy(last)
            atomic_torch_save(dict(identity=identity,k=k,rows=rows,first=first,last=last),cache_path)
            atomic_json_save(rows,out/'chain.json');log(f'kernel={args.arm} k={k} END s={row["s"]} norm={norm}')
        restore(native_learner._network,native_opt,adapter.engine,first)
        adapter.engine.arm='vna_gir'
        adapter.events.append(rows[1]['event']);adapter.last_iteration=2001
        adapter.interval=full_schedule['K0']*(2 if rows[1]['event'].get('budget_guard') else 1)
        if not rows[1]['event'].get('skipped') and rows[1]['event'].get('s',0)>0:
            adapter.followups.append(dict(task=2,iteration=2001,status='RUNNING',samples=[]))
        # The eight copied attempts must not consume RNG from the continuing native loop.
        restore_rng(native['rng'])
        audit.update(status='ANCHOR_VERIFIED',flags=flags(),anchor_sha256=digest(common))
        atomic_json_save(audit,out/'anchor_audit.json')
    def epoch(hook,learner,opt,frame):
        original_epoch(hook,learner,opt,frame)
        if args.mode=='replay' and learner._cur_task==1 and hook.current_epoch==43:
            got=[e for e in hook.intervention.events if e.get('task')==2]
            end=hook.steps
            want=[e for e in reference_events if e.get('task')==2 and e.get('iteration',0)<=end]
            differences=compare_events(want,got)
            first_intervention=differences[0] if differences else None
            lookup={(e['task'],e['epoch']):e for e in reference_epochs}
            for e in hook.epochs:
                if e['task']!=2:continue
                ref=lookup.get((2,e['epoch']))
                if ref is None or any(e.get(k)!=ref.get(k) for k in ('train_loss','lr')):
                    differences.append(dict(epoch=e['epoch'],reason='epoch_loss_or_lr_mismatch',replay=e,original=ref))
            audit.update(status='REPLAY_MATCHED' if not differences else 'REPLAY_MISMATCH',differences=differences,
                first_intervention_mismatch=first_intervention,
                first_epoch_mismatch=next((d for d in differences if d.get('reason')=='epoch_loss_or_lr_mismatch'),None),
                flags=flags(),original_flags=protocol.get('determinism','unrecorded'),epoch=43,
                conclusion_allowed=not differences and protocol.get('determinism')==flags())
            atomic_json_save(audit,out/'replay.json');raise ReplayDone()
    def record(hook,learner,result):
        original_record(hook,learner,result)
        atomic_json_save(hook.tasks,out/'tasks.json')
    main.Intervention.after_step=after_step;main.MainHook.after_epoch=epoch;main.MainHook.record_task=record
    try:
        result=main.run(cli)
        if args.mode=='replay':raise RuntimeError('replay did not reach task2 epoch43')
        result['interpretation_allowed']=True if args.seed==0 else read(replay_path).get('conclusion_allowed',False)
        atomic_json_save(result,out/'summary.json')
    except ReplayDone:pass
    finally:
        vna.V3Engine=BaseEngine;main.Progress=Progress
        main.Intervention.after_step=original_after;main.MainHook.after_epoch=original_epoch;main.MainHook.record_task=original_record


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=(0,1),required=True)
    p.add_argument('--arm',choices=ARMS,default='vna')
    p.add_argument('--mode',choices=('ablation','replay'),default='ablation')
    p.add_argument('--checkpoint',required=True);p.add_argument('--baseline-checkpoint')
    p.add_argument('--original-run',required=True);p.add_argument('--anchor-dir',required=True)
    p.add_argument('--data-dir',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    return p


if __name__=='__main__':run(parser().parse_args())
