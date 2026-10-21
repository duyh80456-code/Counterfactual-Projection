"""One-T4 old/new profiling and task-2 A–D replay from the archived task-1 checkpoint.

Run this file by absolute path; --repo selects either an exported d88abc4 tree or
current source. No diagnostic result is accepted as a main comparison result.
"""
import argparse
from copy import deepcopy
import inspect
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time


def checkpoint_from_archive(archive, target):
    with tarfile.open(archive) as source:
        matches = [m for m in source if m.isfile() and
                   m.name.endswith('/icarl_full__nobn__superclass__s0/latest.pt')]
        if len(matches) != 1: raise ValueError('Need exactly one archived full-arm latest.pt')
        target.write_bytes(source.extractfile(matches[0]).read())
    return target


def verify_repository(repo, modules):
    root = Path(repo).resolve()
    paths = {module.__name__: str(Path(module.__file__).resolve()) for module in modules}
    for name, path in paths.items():
        if not Path(path).is_relative_to(root):
            raise RuntimeError(f'Diagnostic import mismatch: {name} loaded from {path}; expected source inside {root}')
    return paths


def run(args):
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import torch
    from experiments import main_cil as main
    verify_repository(args.repo, (main,))
    main.cil.upstream_imports()
    from adapters.icarl_hooks import Prediction, sites_for, loss_context
    from methods import vna_v3 as vna
    from methods.vna_ggn import GGNProducts
    from experiments.vna_resume import Progress
    from experiments.io_utils import atomic_json_save
    module_paths = verify_repository(args.repo, (main, vna, inspect.getmodule(Prediction),
        inspect.getmodule(GGNProducts), inspect.getmodule(Progress)))
    if args.branch in ('B','C') and not callable(getattr(vna, 'conjugate_gradient', None)):
        raise RuntimeError('Historical d88abc4 solver is missing; rebuild the reference checkout before profiling B/C')
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This diagnostic requires one real T4; no substitute timing')
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else checkpoint_from_archive(
        args.archive, out/'task1_source.pt')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    source_identity = state.get('identity',{})
    expected = dict(arm='icarl_full', seed=0, backbone='resnet32_nobn', class_order_kind='superclass', nobn_lr_scale=.5)
    if any(source_identity.get(k) != v for k,v in expected.items()) or not source_identity.get('code_commit','').startswith('d88abc4'):
        raise ValueError('Need the registered d88abc4 full/noBN/superclass/seed-0 checkpoint')
    if (state.get('completed'), state.get('mid'), state.get('epoch')) != (1, False, 100):
        raise ValueError('Expected completed task 1, epoch 100 checkpoint')
    native_commit = main.code_commit()
    old = args.branch in ('B','C')
    if old and not native_commit.startswith('d88abc4'):
        raise ValueError('B/C must import d88abc4 source')
    identity = dict(branch=args.branch, code_commit=native_commit, profile=args.profile,
        source_digest=os.environ.get('VNA_SOURCE_DIGEST'), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    if (out/'diagnostic.json').exists():
        prior=json.loads((out/'diagnostic.json').read_text())
        if prior.get('identity') != identity: raise ValueError('diagnostic result identity mismatch')
        if prior.get('status') in ('PROFILE_COMPLETED','COMPLETED_5_EPOCHS','FAILED_DIVERGED'):
            print(prior['status'],out);return
    report = dict(identity=identity, status='RUNNING', diagnostic_only=True, branch=args.branch,
        code_commit=native_commit, source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
        checkpoint=str(checkpoint), device=torch.cuda.get_device_name(0),
        profiling=args.profile, module_paths=module_paths, first_nonfinite=None, threshold=40., lr=.05,
        timers=[], cg=[], jvp_vjp={}, applications=[])
    atomic_json_save(report,out/'diagnostic.json')
    active = dict(hook=None, model=None, applications=0, stage=[], last_lambda_step=None)
    class Stop(Exception): pass
    class ReplayProgress(Progress):
        def __init__(self, directory, protocol, name='latest.pt'):
            self.out,self.identity,self.path = Path(directory),deepcopy(protocol),Path(directory)/name
            self.state = torch.load(self.path,map_location='cpu',weights_only=False) if self.path.exists() else deepcopy(state)
            if self.path.exists() and (self.state['identity'] != self.identity or self.state.get('diagnostic_identity') != identity): raise ValueError('diagnostic resume identity mismatch')
            if not self.path.exists():
                self.state['identity'] = deepcopy(self.identity)
                hookstate = self.state['hook']
                hookstate['rows'], hookstate['tasks'] = [], []
                if args.branch == 'A': hookstate.pop('intervention_state',None)
                if args.branch == 'D':
                    engine=vna.V3Engine('vna_gir',seed=0,K0=1)
                    oldengine = hookstate.get('intervention_state',{}).get('engine',{})
                    engine.mu = oldengine.get('mu')
                    engine.install_ages = deepcopy(oldengine.get('install_ages',{}))
                    engine.protocol['mu0'] = oldengine.get('protocol',{}).get('mu0')
                    hookstate['intervention_state']=main.Intervention(engine,1).state_dict()
            history=out/'sgd.jsonl'
            if history.exists():
                rows=[json.loads(line) for line in history.read_text().splitlines() if line.strip()]
                rows=[r for r in rows if r['step'] <= self.state['hook']['steps']]
                history.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                active['last_lambda_step']=next((r['step'] for r in reversed(rows) if 'lambda_max_ggn' in r),None)
        def save(self, payload):
            super().save({**payload, 'diagnostic_identity': identity})
        def restore_artifacts(self): pass
    class ReplayHook(main.MainHook):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);active['hook']=self
        def record_task(self,*a): pass
        def after_epoch(self,learner,optimizer,frame):
            super().after_epoch(learner,optimizer,frame)
            if self.current_epoch >= 5:
                report['status']='COMPLETED_5_EPOCHS';raise Stop()
    def write(): atomic_json_save(report,out/'diagnostic.json')
    def nonfinite(model,optimizer):
        for name,p in model.named_parameters():
            if not bool(torch.isfinite(p).all()): return 'parameter:'+name
            if p.grad is not None and not bool(torch.isfinite(p.grad).all()): return 'gradient:'+name
        for p,values in optimizer.state.items():
            for key,value in values.items():
                if isinstance(value,torch.Tensor) and not bool(torch.isfinite(value).all()): return 'optimizer:'+key
        return None
    original_step=torch.optim.SGD.step
    def step(optimizer,*a,**kw):
        hook=active['hook'];learner=hook.learner
        if learner._cur_task != 1: return original_step(optimizer,*a,**kw)
        frame=inspect.currentframe().f_back
        while frame is not None and 'loss' not in frame.f_locals: frame=frame.f_back
        loss=float(frame.f_locals['loss']) if frame else float('nan')
        epoch=int(frame.f_locals.get('epoch',hook.current_epoch))+1 if frame else hook.current_epoch+1
        model=Prediction(learner._network);sites=sites_for(model)
        bad=nonfinite(model,optimizer)
        if not torch.isfinite(torch.tensor(loss)): bad='loss'
        if bad:
            report.update(status='FAILED_DIVERGED',first_nonfinite=dict(task=2,epoch=epoch,step=hook.steps,tensor=bad))
            write();raise Stop()
        result=original_step(optimizer,*a,**kw)
        torch.cuda.synchronize()
        iteration_seconds=max(time.monotonic()-hook.iteration_started,1e-9)
        norms={s.module_name:sum(float(dict(model.named_parameters())[n].detach().double().square().sum()) for n in s.parameter_names)**.5 for s in sites}
        row=dict(task=2,epoch=epoch,step=hook.steps,loss=loss,
            iteration_seconds=iteration_seconds,
            grad_norm=sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None)**.5,
            param_max_abs=max(float(p.detach().abs().max()) for p in model.parameters()),site_norms=norms)
        engine=hook.intervention.engine if hook.intervention else None
        row['new_outgoing_norms']={f'{name}:{unit}':float(s.second_layer.layer.weight[:,unit].detach().double().norm())
            for (name,unit) in (engine.install_ages if engine else {}) for s in sites if s.module_name==name}
        if active['last_lambda_step'] is None or hook.steps-active['last_lambda_step']>=10:
            batch=tuple(t.to(learner._device) for t in hook.current_batch)
            spec,ctx=loss_context(learner,batch)
            with vna.training_products(model,sites,vna.seed_proposals(sites),batch[0]) as p:
                op=GGNProducts(p,batch[1],spec,ctx)
                row['lambda_max_ggn']=vna.power_lambda_max(op.F,op.size,
                    vna.dedicated_generator(0,'nan-ggn',learner._device),iterations=10,device=learner._device)
            row['above_2_over_lr']=row['lambda_max_ggn']>40.
            active['last_lambda_step']=hook.steps
        with (out/'sgd.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        bad=nonfinite(model,optimizer)
        if bad:
            report.update(status='FAILED_DIVERGED',first_nonfinite=dict(task=2,epoch=epoch,step=hook.steps,tensor=bad));write();raise Stop()
        return result
    torch.optim.SGD.step=step
    def after(intervention,hook,learner,optimizer):
        if active['applications'] or any(e.get('task') == 2 for e in intervention.events): return
        active['applications']+=1
        model=Prediction(learner._network)
        batch=tuple(t.to(learner._device) for t in hook.current_batch)
        spec,ctx=loss_context(learner,batch)
        active['model']=model
        if args.profile:
            for dtype in (torch.float32,torch.float64):
                clone=deepcopy(model).to(dtype=dtype); inputs=batch[0].to(dtype)
                with vna.training_products(clone,sites_for(clone),vna.seed_proposals(sites_for(clone)),inputs) as p:
                    params=sum(x.numel() for _,x in p.parameter_refs)
                    vp=torch.ones(params,device=inputs.device,dtype=dtype)
                    vo=torch.ones(p._call({}).numel(),device=inputs.device,dtype=dtype)
                    timings={}
                    for name,fn,arg in (('jvp',p.Jv,vp),('vjp',p.JT,vo)):
                        torch.cuda.synchronize();start=time.perf_counter();fn(arg);torch.cuda.synchronize()
                        timings[name+'_seconds']=time.perf_counter()-start
                    report['jvp_vjp'][str(dtype)]=dict(**timings,parameter_dtype=str(p.parameter_refs[0][1].dtype),vector_dtype=str(vp.dtype))
            del p, clone, vp, vo
        engine=intervention.engine
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        event=engine.apply(model,sites_for(model),batch,spec,ctx,optimizer,hook.steps,force=True)
        torch.cuda.synchronize();report['application_seconds']=time.perf_counter()-start
        report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        report['applications'].append(event);intervention.events.append(event)
        report['systems_solved']=len(report['cg']) if old else event.get('systems_solved',len(event.get('solves',[])))
        if not args.profile:
            bad=nonfinite(model,optimizer)
            if bad:
                report.update(status='FAILED_DIVERGED',first_nonfinite=dict(task=2,epoch=1,step=hook.steps,tensor=bad));write();raise Stop()
            with vna.training_products(model,sites_for(model),vna.seed_proposals(sites_for(model)),batch[0]) as p:
                op=GGNProducts(p,batch[1],spec,ctx)
                report['lambda_after_application']=vna.power_lambda_max(op.F,op.size,
                    vna.dedicated_generator(0,'nan-ggn',learner._device),iterations=10,device=learner._device)
        write()
        if args.profile: report['status']='PROFILE_COMPLETED';raise Stop()
    main.Intervention.after_step=after
    if args.branch == 'C':
        original_install=vna.V3Engine.install_gir
        def install(engine,model,site,proposals,units,optimizer):
            saved=deepcopy(optimizer.state_dict())
            try:return original_install(engine,model,site,proposals,units,optimizer)
            finally:optimizer.load_state_dict(saved)
        vna.V3Engine.install_gir=install
    if args.profile:
        for method in ('_direction','_select','_slot_jump','_slot_contribution','prepare_gir','_run_gir'):
            original=getattr(vna.V3Engine,method)
            def wrap(original=original,name=method):
                def timed(*a,**kw):
                    torch.cuda.synchronize();start=time.perf_counter();active['stage'].append(name)
                    try:return original(*a,**kw)
                    finally:
                        torch.cuda.synchronize();report['timers'].append(dict(name=name,seconds=time.perf_counter()-start));active['stage'].pop()
                return timed
            setattr(vna.V3Engine,method,wrap())
        if old:
            cg=vna.conjugate_gradient
            def measured(action,rhs,*a,**kw):
                torch.cuda.synchronize();start=time.perf_counter();result=cg(action,rhs,*a,**kw);torch.cuda.synchronize()
                caller=inspect.currentframe().f_back
                report['cg'].append(dict(name=caller.f_locals.get('name',caller.f_code.co_name),
                    beta=caller.f_locals.get('beta'),stage=list(active['stage']),iterations=result.iterations,
                    relative_residual=result.relative_residual,seconds=time.perf_counter()-start,
                    vector_dtype=str(rhs.dtype),parameter_dtype=str(next(active['model'].parameters()).dtype)))
                return result
            vna.conjugate_gradient=measured
    main.Progress,main.MainHook=ReplayProgress,ReplayHook
    # Record only the requested five task-2 epochs, retaining the registered training recipe.
    main.measure_task=lambda *a,**kw: ({}, {})
    schedule=dict(authoritative=True,code_commit=main.code_commit(),source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
                  backbone='resnet32_nobn',nobn_lr_scale=.5,synthetic=False,K0=1)
    (out/'schedule.json').write_text(json.dumps(schedule))
    runargs=main.parser().parse_args(['--arm','icarl' if args.branch=='A' else 'icarl_full',
        '--nobn-lr-scale','.5','--seed','0','--device','cuda:0','--data-dir',args.data_dir,
        '--out',str(out/'run'),'--schedule',str(out/'schedule.json')])
    try:
        outcome=main.run(runargs)
        if outcome.get('status') in ('FAILED_DIVERGED','DIVERGED'):
            report.update(status='FAILED_DIVERGED',first_nonfinite=outcome.get('diverged'))
    except Stop:pass
    except SystemExit as exc:
        if exc.code==75:report['status']='PAUSED';write()
        raise
    except Exception as exc:
        report.update(status='FAILED_DIAGNOSTIC',error=repr(exc));write();raise
    finally:
        report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        history=out/'sgd.jsonl'
        if history.exists():
            rows=[json.loads(line) for line in history.read_text().splitlines() if line.strip()]
            lambdas=[r['lambda_max_ggn'] for r in rows if 'lambda_max_ggn' in r]
            if 'lambda_after_application' in report:lambdas.append(report['lambda_after_application'])
            report['max_lambda_max_ggn']=max(lambdas,default=None)
            report['above_2_over_lr_before_failure']=any(v>40. for v in lambdas)
            report['sgd_steps_recorded']=len(rows)
        if args.profile:
            name='profile_d88abc4.json' if old else 'profile_exact.json'
            atomic_json_save(report,out/name)
        write()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    p.add_argument('--branch',choices=list('ABCD'),required=True)
    p.add_argument('--archive');p.add_argument('--checkpoint');p.add_argument('--data-dir',required=True)
    p.add_argument('--profile',action='store_true')
    return p


if __name__=='__main__':run(parser().parse_args())
