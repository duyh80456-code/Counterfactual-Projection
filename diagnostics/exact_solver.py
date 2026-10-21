"""Old/new profiling and task-2 A–D replay, with one T4 per subprocess.

Run this file by absolute path; --repo selects either an exported d88abc4 tree or
current source. No diagnostic result is accepted as a main comparison result.
"""
import argparse
from copy import deepcopy
import inspect
import io
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import zipfile


def compare_profiles(reference, current):
    import math
    if any(r.get('status')!='PROFILE_COMPLETED' or not r.get('applications') for r in (reference,current)):
        return dict(status='INCOMPLETE',reference_status=reference.get('status'),current_status=current.get('status'))
    def equal(a,b):
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            return math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-8)
        return a==b
    left,right=reference['applications'][0],current['applications'][0]
    fields=('mu_star','rho_tr','L_before','L_after','KD_before','KD_after')
    checks={name:dict(reference=left.get(name),current=right.get(name),
                      matches=equal(left.get(name),right.get(name))) for name in fields}
    slots=lambda row:[(r['site'],r['unit']) for r in row.get('swap_records',[])]
    checks['swap_slots']=dict(reference=slots(left),current=slots(right),matches=slots(left)==slots(right))
    old_loss,new_loss=reference.get('final_L_KD'),current.get('final_L_KD')
    checks['final_L_KD']=dict(reference=old_loss,current=new_loss,
        matches=old_loss is not None and new_loss is not None and len(old_loss)==len(new_loss) and
                all(equal(a,b) for a,b in zip(old_loss,new_loss)))
    oldr,newr=reference.get('candidate_rankings',[]),current.get('candidate_rankings',[])
    checks['candidate_rankings']=dict(matches=len(oldr)==len(newr) and
        all([p['id'] for p in a['candidates']]==[p['id'] for p in b['candidates']] for a,b in zip(oldr,newr)),
        reference_calls=len(oldr),current_calls=len(newr))
    oldg,newg=left.get('gir_trials',[]),right.get('gir_trials',[])
    checks['ordered_Gamma']=dict(matches=len(oldg)==len(newg) and all(
        (a['site'],a['unit'])==(b['site'],b['unit']) and equal(a['Gamma_woodbury'],b['Gamma_woodbury'])
        for a,b in zip(oldg,newg)),reference_trials=len(oldg),current_trials=len(newg))
    return dict(status='COMPARED',checks=checks,application_seconds=dict(
        reference=reference['application_seconds'],current=current['application_seconds']),
        interpretation='Exact eigenvalues may change ranking and swaps; ordered Gamma is comparable only when candidate rankings match.')


def profile_capacity(model, batch, boundary=lambda: None):
    """Capacity proxy with a 100-class clone, never the live training model."""
    import gc
    import torch
    from diagnostics.vna_align import _global_state
    from methods import vna_v3 as vna
    from adapters.icarl_hooks import sites_for
    rows=[]
    with _global_state():
        clone=deepcopy(model)
        device,dtype=next(clone.parameters()).device,next(clone.parameters()).dtype
        clone.native.update_fc(100)
        clone.to(device=device,dtype=dtype)
        for n in (256,128):
            boundary()
            repeats=(n+len(batch[0])-1)//len(batch[0])
            x=batch[0].repeat((repeats,)+(1,)*(batch[0].ndim-1))[:n]
            y=batch[1].repeat(repeats)[:n]
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            row=dict(probe_size=n,classes=100,m=n*100,gram_bytes=8*(n*100)**2,
                source='task-2 weights, extended 100-class head, repeated window; capacity proxy, not actual task-20 data')
            original_gram=vna.gram
            def measured_gram(*a,**kw):
                gram_start=time.perf_counter()
                try:
                    G,timing=original_gram(*a,**kw)
                    row.update(timing,gram_status='COMPLETED')
                    return G,timing
                finally:
                    row.setdefault('gram_seconds',time.perf_counter()-gram_start)
                    row.setdefault('gram_status','FAILED')
            vna.gram=measured_gram
            start=time.perf_counter()
            try:
                row.update(vna.diagnose(clone,(x,y),sites=sites_for(clone)),status='COMPLETED')
            except torch.cuda.OutOfMemoryError as exc:
                row.update(status='CUDA_OOM',error=str(exc))
            except RuntimeError as exc:
                if not str(exc).startswith('GRAM_LIMIT:'): raise
                row.update(status='GRAM_LIMIT',error=str(exc))
            finally:
                vna.gram=original_gram
                torch.cuda.synchronize()
                row.update(seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
            rows.append(row)
            del x,y
            gc.collect();torch.cuda.empty_cache()
    return dict(rows=rows,actual_task20_measured=False,probe_change_applied=False,
        recommendation='Consider preregistering 128 only after reviewing these proxy measurements and actual task-20 capacity.')


def is_task1_checkpoint(state):
    if not isinstance(state, dict) or not isinstance(state.get('identity'), dict): return False
    expected = dict(arm='icarl_full', seed=0, backbone='resnet32_nobn', class_order_kind='superclass',
                    nobn_lr_scale=.5, code_commit='d88abc496eb1fe21d1b6df66d90b47cf93d7ded1')
    return (all(state['identity'].get(k) == v for k,v in expected.items()) and
            (state.get('completed'), state.get('mid'), state.get('epoch')) == (1, False, 100))


def find_task1_checkpoint(source, target):
    """Select by checkpoint metadata; identical attached copies count as one source."""
    import torch
    from diagnostics.vna_align import _global_state
    source, target = Path(source), Path(target)
    files, visited = [], set()
    if source.is_file(): files = [source]
    else:
        for directory, dirs, names in os.walk(source, followlinks=True):
            real = Path(directory).resolve()
            if real in visited:
                dirs[:] = []
                continue
            visited.add(real)
            files.extend(Path(directory)/name for name in sorted(names))
    matches, rejected, scanned = {}, [], 0
    def inspect_checkpoint(label, data):
        nonlocal scanned
        scanned += 1
        try:
            with _global_state():
                state = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
            if not is_task1_checkpoint(state):
                rejected.append(f'{label}: metadata does not match full/noBN/superclass/s0 task-1 end')
                return
        except Exception as exc:
            rejected.append(f'{label}: {type(exc).__name__}: {exc}')
            return
        digest = hashlib.sha256(data).hexdigest()
        if digest not in matches: matches[digest] = (label, data)
        print(f'Compatible task-1 checkpoint: {label} [sha256 {digest}]', flush=True)
    archive_count = 0
    for path in sorted(files):
        name = path.name.lower()
        try:
            if path.suffix.lower() in ('.pt', '.pth'):
                inspect_checkpoint(str(path), path.read_bytes())
            elif name.endswith(('.tar.gz','.tgz','.tar','.tar.bz2','.tbz2','.tar.xz','.txz')):
                archive_count += 1
                with tarfile.open(path) as archive:
                    for member in archive:
                        if member.isfile() and Path(member.name).suffix.lower() in ('.pt','.pth'):
                            inspect_checkpoint(f'{path}::{member.name}', archive.extractfile(member).read())
            elif name.endswith('.zip'):
                archive_count += 1
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        if not member.is_dir() and Path(member.filename).suffix.lower() in ('.pt','.pth'):
                            inspect_checkpoint(f'{path}::{member.filename}', archive.read(member))
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            rejected.append(f'{path}: {type(exc).__name__}: {exc}')
    print(f'Checkpoint search: {source}; archives={archive_count}, checkpoints={scanned}, unique matches={len(matches)}', flush=True)
    if not matches:
        inventory = '\n'.join(rejected[:20]) or '\n'.join(str(p) for p in sorted(files)[:20]) or '(input is empty or unavailable)'
        raise FileNotFoundError(f'No compatible d88abc4 full/noBN/superclass/s0 checkpoint at completed task 1, epoch 100 in {source}.\n{inventory}')
    if len(matches) > 1:
        raise ValueError('Different compatible task-1 checkpoints found; keep one source or use --checkpoint explicitly:\n'+
                         '\n'.join(label for label, _ in matches.values()))
    label, data = next(iter(matches.values()))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name+'.tmp')
    temporary.write_bytes(data); temporary.replace(target)
    print(f'Selected task-1 checkpoint: {label} -> {target}', flush=True)
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
    checkpoint = Path(args.checkpoint) if args.checkpoint else find_task1_checkpoint(
        args.archive or args.input_root, out/'task1_source.pt')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not is_task1_checkpoint(state):
        raise ValueError('Need d88abc4 full/noBN/superclass/s0 checkpoint at completed task 1, epoch 100')
    native_commit = main.code_commit()
    old = args.branch in ('B','C')
    if old and not native_commit.startswith('d88abc4'):
        raise ValueError('B/C must import d88abc4 source')
    identity = dict(branch=args.branch, code_commit=native_commit, profile=args.profile,
        source_digest=os.environ.get('VNA_SOURCE_DIGEST'), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        capacity_profile=args.capacity_profile)
    if (out/'diagnostic.json').exists():
        prior=json.loads((out/'diagnostic.json').read_text())
        if prior.get('identity') != identity: raise ValueError('diagnostic result identity mismatch')
        if prior.get('status') in ('PROFILE_COMPLETED','COMPLETED_5_EPOCHS','FAILED_DIVERGED'):
            print(prior['status'],out);return
    report = dict(identity=identity, status='RUNNING', diagnostic_only=True, branch=args.branch,
        code_commit=native_commit, source_digest=os.environ.get('VNA_SOURCE_DIGEST'),
        checkpoint=str(checkpoint), device=torch.cuda.get_device_name(0),
        profiling=args.profile, module_paths=module_paths,
        execution=dict(parallel_lanes=int(os.environ.get('VNA_DIAGNOSTIC_LANES','1')),
                       cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')), first_nonfinite=None, threshold=40., lr=.05,
        timers=[], cg=[], jvp_vjp={}, applications=[],candidate_rankings=[])
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
        torch.cuda.synchronize()
        training_seconds=max(time.monotonic()-hook.iteration_started,0.)
        frame=inspect.currentframe().f_back
        while frame is not None and 'loss' not in frame.f_locals: frame=frame.f_back
        loss=float(frame.f_locals['loss'].detach()) if frame else float('nan')
        epoch=int(frame.f_locals.get('epoch',hook.current_epoch))+1 if frame else hook.current_epoch+1
        model=Prediction(learner._network);sites=sites_for(model)
        bad=nonfinite(model,optimizer)
        if not torch.isfinite(torch.tensor(loss)): bad='loss'
        if bad:
            report.update(status='FAILED_DIVERGED',first_nonfinite=dict(task=2,epoch=epoch,step=hook.steps,tensor=bad))
            write();raise Stop()
        torch.cuda.synchronize();sgd_start=time.monotonic()
        result=original_step(optimizer,*a,**kw)
        torch.cuda.synchronize()
        iteration_seconds=max(training_seconds+time.monotonic()-sgd_start,1e-9)
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
            del p, clone, vp, vo, fn, arg, inputs
        engine=intervention.engine
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        event=engine.apply(model,sites_for(model),batch,spec,ctx,optimizer,hook.steps,force=True)
        torch.cuda.synchronize();report['application_seconds']=time.perf_counter()-start
        report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        report['applications'].append(event);intervention.events.append(event)
        report['systems_solved']=len(report['cg']) if old else event.get('systems_solved',len(event.get('solves',[])))
        if args.profile and args.branch=='D' and not event.get('skipped'):
            report['final_L_KD']=list(engine._evaluate_current(model,batch,spec,ctx))
            trials=event.get('gir_trials',[])
            costs=[r.get('gamma_total_seconds',r['gamma_seconds']) for r in trials]
            report['gamma_cost']=dict(trials=len(trials),total_seconds=sum(costs),
                mean_seconds=sum(costs)/len(costs) if costs else None,
                max_seconds=max(costs,default=None),
                interpretation='Includes native slot columns once per slot; candidate Jacobians charged to prepare_gir when cached.')
            report['performance_targets']=dict(application_target_seconds=180,
                application_under_180_seconds=report['application_seconds']<180,
                gram_1280_seconds=event.get('gram_seconds'))
            samples=[json.loads(line)['iteration_seconds'] for line in (out/'sgd.jsonl').read_text().splitlines()]
            epochs=50
            point=dict(task=2,vna_seconds=report['application_seconds'],iteration_seconds=sum(samples)/len(samples),
                       task_iterations=len(learner.train_loader)*epochs)
            report['budget_estimate']=vna.schedule_from_points([point],len(learner.train_loader))
            report['budget_estimate'].update(authoritative=False,baseline_samples=len(samples),
                scope='task 2 only; final-task application cost and task-1 calibration remain unmeasured')
            points=[dict(point,task=t,task_iterations=(100*((2500+learner.train_loader.batch_size-1)//learner.train_loader.batch_size)
                        if t==1 else point['task_iterations'])) for t in range(1,21)]
            projection=vna.schedule_from_points(points,len(learner.train_loader))
            projection.update(authoritative=False,assumption='constant task-2 application and SGD costs for all 20 tasks; not a final-size cost model',
                applications_per_task={t:round(seconds/report['application_seconds']) for t,seconds in projection['projected_vna_seconds'].items()})
            report['budget_estimate_20_task_scenario']=projection
            write()
            if args.capacity_profile:
                report['capacity_profile']=profile_capacity(model,batch,hook.measurement_boundary)
                capacity=next(r for r in report['capacity_profile']['rows'] if r['m']==12800)
                report['performance_targets'].update(gram_12800_seconds=capacity.get('gram_seconds'),
                    gram_12800_limit_seconds=600,gram_12800_ideal_seconds=60,
                    gram_12800_under_600_seconds=capacity.get('gram_status')=='COMPLETED' and
                                                capacity.get('gram_seconds',float('inf'))<600)
        if not args.profile:
            bad=nonfinite(model,optimizer)
            if bad:
                report.update(status='FAILED_DIVERGED',first_nonfinite=dict(task=2,epoch=1,step=hook.steps,tensor=bad));write();raise Stop()
            with vna.training_products(model,sites_for(model),vna.seed_proposals(sites_for(model)),batch[0]) as p:
                op=GGNProducts(p,batch[1],spec,ctx)
                report['lambda_after_application']=vna.power_lambda_max(op.F,op.size,
                    vna.dedicated_generator(0,'nan-ggn',learner._device),iterations=10,device=learner._device)
        write()
        if args.profile:
            report['status']='PROFILE_SKIPPED' if event.get('skipped') else 'PROFILE_COMPLETED'
            raise Stop()
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
                    try:
                        result=original(*a,**kw)
                        if name=='prepare_gir':
                            candidates=[]
                            for proposal in result:
                                h=hashlib.sha256()
                                for key,value in sorted(proposal['A'].items()):
                                    h.update(key.encode())
                                    if value is not None: h.update(value.detach().cpu().contiguous().numpy().tobytes())
                                candidates.append(dict(id=h.hexdigest(),score=proposal['score'],curvature=proposal.get('curvature')))
                            report['candidate_rankings'].append(dict(site=a[2].module_name,candidates=candidates))
                        return result
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
        report['peak_cuda_bytes']=max(report.get('peak_cuda_bytes',0),torch.cuda.max_memory_allocated(),
            max((r['peak_cuda_bytes'] for r in report.get('capacity_profile',{}).get('rows',[])),default=0))
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
    p.add_argument('--archive');p.add_argument('--checkpoint');p.add_argument('--input-root',default='/kaggle/input')
    p.add_argument('--data-dir',required=True)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--capacity-profile',action='store_true')
    return p


if __name__=='__main__':run(parser().parse_args())
