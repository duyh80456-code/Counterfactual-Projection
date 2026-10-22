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


def compare_profiles(reference, current, strict=False, allow_curvature_change=False):
    import math
    if any(r.get('status')!='PROFILE_COMPLETED' or not r.get('applications') for r in (reference,current)):
        return dict(status='INCOMPLETE',reference_status=reference.get('status'),current_status=current.get('status'))
    def equal(a,b):
        if isinstance(a,int) and isinstance(b,int): return a==b
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-5,abs_tol=0.)
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
        all(a.get('site')==b.get('site') and [p['id'] for p in a['candidates']]==[p['id'] for p in b['candidates']] for a,b in zip(oldr,newr)),
        reference_calls=len(oldr),current_calls=len(newr))
    checks['candidate_values']=dict(matches=len(oldr)==len(newr) and all(
        len(a['candidates'])==len(b['candidates']) and all(
            x['id']==y['id'] and all(equal(x.get(k),y.get(k)) for k in (('score','curvature') if allow_curvature_change else ('score','curvature','power_generator_seed','power_iterations')))
            for x,y in zip(a['candidates'],b['candidates'])) for a,b in zip(oldr,newr)))
    oldg,newg=left.get('gir_trials',[]),right.get('gir_trials',[])
    checks['ordered_Gamma']=dict(matches=len(oldg)==len(newg) and all(
        (a['site'],a['unit'])==(b['site'],b['unit']) and equal(a.get('Gamma_cg',a.get('Gamma_woodbury')),b.get('Gamma_cg',b.get('Gamma_woodbury')))
        for a,b in zip(oldg,newg)),reference_trials=len(oldg),current_trials=len(newg))
    old_swaps,new_swaps=reference.get('applied_swaps',[]),current.get('applied_swaps',[])
    def same_values(a,b):
        if isinstance(a,list) and isinstance(b,list): return len(a)==len(b) and all(same_values(x,y) for x,y in zip(a,b))
        return equal(a,b)
    checks['installed_candidates']=dict(matches=len(old_swaps)==len(new_swaps) and all(
        (a['site'],a['unit'],a['candidate_id'])==(b['site'],b['unit'],b['candidate_id']) and
        a['A'].keys()==b['A'].keys() and all(same_values(a['A'][k],b['A'][k]) for k in a['A'])
        for a,b in zip(old_swaps,new_swaps)))
    complete=bool(oldr and newr) and all(
        isinstance(left.get(k),(int,float)) and isinstance(right.get(k),(int,float)) for k in fields)
    complete=complete and bool(oldg and newg) and len(old_swaps)==len(left.get('swap_records',[])) and len(new_swaps)==len(right.get('swap_records',[]))
    complete=complete and all((p.get('curvature_method')=='eigvalsh_downstream' or
        isinstance(p.get('power_generator_seed'),int) and p.get('power_iterations')==10) and all(isinstance(p.get(k),(int,float)) and math.isfinite(p[k]) for k in ('score','curvature'))
        for calls in (oldr,newr) for call in calls for p in call['candidates'])
    checks['data_complete']=dict(matches=complete)
    if strict:
        policies=[r.get('identity',{}).get('policy_version') for r in (reference,current)]
        if any(policies):
            checks['same_policy']=dict(matches=policies[0]==policies[1] and
                reference.get('identity',{}).get('gir_live')==current.get('identity',{}).get('gir_live'))
        sha=reference.get('identity',{}).get('checkpoint_sha256')
        checks['same_checkpoint']=dict(matches=bool(sha) and sha==current.get('identity',{}).get('checkpoint_sha256'))
        if not allow_curvature_change:
            checks['same_curvature_mode']=dict(matches=reference.get('curvature')==current.get('curvature') and
                reference.get('curvature') in ('power10','exact'))
    ref_rhos=dict(zip(left.get('mu_trajectory',[]),left.get('rho_trajectory',[])))
    numeric=[dict(mu=t['mu'],reference_rho=ref_rhos.get(t['mu']),failure_reason=t.get('failure_reason'))
             for t in right.get('lm_trials',[]) if t.get('outcome')=='numeric_failure']
    lm_audit=dict(numeric_failures=numeric,
        positive_reference_rejections=[t for t in numeric if t['reference_rho'] is not None and t['reference_rho']>0],
        unknown_reference_trials=[t['mu'] for t in numeric if t['reference_rho'] is None],
        scope='Only this checkpoint/application; reference Gram arithmetic is not an exact float64 solve.')
    ranking_changes = None
    if reference.get('curvature')=='power10' and current.get('curvature')=='exact':
        calls=[]
        for a,b in zip(oldr,newr):
            if (a.get('site'),a.get('used_directions'))!=(b.get('site'),b.get('used_directions')): continue
            old={p['id']:p for p in a['candidates']}
            candidates=[dict(p,curvature_power10=old[p['id']]['curvature'],score_power10=old[p['id']]['score'])
                        for p in b['candidates'] if p['id'] in old]
            calls.append(dict(b,candidates=candidates))
        ranking_changes=curvature_differences(dict(candidate_rankings=calls))
        ranking_changes['context']='Matched candidate IDs on independent trajectories; residuals/directions may differ after the first selection. Not a same-state curvature audit.'
    passed=complete and all(check['matches'] for check in checks.values())
    records=lambda event:[{k:r.get(k) for k in ('site','unit','Gamma','ell_beta','bias_q')} for r in event.get('swap_records',[])]
    trials=lambda event:[dict(site=r.get('site'),unit=r.get('unit'),Gamma=r.get('Gamma_cg',r.get('Gamma_woodbury'))) for r in event.get('gir_trials',[])]
    return dict(status=('PASS' if passed else 'FAIL') if strict else 'COMPARED',checks=checks,relative_tolerance=1e-5,
        lm_numeric_audit=lm_audit,ranking_changes=ranking_changes,
        curvature_modes=dict(reference=reference.get('curvature'),current=current.get('curvature'),
            intentional_change=allow_curvature_change),
        swaps=dict(reference=records(left),current=records(right)),
        installed_swaps=dict(reference=old_swaps,current=new_swaps),
        Gamma_trials=dict(reference=trials(left),current=trials(right)),
        application_seconds=dict(
        reference=reference['application_seconds'],current=current['application_seconds']),
        interpretation=('Strict numeric comparison on the same checkpoint with an intentional curvature change; ranking/swaps may differ.'
            if allow_curvature_change else
            'Compare the same checkpoint and curvature mode; failures remain explicit, including near-zero values at abs tol zero.'))


def record_profile_loss(report,step,loss):
    if report.get('status')!='PROFILE_FOLLOWUP':return
    samples=report.setdefault('post_application_sgd',[])
    origin=report['application_step']
    if step<=origin or samples and step<=samples[-1]['step']:return
    samples.append(dict(step=step,relative_step=step-origin,loss=loss))
    if len(samples)>=20:report['status']='PROFILE_COMPLETED'


def curvature_differences(report):
    import math
    from scipy.stats import kendalltau
    result=[]
    for call in report.get('candidate_rankings',[]):
        candidates=call['candidates']
        valid=[p for p in candidates if all(isinstance(p.get(k),(int,float)) and math.isfinite(p[k])
               for k in ('curvature','curvature_power10','score','score_power10'))]
        exact=sorted(valid,key=lambda p:p['score'],reverse=True)
        power=sorted(valid,key=lambda p:p['score_power10'],reverse=True)
        tau=float(kendalltau([p['score'] for p in valid],[p['score_power10'] for p in valid],variant='b').statistic) if len(valid)>1 else float('nan')
        result.append(dict(site=call['site'],iteration=call.get('iteration'),used_directions=call.get('used_directions'),
            mu=call.get('mu'),candidate_count=len(candidates),valid_count=len(valid),
            kendall_tau_b=tau if math.isfinite(tau) else None,
            top_k_overlap={str(k):dict(k=min(k,len(valid)),common=len({p['id'] for p in exact[:k]} & {p['id'] for p in power[:k]}),
                fraction=len({p['id'] for p in exact[:k]} & {p['id'] for p in power[:k]})/min(k,len(valid)) if valid else None)
                           for k in (1,2,4,8)},
            candidates=[dict(id=p['id'],lambda_exact=p['curvature'],lambda_power10=p['curvature_power10'],
                lambda_exact_plus_mu=p['curvature']+call['mu'] if call.get('mu') is not None else None,
                lambda_power10_plus_mu=p['curvature_power10']+call['mu'] if call.get('mu') is not None else None,
                delta=p['curvature']-p['curvature_power10'],
                relative_error=abs(p['curvature']-p['curvature_power10'])/abs(p['curvature']) if p['curvature'] else None,
                ell_exact=p['score'],ell_power10=p['score_power10']) for p in valid]))
    return dict(context='Both estimates use the same candidates, residual and H_beta on the D-exact trajectory.',calls=result)


def profile_capacity(model, batch, boundary=lambda: None, mu=None):
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
            row=dict(probe_size=n,classes=100,m=n*100,dense_gram_reference_bytes=8*(n*100)**2,
                source='task-2 weights, extended 100-class head, repeated window; capacity proxy, not actual task-20 data')
            start=time.perf_counter()
            try:
                diagnostic=vna.diagnose(clone,(x,y),sites=sites_for(clone),mu=mu)
                row.update(diagnostic,status='COMPLETED' if diagnostic.get('diagnostic_flag') is None else 'CG_NOT_CONVERGED')
                sites=sites_for(clone)
                if n==128 and sites and hasattr(vna,'Downstream') and vna.separable(clone):
                    ds=None
                    try:
                        with vna.training_products(clone,[],{},x) as products:
                            G=vna.FrozenGram(products)
                            seed=torch.ones(n*100,device=device,dtype=torch.float64)
                            G.JT(seed);G.high.JT(seed)
                            ds=vna.Downstream(G.high.products.model,sites[0],x.double())
                            ds.measure()
                            row['downstream_capacity']=dict(status='MEASURED',bytes=ds.bytes,
                                seconds=ds.seconds,vjp_calls=ds.vjp_calls,dtype=str(ds.dtype),
                                block_samples=ds.block_samples,blocks=ds.blocks,retains_full_D=False,
                                scope='One streamed stage-1 D block plus frozen native/double VJP graphs; not a full application.')
                    except torch.cuda.OutOfMemoryError as exc:
                        row['downstream_capacity']=dict(status='CUDA_OOM',error=str(exc))
                    except ValueError as exc:
                        row['downstream_capacity']=dict(status='UNSUPPORTED',error=str(exc))
                    finally:
                        del ds
            except torch.cuda.OutOfMemoryError as exc:
                row.update(status='CUDA_OOM',error=str(exc))
            except RuntimeError as exc:
                if not str(exc).startswith('GRAM_LIMIT:'): raise
                row.update(status='GRAM_LIMIT',error=str(exc))
            finally:
                torch.cuda.synchronize()
                row.update(seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
            rows.append(row)
            del x,y
            gc.collect();torch.cuda.empty_cache()
    return dict(rows=rows,actual_task20_measured=False,probe_change_applied=False,
        recommendation='Keep probe=256; review actual CG convergence, elapsed time and memory at both sizes before M1.')


def is_task1_checkpoint(state):
    if not isinstance(state, dict) or not isinstance(state.get('identity'), dict): return False
    expected = dict(arm='icarl_full', seed=0, backbone='resnet32_nobn', class_order_kind='superclass',
                    nobn_lr_scale=.5, code_commit='d88abc496eb1fe21d1b6df66d90b47cf93d7ded1')
    return (all(state['identity'].get(k) == v for k,v in expected.items()) and
            (state.get('completed'), state.get('mid'), state.get('epoch')) == (1, False, 100))


def find_task1_checkpoint(source, target):
    """Select by checkpoint metadata; identical attached copies count as one source."""
    import importlib
    import torch
    from diagnostics.vna_align import _global_state
    # A reused notebook kernel can retain the submodule without its package attribute.
    if not hasattr(torch, '_utils'):
        torch._utils = importlib.import_module('torch._utils')
        print('Restored torch._utils package binding before checkpoint discovery', flush=True)
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
    if old: args.curvature='power10'
    supports_curvature='curvature' in inspect.signature(vna.V3Engine).parameters
    supports_solver='solver' in inspect.signature(vna.V3Engine).parameters
    supports_live='gir_live' in inspect.signature(vna.V3Engine).parameters
    if args.gir_live and not supports_live: raise ValueError('Live-slot ablation unavailable in selected source')
    args.solver=args.solver or ('cg_ir' if hasattr(vna,'OutputCG') else 'historical-cg' if old else 'gram')
    if not supports_solver and not old and args.solver!=('cg_ir' if hasattr(vna,'OutputCG') else 'gram'):
        raise ValueError('Requested solver unavailable in selected source')
    if args.branch=='D' and not supports_curvature and args.curvature!='power10':
        raise ValueError('Historical D supports power10 only')
    if old and not native_commit.startswith('d88abc4'):
        raise ValueError('B/C must import d88abc4 source')
    identity = dict(branch=args.branch, code_commit=native_commit, profile=args.profile,
        source_digest=os.environ.get('VNA_SOURCE_DIGEST'), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        capacity_profile=args.capacity_profile,curvature=args.curvature,
        solver_backend=args.solver,followup_steps=20)
    if supports_live: identity.update(policy_version=vna.POLICY_VERSION,gir_live=args.gir_live)
    prior=None
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
        curvature=args.curvature,timers=[], cg=[], jvp_vjp={}, applications=[],candidate_rankings=[],applied_swaps=[])
    atomic_json_save(report,out/'diagnostic.json')
    active = dict(hook=None, model=None, applications=0, stage=[], last_lambda_step=None,
                  audit_seconds=0.,proposal_rows=None,power_values=None)
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
                    engine=vna.V3Engine('vna_gir',seed=0,K0=1,**({'curvature':args.curvature} if supports_curvature else {}),
                        **({'solver':args.solver} if supports_solver else {}),
                        **({'gir_live':args.gir_live} if supports_live else {}))
                    oldengine = hookstate.get('intervention_state',{}).get('engine',{})
                    if not supports_live:
                        engine.mu = oldengine.get('mu')
                        engine.install_ages = deepcopy(oldengine.get('install_ages',{}))
                        engine.protocol['mu0'] = oldengine.get('protocol',{}).get('mu0')
                    hookstate['intervention_state']=main.Intervention(engine,1).state_dict()
            if prior and prior.get('status')=='PROFILE_FOLLOWUP':
                origin=prior['application_step']
                events=self.state['hook'].get('intervention_state',{}).get('events',[])
                if any(e.get('iteration')==origin for e in events):
                    report.clear();report.update(deepcopy(prior));active['applications']=1
                    report['post_application_sgd']=[r for r in report.get('post_application_sgd',[])
                        if r['step']<=self.state['hook']['steps']]
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
        if args.profile and report.get('status')=='PROFILE_FOLLOWUP':
            record_profile_loss(report,hook.steps,loss);write()
            if report['status']=='PROFILE_COMPLETED':raise Stop()
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
        audit_start=active['audit_seconds'];start=time.perf_counter()
        event=engine.apply(model,sites_for(model),batch,spec,ctx,optimizer,hook.steps,force=True,
            **({'probe_batch':main.fixed_probe(learner,256)[0]} if supports_live else {}))
        if event.get('skipped') and not event.get('swap_records') and report['applied_swaps']:
            report['rolled_back_swaps'],report['applied_swaps']=report['applied_swaps'],[]
        torch.cuda.synchronize();report['application_wall_seconds']=time.perf_counter()-start
        report['curvature_audit_seconds']=active['audit_seconds']-audit_start
        report['peak_cuda_includes_curvature_audit']=report['curvature_audit_seconds']>0
        report['stability_seconds']=event.get('stability_seconds',0.)
        report['application_seconds']=report['application_wall_seconds']-report['curvature_audit_seconds']-report['stability_seconds']
        event['raw_seconds']=event.get('seconds')
        for key in ('seconds','gir_seconds'):
            if event.get(key) is not None: event[key]-=report['curvature_audit_seconds']
        report['application_peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        report['peak_cuda_bytes']=report['application_peak_cuda_bytes']
        report['applications'].append(event);intervention.events.append(event)
        if hasattr(vna,'OutputCG'):
            report['cg']=[r for r in event.get('solves',[])+event.get('gir_solves',[]) if r.get('solver') in ('cg','cg_ir')]
        report['ir_summary']=dict(actual_solves=sum(not r.get('cached',False) for r in report['cg']),
            cached_requests=sum(bool(r.get('cached')) for r in report['cg']),
            outer_corrections=sum(r.get('refinements',0) for r in report['cg']),
            high_seconds=sum(r.get('high_seconds',0.) for r in report['cg']),
            inner_seconds=sum(r.get('inner_seconds',0.) for r in report['cg']))
        report['systems_solved']=len(report['cg']) if old else event.get('systems_solved',len(event.get('solves',[])))
        if args.profile and args.branch=='D' and not event.get('skipped'):
            report['final_L_KD']=list(engine._evaluate_current(model,batch,spec,ctx))
            trials=event.get('gir_trials',[])
            costs=[r.get('gamma_total_seconds',r['gamma_seconds']) for r in trials]
            report['gamma_cost']=dict(trials=len(trials),total_seconds=sum(costs),
                mean_seconds=sum(costs)/len(costs) if costs else None,
                max_seconds=max(costs,default=None),
                interpretation=('Includes on-demand candidate columns inside Gamma; native slot coordinates are implicit.'
                    if hasattr(vna,'Downstream') else
                    'Includes native slot columns once per slot; candidate Jacobians charged to prepare_gir when cached.'))
            ratios=[abs(r['Gamma_cg'])/r['linear_solve_error_bound'] for r in trials
                    if r.get('linear_solve_error_bound',0)>0 and 'Gamma_cg' in r]
            report['gamma_precision']=dict(min_abs_gamma_over_linear_solve_bound=min(ratios,default=None),
                ratios_below_one=sum(r<1 for r in ratios),measured_trials=len(ratios),decision_gate=bool(supports_live),
                decision_scope='Dead/live gain must exceed its corresponding residual bound' if supports_live else 'observational',
                scope='Linear solves in the represented frozen system only; excludes metric/Jacobian assembly and jump rounding.')
            report['performance_targets']=dict(application_target_seconds=180,
                application_under_180_seconds=report['application_seconds']<180,
                application_under_400_seconds=report['application_seconds']<400,
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
                report['capacity_profile']=profile_capacity(model,batch,hook.measurement_boundary,mu=engine.mu_star)
                capacity=next(r for r in report['capacity_profile']['rows'] if r['m']==12800)
                report['performance_targets'].update(capacity_12800_seconds=capacity.get('seconds'),
                    capacity_12800_completed=capacity.get('status')=='COMPLETED')
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
            report['application_step']=hook.steps
            report['status']='PROFILE_SKIPPED' if event.get('skipped') else 'PROFILE_FOLLOWUP'
            write()
            if event.get('skipped'): raise Stop()
    main.Intervention.after_step=after
    if args.branch == 'C':
        original_install=vna.V3Engine.install_gir
        def install(engine,model,site,proposals,units,optimizer):
            saved=deepcopy(optimizer.state_dict())
            try:return original_install(engine,model,site,proposals,units,optimizer)
            finally:optimizer.load_state_dict(saved)
        vna.V3Engine.install_gir=install
    if args.profile:
        original_rows=vna.rows_proposal
        def rows(site,values):
            result=original_rows(site,values)
            if active['proposal_rows'] is not None:
                active['proposal_rows'].append((values[0].detach().clone(),result))
            return result
        vna.rows_proposal=rows
        original_power=vna.power_lambda_max
        def power(action,size,generator,*a,**kw):
            value=original_power(action,size,generator,*a,**kw)
            if active['power_values'] is not None:
                local=inspect.currentframe().f_back.f_locals
                key=('seed',local['seed_index']) if 'seed_index' in local else ('A',id(local.get('candidate')))
                active['power_values'][key]=dict(curvature=value,power_generator_seed=generator.initial_seed(),
                    power_iterations=kw.get('iterations',20))
            return value
        vna.power_lambda_max=power
        profile_original_install=vna.V3Engine.install_gir
        def profiled_install(engine,model,site,proposals,units,optimizer):
            used=profile_original_install(engine,model,site,proposals,units,optimizer)
            for proposal,unit in zip(proposals[:used],units[:used]):
                report['applied_swaps'].append(dict(site=site.module_name,unit=int(unit),
                    candidate_id=proposal.get('_profile_id'),A={key:None if value is None else value.detach().cpu().tolist()
                    for key,value in proposal['A'].items()}))
            return used
        vna.V3Engine.install_gir=profiled_install
        for method in ('_direction','_select','_slot_jump','_slot_contribution','prepare_gir','_run_gir'):
            original=getattr(vna.V3Engine,method)
            def wrap(original=original,name=method):
                def timed(*a,**kw):
                    torch.cuda.synchronize();start=time.perf_counter();active['stage'].append(name)
                    audit_start=active['audit_seconds']
                    if name=='prepare_gir': active['proposal_rows'],active['power_values']=[],{}
                    try:
                        result=original(*a,**kw)
                        if name=='prepare_gir':
                            candidates=[]
                            for proposal in result:
                                row_index=next(i for i,(direction,_) in enumerate(active['proposal_rows'])
                                               if torch.equal(direction,proposal['direction']))
                                base=active['proposal_rows'][row_index][1]
                                key='weight' if bool(base['weight'].ne(0).any()) else 'bias'
                                sign=1 if torch.equal(base[key],proposal['A'][key]) else -1
                                proposal['_profile_id']=f"{row_index}:{sign}:{proposal.get('bias_q')}"
                                measured=active['power_values'].get(('A',id(proposal['A'])),
                                    active['power_values'].get(('seed',proposal.get('power_seed_index')),{}))
                                candidates.append(dict(id=proposal['_profile_id'],score=proposal['score'],
                                    curvature=proposal.get('curvature',measured.get('curvature')),curvature_method=proposal.get('curvature_method'),
                                    curvature_power10=proposal.get('curvature_power10'),score_power10=proposal.get('score_power10'),
                                    power_generator_seed=measured.get('power_generator_seed'),power_iterations=measured.get('power_iterations')))
                            active['audit_seconds']+=sum(p.get('audit_seconds',0.) for p in result)
                            report['candidate_rankings'].append(dict(site=a[2].module_name,iteration=a[8],
                                used_directions=len(kw.get('used_directions',())),mu=kw.get('mu'),candidates=candidates))
                        return result
                    finally:
                        if name=='prepare_gir': active['proposal_rows'],active['power_values']=None,None
                        torch.cuda.synchronize();raw=time.perf_counter()-start
                        audit=active['audit_seconds']-audit_start
                        report['timers'].append(dict(name=name,seconds=raw-audit,wall_seconds=raw,audit_seconds=audit));active['stage'].pop()
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
                  backbone='resnet32_nobn',nobn_lr_scale=.5,synthetic=False,K0=1,curvature=args.curvature)
    if supports_solver:schedule['solver']=args.solver
    if supports_live:schedule.update(policy_version=vna.POLICY_VERSION,gir_live=args.gir_live)
    (out/'schedule.json').write_text(json.dumps(schedule))
    main_parser=main.parser()
    runargs=main_parser.parse_args(['--arm','icarl' if args.branch=='A' else 'icarl_full',
        '--nobn-lr-scale','.5','--seed','0','--device','cuda:0','--data-dir',args.data_dir,
        '--out',str(out/'run'),'--schedule',str(out/'schedule.json')]+
        (['--curvature',args.curvature] if '--curvature' in main_parser._option_string_actions else [])+
        (['--solver',args.solver] if '--solver' in main_parser._option_string_actions else [])+
        (['--gir-live'] if args.gir_live else []))
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
            name='profile_d88abc4.json' if old else 'profile_cg.json' if hasattr(vna,'OutputCG') else 'profile_gram.json'
            atomic_json_save(report,out/name)
        write()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    p.add_argument('--branch',choices=list('ABCD'),required=True)
    p.add_argument('--archive');p.add_argument('--checkpoint');p.add_argument('--input-root',default='/kaggle/input')
    p.add_argument('--data-dir',required=True)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--gir-live',action='store_true')
    p.add_argument('--capacity-profile',action='store_true')
    p.add_argument('--solver',choices=('gram','cg_ir'))
    p.add_argument('--curvature',choices=('power10','exact'),default='exact')
    return p


if __name__=='__main__':run(parser().parse_args())
