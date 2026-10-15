"""Stage 2 toy and matrix-free real-checkpoint solver audits.

Default toy mode accepts a tiny-model checkpoint. Use --mode recipe with a
Phase-0 checkpoint and CIFAR root for trained-model GPU validation.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import time
import torch
from experiments.vna_resume import Progress, file_hash, log, pause_if_due
from experiments.vna_support import tiny_model, tiny_candidate, isolated_rng
from methods.vna import (dedicated_generator, extract_tiny_A, block_products,
                         vna_block_correction, VNAConfig)
from methods.vna_ggn import (GGNConfig, GGNProducts, vna_exact_ggn, vna_lite_ggn,
                            ngd_control, filtered_cotangent, project_bn_tangent)
from methods.vna_kfac import KFACBlockPreconditioner, vna_kfac
from methods.vna_select import (tiny_V, greedy_V, normalize_A, score_A,
                               trace_matched_random_A)


def cosine(a, b):
    denom = float(a.norm() * b.norm())
    return float(a.dot(b)) / denom if denom else 0.


def flatten(delta, refs):
    return torch.cat([delta.get(n, torch.zeros_like(p)).flatten() for n, p in refs]).double()


def run(output, checkpoint=None, seed=20261014, damping_rel=1e-2, damping_norm="outputs"):
    torch.set_num_threads(1)
    generator = dedicated_generator(seed, 'solver-validation')
    with isolated_rng(generator):
        model = tiny_model()
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state.get('model', state))
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    # Round trip the synthetic checkpoint before any diagnostic.
    torch.save({'model': model.state_dict(), 'synthetic': checkpoint is None}, output/'tiny_checkpoint.pt')
    model.load_state_dict(torch.load(output/'tiny_checkpoint.pt', weights_only=False)['model'])
    site = model.growing_blocks()[0].name
    batch = (torch.randn(3,3,8,8,generator=generator), torch.tensor([0,1,2]))
    stats_batch = (torch.randn(12,3,8,8,generator=generator), torch.arange(12)%3)
    candidate = tiny_candidate(model, site, stats_batch, 2, seed=seed)
    A,_ = extract_tiny_A(model,candidate)
    A = normalize_A(model,site,A)
    cfg = GGNConfig(seed=seed, damping_norm=damping_norm, cg_iters=500, cg_tol=1e-5)
    with block_products(model,site,A,batch[0]) as p:
        op=GGNProducts(p,batch[1]); refs=p.parameter_refs
        trace=op.trace(cfg); gradient=op.gradient
        eye=torch.eye(op.g.numel())
        J=torch.stack([p.JT(e).double() for e in eye]); E=torch.stack([p.JET(e).double() for e in eye])
        H=torch.stack([op.H(e) for e in eye],1)
        _ = (J, E, H)  # retained dense toy-oracle work
    # Measure actual optimizer steps under a restorable independent baseline.
    with isolated_rng(dedicated_generator(seed, "sgd-baseline-model")):
        clone = tiny_model()
    clone.load_state_dict(model.state_dict()); clone.train()
    optimizer = torch.optim.SGD(clone.parameters(),lr=.01)
    times=[]
    baseline_batch=(torch.randn(64,3,8,8,generator=generator),torch.arange(64)%3)
    for _ in range(4):
        clone.load_state_dict(model.state_dict())
        start=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(clone(baseline_batch[0]),baseline_batch[1]).backward(); optimizer.step()
        times.append(time.perf_counter()-start)
    sgd=statistics.median(times[1:])
    preconditioner=KFACBlockPreconditioner()
    start=time.perf_counter(); preconditioner.refresh(model,site,batch); refresh=time.perf_counter()-start
    rows=[]; selection=[]
    for relative in sorted({1e-4,1e-3,1e-2,1e-1,damping_rel}):
        mu=(max(relative*trace/len(gradient),1e-12) if cfg.damping_norm == "params" else relative*trace/len(op.g))
        exact,de=vna_exact_ggn(model,site,A,batch,cfg,mu=mu,trace_F=trace)
        reference=flatten(exact,refs)
        variants={'exact':(exact,de),
            'kfac':vna_kfac(model,site,A,batch,preconditioner,cfg,mu=mu,trace_F=trace),
            'lite':vna_lite_ggn(model,site,A,batch,cfg,mu=mu,trace_F=trace),
            'ngd':ngd_control(model,site,A,batch,cfg,mu=mu,trace_F=trace,reference=exact)}
        old,do=vna_block_correction(model,site,A,batch,VNAConfig(seed=seed,cg_iters=500,cg_tol=1e-5,damping_rel=relative),apply_gain_guard=False)
        old_raw_gain=float(gradient.dot(flatten(old,refs)))
        old=project_bn_tangent(model,site,old)
        variants['euclidean']=(old,dict(do,seconds=do['wall_time'],raw_gain=old_raw_gain,realized_gain=float(gradient.dot(flatten(old,refs)))))
        # Ablate ONLY W1 rows of the exact cross operator, under the same
        # block-local KFAC inverse. This is precisely the prohibited local
        # cross-factorization: KFAC cannot restore a missing W1 cross row.
        start=time.perf_counter()
        with block_products(model,site,A,batch[0]) as p:
            abop=GGNProducts(p,batch[1])
            w1mask=torch.cat([torch.full((r.numel(),),'.first_layer.layer.weight' in n,dtype=torch.bool) for n,r in refs])
            u=preconditioner.apply(abop.gradient,refs,mu)
            u[w1mask]=0
            rhs=abop.COE(abop.CEO(u)); rhs[w1mask]=0
            dropped=preconditioner.apply(rhs,refs,mu)
            delta={}; offset=0
            for n,r in refs:
                delta[n]=dropped[offset:offset+r.numel()].reshape_as(r); offset+=r.numel()
            delta=project_bn_tangent(model,site,delta)
            dd={'raw_gain':float(gradient.dot(dropped)),
                'realized_gain':float(gradient.dot(flatten(delta,refs)))}
            for key in ('forward_calls','jvp_calls','vjp_calls','extension_jvp_calls','extension_vjp_calls'):
                dd[key]=getattr(p,key)
        dd['seconds']=time.perf_counter()-start
        variants['cross_dropped']=(delta,dd)
        for name,(d,diag) in variants.items():
            value=flatten(d,refs)
            row={'relative_mu':relative,'mu':mu,'variant':name,'cosine':cosine(value,reference),
                 'raw_gain':diag['raw_gain'],'realized_gain':diag['realized_gain'],
                 'seconds':diag['seconds'],'sgd_units':None if diag['seconds'] is None else diag['seconds']/sgd,
                 'cg_converged':all(r['converged'] for r in diag.get('cg',[]))}
            for group,match in [('W1','.first_layer.layer.weight'),('W2','.second_layer.layer.weight'),('BN','post_layer_function')]:
                indices=torch.cat([torch.full((r.numel(),),match in n,dtype=torch.bool) for n,r in refs])
                row['cos_'+group]=cosine(value[indices],reference[indices]); row['norm_'+group]=float(value[indices].norm())
            for key in ('forward_calls','jvp_calls','vjp_calls','extension_jvp_calls','extension_vjp_calls'):
                row[key]=diag.get(key)
            row['product_passes']=sum(diag.get(key,0) for key in ('jvp_calls','vjp_calls','extension_jvp_calls','extension_vjp_calls'))
            rows.append(row)
        zstats=[filtered_cotangent(model,site,A,(x,y),cfg,mu=mu) for x,y in zip(stats_batch[0].split(6),stats_batch[1].split(6))]
        av,_=tiny_V(model,site,list(zip(stats_batch[0].split(6),stats_batch[1].split(6))),2,cotangents=zstats,cfg=cfg)
        z=filtered_cotangent(model,site,A,batch,cfg,mu=mu)
        ag,gd=greedy_V(model,site,av,batch[0],z,random_count=2,r_max=2,theta=1.)
        rank=len(ag['weight'])
        truncate=lambda a:{n:None if v is None else v[:rank] for n,v in a.items()}
        au,av=truncate(A),truncate(av)
        random,rd=trace_matched_random_A(model,site,av,batch[0],seed=seed)
        for name,a in [('tiny_U',au),('tiny_V',av),('greedy_V',ag),('random_A',random)]:
            selection.append({'relative_mu':relative,'selection':name,'rank':len(a['weight']),
                              'V':score_A(model,site,a,batch[0],z)})
    summary={'synthetic':checkpoint is None,'seed':seed,'site':site,'trace_F_hutchinson':trace,
             'training_damping_rel':damping_rel, 'damping_norm':cfg.damping_norm,
             'exact_converged_at_train_mu':all(r['cg_converged'] for r in rows if r['variant']=='exact' and r['relative_mu']==damping_rel),
             'sgd_seconds':sgd,'sgd_batch':64,'kfac_refresh_seconds':refresh,
             'kfac_refresh_sgd_units':refresh/sgd,'rows':rows,'selection':selection,
             'cross_ablation':'Only W1 rows of C dropped on both sides, same KFAC inverse; W1 correction is zero',
             'cost_note':'CPU single run; correction timings reuse trace; refresh separately charged; all matrix-free product counts included',
             'projection_note':'PSD guarantees apply before BN tangent projection; realized gains are measured afterward'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for filename,data in [('solvers.csv',rows),('selection.csv',selection)]:
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    print(json.dumps(summary,indent=2))
    if not summary['exact_converged_at_train_mu']:
        raise RuntimeError('Exact reference did not converge at training damping; diagnostics saved.')
    return summary


def run_checkpoint(args, *, stage_metadata=None, require_convergence=True):
    """Matrix-free real recipe checkpoint audit; never touches official test."""
    audit_start=time.perf_counter()
    import copy
    from experiments.run_vna_training import parser as training_parser
    from experiments.vna_support import setup_data, build_model, augmented_batch
    from experiments.vna_training_support import TrainingVNA, sync
    from experiments.shared_protocol import atomic_json_save
    device=torch.device(args.device)
    if args.fake_data: torch.set_num_threads(1)
    generator=dedicated_generator(args.seed,'real-validation-data')
    train,_,indices,_,_,_=setup_data(args.data_root,args.fake_data,args.seed)
    with isolated_rng(generator): model=build_model(device,args.fake_data)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(payload.get('model',payload))
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    config=training_parser().parse_args([])
    for key in vars(config):
        if hasattr(args,key): setattr(config,key,getattr(args,key))
    config.solver='exact'
    engine=TrainingVNA(config,device)
    refresh_cost=engine.refresh(model,train,indices,1)
    batch=augmented_batch(train,indices,args.vna_batch,generator,device)
    baseline_batch=augmented_batch(train,indices,args.batch_size,generator,device)
    clone=copy.deepcopy(model); clone.train()
    optimizer=torch.optim.SGD(clone.parameters(),lr=.1,momentum=.9,weight_decay=5e-4)
    state=copy.deepcopy(clone.state_dict()); times=[]
    for _ in range(4):
        clone.load_state_dict(state); optimizer.state.clear()
        sync(device); t=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(clone(baseline_batch[0]),baseline_batch[1]).backward(); optimizer.step()
        sync(device); times.append(time.perf_counter()-t)
    sgd=statistics.median(times[1:]); del clone,optimizer,state
    rows=[]; factor_cost={}; batch_checks=[]
    cfg=engine.cfg
    if stage_metadata is not None:
        from dataclasses import replace
        cfg=replace(cfg, damping_norm="outputs")
    progress = None
    completed = []
    last_unit_seconds = 0.
    if stage_metadata is not None:
        identity = {'args': vars(args), 'checkpoint_sha256': file_hash(args.checkpoint),
                    'code_commit': stage_metadata['code_commit']}
        progress = Progress(output, identity, name='progress.pt')
        if progress.state:
            saved = progress.state
            rows, batch_checks, factor_cost = saved['rows'], saved['batch_checks'], saved['factor_cost']
            completed = saved['completed']
            generator.set_state(saved['generator'].cpu())
            for key, value in saved['generators'].items(): engine.generators[key].set_state(value.cpu())
            last_unit_seconds = saved['last_unit_seconds']
    relatives = sorted({1e-4,1e-3,1e-2,1e-1,config.damping_rel})
    for site,entry in engine.cache.items():
        if all((site, relative) in completed for relative in relatives): continue
        A=entry['A']; trace=entry['trace_F']
        normalization=None
        if stage_metadata is not None:
            from methods.vna_select import prepare_A
            A,normalization=prepare_A(model,site,A,args.selection)
        kfac=KFACBlockPreconditioner()
        sync(device); t=time.perf_counter(); kfac.refresh(model,site,batch); sync(device)
        factor_cost[site]=time.perf_counter()-t
        with block_products(model,site,A,batch[0],cache_block_input=cfg.cache_block_input) as p:
            refs=p.parameter_refs; size=sum(p.numel() for _,p in refs)
            op=GGNProducts(p,batch[1])
            n_out=op.g.numel()
            block_gradient_norm=float(op.gradient.norm())
        for relative in relatives:
            if (site, relative) in completed: continue
            if progress: pause_if_due(seconds=last_unit_seconds)
            unit_started = time.monotonic()
            log(f'solver epoch={payload.get("epoch")} site={site} rank={entry["rank"]} mu={relative} start')
            mu=(max(relative*trace/size,1e-12) if cfg.damping_norm == "params" else relative*trace/n_out)
            variants={}
            with isolated_rng(engine.generators['construction']):
                for name in ('exact','kfac','lite','euclidean','ngd'):
                    sync(device); t=time.perf_counter()
                    kwargs={'mu':mu,'trace_F':trace}
                    if name=='exact': d,diag=vna_exact_ggn(model,site,A,batch,cfg,**kwargs)
                    elif name=='kfac': d,diag=vna_kfac(model,site,A,batch,kfac,cfg,**kwargs)
                    elif name=='lite': d,diag=vna_lite_ggn(model,site,A,batch,cfg,**kwargs)
                    elif name=='ngd': d,diag=ngd_control(model,site,A,batch,cfg,reference=variants['exact'][0],**kwargs)
                    else:
                        d,diag=vna_block_correction(model,site,A,batch,VNAConfig(seed=args.seed,
                            cg_iters=args.cg_iters,cg_tol=args.cg_tol,damping_rel=relative),apply_gain_guard=False)
                        d=project_bn_tangent(model,site,d)
                        with block_products(model,site,A,batch[0],cache_block_input=cfg.cache_block_input) as p:
                            gradient=GGNProducts(p,batch[1]).gradient
                            diag['realized_gain']=float(gradient.dot(flatten(d,refs)))
                    sync(device); diag['seconds']=time.perf_counter()-t
                    variants[name]=(d,diag)
                    log(f'solver epoch={payload.get("epoch")} site={site} rank={entry["rank"]} '
                        f'mu={relative} variant={name} seconds={diag["seconds"]:.3f} '
                        f'CG iterations={[r.get("iterations") for r in diag.get("cg", [])]} '
                        f'converged={all(r["converged"] for r in diag.get("cg", []))}')
            reference=flatten(variants['exact'][0],refs)
            for name,(d,diag) in variants.items():
                value=flatten(d,refs)
                row={'site':site,'epoch':payload.get('epoch'),'relative_mu':relative,'mu':mu,
                    'variant':name,'cosine':cosine(value,reference),'raw_gain':diag.get('raw_gain'),'realized_gain':diag['realized_gain'],
                    'seconds':diag['seconds'],'sgd_units':diag['seconds']/sgd,
                    'cg_converged':all(r['converged'] for r in diag.get('cg',[])),
                    'cg':diag.get('cg',[]),'rank':entry['rank']}
                row.update(mu_rel=relative, damping_norm=cfg.damping_norm, trace_F=trace,
                           n_out=n_out, p=size, mu_rel_equiv_params=mu/(trace/size) if trace else None)
                if name == 'euclidean':
                    row.update(mu=diag.get('mu'), damping_metric='phase0_euclidean_trace_K')
                    row['mu_rel_equiv_params']=diag.get('mu')/(trace/size) if trace else None
                row['lambda_d_over_g']=abs(config.lambda_vna)*float(value.norm())/max(block_gradient_norm,1e-30)
                if normalization is not None:
                    row['normalization_constant']=normalization['normalization_constant']
                for group,match in [('W1','.first_layer.layer.weight'),('W2','.second_layer.layer.weight'),('BN','post_layer_function')]:
                    mask=torch.cat([torch.full((p.numel(),),match in n,device=device,dtype=torch.bool) for n,p in refs])
                    row['cos_'+group]=cosine(value[mask],reference[mask])
                rows.append(row)
            if relative==config.damping_rel:
                # One augmented 256-example pool: nested 16/64/128 reference,
                # and a strictly disjoint 128-example comparison at the same mu.
                if len(indices)<256: raise ValueError('not enough samples for disjoint 128 audit')
                audit_size=128
                order=torch.randperm(len(indices),generator=generator).tolist()
                pools=[[indices[i] for i in order[j*audit_size:(j+1)*audit_size]] for j in range(2)]
                reference_batches=[augmented_batch(train,pool,audit_size,generator,device) for pool in pools]
                large=reference_batches[0]
                nested=[(large[0][:b],large[1][:b]) for b in (16,64)]
                directions=[flatten(vna_exact_ggn(model,site,A,b,cfg,mu=mu,trace_F=trace)[0],refs)
                            for b in [*nested,*reference_batches,
                                      (reference_batches[1][0][:16],reference_batches[1][1][:16]),
                                      (reference_batches[1][0][:64],reference_batches[1][1][:64])]]
                check={'site':site,'epoch':payload.get('epoch'),'vna_batch':config.vna_batch,
                    'cos16':cosine(directions[0],directions[2]),
                    'cos64':cosine(directions[1],directions[2]),
                    'cos_batch_128':cosine(directions[0 if config.vna_batch==16 else 1 if config.vna_batch==64 else 2],directions[2]),
                    'cos_disjoint16':cosine(directions[0],directions[4]),
                    'cos_disjoint64':cosine(directions[1],directions[5]),
                    'cos_disjoint128':cosine(directions[2],directions[3]),
                    'cos_disjoint_batches':cosine(directions[0],directions[4]) if config.vna_batch==16 else
                        cosine(directions[1],directions[5]) if config.vna_batch==64 else cosine(directions[2],directions[3]),
                    'batch_reference_size':audit_size}
                batch_checks.append(check)
                for row in rows:
                    if row['site']==site and row['relative_mu']==relative: row.update(check)
            if progress:
                last_unit_seconds = time.monotonic()-unit_started
                completed.append((site, relative))
                progress.save({'rows': rows, 'batch_checks': batch_checks, 'factor_cost': factor_cost,
                               'completed': completed, 'generator': generator.get_state(),
                               'generators': {key: g.get_state() for key, g in engine.generators.items()},
                               'last_unit_seconds': last_unit_seconds})
    summary={'checkpoint':str(args.checkpoint),'epoch':payload.get('epoch'),'synthetic':args.fake_data,
        'device':str(device),'seed':args.seed,'sites':list(engine.cache),'sgd_seconds':sgd,
        'refresh_seconds':refresh_cost,'kfac_refresh_seconds':factor_cost,'rows':rows,
        'training_damping_rel':config.damping_rel,'training_config':vars(config),'batch_checks':batch_checks,
        'exact_converged_at_train_mu':all(r['cg_converged'] for r in rows if r['variant']=='exact' and r['relative_mu']==config.damping_rel),
        'exact_converged':all(r['cg_converged'] for r in rows if r['variant']=='exact'),
        'cost_note':'Synchronized correction cost; refresh separately measured; training-batch recipe SGD units.'}
    if stage_metadata is not None:
        stage_metadata=dict(stage_metadata,wall_seconds=time.perf_counter()-audit_start)
        summary.update(stage_metadata)
        for row in rows:
            row.update(stage_metadata)
        for check in batch_checks:
            check.update(stage_metadata)
    atomic_json_save(summary,output/'summary.json')
    with (output/'solvers.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row))); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary,indent=2))
    if require_convergence and not summary['exact_converged_at_train_mu']:
        raise RuntimeError('Exact reference did not converge; residuals saved. Increase --cg-iters or inspect precision.')
    return summary


def parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='docs/vna/stage2_validation')
    parser.add_argument('--checkpoint'); parser.add_argument('--seed',type=int,default=20261014)
    parser.add_argument('--mode',choices=['toy','recipe'],default='toy')
    parser.add_argument('--data-root',default='data')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fake-data',action='store_true')
    from experiments.vna_training_config import add_vna_arguments
    add_vna_arguments(parser)
    from experiments.run_vna_training import parser as training_parser
    parser.add_argument('--batch-size',type=int,default=training_parser().get_default('batch_size'))
    return parser


def main():
    p=parser()
    args=p.parse_args()
    if args.mode=='recipe':
        if not args.checkpoint: p.error('--mode recipe requires --checkpoint')
        run_checkpoint(args)
    else: run(args.output,args.checkpoint,args.seed,args.damping_rel,args.damping_norm)


if __name__=='__main__':
    main()
