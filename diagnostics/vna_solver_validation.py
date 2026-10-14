"""CPU-only Stage 2 solver and equal-rank selection audit (no training runner).

Run: .venv/bin/python -m diagnostics.vna_solver_validation --output PATH
An optional checkpoint must be a tiny_model state dict or {'model': state}.
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import statistics
import time
import torch
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


def run(output, checkpoint=None, seed=20261014):
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
    cfg = GGNConfig(seed=seed, cg_iters=500, cg_tol=1e-5)
    with block_products(model,site,A,batch[0]) as p:
        op=GGNProducts(p,batch[1]); refs=p.parameter_refs
        trace=op.trace(cfg); gradient=op.gradient
        eye=torch.eye(op.g.numel())
        J=torch.stack([p.JT(e).double() for e in eye]); E=torch.stack([p.JET(e).double() for e in eye])
        H=torch.stack([op.H(e) for e in eye],1)
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
    for relative in (1e-4,1e-3,1e-2,1e-1):
        mu=relative*trace/len(gradient)
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
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='docs/vna/stage2_validation')
    parser.add_argument('--checkpoint'); parser.add_argument('--seed',type=int,default=20261014)
    args=parser.parse_args(); run(args.output,args.checkpoint,args.seed)


if __name__=='__main__':
    main()
