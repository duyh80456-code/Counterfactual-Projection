"""Stateful Stage 2a corrections; no persistent model expansion.

Lite uses M = I/(tr(F)/number_of_logits), hence NGD_lite=M F M h.
Selection uses z=H J M h for all three solvers. Training shares one CG
curvature across sites; refreshed curvature invalidates the warm starts.
"""
from dataclasses import asdict
import time
import torch
from experiments.vna_support import augmented_batch, tiny_candidate, isolated_rng
from methods.vna import (dedicated_generator, block_products, extract_tiny_A,
                         _unflatten, assert_disjoint_scopes)
from methods.vna_ggn import GGNConfig, GGNProducts, project_bn_tangent, norm_match, relative_damping
from methods.vna_kfac import KFACBlockPreconditioner
from methods.vna_select import tiny_V, greedy_V, normalize_A, score_A, trace_matched_random_A
from projection.cg import conjugate_gradient



def preserve_buffers_and_modes(method):
    """Undo upstream lazy buffer registration as well as BN value changes."""
    from functools import wraps
    @wraps(method)
    def call(self, model, *args, **kwargs):
        snapshots=[(m,m.training,dict(m._buffers),
                    {n:None if b is None else b.detach().clone() for n,b in m._buffers.items()})
                   for m in model.modules()]
        try:
            return method(self,model,*args,**kwargs)
        finally:
            with torch.no_grad():
                for module,mode,buffers,values in snapshots:
                    module._buffers.clear(); module._buffers.update(buffers)
                    for name,buffer in buffers.items():
                        if buffer is not None: buffer.copy_(values[name])
                    module.training=mode
    return call


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def norm(delta):
    return sum(float(v.double().square().sum()) for v in delta.values()) ** .5


class TrainingVNA:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.generators = {p: dedicated_generator(args.seed, 'training-'+p)
                           for p in ('statistics', 'correction', 'selection', 'hutchinson', 'random_A', 'construction')}
        self.cache, self.factors, self.warm = {}, {}, {}
        self.spread = {}
        self.controller = {'state': 'active', 'refresh_epoch': 0, 'refreshes': 0}
        self.cfg = GGNConfig(damping_norm=args.damping_norm, damping_rel=args.damping_rel, cg_iters=args.cg_iters,
                             cg_tol=args.cg_tol, hutchinson_probes=args.hutchinson_probes,
                             seed=args.seed, project_bn=not args.no_bn_projection,
                             cache_block_input=args.cache_block_input)

    @property
    def coefficient(self):
        return self.args.lambda_vna*(self.args.vna_every if self.args.lambda_mode=='time_avg' else 1)

    def seed(self, purpose):
        return int(torch.randint(2**31, (), generator=self.generators[purpose]))

    def state_dict(self):
        return {'cache': self.cache, 'factors': {s: f.state_dict() for s,f in self.factors.items()},
                'warm': self.warm, 'spread': self.spread, 'controller': self.controller, 'config': asdict(self.cfg),
                'generators': {p: g.get_state() for p,g in self.generators.items()}}

    def load_state_dict(self, state):
        def move(value):
            if torch.is_tensor(value): return value.to(self.device)
            if isinstance(value, dict): return {k: move(v) for k,v in value.items()}
            if isinstance(value, list): return [move(v) for v in value]
            return value
        self.cache, self.warm = move(state['cache']), move(state['warm'])
        self.spread = move(state.get('spread', {}))
        self.controller = state['controller']
        for s, payload in state['factors'].items():
            factor = KFACBlockPreconditioner(); factor.load_state_dict(move(payload)); self.factors[s] = factor
        for p, value in state['generators'].items(): self.generators[p].set_state(value.cpu())

    def queue_spread(self, site, delta, skipped=False):
        # A rejected replacement must also clear any stale state at this site.
        self.spread.pop(site, None)
        if not skipped and delta and all(torch.isfinite(v).all() for v in delta.values()):
            self.spread[site] = {'delta': {n: v.detach().clone() for n,v in delta.items()},
                                 'remaining': self.args.vna_every}

    def apply_spread(self, model, lr=1.):
        refs = dict(model.named_parameters()); ratios = []
        for site, state in list(self.spread.items()):
            delta = state['delta']
            grad_norm = sum(float(refs[n].grad.detach().double().square().sum())
                            for n in delta if refs[n].grad is not None)**.5
            ratios.append({'site': site, 'applied_delta_ratio':
                           abs(self.coefficient)*norm(delta)/max(grad_norm, 1e-30)})
            if self.coefficient != 0:
                with torch.no_grad():
                    for name, value in delta.items(): refs[name].add_(value.to(refs[name]), alpha=-lr*self.coefficient)
            state['remaining'] -= 1
            if state['remaining'] == 0: del self.spread[site]
        return ratios

    @preserve_buffers_and_modes
    def correction_joint(self, model, batch):
        """All selected sites share the full original-parameter curvature."""
        from dataclasses import replace
        from methods.vna_joint import joint_products
        from methods.vna import _cg_record
        sites = list(self.cache)
        cfg = replace(self.cfg, cg_iters=50, cg_tol=1e-3, seed=self.seed('hutchinson'))
        records = []
        warm = self.warm.setdefault('__joint__', {})
        with joint_products(model, sites, {s:self.cache[s]['A'] for s in sites}, batch[0]) as p:
            op = GGNProducts(p, batch[1])
            trace = op.trace(cfg); mu = relative_damping(cfg, trace, op)
            def solve(rhs, key):
                result = conjugate_gradient(lambda v:op.F(v)+mu*v, rhs.double(),
                    x0=warm.get(key), max_iter=50, tolerance=1e-3, eps=1e-30)
                records.append(_cg_record(result, include_breakdown=True))
                if torch.isfinite(result.solution).all(): warm[key] = result.solution.detach().clone()
                return result.solution
            u = solve(op.gradient, 'u')
            z = op.H(p.Jv(u)); V = float(p.JET(z).double().square().sum())
            d = solve(op.COE(p.JET(z).double()), 'vna')
            vna_raw_gain = float(op.gradient.dot(d))
            def projected(value):
                delta = _unflatten(value, p.parameter_refs)
                for site in sites: delta = project_bn_tangent(model, site, delta, cfg.project_bn)
                return delta
            reference = projected(d); delta = reference
            raw_gain = vna_raw_gain
            if self.args.arm == 'ngd':
                d = solve(op.F(u), 'ngd')
                delta, scale = norm_match(projected(d), reference)
                raw_gain = float(op.gradient.dot(d))*scale
            flat = torch.cat([delta[n].flatten() for n,_ in p.parameter_refs]).double()
            ref_flat = torch.cat([reference[n].flatten() for n,_ in p.parameter_refs]).double()
            gain = float(op.gradient.dot(flat)); ref_gain = float(op.gradient.dot(ref_flat))
            finite = bool(torch.isfinite(flat).all() and torch.isfinite(ref_flat).all())
            skipped = not finite or not (gain > 0 and ref_gain > 0)
            rho = self.coefficient*V/max(float(op.gradient.square().sum()),1e-30)
            diag = dict(site='__joint__', sites=sites, rank=sum(e['rank'] for e in self.cache.values()),
                correction_solver='cg_shared', parameter_scope='full_trainable', cg_tol=1e-3,cg_max_iter=50,
                mu=mu, trace_F=trace, rho=rho,V=V,raw_gain=raw_gain,realized_gain=gain,
                vna_raw_gain=vna_raw_gain,vna_realized_gain=ref_gain,coefficient=self.coefficient,
                vna_norm=norm(reference),correction_norm=norm(delta),cg=records,
                G1=any(e['G1'] for e in self.cache.values()),G2=rho>.3,G3=self.controller.get('G3'),
                state='active',bn_mode='eval',skipped=skipped,
                skip_reason='nonfinite' if not finite else 'nonpositive_gain' if skipped else None)
        return ({} if skipped else delta),diag

    def inverse(self, op, refs, site, rhs, mu, trace, key=None, records=None):
        if self.args.solver == 'kfac': return self.factors[site].apply(rhs, refs, mu)
        if self.args.solver == 'lite': return rhs / max(trace/op.g.numel(), 1e-15)
        result = conjugate_gradient(lambda v: op.F(v)+mu*v, rhs.double(),
            x0=self.warm.get(site, {}).get(key) if key else None,
            max_iter=self.cfg.cg_iters, tolerance=self.cfg.cg_tol, eps=1e-30)
        if key and torch.isfinite(result.solution).all():
            self.warm.setdefault(site, {})[key] = result.solution.detach().clone()
        if records is not None:
            records.append({'iterations': result.iterations, 'converged': result.converged,
                            'relative_residual': result.relative_residual})
        return result.solution

    def cotangent(self, model, site, A, batch, trace, mu):
        with block_products(model, site, A, batch[0], cache_block_input=self.cfg.cache_block_input) as p:
            op=GGNProducts(p,batch[1])
            u=self.inverse(op,p.parameter_refs,site,op.gradient,mu,trace)
            return op.H(p.Jv(u)).reshape_as(op.probs).detach()

    @preserve_buffers_and_modes
    def refresh(self, model, dataset, indices, epoch):
        sync(self.device); start=time.perf_counter()
        batches=[augmented_batch(dataset, indices, self.args.stat_batch,
                  self.generators['statistics'], self.device) for _ in range(self.args.stat_batches)]
        statistics=(torch.cat([b[0] for b in batches]),torch.cat([b[1] for b in batches]))
        measurement=augmented_batch(dataset,indices,self.args.vna_batch,self.generators['statistics'],self.device)
        names=[b.name for b in model.growing_blocks()]
        if self.args.sites != 'auto':
            names=self.args.sites.split(',')
            if not set(names) <= {b.name for b in model.growing_blocks()}: raise ValueError('unknown site')
        self.warm={}
        entries={}; components={'factors':0.,'selection':0.,'trace':0.,'gate_statistics':0.}
        modes={m:m.training for m in model.modules()}; model.eval()
        try:
            # Upstream statistics / virtual layer construction have private global RNG.
            with isolated_rng(self.generators['construction']):
                for site in names:
                    sync(self.device); t=time.perf_counter()
                    candidate=tiny_candidate(model,site,statistics,self.args.rank,seed=self.seed('selection'))
                    A,_=extract_tiny_A(model,candidate); A=normalize_A(model,site,A)
                    sync(self.device); components['selection']+=time.perf_counter()-t
                    t=time.perf_counter()
                    if self.args.solver=='kfac':
                        f=self.factors.setdefault(site,KFACBlockPreconditioner())
                        f.refresh(model,site,measurement)
                    sync(self.device); components['factors']+=time.perf_counter()-t
                    t=time.perf_counter()
                    from dataclasses import replace
                    with block_products(model,site,A,measurement[0]) as p:
                        op=GGNProducts(p,measurement[1]); trace=op.trace(replace(self.cfg,seed=self.seed('hutchinson')))
                        mu=relative_damping(self.cfg, trace, op)
                    sync(self.device); components['trace']+=time.perf_counter()-t
                    t=time.perf_counter()
                    zs=[self.cotangent(model,site,A,b,trace,mu) for b in batches]
                    selection_stats={}
                    if self.args.selection!='tiny_U':
                        A,selection_stats=tiny_V(model,site,batches,self.args.rank,cotangents=zs,cfg=self.cfg)
                    z=self.cotangent(model,site,A,measurement,trace,mu)
                    if self.args.selection=='greedy_V':
                        A,selection_stats=greedy_V(model,site,A,measurement[0],z,
                            r_max=self.args.rank,seed=self.seed('selection'),theta=self.args.dorfler_theta)
                    V=score_A(model,site,A,measurement[0],z)
                    if self.args.arm=='vna_random':
                        A,random_stats=trace_matched_random_A(model,site,A,measurement[0],seed=self.seed('random_A'))
                    else: random_stats=None
                    sync(self.device); components['selection']+=time.perf_counter()-t
                    # G1 is native TINY U-spectrum significance, diagnostic only.
                    t=time.perf_counter()
                    eigen=torch.as_tensor(candidate.payload['tiny_eigenvalues'],dtype=torch.float64).flatten()
                    boot=[]
                    x,y=statistics
                    for xb,yb in zip(x.chunk(self.args.gate_boot),y.chunk(self.args.gate_boot)):
                        c=tiny_candidate(model,site,(xb,yb),self.args.rank,seed=self.seed('selection'))
                        e=torch.as_tensor(c.payload['tiny_eigenvalues']); boot.append(float(e.max()) if e.numel() else 0.)
                    noise=float(torch.tensor(boot,dtype=torch.float64).mean()+2*torch.tensor(boot,dtype=torch.float64).std(unbiased=False))
                    sync(self.device); components['gate_statistics']+=time.perf_counter()-t
                    entries[site]={'A':A,'rank':len(A['weight']),'V':V,'trace_F':trace,'mu':mu,
                        'lambda_1':float(eigen.max()) if eigen.numel() else 0.,'noise':noise,
                        'G1':bool(eigen.numel() and float(eigen.max())>noise), 'random_trace':random_stats}
        finally:
            for m,mode in modes.items(): m.training=mode
        ordered=sorted(entries,key=lambda s:entries[s]['V'],reverse=True)
        if self.args.sites=='auto':
            selected=[]; total=sum(e['V'] for e in entries.values()); captured=0.
            for s in ordered[:self.args.max_sites]:
                selected.append(s); captured+=entries[s]['V']
                if captured>=self.args.dorfler_theta*total: break
        else: selected=names
        assert_disjoint_scopes(model,selected)
        self.cache={s:entries[s] for s in selected}
        self.factors={s:f for s,f in self.factors.items() if s in selected}
        self.controller.update(refresh_epoch=epoch,factor_refresh_epoch=epoch,refreshes=self.controller['refreshes']+1,
            candidates={s:{k:v for k,v in e.items() if k!='A'} for s,e in entries.items()})
        sync(self.device); components['refresh']=time.perf_counter()-start
        return components

    @preserve_buffers_and_modes
    def refresh_factors(self, model, dataset, indices, epoch):
        """Update curvature without changing selected sites or A."""
        sync(self.device); start=time.perf_counter()
        batch=augmented_batch(dataset,indices,self.args.vna_batch,self.generators['statistics'],self.device)
        self.warm={}
        with isolated_rng(self.generators['construction']):
            for site,entry in self.cache.items():
                if self.args.solver=='kfac': self.factors[site].refresh(model,site,batch)
                from dataclasses import replace
                with block_products(model,site,entry['A'],batch[0],cache_block_input=self.cfg.cache_block_input) as p:
                    op=GGNProducts(p,batch[1])
                    trace=op.trace(replace(self.cfg,seed=self.seed('hutchinson')))
                    entry.update(trace_F=trace,mu=relative_damping(self.cfg, trace, op))
        self.controller['factor_refresh_epoch']=epoch
        sync(self.device)
        return {'factor_refresh':time.perf_counter()-start}

    @preserve_buffers_and_modes
    def correction(self, model, site, batch):
        entry=self.cache[site]; A=entry['A']; trace=entry['trace_F']; mu=entry['mu']
        records=[]
        with block_products(model,site,A,batch[0],cache_block_input=self.cfg.cache_block_input) as p:
            op=GGNProducts(p,batch[1]); refs=p.parameter_refs
            u=self.inverse(op,refs,site,op.gradient,mu,trace,'u',records)
            z=op.H(p.Jv(u)); V=float(p.JET(z).double().square().sum())
            d=self.inverse(op,refs,site,op.COE(op.CEO(u)),mu,trace,'vna',records)
            vna_raw_gain=float(op.gradient.dot(d))
            reference=project_bn_tangent(model,site,_unflatten(d,refs),self.cfg.project_bn)
            delta=reference
            if self.args.arm=='ngd':
                d=self.inverse(op,refs,site,op.F(u),mu,trace,'ngd',records)
                delta=project_bn_tangent(model,site,_unflatten(d,refs),self.cfg.project_bn)
                delta,scale=norm_match(delta,reference)
            raw_gain=float(op.gradient.dot(d))
            if self.args.arm=='ngd': raw_gain*=scale
            flat=torch.cat([delta[n].flatten() for n,_ in refs]).double()
            gain=float(op.gradient.dot(flat)); ref_flat=torch.cat([reference[n].flatten() for n,_ in refs]).double()
            ref_gain=float(op.gradient.dot(ref_flat))
            rho=self.coefficient*V/max(float(op.gradient.square().sum()),1e-30)
            finite=bool(torch.isfinite(flat).all() and torch.isfinite(ref_flat).all())
            skipped=not finite or not (gain>0 and ref_gain>0)
            diag={'site':site,'rank':entry['rank'],'rho':rho,'V':V,'raw_gain':raw_gain,'realized_gain':gain,'vna_raw_gain':vna_raw_gain,'coefficient':self.coefficient,
                'vna_realized_gain':ref_gain,'vna_norm':norm(reference),'correction_norm':norm(delta),
                'cg':records,'G1':entry['G1'],'G2':rho>.3,'G3':self.controller.get('G3'),'state':'active','bn_mode':'eval',
                'skipped':skipped,'skip_reason':'nonfinite' if not finite else 'nonpositive_gain' if skipped else None}
        return ({} if skipped else delta),diag
