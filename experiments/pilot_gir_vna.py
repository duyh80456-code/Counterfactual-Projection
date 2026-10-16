"""Shared exact training interventions. All stochastic draws use private generators."""
from contextlib import contextmanager, redirect_stdout
import io
import time
import torch
from diagnostics.vna_align import _global_state
from diagnostics.measure_efficiency import SolveCache
from methods.vna import dedicated_generator, random_A_like
from methods.vna_ggn import GGNConfig, ngd_control
from methods.vna_joint import joint_correction
from methods.vna_select import tiny_V, neuron_norms, prepare_A


@contextmanager
def inference_state(model):
    modes = {m: m.training for m in model.modules()}
    with _global_state(), redirect_stdout(io.StringIO()):
        try:
            model.eval()
            yield
        finally:
            for module, mode in modes.items(): module.training = mode


def scale_rows(a, norm):
    scales = norm / neuron_norms(a).clamp_min(1e-30)
    return {k: None if v is None else v * scales.reshape((-1,) + (1,) * (v.ndim-1)) for k, v in a.items()}


def slice_rows(a, start, stop=None):
    return {k: None if v is None else v[start:stop].detach().clone() for k, v in a.items()}


def exact_cache(model, sites, batch, spec, ctx, cfg):
    cache = SolveCache(model, sites[0], batch, spec, ctx, cfg, solver='auto', trace=0., mu=1e-12)
    factor = cache.factor(batch, ctx, prepare=False)
    # tr(R J J^T R) = tr(F), available exactly from the constructed matrix.
    cache.trace = float(factor.matrix.diagonal().sum())
    cache.mu = max(cfg.damping_rel * cache.trace / factor.size, 1e-12)
    cache.ctx['mu'] = cache.mu
    factor.prepare(cache.mu)
    if factor.factor_error: raise RuntimeError(factor.factor_error)
    return cache, factor


class PilotEngine:
    def __init__(self, *, seed=0, gir=False, random=False, upstream_random=False,
                 coefficient=0., ngd=False, every=100, pool=8, refresh=1000,
                 rho_min=0., rank=1, selection='tiny_V'):
        if every < 1 or pool < 1 or refresh < 1: raise ValueError('pilot intervals/pool must be positive')
        self.seed, self.gir, self.random, self.upstream_random = seed, gir, random, upstream_random
        self.coefficient, self.ngd, self.every = coefficient, ngd, every
        self.pool_size, self.refresh, self.rho_min, self.rank = pool, refresh, rho_min, rank
        if selection not in ('tiny_V', 'tiny_V_euclid', 'random_frob'):
            raise ValueError('unknown selection: '+selection)
        self.selection = selection
        self.pools, self.positions, self.pool_steps = {}, {}, {}
        self.events, self.activities, self.generators = [], [], {}
        self.task = None

    def state_dict(self):
        return {k: v for k, v in vars(self).items() if k != 'generators'} | {
            'generators': {k: g.get_state() for k, g in self.generators.items()}}

    def load_state_dict(self, state):
        for k, v in state.items():
            if k != 'generators': setattr(self, k, v)
        self.generators = {}
        for device, value in state['generators'].items():
            g = dedicated_generator(self.seed, 'pilot-interventions', torch.device(device))
            g.set_state(value.cpu()); self.generators[device] = g

    def generator(self, device):
        key = str(device)
        if key not in self.generators:
            self.generators[key] = dedicated_generator(self.seed, 'pilot-interventions', device)
        return self.generators[key]

    def begin_task(self, task):
        if task != self.task:
            self.task = task
            self.pools.clear(); self.positions.clear(); self.pool_steps.clear()

    @property
    def cfg(self):
        return GGNConfig(seed=self.seed, damping_rel=.01, damping_norm='outputs', project_bn=False)

    def refresh_pool(self, model, sites, batch, spec, ctx, step):
        started = time.monotonic()
        with inference_state(model):
            cache, factor = exact_cache(model, sites, batch, spec, ctx, self.cfg)
            with torch.no_grad(): e = spec.residual(model(batch[0]), ctx)
            z = factor.z(e, cache.mu)
            for site in sites:
                a, capture = self.select(model, site, batch, z, self.pool_size)
                positive = int((capture['generalized_eigenvalues'][:len(a['weight'])] > 0).sum())
                key = site.module_name
                self.pools[key] = slice_rows(a, 0, positive)
                self.positions[key], self.pool_steps[key] = 0, step
        self.events.append(dict(kind='gir_refresh', task=self.task, step=step, seconds=time.monotonic()-started,
                                directions={k: len(a['weight']) for k, a in self.pools.items()}))

    def select(self, model, site, batch, z, rank):
        kind = 'tiny_V_euclid' if self.selection == 'tiny_V_euclid' else 'tiny_V'
        a, capture = tiny_V(model, site, [batch], rank, cotangents=[z], cfg=self.cfg,
                            selection=kind)
        if self.selection == 'random_frob':
            random = random_A_like(a, generator=self.generator(a['weight'].device))
            a, _ = prepare_A(model, site, random, 'random_frob', reference=a)
        return a, capture

    def take(self, model, site, batch, spec, ctx, count, norm, step):
        key = site.module_name
        if (key not in self.pools or self.positions[key] >= len(self.pools[key]['weight'])
                or step-self.pool_steps[key] >= self.refresh):
            self.refresh_pool(model, [site], batch, spec, ctx, step)
        position = self.positions[key]
        a = slice_rows(self.pools[key], position, position+count)
        used = len(a['weight']); self.positions[key] += used
        if self.random and used:
            a = {k: None if v is None else torch.randn(v.shape, device=v.device, dtype=v.dtype,
                    generator=self.generator(v.device)) for k, v in a.items()}
            if a['bias'] is not None: a['bias'].zero_()
        self.events.append(dict(kind='gir_replace', task=self.task, step=step, site=key,
                                replaced=count, directions_used=used, gir_random_fill=count-used))
        return scale_rows(a, norm)

    def correction(self, model, sites, batch, spec, ctx, step, lr):
        if self.coefficient == 0 or step % self.every: return None
        started = time.monotonic()
        with inference_state(model):
            cache, factor = exact_cache(model, sites, batch, spec, ctx, self.cfg)
            with torch.no_grad(): e = spec.residual(model(batch[0]), ctx)
            z = factor.z(e, cache.mu)
            proposals = {}
            for site in sites:
                m = self.positions.get(site.module_name, 0) if self.gir else 0
                a, _ = self.select(model, site, batch, z, m+self.rank)
                proposals[site] = slice_rows(a, m, m+self.rank)
            if not any(len(a['weight']) for a in proposals.values()):
                row = dict(kind='correction', task=self.task, step=step, K=self.every,
                           selection=self.selection, rho=0., skipped=True,
                           reason='no unreserved positive directions', seconds=time.monotonic()-started)
                self.events.append(row); return row
            proposals = {s: a for s, a in proposals.items() if len(a['weight'])}
            active = list(proposals)
            delta, _, rho, record = joint_correction(model, active, proposals, batch, self.cfg,
                mu=cache.mu, trace_F=cache.trace, loss_spec=spec, ctx=ctx, output_factor=factor)
            vna_norm = sum(float(v.double().square().sum()) for v in delta.values())**.5
            if self.ngd:
                delta, ngd_record = ngd_control(model, active[0], proposals[active[0]], batch, self.cfg,
                    reference=delta, norm_match=True, output_factor=factor,
                    mu=cache.mu, trace_F=cache.trace, loss_spec=spec, ctx=ctx)
                record['realized_gain'] = ngd_record['realized_gain']
            skipped = rho < self.rho_min or not record['finite']
            if not skipped:
                refs = dict(model.named_parameters())
                with torch.no_grad():
                    for name, value in delta.items(): refs[name].add_(value.to(refs[name]), alpha=-lr*self.coefficient)
        row = dict(kind='correction', task=self.task, step=step, K=self.every,
                   selection=self.selection, rho=rho, skipped=skipped,
                   rho_gate_skip=rho < self.rho_min, vna_norm=vna_norm,
                   correction_norm=sum(float(v.double().square().sum()) for v in delta.values())**.5,
                   gradient_dot_direction=record['realized_gain'], seconds=time.monotonic()-started,
                   **factor.metadata())
        self.events.append(row)
        return row


def choose_interval(points, iterations_per_epoch):
    """Pre-outcome budget: linear n_out extrapolation to task 10, t_VNA/K <= t_iter."""
    projected = max(p['vna_seconds'] * 12800/p['n_out'] for p in points)
    iteration = sum(p['iteration_seconds'] for p in points)/len(points)
    choices = sorted(set([50, 100, 200, 400, 800, iterations_per_epoch]))
    feasible = [k for k in choices if projected/k <= iteration]
    if not feasible: raise ValueError('no registered K meets the correction budget')
    return dict(K=min(feasible), every_epoch=min(feasible) not in (50,100,200,400,800), projected_vna_seconds=projected, iteration_seconds=iteration,
                iterations_per_epoch=iterations_per_epoch, smoke_points=points,
                rule='max(t_VNA * 12800/n_out) / K <= mean(t_iter)')
