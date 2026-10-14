"""Stage A: read-only checkpoint diagnostics, schema-isolated and resumable."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from functools import wraps

import torch
from experiments.shared_protocol import atomic_json_save
from experiments.vna_provenance import code_commit
from experiments.vna_support import build_model, setup_data, augmented_batch, isolated_rng, tiny_candidate
from methods.vna import dedicated_generator, _mean_ce_logit_grad, extract_tiny_A, VNAConfig, vna_block_correction, _resolve_site
from methods.vna_ggn import GGNConfig, filtered_cotangent
from methods.vna_joint import joint_correction, joint_products
from methods.vna_select import (tiny_U_from_cotangent, tiny_V, greedy_V, prepare_A,
                                subspace_sim, random_A_like)

SCHEMA_VERSION = 'stage1d-v1'
DAMPING_FIELDS = ('mu', 'mu_rel', 'damping_norm', 'trace_F', 'n_out', 'p', 'mu_rel_equiv_params')
MU_GRID = (1e-1, 1e-2, 1e-3, 1e-4)
MU_RATIO = 3.
MU_FRACTION = .5
EPOCHS = {'null': tuple(range(20, 201, 20)), 'mu': (40, 80, 100, 120, 140, 160),
          'selection': (40, 80, 120, 140, 160), 'joint': (100, 120, 140, 160),
          'solver': (40, 100, 160), 'a5': (90, 100, 110)}
FILES = {'null': 'null.csv', 'mu': 'mu_sweep.csv', 'selection': 'selection.csv',
         'joint': 'joint.csv', 'solver': 'solver.csv', 'a5': 'a5.csv'}
SELECTION_KINDS = ('tiny_U', 'tiny_V', 'greedy_V', 'random_trace', 'random_frob')


@contextmanager
def invariant(model):
    """Assert bitwise parameters, all registered buffers, modes, grads and RNG."""
    before = {n: t.detach().clone() for n, t in model.state_dict().items()}
    buffers = {n: t.detach().clone() for n, t in model.named_buffers()}
    modes = {m: m.training for m in model.modules()}
    grads = {p: None if p.grad is None else p.grad.detach().clone() for p in model.parameters()}
    count = sum(p.numel() for p in model.parameters())
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        yield
    finally:
        after = model.state_dict()
        assert before.keys() == after.keys(), 'measurement changed state structure'
        assert all(torch.equal(t, after[n]) for n, t in before.items()), 'measurement mutated state'
        current = dict(model.named_buffers())
        assert buffers.keys() == current.keys()
        assert all(torch.equal(t, current[n]) for n, t in buffers.items())
        assert all(m.training == mode for m, mode in modes.items()), 'measurement changed modes'
        assert sum(p.numel() for p in model.parameters()) == count
        assert all((p.grad is None if g is None else p.grad is not None and torch.equal(g, p.grad))
                   for p, g in grads.items()), 'measurement changed gradients'
        assert torch.equal(rng, torch.get_rng_state()), 'measurement changed RNG'
        assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))


def residual(model, batch):
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()
        with torch.no_grad():
            logits = model(batch[0])
        return _mean_ce_logit_grad(logits, batch[1]).reshape_as(logits)
    finally:
        for m, mode in modes.items():
            m.training = mode


def measurement_entry(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with invariant(self.model):
            return method(self, *args, **kwargs)
    return call


def null_residual(g, kind, generator):
    """N1 preserves the residual distribution; N2 preserves each sample norm.

    N0 needs labels/logits and is computed separately by measure_null.
    """
    if kind == 'N1':
        return g[torch.randperm(len(g), generator=generator, device=g.device)].clone()
    if kind == 'N2':
        noise = torch.randn(g.shape, dtype=g.dtype, device=g.device, generator=generator)
        noise -= noise.mean(-1, keepdim=True)
        return noise * (g.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True).clamp_min(1e-30))
    raise ValueError('null_residual supports N1/N2; N0 requires labels')


def empirical_p(real, nulls):
    nulls = list(nulls)
    if not nulls or not all(math.isfinite(float(v)) for v in [real, *nulls]):
        raise ValueError('p-value requires finite real and nonempty nulls')
    return (1 + sum(v >= real for v in nulls)) / (1 + len(nulls))


def diagnostic_ratio(tiny, random):
    """Do not turn missing/nonfinite controls or guarded directions into zeros."""
    if not all(math.isfinite(float(v)) for v in (tiny, random)) or random <= 0:
        return None
    return tiny / random


def metadata(args, seconds):
    return {'schema_version': SCHEMA_VERSION, 'code_commit': code_commit(),
            'seed': args.seed, 'device': args.device, 'wall_seconds': seconds}


def load_json(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'incompatible schema_version in {path}')
    return data


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(f'incompatible schema_version in {path}')
        for key, value in list(row.items()):
            if value in ('True', 'False'):
                row[key] = value == 'True'
            elif value == '':
                row[key] = None
            elif key not in ('code_commit', 'schema_version', 'site', 'device', 'kind', 'null_kind', 'stat', 'variant'):
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    tmp = Path(path).with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)
    tmp.replace(path)


class Measurement:
    def __init__(self, model, datasets, args, epoch):
        self.model, self.args, self.epoch = model, args, epoch
        self.train, self.evaluation, self.indices, trigger, *_ = datasets
        self.device = torch.device(args.device)
        self.sites = [r.name for r in model.growing_blocks()]
        self.cfg = GGNConfig(seed=args.seed, damping_norm="outputs", cache_block_input=True, cg_iters=1000, cg_tol=1e-6)
        if args.cost_smoke:
            self.sites = self.sites[:1]
        if args.smoke:
            self.cfg = replace(self.cfg, cg_iters=30, cg_tol=1e-4)
        self.rank = 1 if args.smoke else 16
        self.stat_size, self.batch_size = (4, 1) if args.smoke else (256, 64)
        # Stat/correction pools are disjoint by dataset index, not just draws.
        order = torch.randperm(len(self.indices), generator=dedicated_generator(args.seed, 'stage-a-pools')).tolist()
        split = len(order) // 2
        self.stat_pool = [self.indices[i] for i in order[:split]]
        self.correction_pool = [self.indices[i] for i in order[split:]]
        self.check = augmented_batch(self.evaluation, trigger, 2 if args.smoke else 512,
            dedicated_generator(args.seed, 'heldout'), self.device)
        self._top = None

    def draw(self, purpose, count, *, stat=False):
        return augmented_batch(self.train, self.stat_pool if stat else self.correction_pool,
            count, dedicated_generator(self.args.seed, f'stage-a:{self.epoch}:{purpose}'), self.device)

    def select_u(self, site, batch, rank, g=None):
        g = residual(self.model, batch) if g is None else g
        A, values = tiny_U_from_cotangent(self.model, site, batch[0], g, rank, self.cfg)
        A, norm = prepare_A(self.model, site, A, 'tiny_U')
        return A, values, norm

    def top_sites(self):
        if self._top is None:
            # Keep Phase-0 named selection streams and selection statistic;
            # never select sites using any reported correction/heldout batch.
            stat = augmented_batch(self.train, self.indices, self.stat_size,
                dedicated_generator(self.args.seed, 'rank-selection-statistics'), self.device)
            batch = augmented_batch(self.train, self.indices, self.batch_size,
                dedicated_generator(self.args.seed, 'rank-selection'), self.device)
            clone = copy.deepcopy(self.model)
            scores = {}
            for site in self.sites:
                candidate = tiny_candidate(clone, site, stat, self.rank, self.args.seed)
                A, _ = extract_tiny_A(clone, candidate)
                A, _ = prepare_A(clone, site, A, 'tiny_U')
                _, diag = vna_block_correction(clone, site, A, batch,
                    VNAConfig(seed=self.args.seed, cache_block_input=True), apply_gain_guard=False)
                if not diag['nonfinite'] and math.isfinite(diag['V']):
                    scores[site] = diag['V']
            self._top = sorted(scores, key=lambda s: (-scores[s], s))[:3]
        return self._top[:1] if self.args.cost_smoke else self._top

    def correction(self, sites, A, batch, *, mu_rel=1e-2, cotangent=None, mu=None):
        d, V, rho, diag = joint_correction(self.model, sites, A, batch,
            replace(self.cfg, damping_rel=mu_rel), cotangent=cotangent, mu=mu,
            o_params=_resolve_site(self.model, sites[0])[2] if len(sites) == 1 else None)
        return d, diag

    def derivative(self, sites, A, d):
        with joint_products(self.model, sites, A, self.check[0], list(d)) as p:
            g = _mean_ce_logit_grad(p._call({}).reshape(len(self.check[1]), -1), self.check[1])
            flat = torch.cat([d[n].flatten() for n, _ in p.parameter_refs])
            return -float(g.dot(p.Jv(flat).double()))

    @measurement_entry
    def measure_null(self):
        rows = []
        stat = self.draw('null-stat', self.stat_size, stat=True)
        batch = self.draw('null-correction', self.batch_size)
        gs, gc = residual(self.model, stat), residual(self.model, batch)
        for site in self.sites:
            real_A, eigen, real_norm = self.select_u(site, stat, self.rank, gs)
            _, real_diag = self.correction([site], {site: real_A}, batch, cotangent=gc)
            real = {'lambda1': float(eigen[0]), 'rho': real_diag['rho']}
            for kind in ('N0', 'N1', 'N2'):
                values = {k: [] for k in real}
                normalized = []
                convergence = []
                normalizations = []
                for replicate in range(1 if self.args.cost_smoke else 8):
                    purpose = f'null:{site}:{kind}:{replicate}'
                    gen = dedicated_generator(self.args.seed, f'{self.epoch}:{purpose}:stat', self.device)
                    cgen = dedicated_generator(self.args.seed, f'{self.epoch}:{purpose}:correction', self.device)
                    if kind == 'N0':
                        ys = stat[1][torch.randperm(len(gs), generator=gen, device=self.device)]
                        yc = batch[1][torch.randperm(len(gc), generator=cgen, device=self.device)]
                        ns = residual(self.model, (stat[0], ys))
                        nc = residual(self.model, (batch[0], yc))
                    else:
                        ns, nc = null_residual(gs, kind, gen), null_residual(gc, kind, cgen)
                    A, eig, normalization = self.select_u(site, stat, self.rank, ns)
                    normalizations.append(normalization)
                    _, diag = self.correction([site], {site: A}, batch, cotangent=nc, mu=real_diag['mu'])
                    values['lambda1'].append(float(eig[0]))
                    values['rho'].append(diag['rho'])
                    normalized.append(float(eig[0]) / float(ns.square().sum()))
                    convergence.append(diag['cg_converged'])
                for key in real:
                    p = empirical_p(real[key], values[key])
                    rows.append({'site': site, 'null_kind': kind, 'stat': key, 'real': real[key],
                        'null_mean': statistics.mean(values[key]), 'null_max': max(values[key]),
                        'null_values': values[key], 'p_value': p, 'significant': p <= 1/9,
                        'normalization_constant': real_norm['normalization_constant'],
                        'null_normalization': normalizations,
                        'cg_converged': real_diag['cg_converged'] and all(convergence),
                        'real_lambda_over_g_norm_sq': float(eigen[0]) / float(gs.square().sum()),
                        'null_lambda_over_g_norm_sq_mean': statistics.mean(normalized),
                        **{k: real_diag[k] for k in DAMPING_FIELDS}, 'rank': self.rank})
        return rows

    def joint_A(self, purpose):
        stat = self.draw(purpose, self.stat_size, stat=True)
        selected = {site: self.select_u(site, stat, self.rank) for site in self.sites}
        self.normalizations = {site: result[2] for site, result in selected.items()}
        return {site: result[0] for site, result in selected.items()}

    @measurement_entry
    def measure_mu(self):
        if self.args.cost_smoke:
            self.sites = [r.name for r in self.model.growing_blocks()]
        rows = []
        A = self.joint_A('mu-stat')
        batch = self.draw('mu-correction', self.batch_size)
        for site in [*self.top_sites(), 'joint']:
            sites = self.sites if site == 'joint' else [site]
            for mu_rel in MU_GRID:
                d, diag = self.correction(sites, A, batch, mu_rel=mu_rel)
                rows.append({'site': site, 'mu_rel': mu_rel, **diag,
                    'normalization_constant': {s: self.normalizations[s] for s in sites},
                    'heldout_derivative': self.derivative(sites, A, d), 'rank': self.rank})
        return rows

    def select(self, site, kind, batch, rank, purpose):
        au, _, norm = self.select_u(site, batch, rank)
        if kind == 'tiny_U':
            return au, norm
        batches = list(zip(batch[0].split(64), batch[1].split(64)))
        z = [filtered_cotangent(self.model, site, au, b, self.cfg) for b in batches]
        selection_converged = True
        if kind in ('tiny_V', 'greedy_V'):
            raw, _ = tiny_V(self.model, site, batches, rank, cotangents=z, cfg=self.cfg, normalize=False)
            if kind == 'greedy_V':
                raw, selection_record = greedy_V(self.model, site, raw, batch[0], torch.cat(z),
                    r_max=rank, theta=1., seed=dedicated_generator(self.args.seed,
                        f'{self.epoch}:{purpose}:{site}:greedy-pool').initial_seed(),
                    matrix_free=not self.args.smoke,
                    random_count=2 if self.args.smoke else 8)
                selection_converged = selection_record.get('cg_converged', True)
        else:
            raw = random_A_like(au, generator=dedicated_generator(self.args.seed,
                f'{self.epoch}:{purpose}:{site}:{kind}:{rank}', self.device))
        prepared, normalization = prepare_A(self.model, site, raw, kind, reference=au, inputs=batch[0], trace_estimator='auto',
            seed=dedicated_generator(self.args.seed, f'{self.epoch}:{purpose}:{site}:trace-crn').initial_seed())
        normalization['selection_cg_converged'] = selection_converged
        return prepared, normalization

    @measurement_entry
    def measure_selection(self):
        rows = []
        ranks = (1,) if self.args.smoke else ((64,) if self.args.cost_smoke else (4, 16, 64))
        batch = self.draw('selection-correction', self.batch_size)
        for site in self.top_sites():
            for rank in ranks:
                for kind in SELECTION_KINDS:
                    samples = []
                    for draw in range(1 if self.args.cost_smoke else (2 if self.args.smoke else 3)):
                        stat = self.draw(f'selection-stat:{draw}', self.stat_size, stat=True)
                        A, norm = self.select(site, kind, stat, rank, f'selection:{draw}')
                        d, diag = self.correction([site], {site: A}, batch)
                        halves = [(x, y) for x, y in zip(stat[0].chunk(2), stat[1].chunk(2))]
                        split = [self.select(site, kind, h, rank, f'split:{draw}:{i}')[0] for i, h in enumerate(halves)]
                        d_in = A['weight'][0].numel() + int(A['bias'] is not None)
                        samples.append({**{k: diag[k] for k in DAMPING_FIELDS}, 'trace_record': norm,
                            'trace_estimator': norm.get('trace_estimator'),
                            'trace_std': norm.get('trace_target_record', {}).get('std'), 'V': diag['V'], 'rho': diag['rho'],
                            'heldout_derivative': self.derivative([site], {site: A}, d),
                            'subspace_sim': subspace_sim(*split),
                            'random_expectation': min(len(A['weight']), d_in) / d_in,
                            'effective_rank': len(A['weight']), 'cg_converged': diag['cg_converged'] and norm.get('selection_cg_converged', True),
                            'normalization_constant': norm['normalization_constant'],
                            'match_scale': norm['match_scale']})
                    row = {**{k: [sample[k] for sample in samples] for k in DAMPING_FIELDS},
                           'mu_rel': self.cfg.damping_rel, 'damping_norm': self.cfg.damping_norm,
                           'site': site, 'kind': kind, 'rank': rank, 'draws': len(samples),
                           'cg_converged': all(s['cg_converged'] for s in samples),
                           'normalization_constant': [s['normalization_constant'] for s in samples],
                           'match_scale': [s['match_scale'] for s in samples],
                           'trace_estimator': [s['trace_estimator'] for s in samples],
                           'trace_std': [s['trace_std'] for s in samples], 'samples': samples}
                    for key in ('V', 'rho', 'heldout_derivative', 'subspace_sim', 'random_expectation', 'effective_rank'):
                        row[key] = statistics.mean(s[key] for s in samples)
                        row[key + '_std'] = statistics.stdev(s[key] for s in samples) if len(samples) > 1 else 0.
                    rows.append(row)
        return rows

    @measurement_entry
    def measure_joint(self):
        if self.args.cost_smoke:
            self.sites = [r.name for r in self.model.growing_blocks()]
        A = self.joint_A('joint-stat')
        batch = self.draw('joint-correction', self.batch_size)
        rows = []
        for mu_rel in (1e-2, 1e-3):
            _, diag = self.correction(self.sites, A, batch, mu_rel=mu_rel)
            single = [self.correction([site], A, batch, mu_rel=mu_rel)[1] for site in self.sites]
            rows.append({'site': 'joint', 'mu_rel': mu_rel, 'rank': self.rank,
                'rho_joint': diag['rho'], 'sum_rho_site': sum(d['rho'] for d in single),
                'max_rho_site': max(d['rho'] for d in single), **diag,
                'normalization_constant': self.normalizations,
                'cg_converged': diag['cg_converged'] and all(d['cg_converged'] for d in single)})
        return rows

    @measurement_entry
    def measure_a5(self, checkpoint):
        # Match signal_curve.compute defaults, including unnormalised native A,
        # Phase-0 batches/purposes and asymmetric gain guards.
        from diagnostics.signal_curve import serial_diag
        from methods.vna import random_A_like as phase0_random_A
        cfg = VNAConfig(seed=self.args.seed, cache_block_input=not self.args.smoke)
        if self.args.smoke:
            cfg = replace(cfg, stat_samples=2, vna_batch=1, r_max=1, cg_iters=3, hutchinson_probes=1)
        protocol = checkpoint.get('protocol', {})
        model = copy.deepcopy(self.model)
        train = self.train
        if protocol.get('gpu_data', False) and not self.args.fake_data:
            from experiments.vna_support import DeviceCIFAR
            train = DeviceCIFAR(train, self.device)
        if protocol.get('channels_last', False):
            model.to(memory_format=torch.channels_last)
        stat = augmented_batch(train, self.indices, cfg.stat_samples,
            dedicated_generator(self.args.seed, 'statistics'), self.device)
        batch = augmented_batch(train, self.indices, cfg.vna_batch,
            dedicated_generator(self.args.seed, 'vna'), self.device)
        if protocol.get('channels_last', False):
            stat = (stat[0].contiguous(memory_format=torch.channels_last), stat[1])
            batch = (batch[0].contiguous(memory_format=torch.channels_last), batch[1])
        modes = {m: m.training for m in model.modules()}
        try:
            model.eval()
            with torch.no_grad(): logits = model(batch[0]).detach()
        finally:
            for m, mode in modes.items(): m.training = mode
        g = _mean_ce_logit_grad(logits, batch[1])
        optimizer = checkpoint.get('optimizer', checkpoint.get('optimizer_state_dict', {}))
        scheduler = checkpoint.get('scheduler', checkpoint.get('scheduler_state_dict'))
        inferred = recipe_scheduler_state(checkpoint['epoch'])
        rows = []
        for site in self.sites:
            candidate = tiny_candidate(model, site, stat, cfg.r_max, self.args.seed)
            A, rank = extract_tiny_A(model, candidate)
            random_A = phase0_random_A(A, generator=dedicated_generator(
                self.args.seed, f'random-A:{site}:{cfg.r_max}', self.device))
            row = {'site': site, 'rank': rank, 'requested_rank': cfg.r_max,
                   'checkpoint_epoch': checkpoint['epoch'],
                   'checkpoint_lr': checkpoint.get('lr', [p.get('lr') for p in optimizer.get('param_groups', [])] or None),
                   'checkpoint_scheduler_state': scheduler,
                   'checkpoint_state_present': scheduler is not None and bool(optimizer),
                   'recipe_scheduler_state_reconstructed': inferred,
                   'recipe_lr_reconstructed': inferred['_last_lr'],
                   'phase0_gpu_data': protocol.get('gpu_data', False),
                   'phase0_channels_last': protocol.get('channels_last', False),
                   'phase0_cache_block_input': cfg.cache_block_input, 'g_norm': float(g.norm())}
            for prefix, fixed in (('old_', False), ('fixed_', True)):
                audit_cfg = cfg
                common = dict(shared_logits=logits, shared_g=g, cg_breakdown_flags=True, rhs_normalized_cg=fixed)
                _, tiny = vna_block_correction(model, site, A, batch, audit_cfg, **common)
                _, random = vna_block_correction(model, site, random_A, batch, audit_cfg,
                    damping_cache=tiny.get('damping_cache'), apply_gain_guard=False, **common)
                row[prefix + 'ratio'] = diagnostic_ratio(tiny.get('V', float('nan')), random.get('V', float('nan')))
                for label, diag in (('', tiny), ('random_', random)):
                    for key, value in serial_diag(diag).items():
                        row[prefix + label + key] = value
                    row[prefix + label + 'finite'] = not diag['nonfinite'] and math.isfinite(diag.get('V', float('nan')))
                    for solve in ('cg_z', 'cg_q'):
                        record = diag.get(solve) or {}
                        for key in ('iterations', 'converged', 'relative_residual'):
                            row[prefix + label + solve + '_' + key] = record.get(key)
                        row[prefix + label + solve + '_breakdown'] = record.get('breakdown', False)
            with joint_products(model, [site], {site: A}, batch[0], _resolve_site(model, site)[2]) as products:
                p = sum(ref.numel() for _, ref in products.parameter_refs)
                from methods.vna_ggn import GGNProducts
                trace_F = GGNProducts(products, batch[1]).trace(
                    replace(self.cfg, hutchinson_probes=1) if self.args.smoke else self.cfg)
            row.update({key[4:]: value for key, value in list(row.items())
                        if key.startswith('old_') and key != 'old_site'})
            mu = row.get('old_mu')
            row.update(mu=mu, mu_rel=cfg.damping_rel, damping_norm='outputs',
                       trace_F=trace_F, n_out=g.numel(), p=p,
                       mu_rel_equiv_params=mu / (trace_F/p) if trace_F else None,
                       damping_metric='phase0_euclidean_trace_K')
            rows.append(row)
        return rows

    @measurement_entry
    def measure_solver(self, checkpoint, output):
        from diagnostics.vna_solver_validation import parser, run_checkpoint
        args = parser().parse_args(['--mode', 'recipe', '--checkpoint', str(checkpoint),
            '--data-root', self.args.data_root, '--output', str(output), '--device', self.args.device,
            '--seed', str(self.args.seed), *(['--fake-data'] if self.args.fake_data else [])])
        if self.args.cost_smoke:
            args.max_sites = 1
        # Shared training defaults are left intact, including batch 16/rank16.
        # CPU smoke trims only the fixture, not the production protocol.
        if self.args.smoke:
            args.rank, args.stat_batch, args.stat_batches = 1, 2, 2
            args.max_sites, args.gate_boot = 1, 1
            args.vna_batch, args.batch_size = 2, 4
            args.cg_iters, args.cg_tol = 30, 1e-4
        with isolated_rng(dedicated_generator(self.args.seed, f'stage-a:solver:{self.epoch}')):
            result = run_checkpoint(args, stage_metadata=metadata(self.args, 0.), require_convergence=False)
        checks = {c['site']: c for c in result['batch_checks']}
        rows = []
        for row in result['rows']:
            if row['relative_mu'] != args.damping_rel:
                continue
            check = checks.get(row['site'], {})
            rows.append({**row, 'cos_d16_d128': check.get('cos16'),
                'cos_disjoint_batch16': check.get('cos_disjoint16'),
                'overhead': len(result['sites']) * row['sgd_units'] / args.vna_every})
        return rows


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def merge_measurement(output, measurement):
    rows = []
    for path in sorted((output / 'completed' / measurement).glob('*.json')):
        rows.extend(load_json(path)['rows'])
    write_csv(output / FILES[measurement], rows)


def recipe_scheduler_state(epoch):
    # Periodic Phase-0 checkpoints omit optimizer/scheduler state. Preserve
    # missing raw state and separately label the deterministic reconstruction.
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=.1, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=.1)
    for _ in range(epoch):
        optimizer.step()
        scheduler.step()
    return scheduler.state_dict()


def a5_decision(rows):
    eligible = [r for r in rows if r['epoch'] == 100 and r.get('old_ratio') == 0
                and r.get('fixed_ratio') is not None and math.isfinite(r['fixed_ratio'])
                and r['fixed_ratio'] > 0]
    return 'resolved' if eligible else 'unresolved'


def summarize(args):
    output = Path(args.output)
    start = time.perf_counter()
    mu = read_csv(output / FILES['mu'])
    groups = {}
    for r in mu:
        if r['cg_converged']:
            groups.setdefault((r['epoch'], r['site']), {})[r['mu_rel']] = r['rho']
    pairs = [g for g in groups.values() if 1e-2 in g and 1e-3 in g]
    hits = sum(g[1e-3] >= MU_RATIO * g[1e-2] or
               math.isclose(g[1e-3], MU_RATIO * g[1e-2], rel_tol=1e-12, abs_tol=0.) for g in pairs)
    fraction = hits / len(pairs) if pairs else 0.
    selection = read_csv(output / FILES['selection'])
    fractions = {}
    for rank in sorted({r['rank'] for r in selection}):
        groups = {}
        for r in selection:
            if r['rank'] == rank and r['cg_converged']:
                groups.setdefault((r['epoch'], r['site']), {})[r['kind']] = r['V']
        eligible = [g for g in groups.values() if 'tiny_V' in g and 'random_frob' in g]
        wins = sum(g['tiny_V'] > g['random_frob'] for g in eligible)
        fractions[str(int(rank))] = {'wins': wins, 'points': len(eligible),
                                    'fraction': wins / len(eligible) if eligible else None}
    null = read_csv(output / FILES['null'])
    n1 = {}
    for stat in ('lambda1', 'rho'):
        eligible = [r for r in null if r['null_kind'] == 'N1' and r['stat'] == stat and r['cg_converged']]
        hits_n1 = sum(r['significant'] for r in eligible)
        n1[stat] = {'significant': hits_n1, 'points': len(eligible),
                    'fraction': hits_n1 / len(eligible) if eligible else None}
    a5 = read_csv(output / FILES['a5'])
    evidence = [dict(site=r['site'], epoch=r['epoch'],
        columns=[k for k in ('old_V', 'old_random_V', 'old_ratio', 'fixed_V', 'fixed_random_V',
                             'fixed_ratio', 'old_cg_z_breakdown', 'old_cg_q_breakdown',
                             'old_nonpositive_gain', 'old_nonfinite') if k in r],
        cause='absolute CG breakdown' if r.get('old_cg_z_breakdown') or r.get('old_cg_q_breakdown') else 'ratio recovered; inspect raw CG/finite/guard columns')
        for r in a5 if r['epoch'] == 100 and r.get('old_ratio') == 0 and r.get('fixed_ratio') is not None and r['fixed_ratio'] > 0]
    result = {**metadata(args, time.perf_counter()-start),
        'mu_rel_default': 1e-3 if pairs and fraction >= MU_FRACTION else 1e-2,
        'mu_rule': {'ratio': MU_RATIO, 'threshold_fraction': MU_FRACTION,
                    'qualifying_points': hits, 'converged_pairs': len(pairs), 'fraction': fraction},
        'tiny_V_gt_random_frob': fractions, 'significant_N1': n1,
        'A5_status': a5_decision(a5), 'A5_evidence': evidence}
    if (output / 'summary.json').exists():
        load_json(output / 'summary.json')
    atomic_json_save(result, output / 'summary.json')
    return result


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        read_csv(output / filename)
    if (output / 'summary.json').exists():
        load_json(output / 'summary.json')
    if args.measurement == 'trace-cost':
        return real_trace_smoke(args)
    if args.measurement == 'cost':
        return cost_smoke(args)
    if args.measurement == 'summary':
        for measurement in FILES:
            merge_measurement(output, measurement)
        return summarize(args)
    if args.fake_data:
        torch.set_num_threads(1)
    with isolated_rng(dedicated_generator(args.seed, 'model')):
        model = build_model(torch.device(args.device), args.fake_data)
    datasets = setup_data(args.data_root, args.fake_data, args.seed)
    paths = [Path(p) for p in args.checkpoints] or sorted(Path(args.checkpoint_dir).glob('epoch_*.pt'))
    if not paths and args.smoke and args.fake_data:
        # A synthetic recipe checkpoint is an explicit smoke fixture, never a
        # fallback for missing real data/checkpoints.
        path = output / 'smoke_recipe.pt'
        if not path.exists():
            torch.save({'kind': 'vna_recipe', 'epoch': 100, 'model': model.state_dict()}, path)
        paths = [path]
    if not paths:
        raise ValueError('no Phase-0 recipe checkpoints found')
    measurements = list(FILES) if args.measurement == 'all' else [args.measurement]
    args_identity = {k: v for k, v in vars(args).items()
                     if k not in ('measurement', 'output', 'checkpoint_dir', 'checkpoints')}
    for path in paths:
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('kind') != 'vna_recipe':
            raise ValueError('expected vna_recipe checkpoint')
        epoch = payload['epoch']
        model.load_state_dict(payload['model'])
        measurement = Measurement(model, datasets, args, epoch)
        digest = checkpoint_hash(path)
        for name in measurements:
            if not (args.smoke or args.cost_smoke) and epoch not in EPOCHS[name]:
                continue
            marker = output / 'completed' / name / f'epoch_{epoch:03d}.json'
            identity = {'schema_version': SCHEMA_VERSION, 'args': args_identity,
                        'checkpoint_sha256': digest, 'measurement': name, 'epoch': epoch, 'implementation_revision': 'review-round-1'}
            if marker.exists():
                previous = load_json(marker)
                if previous['identity'] != identity:
                    raise ValueError(f'resume identity mismatch in {marker}')
                print(f'skip {name} epoch {epoch}', flush=True)
                continue
            start = time.perf_counter()
            with invariant(model):
                if args.cost_smoke:
                    measurement.sites = [r.name for r in model.growing_blocks()][:1]
                if name == 'a5':
                    rows = measurement.measure_a5(payload)
                elif name == 'solver':
                    rows = measurement.measure_solver(path, output / 'solver_audit' / f'epoch_{epoch:03d}')
                else:
                    rows = getattr(measurement, 'measure_' + name)()
            meta = metadata(args, time.perf_counter() - start)
            rows = [{'epoch': epoch, **row, **meta} for row in rows]
            marker.parent.mkdir(parents=True, exist_ok=True)
            atomic_json_save({**meta, 'identity': identity, 'rows': rows}, marker)
            print(f'completed {name} epoch {epoch}: {len(rows)} rows in {meta["wall_seconds"]:.2f}s', flush=True)
        for name in measurements:
            merge_measurement(output, name)
    return summarize(args) if args.measurement == 'all' else None


def real_trace_smoke(args):
    """Real ResNet-18/CIFAR shape; CPU batch-one time extrapolated to stat256."""
    from methods.vna_select import trace_cost
    device = torch.device(args.device)
    with isolated_rng(dedicated_generator(args.seed, 'model')):
        model = build_model(device, False)
    site = model.growing_blocks()[0].name
    A = {'weight': model.block(site).first_layer.layer.weight.detach()[:16].clone(), 'bias': None}
    inputs = torch.randn(1, 3, 32, 32, device=device,
        generator=dedicated_generator(args.seed, 'trace-cost-inputs', device))
    with invariant(model): result = trace_cost(model, site, A, inputs)
    result.update(stat_batch=256, rank=16, parameter_count=sum(p.numel() for p in model.parameters()),
                  extrapolation='coordinate time and output count each scaled by batch ratio',
                  projected_a3_exact_trace_seconds_stat256=result['projected_a3_exact_trace_seconds'] * 256**2)
    result.update(metadata(args, result['coordinate_vjp_seconds'] * 4))
    atomic_json_save(result, Path(args.output) / 'trace_cost.json')
    return result


def cost_smoke(args):
    """Two real checkpoint equivalents, full shapes/CG with sampled units.

    A3 times rank64 and conservatively charges it for all three ranks. Costs
    include selection and split halves. Joint units use all eight sites.
    """
    if args.fake_data or args.smoke:
        raise ValueError('cost smoke requires real model/data/checkpoints')
    paths = [Path(p) for p in args.checkpoints] or sorted(Path(args.checkpoint_dir).glob('epoch_*.pt'))
    if len(paths) < 2:
        raise ValueError('cost smoke requires two distinct checkpoints')
    epochs = [torch.load(p, map_location='cpu', weights_only=False)['epoch'] for p in paths[:2]]
    if len(set(epochs)) != 2: raise ValueError('cost smoke requires distinct epochs')
    pilot = copy.copy(args)
    pilot.measurement, pilot.cost_smoke = 'all', True
    pilot.checkpoints = [str(p) for p in paths[:2]]
    pilot.output = str(Path(args.output) / 'cost_units')
    run(pilot)
    # Scale full-size sampled checkpoint units to all planned production units.
    multipliers = {'a5': 8, 'null': 8*8, 'selection': 3*3*3,
                   'joint': 1, 'mu': 2, 'solver': 3}
    projections = {}
    measured = 0.
    for name in FILES:
        units = [load_json(p) for p in (Path(pilot.output)/'completed'/name).glob('*.json')]
        if len(units) != 2: raise ValueError(f'incomplete cost smoke: {name}')
        times = [u['wall_seconds'] for u in units]
        measured += sum(times)
        projections[name] = {'checkpoint_seconds': times, 'sample_multiplier': multipliers[name],
            'projected_gpu_hours': max(times) * multipliers[name] * len(EPOCHS[name]) / 3600}
    result = {**metadata(args, measured), 'cost_projection': projections,
              'total_projected_gpu_hours': sum(p['projected_gpu_hours'] for p in projections.values()),
              'smoke_gpu_hours': measured/3600, 'epochs': epochs,
              'projection_basis': 'maximum of two checkpoint equivalents; full shapes/CG; rank64 charged to all A3 ranks'}
    atomic_json_save(result, Path(args.output) / 'cost_projection.json')
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('measurement', choices=(*FILES, 'summary', 'all', 'cost', 'trace-cost'))
    p.add_argument('--checkpoint-dir', default='outputs/vna_phase0_seed0/recipe')
    p.add_argument('--checkpoints', nargs='*', default=[])
    p.add_argument('--output', default='outputs/stageA')
    p.add_argument('--data-root', default='data')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cost-smoke', action='store_true', help='representative full-shape cost units only; not scientific results')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--fake-data', action='store_true')
    return p


def main():
    args = parser().parse_args()
    result = run(args)
    if result is not None:
        print(json.dumps(result, indent=2))


def restore_archives(input_root, output):
    """Restore only Stage-A members; reject traversal and conflicting files."""
    import tarfile
    import zipfile
    output = Path(output)
    def save(name, data):
        parts = Path(name).parts
        if '..' in parts or Path(name).is_absolute():
            raise ValueError('unsafe archive member')
        if 'stageA' not in parts:
            return
        relative = Path(*parts[parts.index('stageA')+1:])
        if not relative.parts:
            return
        target = output / relative
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f'conflicting Stage-A restore file {relative}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for path in sorted(Path(input_root).rglob('*')):
        if path.is_file() and 'stageA' in path.parts:
            save('/'.join(path.parts[path.parts.index('stageA'):]), path.read_bytes())
        elif path.is_file() and (path.name.endswith('.tar.gz') or path.suffix == '.tar'):
            with tarfile.open(path) as archive:
                for member in archive.getmembers():
                    if member.isfile() and 'stageA' in Path(member.name).parts:
                        save(member.name, archive.extractfile(member).read())
        elif path.is_file() and path.suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if not name.endswith('/') and 'stageA' in Path(name).parts:
                        save(name, archive.read(name))


if __name__ == '__main__':
    main()
