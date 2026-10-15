"""Disjoint Stage-B diagnostics with shared solves.

Each row's null p-value is valid; p-values across sites/ranks of a point are
correlated because they reuse the same null residuals.
"""
import time
import torch
from adapters.icarl_hooks import sites_for, loss_context
from diagnostics.stage_a import diagnostic_ratio
from diagnostics.measure_efficiency import (SolveCache, draw_nulls, virtual_energy,
                                             null_columns, verify_energy)
from methods.vna import random_A_like
from methods.vna_ggn import GGNConfig
from methods.vna_joint import joint_correction
from methods.vna_select import tiny_V, tiny_U_from_cotangent, prepare_A, subspace_sim
from experiments.vna_resume import measurement_update


class B3Measurement:
    def __init__(self, metadata, *, mu_rel_default=.01, cg_iters=1000, sites_fn=sites_for,
                 context_fn=None, ranks=(4, 8, 16), groups=True, null_repeats=8, solver="auto", output_exact_max=4096, cg_tol=1e-6):
        self.metadata = metadata
        self.mus = [1e-3, 1e-2]
        self.cg_iters = cg_iters
        self.solver, self.output_exact_max, self.cg_tol = solver, output_exact_max, cg_tol
        self.sites_fn, self.ranks, self.groups = sites_fn, ranks, groups
        self.context_fn = context_fn or (lambda ctx, batch: loss_context(ctx['learner'], batch))
        self.null_repeats = null_repeats

    def __call__(self, model, batch, loss_spec, ctx, *, generator):
        start = time.monotonic()
        model.eval()
        order = torch.randperm(len(batch[0]), device=batch[0].device, generator=generator)
        half = len(order)//2
        if not half: raise ValueError('measurement needs at least two buffered samples')
        stat = tuple(t[order[:half]] for t in batch)
        evaluation = tuple(t[order[half:]] for t in batch)
        _, stat_ctx = self.context_fn(ctx, stat)
        _, eval_ctx = self.context_fn(ctx, evaluation)
        sites = self.sites_fn(model)
        rows = []
        with torch.no_grad():
            g = loss_spec.residual(model(stat[0]), stat_ctx)
            e = loss_spec.residual(model(evaluation[0]), eval_ctx)
        factors = {}
        prepared_groups = {}
        for mu_rel in self.mus:
            cfg = GGNConfig(damping_rel=mu_rel, cg_iters=self.cg_iters, cg_tol=self.cg_tol,
                            seed=self.metadata['seed'], damping_norm='outputs', project_bn=False)
            cache = SolveCache(model, sites[0], stat, loss_spec, stat_ctx, cfg, solver=self.solver, output_exact_max=self.output_exact_max, factors=factors)
            z = cache.solve('real:stat', g, stat, stat_ctx)
            ze = cache.solve('real:eval', e, evaluation, eval_ctx)
            cache.null_draw_id = f"{self.metadata['seed']}:{ctx.get('task')}:{ctx.get('epoch')}:{ctx['step']}:{mu_rel}"
            nulls = draw_nulls(cache, g, e, stat, evaluation, stat_ctx, eval_ctx, generator, self.null_repeats)
            groups = {}
            for label, mask in (('exemplar', evaluation[1] < ctx['known_classes']),
                                ('new', evaluation[1] >= ctx['known_classes'])) if self.groups else ():
                if label not in prepared_groups:
                    subset = tuple(t[mask] for t in evaluation)
                    if mask.any():
                        _, context = self.context_fn(ctx, subset)
                        with torch.no_grad(): residual = loss_spec.residual(model(subset[0]), context)
                        prepared_groups[label] = subset, context, residual
                    else: prepared_groups[label] = None
                prepared = prepared_groups[label]
                if prepared:
                    subset, context, residual = prepared
                    cache.factor(subset, context, parent=cache.factor(evaluation, eval_ctx), mask=mask)
                    zg = cache.solve('real:'+label, residual, subset, context)
                    groups[label] = (subset, residual, zg)
                else: groups[label] = None
            for rank in self.ranks:
                records = []
                group_sums = {label: 0. for label in groups}
                for site in sites:
                    if rank > site.first_layer.layer.out_channels: continue
                    measurement_update(mu=cache.mu, mu_rel=mu_rel, site=site.module_name, rank=rank)
                    A, capture = tiny_V(model, site, [stat], rank, cotangents=[z], cfg=cfg)
                    def score(a, cotangent=e, filtered=ze):
                        return virtual_energy(model, [site], {site: a}, evaluation, filtered, cotangent)
                    real = score(A)
                    measurement_update(solve='tiny_V:correction')
                    checked = joint_correction(model, [site], {site: A}, evaluation, cfg, cache.ctx['names'],
                        mu=cache.mu, trace_F=cache.trace, loss_spec=loss_spec, ctx=eval_ctx,
                        output_factor=cache.factor(evaluation, eval_ctx))[3]
                    consistency = verify_energy(real, checked)
                    random, _ = prepare_A(model, site, random_A_like(A, generator=generator), 'random_frob', reference=A)
                    random_record = score(random)
                    incoming, _ = tiny_U_from_cotangent(model, site, stat[0], g, rank, cfg)
                    incoming, _ = prepare_A(model, site, incoming, 'tiny_U')
                    u_record = score(incoming)
                    split = len(stat[0])//2
                    similarity = None
                    if split:
                        pieces = [tiny_V(model, site, [tuple(t[s] for t in stat)], rank,
                                  cotangents=[z[s]], cfg=cfg)[0]
                                  for s in (slice(None, split), slice(split, None))]
                        similarity = subspace_sim(*pieces)
                    values = {}
                    for kind, draws in nulls.items():
                        lambdas, rhos, energies = [], [], []
                        for null_z, null_ze, null_e in draws:
                            null_A, cap = tiny_V(model, site, [stat], rank, cotangents=[null_z], cfg=cfg)
                            lambdas.append(float(cap['generalized_eigenvalues'][0]))
                            energy = score(null_A, null_e, null_ze)
                            rhos.append(energy['rho']); energies.append(energy)
                        values[kind] = lambdas, rhos, energies
                    for label, group in groups.items():
                        if group:
                            data, residual, filtered = group
                            energy = virtual_energy(model, [site], {site: A}, data, filtered, residual)
                            group_sums[label] += energy['rho']
                    lam = float(capture['generalized_eigenvalues'][0])
                    real_ok = cache.converged() and checked['cg_converged'] and consistency <= 1e-4
                    null_ok = cache.converged(null=True)
                    row = {**self.metadata, **{k: ctx[k] for k in ('task', 'epoch', 'step', 'end_of_task', 'train_acc')},
                           'site': site.module_name, 'rank': rank, 'selected_rank': len(A['weight']),
                           'mu_rel': mu_rel, 'mu': cache.mu, 'trace': cache.trace, 'damping_norm': 'outputs',
                           'buffer_m': len(batch[0]), 'measurement_m': len(batch[0]),
                           'stat_m': len(stat[0]), 'eval_m': len(evaluation[0]), 'correction_m': len(evaluation[0]),
                           **{k: real[k] for k in ('U', 'V', 'eta', 'rho', 'e_norm2', 'nu')},
                           **{k: checked[k] for k in ('cg_iters_used', 'cg_rel_residual', 'raw_gain', 'realized_gain', 'finite')},
                           'cg_converged_real': real_ok, 'cg_converged_null': null_ok, 'cg_converged': real_ok and null_ok,
                           'V_consistency': consistency, **cache.metadata(evaluation, eval_ctx), 'V_tiny_V': real['V'], 'V_random_frob': random_record['V'],
                           'V_tiny_U': u_record['V'], 'tiny_V_random_ratio': diagnostic_ratio(real['V'], random_record['V']),
                           'lambda1': lam, 'subspace_sim': similarity,
                           'null_draw_id': f"{self.metadata['seed']}:{ctx['task']}:{ctx['epoch']}:{ctx['step']}:{mu_rel}",
                           **null_columns(lam, real['rho'], values, real['nu'])}
                    records.append(row)
                joint_rho = sum(row['rho'] for row in records)
                for row in records:
                    row.update(rho_joint=joint_rho)
                    for label, group in groups.items():
                        row[f'{label}_m'] = len(group[0][0]) if group else 0
                        row[f'rho_{label}'] = group_sums[label] if group else None
                    rows.append(row)
        for row in rows: row['wall_seconds'] = time.monotonic()-start
        return rows
