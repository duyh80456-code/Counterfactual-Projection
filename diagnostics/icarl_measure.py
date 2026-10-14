"""B3 GGN diagnostics on disjoint draws from already-seen training samples."""
import json
import time
import torch
from adapters.icarl_hooks import sites_for, loss_context
from diagnostics.stage_a import null_residual, empirical_p, diagnostic_ratio
from methods.vna import random_A_like
from methods.vna_ggn import GGNConfig, GGNProducts, relative_damping
from methods.vna_joint import joint_products, joint_correction, _inverse
from methods.vna_select import tiny_V, tiny_U_from_cotangent, prepare_A, subspace_sim


class B3Measurement:
    def __init__(self, metadata, *, mu_rel_default=.01, cg_iters=1000):
        self.metadata = metadata
        self.mus = sorted({mu_rel_default, .01})
        self.cg_iters = cg_iters

    def __call__(self, model, batch, loss_spec, ctx, *, generator):
        start = time.monotonic()
        model.eval()
        order = torch.randperm(len(batch[0]), device=batch[0].device, generator=generator)
        half = len(order)//2
        if not half: raise ValueError('B3 measurement needs at least two buffered samples')
        stat = tuple(t[order[:half]] for t in batch)
        correction = tuple(t[order[half:]] for t in batch)
        _, stat_ctx = loss_context(ctx['learner'], stat)
        _, corr_ctx = loss_context(ctx['learner'], correction)
        names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
        sites = sites_for(model)
        rows = []
        with torch.no_grad(): logits = model(stat[0])
        g = loss_spec.residual(logits, stat_ctx)
        for mu_rel in self.mus:
            cfg = GGNConfig(damping_rel=mu_rel, cg_iters=self.cg_iters, cg_tol=1e-6,
                            seed=self.metadata['seed'], damping_norm='outputs', project_bn=False)
            for rank in (4, 8, 16):
                proposals, records = {}, []
                for site in sites:
                    layer = site.first_layer.layer
                    if rank > layer.out_channels: continue
                    seed = {'weight': layer.weight[:1].detach(), 'bias': None}
                    def filtered(residual=None):
                        with joint_products(model, [site], {site: seed}, stat[0], names) as p:
                            op = GGNProducts(p, stat[1], loss_spec, stat_ctx)
                            mu = relative_damping(cfg, op.trace(cfg), op)
                            gradient = op.gradient if residual is None else p.JT(residual.flatten()).double()
                            u, status = _inverse(lambda v: op.F(v)+mu*v, gradient, cfg)
                            return op.H(p.Jv(u)).reshape_as(logits), mu, status
                    z, mu, solved = filtered()
                    A, capture = tiny_V(model, site, [stat], rank, cotangents=[z], cfg=cfg)
                    proposals[site] = A
                    def score(a, cotangent=None, data=correction, context=corr_ctx):
                        return joint_correction(model, [site], {site: a}, data, cfg, names,
                            mu=mu, cotangent=cotangent, loss_spec=loss_spec, ctx=context)[3]
                    real = score(A)
                    random, _ = prepare_A(model, site, random_A_like(A, generator=generator), 'random_frob', reference=A)
                    random_record = score(random)
                    incoming, _ = tiny_U_from_cotangent(model, site, stat[0], g, rank, cfg)
                    incoming, _ = prepare_A(model, site, incoming, 'tiny_U')
                    u_record = score(incoming)
                    split = len(stat[0])//2
                    similarity = None
                    if split:
                        pieces = [tiny_V(model, site, [(stat[0][s], stat[1][s])], rank,
                                  cotangents=[z[s]], cfg=cfg)[0]
                                  for s in (slice(None, split), slice(split, None))]
                        similarity = subspace_sim(*pieces)
                    converged = solved['converged'] and all(r['cg_converged'] for r in (real, random_record, u_record))
                    null_stats = {}
                    for kind in ('N1', 'N2'):
                        lambdas, rhos = [], []
                        for _ in range(8):
                            null = null_residual(g, kind, generator, residual_is_zero_sum=loss_spec.residual_is_zero_sum)
                            null_z, _, status = filtered(null)
                            null_A, cap = tiny_V(model, site, [stat], rank, cotangents=[null_z], cfg=cfg)
                            # Null correction residual has the correction batch's own shape/scaling.
                            with torch.no_grad(): corr_logits = model(correction[0])
                            corr_g = loss_spec.residual(corr_logits, corr_ctx)
                            null_record = score(null_A, null_residual(corr_g, kind, generator,
                                                residual_is_zero_sum=loss_spec.residual_is_zero_sum))
                            converged &= status['converged'] and null_record['cg_converged']
                            lambdas.append(float(cap['generalized_eigenvalues'][0]))
                            rhos.append(null_record['rho'])
                        p_lam = empirical_p(float(capture['generalized_eigenvalues'][0]), lambdas)
                        null_stats.update({f'p_lambda_{kind}': p_lam, f'significant_{kind}': p_lam <= 1/9,
                                           f'p_rho_{kind}': empirical_p(real['rho'], rhos),
                                           f'significant_rho_{kind}': empirical_p(real['rho'], rhos) <= 1/9,
                                           f'null_lambda_{kind}': json.dumps(lambdas),
                                           f'null_rho_{kind}': json.dumps(rhos)})
                    row = {**self.metadata, **{k: ctx[k] for k in ('task', 'epoch', 'step', 'end_of_task', 'train_acc')},
                           'site': site.module_name, 'rank': rank, 'selected_rank': len(A['weight']),
                           'mu_rel': mu_rel, 'damping_norm': 'outputs', 'buffer_m': 128,
                           'measurement_m': len(batch[0]), 'stat_m': len(stat[0]), 'correction_m': len(correction[0]),
                           **{k: real[k] for k in ('U', 'V', 'eta', 'rho', 'mu', 'cg_iters_used', 'cg_rel_residual')},
                           'cg_converged': converged, 'V_tiny_V': real['V'], 'V_random_frob': random_record['V'],
                           'V_tiny_U': u_record['V'], 'tiny_V_random_ratio': diagnostic_ratio(real['V'], random_record['V']),
                           'lambda1': float(capture['generalized_eigenvalues'][0]), 'subspace_sim': similarity, **null_stats}
                    records.append(row)
                joint = joint_correction(model, list(proposals), proposals, correction, cfg, names,
                                         loss_spec=loss_spec, ctx=corr_ctx)[3]
                groups = {}
                for label, mask in (('exemplar', correction[1] < ctx['known_classes']),
                                    ('new', correction[1] >= ctx['known_classes'])):
                    groups[f'{label}_m'] = int(mask.sum())
                    groups[f'rho_{label}'] = None
                    if mask.any():
                        subset = tuple(t[mask] for t in correction)
                        _, subset_ctx = loss_context(ctx['learner'], subset)
                        record = joint_correction(model, list(proposals), proposals, subset, cfg, names,
                                                  loss_spec=loss_spec, ctx=subset_ctx)[3]
                        groups[f'rho_{label}'] = record['rho']
                        joint['cg_converged'] &= record['cg_converged']
                for row in records:
                    row.update(rho_joint=joint['rho'], **groups)
                    row['cg_converged'] &= joint['cg_converged']
                    rows.append(row)
        for row in rows: row['wall_seconds'] = time.monotonic()-start
        return rows
