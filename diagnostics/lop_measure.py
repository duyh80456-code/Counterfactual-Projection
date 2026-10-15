"""B5 uses disjoint stat/eval windows. Shared nulls preserve row p-values;
p-values across sites/ranks of one point are correlated.
"""
import copy
import json
import time
import torch
from adapters.lop_hooks import Prediction, sites_for
from diagnostics.stage_a import null_residual, empirical_p, diagnostic_ratio
from methods.vna_ggn import GGNConfig, GGNProducts
from methods.vna_joint import joint_correction, joint_products
from diagnostics.measure_efficiency import SolveCache, draw_nulls, virtual_energy, solve_z, verify_energy
from experiments.vna_resume import measurement_update
from methods.vna_select import tiny_V, tiny_U_from_cotangent, normalize_A, prepare_A, subspace_sim, neuron_norms


def random_init_A(site, reference, shadow, index, generator):
    bound = shadow.bounds[index]
    value = {'weight': torch.empty_like(reference['weight']).uniform_(-bound, bound, generator=generator),
             'bias': None if reference['bias'] is None else torch.zeros_like(reference['bias'])}
    return normalize_A(site.model, site, value)


def swap_directions(tiny, random, target_norm):
    """Complete TINY with a shared random draw, then match both init norms."""
    count, k = len(random['weight']), len(tiny['weight'])
    if k > count:
        raise ValueError('TINY returned more directions than replacement units')
    def init_scale(a):
        scale = target_norm / neuron_norms(a).clamp_min(1e-30)
        return {name: None if value is None else value * scale.reshape((-1,) + (1,) * (value.ndim-1))
                for name, value in a.items()}
    random = init_scale(random)
    tiny = init_scale(tiny)
    mixed = {name: None if value is None else torch.cat((tiny[name], value[:count-k]), dim=0)
             for name, value in random.items()}
    return mixed, random, k


def activity_stats(model, batch):
    with torch.no_grad():
        _, features = model.native.predict(batch[0])
    result = []
    for i, feature in enumerate(features):
        width = model.native.layers[2*i].weight.shape[0]
        matrix = feature.detach().reshape(len(feature), width, -1).transpose(1, 2).reshape(-1, width)
        score = matrix.abs().mean(0)
        dormant = score / score.mean().clamp_min(1e-30) <= .1
        singular = torch.linalg.svdvals(matrix.double())
        prob = singular / singular.sum().clamp_min(1e-30)
        effective = float(torch.exp(-(prob * prob.clamp_min(1e-30).log()).sum())) if singular.sum() else 0.
        result.append({'dormant_fraction': float(dormant.float().mean()), 'effective_rank': effective,
                       'dormant_indices': torch.where(dormant)[0]})
    return result


def spectrum(model, sites, A, batch, spec, ctx, mu, *, top_k=20):
    """Exact output-space GGN spectrum, including residual spectral energy.

    H^(1/2) K H^(1/2) has the nonzero eigenvalues of J^T H J.
    Stream columns without materializing the parameter Jacobian.
    """
    with joint_products(model, sites, A, batch[0]) as products:
        op = GGNProducts(products, batch[1], spec, ctx)
        n, c = op.logits.shape
        eye = torch.eye(c, device=batch[0].device, dtype=torch.float64)
        blocks = torch.stack([spec.metric_apply(op.logits, eye[j].expand(n, c), op.ctx) for j in range(c)], -1)
        e, q = torch.linalg.eigh(blocks)
        root = (q * e.clamp_min(0).sqrt().unsqueeze(-2)) @ q.transpose(-1, -2)
        transform = lambda v: torch.einsum('nij,nj->ni', root, v.double().reshape(n, c)).flatten()
        size = n*c
        coordinate = torch.zeros(size, device=batch[0].device, dtype=torch.float64)
        columns = []
        for i in range(size):
            coordinate.zero_(); coordinate[i] = 1
            columns.append(transform(products.K(transform(coordinate))).double())
        K = torch.stack(columns, -1)
        eigen, basis = torch.linalg.eigh((K+K.T)/2)
        # g is an output residual; report its energy in these output directions.
        energy = (basis.T @ op.g).square()
        low = float(energy[eigen < mu].sum()/energy.sum()) if energy.sum() else 0.
        return {'K_top_eigenvalues': json.dumps(eigen.flip(0)[:top_k].clamp_min(0).tolist()),
                'residual_energy_lambda_lt_mu': low, 'K_spectrum_method': 'exact_output_GGN'}


class B5Measurement:
    def __init__(self, benchmark, metadata, *, mu_rel_default=.01, ranks=None, cg_iters=1000):
        self.benchmark, self.metadata = benchmark, {'setting': 'B5', **metadata}
        self.mus = [1e-3, 1e-2]
        self.ranks = ranks or ([1, 2, 4] if benchmark == 'regression' else [4, 16, 64])
        self.cg_iters = cg_iters

    def __call__(self, model, batch, loss_spec, ctx, *, generator):
        start = time.monotonic()
        model.eval()
        order = torch.randperm(len(batch[0]), device=batch[0].device, generator=generator)
        half = len(order)//2
        if not half: raise ValueError('B5 measurement needs at least two samples')
        stat = tuple(t[order[:half]] for t in batch)
        evaluation = tuple(t[order[half:]] for t in batch)
        sites = sites_for(model)
        stats = activity_stats(model, stat)
        rows = []
        all_names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
        loss_ctx = {'targets': stat[1], 'loss_fn': ctx['loss_fn']}
        eval_ctx = {'targets': evaluation[1], 'loss_fn': ctx['loss_fn']}
        with torch.no_grad(): logits = model(stat[0])
        g = loss_spec.residual(logits, loss_ctx)
        with torch.no_grad(): e = loss_spec.residual(model(evaluation[0]), eval_ctx)
        for mu_rel in self.mus:
            cfg = GGNConfig(damping_rel=mu_rel, cg_iters=self.cg_iters, cg_tol=1e-6, project_bn=False,
                            seed=self.metadata['seed'], damping_norm='outputs')
            cache = SolveCache(model, sites[0], stat, loss_spec, loss_ctx, cfg)
            mu, trace = cache.mu, cache.trace
            z = cache.solve('real:stat', g, stat, loss_ctx)
            ze = cache.solve('real:eval', e, evaluation, eval_ctx)
            cache.null_draw_id = f"{self.metadata['seed']}:{ctx.get('task')}:{ctx.get('epoch')}:{ctx['step']}:{mu_rel}"
            nulls = draw_nulls(cache, g, e, stat, evaluation, loss_ctx, eval_ctx, generator, 8)
            swaps = {}
            spectral = None
            for rank in self.ranks:
                proposals, records, controls = {}, {}, {}
                for index, site in enumerate(sites):
                    width = site.first_layer.layer.weight.shape[0]
                    if rank > width: continue
                    measurement_update(mu=mu, mu_rel=mu_rel, site=site.module_name, rank=rank)
                    A, cap = tiny_V(model, site, [stat], rank, cotangents=[z], cfg=cfg)
                    proposals[site] = A
                    def correction(a, cotangent=e, filtered=ze):
                        return virtual_energy(model, [site], {site: a}, evaluation, filtered, cotangent)
                    real = correction(A)
                    measurement_update(solve='tiny_V:correction')
                    checked = joint_correction(model, [site], {site: A}, evaluation, cfg, all_names,
                        mu=mu, trace_F=trace, loss_spec=loss_spec, ctx=eval_ctx)[3]
                    consistency = verify_energy(real, checked)
                    real.update({k: checked[k] for k in ('cg_converged', 'cg_iters_used', 'cg_rel_residual',
                                                        'raw_gain', 'realized_gain', 'finite')}, mu=mu)
                    random = random_init_A(site, A, ctx['shadow'], index, generator)
                    random, random_normalization = prepare_A(model, site, random, 'random_frob', reference=A)
                    # normalize_A makes every row match the same init row norm;
                    # hence this random draw has exactly the same Frobenius norm.
                    random_record = correction(random)
                    incoming, _ = tiny_U_from_cotangent(model, site, stat[0], g, rank, cfg)
                    incoming, _ = prepare_A(model, site, incoming, 'tiny_U')
                    u_record = correction(incoming)
                    half = len(stat[0])//2
                    similarity = None
                    if half:
                        first, _ = tiny_V(model, site, [(stat[0][:half], stat[1][:half])], rank,
                                          cotangents=[z[:half]], cfg=cfg)
                        second, _ = tiny_V(model, site, [(stat[0][half:], stat[1][half:])], rank,
                                           cotangents=[z[half:]], cfg=cfg)
                        similarity = subspace_sim(first, second)
                    null_stats = {}
                    for kind in ('N1', 'N2'):
                        null_lambda, null_rho = [], []
                        for null_z, null_ze, null_e in nulls[kind]:
                            null_A, capture = tiny_V(model, site, [stat], rank, cotangents=[null_z], cfg=cfg)
                            record = correction(null_A, null_e, null_ze)
                            null_lambda.append(float(capture['generalized_eigenvalues'][0]))
                            null_rho.append(record['rho'])
                        lam = float(cap['generalized_eigenvalues'][0])
                        p_lam = empirical_p(lam, null_lambda)
                        significant_lam = p_lam <= 1/9
                        p_rho = empirical_p(real['rho'], null_rho)
                        significant_rho = p_rho <= 1/9
                        null_stats.update({f'p_lambda_{kind}': p_lam, f'p_rho_{kind}': p_rho,
                                           f'significant_{kind}': significant_lam,
                                           f'significant_rho_{kind}': significant_rho,
                                           f'null_lambda_{kind}': json.dumps(null_lambda),
                                           f'null_rho_{kind}': json.dumps(null_rho)})
                    row = {**self.metadata, 'task': ctx['task'], 'step': ctx['step'], 'point': ctx['point'],
                           'within_task_step': ctx['within_task_step'], 'buffer_task_purity': ctx['buffer_task_purity'],
                           'site': site.module_name, 'rank': rank, 'mu_rel': mu_rel, 'selection': 'tiny_V',
                           'null_kind': 'N2', 'n2_centered': loss_spec.residual_is_zero_sum,
                           'selected_rank': len(A['weight']),
                           'measurement_m': len(batch[0]), 'buffer_m': len(batch[0]),
                           'stat_m': len(stat[0]), 'eval_m': len(evaluation[0]), 'trace': trace,
                           'V_consistency': consistency,
                           'null_draw_id': f"{self.metadata['seed']}:{ctx['task']}:{ctx['step']}:{mu_rel}",
                           **{k: real[k] for k in ('U', 'V', 'eta', 'rho', 'mu', 'cg_iters_used', 'cg_rel_residual', 'raw_gain', 'realized_gain', 'finite')},
                           'damping_norm': 'outputs', 'lambda1': lam,
                           'V_tiny_V': real['V'], 'V_random_frob': random_record['V'],
                           'V_tiny_U': u_record['V'], 'tiny_V_random_ratio': diagnostic_ratio(real['V'], random_record['V']),
                           'subspace_sim': similarity, 'A_row_norm': float(neuron_norms(A).mean()),
                           'A_frob': float(neuron_norms(A).norm()), 'random_A_frob': float(neuron_norms(random).norm()),
                           'random_match_scale': random_normalization['match_scale'],
                           'dormant_fraction': stats[index]['dormant_fraction'],
                           'effective_rank': stats[index]['effective_rank'],
                           'weight_magnitude': float(site.first_layer.layer.weight.detach().abs().mean()),
                           'weight_magnitude_per_layer': json.dumps([float(layer.weight.detach().abs().mean())
                               for layer in list(model.native.layers)[::2]]),
                           **null_stats}
                    row['cg_converged_real'] = cache.converged() and checked['cg_converged']
                    row['cg_converged_null'] = cache.converged(null=True)
                    # Counterfactuals are private copies, with outgoing weights
                    # zeroed at the declared selected coordinates only.
                    if site not in swaps:
                        swap_row = {'cg_converged_real': True}
                        candidates = [(label, shadow, selected[index]) for label, shadow, selected in ctx['cbp_candidates']]
                        candidates.append(('redo', ctx['shadow'], stats[index]['dormant_indices']))
                        for criterion, shadow, units in candidates:
                            count = len(units)
                            swap_row[f'{criterion}_m'] = count
                            swap_row[f'{criterion}_tiny_directions'] = 0
                            swap_row[f'{criterion}_fraction'] = count / width
                            if not count:
                                for name in ('V_swap_tiny', 'V_swap_random', 'ratio', 'eta_before', 'eta_after'):
                                    swap_row[f'{criterion}_{name}'] = None
                                continue
                            old_A = {'weight': site.first_layer.layer.weight.detach()[units],
                                     'bias': None if site.first_layer.layer.bias is None else site.first_layer.layer.bias.detach()[units]}
                            before_swap = correction(old_A)
                            swap_A, _ = tiny_V(model, site, [stat], count, cotangents=[z], cfg=cfg)
                            # Draw all m rows once; missing TINY directions use
                            # the first m-k rows of this same random arm.
                            swap_random = random_init_A(site, old_A, shadow, index, generator)
                            swap_A, swap_random, k = swap_directions(swap_A, swap_random, ctx['init_row_norms'][index])
                            swap_row[f'{criterion}_tiny_directions'] = k
                            gains = []
                            for a in (swap_A, swap_random):
                                clone = Prediction(copy.deepcopy(model.native))
                                clone_site = sites_for(clone)[index]
                                first, second = clone_site.first_layer.layer, clone_site.second_layer.layer
                                with torch.no_grad():
                                    first.weight[units] = a['weight']
                                    if first.bias is not None: first.bias[units] = a['bias']
                                    outgoing_units = units
                                    if site.kind == 'conv_linear':
                                        spatial = second.in_features//first.out_channels
                                        outgoing_units = (units[:, None]*spatial+torch.arange(spatial, device=units.device)).flatten()
                                    second.weight[:, outgoing_units] = 0
                                clone_cache = SolveCache(clone, clone_site, stat, loss_spec, loss_ctx, cfg, mu=mu, trace=trace)
                                with torch.no_grad(): clone_e = loss_spec.residual(clone(evaluation[0]), eval_ctx)
                                clone_z = clone_cache.solve(f'real:swap:{criterion}:{len(gains)}', clone_e, evaluation, eval_ctx)
                                gains.append(virtual_energy(clone, [clone_site], {clone_site: a}, evaluation, clone_z, clone_e))
                                swap_row['cg_converged_real'] &= clone_cache.converged()
                            swap_row.update({f'{criterion}_V_swap_tiny': gains[0]['V'], f'{criterion}_V_swap_random': gains[1]['V'],
                                        f'{criterion}_ratio': 1.0 if k == 0 else diagnostic_ratio(gains[0]['V'], gains[1]['V']),
                                        f'{criterion}_eta_before': before_swap['eta'], f'{criterion}_eta_after': gains[0]['eta']})
                        swaps[site] = swap_row
                    swap_fields = swaps[site]
                    real_ok = row['cg_converged_real'] and swap_fields['cg_converged_real']
                    row.update({k: v for k, v in swap_fields.items() if k != 'cg_converged_real'})
                    row['cg_converged_real'] = real_ok
                    row['cg_converged'] = real_ok and row['cg_converged_null']
                    records[site] = row; controls[site] = (random, incoming)
                if not proposals: continue
                joint_rho = sum(row['rho'] for row in records.values())
                if spectral is None:
                    spectral = spectrum(model, list(proposals), proposals, evaluation, loss_spec, eval_ctx, mu)
                for site, row in records.items():
                    row.update(rho_joint=joint_rho, **spectral)
                    rows.append(row)
        for row in rows: row['wall_seconds'] = time.monotonic()-start
        return rows
