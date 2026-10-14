"""B5 measurements use only the already-seen sliding window."""
import copy
import json
import time
import torch
from adapters.lop_hooks import Prediction, sites_for
from diagnostics.stage_a import null_residual, empirical_p, diagnostic_ratio
from methods.vna_ggn import GGNConfig, GGNProducts
from methods.vna_joint import joint_correction, joint_products
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
        self.benchmark, self.metadata = benchmark, metadata
        self.mus = sorted({mu_rel_default, .01})
        self.ranks = ranks or ([1, 2, 4] if benchmark == 'regression' else [4, 16, 64])
        self.cg_iters = cg_iters

    def __call__(self, model, batch, loss_spec, ctx, *, generator):
        start = time.monotonic()
        sites = sites_for(model)
        stats = activity_stats(model, batch)
        rows = []
        all_names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
        loss_ctx = {'targets': batch[1], 'loss_fn': ctx['loss_fn']}
        with torch.no_grad(): logits = model(batch[0])
        g = loss_spec.residual(logits, loss_ctx)
        for mu_rel in self.mus:
            cfg = GGNConfig(damping_rel=mu_rel, cg_iters=self.cg_iters, cg_tol=1e-6, project_bn=False,
                            seed=self.metadata['seed'], damping_norm='outputs')
            spectral = None
            for rank in self.ranks:
                proposals, records, controls = {}, {}, {}
                for index, site in enumerate(sites):
                    width = site.first_layer.layer.weight.shape[0]
                    if rank > width: continue
                    seed = {'weight': site.first_layer.layer.weight[:1].detach(),
                            'bias': None if site.first_layer.layer.bias is None else site.first_layer.layer.bias[:1].detach()}
                    # Whole-network K for both selection and all corrections.
                    with joint_products(model, [site], {site: seed}, batch[0], all_names) as p:
                        op = GGNProducts(p, batch[1], loss_spec, loss_ctx)
                        from methods.vna_joint import _inverse
                        from methods.vna_ggn import relative_damping
                        trace = op.trace(cfg)
                        mu = relative_damping(cfg, trace, op)
                        u, solve = _inverse(lambda v: op.F(v)+mu*v, op.gradient, cfg)
                        z = op.H(p.Jv(u)).reshape_as(logits)
                    A, cap = tiny_V(model, site, [batch], rank, cotangents=[z], cfg=cfg)
                    proposals[site] = A
                    def correction(a, cotangent=None):
                        return joint_correction(model, [site], {site: a}, batch, cfg, all_names,
                                                mu=mu, cotangent=cotangent, loss_spec=loss_spec, ctx=loss_ctx)[3]
                    real = correction(A)
                    real['cg_converged'] &= solve['converged']
                    random = random_init_A(site, A, ctx['shadow'], index, generator)
                    random, random_normalization = prepare_A(model, site, random, 'random_frob', reference=A)
                    # normalize_A makes every row match the same init row norm;
                    # hence this random draw has exactly the same Frobenius norm.
                    random_record = correction(random)
                    incoming, _ = tiny_U_from_cotangent(model, site, batch[0], g, rank, cfg)
                    incoming, _ = prepare_A(model, site, incoming, 'tiny_U')
                    u_record = correction(incoming)
                    half = len(batch[0])//2
                    similarity = None
                    if half:
                        first, _ = tiny_V(model, site, [(batch[0][:half], batch[1][:half])], rank,
                                          cotangents=[z[:half]], cfg=cfg)
                        second, _ = tiny_V(model, site, [(batch[0][half:], batch[1][half:])], rank,
                                           cotangents=[z[half:]], cfg=cfg)
                        similarity = subspace_sim(first, second)
                    null_stats = {}
                    null_converged = True
                    for kind in ('N1', 'N2'):
                        null_lambda, null_rho = [], []
                        for _ in range(8):
                            null = null_residual(g, kind, generator,
                                                 residual_is_zero_sum=loss_spec.residual_is_zero_sum)
                            with joint_products(model, [site], {site: seed}, batch[0], all_names) as p:
                                op = GGNProducts(p, batch[1], loss_spec, loss_ctx)
                                gradient = p.JT(null.flatten()).double()
                                u, status = _inverse(lambda v: op.F(v)+mu*v, gradient, cfg)
                                null_z = op.H(p.Jv(u)).reshape_as(logits)
                            null_A, capture = tiny_V(model, site, [batch], rank, cotangents=[null_z], cfg=cfg)
                            record = correction(null_A, null)
                            null_converged &= status['converged'] and record['cg_converged']
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
                           'measurement_m': len(batch[0]), 'buffer_m': ctx['buffer_m'],
                           **{k: real[k] for k in ('U', 'V', 'eta', 'rho', 'mu', 'cg_converged', 'cg_iters_used', 'cg_rel_residual')},
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
                    row['cg_converged'] &= random_record['cg_converged'] and u_record['cg_converged'] and null_converged
                    # Counterfactuals are private copies, with outgoing weights
                    # zeroed at the declared selected coordinates only.
                    candidates = [(label, shadow, selected[index]) for label, shadow, selected in ctx['cbp_candidates']]
                    candidates.append(('redo', ctx['shadow'], stats[index]['dormant_indices']))
                    for criterion, shadow, units in candidates:
                        count = len(units)
                        row[f'{criterion}_m'] = count
                        row[f'{criterion}_tiny_directions'] = 0
                        row[f'{criterion}_fraction'] = count / width
                        if not count:
                            for name in ('V_swap_tiny', 'V_swap_random', 'ratio', 'eta_before', 'eta_after'):
                                row[f'{criterion}_{name}'] = None
                            continue
                        old_A = {'weight': site.first_layer.layer.weight.detach()[units],
                                 'bias': None if site.first_layer.layer.bias is None else site.first_layer.layer.bias.detach()[units]}
                        before_swap = correction(old_A)
                        swap_A, _ = tiny_V(model, site, [batch], count, cotangents=[z], cfg=cfg)
                        # Draw all m rows once; missing TINY directions use
                        # the first m-k rows of this same random arm.
                        swap_random = random_init_A(site, old_A, shadow, index, generator)
                        swap_A, swap_random, k = swap_directions(swap_A, swap_random, ctx['init_row_norms'][index])
                        row[f'{criterion}_tiny_directions'] = k
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
                            gains.append(joint_correction(clone, [clone_site], {clone_site: a}, batch, cfg,
                                mu=mu, loss_spec=loss_spec, ctx=loss_ctx)[3])
                        row.update({f'{criterion}_V_swap_tiny': gains[0]['V'], f'{criterion}_V_swap_random': gains[1]['V'],
                                    f'{criterion}_ratio': 1.0 if k == 0 else diagnostic_ratio(gains[0]['V'], gains[1]['V']),
                                    f'{criterion}_eta_before': before_swap['eta'], f'{criterion}_eta_after': gains[0]['eta']})
                    records[site] = row; controls[site] = (random, incoming)
                if not proposals: continue
                joint = joint_correction(model, list(proposals), proposals, batch, cfg, all_names,
                                          loss_spec=loss_spec, ctx=loss_ctx)[3]
                if spectral is None:
                    spectral = spectrum(model, list(proposals), proposals, batch, loss_spec, loss_ctx, joint['mu'])
                for site, row in records.items():
                    row.update(rho_joint=joint['rho'], **spectral)
                    row['cg_converged'] &= joint['cg_converged']
                    rows.append(row)
        for row in rows: row['wall_seconds'] = time.monotonic()-start
        return rows
