"""Projection-aware virtual-neuron selection using pinned Gromo statistics.

TINY is a linearized-ReLU proposal only; every pool score uses the true native
ReLU J_E at B=0. Statistical projection is an independent ablation from BN
parameter tangent projection.
"""
import copy
import torch
from methods.vna import block_products, _as_a_dict, random_A_like, dedicated_generator
from methods.vna_ggn import filtered_cotangent, GGNConfig


def neuron_norms(A):
    a = _as_a_dict(A)
    norms = a['weight'].flatten(1).square().sum(1)
    if a['bias'] is not None:
        norms += a['bias'].square()
    return norms.sqrt()


def normalize_A(model, site, A):
    layer = site.first_layer.layer if hasattr(site, 'resolve') else model.block(site).first_layer.layer
    existing = {'weight': layer.weight.detach(), 'bias': None if layer.bias is None else layer.bias.detach()}
    a = _as_a_dict(A)
    scale = neuron_norms(existing).mean() / neuron_norms(a).clamp_min(1e-30)
    return {'weight': a['weight'] * scale.reshape((-1,) + (1,) * (a['weight'].ndim - 1)),
            'bias': None if a['bias'] is None else a['bias'] * scale}


def _neuron(A, i, sign=1):
    return {n: None if v is None else sign * v[i:i+1] for n, v in A.items()}


def _concat(items):
    return {n: None if items[0][n] is None else torch.cat([a[n] for a in items]) for n in items[0]}


def score_A(model, site, A, inputs, z):
    with block_products(model, site, A, inputs) as p:
        return float(p.JET(z.flatten()).double().square().sum())


def choose_signs(model, site, A, inputs, z):
    choices = []
    for i in range(len(A['weight'])):
        plus, minus = _neuron(A, i), _neuron(A, i, -1)
        choices.append(plus if score_A(model, site, plus, inputs, z) >= score_A(model, site, minus, inputs, z) else minus)
    return _concat(choices)


def _profile_tiny(function):
    from functools import wraps
    @wraps(function)
    def measured(model, site, *args, **kwargs):
        from experiments.vna_resume import current_measurement
        monitor = current_measurement.get()
        device = next(model.parameters()).device
        profile = monitor is not None and device.type == 'cuda'
        if profile: torch.cuda.reset_peak_memory_stats(device)
        try:
            return function(model, site, *args, **kwargs)
        finally:
            if profile:
                peak = torch.cuda.max_memory_allocated(device)
                monitor.tiny_V_peak_cuda_bytes = max(monitor.tiny_V_peak_cuda_bytes, peak)
    return measured


@_profile_tiny
def tiny_V(model, site, batches, rank, *, cotangents=None, cfg=GGNConfig(),
           existing_weight_projection=False, normalize=True, score_signs=True,
           numerical_threshold=1e-6, loss_spec=None, contexts=None):
    """Reuse native TINY's thresholded generalized eigensolve.

    Pseudo loss sum(f*z) has logit derivative exactly z. Native N=-M_prev
    (plus the optional existing-weight residual), so its sign matches CE TINY.
    Gromo averages N_z and S by sample count across all supplied batches.
    For z=g parity use sum-CE cotangents (batch*g), as upstream expects sums.
    Default z=H J (F+mu)^-1 J^T g (or the supplied solver's cotangent).
    """
    if hasattr(site, 'resolve'):
        if site.native_path:
            return tiny_V(model, site.native_name, batches, rank, cotangents=cotangents,
                          cfg=cfg, existing_weight_projection=existing_weight_projection,
                          normalize=normalize, score_signs=score_signs,
                          numerical_threshold=numerical_threshold, loss_spec=loss_spec, contexts=contexts)
        from methods.vna_sites import hook_tiny
        return hook_tiny(model, site, batches, rank, cotangents=cotangents, cfg=cfg,
                         existing_weight_projection=existing_weight_projection,
                         normalize=normalize, score_signs=score_signs,
                         numerical_threshold=numerical_threshold, loss_spec=loss_spec, contexts=contexts)
    from experiments.vna_support import isolated_rng
    from gromo.utils.training_utils import compute_statistics
    from dual_growth.adapters import TinyAdapter
    batches = list(batches)
    if not batches or rank < 1:
        raise ValueError('need batches and positive rank')
    clone = copy.deepcopy(model)
    block = clone.block(site)
    # Pinned Gromo creates this lazy convolution identity in float32 even
    # for float64 models. Initialize it on the private statistics clone.
    second = block.second_layer
    if hasattr(second, 'bordering_weight') and second.bordering_weight is None:
        from gromo.utils.tools import create_bordering_effect_weight
        previous = second.previous_module
        channels = previous.in_channels * previous.kernel_size[0] * previous.kernel_size[1] + previous.use_bias
        second.bordering_weight = create_bordering_effect_weight(channels, convolution=second.layer).to(second.layer.weight)
    block.second_layer.target_in_neurons = block.second_layer.in_neurons + rank
    index = next(i for i, b in enumerate(clone.core._growable_layers) if b is block)
    clone.core.set_growing_layers(scheduling_method='sequential', index=index)
    if cotangents is None:
        seed_A = {'weight': model.block(site).first_layer.layer.weight.detach()[:1], 'bias': None}
        contexts = [None] * len(batches) if contexts is None else list(contexts)
        if len(contexts) != len(batches):
            raise ValueError('one context required per batch')
        cotangents = [filtered_cotangent(model, site, seed_A, batch, cfg, loss_spec=loss_spec, ctx=ctx)
                      for batch, ctx in zip(batches, contexts)]
    cotangents = list(cotangents)
    if len(cotangents) != len(batches):
        raise ValueError('one cotangent required per batch')
    pseudo_batches = [(x, z.detach().reshape(len(x), -1)) for (x, _), z in zip(batches, cotangents)]
    capture = {}
    original = second._compute_optimal_added_parameters
    def capture_solve(*args, **kwargs):
        capture['S'] = second.tensor_s_growth().detach().double().clone()
        capture['N_z'] = (second.tensor_n if existing_weight_projection else -second.tensor_m_prev()).detach().double().clone()
        return original(*args, **kwargs)
    second._compute_optimal_added_parameters = capture_solve
    try:
        with isolated_rng(dedicated_generator(cfg.seed, 'tiny-V-statistics')):
            compute_statistics(clone.core, pseudo_batches, loss_function=lambda f, z: (f * z.to(f)).sum(),
                               device=next(clone.parameters()).device)
            options = TinyAdapter(quantum_params=10**9, numerical_threshold=numerical_threshold,
                                  statistical_threshold=0.)._optimal_update_kwargs(rank)
            options['use_projection'] = existing_weight_projection
            clone.core.compute_optimal_updates(**options)
            a = block.first_layer.extended_output_layer
            A = {'weight': a.weight.detach().clone(), 'bias': None if a.bias is None else a.bias.detach().clone()}
            capture['singular_values'] = block.eigenvalues_extension.detach().double().clone()
    finally:
        second._compute_optimal_added_parameters = original
        clone.core.reset_computation()
    # Expose the generalized eigenvalues as a mathematical audit trail; the
    # actual A above comes exclusively from the pinned upstream solve.
    S, N = capture['S'], capture['N_z']
    es, qs = torch.linalg.eigh(S)
    inv = torch.where(es > numerical_threshold, es.clamp_min(numerical_threshold).rsqrt(), 0.)
    whitening = (qs * inv) @ qs.T
    capture['generalized_eigenvalues'] = torch.linalg.eigvalsh(whitening @ N @ N.T @ whitening).flip(0).clamp_min(0)
    capture['existing_weight_projection'] = existing_weight_projection
    if normalize:
        A = normalize_A(model, site, A)
    if score_signs:
        # Several batches contribute to each sign's true-ReLU gain.
        chosen = []
        for i in range(len(A['weight'])):
            plus, minus = _neuron(A, i), _neuron(A, i, -1)
            gain_plus = sum(score_A(model, site, plus, x, z) for (x, _), z in zip(batches, cotangents))
            gain_minus = sum(score_A(model, site, minus, x, z) for (x, _), z in zip(batches, cotangents))
            chosen.append(plus if gain_plus >= gain_minus else minus)
        A = _concat(chosen)
    return A, capture


def greedy_V(model, site, tiny_A, inputs, z, *, random_count=8, r_max=16,
             theta=.6, nu=1e-8, seed=20261014, matrix_free=False):
    """Group OMP: a neuron includes all its outgoing B coordinates.

    Gram is the logit-space Gram of the true-ReLU feature Jacobian pool.
    A group can be chosen once; duplicate input maps are removed before scoring.
    """
    if not 0 < theta <= 1 or nu <= 0 or r_max < 1 or random_count < 0:
        raise ValueError('invalid greedy selection configuration')
    tiny_A = normalize_A(model, site, tiny_A)
    items = [_neuron(tiny_A, i, sign) for i in range(len(tiny_A['weight'])) for sign in (1, -1)]
    gen = dedicated_generator(seed, 'greedy-V-pool', inputs.device)
    for _ in range(random_count):
        a = normalize_A(model, site, random_A_like(_neuron(tiny_A, 0), generator=gen))
        items.extend([a, {n: None if v is None else -v for n, v in a.items()}])
    unique = []
    for a in items:
        if not any(all((a[n] is None and b[n] is None) or
                       (a[n] is not None and b[n] is not None and torch.equal(a[n], b[n]))
                       for n in a) for b in unique):
            unique.append(a)
    if matrix_free:
        return _greedy_matrix_free(model, site, unique, inputs, z,
                                   r_max=r_max, theta=theta, nu=nu)
    features, groups = [], []
    offset = 0
    for a in unique:
        with block_products(model, site, a, inputs) as p:
            size = sum(v.numel() for _, v in p.extension_refs)
            eye = torch.eye(size, device=inputs.device, dtype=inputs.dtype)
            feature = torch.stack([p.JEv(v).double() for v in eye], 1)
        features.append(feature)
        groups.append(list(range(offset, offset + size)))
        offset += size
    J = torch.cat(features, 1)
    G = J.T @ J
    c = J.T @ z.flatten().double()
    def gain(indices):
        if not indices:
            return 0.
        idx = torch.tensor(indices, device=G.device)
        sub = G[idx[:, None], idx] + nu * torch.eye(len(idx), device=G.device, dtype=G.dtype)
        return float(c[idx].dot(torch.linalg.solve(sub, c[idx])))
    pool_gain = gain(list(range(offset)))
    selected, columns, gains = [], [], [0.]
    while len(selected) < min(r_max, len(unique)) and gains[-1] < theta * pool_gain:
        options = [(gain(columns + group), i) for i, group in enumerate(groups) if i not in selected]
        value, index = max(options)
        selected.append(index)
        columns.extend(groups[index])
        gains.append(value)
    # For an identically zero pool, retain a valid rank-one proposal.
    if not selected:
        selected = [0]
        gains.append(gain(groups[0]))
    return _concat([unique[i] for i in selected]), {'selected': selected, 'gains': gains,
        'pool_gain': pool_gain, 'threshold_reached': gains[-1] >= theta * pool_gain,
        'theta': theta, 'pool_size': len(unique), 'nu': nu}


def _greedy_matrix_free(model, site, pool, inputs, z, *, r_max, theta, nu):
    """Same ridge group-OMP objective using KE, without a dense feature Gram.

    c^T(G+nu I)^-1 c = z^T KE(KE+nu I)^-1 z. The latter
    needs only logit-space vectors; a CIFAR outgoing group can contain 4608
    coordinates, making the explicit full-pool Gram prohibitively large.
    """
    from methods.vna_joint import _inverse
    records = []
    z = z.flatten().double()
    def gain(indices):
        if not indices:
            return 0.
        A = _concat([pool[i] for i in indices])
        with block_products(model, site, A, inputs) as p:
            solution, record = _inverse(lambda v: p.KE(v).double()+nu*v, z,
                GGNConfig(cg_iters=1000, cg_tol=1e-6))
            records.append(record)
            return float(z.dot(p.KE(solution).double()))
    pool_gain = gain(list(range(len(pool))))
    selected, gains = [], [0.]
    while len(selected) < min(r_max, len(pool)) and gains[-1] < theta*pool_gain:
        value, index = max((gain(selected+[i]), i) for i in range(len(pool)) if i not in selected)
        selected.append(index); gains.append(value)
    if not selected:
        selected = [0]; gains.append(gain(selected))
    return _concat([pool[i] for i in selected]), {'selected': selected, 'gains': gains,
        'pool_gain': pool_gain, 'threshold_reached': gains[-1] >= theta*pool_gain,
        'theta': theta, 'pool_size': len(pool), 'nu': nu, 'matrix_free': True,
        'cg_converged': all(r['converged'] for r in records), 'cg': records}


def extension_trace(model, site, A, inputs):
    """Exact tr(K_E), intended for small CPU controls and refresh only."""
    with block_products(model, site, A, inputs) as p:
        n = p._call({}).numel()
        # Stream coordinate vectors: an n-by-n identity is prohibitive on
        # CIFAR statistics batches (n=25600 logits).
        v = torch.zeros(n, dtype=inputs.dtype, device=inputs.device)
        total = 0.
        for i in range(n):
            v.zero_(); v[i] = 1
            total += float(p.JET(v).double().square().sum())
        return total


def extension_trace_estimate(model, site, A, inputs, *, seed, probes=64):
    """Rademacher tr(K_E), using a repeatable shared output-probe stream."""
    if probes < 64:
        raise ValueError('trace estimate requires at least 64 probes')
    with block_products(model, site, A, inputs) as p:
        n = p._call({}).numel()
        gen = dedicated_generator(seed, 'extension-trace-crn', inputs.device)
        terms = []
        for _ in range(probes):
            v = torch.empty(n, dtype=inputs.dtype, device=inputs.device).bernoulli_(.5, generator=gen).mul_(2).sub_(1)
            terms.append(float(p.JET(v).double().square().sum()))
    values = torch.tensor(terms, dtype=torch.float64)
    return float(values.mean()), {'estimator': 'hutchinson', 'probes': probes,
                                 'std': float(values.std()),
                                 'standard_error': float(values.std() / probes**.5),
                                 'probe_seed': seed, 'common_random_numbers': True}


def trace_cost(model, site, A, inputs, *, coordinates=4):
    """Time a bounded coordinate sample and extrapolate the exact A3 trace work."""
    import time
    with block_products(model, site, A, inputs) as p:
        n = p._call({}).numel()
        v = torch.zeros(n, dtype=inputs.dtype, device=inputs.device)
        p.JET(v)  # warm up
        if inputs.is_cuda: torch.cuda.synchronize(inputs.device)
        start = time.perf_counter()
        for i in range(min(coordinates, n)):
            v.zero_(); v[i] = 1
            p.JET(v)
        if inputs.is_cuda: torch.cuda.synchronize(inputs.device)
        seconds = (time.perf_counter() - start) / min(coordinates, n)
    # 5 epochs * 3 sites * 3 ranks * 3 draws * (full + two halves), two traces.
    a3_seconds = seconds * n * 2 * (5 * 3 * 3 * 3) * 2
    return {'coordinate_vjp_seconds': seconds, 'n_out': n,
            'exact_pair_seconds': 2 * seconds * n,
            'projected_a3_exact_trace_seconds': a3_seconds,
            'device': str(inputs.device)}


def trace_matched_random_A(model, site, reference, inputs, *, seed=20261014, A=None,
                           estimator='exact'):
    a = (random_A_like(reference, generator=dedicated_generator(seed, 'random-A-trace', inputs.device))
         if A is None else _as_a_dict(A))
    cost = {}
    if estimator == 'auto':
        # Only CIFAR-scale output spaces need the cost-based switch.
        with block_products(model, site, reference, inputs) as p:
            n = p._call({}).numel()
        if n >= 1000:
            cost = trace_cost(model, site, reference, inputs)
            estimator = 'hutchinson' if cost['projected_a3_exact_trace_seconds'] > 1800 else 'exact'
        else:
            estimator = 'exact'
    if estimator == 'hutchinson':
        target, target_record = extension_trace_estimate(model, site, reference, inputs, seed=seed)
        actual, actual_record = extension_trace_estimate(model, site, a, inputs, seed=seed)
    elif estimator == 'exact':
        target, actual = extension_trace(model, site, reference, inputs), extension_trace(model, site, a, inputs)
        target_record = actual_record = {'estimator': 'exact', 'std': 0., 'probes': 0}
    else:
        raise ValueError('unknown trace estimator')
    if actual == 0 and target > 0:
        raise RuntimeError('random-A has zero trace and cannot be trace matched')
    scale = (target / actual) ** .5 if actual else 1.
    return {n: None if v is None else v * scale for n, v in a.items()}, {
        'trace_target': target, 'trace_before': actual, 'scale': scale,
        'trace_estimator': estimator, 'trace_target_record': target_record,
        'trace_random_record': actual_record, 'trace_cost': cost}


def tiny_U_from_cotangent(model, site, inputs, cotangent, rank, cfg=GGNConfig()):
    """Native projected TINY closed form driven by a mean-loss cotangent.

    Native statistics consume sum-loss derivatives. Convert the supplied
    mean-batch derivative once, before Gromo's sample averaging. No labels
    enter the pseudo-loss or the generalized singular-value solve.
    """
    cotangent = cotangent.detach().reshape(len(inputs), -1)
    batches = [(x, torch.zeros(len(x), dtype=torch.long, device=inputs.device))
               for x in inputs.split(64)]
    A, capture = tiny_V(model, site, batches, rank,
        cotangents=list((cotangent * len(inputs)).split(64)), cfg=cfg,
        existing_weight_projection=True, normalize=False, score_signs=False)
    return A, capture['singular_values']


def prepare_A(model, site, A, kind, *, reference=None, inputs=None, seed=20261014, trace_estimator="exact"):
    """Single Stage-A normalization boundary, including matched controls.

    Row scales are the normalization constants; a final scalar records the
    exact Frobenius/trace matching operation after row normalization.
    """
    before = neuron_norms(A)
    normalized = normalize_A(model, site, A)
    scales = neuron_norms(normalized) / before.clamp_min(1e-30)
    match = 1.
    trace_record = {}
    if kind == 'random_frob':
        if reference is None:
            raise ValueError('random_frob requires normalized tiny_U reference')
        norm = lambda a: sum(v.double().square().sum() for v in a.values() if v is not None).sqrt()
        match = float(norm(reference) / norm(normalized).clamp_min(1e-30))
    elif kind == 'random_trace':
        if reference is None or inputs is None:
            raise ValueError('random_trace requires reference and inputs')
        _, trace_record = trace_matched_random_A(model, site, reference, inputs,
                                                seed=seed, A=normalized, estimator=trace_estimator)
        match = trace_record['scale']
    normalized = {n: None if v is None else v * match for n, v in normalized.items()}
    return normalized, {'normalization_constant': scales.detach().cpu().tolist(),
                        'match_scale': match, 'kind': kind, **trace_record}


def subspace_sim(A, B):
    """Squared overlap of orthonormal row spaces (bias is a homogeneous input)."""
    def basis(a):
        a = _as_a_dict(a)
        rows = a['weight'].flatten(1).double()
        if a['bias'] is not None:
            rows = torch.cat((rows, a['bias'].double()[:, None]), 1)
        _, s, vh = torch.linalg.svd(rows, full_matrices=False)
        keep = s > (s.max() * max(rows.shape) * torch.finfo(s.dtype).eps)
        return vh[keep].T
    q1, q2 = basis(A), basis(B)
    rank = min(q1.shape[1], q2.shape[1])
    return float((q1.T @ q2).square().sum() / rank) if rank else 0.
