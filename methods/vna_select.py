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
    layer = model.block(site).first_layer.layer
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


def tiny_V(model, site, batches, rank, *, cotangents=None, cfg=GGNConfig(),
           existing_weight_projection=False, normalize=True, score_signs=True,
           numerical_threshold=1e-6):
    """Reuse native TINY's thresholded generalized eigensolve.

    Pseudo loss sum(f*z) has logit derivative exactly z. Native N=-M_prev
    (plus the optional existing-weight residual), so its sign matches CE TINY.
    Gromo averages N_z and S by sample count across all supplied batches.
    For z=g parity use sum-CE cotangents (batch*g), as upstream expects sums.
    Default z=H J (F+mu)^-1 J^T g (or the supplied solver's cotangent).
    """
    from experiments.vna_support import isolated_rng
    from gromo.utils.training_utils import compute_statistics
    from dual_growth.adapters import TinyAdapter
    batches = list(batches)
    if not batches or rank < 1:
        raise ValueError('need batches and positive rank')
    clone = copy.deepcopy(model)
    block = clone.block(site)
    block.second_layer.target_in_neurons = block.second_layer.in_neurons + rank
    index = next(i for i, b in enumerate(clone.core._growable_layers) if b is block)
    clone.core.set_growing_layers(scheduling_method='sequential', index=index)
    if cotangents is None:
        seed_A = {'weight': model.block(site).first_layer.layer.weight.detach()[:1], 'bias': None}
        cotangents = [filtered_cotangent(model, site, seed_A, batch, cfg) for batch in batches]
    cotangents = list(cotangents)
    if len(cotangents) != len(batches):
        raise ValueError('one cotangent required per batch')
    pseudo_batches = [(x, z.detach().reshape(len(x), -1)) for (x, _), z in zip(batches, cotangents)]
    capture = {}
    second = block.second_layer
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
             theta=.6, nu=1e-8, seed=20261014):
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


def extension_trace(model, site, A, inputs):
    """Exact tr(K_E), intended for small CPU controls and refresh only."""
    with block_products(model, site, A, inputs) as p:
        n = p._call({}).numel()
        eye = torch.eye(n, dtype=inputs.dtype, device=inputs.device)
        return sum(float(p.JET(v).double().square().sum()) for v in eye)


def trace_matched_random_A(model, site, reference, inputs, *, seed=20261014):
    a = random_A_like(reference, generator=dedicated_generator(seed, 'random-A-trace', inputs.device))
    target, actual = extension_trace(model, site, reference, inputs), extension_trace(model, site, a, inputs)
    if actual == 0 and target > 0:
        raise RuntimeError('random-A has zero trace and cannot be trace matched')
    scale = (target / actual) ** .5 if actual else 1.
    return {n: None if v is None else v * scale for n, v in a.items()}, {'trace_target': target, 'trace_before': actual, 'scale': scale}
