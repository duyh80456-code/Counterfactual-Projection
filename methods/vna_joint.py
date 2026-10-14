"""Matrix-free joint virtual kernel with one shared pair of GGN solves."""
from contextlib import contextmanager, ExitStack
import time
import torch
from methods.vna import (_Products, _resolve_site, _virtual_b_context, _as_a_dict,
                         _unflatten, _hutchinson_trace_and_diag, VNAConfig, _cg_record)
from methods.vna_ggn import GGNProducts, GGNConfig, project_bn_tangent, damping_metadata, relative_damping
from projection.functional import _preserve_functional_state
from projection.cg import conjugate_gradient
from probe.gromo_adapter import _snapshot, _assert_unchanged


@contextmanager
def joint_products(model, sites, A_by_site, inputs, o_params=None, *, cache_block_input=False):
    """At B=0 cross terms between virtual sites vanish; concatenate B scopes."""
    sites = list(sites)
    if not sites or len(set(sites)) != len(sites):
        raise ValueError('need distinct nonempty sites')
    original = dict(model.named_parameters())
    if o_params is None:
        names = tuple(n for n, p in original.items() if p.requires_grad)
    else:
        supplied = list(o_params)
        requested_names = {p for p in supplied if isinstance(p, str)}
        ids = {id(p) for p in supplied if not isinstance(p, str)}
        names = tuple(n for n, p in original.items() if n in requested_names or id(p) in ids)
        if len(names) != len(supplied):
            raise ValueError('o_params must name distinct model parameters')
    if not names:
        raise ValueError('need a nonempty original-parameter scope')
    if any(hasattr(site, 'resolve') for site in sites):
        if not all(hasattr(site, 'resolve') for site in sites):
            raise ValueError('joint sites must use the same resolver')
        from methods.vna_sites import hook_products
        with hook_products(model, sites, A_by_site, inputs, names,
                           cache_block_input=cache_block_input) as products:
            yield products
        return
    before = _snapshot(model)
    modes = {m: m.training for m in model.modules()}
    try:
        model.eval()
        with ExitStack() as stack:
            stack.enter_context(_preserve_functional_state(model))
            ids = set()
            for site in sites:
                _, block, _ = _resolve_site(model, site)
                outgoing = stack.enter_context(_virtual_b_context(model, block, _as_a_dict(A_by_site[site])))
                ids.update(id(p) for p in outgoing.parameters())
            ext = tuple(n for n, p in model.named_parameters() if id(p) in ids)
            local_scope = len(sites) == 1 and set(names) == set(_resolve_site(model, sites[0])[2])
            with torch.autocast(device_type=inputs.device.type, enabled=False):
                yield _Products(model, inputs, names, ext, bn_mode='eval',
                                cache_block_input=cache_block_input and local_scope)
    finally:
        for m, mode in modes.items():
            m.training = mode
        _assert_unchanged(model, before)


def _inverse(action, rhs, cfg):
    # Scale-invariant CG: absolute breakdown guards must not erase small CE
    # residuals at interpolating checkpoints. Solve for a unit-norm RHS.
    scale = rhs.norm()
    unit = rhs / scale if scale > 0 else rhs
    result = conjugate_gradient(action, unit, max_iter=cfg.cg_iters,
                                tolerance=cfg.cg_tol, eps=1e-30)
    return result.solution * scale, _cg_record(result, include_breakdown=True)


def joint_correction(model, sites, A_by_site, batch, cfg=GGNConfig(), o_params=None,
                     *, mu=None, cotangent=None, loss_spec=None, ctx=None):
    """Return (d, V, rho, record), without installing or applying neurons.

    GGNConfig selects trace(F)/p or trace(F)/n_out via damping_norm.
    VNAConfig is an explicit legacy Euclidean compatibility audit, for the
    required comparison to vna_block_correction; Stage A uses GGNConfig.
    """
    start = time.perf_counter()
    with joint_products(model, sites, A_by_site, batch[0], o_params,
                        cache_block_input=cfg.cache_block_input) as p:
        op = GGNProducts(p, batch[1], loss_spec, ctx)
        if cotangent is not None:
            op.g = cotangent.detach().flatten().double()
            op.gradient = p.JT(op.g).double()
        if isinstance(cfg, VNAConfig):
            trace, jacobi = _hutchinson_trace_and_diag(p, op.g.numel(), cfg, op.g.device)
            damping = cfg.damping_rel * trace / op.g.numel() if mu is None else mu
            action = lambda v: p.K(v).double() + damping * v
            z, first = _inverse(action, p.Jv(op.gradient).double(), cfg)
            q, second = _inverse(action, p.KE(z).double(), cfg)
            d = p.JT(q).double() * cfg.lambda_vna
        else:
            trace = op.trace(cfg)
            damping = relative_damping(cfg, trace, op) if mu is None else mu
            action = lambda v: op.F(v) + damping * v
            u, first = _inverse(action, op.gradient, cfg)
            z = op.H(p.Jv(u))
            d, second = _inverse(action, op.COE(p.JET(z).double()), cfg)
        V = float(p.JET(z).double().square().sum())
        U = float(p.JET(op.g).double().square().sum())
        norm = float(op.gradient.square().sum())
        rho = V / norm if norm else 0.
        if isinstance(cfg, VNAConfig):
            rho *= cfg.lambda_vna
        delta = _unflatten(d, p.parameter_refs)
        if getattr(cfg, 'project_bn', False):
            for site in sites:
                delta = project_bn_tangent(model, site, delta)
        flat = torch.cat([delta[n].flatten() for n, _ in p.parameter_refs]).double()
        record = {'cg': [first, second], 'cg_converged': first['converged'] and second['converged'],
                  'cg_iters_used': first['iterations'] + second['iterations'],
                  'cg_rel_residual': max(first['relative_residual'], second['relative_residual']),
                  'mu': float(damping), 'trace': trace, 'U': U, 'V': V,
                  'eta': V / U if U else 0., 'rho': rho,
                  'raw_gain': float(op.gradient.dot(d)),
                  'realized_gain': float(op.gradient.dot(flat)),
                  'finite': bool(torch.isfinite(flat).all())}
    if isinstance(cfg, GGNConfig):
        record.update(damping_metadata(cfg, trace, op, damping))
    else:
        record.update(mu_rel=cfg.damping_rel, damping_norm='outputs', trace_K=trace,
                      n_out=op.g.numel(), p=op.size, damping_metric='phase0_euclidean_trace_K')
    record['seconds'] = time.perf_counter() - start
    return delta, V, rho, record
