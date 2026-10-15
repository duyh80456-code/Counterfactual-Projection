"""Loss-generic GGN solvers. H contains the single 1/batch normalization.

Network products use the native dtype; curvature actions, CG and scalar
reductions use float64. All public corrections are read-only and eval-BN.
"""
from dataclasses import dataclass
import time
import torch
from methods.vna import (block_products, loss_logit_grad, _unflatten,
                         dedicated_generator, _resolve_site)
from projection.cg import conjugate_gradient
from methods.vna_losses import CE


@dataclass(frozen=True)
class GGNConfig:
    damping_rel: float = 1e-2
    cg_iters: int = 200
    cg_tol: float = 1e-6
    hutchinson_probes: int = 8
    seed: int = 20261014
    project_bn: bool = True
    cache_block_input: bool = False
    damping_norm: str = "outputs"

    def __post_init__(self):
        if self.damping_norm not in ("params", "outputs"):
            raise ValueError("damping_norm must be params or outputs")


class GGNProducts:
    """F, cross-curvature and H without a batch-by-batch dense matrix."""
    def __init__(self, products, targets, loss_spec=None, ctx=None):
        self.p = products
        logits = products._call({}).reshape(len(products.inputs), -1).detach()
        self.probs = logits.double().softmax(-1)
        self.loss_spec = CE() if loss_spec is None else loss_spec
        self.ctx = {'targets': targets, **(ctx or {})}
        self.logits = logits.double()
        self.g = loss_logit_grad(logits, targets, loss_spec, ctx)
        self.gradient = products.JT(self.g).double()
        self.size = self.gradient.numel()

    def H(self, v):
        v = v.double().reshape_as(self.probs)
        return self.loss_spec.metric_apply(self.logits, v, self.ctx).flatten()

    def F(self, v):
        return self.p.JT(self.H(self.p.Jv(v))).double()

    def COE(self, v):
        return self.p.JT(self.H(self.p.JEv(v))).double()

    def CEO(self, v):
        return self.p.JET(self.H(self.p.Jv(v))).double()

    def trace(self, cfg):
        if cfg.hutchinson_probes < 1:
            raise ValueError('hutchinson_probes must be positive')
        gen = dedicated_generator(cfg.seed, 'ggn-trace', self.gradient.device)
        total = 0.
        for _ in range(cfg.hutchinson_probes):
            v = torch.empty_like(self.gradient).bernoulli_(.5, generator=gen).mul_(2).sub_(1)
            total += float(v.dot(self.F(v)))
        return max(total / cfg.hutchinson_probes, 0.)


def bn_conv_weights(model, site):
    """Locate convs followed by BN within each native residual growing layer."""
    path, block, _ = _resolve_site(model, site)
    result = {}
    if hasattr(site, 'resolve'):
        modules = dict(model.named_modules())
        if isinstance(modules[site.first], torch.nn.Conv2d) and any(
                isinstance(modules[n], torch.nn.modules.batchnorm._BatchNorm)
                or 'BatchNorm' in type(modules[n]).__name__ for n in site.between):
            result[site.first + '.weight'] = modules[site.first].weight.detach()
        if site.native_path:
            result.update(bn_conv_weights(model, site.native_path))
        return result
    for label in ('first_layer', 'second_layer'):
        layer = getattr(block, label)
        if isinstance(layer.layer, torch.nn.Conv2d) and any(
                isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                for m in layer.post_layer_function.modules()):
            result[path + '.' + label + '.layer.weight'] = layer.layer.weight.detach()
    return result


def project_bn_tangent(model, site, delta, enabled=True):
    result = {n: v.clone() for n, v in delta.items()}
    if enabled:
        for n, w in bn_conv_weights(model, site).items():
            if n in result:
                wd, d = w.double(), result[n].double()
                denom = wd.square().sum()
                if denom > 0:
                    result[n] = (d - d.mul(wd).sum() / denom * wd).to(result[n])
    return result


def bn_weight_norms(model, site):
    return {n: float(w.double().norm()) for n, w in bn_conv_weights(model, site).items()}


def norm_match(delta, reference):
    norm = sum(float(v.double().square().sum()) for v in delta.values()) ** .5
    target = sum(float(v.double().square().sum()) for v in reference.values()) ** .5
    scale = target / norm if norm else 0.
    return {n: v * scale for n, v in delta.items()}, scale


def solve_ggn(op, rhs, mu, cfg):
    return conjugate_gradient(lambda v: op.F(v) + mu * v, rhs.double(),
                              max_iter=cfg.cg_iters, tolerance=cfg.cg_tol, eps=1e-30)


def _counts(p):
    return {k: getattr(p, k) for k in ('forward_calls', 'jvp_calls', 'vjp_calls',
                                      'extension_jvp_calls', 'extension_vjp_calls')}


def _correction(model, site, A, batch, cfg, variant, *, mu=None, trace_F=None, preconditioner=None, loss_spec=None, ctx=None):
    start = time.perf_counter()
    with block_products(model, site, A, batch[0], cache_block_input=cfg.cache_block_input) as p:
        op = GGNProducts(p, batch[1], loss_spec, ctx)
        trace = op.trace(cfg) if trace_F is None else float(trace_F)
        damping = relative_damping(cfg, trace, op) if mu is None else float(mu)
        if damping <= 0:
            raise ValueError('mu must be positive')
        records = []
        def inverse(v):
            if preconditioner is not None:
                return preconditioner.apply(v, p.parameter_refs, damping)
            r = solve_ggn(op, v, damping, cfg)
            records.append({'iterations': r.iterations, 'converged': r.converged,
                            'relative_residual': r.relative_residual})
            return r.solution
        if variant == 'lite':
            # tr(H K) = tr(F), n is the number of flattened logits.
            c = max((trace / op.g.numel()) ** 2, 1e-30)
            z = op.H(p.Jv(op.gradient))
            d = p.JT(op.H(p.KE(z))).double() / c
        else:
            u = inverse(op.gradient)
            z = op.H(p.Jv(u))
            rhs = op.F(u) if variant == 'ngd' else op.COE(op.CEO(u))
            d = inverse(rhs)
            c = None
        raw_gain = float(op.gradient.dot(d))
        delta = project_bn_tangent(model, site, _unflatten(d, p.parameter_refs), cfg.project_bn)
        flat = torch.cat([delta[n].flatten() for n, _ in p.parameter_refs]).double()
        diag = {'variant': variant, 'mu': damping, 'trace_F': trace,
                'trace_F_per_parameter': trace / op.size, **damping_metadata(cfg, trace, op, damping), 'cg': records, 'c': c,
                'V': float(p.JET(z).double().square().sum()),
                'raw_gain': raw_gain, 'realized_gain': float(op.gradient.dot(flat)),
                'project_bn': cfg.project_bn, 'weight_norms': bn_weight_norms(model, site),
                'finite': bool(torch.isfinite(flat).all()), **_counts(p)}
    diag['seconds'] = time.perf_counter() - start
    diag['skipped'] = not diag['finite']
    diag['skip_reason'] = 'nonfinite' if diag['skipped'] else None
    return ({} if diag['skipped'] else delta), diag


def vna_exact_ggn(model, site, A, batch, cfg=GGNConfig(), **kwargs):
    return _correction(model, site, A, batch, cfg, 'exact', **kwargs)


def ngd_control(model, site, A, batch, cfg=GGNConfig(), *, reference=None, **kwargs):
    delta, diag = _correction(model, site, A, batch, cfg, 'ngd', **kwargs)
    if reference is not None:
        delta, diag['norm_match_scale'] = norm_match(delta, reference)
        diag['realized_gain'] *= diag['norm_match_scale']
    return delta, diag


def vna_lite_ggn(model, site, A, batch, cfg=GGNConfig(), **kwargs):
    return _correction(model, site, A, batch, cfg, 'lite', **kwargs)


def filtered_cotangent(model, site, A, batch, cfg=GGNConfig(), *, mu=None, preconditioner=None, loss_spec=None, ctx=None):
    """z = H J M J^T g; V(A) = ||J_E(A)^T z||² before projection.

    For exact M, z = (HK + mu I)^-1 H K g. H is applied
    outside the parameter inverse; omitting it selects a different metric.
    """
    with block_products(model, site, A, batch[0]) as p:
        op = GGNProducts(p, batch[1], loss_spec, ctx)
        damping = relative_damping(cfg, op.trace(cfg), op) if mu is None else mu
        u = (solve_ggn(op, op.gradient, damping, cfg).solution if preconditioner is None
             else preconditioner.apply(op.gradient, p.parameter_refs, damping))
        return op.H(p.Jv(u)).reshape_as(op.probs).detach()


def damping_metadata(cfg, trace, op, mu):
    return {'mu': float(mu), 'mu_rel': cfg.damping_rel,
            'damping_norm': cfg.damping_norm, 'trace_F': float(trace),
            'n_out': op.g.numel(), 'p': op.size,
            'mu_rel_equiv_params': float(mu) / (trace / op.size) if trace > 0 else None}


def relative_damping(cfg, trace, op):
    if cfg.damping_norm == 'params':
        return max(cfg.damping_rel * trace / op.size, 1e-12)
    # Same positive floor as "params": a saturated softmax gives tr(F) = 0.
    return max(cfg.damping_rel * trace / op.g.numel(), 1e-12)
