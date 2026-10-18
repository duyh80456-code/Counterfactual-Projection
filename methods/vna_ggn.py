"""Loss-generic matrix-free curvature products."""
from dataclasses import dataclass
import torch
from methods.vna import block_products, loss_logit_grad, dedicated_generator
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

def solve_ggn(op, rhs, mu, cfg):
    return conjugate_gradient(lambda v: op.F(v) + mu * v, rhs.double(),
                              max_iter=cfg.cg_iters, tolerance=cfg.cg_tol, eps=1e-30)

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

def relative_damping(cfg, trace, op):
    if cfg.damping_norm == 'params':
        return max(cfg.damping_rel * trace / op.size, 1e-12)
    # Same positive floor as "params": a saturated softmax gives tr(F) = 0.
    return max(cfg.damping_rel * trace / op.g.numel(), 1e-12)
