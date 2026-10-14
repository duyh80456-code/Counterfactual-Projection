"""Output loss curvature; residual and metric both contain exactly one 1/n.

ctx is a mapping with ``targets`` and, for real measurements, ``loss_fn``:
loss_fn(z, ctx) must call the actual upstream loss object, with its reduction.
No empirical Fisher is used. Squared error defaults to sum of output errors
per sample, mean over samples; reduction='mean' additionally divides by d.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F


def _cov(p, v):
    return p * (v - (p * v).sum(-1, keepdim=True))


@dataclass
class LossSpec:
    output_dim: int | None = None
    name: str = 'loss'

    def loss(self, z, ctx):
        raise NotImplementedError

    def residual(self, z, ctx):
        """Autograd rows dL/dz, including the upstream mean-batch 1/n once.

        Evaluate in float64 (as legacy mean-CE products do), on a detached
        copy, without attaching a graph to the baseline or accumulating grads.
        The supplied loss_fn(z, ctx) is authoritative for the reduction.
        """
        with torch.enable_grad():
            output = z.detach().double().requires_grad_(True)
            loss = ctx['loss_fn'](output, ctx) if 'loss_fn' in ctx else self.loss(output, ctx)
            return torch.autograd.grad(loss, output)[0].detach()

    def metric_apply(self, z, v, ctx):
        raise NotImplementedError


@dataclass
class CE(LossSpec):
    name: str = 'CE'

    def loss(self, z, ctx):
        return F.cross_entropy(z, ctx['targets'])

    def metric_apply(self, z, v, ctx):
        return _cov(z.double().softmax(-1), v.double().reshape_as(z)) / len(z)


@dataclass
class MSE(LossSpec):
    name: str = 'MSE'
    reduction: str = 'batchmean'

    def __post_init__(self):
        if self.reduction not in ('batchmean', 'mean'):
            raise ValueError('MSE reduction must be batchmean or mean')

    def loss(self, z, ctx):
        value = (z - ctx['targets']).square()
        return value.sum() / len(z) if self.reduction == 'batchmean' else value.mean()

    def metric_apply(self, z, v, ctx):
        if 'loss_fn' in ctx:
            # Upstream MSE can average output coordinates as well as samples.
            # Read the quadratic coefficient from its actual reduction.
            with torch.enable_grad():
                output = z.detach().double().requires_grad_(True)
                g = torch.autograd.grad(ctx['loss_fn'](output, ctx), output, create_graph=True)[0]
                return torch.autograd.grad(g, output, v.double().reshape_as(output))[0].detach()
        denominator = len(z) if self.reduction == 'batchmean' else z.numel()
        return 2 * v.double().reshape_as(z) / denominator


@dataclass
class KDRepDistiller(CE):
    name: str = 'KDRepDistiller'
    alpha: float = .1
    beta: float = .9
    T: float = 4.

    def loss(self, z, ctx):
        # RepDistiller/distiller_zoo/KD.py: DistillKL.forward.
        kd = F.kl_div(F.log_softmax(z / self.T, dim=1),
                      F.softmax(ctx['teacher_logits'] / self.T, dim=1),
                      reduction='sum') * (self.T ** 2) / z.shape[0]
        return self.alpha * F.cross_entropy(z, ctx['targets']) + self.beta * kd

    def metric_apply(self, z, v, ctx):
        z, v = z.double(), v.double().reshape_as(z)
        return (self.alpha * _cov(z.softmax(-1), v) +
                self.beta * _cov((z / self.T).softmax(-1), v)) / len(z)


@dataclass
class ICaRLPyCIL(CE):
    name: str = 'ICaRLPyCIL'
    T: float = 2.
    known_classes: int = 0

    def loss(self, z, ctx):
        known = ctx.get('known_classes', self.known_classes)
        loss = F.cross_entropy(z, ctx['targets'])
        if known:
            # PyCIL/models/icarl.py: _KD_loss (no T squared).
            pred = torch.log_softmax(z[:, :known] / self.T, dim=1)
            soft = torch.softmax(ctx['teacher_logits'][:, :known] / self.T, dim=1)
            loss = loss + -1 * torch.mul(soft, pred).sum() / pred.shape[0]
        return loss

    def metric_apply(self, z, v, ctx):
        z, v = z.double(), v.double().reshape_as(z)
        result = _cov(z.softmax(-1), v)
        known = ctx.get('known_classes', self.known_classes)
        if known:
            result[:, :known] += _cov((z[:, :known] / self.T).softmax(-1),
                                     v[:, :known]) / self.T ** 2
        return result / len(z)
