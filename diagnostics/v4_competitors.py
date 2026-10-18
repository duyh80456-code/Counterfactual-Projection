"""Read-only CE reach and ReDo activity diagnostics for competitor probes."""
import torch
from torch import nn
from adapters.v4_competitors import branch_pairs, dormant_score
from experiments.v4_competitor_setup import read_only
from methods.vna import _Products
from methods.vna_ggn import GGNProducts, GGNConfig
from projection.cg import conjugate_gradient


class Logits(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x): return self.net(x)['logits']


def diagnose(model, batch):
    """Use the same CE metric on all arms; never advance a training engine."""
    with read_only(model):
        activities, handles = {}, []
        for name, block in branch_pairs(model):
            def capture(module, args, name=name):
                value = args[0].detach()
                axes = (0, *range(2, value.ndim))
                activities[name] = dict(dead=float((value.abs().sum(axes) == 0).double().mean()),
                    dormant=float((dormant_score(value) <= .025).double().mean()))
            handles.append(block.conv_b.register_forward_pre_hook(capture))
        try:
            with torch.no_grad(): model(batch[0])
        finally:
            for handle in handles: handle.remove()
        wrapper = Logits(model)
        products = _Products(wrapper, batch[0], tuple(n for n,p in wrapper.named_parameters()
                             if p.requires_grad), (), bn_mode='eval')
        op = GGNProducts(products, batch[1])
        trace = op.trace(GGNConfig())
        mu = .01*trace/max(1, op.size)
        result = conjugate_gradient(lambda v: op.F(v)+mu*v, op.gradient,
                                    max_iter=200, tolerance=1e-3)
        B = float(op.gradient.dot(result.solution))
        eye = torch.eye(op.logits.shape[1], dtype=torch.float64, device=op.logits.device)
        metric = torch.stack([op.H(eye[i].expand_as(op.logits)).reshape_as(op.logits)
                              for i in range(len(eye))], dim=-1)
        residual = op.g.reshape_as(op.logits)
        norm = float(torch.einsum('ni,nij,nj->', residual, torch.linalg.pinv(metric, hermitian=True), residual))
        reach = B/norm if norm else 0.
        return dict(B=B, w_norm_sq=norm, reach=reach, unreach=1-reach, mu=mu,
                    relative_residual=result.relative_residual, sites=activities, metric='CE')
