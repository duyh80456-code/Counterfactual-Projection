"""Fast fake-data smoke: loss curvature and a zero-B forward, never a gate row."""
import time
import torch
from methods.vna_joint import joint_products


class FakeMeasurement:
    def __init__(self, metadata, sites_fn, context_fn):
        self.metadata, self.sites_fn, self.context_fn = metadata, sites_fn, context_fn

    def __call__(self, model, batch, spec, ctx, *, generator):
        start = time.monotonic()
        model.eval()
        batch = tuple(t[:2] for t in batch)
        spec, context = self.context_fn(ctx, batch)
        with torch.no_grad(): logits = model(batch[0])
        residual = spec.residual(logits, context)
        curvature = spec.metric_apply(logits, torch.ones_like(logits), context)
        assert torch.isfinite(residual).all() and torch.isfinite(curvature).all()
        site = self.sites_fn(model)[0]
        A = {'weight': site.first_layer.layer.weight[:1].detach(), 'bias': None}
        with joint_products(model, [site], {site: A}, batch[0]) as products:
            assert torch.equal(products._call({}), logits.flatten())
        return [{**self.metadata, 'epoch': ctx['epoch'], 'train_acc': ctx['train_acc'],
                 'site': site.module_name, 'rank': 1, 'mu_rel': .01, 'cg_converged': False,
                 'smoke_check': 'loss curvature + zero B', 'wall_seconds': time.monotonic()-start}]
