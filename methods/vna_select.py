"""Virtual-neuron row normalization."""
import torch
from methods.vna import _as_a_dict

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
