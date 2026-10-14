"""Minimal Gromo K-FAC factors from exact softmax GGN (never observed labels).

Spatial expand convention: A averages batch and locations; G sums locations
and averages samples. EKFAC optionally replaces eigenvalue products by the
exact diagonal GGN in this eigenbasis. Damping is added to each eigenvalue,
not separately to A and G. Refresh is explicit; apply never refreshes.
"""
import copy
import torch
import torch.nn.functional as F
from methods.vna import _resolve_site
from methods.vna_ggn import GGNConfig, _correction
from projection.functional import _preserve_functional_state


class KFACBlockPreconditioner:
    def __init__(self, ema=.5, eigenvalue_correction=True):
        if not 0 <= ema < 1:
            raise ValueError('ema must be in [0,1)')
        self.ema = ema
        self.eigenvalue_correction = eigenvalue_correction
        self.blocks = {}
        self.diagonal = {}
        self.refreshes = 0

    def refresh(self, model, site, batch):
        path, block, names = _resolve_site(model, site)
        params = dict(model.named_parameters())
        refs = tuple((n, params[n]) for n in names)
        modes = {m: m.training for m in model.modules()}
        handles, captured = [], {}
        convs = {}
        try:
            model.eval()
            for label in ('first_layer', 'second_layer'):
                conv = getattr(block, label).layer
                if not isinstance(conv, torch.nn.Conv2d) or conv.groups != 1:
                    raise TypeError('KFAC requires ungrouped residual Conv2d')
                prefix = path + '.' + label + '.layer'
                convs[prefix] = conv
                def hook(_m, args, output, key=prefix):
                    captured[key] = (args[0].detach(), output)
                handles.append(conv.register_forward_hook(hook))
            with torch.autocast(device_type=batch[0].device.type, enabled=False), _preserve_functional_state(model):
                logits = model(batch[0])
                probs = logits.detach().double().softmax(-1)
                b, classes = probs.shape
                factors = {}
                for prefix, conv in convs.items():
                    x, out = captured[prefix]
                    patches = F.unfold(x, conv.kernel_size, dilation=conv.dilation,
                                       padding=conv.padding, stride=conv.stride).double()
                    if conv.bias is not None:
                        patches = torch.cat((patches, torch.ones_like(patches[:, :1])), 1)
                    rows = patches.transpose(1, 2).reshape(-1, patches.shape[1])
                    factors[prefix] = [rows.T @ rows / len(rows), torch.zeros(
                        out.shape[1], out.shape[1], dtype=torch.float64, device=out.device)]
                diagonal = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in refs
                            if not any(n.startswith(k + '.') for k in convs)}
                # Exact class-Hessian square root: each column is a Fisher
                # output cotangent. autograd.grad leaves .grad untouched.
                roots = []
                for i in range(b):
                    h = torch.diag(probs[i]) - probs[i, :, None] * probs[i, None, :]
                    eig, q = torch.linalg.eigh(h)
                    roots.append(q * eig.clamp_min(0).sqrt()[None, :])

                def cotangents():
                    for i, root in enumerate(roots):
                        for c in range(classes):
                            value = torch.zeros_like(logits)
                            value[i] = root[:, c].to(logits) / b ** .5
                            yield value

                bn_refs = tuple((n, p) for n, p in refs if n in diagonal)
                outputs = tuple(captured[k][1] for k in convs)
                for cotangent in cotangents():
                    grads = torch.autograd.grad(logits, outputs + tuple(p for _, p in bn_refs),
                                                cotangent, retain_graph=True, allow_unused=True)
                    for prefix, grad in zip(convs, grads[:len(convs)]):
                        if grad is not None:
                            rows = grad.double().permute(0, 2, 3, 1).reshape(-1, grad.shape[1])
                            factors[prefix][1].add_(rows.T @ rows)
                    for (n, _), grad in zip(bn_refs, grads[len(convs):]):
                        if grad is not None:
                            diagonal[n].add_(grad.detach().double().square())
                new_blocks = {}
                for prefix, (a, g) in factors.items():
                    old = self.blocks.get(prefix)
                    if old is not None:
                        a = self.ema * old['A'] + (1 - self.ema) * a
                        g = self.ema * old['G'] + (1 - self.ema) * g
                    ea, qa = torch.linalg.eigh(a)
                    eg, qg = torch.linalg.eigh(g)
                    lam = eg.clamp_min(0)[:, None] * ea.clamp_min(0)[None, :]
                    new_blocks[prefix] = {'A': a.detach(), 'G': g.detach(), 'QA': qa.detach(),
                                          'QG': qg.detach(), 'eigenvalues': lam.detach(),
                                          'bias': convs[prefix].bias is not None}
                if self.eigenvalue_correction:
                    # A second streamed backward sweep avoids retaining BC
                    # full parameter gradients (unbounded at CIFAR scale).
                    conv_refs = tuple((n, p) for n, p in refs if n not in diagonal)
                    corrections = {k: torch.zeros_like(v['eigenvalues']) for k, v in new_blocks.items()}
                    for cotangent in cotangents():
                        grads = torch.autograd.grad(logits, tuple(p for _, p in conv_refs),
                                                    cotangent, retain_graph=True, allow_unused=True)
                        pg = {n: torch.zeros_like(p, dtype=torch.float64) if g is None else g.detach().double()
                              for (n, p), g in zip(conv_refs, grads)}
                        for prefix, new in new_blocks.items():
                            matrix = pg[prefix + '.weight'].flatten(1)
                            if new['bias']:
                                matrix = torch.cat((matrix, pg[prefix + '.bias'][:, None]), 1)
                            corrections[prefix].add_((new['QG'].T @ matrix @ new['QA']).square())
                    for prefix, new in new_blocks.items():
                        corrected = corrections[prefix]
                        old = self.blocks.get(prefix)
                        if old is not None:
                            # Rotate the previous diagonal into the refreshed
                            # basis before EMA (squared overlap coefficients).
                            qa, qg = new['QA'], new['QG']
                            previous = (qg.T @ old['QG']).square() @ old['eigenvalues'] @ (old['QA'].T @ qa).square()
                            corrected = self.ema * previous + (1 - self.ema) * corrected
                        new['eigenvalues'] = corrected.detach()
                self.blocks = new_blocks
                self.diagonal = {n: (self.ema * self.diagonal[n] + (1-self.ema) * d
                                     if n in self.diagonal else d).detach() for n, d in diagonal.items()}
                self.refreshes += 1
        finally:
            for handle in handles:
                handle.remove()
            for m, training in modes.items():
                m.training = training

    def apply(self, vector, refs, mu):
        if mu <= 0:
            raise ValueError('mu must be positive')
        if not self.blocks:
            raise RuntimeError('refresh KFAC before apply')
        pieces = vector.double().split([p.numel() for _, p in refs])
        values = {n: v.reshape_as(p) for (n, p), v in zip(refs, pieces)}
        result = {}
        for prefix, block in self.blocks.items():
            key = prefix + '.weight'
            matrix = values[key].flatten(1)
            if block['bias']:
                matrix = torch.cat((matrix, values[prefix + '.bias'][:, None]), 1)
            qa, qg = block['QA'], block['QG']
            inverse = qg @ ((qg.T @ matrix @ qa) / (block['eigenvalues'] + mu)) @ qa.T
            result[key] = inverse[:, :values[key].flatten(1).shape[1]].reshape_as(values[key])
            if block['bias']:
                result[prefix + '.bias'] = inverse[:, -1]
        for n, d in self.diagonal.items():
            result[n] = values[n] / (d + mu)
        return torch.cat([result[n].flatten() for n, _ in refs])

    def state_dict(self):
        return copy.deepcopy({'ema': self.ema, 'eigenvalue_correction': self.eigenvalue_correction,
                              'blocks': self.blocks, 'diagonal': self.diagonal, 'refreshes': self.refreshes})

    def load_state_dict(self, state):
        self.ema = state['ema']
        self.eigenvalue_correction = state['eigenvalue_correction']
        self.blocks = copy.deepcopy(state['blocks'])
        self.diagonal = copy.deepcopy(state['diagonal'])
        self.refreshes = state['refreshes']


def vna_kfac(model, site, A, batch, preconditioner, cfg=GGNConfig(), **kwargs):
    """Exact cross products on both sides of the symmetric KFAC inverse."""
    return _correction(model, site, A, batch, cfg, 'kfac', preconditioner=preconditioner, **kwargs)
