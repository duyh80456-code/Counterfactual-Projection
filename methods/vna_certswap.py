"""T32: certify each structural swap against actual window CE and KD."""
from contextlib import ExitStack
import math
import time
import torch
from torch import nn
from torch.func import functional_call, jvp
from diagnostics.vna_align import _global_state
from methods.vna_losses import CE
from methods.vna_downstream import separable


class PrefixCache:
    """Pinned CIFAR ResNet suffix; block inputs remain valid in reverse site order."""
    def __init__(self, model, sites, inputs, measure=None):
        if not separable(model):
            raise ValueError('CertSwap prefix cache requires a sample-separable noBN network')
        self.model = model
        self.native = model.native
        self.backbone = self.native.convnet
        self.blocks = [block for name in ('stage_1', 'stage_2', 'stage_3')
                       for block in getattr(self.backbone, name)]
        self.inputs = {}
        self.indices = {}
        with ExitStack() as stack, torch.no_grad():
            for site in sites:
                block = model.get_submodule(site.first.rsplit('.', 1)[0])
                if site.first.rsplit('.', 1)[-1] != 'conv_a' or site.second.rsplit('.', 1)[-1] != 'conv_b':
                    raise ValueError('CertSwap requires an A12 internal basic-block channel')
                index = next(i for i, value in enumerate(self.blocks) if value is block)
                self.indices[site.module_name] = index
                def capture(_module, args, key=site.module_name):
                    self.inputs[key] = args[0].detach().clone()
                handle = block.register_forward_pre_hook(capture)
                stack.callback(handle.remove)
            if measure is None:
                self.logits = model(inputs).detach().reshape(len(inputs), -1).double()
            else:
                def output(_module, _args, value):
                    self.logits = value.detach().reshape(len(inputs), -1).double()
                handle = model.register_forward_hook(output)
                stack.callback(handle.remove)
                self.losses = measure()

    def forward(self, site, overrides=None):
        index = self.indices[site.module_name]
        block = self.blocks[index]
        params = {n: p.detach() for n, p in block.named_parameters()}
        params.update(overrides or {})
        value = functional_call(block, params, (self.inputs[site.module_name],))
        for block in self.blocks[index+1:]:
            value = functional_call(block, {n: p.detach() for n, p in block.named_parameters()}, (value,))
        value = self.backbone.avgpool(value).flatten(1)
        output = functional_call(self.native.fc,
            {n: p.detach() for n, p in self.native.fc.named_parameters()}, (value,))
        if isinstance(output, dict): output = output['logits']
        return output.reshape(len(value), -1).double()


def losses(logits, spec, ctx):
    ce = nn.functional.cross_entropy(logits, ctx['targets'])
    kd = spec.loss(logits, ctx)-ce if ctx.get('known_classes', 0) else ce.new_zeros(())
    return float(ce.detach()), float(kd.detach())


def eigen_step(eigenvalues, eigenvectors, gradient, mu):
    return eigenvectors @ ((eigenvectors.T @ gradient)/(eigenvalues+mu))


def certify(forward, size, reference, before, spec, ctx, *, mu0=None, budget=20):
    """Functional trial: rejection cannot mutate model parameters or optimizer state."""
    started = time.perf_counter()
    zero = reference.new_zeros(size, requires_grad=True)
    with torch.no_grad(): output = forward(zero)
    swapped = losses(output, spec, ctx)
    jL, jKD = swapped[0]-before[0], swapped[1]-before[1]
    record = dict(jL=jL, jKD=jKD, lm_trials=[], jump_seconds=time.perf_counter()-started)
    if not math.isfinite(jL) or jL > 0: return None, dict(record, reason='jL_pos')
    if not math.isfinite(jKD) or jKD > 0: return None, dict(record, reason='jKD_pos')
    started = time.perf_counter()
    ce = nn.functional.cross_entropy(forward(zero), ctx['targets'])
    gradient = torch.autograd.grad(ce, zero)[0].detach().double()
    record['gradient_seconds'] = time.perf_counter()-started
    if not bool(torch.isfinite(gradient).all()) or not bool(gradient.ne(0).any()):
        return None, dict(record, reason='g_zero')
    # Only the outgoing-slot Jacobian is materialized, never an output Gram matrix.
    started = time.perf_counter()
    basis = torch.eye(size, device=zero.device, dtype=zero.dtype)
    columns = [torch.vmap(lambda direction: jvp(forward, (zero.detach(),), (direction,))[1])(
        basis[lo:lo+4]).detach().flatten(1) for lo in range(0, size, 4)]
    J = torch.cat(columns).T.reshape(*output.shape, size)
    H = CE().metric_blocks(output.detach(), ctx)
    F = torch.einsum('nci,ncd,ndj->ij', J, H, J)
    F = (F+F.T)/2
    record['curvature_seconds'] = time.perf_counter()-started
    if not bool(torch.isfinite(F).all()): return None, dict(record, reason='lm_budget', numeric_failure='curvature')
    started = time.perf_counter()
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(F)
    except torch.linalg.LinAlgError:
        return None, dict(record, reason='lm_budget', numeric_failure='eigh')
    record['eigh_seconds'] = time.perf_counter()-started
    eigenvalues = eigenvalues.clamp_min(0)
    mu = float(mu0) if mu0 is not None else 1e-8*float(F.trace())/size
    # A nonzero gradient with exactly zero curvature still needs positive damping.
    mu = max(mu, torch.finfo(torch.float64).tiny)
    last_kd_fail = False
    started = time.perf_counter()
    for trial in range(budget):
        x = eigen_step(eigenvalues, eigenvectors, gradient, mu)
        pred = float(gradient@x-.5*x@F@x)
        with torch.no_grad():
            logits = forward(-x.to(reference))
            after = losses(logits, spec, ctx)
        rho = (swapped[0]-after[0])/pred if pred > 0 and math.isfinite(after[0]) else -math.inf
        last_kd_fail = not math.isfinite(after[1]) or after[1] > before[1]
        record['lm_trials'].append(dict(mu=mu, rho=rho if math.isfinite(rho) else None,
                                        pred_B=pred, L=after[0], KD=after[1], kd_fail=last_kd_fail))
        if math.isfinite(rho) and rho >= .25 and not last_kd_fail:
            return (-x.to(reference).detach(), logits.detach()), dict(record, reason=None,
                mu_star=mu, rho=rho, pred_B=pred, L_before=before[0], L_after=after[0],
                KD_before=before[1], KD_after=after[1], lm_seconds=time.perf_counter()-started)
        mu *= 2.**(trial+1)
        if not math.isfinite(mu): break
    return None, dict(record, reason='lm_budget', kd_fail=last_kd_fail,
                      lm_seconds=time.perf_counter()-started)


@_global_state()
def run(engine, model, sites, proposals, batch, spec, ctx, optimizer, iteration, values):
    start = time.perf_counter()
    report = dict(sites=[], accepted=[], trials=[], kd_trial_failures=0, rejections={key: 0 for key in
                  ('jL_pos','jKD_pos','g_zero','lm_budget','kd_fail')})
    if not ('gir' in engine.arm and 'cbp' not in engine.arm):
        report['seconds'] = time.perf_counter()-start
        return report
    sites = [s for s in sites if s.module_name in proposals]
    cache = values.get('certswap_cache') or PrefixCache(model, sites, batch[0])
    accepted_diagnostics = []
    for site in sorted(sites, key=lambda s: cache.indices[s.module_name], reverse=True):
        with torch.no_grad(): current = cache.forward(site, None)
        first, second = site.first_layer.layer, site.second_layer.layer
        x = cache.inputs[site.module_name]
        with torch.no_grad():
            hidden = first(x)
            for path in site.between: hidden = model.get_submodule(path)(hidden)
            scores = hidden.abs().movedim(1, 0).flatten(1).double().mean(1)
            mean = scores.mean()
            scores = scores/mean if mean > 0 else torch.zeros_like(scores)
        A = proposals[site.module_name]
        count = min(engine.certswap_topk, len(A['weight']), len(scores))
        units = torch.argsort(scores, stable=True)[:count].tolist()
        report['sites'].append(dict(site=site.module_name, eligible_channels=len(scores),
            dead=int((scores == 0).sum()), dormant=int((scores <= .025).sum()), n_candidates=count))
        last_mu = None
        for rank, unit in enumerate(units):
            candidate_start = time.perf_counter()
            weight = A['weight'][rank:rank+1].to(first.weight)
            bias = None if A.get('bias') is None else A['bias'][rank:rank+1].to(first.weight)
            if (first.bias is None) != (bias is None):
                raise ValueError('tiny_V bias must match the unchanged native slot architecture')
            with torch.no_grad():
                features = nn.functional.conv2d(x, weight, bias, first.stride, first.padding, first.dilation)
                for path in site.between: features = model.get_submodule(path)(features)
                activation_fraction = float(features.gt(0).double().mean())
                index = torch.tensor([unit], device=weight.device)
                incoming = first.weight.detach().index_copy(0, index, weight)
                overrides = {'conv_a.weight': incoming}
                if first.bias is not None:
                    incoming_bias = first.bias.detach().index_copy(0, index, bias)
                    overrides['conv_a.bias'] = incoming_bias
                outgoing = second.weight.detach().clone()
                outgoing[:, unit] = 0
            def forward(vector):
                replacement = outgoing.index_copy(1, index,
                    vector.reshape(outgoing.shape[0], 1, *outgoing.shape[2:]))
                return cache.forward(site, {**overrides, 'conv_b.weight': replacement})
            before = losses(current, spec, ctx)
            result, record = certify(forward, outgoing[:, unit].numel(), outgoing, before, spec, ctx,
                                     mu0=last_mu, budget=engine.certswap_lm_budget)
            record['activation_fraction'] = activation_fraction
            report['trials'].append(dict(record, site=site.module_name, unit=int(unit)))
            report['kd_trial_failures'] += sum(t['kd_fail'] for t in record['lm_trials'])
            if result is None:
                report['rejections'][record['reason']] += 1
                if any(t['kd_fail'] for t in record['lm_trials']): report['rejections']['kd_fail'] += 1
                continue
            vector, current = result
            with torch.no_grad():
                first.weight[unit].copy_(weight[0])
                if first.bias is not None: first.bias[unit].copy_(bias[0])
                second.weight[:, unit].copy_(vector.reshape_as(second.weight[:, unit]))
                entries = [(first.weight, unit), (second.weight, (slice(None), unit))]
                if first.bias is not None: entries.append((first.bias, unit))
                for parameter, selected in entries:
                    for value in optimizer.state.get(parameter, {}).values():
                        if isinstance(value, torch.Tensor) and value.shape == parameter.shape: value[selected] = 0
            last_mu = record['mu_star']
            record.update(site=site.module_name, unit=int(unit), redo_score=float(scores[unit]),
                was_dead=bool(scores[unit] == 0), seconds=time.perf_counter()-candidate_start,
                task=engine.task, iteration=iteration, slot_kind='certswap', ell_beta=None)
            engine.install_ages[(site.module_name, int(unit))] = dict(site=site.module_name,
                unit=int(unit), task=engine.task, iteration=iteration)
            report['accepted'].append(record)
            accepted_diagnostics.append((site, dict(weight=weight, bias=bias), record))
    # Frozen Schur values are optional diagnostics; never a certificate for a T32 decision.
    for site, A, record in accepted_diagnostics:
        record['schur_gain'] = None
        if not engine.certswap_log_schur: continue
        allowance = engine.K0*float(engine.protocol.get('baseline_iteration_seconds', 0.))
        if allowance <= 0 or time.perf_counter()-start >= allowance:
            record['schur_disabled'] = 'application_time_budget'
            continue
        callback = values.get('certswap_schur')
        if callback is None:
            record['schur_disabled'] = 'frozen_reference_unavailable'
            continue
        diagnostic_start = time.perf_counter()
        try:
            record.update(callback(site, A, record))
        except (RuntimeError, ValueError) as error:
            record['schur_disabled'] = type(error).__name__+': '+str(error)
        record['schur_seconds'] = time.perf_counter()-diagnostic_start
    report['seconds'] = time.perf_counter()-start
    return report
