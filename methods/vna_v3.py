"""Training VNA–GIR v3: matrix-free certified solves, protected correction and frozen GIR."""
from contextlib import contextmanager, ExitStack
from copy import copy, deepcopy
import math
import time
import torch
from torch import nn
from torch.func import functional_call
from diagnostics.vna_align import _global_state
from methods.vna import dedicated_generator, _unflatten
from methods.vna_ggn import GGNConfig, GGNProducts
from methods.vna_losses import CE
from methods.vna_sites import HookProducts, HookSite
from methods.vna_exact import gram, columns, OutputSolve, FrozenGram, OutputCG, CGConvergenceError, solve_image, solve_norm2
from experiments.vna_resume import rng_state, restore_rng
from projection.functional import _preserve_functional_state


@contextmanager
def training_products(model, sites, proposals, inputs):
    modes = {m: m.training for m in model.modules()}
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    virtual_sites = []
    for site in sites:
        proxy = copy(site)
        proposal = proposals[site.module_name]
        def features(x, a, site=site, proposal=proposal):
            value = a(x)
            for path in site.between:
                module = model.get_submodule(path)
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    axes = (0, *range(2, value.ndim))
                    value = (value-value.mean(axes, keepdim=True)) / (
                        value.var(axes, unbiased=False, keepdim=True)+module.eps).sqrt()
                    shape = (1, -1, *((1,)*(value.ndim-2)))
                    gamma = proposal.get('bn_weight', torch.ones(value.shape[1], device=value.device))
                    shift = proposal.get('bn_bias', torch.zeros(value.shape[1], device=value.device))
                    value = value*gamma.to(value).reshape(shape)+shift.to(value).reshape(shape)
                else:
                    value = module(value)
            if site.kind == 'conv_linear': value = value.flatten(1)
            return value*proposal.get('feature_scale', 1.)
        proxy.virtual_features = features
        virtual_sites.append(proxy)
    try:
        with _global_state(), _preserve_functional_state(model), torch.enable_grad(), torch.autocast(device_type=inputs.device.type, enabled=False):
            model.train()
            names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
            products = HookProducts(model, virtual_sites,
                {s: proposals[s.module_name] for s in virtual_sites}, inputs, names, bn_mode='train')
            # A virtual neuron contributes outgoing weights only, not a second-layer bias.
            refs = []
            for _, _, b, prefix in products.virtual:
                b.register_parameter('bias', None)
                refs.extend((prefix+n, p.detach()) for n, p in b.named_parameters())
            products.extension_refs = tuple(refs)
            yield products
    finally:
        with torch.no_grad():
            for n, b in model.named_buffers(): b.copy_(buffers[n])
        for m, mode in modes.items(): m.training = mode


def statistics(model, site, batch, cotangents):
    """Full-window centred input/cotangent statistics, without updating BN or RNG."""
    captured = {}
    first, second = site.first_layer.layer, site.second_layer.layer
    def incoming(module, args): captured['x'] = args[0]
    def outgoing(module, args, output): captured['out'] = output
    buffers = {n: b.clone() for n, b in model.named_buffers()}
    modes = {m: m.training for m in model.modules()}
    try:
        with _global_state(), _preserve_functional_state(model), torch.enable_grad(), ExitStack() as stack:
            model.train()
            h = first.register_forward_pre_hook(incoming); stack.callback(h.remove)
            h = second.register_forward_hook(outgoing); stack.callback(h.remove)
            output = functional_call(model, ({n: p for n, p in model.named_parameters()}, buffers), (batch[0],))
            residuals = [torch.autograd.grad(output, captured['out'], z.reshape_as(output).to(output),
                         retain_graph=i<len(cotangents)-1)[0].detach().double()
                         for i, z in enumerate(cotangents)]
            x = captured['x'].detach().double()
    finally:
        for m, mode in modes.items(): m.training = mode
    if site.kind == 'linear':
        x = x.reshape(-1, first.in_features)
        if first.bias is not None: x = torch.cat((x, torch.ones_like(x[:, :1])), 1)
        if any(isinstance(model.get_submodule(n), nn.modules.batchnorm._BatchNorm) for n in site.between):
            x = x-x.mean(0)
        S = x.T@x/len(x)
        return S, [x.T@r.reshape(len(x), -1) for r in residuals]
    if site.kind != 'conv': raise ValueError('v3 training sites must be linear or basic-block conv1')
    input_height = x.shape[-2]
    x = nn.functional.unfold(x, first.kernel_size, dilation=first.dilation,
                             padding=first.padding, stride=first.stride)
    if first.bias is not None: x = torch.cat((x, torch.ones_like(x[:, :1])), 1)
    if any(isinstance(model.get_submodule(n), nn.modules.batchnorm._BatchNorm) for n in site.between):
        x = x-x.mean((0, 2), keepdim=True)
    S = torch.einsum('ndl,nel->de', x, x)/(x.shape[0]*x.shape[2])
    height = (input_height+2*first.padding[0]-first.dilation[0]*(first.kernel_size[0]-1)-1)//first.stride[0]+1
    width = x.shape[2]//height
    patches = nn.functional.unfold(x.reshape(len(x), x.shape[1], height, width),
        second.kernel_size, dilation=second.dilation, padding=second.padding, stride=second.stride)
    patches = patches.reshape(len(x), x.shape[1], -1, patches.shape[-1])
    return S, [torch.einsum('ndkl,nol->dko', patches, r.flatten(2)).flatten(1) for r in residuals]


def generalized_rows(matrix, S, rank, *, euclidean=False):
    if euclidean:
        # GradMax takes N directly: SVD avoids squaring its condition number in NNᵀ.
        scale = matrix.abs().max()
        if scale == 0: return matrix.new_empty((0, len(S)))
        vectors, values, _ = torch.linalg.svd(matrix/scale, full_matrices=False)
        return vectors[:, torch.where(values > 0)[0][:rank]].T
    ridge = 1e-6*float(S.trace())/len(S)
    if ridge <= 0: return S.new_empty((0, len(S)))
    L = torch.linalg.cholesky(S+ridge*torch.eye(len(S), device=S.device, dtype=S.dtype))
    whitened = torch.linalg.solve_triangular(L, matrix, upper=False)
    whitened = torch.linalg.solve_triangular(L, whitened.T, upper=False).T
    values, vectors = torch.linalg.eigh((whitened+whitened.T)/2)
    rows = torch.linalg.solve_triangular(L.T, vectors, upper=True).T
    keep = torch.where(values > 0)[0].flip(0)[:rank]
    return rows[keep]


def rows_proposal(site, rows):
    first = site.first_layer.layer
    norm = first.weight.detach().flatten(1).double().norm(dim=1).mean()
    if first.bias is not None:
        norm = torch.cat((first.weight.detach().flatten(1).double(), first.bias.detach().double()[:, None]), 1).norm(dim=1).mean()
    rows = rows*norm/rows.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return dict(weight=rows[:, :first.weight[0].numel()].reshape(-1, *first.weight.shape[1:]).to(first.weight),
                bias=rows[:, -1].to(first.bias) if first.bias is not None else None)


def seed_proposals(sites):
    return {s.module_name: dict(weight=s.first_layer.layer.weight[:1].detach(),
        bias=None if s.first_layer.layer.bias is None else s.first_layer.layer.bias[:1].detach()) for s in sites}


def beta_rule(solve, rhs, action_B, H, uN, VO, chi, eold):
    d0 = solve('d_N', rhs, 0.)
    leak = lambda d: max(float(d@action_B(d)), 0.)
    if chi == 0 or float(eold.norm()) == 0:
        return d0, dict(beta=0., beta_branch='a', leak=leak(d0), epsilon_star=None, beta_suf=None)
    if VO == 0:
        return d0, dict(beta=0., beta_branch='b', leak=leak(d0), epsilon_star=None, beta_suf=None)
    epsilon = VO**2/chi
    suf = chi*max(float(uN@H(uN)), 0.)/VO**2
    if leak(d0) <= epsilon:
        return d0, dict(beta=0., beta_branch='c_zero', leak=leak(d0), epsilon_star=epsilon, beta_suf=suf)
    if not math.isfinite(suf) or suf <= 0:
        return d0, dict(beta=0., beta_branch='nonfinite_chi', leak=leak(d0), epsilon_star=epsilon, beta_suf=suf, invalid_beta=True)
    lo, hi = math.log(suf)-8*math.log(10.), math.log(suf)
    candidate = solve('d_N', rhs, math.exp(hi))
    if leak(candidate) > epsilon:
        return candidate, dict(beta=math.exp(hi), beta_branch='uncertified_endpoint', leak=leak(candidate), epsilon_star=epsilon, beta_suf=suf, invalid_beta=True)
    low = solve('d_N', rhs, math.exp(lo))
    if leak(low) <= epsilon:
        hi, candidate = lo, low
    else:
        for _ in range(4):
            mid = (lo+hi)/2
            value = solve('d_N', rhs, math.exp(mid))
            if leak(value) <= epsilon: hi, candidate = mid, value
            else: lo = mid
    return candidate, dict(beta=math.exp(hi), beta_branch='c_bisect', leak=leak(candidate),
                          epsilon_star=epsilon, beta_suf=suf)


def power_lambda_max(action, size, generator, iterations=20, dtype=torch.float64, device='cpu'):
    """lambda_max of a symmetric PSD operator by matrix-free power iteration (Rayleigh quotient)."""
    if size == 0: return 0.
    v = torch.randn(size, dtype=dtype, device=device, generator=generator)
    v /= v.norm().clamp_min(1e-300)
    value = 0.
    for _ in range(iterations):
        w = action(v).double()
        value = float(v @ w)
        norm = float(w.norm())
        if norm == 0 or not math.isfinite(norm): return max(value, 0.)
        v = w/norm
    return max(value, 0.)


def trust_region_step(g, d, Fd, mu, delta):
    """T26(e): choose the damped direction scale inside the GN trust region."""
    gain = float(g @ d)
    if gain <= 0:
        return 0., 0., 'none'
    norm_d = float(d.norm())
    if norm_d == 0:
        return 0., 0., 'none'
    denom = float(d @ Fd)
    s_f = gain / denom if denom > 0 else float('inf')
    s_delta = float(delta) / norm_d
    s = min(s_f, s_delta)
    return s, s_f, 's_F' if s_f <= s_delta else 'Delta'


def net_gain(e_beta, H_beta, J, J_prime, f_u, mu):
    """T28(a) net gain for a temporarily installed live-slot candidate."""
    metric = H_beta if callable(H_beta) else lambda value: H_beta @ value
    def q(Jx, e):
        gram = Jx.T @ metric(Jx) + mu * torch.eye(Jx.shape[1], dtype=Jx.dtype, device=Jx.device)
        rhs = Jx.T @ e
        return -.5 * float(rhs @ torch.linalg.solve(gram, rhs))
    e_prime = e_beta - metric(f_u)
    jump = -float(e_beta @ f_u) + .5 * float(f_u @ metric(f_u))
    return -jump + q(J, e_beta) - q(J_prime, e_prime)


def diagnose(model, batch, *, sites=None, spec=None, ctx=None, mu=None):
    """Read-only T25 probe; all model state, buffers and RNG are restored."""
    spec = spec or CE()
    ctx = dict(ctx or {})
    if 'targets' not in ctx:
        ctx['targets'] = batch[1]
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    buffer_state = {n: b.clone() for n, b in buffers.items()}
    parameter_state = {n: p.detach().clone() for n, p in parameters}
    gradients = {n: None if p.grad is None else p.grad.detach().clone() for n, p in parameters}
    modes = {m: m.training for m in model.modules()}
    try:
        with _global_state(), torch.enable_grad():
            model.train()
            with torch.no_grad():
                output = model(batch[0])
                if isinstance(output, dict): output = output['logits']
                logits = output.reshape(len(batch[0]), -1).double()
            e = spec.residual(logits, ctx).reshape(-1).double()
            with training_products(model, sites or [], seed_proposals(sites or []), batch[0]) as products:
                op = GGNProducts(products, batch[1], spec, ctx)
                G = FrozenGram(products)
                timing = dict(solver='cg', m=e.numel(), gram_bytes=0)
                if mu is None:
                    mu = max(1e-12, 1e-2*op.trace(GGNConfig())/max(op.size, 1))
                blocks = spec.metric_blocks(logits, ctx)
                factor = None
                try:
                    factor = OutputCG(G, blocks, mu)
                    y = factor.solve(e)
                    B = max(float(e @ solve_image(G,y)), 0.)
                    flag = None
                except CGConvergenceError:
                    B, flag = None, 'cg_not_converged'
                except torch.linalg.LinAlgError:
                    B, flag = None, 'factor_fail'
                timing['solves'] = factor.records if factor is not None else []
            target = e.reshape_as(logits)
            norm_w = max(float(torch.einsum('ni,nij,nj->', target,
                torch.linalg.pinv(blocks, hermitian=True), target)), 0.)
            result = dict(B=B, norm_w2=norm_w, mu_diag=mu, diagnostic_flag=flag,
                          **timing, sites={})
        if sites is not None:
            captures = {}
            handles = [site.second_layer.layer.register_forward_pre_hook(
                lambda _m, args, name=site.module_name: captures.__setitem__(name, args[0].detach()))
                for site in sites]
            with _global_state(), torch.no_grad(): model(batch[0])
            for handle in handles: handle.remove()
            for site in sites:
                value = captures.get(site.module_name)
                if value is None: continue
                flat = value.movedim(1, -1).reshape(-1, value.shape[1])
                dormant = flat.abs().mean(0) / flat.abs().mean().clamp_min(1e-30) <= .025
                result['sites'][site.module_name] = dict(
                    dead=float(flat.eq(0).all(0).double().mean()),
                    dormant=float(dormant.double().mean()))
        return result
    finally:
        with torch.no_grad():
            model.load_state_dict({**parameter_state, **buffer_state}, strict=False)
            for name, value in gradients.items(): dict(model.named_parameters())[name].grad = None if value is None else value.clone()
        for module, mode in modes.items(): module.training = mode
        # _global_state restores torch, Python, and NumPy RNG state.


def schedule_from_points(points, iterations_per_epoch, final_classes=100):
    points = list(points)
    if not points: raise ValueError('schedule calibration needs at least one point')
    tasks = sorted({p.get('task', 1) for p in points})
    baseline = sum(float(p.get('iteration_seconds', 0.)) for p in points) / len(points)
    application = sum(float(p['vna_seconds']) for p in points) / len(points)
    candidates = sorted(set([1, 2, 4, 8, 16, 32, 64, 128, 256, 512] +
                            [m*iterations_per_epoch for m in (1, 2, 5, 10)]))

    def count(K, length):
        if length < 1: return 0
        used, gap = 1, K
        while gap < length:
            used += 1
            gap += K * (2 ** (used-1))
        return used

    lengths = {task: max(1, int(next((p.get('task_iterations', p.get('iterations_per_epoch', iterations_per_epoch))
                                      for p in points if p.get('task') == task), iterations_per_epoch)))
               for task in tasks}
    feasible = [K for K in candidates if all(count(K, length)*application <= length*baseline
                                             for length in lengths.values())]
    K0 = min(feasible) if feasible else max(candidates)
    return dict(K0=K0, budget=1.0, seconds_per_application=application,
        budget_exceeded=not bool(feasible),
        t_vna_task1=[p['vna_seconds'] for p in points if p.get('task') == 1],
        t_vna_task2=[p['vna_seconds'] for p in points if p.get('task') == 2],
        projected_vna_seconds={task: count(K0, length)*application for task, length in lengths.items()},
        iteration_seconds=baseline, candidates=candidates,
        iterations_per_epoch=iterations_per_epoch, calibration=deepcopy(points),
        task_iterations=lengths, K_units='iterations')


class V3Engine:
    def __init__(self, arm='vna', *, seed=0, coefficient=1., K0=100, curvature='power10'):
        if curvature != 'power10': raise ValueError('engine curvature must be power10; exact is a historical reference only')
        self.curvature = curvature
        self.arm, self.seed, self.coefficient = arm, seed, coefficient
        self.preflight_only = False
        self.K0 = self.K_next = K0
        self.mu, self.nu = None, 2.
        self.protocol = {'live_candidates': 4, 'gamma_candidates': 2, 'trust_region': True}
        self.task, self.last_iteration = None, None
        self.applications = 0
        self.A = {}
        self.mu_star, self.mu_star_task = None, None
        self.install_ages = {}
        self.events, self.activities, self.calibration = [], [], []
        self.gir_fallback = False
        self.diverged = False
        self.algorithm_seconds = 0.

    @property
    def cfg(self):
        return GGNConfig(damping_rel=0., hutchinson_probes=8, seed=self.seed, project_bn=False)

    def state_dict(self): return deepcopy(vars(self))
    def load_state_dict(self, state):
        if state.get('curvature','power10') != 'power10': raise ValueError('checkpoint curvature mismatch')
        vars(self).update(deepcopy(state))

    @property
    def kernel_self(self): return self.arm.startswith('kernel_self')

    @property
    def gir_only(self): return self.arm == 'gir'

    def configure_schedule(self, schedule):
        if schedule.get('curvature',self.curvature)!=self.curvature:
            raise ValueError('schedule curvature mismatch')
        self.K0 = self.K_next = int(schedule['K0'])
        if schedule.get('iteration_seconds'):
            self.protocol['baseline_iteration_seconds'] = float(schedule['iteration_seconds'])
        self.protocol['budget'] = float(schedule.get('budget', 1.0))

    def begin_task(self, task):
        if task != self.task:
            self.task, self.last_iteration, self.K_next = task, None, self.K0
            self.applications = 0
            self.A.clear()

    @staticmethod
    def _loss_parts(model, batch, spec, ctx):
        logits = model(batch[0]).reshape(len(batch[0]), -1).double()
        total = float(spec.loss(logits, ctx).detach())
        ce = float(torch.nn.functional.cross_entropy(logits, ctx['targets']).detach())
        return total, total-ce if ctx.get('known_classes', 0) else None

    @_global_state()
    def _measure_ratio(self, model, batch, spec, ctx, refs, x):
        params = dict(model.named_parameters())
        before = {n: p.detach().clone() for n, p in params.items()}
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        rng = torch.random.get_rng_state()
        modes = {m: m.training for m in model.modules()}
        try:
            model.train()
            with torch.no_grad():
                for (name, parameter), value in zip(refs, x.split([p.numel() for _, p in refs])):
                    params[name].add_(value.reshape_as(parameter).to(parameter), alpha=-1.)
            after = self._loss_parts(model, batch, spec, ctx)
            with torch.no_grad():
                for name, value in before.items(): params[name].copy_(value)
                for name, value in buffers.items(): model.get_buffer(name).copy_(value)
            before_loss = self._loss_parts(model, batch, spec, ctx)
            return before_loss, after
        finally:
            with torch.no_grad():
                for name, value in before.items(): params[name].copy_(value)
                for name, value in model.named_buffers(): value.copy_(buffers[name])
            for module, mode in modes.items(): module.training = mode
            torch.random.set_rng_state(rng)

    @staticmethod
    def _restore_params(model, state):
        with torch.no_grad():
            for name, value in model.named_parameters(): value.copy_(state[name])

    @_global_state()
    def _evaluate_current(self, model, batch, spec, ctx):
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        rng = torch.random.get_rng_state()
        modes = {m: m.training for m in model.modules()}
        try:
            model.train()
            return self._loss_parts(model, batch, spec, ctx)
        finally:
            with torch.no_grad():
                for name, value in model.named_buffers(): value.copy_(buffers[name])
            for module, mode in modes.items(): module.training = mode
            torch.random.set_rng_state(rng)

    @staticmethod
    def _output_solver(G, blocks, old_blocks, mu, records):
        cached_beta, factor = None, None
        def solve(name, rhs, beta=0.):
            nonlocal cached_beta, factor
            start = time.perf_counter()
            if cached_beta != beta:
                factor = (OutputSolve if isinstance(G, torch.Tensor) else OutputCG)(G, blocks+beta*old_blocks, mu)
                cached_beta = beta
            try:
                y = factor.solve(rhs.double())
            except CGConvergenceError as error:
                records.append(dict(error.record, name=name, beta=beta))
                raise
            image = solve_image(G,y)
            record = dict(name=name, beta=beta, seconds=time.perf_counter()-start,
                          relative_residual=(factor.records[-1]['relative_residual'] if isinstance(factor,OutputCG) else
                              float((factor.metric(image)+mu*y-rhs).norm()/rhs.norm().clamp_min(1e-300))))
            if isinstance(factor, OutputCG): record.update(factor.records[-1])
            if name == 'd_N':
                record['leak'] = max(float(image @ torch.einsum('nij,nj->ni', old_blocks,
                    image.reshape(old_blocks.shape[:2])).flatten()), 0.)
            records.append(record)
            return y
        return solve

    @staticmethod
    def old_blocks(op):
        blocks = torch.zeros_like(op.loss_spec.metric_blocks(op.logits, op.ctx))
        known = op.ctx.get('known_classes', 0)
        if known:
            T = getattr(op.loss_spec, 'T', 2.)
            blocks[:, :known, :known] = CE().metric_blocks(op.logits[:, :known]/T, op.ctx)/T**2
        return blocks

    @staticmethod
    def old_parts(op):
        n, c = op.logits.shape
        known = op.ctx.get('known_classes', 0)
        eold = torch.zeros_like(op.logits)
        if not known:
            return eold.flatten(), lambda v: torch.zeros_like(v), 0.
        temperature = getattr(op.loss_spec, 'T', 2.)
        ps = (op.logits[:, :known]/temperature).softmax(1)
        pt = (op.ctx['teacher_logits'][:, :known].double()/temperature).softmax(1)
        eold[:, :known] = (ps-pt)/(n*temperature)
        def H_old(v):
            out = torch.zeros_like(op.logits)
            v = v.reshape_as(out)[:, :known]
            out[:, :known] = ps*(v-(ps*v).sum(1, keepdim=True))/(n*temperature**2)
            return out.flatten()
        chi = float(((ps-pt).square()/ps).sum()/n)
        return eold.flatten(), H_old, chi

    def _direction(self, p, op, mu, *, G=None, split=None, split_only=False):
        records = []
        if G is None: G = FrozenGram(p)
        blocks, old_blocks = op.loss_spec.metric_blocks(op.logits, op.ctx), self.old_blocks(op)
        eold, H_old, chi = self.old_parts(op)
        solve = self._output_solver(G, blocks, old_blocks, mu, records)
        if split is None:
            y = solve('z', op.g)
            yO = solve('z_O', eold) if bool(eold.any()) else torch.zeros_like(y)
            z, zO = op.H(solve_image(G,y)), op.H(solve_image(G,yO))
        else: y, yO, z, zO = split
        zN = z-zO
        base = dict(y=y, yO=yO, z=z, zO=zO, zN=zN, eold=eold, records=records,
                    G=G, blocks=blocks, old_blocks=old_blocks, H_old=H_old, chi2=chi)
        if split_only: return base
        kernel = (lambda v: G@v) if self.kernel_self else p.KE
        u, uR, uN = kernel(z).double(), kernel(zO).double(), kernel(zN).double()
        V, VO = max(float(z@u), 0.), max(float(zO@uR), 0.)
        # BetaRule uses the same parameter quadratic form in either representation.
        B = lambda v: G.JT(H_old(G.Jv(v))) if isinstance(G,FrozenGram) else G@H_old(G@v)
        if not bool(eold.any()):
            dN = solve('d', op.H(u)); drep = torch.zeros_like(dN)
            beta = dict(beta=0., beta_branch='unprotected', leak=0., epsilon_star=None, beta_suf=None)
        else:
            drep = solve('d_rep', op.H(uR))
            if self.arm.endswith('_beta0'):
                dN = solve('d_N', op.H(uN))
                beta = dict(beta=0., beta_branch='ablation_beta0', leak=max(float(dN@B(dN)), 0.), epsilon_star=None, beta_suf=None)
            else:
                dN, beta = beta_rule(solve, op.H(uN), B, op.H, uN, VO, chi, eold)
        leaks = sorted((r['beta'], r['leak']) for r in records if 'leak' in r)
        monotonic = any(b > a+1e-10*max(1., abs(a)) for (_, a), (_, b) in zip(leaks, leaks[1:]))
        return dict(**base, dy=drep+dN, d_N=dN, d_rep=drep, V=V, V_O=VO, uN=uN,
                    beta_leaks=leaks, leak_monotonicity_violation=monotonic, **beta)

    def _lm(self, model, batch, spec, ctx, p, op, G, mu):
        mus, rhos, records = [], [], []
        nu = 2.
        for _ in range(40):
            mus.append(mu)
            solve = self._output_solver(G, spec.metric_blocks(op.logits, ctx), self.old_blocks(op), mu, records)
            try:
                y = solve('lm', op.g)
            except CGConvergenceError:
                rhos.append(float('-inf'))
                mu *= nu; nu *= 2
                if not math.isfinite(mu): break
                continue
            except torch.linalg.LinAlgError:
                return dict(reason='chol_fail', mu=mu, mu_trajectory=mus, rho_trajectory=rhos, records=records)
            image = solve_image(G,y)
            pred = float(op.g@image)-.5*float(image@op.H(image))
            if not torch.isfinite(y).all() or not math.isfinite(pred):
                return dict(reason='nonfinite', mu=mu, mu_trajectory=mus, rho_trajectory=rhos, records=records)
            # Dense references return output cotangents; CG returns the parameter step directly.
            before, after = self._measure_ratio(model, batch, spec, ctx, p.parameter_refs, y if isinstance(G,FrozenGram) else p.JT(y))
            rho = ((before[0]-after[0])/pred if pred > 0 and
                   all(math.isfinite(v) for v in (before[0], after[0])) else float('-inf'))
            rhos.append(rho)
            if math.isfinite(rho) and rho > 0:
                self.mu, self.nu = mu*(1/3 if rho >= 1 else max(1/3, 1-(2*rho-1)**3)), 2.
                self.mu_star, self.mu_star_task = mu, self.task
                return dict(reason=None, mu=mu, y=y, before=before, after=after,
                            rho=rho, mu_trajectory=mus, rho_trajectory=rhos, records=records)
            mu *= nu; nu *= 2
            if not math.isfinite(mu): break
        self.mu, self.nu = mu, nu
        return dict(reason='lm_budget', mu=mu, mu_trajectory=mus, rho_trajectory=rhos, records=records)

    def _select(self, model, sites, batch, values, *, matrix_values=None):
        proposals = {}
        for site in sites:
            cotangents = [values['z']] if matrix_values is None else matrix_values
            S, Ns = statistics(model, site, batch, cotangents)
            if matrix_values is None: M = Ns[0]@Ns[0].T
            else:
                Nz, NzO, Ny, NzN = Ns
                M = (Nz@NzO.T+NzO@Nz.T+Ny@NzN.T+NzN@Ny.T)/2
            rows = generalized_rows(M, S, 16)
            if not len(rows): continue
            a = rows_proposal(site, rows)
            for i in range(len(rows)):
                trial = {k: None if v is None else v[i:i+1] for k, v in a.items()}
                signs = []
                for sign in (1., -1.):
                    signed = {k: None if v is None else sign*v for k, v in trial.items()}
                    with training_products(model, [site], {site.module_name: signed}, batch[0]) as p:
                        signs.append(float(p.JET(values['z']).double().square().sum()))
                if signs[1] > signs[0]:
                    for v in a.values():
                        if v is not None: v[i].neg_()
            proposals[site.module_name] = a
        return proposals

    @staticmethod
    def _trace_extension(products):
        nout = products._call({}).numel()
        ne = sum(v.numel() for _, v in products.extension_refs)
        total = 0.
        for start in range(0, min(nout, ne), 8):
            indices = torch.arange(start, min(start+8, min(nout, ne)), device=products.inputs.device)
            basis = nn.functional.one_hot(indices, nout if nout < ne else ne).to(products.inputs)
            action = products.JET if nout < ne else products.JEv
            total += float(torch.vmap(action)(basis).double().square().sum())
        return total

    def _randomize(self, model, sites, batch, reference):
        result = {}
        generator = dedicated_generator(self.seed, f'v3-random:{self.task}:{self.applications}', batch[0].device)
        for site in sites:
            name = site.module_name
            if name not in reference: continue
            proposal = {k: None if v is None else torch.randn(v.shape, dtype=v.dtype, device=v.device,
                generator=generator) for k, v in reference[name].items()}
            with training_products(model, [site], {name: reference[name]}, batch[0]) as p: target = self._trace_extension(p)
            with training_products(model, [site], {name: proposal}, batch[0]) as p: actual = self._trace_extension(p)
            proposal['feature_scale'] = math.sqrt(target/actual) if actual > 0 else 0.
            proposal['trace_target'], proposal['trace_before'] = target, actual
            result[name] = proposal
        return result

    def preflight(self, model, sites, batch, spec, ctx, optimizer, iteration=1, remaining=None):
        """Measure one read-only point; the adapter calls this three times per task."""
        # Time a complete application (VNA step, exact check and GIR swaps), then restore
        # model, optimizer, RNG and engine state bitwise: GIR is the most expensive part of K0.
        saved = self.state_dict()
        training = self._training_state(model, optimizer)
        started = time.perf_counter()
        try:
            row = self.apply(model, sites, batch, spec, ctx, optimizer, iteration,
                             remaining=remaining, force=True, apply=True)
        finally:
            self._restore_training_state(model, optimizer, training)
            self.load_state_dict(saved)
        point = {**(row or {}), 'task': self.task,
                 'vna_seconds': time.perf_counter()-started,
                 'iteration_seconds': float(self.protocol.get('baseline_iteration_seconds', 0.)),
                 'task_iterations': int(self.protocol.get('task_iterations', remaining or 1))}
        points = self.protocol.setdefault('preflight_points', [])
        points.append(point)
        if len(points) >= 6:
            schedule = schedule_from_points(points, max(1, int(self.protocol.get('iterations_per_epoch', remaining or 1))))
            self.K0 = self.K_next = schedule['K0']
            self.protocol.update(K0=schedule['K0'], seconds_per_application=schedule['seconds_per_application'],
                                budget=schedule['budget'])
        return dict(K0=self.protocol.get('K0', self.K0),
                    seconds_per_application=self.protocol.get('seconds_per_application'),
                    budget=self.protocol.get('budget', 1.0), trials=[point])

    @staticmethod
    def _norms(model, sites):
        params = dict(model.named_parameters())
        return dict(param_norm=math.sqrt(sum(float(p.detach().double().square().sum()) for p in params.values())),
                    param_max_abs=max((float(p.detach().abs().max()) for p in params.values()), default=0.),
                    site_norms={s.module_name: math.sqrt(sum(float(params[n].detach().double().square().sum())
                                for n in s.parameter_names)) for s in sites})

    @staticmethod
    def _finite(model, optimizer):
        def finite(value):
            if isinstance(value, torch.Tensor): return bool(torch.isfinite(value).all())
            if isinstance(value, dict): return all(finite(v) for v in value.values())
            if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
            return True
        return finite(model.state_dict()) and finite(optimizer.state_dict())

    def apply(self, model, sites, batch, spec, ctx, optimizer, iteration, remaining=None, force=False, apply=True):
        if self.coefficient == 0 or self.arm in ('baseline', 'bp', 'cbp', 'icarl') or ('cbp' in self.arm and 'vna' not in self.arm): return None
        if not force and self.last_iteration is not None and iteration-self.last_iteration < self.K_next: return None
        saved = self.state_dict()
        training = self._training_state(model, optimizer)
        before = self._norms(model, sites)
        start = time.perf_counter()
        try:
            with _global_state():
                try:
                    row = self._apply(model, sites, batch, spec, ctx, optimizer, iteration, remaining, force, apply)
                except CGConvergenceError as error:
                    self._restore_training_state(model, optimizer, training)
                    self.load_state_dict(saved)
                    row = dict(skipped=True, skip_reason='cg_not_converged', solves=[error.record], swaps=0, swap_records=[])
                except torch.linalg.LinAlgError:
                    self._restore_training_state(model, optimizer, training)
                    self.load_state_dict(saved)
                    row = dict(skipped=True, skip_reason='chol_fail', chol_fail=True, swaps=0, swap_records=[])
                after_norms = self._norms(model, sites)
                row.update(before=before, after=after_norms, **after_norms)
                if not self._finite(model, optimizer):
                    self._restore_training_state(model, optimizer, training)
                    self.load_state_dict(saved)
                    row.update(nonfinite_after_apply=True, skipped=True, skip_reason='nonfinite_after_apply',
                               swaps=0, swap_records=[], after_restore=self._norms(model, sites))
            row.update(kind='correction', task=self.task, iteration=iteration, step=iteration,
                       K=self.K_next, K_next=self.K_next*2, seconds=time.perf_counter()-start)
            row['skip_rate'] = (sum(bool(r.get('skipped')) for r in self.events)+int(row['skipped']))/(len(self.events)+1)
            if apply:
                self.last_iteration, self.K_next = iteration, self.budget_guard(row, iteration)
                self.applications += 1; self.events.append(row)
            return row
        finally:
            if not apply:
                self._restore_training_state(model, optimizer, training)
                self.load_state_dict(saved)

    @staticmethod
    def _training_state(model, optimizer):
        return (deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()),
                {m: m.training for m in model.modules()}, rng_state(),
                {n: None if p.grad is None else p.grad.detach().clone() for n, p in model.named_parameters()})

    @staticmethod
    def _restore_training_state(model, optimizer, state):
        model_state, optimizer_state, modes, rng, grads = state
        with torch.no_grad(): model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        for module, mode in modes.items(): module.training = mode
        for n, p in model.named_parameters(): p.grad = grads[n]
        restore_rng(rng)

    @staticmethod
    def _forward_logits(model, x):
        output = model(x)
        if isinstance(output, dict): output = output['logits']
        return output.reshape(len(x), -1).double()

    @_global_state()
    def _slot_contribution(self, model, site, batch, spec, ctx, unit):
        state = {n: p.detach().clone() for n, p in model.named_parameters()}
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        modes = {m: m.training for m in model.modules()}
        rng = torch.random.get_rng_state().clone()
        try:
            model.train()
            with torch.no_grad():
                full = self._forward_logits(model, batch[0])
                site.second_layer.layer.weight[:, unit] = 0
                without = self._forward_logits(model, batch[0])
            return (full-without).flatten()
        finally:
            with torch.no_grad():
                for name, value in state.items(): dict(model.named_parameters())[name].copy_(value)
                for name, value in buffers.items(): dict(model.named_buffers())[name].copy_(value)
            for module, mode in modes.items(): module.training = mode
            torch.random.set_rng_state(rng)

    @_global_state()
    def _slot_jump(self, model, site, batch, spec, ctx, unit):
        state = {n: p.detach().clone() for n, p in model.named_parameters()}
        buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
        modes = {m: m.training for m in model.modules()}
        rng = torch.random.get_rng_state().clone()
        try:
            before = self._evaluate_current(model, batch, spec, ctx)
            with torch.no_grad(): site.second_layer.layer.weight[:, unit] = 0
            after = self._evaluate_current(model, batch, spec, ctx)
            return after[0]-before[0], (None if before[1] is None else after[1]-before[1])
        finally:
            with torch.no_grad():
                for name, value in state.items(): dict(model.named_parameters())[name].copy_(value)
                for name, value in buffers.items(): dict(model.named_buffers())[name].copy_(value)
            for module, mode in modes.items(): module.training = mode
            torch.random.set_rng_state(rng)

    @staticmethod
    def _slot_columns(p, site, unit, replacements=None, dead=False, materialize=True):
        selected, outgoing, offset = [], None, 0
        paths = {site.first+'.weight': unit, site.second+'.weight': (slice(None), unit)}
        if site.first_layer.layer.bias is not None: paths[site.first+'.bias'] = unit
        for path in site.between:
            module = site.model.get_submodule(path)
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.affine:
                paths.update({path+'.weight': unit, path+'.bias': unit})
        size = sum(v.numel() for _, v in p.parameter_refs)
        ref = p.parameter_refs[0][1]
        for name, value in p.parameter_refs:
            native = dict(site.model.named_parameters())[name]
            match = next((index for path, index in paths.items()
                          if site.model.get_parameter(path) is native), None)
            if match is not None:
                ids = torch.arange(value.numel(), device=value.device).reshape_as(value)[match].flatten()
                selected.append(ids+offset)
                if native is site.second_layer.layer.weight: outgoing = ids+offset
            offset += value.numel()
        ids = torch.cat(selected)
        if not materialize: return None, ids, outgoing
        J = (torch.zeros((p._call({}).numel(), len(ids)), dtype=torch.float64, device=ref.device)
             if dead else columns(p.Jv, ids, size, ref))
        if replacements is not None:
            for column, index in enumerate(ids.tolist()):
                if index in replacements:
                    value = replacements[index]
                    J[:, column] = 0. if value is None else value
        if dead: J = J[:, J.ne(0).any(0)]
        return J, ids, outgoing

    def _run_gir(self, model, sites, frozen, frozen_sites, batch, spec, ctx, optimizer,
                 units_by_site, iteration, values, mu):
        if not ('gir' in self.arm and 'cbp' not in self.arm): return []
        factor = values['factor']
        ecur, shift = values['e_beta'].clone(), torch.zeros_like(values['e_beta'])
        init = next((t[len('girinit-'):] for t in self.arm.split('_') if t.startswith('girinit-')), None)
        records = []
        values['gir_trials'], values['gir_solves'] = [], []
        fixed_dead = self.dead_slots(frozen, frozen_sites, batch)
        scopes = [set(s.parameter_names) for s in frozen_sites]
        replacements = {} if any(a & b for i, a in enumerate(scopes) for b in scopes[i+1:]) else None
        for site, fixed in zip(sites, frozen_sites):
            units = units_by_site.get(site.module_name, [])
            if not units: continue
            used, cache = [], {}
            def proposals():
                if ecur.is_cuda: torch.cuda.synchronize(ecur.device)
                residual_start = time.perf_counter()
                residual = ecur-factor.metric(solve_image(factor.G,factor.solve(ecur)))
                if ecur.is_cuda: torch.cuda.synchronize(ecur.device)
                values['gir_solves'].append(dict(name='gir_residual', seconds=time.perf_counter()-residual_start))
                if isinstance(factor, OutputCG): values['gir_solves'][-1].update(factor.records[-1])
                if init in ('gradmax', 'tiny'): residual = values['e']
                cache.pop('stats', None)
                return self.prepare_gir(frozen, fixed, batch, spec, ctx, optimizer, units, iteration,
                    residual=residual, used_directions=used, gradmax=init == 'gradmax',
                    random_rows=init == 'random', H=factor.metric, cache=cache, mu=mu)
            start = time.perf_counter()
            choices = proposals()
            prepare_seconds = time.perf_counter()-start
            if not choices: continue
            start = time.perf_counter()
            contributions = {u: self._slot_contribution(frozen, fixed, batch, spec, ctx, u) for u in units}
            contribution_seconds = time.perf_counter()-start
            norms = {u: math.sqrt(max(float(f@factor.metric(f)), 0.)) for u, f in contributions.items()}
            dead = [u for u in units if u in fixed_dead.get(fixed.module_name, [])]
            live = sorted((u for u in units if u not in dead), key=lambda u: norms[u])[:int(self.protocol['live_candidates'])]
            for unit in dead+live:
                if not choices: break
                start = time.perf_counter()
                jL, jKD = self._slot_jump(model, site, batch, spec, ctx, unit)
                jump_seconds = time.perf_counter()-start
                if not math.isfinite(jL) or jL > 0 or (jKD is not None and (not math.isfinite(jKD) or jKD > 0)): continue
                columns_start = time.perf_counter()
                with training_products(frozen, [], {}, batch[0]) as p:
                    Ju, native_ids, outgoing_ids = self._slot_columns(p, fixed, unit, replacements, unit in dead,
                        materialize=not isinstance(factor, OutputCG))
                    native_jvps = 0 if unit in dead or isinstance(factor, OutputCG) else len(native_ids)
                if ecur.is_cuda: torch.cuda.synchronize(ecur.device)
                native_columns_seconds = time.perf_counter()-columns_start
                gain_start = time.perf_counter()
                factor.q(ecur)
                values['gir_solves'].append(dict(name='baseline_gain', seconds=time.perf_counter()-gain_start))
                best = None
                for trial_index, proposal in enumerate(choices[:int(self.protocol['gamma_candidates'])]):
                    start = time.perf_counter()
                    Jv = proposal.get('_Jv')
                    cached = Jv is not None
                    if not cached:
                        with training_products(frozen, [fixed], {fixed.module_name: proposal['A']}, batch[0]) as pv:
                            ne = sum(v.numel() for _, v in pv.extension_refs)
                            ref = pv.extension_refs[0][1]
                            Jv = columns(pv.JEv, torch.arange(ne, device=ref.device), ne, ref)
                            proxy, incoming, _, _ = pv.virtual[0]
                            features = proxy.virtual_features(cache['x'], incoming)
                            proposal['_activation'] = float(features.ne(0).double().mean())
                        proposal['_Jv'] = Jv
                    ne, activation = Jv.shape[1], proposal['_activation']
                    swap = (factor.swapped(Ju,Jv,native_ids=native_ids,outgoing_ids=outgoing_ids)
                            if isinstance(factor,OutputCG) else factor.swapped(Ju,Jv))
                    gamma = swap.gamma(ecur, contributions[unit])
                    if isinstance(factor, OutputCG):
                        values['gir_solves'].extend(dict(r, name='Gamma') for r in swap.factor.records)
                    seconds = time.perf_counter()-start
                    values['gir_trials'].append(dict(site=site.module_name, unit=int(unit),
                        Gamma_woodbury=gamma, gamma_seconds=seconds, n_jvp=native_jvps+ne,
                        native_columns_seconds=native_columns_seconds if trial_index==0 else 0.,
                        gamma_total_seconds=seconds+(native_columns_seconds if trial_index==0 else 0.),
                        n_jvp_evaluated=(native_jvps if trial_index==0 else 0)+(0 if cached else ne),
                        candidate_jacobian_cached=cached, candidate_jvp_evaluated=0 if cached else ne))
                    if isinstance(factor, OutputCG):
                        values['gir_trials'][-1]['Gamma_cg'] = values['gir_trials'][-1].pop('Gamma_woodbury')
                    if math.isfinite(gamma) and gamma > 0 and (best is None or gamma > best[0]):
                        best = gamma, proposal, swap, seconds, activation, ne
                if best is None: continue
                gamma, proposal, swap, seconds, activation, ne = best
                self.install_gir(model, site, [proposal], [unit], optimizer)
                if not self._finite(model, optimizer): return records
                swap.commit()
                if replacements is not None and not isinstance(factor,OutputCG):
                    # Adjacent MLP sites share weights: later swaps remove the current columns once.
                    replacements.update((i, None) for i in native_ids.tolist())
                    replacements.update((i, column) for i, column in zip(outgoing_ids.tolist(), swap.add.T))
                fu = contributions[unit]
                shift -= fu; ecur -= factor.metric(fu)
                used.append(proposal['direction'])
                incoming = proposal['A']['weight'].double().flatten()
                native = fixed.first_layer.layer.weight.detach().flatten(1).double()
                if proposal['A'].get('bias') is not None:
                    incoming = torch.cat((incoming, proposal['A']['bias'].double().flatten()))
                    native = torch.cat((native, fixed.first_layer.layer.bias.detach().double()[:, None]), 1)
                incoming_norm = float(incoming.norm())
                median = float(native.norm(dim=1).median())
                record = dict(site=site.module_name, unit=int(unit), slot_kind='dead' if unit in dead else 'live',
                    task=self.task, iteration=iteration, Gamma=gamma, Gamma_woodbury=gamma,
                    ell_beta=proposal['score'], f_u_norm_Hb=norms[unit], j_L=jL, j_KD=jKD,
                    bias_q=proposal.get('bias_q'), incoming_norm=incoming_norm, incoming_norm_median=median,
                    incoming_norm_ratio=incoming_norm/median if median else None, activation_fraction=activation,
                    n_jvp=native_jvps+ne, gamma_seconds=seconds, prepare_gir_seconds=prepare_seconds,
                    slot_jump_seconds=jump_seconds, slot_contribution_seconds=contribution_seconds,
                    output_shift_norm=float(shift.norm()), contribution_norm={1: None, 5: None, 10: None})
                records.append(record)
                if isinstance(factor, OutputCG): record['Gamma_cg'] = record.pop('Gamma_woodbury')
                self.install_ages[(site.module_name, int(unit))] = dict(site=site.module_name,
                    unit=int(unit), task=self.task, iteration=iteration)
                start = time.perf_counter(); choices = proposals()
                prepare_seconds = time.perf_counter()-start
        return records

    def budget_guard(self, row, iteration):
        """Compute-budget guard (§2.5): if algorithm time exceeds budget x training time so far,
        double the next gap. Changes only the schedule, never the algorithm."""
        self.algorithm_seconds += float(row.get('seconds') or 0.)
        per_iteration = float(self.protocol.get('baseline_iteration_seconds') or 0.)
        ratio = self.algorithm_seconds/(max(int(iteration), 1)*per_iteration) if per_iteration > 0 else None
        guard = ratio is not None and ratio > float(self.protocol.get('budget', 1.0))
        row.update(budget_ratio=ratio, budget_guard=guard)
        if guard: row['K_next'] = 2*row['K_next']
        return row['K_next']

    def _apply(self, model, sites, batch, spec, ctx, optimizer, iteration, remaining=None, force=False, apply=True):
        sites = [site for site in sites if self.valid_slot(site)]
        frozen = deepcopy(model)
        frozen_sites = [HookSite(frozen, s.first, s.second, s.kind, between=s.between,
                                o_params=s.parameter_names) for s in sites]
        initial = seed_proposals(frozen_sites)
        with training_products(frozen, frozen_sites, initial, batch[0]) as p:
            op = GGNProducts(p, batch[1], spec, ctx)
            if not bool(torch.isfinite(op.g).all()):
                return dict(skipped=True, skip_reason='nonfinite', swaps=0)
            G = FrozenGram(p)
            timing = dict(solver='cg', m=op.g.numel(), gram_bytes=0)
            gradient_norm = float(op.gradient.norm())
            if gradient_norm == 0: return dict(**timing, skipped=True, skip_reason='zero_gradient', swaps=0)
            if self.mu is None:
                self.mu = max(1e-12, 1e-2*op.trace(self.cfg)/max(op.size, 1))
                self.protocol['mu0'] = self.mu
            lm = self._lm(model, batch, spec, ctx, p, op, G, self.mu)
            row = dict(**timing, skipped=lm['reason'] is not None, skip_reason=lm['reason'],
                       mu_star=lm['mu'] if lm['reason'] is None else None, mu=lm['mu'], nu=self.nu,
                       lm_tries=len(lm['mu_trajectory']), mu_trajectory=lm['mu_trajectory'],
                       rho_trajectory=lm['rho_trajectory'], rho_tr=lm.get('rho'), swaps=0, swap_records=[],
                       gradient_norm=gradient_norm, solves=lm['records'], chol_fail=lm['reason']=='chol_fail')
            if lm['reason'] is not None: return row
            mu = lm['mu']
            preliminary = self._direction(p, op, mu, G=G, split_only=True)
        start = time.perf_counter()
        A = initial if self.kernel_self or self.gir_only else self._select(frozen, frozen_sites, batch, preliminary)
        if 'rand' in self.arm.split('_'): A = self._randomize(frozen, frozen_sites, batch, A)
        row['tiny_V_seconds'] = time.perf_counter()-start
        active = [s for s in frozen_sites if s.module_name in A]
        if not active:
            row.update(skipped=True, skip_reason='no_positive_directions'); return row
        start = time.perf_counter()
        with training_products(frozen, active, A, batch[0]) as p:
            if self.gir_only:
                values = dict(**preliminary, dy=torch.zeros_like(preliminary['y']), beta=0.,
                    beta_branch='gir_only', leak=0., epsilon_star=None, V=0., V_O=0.,
                    beta_leaks=[], leak_monotonicity_violation=False)
            else:
                values = self._direction(p, op, mu, G=G,
                    split=(preliminary['y'], preliminary['yO'], preliminary['z'], preliminary['zO']))
            dy, image = values['dy'], solve_image(G,values['dy'])
            Delta = math.sqrt(max(solve_norm2(G,lm['y']),0.))
            gh, hh = float(op.g@image), float(values['eold']@image)
            reason = ('nonfinite' if not bool(torch.isfinite(dy).all()) or not math.isfinite(Delta) else
                      'invalid_beta' if values.get('invalid_beta') else
                      'negative_certificate' if gh <= 0 or (bool(values['eold'].any()) and hh < 0) else None)
            if self.gir_only: reason = None
            if reason:
                row.update(skipped=True, skip_reason=reason, g_dot_d=gh, h_dot_d=hh, s=0.); return row
            normd = math.sqrt(max(solve_norm2(G,dy),0.))
            curvature = float(image@op.H(image))
            optimum = gh/curvature if curvature > 0 else float('inf')
            scale = min(optimum, Delta/normd) if normd and gh > 0 and not self.gir_only else 0.
            e_beta, beta = op.g, 0. if self.gir_only else values['beta']
            if beta:
                delta_old = torch.zeros_like(op.logits)
                known = int(ctx.get('known_classes', 0))
                delta_old[:, :known] = -solve_image(G,values['yO']).reshape_as(op.logits)[:, :known]
                e_beta = op.g-beta*values['H_old'](delta_old.flatten())
            Hb_blocks = values['blocks']+beta*values['old_blocks']
            factor = OutputCG(G, Hb_blocks, mu)
            values.update(e=op.g, e_beta=e_beta, H_beta=factor.metric, factor=factor)
            ne = sum(v.numel() for _, v in p.extension_refs)
            lam = power_lambda_max(lambda v: p.JET(op.H(p.JEv(v))), ne,
                dedicated_generator(self.seed, f'v4-lambdaE:{self.task}:{self.applications}', batch[0].device),
                device=batch[0].device)
            B = max(float(op.g@solve_image(G,lm['y'])),0.)
            e = op.g.reshape_as(op.logits)
            normw = float(torch.einsum('ni,nij,nj->', e, torch.linalg.pinv(values['blocks'], hermitian=True), e))
            row.update(V=values['V'], B=B, lambda_E=lam, norm_w2=normw,
                ratio=values['V']/(lam*B) if lam*B else None, V_O=values['V_O'],
                g_dot_d=gh, h_dot_d=hh, Delta=Delta, beta=beta, beta_branch=values['beta_branch'],
                leak=values['leak'], epsilon_star=values['epsilon_star'], chi2=values['chi2'],
                beta_leaks=values['beta_leaks'], leak_monotonicity_violation=values['leak_monotonicity_violation'],
                s=scale, s_F_star=optimum, gradient_norm=gradient_norm,
                solves=lm['records']+preliminary['records']+([] if self.gir_only else values['records']))
            delta = (dy if isinstance(G,FrozenGram) else p.JT(dy)) if scale > 0 else None
        row['certificate_seconds'] = time.perf_counter()-start
        before = {n: v.detach().clone() for n, v in model.named_parameters()}
        rejected, after = False, lm['before']
        if apply and scale > 0:
            refs = dict(model.named_parameters())
            with torch.no_grad():
                for (name, ref), value in zip(p.parameter_refs, delta.split([v.numel() for _, v in p.parameter_refs])):
                    refs[name].add_(value.reshape_as(ref).to(ref), alpha=-scale)
            after = self._evaluate_current(model, batch, spec, ctx)
            rejected = (not math.isfinite(after[0]) or after[0] > lm['before'][0] or
                        (after[1] is not None and (not math.isfinite(after[1]) or after[1] > lm['before'][1])))
            if rejected: self._restore_params(model, before)
        row.update(rejected=rejected, s=0. if rejected else scale, L_before=lm['before'][0], L_after=after[0],
                   KD_before=lm['before'][1], KD_after=after[1])
        start = time.perf_counter()
        units = {s.module_name: list(range(s.second_layer.layer.weight.shape[1])) for s in sites}
        swaps = self._run_gir(model, sites, frozen, frozen_sites, batch, spec, ctx, optimizer,
                             units, iteration, values, mu) if apply else []
        row.update(swaps=len(swaps), swap_records=swaps, ell=[r['ell_beta'] for r in swaps],
                   gir_seconds=time.perf_counter()-start, gir_trials=values.get('gir_trials', []),
                   gir_solves=values.get('gir_solves', []), systems_solved=len(row['solves'])+factor.solves)
        if apply and not self.kernel_self: self.A = A
        return row

    @staticmethod
    def valid_slot(site):
        if site.kind == 'linear': return True
        return site.kind == 'conv' and site.first.rsplit('.', 1)[-1] in ('conv_a', 'conv1')

    @staticmethod
    def dead_slots(model, sites, batch):
        captures, handles = {}, []
        def save(name):
            def hook(module, args): captures[name] = args[0].detach()
            return hook
        for site in sites: handles.append(site.second_layer.layer.register_forward_pre_hook(save(site.module_name)))
        try:
            buffers = {n: b.clone() for n, b in model.named_buffers()}
            modes = {m: m.training for m in model.modules()}
            with _global_state(), _preserve_functional_state(model), torch.no_grad():
                model.train()
                functional_call(model, ({n: p for n, p in model.named_parameters()}, buffers), (batch[0],))
        finally:
            for h in handles: h.remove()
            for m, mode in modes.items(): m.training = mode
        return {name: torch.where(value.movedim(1, 0).flatten(1).eq(0).all(1))[0].tolist()
                for name, value in captures.items()}

    def prepare_gir(self, model, site, batch, spec, ctx, optimizer, units, iteration, gradmax=False, residual=None, used_directions=(), random_rows=False, H=None, cache=None, mu=None):
        self.gir_fallback = False
        if not len(units) or not self.valid_slot(site): return []
        mu = mu if mu is not None else self.mu
        if residual is None or H is None or mu is None:
            with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
                op = GGNProducts(p, batch[1], spec, ctx)
                if mu is None: mu = max(1e-12, 1e-2*op.trace(self.cfg)/max(op.size, 1))
                if residual is None:
                    if gradmax: residual = op.g
                    else:
                        values = self._direction(p, op, mu)
                        G, beta = values['G'], values['beta']
                        e_beta = op.g
                        known = int(op.ctx.get('known_classes', 0))
                        if known and beta:
                            delta_old = torch.zeros_like(op.logits)
                            delta_old[:, :known] = -solve_image(G,values['yO']).reshape_as(op.logits)[:, :known]
                            e_beta = op.g-beta*values['H_old'](delta_old.flatten())
                        factor = OutputCG(G, values['blocks']+beta*values['old_blocks'], mu)
                        residual = e_beta-factor.metric(solve_image(G,factor.solve(e_beta)))
                        if H is None: H = factor.metric
                if H is None: H = op.H
        metric = H
        # S_x, N and the site input do not change across units of one site (B_v = 0 swaps).
        cache = {} if cache is None else cache
        if 'stats' not in cache: cache['stats'] = statistics(model, site, batch, [residual])
        S, Ns = cache['stats']
        matrix = Ns[0] if gradmax else Ns[0]@Ns[0].T
        if gradmax and (not torch.isfinite(matrix).all() or not torch.isfinite(S).all()):
            self.gir_fallback = True
            return []
        try:
            if random_rows:
                generator = dedicated_generator(self.seed, f'v3-gir-random:{self.task}:{iteration}', S.device)
                rows = torch.randn((4, len(S)), dtype=S.dtype, device=S.device, generator=generator)
            else:
                rows = generalized_rows(matrix, S, 4, euclidean=gradmax)
        except torch.linalg.LinAlgError:
            if not gradmax: raise
            self.gir_fallback = True
            return []
        proposals = []
        bn = next((model.get_submodule(n) for n in site.between
                   if isinstance(model.get_submodule(n), nn.modules.batchnorm._BatchNorm)), None)
        if 'x' not in cache: cache['x'] = self._site_input(model, site, batch)
        x = cache['x']
        first = site.first_layer.layer
        for row in rows:
            row = row.clone()
            energy = float(row @ S @ row)
            for used in used_directions:
                denominator = used @ S @ used
                if denominator > 0:
                    row -= used * ((used @ S @ row)/denominator)
            # A direction already assigned to a slot leaves only round-off after S_x-orthogonalization.
            if float(row @ S @ row) <= max(1e-20, 1e-8*energy): continue
            base = rows_proposal(site, row[None])
            for sign in (1., -1.):
                trial = {k: None if v is None else sign*v for k, v in base.items()}
                projected = nn.functional.linear(x, trial['weight'], trial['bias']) if site.kind == 'linear' else nn.functional.conv2d(x, trial['weight'], trial['bias'], first.stride, first.padding, first.dilation)
                if bn is not None:
                    axes = (0, *range(2, projected.ndim))
                    projected = (projected-projected.mean(axes, keepdim=True))/(projected.var(axes, unbiased=False, keepdim=True)+bn.eps).sqrt()
                quantiles = (0., .1, .3, .5, .7, .9)
                shifts = [(0., None)] + [(-float(torch.quantile(projected.reshape(-1), q)), q) for q in quantiles[1:]]
                for shift, bias_q in shifts:
                    candidate = deepcopy(trial)
                    if bn is not None:
                        candidate['bn_weight'] = torch.ones(1, device=x.device)
                        candidate['bn_bias'] = torch.tensor([shift], device=x.device)
                    elif first.bias is not None: candidate['bias'] = trial['bias']+shift
                    elif shift != 0: continue
                    if not bool((projected+shift).ne(0).any()): continue
                    proposals.append(dict(A=candidate, bias_q=bias_q, direction=row.detach().clone()))
        if not proposals: return []
        combined = {key: (None if proposals[0]['A'][key] is None else
                         torch.cat([entry['A'][key] for entry in proposals])) for key in proposals[0]['A']}
        scored = []
        with training_products(model, [site], {site.module_name: combined}, batch[0]) as pv:
            proxy, incoming, outgoing, _ = pv.virtual[0]
            ref = pv.extension_refs[0][1]
            size = ref.numel()
            ids = torch.arange(size, device=ref.device).reshape_as(outgoing.weight)
            gradients = pv.JET(residual).double()
            with torch.no_grad(): features = proxy.virtual_features(x, incoming)
            width = outgoing.weight.shape[1]//len(proposals)
            candidate_ids = torch.stack([ids[:, i*width:(i+1)*width].flatten() for i in range(len(proposals))])
            candidate_gradients = gradients[candidate_ids]
            active = torch.where(torch.isfinite(candidate_gradients).all(1) & candidate_gradients.ne(0).any(1))[0].tolist()
            def power_curvature(i, seed_index):
                indices = candidate_ids[i]
                def action(v):
                    tangent = torch.zeros(size, device=ref.device, dtype=ref.dtype).scatter(0, indices, v.to(ref))
                    return pv.JET(metric(pv.JEv(tangent)))[indices]
                return power_lambda_max(action, len(indices),
                    dedicated_generator(self.seed, f'v4-ell:{self.task}:{iteration}:{seed_index}', x.device),
                    iterations=10, device=x.device)
            for i in active:
                curvature = power_curvature(i, len(scored))
                if not math.isfinite(curvature): continue
                proposal, s_v = proposals[i], candidate_gradients[i]
                proposal.update(score=.5*float(s_v.square().sum())/(curvature+mu),
                    curvature=curvature, curvature_method='power10',
                    power_seed_index=len(scored),
                    _activation=float(features[:, i*width:(i+1)*width].ne(0).double().mean()))
                scored.append(proposal)
            scored.sort(key=lambda entry: entry['score'], reverse=True)
            for proposal in scored[:int(self.protocol['gamma_candidates'])]:
                i = next(i for i in active if proposals[i] is proposal)
                proposal['_Jv'] = columns(pv.JEv, candidate_ids[i], size, ref)
            return scored

    @staticmethod
    def _site_input(model, site, batch):
        capture = {}
        def hook(module, args): capture['x'] = args[0].detach()
        handle = site.first_layer.layer.register_forward_pre_hook(hook)
        modes = {m: m.training for m in model.modules()}
        try:
            with _global_state(), _preserve_functional_state(model), torch.no_grad():
                model.train()
                functional_call(model, ({n: p for n, p in model.named_parameters()},
                    {n: b.clone() for n, b in model.named_buffers()}), (batch[0],))
        finally:
            handle.remove()
            for m, mode in modes.items(): m.training = mode
        return capture['x']

    def install_gir(self, model, site, proposals, units, optimizer):
        if not V3Engine.valid_slot(site): return 0
        first, second = site.first_layer.layer, site.second_layer.layer
        count = 0
        with torch.no_grad():
            for unit, proposal in zip(units, proposals):
                a = proposal['A']; indices = [(first.weight, unit), (second.weight, (slice(None), unit))]
                first.weight[unit] = a['weight'][0]; second.weight[:, unit] = 0
                if first.bias is not None:
                    first.bias[unit] = a['bias'][0]; indices.append((first.bias, unit))
                for path in site.between:
                    bn = model.get_submodule(path)
                    if isinstance(bn, nn.modules.batchnorm._BatchNorm):
                        if bn.affine:
                            bn.weight[unit] = a.get('bn_weight', torch.ones(1))[0]
                            bn.bias[unit] = a.get('bn_bias', torch.zeros(1))[0]
                            indices.extend(((bn.weight, unit), (bn.bias, unit)))
                        bn.running_mean[unit], bn.running_var[unit] = 0., 1.
                for parameter, index in indices:
                    for value in optimizer.state.get(parameter, {}).values():
                        if isinstance(value, torch.Tensor) and value.shape == parameter.shape: value[index] = 0
                self.install_ages[(site.module_name, int(unit))] = {
                    'site': site.module_name, 'unit': int(unit), 'task': self.task}
                count += 1
        return count

    def gir_candidates(self, model, site, batch, spec, ctx, optimizer, units, iteration, gradmax=False):
        proposals = self.prepare_gir(model, site, batch, spec, ctx, optimizer, units, iteration, gradmax)
        return self.install_gir(model, site, proposals, units, optimizer)
