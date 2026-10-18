"""Training VNA–GIR v3: joint parameter CG, protected correction and dead slots."""
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
from methods.vna_sites import HookProducts
from projection.cg import conjugate_gradient
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


def diagnose(model, batch, *, sites=None, spec=None, ctx=None):
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
                mu = max(1e-12, 1e-2*op.trace(GGNConfig())/max(op.size, 1))
                solve = conjugate_gradient(lambda v: op.F(v)+mu*v, op.gradient,
                    max_iter=200, tolerance=1e-3, eps=1e-30)
                B = max(float(op.gradient @ solve.solution), 0.)
            blocks = spec.metric_blocks(logits, ctx)
            target = e.reshape_as(logits)
            norm_w = max(float(torch.einsum('ni,nij,nj->', target,
                torch.linalg.pinv(blocks, hermitian=True), target)), 0.)
            reach = min(1., max(0., B / norm_w)) if norm_w else 0.
            result = dict(B=B, norm_w2=norm_w, reach=reach, unreach=1.-reach,
                          cg=dict(relative_residual=solve.relative_residual,
                                   iterations=solve.iterations, converged=solve.converged), sites={})
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
    def __init__(self, arm='vna', *, seed=0, coefficient=1., K0=100):
        self.arm, self.seed, self.coefficient = arm, seed, coefficient
        self.preflight_only = False
        self.K0 = self.K_next = K0
        self.tol = 1e-3
        self.mu, self.nu = None, 2.
        self.protocol = {'live_candidates': 4, 'trust_region': True}
        self.task, self.last_iteration = None, None
        self.applications = 0
        self.warm, self.A = {}, {}
        self.install_ages = {}
        self.events, self.activities, self.calibration = [], [], []
        self.gir_fallback = False
        self.diverged = False

    @property
    def cfg(self):
        return GGNConfig(damping_rel=0., cg_iters=200, cg_tol=self.tol,
                         hutchinson_probes=8, seed=self.seed, project_bn=False)

    def state_dict(self): return deepcopy(vars(self))
    def load_state_dict(self, state): vars(self).update(deepcopy(state))

    @property
    def kernel_self(self): return self.arm.startswith('kernel_self')

    @property
    def gir_only(self): return self.arm == 'gir'

    def configure_schedule(self, schedule):
        self.K0 = self.K_next = int(schedule['K0'])
        if 'cg_tol' in schedule: self.tol = schedule['cg_tol']
        if 'tol' in schedule: self.tol = schedule['tol']

    def begin_task(self, task):
        if task != self.task:
            self.task, self.last_iteration, self.K_next = task, None, self.K0
            self.applications = 0
            self.warm.clear(); self.A.clear()

    @staticmethod
    def _loss_parts(model, batch, spec, ctx):
        logits = model(batch[0]).reshape(len(batch[0]), -1).double()
        total = float(spec.loss(logits, ctx))
        ce = float(torch.nn.functional.cross_entropy(logits, ctx['targets']))
        return total, total-ce if ctx.get('known_classes', 0) else None

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

    def _solve(self, op, mu, B, records, *, tol=None, warm=None, max_iter=None):
        cfg = self.cfg
        def solve(name, rhs, beta=0.):
            start = time.perf_counter()
            action = lambda v: op.F(v)+beta*B(v)+mu*v
            cache = self.warm if warm is None else warm
            initial = cache.get(name)
            if initial is not None and initial.shape != rhs.shape: initial = None
            result = conjugate_gradient(action, rhs.double(), x0=initial,
                max_iter=min(200, cfg.cg_iters if max_iter is None else max_iter),
                tolerance=cfg.cg_tol if tol is None else tol, eps=1e-30)
            if torch.isfinite(result.solution).all(): cache[name] = result.solution.detach().clone()
            records.append(dict(name=name, beta=beta, relative_residual=result.relative_residual,
                iterations=result.iterations, converged=result.converged, rhs_norm=float(rhs.norm()),
                warm_reset=result.warm_reset,
                seconds=time.perf_counter()-start))
            return result.solution
        return solve

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

    def _direction(self, p, op, mu, *, split=None, split_only=False, warm=None):
        records = []
        eold, H_old, chi = self.old_parts(op)
        B = lambda v: p.JT(H_old(p.Jv(v))).double()
        solve = self._solve(op, mu, B, records, warm=warm)
        g, gO = op.gradient, p.JT(eold).double()
        if split is None:
            x = solve('z', g)
            xO = solve('z_O', gO) if bool(eold.any()) else torch.zeros_like(x)
            z, zO = op.H(p.Jv(x)), op.H(p.Jv(xO))
        else: x, xO, z, zO = split
        zN = z-zO
        if split_only:
            return dict(x=x, xO=xO, z=z, zO=zO, zN=zN, eold=eold, g=g, gO=gO, records=records)
        kernel = p.K if self.kernel_self else p.KE
        u, uR, uN = kernel(z).double(), kernel(zO).double(), kernel(zN).double()
        V, VO = max(float(z@u), 0.), max(float(zO@uR), 0.)
        if not bool(eold.any()):
            d = solve('d', p.JT(op.H(u)).double())
            dN, drep = d, torch.zeros_like(d)
            beta = dict(beta=0., beta_branch='unprotected', leak=max(float(d@B(d)), 0.), epsilon_star=None, beta_suf=None)
        else:
            drep = solve('d_rep', p.JT(op.H(uR)).double())
            dN, beta = beta_rule(solve, p.JT(op.H(uN)).double(), B, op.H, uN, VO, chi, eold)
            d = drep+dN
        return dict(d=d, d_N=dN, d_rep=drep, g=g, gO=gO, z=z, zO=zO, zN=zN,
            eold=eold, x=x, xO=xO, V=V, V_O=VO, uN=uN, B=B, H_old=H_old, chi2=chi,
            records=records, solve=solve, kernel=kernel, **beta)

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
        saved = self.state_dict()
        started = time.perf_counter()
        try:
            row = self.apply(model, sites, batch, spec, ctx, optimizer, iteration,
                             remaining=remaining, force=True, apply=False)
        finally:
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

    def apply(self, model, sites, batch, spec, ctx, optimizer, iteration, remaining=None, force=False, apply=True):
        state = self.state_dict() if not apply else None
        try:
            return self._apply(model, sites, batch, spec, ctx, optimizer, iteration, remaining, force, apply)
        finally:
            if state is not None: self.load_state_dict(state)

    @staticmethod
    def _training_state(model, optimizer):
        return (deepcopy(model.state_dict()), deepcopy(optimizer.state_dict()),
                {m: m.training for m in model.modules()}, torch.random.get_rng_state().clone())

    @staticmethod
    def _restore_training_state(model, optimizer, state):
        model_state, optimizer_state, modes, rng = state
        with torch.no_grad():
            model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
        for module, mode in modes.items(): module.training = mode
        torch.random.set_rng_state(rng)

    @staticmethod
    def _forward_logits(model, x):
        output = model(x)
        if isinstance(output, dict): output = output['logits']
        return output.reshape(len(x), -1).double()

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

    def _run_gir(self, model, sites, active, A, batch, spec, ctx, optimizer, units_by_site,
                 iteration, values, mu):
        """Run the T28 dead/live search and return swap logs."""
        if not ('gir' in self.arm and 'cbp' not in self.arm): return []
        e_beta, H_beta = values['e_beta'], values['H_beta']
        rperp = values['rperp_beta']
        records = []
        for site in sites:
            units = units_by_site.get(site.module_name, [])
            if not units: continue
            used_directions = []
            width = site.second_layer.layer.weight.shape[1]
            contributions = {u: self._slot_contribution(model, site, batch, spec, ctx, u)
                             for u in range(width)}
            norms = {u: math.sqrt(max(float(f @ H_beta(f)), 0.)) for u, f in contributions.items()}
            dead = [u for u in units if not bool(contributions[u].any())]
            live = sorted((u for u in units if u not in dead), key=lambda u: norms[u])
            live = live[:int(self.protocol.get('live_candidates', 4))]
            for unit in dead + live:
                proposals = self.prepare_gir(model, site, batch, spec, ctx, optimizer,
                    [unit], iteration, residual=rperp, used_directions=used_directions)
                if not proposals: break
                jump_L, jump_KD = (0., 0.) if unit in dead else self._slot_jump(
                    model, site, batch, spec, ctx, unit)
                if jump_L > 0 or (jump_KD is not None and jump_KD > 0): continue
                best = None
                with training_products(model, active, A, batch[0]) as p0:
                    rhs0 = p0.JT(e_beta).double()
                    current = conjugate_gradient(lambda v: p0.JT(H_beta(p0.Jv(v)))+mu*v,
                        rhs0, max_iter=200, tolerance=self.tol, eps=1e-30)
                    q0 = -.5*float(rhs0 @ current.solution)
                for proposal in proposals:
                    snapshot = self._training_state(model, optimizer)
                    warm, ages = deepcopy(self.warm), deepcopy(self.install_ages)
                    try:
                        with torch.no_grad(): self.install_gir(model, site, [proposal], [unit], optimizer)
                        f_u = contributions[unit]
                        with training_products(model, active, A, batch[0]) as pc:
                            ep = e_beta - H_beta(f_u)
                            rhs = pc.JT(ep).double()
                            action = lambda v: pc.JT(H_beta(pc.Jv(v))) + mu*v
                            cg = conjugate_gradient(action, rhs, max_iter=200,
                                tolerance=self.tol, eps=1e-30)
                            q_prime = -.5*float(rhs @ cg.solution)
                        jump = -float(e_beta @ f_u) + .5*float(f_u @ H_beta(f_u))
                        gamma = -jump + q0-q_prime
                        candidate = dict(site=site.module_name, unit=int(unit),
                            slot_kind='dead' if unit in dead else 'live',
                            f_u_norm_Hb=norms[unit], j_L=jump_L, j_KD=jump_KD,
                            Gamma=gamma, ell_beta=proposal['score'], bias_q=proposal.get('bias_q'),
                            task=self.task, iteration=iteration,
                            cg_iterations=cg.iterations, cg_relative_residual=cg.relative_residual)
                        if gamma > 0 and (unit in dead or (jump_L <= 0 and (jump_KD is None or jump_KD <= 0))):
                            if best is None or gamma > best[0]: best = (gamma, proposal, candidate)
                    finally:
                        self._restore_training_state(model, optimizer, snapshot)
                        self.warm, self.install_ages = warm, ages
                if best is not None:
                    gamma, proposal, record = best
                    self.install_gir(model, site, [proposal], [unit], optimizer)
                    used_directions.append(proposal['direction'])
                    record['contribution_norm'] = {1: None, 5: None, 10: None}
                    self.install_ages[(site.module_name, int(unit))] = dict(
                        site=site.module_name, unit=int(unit), task=self.task, iteration=iteration)
                    records.append(record)
        return records

    def _apply(self, model, sites, batch, spec, ctx, optimizer, iteration, remaining=None, force=False, apply=True):
        if self.coefficient == 0 or self.arm in ('baseline', 'bp', 'cbp', 'icarl') or ('cbp' in self.arm and 'vna' not in self.arm): return None
        if not force and self.last_iteration is not None and iteration-self.last_iteration < self.K_next: return None
        started = time.perf_counter()
        if batch[0].is_cuda: torch.cuda.synchronize(batch[0].device)
        sites = [site for site in sites if self.valid_slot(site)]
        initial = seed_proposals(sites)
        refresh = True
        with training_products(model, sites, initial, batch[0]) as p:
            op = GGNProducts(p, batch[1], spec, ctx)
            trace = op.trace(self.cfg)
            if self.mu is None:
                self.mu = max(1e-12, 1e-2 * trace / max(op.size, 1))
                self.protocol['mu0'] = self.mu
            mu = self.mu
            preliminary = self._direction(p, op, mu, split_only=True)
        if self.gir_only:
            gir_started = time.perf_counter()
            before = self.dead_slots(model, sites, batch)
            swaps, gains, swap_records = 0, [], []
            residual = (preliminary['eold']-preliminary['zO']
                        if bool(preliminary['eold'].any()) else op.g-preliminary['z'])
            for site in sites:
                units = before.get(site.module_name, [])
                if units:
                    proposal = self.prepare_gir(model, site, batch, spec, ctx, optimizer,
                        units, iteration, residual=residual)
                    swaps += self.install_gir(model, site, proposal, units, optimizer)
                    if proposal:
                        gains.append(proposal[0]['score'])
                        swap_records.append(dict(site=site.module_name, unit=int(units[0]),
                            score=proposal[0]['score']))
            next_K = self.K_next * 2
            row = dict(kind='correction', task=self.task, iteration=iteration, step=iteration,
                K=self.K_next, K_next=next_K, refresh_A=False, skipped=False, skip_reason=None,
                d=None, d_N=None, d_rep=None, Delta=0., g_dot_d=0., h_dot_d=0.,
                gradient_norm=float(preliminary['g'].norm()), rho_tr=None, V=0.,
                mu=mu, nu=self.nu, beta=0., beta_branch=None, s=0., s_F_star=0.,
                swaps=swaps, swap_records=swap_records, ell=gains,
                gir_seconds=time.perf_counter()-gir_started,
                seconds=time.perf_counter()-started,
                skip_rate=sum(bool(r.get('skipped')) for r in self.events)/max(1,len(self.events)))
            if apply:
                self.last_iteration, self.K_next = iteration, next_K
                self.applications += 1
                self.events.append(row)
            return row
        selection_started = time.perf_counter()
        A = seed_proposals(sites) if self.kernel_self else self._select(model, sites, batch, preliminary)
        if 'rand' in self.arm: A = self._randomize(model, sites, batch, A)
        tiny_seconds = time.perf_counter()-selection_started
        active = sites if self.kernel_self else [s for s in sites if s.module_name in A]
        if not active:
            next_K = self.K_next * 2
            row = dict(kind='correction', task=self.task, iteration=iteration, step=iteration,
                K=self.K_next, K_next=next_K, refresh_A=refresh, skipped=True,
                skip_reason='no_positive_directions', gradient_norm=float(preliminary['g'].norm()),
                rho_tr=None, V=0., mu=mu, nu=self.nu,
                swaps=0, swap_records=[], seconds=time.perf_counter()-started)
            if apply:
                self.last_iteration, self.K_next = iteration, next_K
                self.applications += 1; self.events.append(row)
            return row
        certificate_started = time.perf_counter()
        with training_products(model, active, A, batch[0]) as p:
            op = GGNProducts(p, batch[1], spec, ctx)
            values = self._direction(p, op, mu, split=(preliminary['x'], preliminary['xO'], preliminary['z'], preliminary['zO']))
            d, g, gO = values['d'], values['g'], values['gO']
            d = d.double()
            Delta = float(values['x'].norm())
            gh, hh = float(g@d), float(gO@d)
            z_record = next((record for record in values['records'] + preliminary['records']
                             if record['name'] == 'z'), None)
            rho = None
            rho_before = rho_after = None
            reason = 'cg_z' if z_record and z_record['relative_residual'] > .1 else ('nonfinite' if not torch.isfinite(d).all() or not math.isfinite(Delta) else 'invalid_beta' if values.get('invalid_beta') else None)
            if reason is None and (gh <= 0 or (bool(values['eold'].any()) and hh < 0)):
                reason = 'negative_certificate'
            if reason is None:
                Jx = p.Jv(values['x']).double()  # float32 nets: keep the T26 model in float64
                pred = float(g @ values['x']) - .5*float(op.H(Jx) @ Jx)
                rho_before, rho_after = self._measure_ratio(model, batch, spec, ctx, p.parameter_refs, values['x'])
                actual = rho_before[0] - rho_after[0]
                rho = actual / pred if pred > 0 else float('-inf')
                if rho > 0:
                    self.mu = mu * max(1/3, 1-(2*rho-1)**3)
                else:
                    self.mu = mu * 2.
                self.nu = 2.
                if rho <= 0: reason = 'untrusted'
            skipped = reason is not None
            only_gir = (self.arm.endswith('_gir') or self.arm == 'gir') and 'vna' not in self.arm
            s, optimum, binding = (0., 0., 'none') if skipped or only_gir else trust_region_step(
                g, d, op.F(d), mu, Delta)
            delta = _unflatten(d, p.parameter_refs)
            known = int(op.ctx.get('known_classes', 0))
            if known and values['beta'] > 0:
                logits_old = p.Jv(values['xO']).reshape_as(op.logits)
                delta_old = torch.zeros_like(op.logits)
                delta_old[:, :known] = -logits_old[:, :known]
                e_beta = op.g-values['beta']*values['H_old'](delta_old.flatten())
                h_beta = lambda v: op.H(v)+values['beta']*values['H_old'](v)
                self.warm['gir_residual'] = values['d_N'].detach().clone()
                gir_x = values['solve']('gir_residual', p.JT(e_beta).double(), values['beta'])
                rperp = e_beta-h_beta(p.Jv(gir_x))
            else:
                e_beta, h_beta, gir_x, rperp = op.g, op.H, values['x'], op.g-values['z']
            values.update(e_beta=e_beta, H_beta=h_beta, gir_x=gir_x, rperp_beta=rperp)
            extension_size = sum(ref.numel() for _, ref in p.extension_refs)
            if extension_size:
                basis = torch.eye(extension_size, dtype=p.extension_refs[0][1].dtype,
                                   device=batch[0].device)
                JE = torch.stack([p.JEv(v).double() for v in basis], 1)
                extension_gram = JE.T @ torch.stack([op.H(column) for column in JE.T], 1)
                lambda_E = max(float(torch.linalg.eigvalsh((extension_gram+extension_gram.T)/2).max()), 0.)
            else:
                lambda_E = 0.
            B_reach = max(float(g @ values['x']), 0.)
            V_t25 = float(values['V'])
            logits_shape = op.logits.shape
            e_output = op.g.reshape(logits_shape)
            blocks = spec.metric_blocks(op.logits, op.ctx)
            range_energy = float(torch.einsum('ni,nij,nj->', e_output,
                torch.linalg.pinv(blocks, hermitian=True), e_output))
            reach = min(1., max(0., B_reach/range_energy)) if range_energy else 0.
            values.update(lambda_E=lambda_E, B_reach=B_reach, V_t25=V_t25,
                          norm_w2=range_energy, reach=reach, unreach=1.-reach,
                          lambda_E_B=lambda_E*B_reach,
                          reach_ratio=V_t25/(lambda_E*B_reach) if lambda_E*B_reach else None)
        certificate_seconds = time.perf_counter()-certificate_started
        before = self.dead_slots(model, sites, batch)
        parameter_before = {n: p.detach().clone() for n, p in model.named_parameters()}
        rejected = False
        if apply and not skipped and s > 0:
            refs = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in delta.items(): refs[name].add_(value.to(refs[name]), alpha=-s)
            after_loss = self._evaluate_current(model, batch, spec, ctx)
            rejected = (rho_before is not None and after_loss[0] > rho_before[0]) or (
                rho_before is not None and rho_before[1] is not None and after_loss[1] > rho_before[1])
            if rejected:
                self._restore_params(model, parameter_before)
                self.mu = mu * 2.
                self.nu = 2.
        else:
            after_loss = rho_before
        after = self.dead_slots(model, sites, batch)
        gir_started = time.perf_counter()
        units_by_site = {site.module_name: list(range(site.second_layer.layer.weight.shape[1])) for site in sites}
        swap_records = self._run_gir(model, sites, active, A, batch, spec, ctx, optimizer,
            units_by_site, iteration, values, mu) if apply and (not skipped or reason == 'untrusted') else []
        swaps = len(swap_records)
        next_K = self.K_next * 2
        if remaining is not None and remaining > 0: next_K = min(next_K, remaining)
        if batch[0].is_cuda: torch.cuda.synchronize(batch[0].device)
        row = dict(kind='correction', task=self.task, iteration=iteration, step=iteration, K=self.K_next,
            K_next=next_K, refresh_A=refresh, rho_tr=rho,
            V=values['V_t25'], B=values['B_reach'], lambda_E=values['lambda_E'],
            norm_w2=values['norm_w2'], reach=values['reach'], unreach=values['unreach'],
            lambda_E_B=values['lambda_E_B'], ratio=values['reach_ratio'],
            V_O=values['V_O'], g_dot_d=gh, h_dot_d=hh,
            Delta=Delta, skipped=skipped, skip_reason=reason, beta=values['beta'], beta_branch=values['beta_branch'],
            beta_over_suf=values['beta']/values['beta_suf'] if values.get('beta_suf') else None,
            leak=values['leak'], epsilon_star=values['epsilon_star'], chi2=values['chi2'],
            s=0. if rejected else s, s_F_star=optimum, cap_binding=binding, rejected=rejected,
            nu=self.nu,
            L_before=None if rho_before is None else rho_before[0], L_after=None if after_loss is None else after_loss[0],
            KD_before=None if rho_before is None else rho_before[1], KD_after=None if after_loss is None else after_loss[1], cg=values['records']+preliminary['records'],
            gradient_norm=float(g.norm()), mu=mu,
            dead_before_step5=sum(map(len, before.values())), dead_after_step5=sum(map(len, after.values())),
            swaps=swaps, swap_records=swap_records, ell=[r['ell_beta'] for r in swap_records], tiny_V_seconds=tiny_seconds,
            certificate_seconds=certificate_seconds, gir_seconds=time.perf_counter()-gir_started,
            solve_seconds=sum(r['seconds'] for r in values['records']+preliminary['records']),
            seconds=time.perf_counter()-started,
            skip_rate=(sum(bool(r.get('skipped')) for r in self.events)+int(skipped))/(len(self.events)+1))
        if apply:
            if not self.kernel_self: self.A = A
            self.last_iteration, self.K_next = iteration, next_K
            self.applications += 1
            self.events.append(row)
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

    def prepare_gir(self, model, site, batch, spec, ctx, optimizer, units, iteration, gradmax=False, residual=None, used_directions=()):
        self.gir_fallback = False
        if not len(units) or not self.valid_slot(site): return []
        with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
            op = GGNProducts(p, batch[1], spec, ctx)
            mu = self.mu if self.mu is not None else max(1e-12, 1e-2*op.trace(self.cfg)/max(op.size, 1))
            if residual is None:
                if gradmax: residual = op.g
                else:
                    values = self._direction(p, op, mu, warm={})
                    known = int(op.ctx.get('known_classes', 0))
                    if known and values['beta'] > 0:
                        delta_old = torch.zeros_like(op.logits)
                        delta_old[:, :known] = -p.Jv(values['xO']).reshape_as(op.logits)[:, :known]
                        e_beta = op.g-values['beta']*values['H_old'](delta_old.flatten())
                        H_beta = lambda v: op.H(v)+values['beta']*values['H_old'](v)
                        self.warm['gir_residual'] = values['d_N'].detach().clone()
                        gir_x = values['solve']('gir_residual', p.JT(e_beta).double(), values['beta'])
                        residual = e_beta-H_beta(p.Jv(gir_x))
                    else:
                        residual = op.g-values['z']
        S, Ns = statistics(model, site, batch, [residual])
        matrix = Ns[0] if gradmax else Ns[0]@Ns[0].T
        if gradmax and (not torch.isfinite(matrix).all() or not torch.isfinite(S).all()):
            self.gir_fallback = True
            return []
        try:
            rows = generalized_rows(matrix, S, 4, euclidean=gradmax)
        except torch.linalg.LinAlgError:
            if not gradmax: raise
            self.gir_fallback = True
            return []
        proposals = []
        bn = next((model.get_submodule(n) for n in site.between
                   if isinstance(model.get_submodule(n), nn.modules.batchnorm._BatchNorm)), None)
        x = self._site_input(model, site, batch)
        first = site.first_layer.layer
        for row in rows:
            row = row.clone()
            for used in used_directions:
                denominator = used @ S @ used
                if denominator > 0:
                    row -= used * ((used @ S @ row)/denominator)
            if float(row @ S @ row) <= 1e-20: continue
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
                    with training_products(model, [site], {site.module_name: candidate}, batch[0]) as pv:
                        size = sum(v.numel() for _, v in pv.extension_refs)
                        Jv = torch.stack([pv.JEv(v).double() for v in torch.eye(size, device=x.device)], 1)
                    if not torch.isfinite(Jv).all() or not bool(Jv.norm()): continue
                    Hv = torch.stack([op.H(v) for v in Jv.T], 1)
                    score = .5*float((Jv.T@residual).square().sum())/(max(float(torch.linalg.eigvalsh(Jv.T@Hv).max()), 0.)+mu)
                    proposals.append(dict(A=candidate, score=score, Jv=Jv, curvature=Jv.T@Hv,
                                          bias_q=bias_q, direction=row.detach().clone()))
        return sorted(proposals, key=lambda value: value['score'], reverse=True)

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
        if count:
            self.warm.clear()
        return count

    def gir_candidates(self, model, site, batch, spec, ctx, optimizer, units, iteration, gradmax=False):
        proposals = self.prepare_gir(model, site, batch, spec, ctx, optimizer, units, iteration, gradmax)
        return self.install_gir(model, site, proposals, units, optimizer)
