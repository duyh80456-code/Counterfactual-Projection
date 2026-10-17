"""Training VNA–GIR v3: joint parameter CG, protected correction and dead slots."""
from contextlib import contextmanager, ExitStack
from copy import copy, deepcopy
from dataclasses import replace
import math
import time
import torch
from torch import nn
from torch.func import functional_call
from diagnostics.vna_align import _global_state
from methods.vna import dedicated_generator, _unflatten
from methods.vna_ggn import GGNConfig, GGNProducts, relative_damping
from methods.vna_sites import HookProducts
from projection.cg import conjugate_gradient


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
        with _global_state(), torch.enable_grad(), torch.autocast(device_type=inputs.device.type, enabled=False):
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
        with _global_state(), torch.enable_grad(), ExitStack() as stack:
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
    ridge = 1e-6*float(S.trace())/len(S)
    if ridge <= 0: return S.new_empty((0, len(S)))
    L = torch.linalg.cholesky(S+ridge*torch.eye(len(S), device=S.device, dtype=S.dtype))
    if euclidean:
        values, vectors = torch.linalg.eigh((matrix+matrix.T)/2)
        rows = vectors.T
    else:
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


def spectral_norm(action, transpose, template, generator, iterations):
    v = torch.randn(template.shape, device=template.device, dtype=torch.float64, generator=generator)
    v /= v.norm().clamp_min(1e-300)
    for _ in range(iterations):
        v = transpose(action(v)).double()
        if float(v.norm()) == 0: return 0.
        v /= v.norm()
    return 1.1*float(action(v).norm())


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


def correction_step(g, d, Fd, mu, lr, K, momentum=0., faithful=False):
    gain = float(g@d)
    denominator = float(d@Fd)+mu*float(d@d)
    if gain <= 0 or denominator <= 0: return 0., 0., 'none'
    optimum = gain/denominator
    cap = K*lr/(1-momentum) if faithful else lr*float(g.norm())/max(float(d.norm()), 1e-300)
    return min(optimum, cap), optimum, 's*' if optimum <= cap else 'norm' if not faithful else 'faithful'


def schedule_from_points(points, iterations_per_epoch, final_classes=100):
    second = [p for p in points if p.get('task') == 2] or points
    cost = sum(p['vna_seconds'] for p in second)/len(second)
    classes = second[-1].get('classes', second[-1].get('n_out', 2560)/128)
    projected = cost*final_classes/classes
    iteration = sum(p['iteration_seconds'] for p in points)/len(points)
    candidates = sorted(set([50, 100, 200, 400, 800]+[m*iterations_per_epoch for m in (1, 2, 5, 10)]))
    feasible = [k for k in candidates if projected/k <= iteration]
    return dict(K_base=min(feasible) if feasible else max(candidates), budget_exceeded=not bool(feasible),
        t_vna_task1=[p['vna_seconds'] for p in points if p.get('task') == 1],
        t_vna_task2=[p['vna_seconds'] for p in points if p.get('task') == 2],
        projected_vna_seconds=projected, iteration_seconds=iteration, candidates=candidates,
        iterations_per_epoch=iterations_per_epoch, calibration=deepcopy(points),
        extrapolation='linear in classes, task2 to final task', K_units='iterations')


class V3Engine:
    def __init__(self, arm='vna', *, seed=0, coefficient=1., K_base=100):
        self.arm, self.seed, self.coefficient = arm, seed, coefficient
        self.K_base, self.every = K_base, K_base
        self.mu_rel, self.tol, self.power_iters = .01, 1e-3, 5
        self.task, self.last_iteration, self.K_next = None, None, 1
        self.applications, self.last_refresh = 0, -10
        self.warm, self.A = {}, {}
        self.events, self.activities, self.calibration = [], [], []
        self.solver_tightened = False
        self.exact_checks, self.false_skips = 0, 0

    @property
    def cfg(self):
        return GGNConfig(damping_rel=self.mu_rel, cg_iters=50, cg_tol=self.tol,
                         hutchinson_probes=8, seed=self.seed, project_bn=False)

    def state_dict(self): return deepcopy(vars(self))
    def load_state_dict(self, state): vars(self).update(deepcopy(state))

    @property
    def kernel_self(self): return self.arm.startswith('kernel_self')

    @property
    def gir_only(self): return self.arm == 'gir'

    def configure_schedule(self, schedule):
        self.K_base = self.every = int(schedule['K_base'])
        if 'cg_tol' in schedule: self.tol = schedule['cg_tol']
        for name in ('mu_rel', 'tol', 'power_iters'):
            if name in schedule: setattr(self, name, schedule[name])

    def begin_task(self, task):
        if task != self.task:
            self.task, self.last_iteration, self.K_next = task, None, 1
            self.applications, self.last_refresh = 0, -10
            self.warm.clear(); self.A.clear()

    def _solve(self, op, mu, B, records, *, tol=None, warm=None, max_iter=None):
        cfg = self.cfg
        def solve(name, rhs, beta=0.):
            start = time.perf_counter()
            action = lambda v: op.F(v)+beta*B(v)+mu*v
            cache = self.warm if warm is None else warm
            initial = cache.get(name)
            if initial is not None and initial.shape != rhs.shape: initial = None
            result = conjugate_gradient(action, rhs.double(), x0=initial,
                max_iter=cfg.cg_iters if max_iter is None else max_iter,
                tolerance=cfg.cg_tol if tol is None else tol, eps=1e-30)
            if torch.isfinite(result.solution).all(): cache[name] = result.solution.detach().clone()
            records.append(dict(name=name, beta=beta, relative_residual=result.relative_residual,
                iterations=result.iterations, converged=result.converged, rhs_norm=float(rhs.norm()),
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

    def _direction(self, p, op, mu, *, exact=False, split=None, split_only=False, warm=None):
        records = []
        eold, H_old, chi = self.old_parts(op)
        B = lambda v: p.JT(H_old(p.Jv(v))).double()
        solve = self._solve(op, mu, B, records, tol=1e-8 if exact else None,
                            warm={} if exact else warm, max_iter=2000 if exact else None)
        g, gO = op.gradient, p.JT(eold).double()
        if split is None or exact:
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
        if 'b0' in self.arm or not bool(eold.any()):
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

    def preflight(self, model, sites, batch, spec, ctx, optimizer, iteration=1):
        saved = self.state_dict()
        trials = []
        try:
            for mu_rel in (.001, .01, .1):
                self.mu_rel = mu_rel
                row = self.apply(model, sites, batch, spec, ctx, optimizer, iteration, force=True, apply=False)
                trials.append({**row, 'mu_rel': mu_rel})
        finally: self.load_state_dict(saved)
        selected = next((r for r in trials if r['Delta_hat'] <= .1*abs(r['g_dot_d'])/max(r['gradient_norm'], 1e-300)), trials[-1])
        return dict(mu_rel=selected['mu_rel'], mu_condition_met=selected['Delta_hat'] <= .1*abs(selected['g_dot_d'])/max(selected['gradient_norm'], 1e-300), trials=trials)

    def apply(self, model, sites, batch, spec, ctx, optimizer, iteration, remaining=None, force=False, apply=True):
        state = self.state_dict() if not apply else None
        try:
            return self._apply(model, sites, batch, spec, ctx, optimizer, iteration, remaining, force, apply)
        finally:
            if state is not None: self.load_state_dict(state)

    def _apply(self, model, sites, batch, spec, ctx, optimizer, iteration, remaining=None, force=False, apply=True):
        if self.coefficient == 0 or self.arm in ('baseline', 'bp', 'cbp', 'icarl', 'er', 'erace') or ('cbp' in self.arm and 'vna' not in self.arm): return None
        if not force and self.last_iteration is not None and iteration-self.last_iteration < self.K_next: return None
        started = time.perf_counter()
        if batch[0].is_cuda: torch.cuda.synchronize(batch[0].device)
        sites = [site for site in sites if self.valid_slot(site)]
        initial = seed_proposals(sites)
        refresh = not self.A or self.applications-self.last_refresh >= 10
        with training_products(model, sites, initial, batch[0]) as p:
            op = GGNProducts(p, batch[1], spec, ctx)
            trace = op.trace(self.cfg)
            mu = relative_damping(self.cfg, trace, op)
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
                        units[:1], iteration, residual=residual)
                    swaps += self.install_gir(model, site, proposal, units[:1], optimizer)
                    if proposal:
                        gains.append(proposal[0]['score'])
                        swap_records.append(dict(site=site.module_name, unit=int(units[0]),
                            score=proposal[0]['score']))
                    break
            next_K = self.K_base if remaining is None or remaining <= 0 else min(self.K_base, remaining)
            row = dict(kind='correction', task=self.task, iteration=iteration, step=iteration,
                K=self.K_next, K_next=next_K, refresh_A=False, skipped=False, skip_reason=None,
                d=None, d_N=None, d_rep=None, Delta_hat=0., g_dot_d=0., h_dot_d=0.,
                gradient_norm=float(preliminary['g'].norm()), rho=0., rho_proj=0., V=0., V_O=0.,
                mu=mu, mu_rel=self.mu_rel, beta=0., beta_branch=None, s=0., s_star=0.,
                swaps=swaps, swap_records=swap_records, ell=gains, exact_cg=None,
                exact_cg_converged=None, false_skip=False, solver_tightened=self.solver_tightened,
                power_iters=self.power_iters, gir_seconds=time.perf_counter()-gir_started,
                seconds=time.perf_counter()-started,
                skip_rate=sum(bool(r.get('skipped')) for r in self.events)/max(1,len(self.events)))
            if apply:
                self.last_iteration, self.K_next = iteration, next_K
                self.applications += 1
                self.events.append(row)
            return row
        selection_started = time.perf_counter()
        A = self.A
        if refresh:
            A = self._select(model, sites, batch, preliminary)
            selected_sites = [s for s in sites if s.module_name in A]
            if A and bool(preliminary['eold'].any()) and 'b0' not in self.arm:
                with training_products(model, selected_sites, A, batch[0]) as p:
                    op = GGNProducts(p, batch[1], spec, ctx)
                    first = self._direction(p, op, mu, split=(preliminary['x'], preliminary['xO'], preliminary['z'], preliminary['zO']))
                    y = op.H(p.Jv(first['solve']('alternating_y', first['g'], first['beta'])))
                A = self._select(model, sites, batch, preliminary,
                    matrix_values=[preliminary['z'], preliminary['zO'], y, preliminary['zN']])
            if 'rand' in self.arm: A = self._randomize(model, sites, batch, A)
        tiny_seconds = time.perf_counter()-selection_started
        active = [s for s in sites if s.module_name in A]
        if not active:
            next_K = min(self.K_base, remaining) if remaining is not None and remaining > 0 else self.K_base
            row = dict(kind='correction', task=self.task, iteration=iteration, step=iteration,
                K=self.K_next, K_next=next_K, refresh_A=refresh, skipped=True,
                skip_reason='no_positive_directions', Delta_hat=float('inf'), g_dot_d=0.,
                h_dot_d=0., gradient_norm=float(preliminary['g'].norm()), rho=0., V=0.,
                V_O=0., mu=mu, mu_rel=self.mu_rel, cos_cg_exact=None,
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
            raw_d = d
            d = torch.cat([v.to(ref).double().flatten() for (_, ref), v in zip(p.parameter_refs,
                d.split([r.numel() for _, r in p.parameter_refs]))])
            generator = dedicated_generator(self.seed, f'v3-power:{self.task}:{self.applications}', batch[0].device)
            nHJ = spectral_norm(lambda v: op.H(p.Jv(v)), lambda v: p.JT(op.H(v)).double(), g, generator, self.power_iters)
            nJHK = spectral_norm(lambda v: p.JT(op.H(values['kernel'](v))).double(),
                lambda v: values['kernel'](op.H(p.Jv(v))).double(), op.g, generator, self.power_iters)
            error_z = float((g-op.F(values['x'])-mu*values['x']).norm())
            error_old = float((gO-op.F(values['xO'])-mu*values['xO']).norm())
            if bool(values['eold'].any()) and values['beta_branch'] != 'unprotected':
                error_new = float(((g-gO)-op.F(values['x']-values['xO'])-mu*(values['x']-values['xO'])).norm())
                rhs_R = p.JT(op.H(values['kernel'](values['zO']))).double()
                rhs_N = p.JT(op.H(values['uN'])).double()
                err_R = float((rhs_R-op.F(values['d_rep'])-mu*values['d_rep']).norm())
                err_N = float((rhs_N-op.F(values['d_N'])-values['beta']*values['B'](values['d_N'])-mu*values['d_N']).norm())
                Delta = nJHK*nHJ*(error_old+error_new)/mu**2+(err_R+err_N)/mu
            else:
                rhs = p.JT(op.H(values['kernel'](values['z']))).double()
                Delta = nJHK*nHJ*error_z/mu**2+float((rhs-op.F(d)-mu*d).norm())/mu
            Delta += float((d-raw_d).norm())
            gh, hh = float(g@d), float(gO@d)
            tau_g, tau_h = float(g.norm())*Delta, float(gO.norm())*Delta
            reason = 'nonfinite' if not torch.isfinite(d).all() or not math.isfinite(Delta) else 'invalid_beta' if values.get('invalid_beta') else 'negative_total_certificate' if gh < -tau_g else 'negative_old_certificate' if hh < -tau_h else None
            skipped = reason is not None
            cos_exact, h_exact, false_skip = None, None, False
            exact_records = []
            if self.task in (1, 2) and self.applications < 3:
                exact = self._direction(p, op, mu, exact=True)
                exact_records = exact['records']
                cos_exact = float(nn.functional.cosine_similarity(d, exact['d'], dim=0)) if d.norm() and exact['d'].norm() else 1.
                h_exact = float(exact['gO']@exact['d'])
                exact_converged = all(r['converged'] for r in exact_records)
                false_skip = bool(exact_converged and skipped and h_exact >= 0)
                if apply and exact_converged:
                    self.exact_checks += 1; self.false_skips += int(false_skip)
                    if cos_exact < .99: self.tol, self.solver_tightened = 1e-4, True
                    if self.false_skips/self.exact_checks > .1: self.power_iters = 20
            only_gir = (self.arm.endswith('_gir') or self.arm == 'gir') and 'vna' not in self.arm
            s, optimum, binding = (0., 0., 'none') if skipped or only_gir else correction_step(g, d,
                op.F(d), mu, optimizer.param_groups[0]['lr'], self.K_next,
                optimizer.param_groups[0].get('momentum', 0.), self.arm.endswith('_lf'))
            delta = _unflatten(d, p.parameter_refs)
            phi_N = float((p.Jv(values['d_N'])-values['uN'])@op.H(p.Jv(values['d_N'])-values['uN']))+mu*float(values['d_N']@values['d_N'])
            projected_gain = float(p.JT(values['z']).double().square().sum())
        certificate_seconds = time.perf_counter()-certificate_started
        before = self.dead_slots(model, sites, batch)
        if apply and not skipped and s > 0:
            refs = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in delta.items(): refs[name].add_(value.to(refs[name]), alpha=-s)
        after = self.dead_slots(model, sites, batch)
        gir_started = time.perf_counter(); swaps, gains, swap_records = 0, [], []
        if apply and not skipped and ('gir' in self.arm and 'cbp' not in self.arm):
            for site in sites:
                units = after.get(site.module_name, [])
                if units:
                    proposal = self.prepare_gir(model, site, batch, spec, ctx, optimizer, units[:1], iteration,
                        residual=(values['eold']-values['zO']) if bool(values['eold'].any()) else (op.g-values['z']))
                    swaps += self.install_gir(model, site, proposal, units[:1], optimizer)
                    if proposal:
                        gains.append(proposal[0]['score'])
                        swap_records.append(dict(site=site.module_name, unit=int(units[0]), score=proposal[0]['score']))
                    break
        rho = values['V']/max(float(g.square().sum()), 1e-300)
        next_K = self.K_base if skipped else max(1, int(math.floor(self.K_base*min(4., max(.25, .3/max(rho, 1e-300))))+.5))
        if remaining is not None and remaining > 0: next_K = min(next_K, remaining)
        if batch[0].is_cuda: torch.cuda.synchronize(batch[0].device)
        row = dict(kind='correction', task=self.task, iteration=iteration, step=iteration, K=self.K_next,
            K_next=next_K, refresh_A=refresh, rho=rho, rho_proj=values['V']/max(projected_gain, 1e-300),
            V=values['V'], V_O=values['V_O'], g_dot_d=gh, h_dot_d=hh, tau_g=tau_g, tau_h=tau_h,
            Delta_hat=Delta, skipped=skipped, skip_reason=reason, beta=values['beta'], beta_branch=values['beta_branch'],
            beta_over_suf=values['beta']/values['beta_suf'] if values.get('beta_suf') else None,
            leak=values['leak'], epsilon_star=values['epsilon_star'], chi2=values['chi2'], Phi_N=phi_N,
            kappa=None, kappa_reason='V_NS is not defined in supplied theory', s=s, s_star=optimum, cap_binding=binding, cg=values['records']+preliminary['records'],
            cos_cg_exact=cos_exact, exact_cg=exact_records,
            exact_cg_converged=all(r['converged'] for r in exact_records) if exact_records else None,
            h_exact=h_exact, false_skip=false_skip, solver_tightened=self.solver_tightened,
            power_iters=self.power_iters, gradient_norm=float(g.norm()), mu=mu, mu_rel=self.mu_rel,
            dead_before_step5=sum(map(len, before.values())), dead_after_step5=sum(map(len, after.values())),
            swaps=swaps, swap_records=swap_records, ell=gains, gain=None, tiny_V_seconds=tiny_seconds,
            certificate_seconds=certificate_seconds, gir_seconds=time.perf_counter()-gir_started,
            solve_seconds=sum(r['seconds'] for r in values['records']+preliminary['records']),
            seconds=time.perf_counter()-started, certificate_norms='power estimates x1.1',
            skip_rate=(sum(bool(r.get('skipped')) for r in self.events)+int(skipped))/(len(self.events)+1))
        if apply:
            self.A = A
            if refresh: self.last_refresh = self.applications
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
            with _global_state(), torch.no_grad():
                model.train()
                functional_call(model, ({n: p for n, p in model.named_parameters()}, buffers), (batch[0],))
        finally:
            for h in handles: h.remove()
            for m, mode in modes.items(): m.training = mode
        return {name: torch.where(value.movedim(1, 0).flatten(1).eq(0).all(1))[0].tolist()
                for name, value in captures.items()}

    def prepare_gir(self, model, site, batch, spec, ctx, optimizer, units, iteration, gradmax=False, residual=None):
        if not len(units) or not self.valid_slot(site): return []
        with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
            op = GGNProducts(p, batch[1], spec, ctx)
            mu = relative_damping(self.cfg, op.trace(self.cfg), op)
            if residual is None:
                values = self._direction(p, op, mu, split_only=True, warm={})
                residual = op.g if gradmax else values['eold']-values['zO'] if bool(values['eold'].any()) else op.g-values['z']
        S, Ns = statistics(model, site, batch, [residual])
        rows = generalized_rows(Ns[0]@Ns[0].T, S, 4, euclidean=gradmax)
        proposals = []
        for row in rows:
            base = rows_proposal(site, row[None])
            for sign in (1., -1.):
                trial = {k: None if v is None else sign*v for k, v in base.items()}
                bn = next((model.get_submodule(n) for n in site.between if isinstance(model.get_submodule(n), nn.modules.batchnorm._BatchNorm)), None)
                x = self._site_input(model, site, batch)
                first = site.first_layer.layer
                projected = nn.functional.linear(x, trial['weight'], trial['bias']) if site.kind == 'linear' else nn.functional.conv2d(x, trial['weight'], trial['bias'], first.stride, first.padding, first.dilation)
                if bn is not None:
                    axes = (0, *range(2, projected.ndim))
                    projected = (projected-projected.mean(axes, keepdim=True))/(projected.var(axes, unbiased=False, keepdim=True)+bn.eps).sqrt()
                for shift in (0., -float(projected.median())):
                    candidate = deepcopy(trial)
                    if bn is not None:
                        candidate['bn_weight'] = torch.ones(1, device=x.device)
                        candidate['bn_bias'] = torch.tensor([shift], device=x.device)
                    elif first.bias is not None: candidate['bias'] = trial['bias']+shift
                    elif shift != 0: continue
                    if not bool((projected+shift > 0).any()): continue
                    with training_products(model, [site], {site.module_name: candidate}, batch[0]) as pv:
                        size = sum(v.numel() for _, v in pv.extension_refs)
                        Jv = torch.stack([pv.JEv(v).double() for v in torch.eye(size, device=x.device)], 1)
                    Hv = torch.stack([op.H(v) for v in Jv.T], 1)
                    score = .5*float((Jv.T@residual).square().sum())/(max(float(torch.linalg.eigvalsh(Jv.T@Hv).max()), 0.)+mu)
                    proposals.append(dict(A=candidate, score=score, Jv=Jv, curvature=Jv.T@Hv))
        selected = []
        if not proposals: return selected
        complement = residual.clone()
        deflations = []
        with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
            live = GGNProducts(p, batch[1], spec, ctx)
            solve = self._solve(live, mu, lambda v: torch.zeros_like(v), [])
            def Hperp(v):
                value = live.H(v)-live.H(p.Jv(solve('gir_deflate', p.JT(live.H(v)).double())))
                for left, inverse in deflations:
                    value = value-left@(inverse@(left.T@v))
                return value
            for _ in range(min(len(units), len(proposals))):
                for candidate in proposals:
                    candidate['score'] = .5*float((candidate['Jv'].T@complement).square().sum())/(max(float(torch.linalg.eigvalsh(candidate['curvature']).max()), 0.)+mu)
                best = max(proposals, key=lambda p: p['score'])
                proposals = [candidate for candidate in proposals if candidate is not best]
                selected.append(dict(A=best['A'], score=best['score']))
                if site.kind == 'linear' and len(selected) < len(units):
                    Jv = best['Jv']
                    left = torch.stack([Hperp(v) for v in Jv.T], 1)
                    schur = Jv.T@left+mu*torch.eye(Jv.shape[1], device=Jv.device)
                    inverse = torch.linalg.inv(schur)
                    complement = complement-left@(inverse@(Jv.T@complement))
                    deflations.append((left, inverse))
        return selected

    @staticmethod
    def _site_input(model, site, batch):
        capture = {}
        def hook(module, args): capture['x'] = args[0].detach()
        handle = site.first_layer.layer.register_forward_pre_hook(hook)
        modes = {m: m.training for m in model.modules()}
        try:
            with _global_state(), torch.no_grad():
                model.train()
                functional_call(model, ({n: p for n, p in model.named_parameters()},
                    {n: b.clone() for n, b in model.named_buffers()}), (batch[0],))
        finally:
            handle.remove()
            for m, mode in modes.items(): m.training = mode
        return capture['x']

    @staticmethod
    def install_gir(model, site, proposals, units, optimizer):
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
                count += 1
        return count

    def gir_candidates(self, model, site, batch, spec, ctx, optimizer, units, iteration, gradmax=False):
        proposals = self.prepare_gir(model, site, batch, spec, ctx, optimizer, units, iteration, gradmax)
        return self.install_gir(model, site, proposals, units, optimizer)
