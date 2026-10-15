"""Shared whole-network solves; null row p-values remain valid but correlated
across sites/ranks of a measurement point because their null draws are shared.
"""
import json
import torch
from diagnostics.stage_a import null_residual, empirical_p
from methods.vna_joint import joint_products, _inverse
from methods.vna_ggn import GGNProducts, relative_damping


def virtual_energy(model, sites, A_by_site, data, z, e):
    """Compute V and U with VJPs only, using the original gradient denominator."""
    with joint_products(model, sites, A_by_site, data[0]) as p:
        V = float(p.JET(z.flatten()).double().square().sum())
        U = float(p.JET(e.flatten()).double().square().sum())
        norm = float(p.JT(e.flatten()).double().square().sum())
    return {'V': V, 'U': U, 'eta': V/U if U else 0., 'rho': V/norm if norm else 0.,
            'gradient_norm2': norm, 'e_norm2': float(e.double().square().sum()),
            'nu': V/float(e.double().square().sum()) if e.square().sum() else 0.}


def solve_z(residual, data, ctx):
    """One whole-network solve; ctx supplies model, scope, damping and loss."""
    with joint_products(ctx['model'], [ctx['site']], {ctx['site']: ctx['seed']},
                        data[0], ctx['names']) as p:
        op = GGNProducts(p, data[1], ctx['loss_spec'], ctx['loss_ctx'])
        gradient = p.JT(residual.flatten()).double()
        u, record = _inverse(lambda v: op.F(v)+ctx['mu']*v, gradient, ctx['cfg'])
        return op.H(p.Jv(u)).reshape_as(residual), record


class SolveCache:
    def __init__(self, model, site, stat, loss_spec, loss_ctx, cfg, *, mu=None, trace=None, solver="cg", output_exact_max=4096, factors=None):
        layer = site.first_layer.layer
        seed = {'weight': layer.weight[:1].detach(),
                'bias': None if layer.bias is None else layer.bias[:1].detach()}
        names = tuple(n for n, p in model.named_parameters() if p.requires_grad)
        with joint_products(model, [site], {site: seed}, stat[0], names) as p:
            op = GGNProducts(p, stat[1], loss_spec, loss_ctx)
            self.trace = op.trace(cfg) if trace is None else trace
            self.mu = relative_damping(cfg, self.trace, op) if mu is None else mu
        self.ctx = dict(model=model, site=site, seed=seed, names=names,
                        loss_spec=loss_spec, cfg=cfg, mu=self.mu)
        self.solver, self.output_exact_max = solver, output_exact_max
        self.factors = {} if factors is None else factors
        self.null_draw_id = 'null'
        self.cache = {}
        self.records = {}

    def solve(self, residual_id, residual, data, context):
        key = (self.mu, residual_id)
        if key not in self.cache:
            from experiments.vna_resume import measurement_update
            measurement_update(solve=str(residual_id), mu=self.mu)
            factor = self.factor(data, context)
            if factor is None:
                z, record = solve_z(residual, data, {**self.ctx, 'loss_ctx': context})
            else:
                import time
                from experiments.vna_resume import current_measurement
                monitor = current_measurement.get()
                if monitor: monitor.begin_solve()
                started = time.perf_counter()
                z = factor.z(residual, self.mu)
                record = dict(converged=factor.range_ok, iterations=0, relative_residual=factor.range_error)
                if monitor: monitor.end_solve({**record, 'seconds': time.perf_counter()-started})
            self.cache[key], self.records[key] = z, record
        return self.cache[key]

    def factor(self, data, context, *, parent=None, mask=None):
        from methods.vna_output import OutputFactor, choose_solver
        key = data[0]  # tensor identity hash also retains the batch for safe reuse
        if key not in self.factors:
            with joint_products(self.ctx['model'], [self.ctx['site']],
                    {self.ctx['site']: self.ctx['seed']}, data[0], self.ctx['names']) as p:
                op = GGNProducts(p, data[1], self.ctx['loss_spec'], context)
                solver, reason = choose_solver(self.solver, op.g.numel(), self.output_exact_max)
                matrix = None
                if parent is not None:
                    indices = mask[:, None].expand(-1, parent.shape[1]).flatten().nonzero().flatten()
                    matrix = parent.matrix[indices.to(parent.matrix.device)][:, indices.to(parent.matrix.device)] * (parent.shape[0]/len(data[0]))
                self.factors[key] = OutputFactor(op, matrix=matrix) if solver == 'output_exact' else None
                from experiments.vna_resume import log
                factor = self.factors[key]
                from experiments.vna_resume import current_measurement
                monitor = current_measurement.get()
                if monitor:
                    if factor: monitor.factors.append(factor.metadata())
                    choice = dict(solver=solver, reason=reason, n_out=op.g.numel())
                    if choice not in monitor.solver_choices: monitor.solver_choices.append(choice)
                else:
                    log(f"solver={solver} reason={reason} n_out={op.g.numel()} " +
                        (str(factor.metadata()) if factor else 'matrix-free CG'))
        return self.factors[key]

    def metadata(self, data, context):
        from methods.vna_output import choose_solver
        with torch.no_grad(): n_out = self.ctx['model'](data[0][:1]).numel()*len(data[0])
        factor = self.factor(data, context)
        reason = choose_solver(self.solver, n_out, self.output_exact_max)[1]
        return {**factor.metadata(), 'solver_reason': reason} if factor else dict(solver='cg', solver_reason=reason, n_out=n_out,
            M_build_seconds=0., factor_seconds=0., M_memory_bytes=0, min_eig=None, max_eig=None,
            residual_in_range_H=None)

    def converged(self, *, null=False):
        return all(r['converged'] for (_, key), r in self.records.items()
                   if str(key).startswith('null') == null)


def draw_nulls(cache, g, e, stat, evaluation, stat_ctx, eval_ctx, generator, repeats):
    from methods.vna import dedicated_generator
    generator = dedicated_generator(cache.ctx['cfg'].seed, cache.null_draw_id, g.device)
    result = {}
    for kind in ('N1', 'N2'):
        draws = []
        for repeat in range(repeats):
            a = null_residual(g, kind, generator,
                             residual_is_zero_sum=cache.ctx['loss_spec'].residual_is_zero_sum)
            b = null_residual(e, kind, generator,
                             residual_is_zero_sum=cache.ctx['loss_spec'].residual_is_zero_sum)
            label = f'null:{kind}:{repeat}'
            za = cache.solve(label+':stat', a, stat, stat_ctx)
            zb = cache.solve(label+':eval', b, evaluation, eval_ctx)
            draws.append((za, zb, b))
        result[kind] = draws
    return result


def null_columns(lambda1, rho, values, nu=None):
    result = {}
    for kind, draws in values.items():
        lambdas, rhos = draws[:2]
        pl, pr = empirical_p(lambda1, lambdas), empirical_p(rho, rhos)
        result.update({f'p_lambda_{kind}': pl, f'p_rho_{kind}': pr,
                       f'significant_{kind}': pl <= 1/9, f'significant_rho_{kind}': pr <= 1/9,
                       f'null_lambda_{kind}': json.dumps(lambdas), f'null_rho_{kind}': json.dumps(rhos)})
        if len(draws) > 2:
            energies = draws[2]
            pn = empirical_p(nu, [r['nu'] for r in energies])
            result.update({f'p_nu_{kind}': pn, f'significant_nu_{kind}': pn <= 1/9,
                           **{f'null_{key}_{kind}': json.dumps([r[key] for r in energies])
                              for key in ('V', 'e_norm2', 'nu')}})
    return result


def verify_energy(energy, correction):
    """Report relative V disagreement; callers exclude inconsistent rows."""
    relative = abs(energy['V']-correction['V']) / max(abs(correction['V']), 1e-300)
    return relative
