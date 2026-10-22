"""T29 matrix-free solves, with dense Gram/Woodbury retained as a small reference."""
import time
import math
from copy import copy, deepcopy
import torch
from projection.cg import conjugate_gradient


class CGConvergenceError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"CG did not converge: {record}")


class FrozenGram:
    """JJ^T products at one frozen linearization, plus accepted column swaps."""
    def __init__(self, products, high=None):
        self.products = products
        device = products.parameter_refs[0][1].device
        self.removed = torch.empty(0,dtype=torch.long,device=device)
        self.added_ids = self.removed
        self.added = None
        self.version = 0
        self.warm = {}
        if high is not None:
            self.high = high
        elif products.parameter_refs[0][1].dtype == torch.float64:
            self.high = self
        else:
            from methods.vna_sites import HookProducts
            model = deepcopy(products.model).double()
            names = tuple(n for n,_ in products.parameter_refs)
            hp = HookProducts(model,[],{},products.inputs.double(),names,bn_mode=products.bn_mode)
            self.high = FrozenGram(hp)

    def Jv(self, v):
        native = v.clone().index_fill(0,self.removed,0.)
        image = self.products.Jv(native).double()
        if self.added is not None: image = image+self.added@v[self.added_ids]
        return image

    def JT(self, v):
        tangent = self.products.JT(v).double().clone().index_fill(0,self.removed,0.)
        if self.added is not None: tangent[self.added_ids] = self.added.T@v
        return tangent

    def __matmul__(self, v):
        return self.Jv(self.JT(v))

    def swapped(self, native_ids, outgoing_ids, add, high_add=None):
        G = copy(self)
        G.warm = dict(self.warm)
        G.removed = torch.unique(torch.cat((self.removed,native_ids)))
        if self.added is None:
            G.added_ids, G.added = outgoing_ids, add
        else:
            keep = ~torch.isin(self.added_ids,native_ids)
            G.added_ids = torch.cat((self.added_ids[keep],outgoing_ids))
            G.added = torch.cat((self.added[:,keep],add),1)
        G.version = self.version+1
        if self.high is self:
            G.high = G
        else:
            if high_add is None: raise ValueError('float64 replacement columns are required for IR')
            G.high = self.high.swapped(native_ids,outgoing_ids,high_add)
        return G


class OutputCG:
    def __init__(self, G, blocks, mu, *, max_iter=200, tolerance=1e-10):
        self.G, self.blocks, self.mu = G, blocks.double(), float(mu)
        if any(p.dtype!=torch.float64 for _,p in G.high.products.parameter_refs):
            raise ValueError('IR residual requires actual float64 model products')
        self.max_iter, self.tolerance = max_iter, tolerance
        self.solves, self.records, self._q_key = 0, [], None
        self._warm = None
        self._solution_key = None

    def metric(self, v):
        return torch.einsum('nij,nj->ni', self.blocks, v.double().reshape(self.blocks.shape[:2])).reshape(v.shape)

    def solve(self, c, x0=None):
        original_c = c
        key = (id(c),c._version,self.G.version,self.blocks._version,self.mu)
        if self._solution_key == key:
            self.records.append(dict(self.records[-1],cached=True,iterations=0,refinements=0,
                                     inner_solves=[],high_seconds=0.,inner_seconds=0.,seconds=0.))
            return self._solution.clone()
        self.solves += 1
        start = time.perf_counter()
        c = c.double()
        high = self.G.high
        action = lambda v: self.mu*v+self.G.JT(self.metric(self.G.Jv(v)))
        exact_action = lambda v: self.mu*v+high.JT(self.metric(high.Jv(v)))
        high_start = time.perf_counter()
        rhs = high.JT(c)
        rhs_norm = float(rhs.norm())
        high_seconds = time.perf_counter()-high_start
        solution = x0 if x0 is not None else self._warm
        cold = solution is None
        solution = torch.zeros_like(rhs) if cold else solution.detach().double().clone()
        high_start = time.perf_counter()
        residual = rhs.clone() if cold else rhs-exact_action(solution)
        float(residual.norm())
        high_seconds += time.perf_counter()-high_start
        warm_reset = not bool(torch.isfinite(solution).all()) or float(residual.norm())>rhs_norm
        if warm_reset: solution.zero_(); residual = rhs.clone()
        history, inner, used, reason, inner_seconds = [], [], 0, None, 0.
        inner_tolerance = self.tolerance if high is self.G else 1e-4
        for refinement in range(9):
            relative = float(residual.norm())/max(rhs_norm,1e-300)
            history.append(relative)
            if math.isfinite(relative) and relative<=self.tolerance: break
            if not math.isfinite(relative): reason='nonfinite'; break
            if len(history)>1 and relative>=.99*history[-2]: reason='ir_stagnation'; break
            if used>=self.max_iter: reason='inner_budget'; break
            if refinement==8: reason='ir_budget'; break
            inner_start = time.perf_counter()
            correction = conjugate_gradient(action,residual,max_iter=self.max_iter-used,
                tolerance=inner_tolerance,eps=1e-30)
            inner_seconds += time.perf_counter()-inner_start
            used += correction.iterations
            inner.append(dict(iterations=correction.iterations,converged=correction.converged,
                              relative_residual=correction.relative_residual))
            if not bool(torch.isfinite(correction.solution).all()): reason='inner_nonfinite'; break
            solution += correction.solution
            high_start = time.perf_counter()
            residual = rhs-exact_action(solution)
            float(residual.norm())
            high_seconds += time.perf_counter()-high_start
            # Only a true double residual accepts a correction, including an inexact inner solve.
        relative = float(residual.norm())/max(rhs_norm,1e-300)
        converged = math.isfinite(relative) and relative<=self.tolerance
        record = dict(solver='cg_ir',iterations=used,refinements=len(inner),cached=False,
            relative_residual=relative,parameter_relative_residual=relative,
            residual_dtype='torch.float64',vector_dtype=str(rhs.dtype),
            parameter_dtype=str(self.G.products.parameter_refs[0][1].dtype),
            tolerance=self.tolerance,inner_tolerance=inner_tolerance,mu=self.mu,
            rhs_norm=rhs_norm,residual_norm=float(residual.norm()),
            residual_history=history,inner_solves=inner,warm_start=x0 is not None or self._warm is not None,
            high_seconds=high_seconds,inner_seconds=inner_seconds,
            q_error_bound=rhs_norm*float(residual.norm())/(2*self.mu),
            warm_reset=warm_reset,converged=converged,failure_reason=None if converged else reason or 'not_converged',
            seconds=time.perf_counter()-start)
        self.records.append(record)
        if not converged: raise CGConvergenceError(record)
        self._warm = solution.detach().clone()
        self._solution_key,self._solution,self._solution_c = key,solution.detach().clone(),original_c
        self._rhs = rhs
        return solution

    def q(self, c):
        key = (id(c), c._version, self.G.version, self.blocks._version, self.mu)
        if self._q_key != key:
            solution = self.solve(c)
            self._q_value = -.5*float(self._rhs@solution)
            self._q_bound = self.records[-1]['q_error_bound']
            self._q_key, self._q_c = key, c
        return self._q_value

    def swapped(self, remove, add, *, native_ids, outgoing_ids, high_add=None):
        return CGSwap(self,add,native_ids,outgoing_ids,high_add)


class CGSwap:
    def __init__(self, base, add, native_ids, outgoing_ids, high_add=None):
        self.base, self.add = base, add
        self.high_add=add if high_add is None else high_add
        self.outgoing_ids=outgoing_ids
        G = base.G.swapped(native_ids,outgoing_ids,add,high_add)
        self.factor = OutputCG(G, base.blocks, base.mu,
                              max_iter=base.max_iter, tolerance=base.tolerance)
        self.factor._warm = base._warm

    def gamma(self, e, fu):
        eprime = e-self.base.metric(fu)
        jump = -float(e@fu)+.5*float(fu@self.base.metric(fu))
        value = -jump+self.base.q(e)-self.factor.q(eprime)
        self.error_bound = self.base._q_bound+self.factor._q_bound
        self.gain_record=dict(gain_method='live_q_difference_ablation',linear_solve_error_bound=self.error_bound,
            base_q_error_bound=self.base._q_bound,swapped_q_error_bound=self.factor._q_bound)
        return value

    def dead_gain(self,e):
        base=self.base
        x=base.solve(e)
        residual=e-base.metric(base.G.high.Jv(x))
        r0=base.records[-1]['residual_norm']
        extended=self.factor.solve(e)
        t=extended[self.outgoing_ids]
        s=self.high_add.T@residual
        trace=metric_trace(self.high_add,base.blocks)
        gain,self.error_bound,self.gain_record=schur_gain(s,t,trace,base.mu,r0,
            self.factor.records[-1]['residual_norm'])
        return gain

    def commit(self):
        self.base.G = self.factor.G
        self.base._warm = self.factor._warm
        self.base._solution_key = None
        self.base.solves += self.factor.solves
        self.base.records.extend(self.factor.records)


def metric_trace(columns,blocks):
    values=columns.to(blocks).reshape(*blocks.shape[:2],-1)
    total=blocks.new_zeros(())
    # Sample-block bmm avoids vmap's candidate x sample x class x class broadcast.
    for value,metric in zip(values.split(16),blocks.split(16)):
        total.add_((value*torch.bmm(metric,value)).sum())
    return float(total)


def schur_gain(s,t,trace,mu,base_residual,augmented_residual):
    """Residual bound for .5*s^T*t, with t the new-coordinate augmented solution."""
    a=math.sqrt(max(trace,0.)/mu)*base_residual
    b=augmented_residual/mu
    bound=.5*(a*float(t.norm())+(float(s.norm())+a)*b)
    gain=.5*float(s@t)
    return gain,bound,dict(gain_method='dead_schur',linear_solve_error_bound=bound,
        base_residual_norm=base_residual,augmented_residual_norm=augmented_residual,
        schur_rhs_error_bound=a,schur_solution_error_bound=b,
        schur_rhs_norm=float(s.norm()),schur_solution_norm=float(t.norm()),metric_trace=trace)


def solve_image(G, solution):
    return G.high.Jv(solution) if isinstance(G, FrozenGram) else G@solution


def solve_norm2(G, solution):
    return float(solution@solution) if isinstance(G, FrozenGram) else float(solution@(G@solution))


def gram(products, chunk=64, vjp_chunk=16, jvp_chunk=16):
    start = time.perf_counter()
    m = products._call({}).numel()
    ref = products.parameter_refs[0][1]
    G = torch.empty((m, m), dtype=torch.float64, device=ref.device)
    for lo in range(0, m, chunk):
        ids = torch.arange(lo, min(lo+chunk, m), device=ref.device)
        E = torch.nn.functional.one_hot(ids, m).to(ref)
        # Bound backward activation memory separately from the stored vector block.
        tangents = torch.vmap(products.JT, chunk_size=vjp_chunk)(E)
        G[:, lo:lo+len(ids)] = torch.vmap(products.Jv, chunk_size=jvp_chunk)(tangents).double().T
        del tangents
        if m >= 12800 and G.is_cuda: torch.cuda.synchronize(G.device)
    G = (G+G.T)*.5
    if G.is_cuda: torch.cuda.synchronize(G.device)
    return G, dict(gram_seconds=time.perf_counter()-start, m=m, gram_dtype=str(G.dtype),
                   parameter_dtype=str(ref.dtype), product_vector_dtype=str(ref.dtype), gram_chunk=chunk,
                   gram_vjp_chunk=vjp_chunk, gram_jvp_chunk=jvp_chunk)


def columns(action, indices, size, ref, chunk=16):
    result = []
    for lo in range(0, len(indices), chunk):
        ids = indices[lo:lo+chunk]
        E = torch.nn.functional.one_hot(ids, size).to(ref)
        result.append(torch.vmap(action)(E).double().T)
    return torch.cat(result, 1) if result else torch.empty((0, 0), device=ref.device, dtype=torch.float64)


class OutputSolve:
    def __init__(self, G, blocks, mu):
        self.G, self.mu = G, float(mu)
        blocks = blocks.double()
        self.blocks = blocks
        n, c, _ = blocks.shape
        eig, Q = torch.linalg.eigh(blocks)
        threshold = torch.finfo(torch.float64).eps*c*eig.abs().amax(-1, keepdim=True)
        keep = eig > threshold
        # Block factors omit the softmax null directions without projecting the Gram.
        self.L = torch.zeros((n*c, int(keep.sum())), dtype=G.dtype, device=G.device)
        offset = 0
        for i in range(n):
            k = int(keep[i].sum())
            self.L[i*c:(i+1)*c, offset:offset+k] = Q[i, :, keep[i]]*eig[i, keep[i]].sqrt()
            offset += k
        K = self.L.T @ G @ self.L
        K = (K+K.T)*.5
        K.diagonal().add_(self.mu)
        self.chol, info = torch.linalg.cholesky_ex(K)
        if bool(info.any()) or not bool(torch.isfinite(self.chol).all()):
            raise torch.linalg.LinAlgError('chol_fail')
        self.inverse = None
        self.solves, self._q_key = 0, None

    def metric(self, v):
        shape = v.shape
        return torch.einsum('nij,nj->ni', self.blocks, v.to(self.blocks).reshape(self.blocks.shape[:2])).reshape(shape)

    def A_solve(self, rhs):
        return self.inverse @ rhs if self.inverse is not None else torch.cholesky_solve(rhs, self.chol)

    def solve(self, c):
        self.solves += 1
        return (c-self.L @ self.A_solve((self.L.T @ (self.G @ c))[:, None])[:, 0])/self.mu

    def q(self, c):
        key = (id(c), c._version, self.G._version)
        if self._q_key != key:
            y=self.solve(c)
            self._q_value = -.5*float(c @ (self.G @ y))
            residual=c-self.metric(self.G@y)-self.mu*y
            self._q_bound=self.norm_bound()*float(c.norm())*float(residual.norm())/(2*self.mu)
            self._q_key, self._q_c = key, c
        return self._q_value

    def norm_bound(self):
        if getattr(self,'_norm_key',None)!=self.G._version:
            self._norm_bound=max(float(block.abs().sum(1).max()) for block in self.G.split(64))
            self._norm_key=self.G._version
        return self._norm_bound

    def swapped(self, remove, add):
        return WoodburySwap(self, remove, add)


class WoodburySwap:
    def __init__(self, base, remove, add):
        self.base, self.remove, self.add = base, remove, add
        W = torch.cat((remove, add), 1)
        self.U = base.L.T @ W
        self.AU = base.A_solve(self.U)
        signs = torch.cat((-torch.ones(remove.shape[1], device=W.device, dtype=W.dtype),
                            torch.ones(add.shape[1], device=W.device, dtype=W.dtype)))
        self.small = torch.diag(signs)+self.U.T @ self.AU

    def Gv(self, v):
        return self.base.G@v-self.remove@(self.remove.T@v)+self.add@(self.add.T@v)

    def A_solve(self, rhs):
        original = self.base.A_solve(rhs)
        return original-self.AU @ torch.linalg.solve(self.small, self.U.T @ original)

    def solve(self, c):
        self.base.solves += 1
        return (c-self.base.L @ self.A_solve((self.base.L.T @ self.Gv(c))[:, None])[:, 0])/self.base.mu

    def gamma(self, e, fu):
        eprime = e-self.base.metric(fu)
        jump = -float(e@fu)+.5*float(fu@self.base.metric(fu))
        y=self.solve(eprime)
        qprime = -.5*float(eprime @ self.Gv(y))
        qbase=self.base.q(e)
        residual=eprime-self.base.metric(self.Gv(y))-self.base.mu*y
        # Removing columns decreases the PSD Gram norm; added columns give this upper bound.
        norm=self.base.norm_bound()+float(self.add.square().sum())
        prime_bound=norm*float(eprime.norm())*float(residual.norm())/(2*self.base.mu)
        self.error_bound=self.base._q_bound+prime_bound
        self.gain_record=dict(gain_method='live_q_difference_ablation',linear_solve_error_bound=self.error_bound,
            base_q_error_bound=self.base._q_bound,swapped_q_error_bound=prime_bound)
        return -jump+qbase-qprime

    def dead_gain(self,e):
        base=self.base
        if self.remove.numel() and bool(self.remove.ne(0).any()):
            raise ValueError('dead Schur gain requires zero removed columns')
        y=base.solve(e)
        residual=e-base.metric(base.G@y)
        r0=math.sqrt(base.norm_bound())*float((residual-base.mu*y).norm())
        extended=self.solve(e)
        error=e-base.metric(self.Gv(extended))-base.mu*extended
        r1=math.sqrt(base.norm_bound()+float(self.add.square().sum()))*float(error.norm())
        s,t=self.add.T@residual,self.add.T@extended
        trace=metric_trace(self.add,base.blocks)
        gain,self.error_bound,self.gain_record=schur_gain(s,t,trace,base.mu,r0,r1)
        return gain

    def commit(self):
        base = self.base
        if base.inverse is None:
            base.inverse = torch.cholesky_inverse(base.chol)
            base.chol = None
        base.inverse -= self.AU @ torch.linalg.solve(self.small, self.AU.T)
        base.inverse = (base.inverse+base.inverse.T)*.5
        base.G.addmm_(self.remove, self.remove.T, beta=1., alpha=-1.)
        base.G.addmm_(self.add, self.add.T)
