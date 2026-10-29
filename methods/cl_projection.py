"""Generalized VNA and a two-halfspace metric projection; no dense parameter inverse."""
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Protocol
import math
import time
import torch
from projection.cg import conjugate_gradient


class LinearOperator(Protocol):
    def __call__(self, vector: torch.Tensor) -> torch.Tensor: ...


class VirtualKernel(LinearOperator, Protocol):
    """A fixed PSD output operator for one linearization."""


class Uncertified(RuntimeError):
    def __init__(self,message,diagnostics=None):
        super().__init__(message);self.diagnostics=diagnostics


@dataclass
class ProjectionOperator:
    jvp: LinearOperator
    vjp: LinearOperator
    metric: LinearOperator
    mu: float
    regularizer: LinearOperator = lambda x: x
    regularizer_floor: float = 1.
    penalty: LinearOperator = lambda x: torch.zeros_like(x)
    tolerance: float = 1e-6
    max_iterations: int = 200
    max_relative_solution_error: float = 0.1
    preconditioner_probes: int = 0
    preconditioner_seed: int = 0
    boundary: Callable = lambda: None
    solves: list = field(default_factory=list)

    def __post_init__(self):
        if not math.isfinite(self.mu) or self.mu<=0 or not math.isfinite(self.regularizer_floor) or self.regularizer_floor<=0:
            raise ValueError('require mu>0 and a valid positive lower bound for R')
        if (not 0<self.tolerance<1 or self.max_iterations<1 or self.preconditioner_probes<0
                or not math.isfinite(self.max_relative_solution_error) or self.max_relative_solution_error<=0):
            raise ValueError('invalid solver tolerance/iterations/preconditioner probes/error bound')
        self._preconditioner = None
        self.audit=None;self.preconditioner_quality={}

    def curvature(self,x):return self.vjp(self.metric(self.jvp(x))).double()

    def __call__(self,x):
        return self.curvature(x)+self.mu*self.regularizer(x)+self.penalty(x)

    def _audit(self,rhs):
        if self.audit is not None:
            if not self.audit['passed']:raise Uncertified('linear solve failed operator audit',list(self.solves))
            return
        generator=torch.Generator(device=rhs.device).manual_seed(self.preconditioner_seed+917)
        u=torch.randn(rhs.shape,generator=generator,device=rhs.device,dtype=rhs.dtype)
        v=torch.randn(rhs.shape,generator=generator,device=rhs.device,dtype=rhs.dtype)
        self.boundary();au,av=self(u),self(v)
        ju=self.jvp(u).double();y=torch.randn(ju.shape,generator=generator,device=ju.device,dtype=ju.dtype)
        jty=self.vjp(y).double()
        adjoint=float(abs(ju.flatten()@y.flatten()-u.flatten()@jty.flatten()))/max(float(ju.norm()*y.norm()),float(u.norm()*jty.norm()),1e-30)
        symmetry=float(abs(u@av-v@au))/max(float(u.norm()*av.norm()),float(v.norm()*au.norm()),1e-30)
        lower=self.mu*self.regularizer_floor
        margins=[float(w@a)-lower*float(w@w) for w,a in ((u,au),(v,av))]
        psd=all(m>=-1e-4*max(float(w.norm()*a.norm()),1e-30) for m,w,a in zip(margins,(u,v),(au,av)))
        self.audit=dict(adjoint_relative_error=adjoint,symmetry_relative_error=symmetry,
            lower_bound_margins=margins,tolerance=1e-4,
            sampled_rayleigh=[float(w@a)/max(float(w@w),1e-30) for w,a in ((u,au),(v,av))],
            scope='seeded finite-precision probes; not a global SPD proof')
        self.audit['passed']=bool(all(torch.isfinite(a).all() for a in (au,av,jty)) and psd and adjoint<=1e-4 and symmetry<=1e-4)
        if not self.audit['passed']:
            self.solves.append(dict(mu=self.mu,status='UNCERTIFIED',operator_audit=self.audit))
            raise Uncertified('linear solve failed operator audit',list(self.solves))

    def _jacobi(self, rhs):
        """Deterministic positive Jacobi estimate; never use an indefinite preconditioner."""
        if self.preconditioner_probes == 0:
            return None
        if self._preconditioner is None:
            floor = self.mu * self.regularizer_floor
            n = rhs.numel()
            if n <= 16:
                # Tiny systems can use an exact diagonal (also an audit reference).
                diag = torch.stack([self(torch.nn.functional.one_hot(
                    torch.tensor(i, device=rhs.device), n).to(rhs.dtype))[i]
                    for i in range(n)])
            else:
                generator = torch.Generator(device=rhs.device)
                generator.manual_seed(self.preconditioner_seed)
                diag = torch.zeros_like(rhs);squares=torch.zeros_like(rhs)
                for _ in range(self.preconditioner_probes):
                    self.boundary()
                    v = torch.empty_like(rhs).bernoulli_(.5, generator=generator).mul_(2).sub_(1)
                    sample=v*self(v);diag.add_(sample);squares.add_(sample.square())
                used=self.preconditioner_probes
                if float((diag/used<floor).double().mean())>.1:
                    for _ in range(min(3*used,48)):
                        self.boundary()
                        v=torch.empty_like(rhs).bernoulli_(.5,generator=generator).mul_(2).sub_(1)
                        sample=v*self(v);diag.add_(sample);squares.add_(sample.square())
                        used+=1
                diag.div_(used)
                self.preconditioner_quality['raw_clipped_fraction']=float((diag<floor).double().mean())
                variance=(squares/used-diag.square()).clamp_min(0)/used
                signal=(diag-floor).clamp_min(0)
                confidence=torch.where(variance==0,torch.ones_like(signal),
                    signal.square()/(signal.square()+variance+torch.finfo(rhs.dtype).tiny))
                # Shrink noisy coordinate estimates toward a positive trace-scale prior.
                diag=floor+confidence*signal+(1-confidence)*signal.mean()
                self.preconditioner_quality.update(shrinkage_fraction=float((1-confidence).mean()),
                    mean_standard_error=float(variance.sqrt().mean()))
            if not torch.isfinite(diag).all():
                raise Uncertified('non-finite Jacobi diagonal')
            # Rademacher diagonal estimates may be negative; the ridge stays SPD.
            self.preconditioner_quality.update(estimator='exact_diagonal' if n<=16 else 'variance_shrunk_hutchinson',clipped_fraction=float((diag<floor).double().mean()),
                minimum=float(diag.min()),maximum=float(diag.max()),floor=floor,actual_probes=used if n>16 else 0)
            self._preconditioner = diag.clamp_min(floor).detach()
        return lambda residual: residual / self._preconditioner

    def solve(self,rhs):
        started=time.perf_counter();rhs=rhs.detach().double()
        self.boundary()
        if not bool(torch.isfinite(rhs).all()):
            raise Uncertified('non-finite linear solve RHS',list(self.solves))
        self._audit(rhs)
        if not bool(rhs.any()):
            # The zero RHS has the exact zero solution for the audited SPD operator.
            self.solves.append(dict(iterations=0,refinements=[],residual=0.,relative_residual=0.,
                error_bound=0.,relative_solution_error_bound=0.,
                max_relative_solution_error=self.max_relative_solution_error,status='CERTIFIED',
                operator_audit=self.audit,preconditioner_quality=self.preconditioner_quality,
                mu=self.mu,preconditioner='none',preconditioner_probes=0,breakdown=False,
                seconds=time.perf_counter()-started))
            return torch.zeros_like(rhs)
        preconditioner=self._jacobi(rhs)
        result=conjugate_gradient(self,rhs,max_iter=self.max_iterations,tolerance=self.tolerance,
            preconditioner=preconditioner,eps=1e-30,callback=lambda _:self.boundary())
        solution=result.solution.detach();residual_vector=rhs-self(solution)
        residual=float(residual_vector.norm());refinements=[];iterations=result.iterations
        if preconditioner is not None:
            for _ in range(2):
                target=min(self.tolerance*float(rhs.norm()),
                    self.max_relative_solution_error*self.mu*self.regularizer_floor*float(solution.norm()))
                if residual<=target:break
                self.boundary()
                correction=conjugate_gradient(self,residual_vector,max_iter=self.max_iterations,
                    tolerance=min(self.tolerance,1e-8),preconditioner=preconditioner,eps=1e-30,
                    callback=lambda _:self.boundary())
                candidate=solution+correction.solution
                candidate_residual=rhs-self(candidate);new_residual=float(candidate_residual.norm())
                refinements.append(dict(before=residual,after=new_residual,iterations=correction.iterations))
                iterations+=correction.iterations
                if not math.isfinite(new_residual) or new_residual>=residual:break
                solution=candidate.detach();residual_vector=candidate_residual;residual=new_residual
        relative=residual/max(float(rhs.norm()),torch.finfo(rhs.dtype).tiny)
        # For SPD A with lambda_min(A)>=mu*R_floor, ||x-x*||<=||b-Ax||/(mu*R_floor).
        error_bound=residual/(self.mu*self.regularizer_floor)
        solution_norm=float(solution.norm())
        relative_error_bound=(error_bound/solution_norm if solution_norm>0 else
                              0. if error_bound==0 else math.inf)
        residual_ok=bool(torch.isfinite(solution).all() and math.isfinite(relative) and relative<=self.tolerance)
        error_ok=bool(math.isfinite(relative_error_bound) and relative_error_bound<=self.max_relative_solution_error)
        certified=residual_ok and error_ok
        self.solves.append(dict(iterations=iterations,refinements=refinements,residual=residual,relative_residual=relative,
            error_bound=error_bound,relative_solution_error_bound=relative_error_bound,
            max_relative_solution_error=self.max_relative_solution_error,
            status='CERTIFIED' if certified else 'UNCERTIFIED',
            operator_audit=self.audit,preconditioner_quality=self.preconditioner_quality,mu=self.mu,preconditioner='jacobi' if preconditioner is not None else 'none',
            preconditioner_probes=self.preconditioner_probes,breakdown=result.breakdown,
            seconds=time.perf_counter()-started))
        if not certified:
            reason='true-residual certificate' if not residual_ok else 'solution-error bound'
            raise Uncertified('linear solve failed '+reason,list(self.solves))
        return solution.detach()

    def split(self,gradient):return self.metric(self.jvp(self.solve(gradient))).double()

    def project(self,gradient,kernel):
        z=self.split(gradient);d=self.solve(self.vjp(self.metric(kernel(z))))
        return d,z


@dataclass
class ConstraintResult:
    direction: torch.Tensor
    multipliers: torch.Tensor
    slack: torch.Tensor
    stationarity: float
    complementarity: float
    objective: float
    status: str


class ConstraintSolver:
    """Enumerate the four dual active sets; dependent/zero gradients are admissible."""
    def __init__(self,operator,tolerance=1e-6):
        self.operator,self.tolerance=operator,tolerance

    def solve(self,candidate,gradients):
        n=len(gradients)
        if n not in (1,2):raise ValueError('one or two halfspaces required; no trust-ball constraint')
        d0=candidate.detach().double()
        C=(gradients if isinstance(gradients,torch.Tensor) else torch.stack(tuple(gradients))).detach().double()
        if C.shape!=(n,d0.numel()) or not torch.isfinite(C).all() or not torch.isfinite(d0).all():
            raise ValueError('finite two-row gradients and aligned candidate required')
        # Scale before taking L2 norms: finite gradients may have overflowing or
        # underflowing squared norms, falsely turning a constraint into a zero row.
        maxabs=C.abs().amax(dim=1)
        safe_maxabs=torch.where(maxabs>0,maxabs,torch.ones_like(maxabs))
        scaled=C/safe_maxabs[:,None]
        norms=scaled.norm(dim=1)
        safe_norms=torch.where(norms>0,norms,torch.ones_like(norms))
        U=scaled/safe_norms[:,None]
        if hasattr(self.operator,'boundary'):self.operator.boundary()
        q=U@d0
        # Only take the zero-solve shortcut with a positive roundoff margin.
        # Zero constraints and the zero candidate are exactly feasible.
        guard=(8*d0.numel()*torch.finfo(d0.dtype).eps)*(U.abs()@d0.abs())
        zero_rows=maxabs==0
        safely_feasible=(not bool(d0.any())) or bool((torch.isfinite(q)&((q>guard)|zero_rows)).all())
        if safely_feasible:
            # Exactly feasible d0 is the unique metric projection: no constraint solves.
            if hasattr(self.operator,'_audit'):self.operator._audit(d0)
            return ConstraintResult(d0,torch.zeros(n,dtype=d0.dtype,device=d0.device),C@d0,0.,0.,0.,'CERTIFIED')
        inverse=torch.stack([self.operator.solve(row) for row in U],1)
        W=U@inverse
        if not torch.isfinite(W).all() or float((W-W.T).norm())>self.tolerance*max(float(W.norm()),1e-30):
            raise Uncertified('non-symmetric approximate dual operator')
        W=(W+W.T)/2;solutions=[]
        for size in range(n+1):
            for active in combinations(range(n),size):
                lam=torch.zeros(n,dtype=d0.dtype,device=d0.device)
                if active:
                    indices=list(active);block=W[indices][:,indices]
                    lam[indices]=torch.linalg.pinv(block,hermitian=True)@(-q[indices])
                d=d0+inverse@lam;slack=U@d
                scale=max(float(d.norm()),float(d0.norm()),1e-30)
                tolerance=self.tolerance*scale
                if not torch.isfinite(d).all() or float(lam.min()) < -self.tolerance*max(float(lam.norm()),1e-30):continue
                if float(slack.min()) < -tolerance:continue
                complement=float((lam*slack).abs().max())
                if complement>self.tolerance*max(float(lam.norm())*scale,1e-30):continue
                delta=d-d0;stationarity=float((self.operator(delta)-U.T@lam).norm())
                if stationarity>self.tolerance*max(float((U.T@lam).norm()),float(self.operator(d0).norm()),1e-30):continue
                objective=float(delta@self.operator(delta))/2
                multipliers=(lam/safe_norms)/safe_maxabs
                if not torch.isfinite(multipliers).all():continue
                solutions.append(ConstraintResult(d,multipliers,C@d,stationarity,complement,objective,'CERTIFIED'))
        if not solutions:raise Uncertified('no active set satisfies primal/dual/stationarity/complementarity checks')
        return min(solutions,key=lambda result:result.objective)


def schur_gain(J,Jv,H,e,mu):
    """Dense small-system verification of added columns, not a live-swap claim."""
    A=J.T@H@J+mu*torch.eye(J.shape[1],device=J.device,dtype=J.dtype)
    cross=J.T@H@Jv
    residual=e-H@J@torch.linalg.solve(A,J.T@e)
    s=Jv.T@residual
    S=Jv.T@H@Jv+mu*torch.eye(Jv.shape[1],device=J.device,dtype=J.dtype)-cross.T@torch.linalg.solve(A,cross)
    return .5*s@torch.linalg.solve(S,s)
