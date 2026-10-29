"""H1: replay-aware correction, independent of the historical V3 policy."""
from copy import deepcopy
from dataclasses import dataclass,asdict
from contextlib import contextmanager
from contextvars import ContextVar
import math
import time
import torch
import torch.nn.functional as F
from methods.cl_projection import ProjectionOperator,ConstraintSolver,Uncertified
from methods.cl_state import State,evaluate,predict,vector,subtract
from methods.vna_losses import CE,ICaRLPyCIL
from methods.vna_ggn import GGNConfig,GGNProducts
from methods.vna_v3 import training_products,V3Engine
from experiments.vna_resume import deadline_due

_compute_deadline=ContextVar('cl_diagnostic_compute_deadline',default=None)


class ComputeBudgetExhausted(RuntimeError):
    pass


@dataclass
class LossComponent:
    name: str
    evaluate: object
    def __call__(self,logits):return self.evaluate(logits)


class _ReplayMetric(ICaRLPyCIL):
    def loss(self,z,ctx):
        if 'loss_fn' in ctx:return ctx['loss_fn'](z,ctx)
        return super().loss(z,ctx)

    def metric_apply(self,z,v,ctx):
        result=CE.metric_apply(self,z,v,ctx)
        if 'ce_weights' in ctx:result=result*ctx['ce_weights'].to(result)[:,None]*len(z)
        known=ctx.get('known_classes',self.known_classes)
        if known:
            p=(z[:,:known].double()/self.T).softmax(-1);w=v.reshape_as(z)[:,:known].double()
            # The fixed float32 teacher weights can sum to slightly more/less than one.
            mass=(ctx['teacher_logits'][:,:known]/self.T).softmax(-1).double().sum(-1,keepdim=True)
            result[:,:known]+=mass*p*(w-(p*w).sum(-1,keepdim=True))/(len(z)*self.T**2)
        return result

    def metric_blocks(self,z,ctx):
        blocks=CE.metric_blocks(self,z,ctx);known=ctx.get('known_classes',self.known_classes)
        if 'ce_weights' in ctx:blocks=blocks*ctx['ce_weights'].to(blocks)[:,None,None]*len(z)
        if known:
            p=(z[:,:known].double()/self.T).softmax(-1)
            mass=(ctx['teacher_logits'][:,:known]/self.T).softmax(-1).double().sum(-1)
            blocks[:,:known,:known]+=(torch.diag_embed(p)-p[:,:,None]*p[:,None,:])*mass[:,None,None]/(len(z)*self.T**2)
        return blocks


class ReplayLoss:
    """Partition mean CE without changing its batch weights; KD is old-only."""
    def __init__(self,targets,replay,*,kind='er',known=0,teacher=None,reference='replay_ce_kd',temperature=2.,reduction='pooled'):
        if kind not in ('er','icarl') or reference not in ('replay_ce_kd','kd'):raise ValueError('unknown replay objective')
        if targets.shape!=replay.shape or targets.numel()==0:raise ValueError('nonempty aligned targets/replay mask required')
        if kind=='er' and reference=='kd':raise ValueError('ER has no KD reference')
        if known and kind=='icarl' and teacher is None:raise ValueError('old-class teacher logits required')
        if reduction not in ('pooled','separate_mean'):raise ValueError('explicit pooled or separate_mean reduction required')
        if not math.isfinite(temperature) or temperature<=0:raise ValueError('positive finite KD temperature required')
        self.targets,self.replay=targets,replay.bool()
        self.reduction=reduction
        self.kind,self.known,self.teacher,self.reference,self.temperature=kind,known,None if teacher is None else teacher.detach().clone(),reference,temperature
        self.spec=_ReplayMetric(T=temperature,known_classes=known) if kind=='icarl' else CE()
        self.ctx=dict(targets=targets,known_classes=known if kind=='icarl' else 0,teacher_logits=self.teacher)
        if reduction=='separate_mean':
            weights=torch.where(self.replay,1/self.replay.sum().double().clamp_min(1),1/(~self.replay).sum().double().clamp_min(1))
            self.ctx.update(ce_weights=weights,loss_fn=lambda z,ctx:self.new_loss(z)+self.old_loss(z))
            self.spec=_ReplayMetric(T=temperature,known_classes=known if kind=='icarl' else 0)
        self.incoming=LossComponent('all_ce' if reference=='kd' else 'incoming_ce',self.new_loss)
        self.reference_loss=LossComponent('old_kd' if reference=='kd' else ('replay_ce_plus_old_kd' if kind=='icarl' else 'replay_ce'),self.old_loss)
        self.components=(self.incoming,self.reference_loss)

    def ce(self,z,mask):
        denominator=len(z) if self.reduction=='pooled' else mask.sum().clamp_min(1)
        return F.cross_entropy(z,self.targets,reduction='none').masked_fill(~mask,0).sum()/denominator

    def kd(self,z):
        if self.kind!='icarl' or not self.known:return z.sum()*0
        soft=(self.teacher[:,:self.known]/self.temperature).softmax(-1).to(z)
        return -(soft*(z[:,:self.known]/self.temperature).log_softmax(-1)).sum()/len(z)

    def new_loss(self,z):
        return (self.ce(z,~self.replay)+self.ce(z,self.replay) if self.reduction=='separate_mean' else F.cross_entropy(z,self.targets)) if self.reference=='kd' else self.ce(z,~self.replay)

    def old_loss(self,z):
        return self.kd(z) if self.reference=='kd' else self.ce(z,self.replay)+self.kd(z)

    def total(self,z):return self.spec.loss(z,self.ctx)


def loss_gate(before,after,dtype,constrained=True):
    """Allow only roundoff-scale changes; significance is classified separately."""
    b=list(before) if constrained else [sum(before)]
    a=list(after) if constrained else [sum(after)]
    tolerances=[1e-12+8*torch.finfo(dtype).eps*max(1.,abs(v)) for v in b]
    finite=all(math.isfinite(v) for v in (*a,*b))
    return dict(passed=finite and all(x<=y+t for x,y,t in zip(a,b,tolerances)),
        finite=finite,tolerances=tolerances,changes=[x-y for x,y in zip(a,b)])


class CurrentTaskLoss:
    """H3 has no old-task replay: optimize current CE with one descent constraint."""
    def __init__(self,targets):
        self.targets=targets;self.spec=CE();self.ctx=dict(targets=targets)
        self.components=(LossComponent('current_task_ce',lambda z:F.cross_entropy(z,targets)),)
    def total(self,z):return self.spec.loss(z,self.ctx)


@dataclass(frozen=True)
class CorrectionConfig:
    kernel: str = 'tiny_V'
    candidate_strategy: str = 'original'  # opt-in: pareto_minimax for pure two-objective VNA
    gir_selection: str = 'original'  # opt-in: certified_plasticity for H3
    physical_update_norm: float | None = None
    physical_budget_policy: str = 'cap'
    constrained: bool = True
    mode: str = 'intervention'
    damping_relative: float = .01
    cg_tolerance: float = 1e-6
    cg_iterations: int = 200
    max_relative_solution_error: float = 0.1
    improvement_relative_tolerance: float = 1e-6
    preconditioner_probes: int = 16
    damping_retries: int = 4
    damping_multiplier: float = 10.0
    eval_acceptance: bool = True
    trace_probes: int = 8
    max_trials: int = 12
    seed: int = 0

    def __post_init__(self):
        if self.physical_budget_policy not in ('cap','matched'):raise ValueError('unknown physical budget policy')
        if self.physical_update_norm is not None and (not math.isfinite(self.physical_update_norm) or self.physical_update_norm<=0):
            raise ValueError('physical update norm must be positive finite or None')
        if self.kernel not in ('tiny_V','random','native','gn','gradient'):raise ValueError('unknown candidate kernel')
        if self.mode not in ('intervention','unified'):raise ValueError('unknown integration mode')
        if self.candidate_strategy not in ('original','pareto_minimax'):raise ValueError('unknown VNA candidate strategy')
        if self.gir_selection not in ('original','certified_plasticity'):raise ValueError('unknown GIR selection policy')
        if not math.isfinite(self.damping_relative) or self.damping_relative<=0 or self.max_trials<1 or self.trace_probes<1:
            raise ValueError('positive finite numerical controls required')
        if (not 0<self.cg_tolerance<1 or self.cg_iterations<1 or self.preconditioner_probes<0 or self.damping_retries<0
                or not math.isfinite(self.max_relative_solution_error) or self.max_relative_solution_error<=0
                or not math.isfinite(self.improvement_relative_tolerance) or self.improvement_relative_tolerance<0):
            raise ValueError('invalid CG / acceptance numerical controls')
        if not math.isfinite(self.damping_multiplier) or self.damping_multiplier<=1:
            raise ValueError('damping multiplier must exceed one')


@dataclass
class CandidateUpdate:
    direction: torch.Tensor
    gradients: torch.Tensor
    operator: ProjectionOperator
    diagnostics: dict


@dataclass
class AcceptanceCertificate:
    accepted: bool
    status: str
    before: tuple
    after: tuple
    step: float
    step_norm: float
    trials: list


class _ChunkedSelector(V3Engine):
    def _trace_extension(self,products):
        nout=products._call({}).numel();ne=sum(p.numel() for _,p in products.extension_refs)
        size=min(nout,ne);total=0.
        action=products.JET if nout<ne else products.JEv
        for index in range(size):
            ReplayCorrection.boundary()
            basis=products.inputs.new_zeros(size);basis[index]=1
            total+=float(action(basis).double().square().sum())
        return total




def pareto_minimax_kernel_target(new_z, old_z, new_target, old_target):
    """Two-objective, fixed-PSD-kernel MGDA in virtual feature space.

    z_i are H J A^{-1} g_i and targets are K_E z_i. Returns a kernel
    target for A^{-1} J^T H K_E z_mix. Exact arithmetic gives strictly
    positive individual gains when the feature-space hull avoids zero.
    This DOES NOT certify approximate PCG or physical acceptance.
    """
    a=float(new_z@new_target);b=float(old_z@old_target)
    cross_new_old=float(new_z@old_target);cross_old_new=float(old_z@new_target)
    c=(cross_new_old+cross_old_new)/2
    scale=max(abs(a),abs(b),abs(c),1e-300)
    # A noticeably non-PSD/asymmetric Gram matrix invalidates the proof.
    if (not all(math.isfinite(x) for x in (a,b,c))
            or a < -1e-8*scale or b < -1e-8*scale
            or abs(cross_new_old-cross_old_new)>1e-6*scale
            or c*c>max(a,0.)*max(b,0.)+1e-7*scale*scale):
        raise Uncertified('virtual kernel Gram matrix failed PSD/symmetry audit')
    a=max(a,0.);b=max(b,0.)
    denom=max(a+b-2*c,0.)
    weight=max(0.,min(1.,(b-c)/denom)) if denom>1e-14*scale else .5
    mix_z=weight*new_z+(1-weight)*old_z
    mix_target=weight*new_target+(1-weight)*old_target
    # Use the actual mixed inner product to avoid cancellation in the Gram formula.
    energy=float(mix_z@mix_target)
    if not math.isfinite(energy) or energy < -1e-8*scale:
        raise Uncertified('virtual minimax energy is not nonnegative')
    if energy <= 1e-14*scale:
        return torch.zeros_like(mix_target),dict(weight_new=weight,weight_old=1-weight,
            virtual_margin=0.,candidate_zero=True,gram=[[a,c],[c,b]])
    return mix_target/math.sqrt(energy),dict(weight_new=weight,weight_old=1-weight,
        virtual_margin=math.sqrt(energy),candidate_zero=False,gram=[[a,c],[c,b]])


class ReplayCorrection:
    def __init__(self,config=CorrectionConfig()):
        self.config=config;self.events=[];self.applications=0

    def state_dict(self):return dict(config=asdict(self.config),events=deepcopy(self.events),applications=self.applications)

    def load_state_dict(self,state):
        if state['config']!=asdict(self.config):raise ValueError('correction resume configuration mismatch')
        self.events=deepcopy(state['events']);self.applications=state['applications']

    @staticmethod
    def boundary():
        if deadline_due():raise SystemExit(75)
        limit=_compute_deadline.get()
        if limit is not None and time.perf_counter()>=limit:raise ComputeBudgetExhausted('diagnostic wall-time budget exhausted')

    def _operator(self,p,op,penalty=None,damping=None):
        cfg=GGNConfig(hutchinson_probes=self.config.trace_probes,seed=self.config.seed,damping_norm='params')
        mu=max(self.config.damping_relative*op.trace(cfg)/max(op.size,1),1e-12) if damping is None else float(damping)
        return ProjectionOperator(lambda x:p.Jv(x).double(),lambda x:p.JT(x).double(),op.H,mu,
            tolerance=self.config.cg_tolerance,max_iterations=self.config.cg_iterations,
            max_relative_solution_error=self.config.max_relative_solution_error,
            preconditioner_probes=self.config.preconditioner_probes,preconditioner_seed=self.config.seed,
            penalty=penalty or (lambda x:torch.zeros_like(x)),boundary=self.boundary)

    def propose(self,model,inputs,losses,sites,optimizer_candidate=None,penalty=None,*,damping=None):
        config=self.config;started=time.perf_counter();construction=0.;counts=dict(jvp=0,vjp=0)
        with training_products(model,[],{},inputs) as p:
            op=GGNProducts(p,losses.targets,losses.spec,losses.ctx)
            A=self._operator(p,op,penalty,damping)
            residuals=[torch.autograd.functional.jacobian(lambda z:component(z),op.logits).detach().flatten()
                       for component in losses.components]
            gradients=torch.stack([p.JT(e).double() for e in residuals]);g=gradients.sum(0)
            z=A.split(g);zR=A.split(gradients[1]) if len(gradients)>1 else torch.zeros_like(z);mu=A.mu;solves=A.solves
            counts['jvp']+=p.jvp_calls;counts['vjp']+=p.vjp_calls
        selector=_ChunkedSelector('vna',seed=config.seed);selector.applications=self.applications
        proposals={};kernel=config.kernel
        if kernel in ('tiny_V','random'):
            stamp=time.perf_counter()
            from methods import vna_v3
            original=vna_v3.training_products
            @contextmanager
            def counted(*args,**kwargs):
                self.boundary()
                with original(*args,**kwargs) as product:
                    try:yield product
                    finally:
                        counts['jvp']+=product.jvp_calls+product.extension_jvp_calls
                        counts['vjp']+=product.vjp_calls+product.extension_vjp_calls
            try:
                vna_v3.training_products=counted
                proposals=selector._select(model,sites,(inputs,losses.targets),dict(z=z))
                if kernel=='random':proposals=selector._randomize(model,sites,(inputs,losses.targets),proposals)
            finally:vna_v3.training_products=original
            construction=time.perf_counter()-stamp
        active=[site for site in sites if site.module_name in proposals]
        with training_products(model,active,proposals,inputs) as p:
            op=GGNProducts(p,losses.targets,losses.spec,losses.ctx)
            A=ProjectionOperator(lambda x:p.Jv(x).double(),lambda x:p.JT(x).double(),op.H,mu,
                tolerance=config.cg_tolerance,max_iterations=config.cg_iterations,solves=solves,
                max_relative_solution_error=config.max_relative_solution_error,
                preconditioner_probes=config.preconditioner_probes,preconditioner_seed=config.seed,
                penalty=penalty or (lambda x:torch.zeros_like(x)),boundary=self.boundary)
            kernel_target=None;cross=None;strategy_info=None;strategy_applied='original'
            if kernel=='gn':d0=A.solve(g)
            elif kernel=='gradient':d0=g.clone()
            elif kernel=='native' or active:
                action=p.K if kernel=='native' else p.KE
                kernel_target=action(z).double();reference_target=action(zR).double();new_z=z-zR;new_target=kernel_target-reference_target
                cross=torch.stack((torch.stack((new_z@new_target,new_z@reference_target)),
                    torch.stack((zR@new_target,zR@reference_target)))).double()
                final_target=kernel_target
                if (config.candidate_strategy=='pareto_minimax' and config.constrained
                        and optimizer_candidate is None and len(gradients)==2
                        and kernel in ('tiny_V','random') and active):
                    final_target,strategy_info=pareto_minimax_kernel_target(
                        new_z,zR,new_target,reference_target)
                    if strategy_info['candidate_zero']:
                        # Rank-deficient/opposed virtual features have no strict
                        # common descent direction; retain the original projected
                        # candidate instead of regressing to an automatic no-op.
                        final_target=kernel_target
                        strategy_applied='original_fallback_zero_virtual_margin'
                    else:strategy_applied='pareto_minimax'
                d0=A.solve(p.JT(op.H(final_target)).double())
            else:d0=torch.zeros_like(g)
            pure=d0.clone()
            if optimizer_candidate is not None:d0=d0+optimizer_candidate
            constraint_started=time.perf_counter()
            qp=ConstraintSolver(A,config.cg_tolerance*10).solve(d0,gradients) if config.constrained else None
            d=qp.direction if qp else d0
            curvature=float(d@A.curvature(d));gain=gradients@d
            geometry_curvature=float(d@A.penalty(d));ridge_curvature=mu*float(d@d)
            norms=gradients.norm(dim=1)
            opposed=len(gradients)==2 and bool((norms>0).all()) and float((gradients[0]/norms[0]+gradients[1]/norms[1]).norm())<=config.cg_tolerance
            diagnostic=dict(mu=mu,curvature_balance_scope='proposed direction at the pre-update linearization',geometry_curvature=geometry_curvature,ridge_curvature=ridge_curvature,
                geometry_to_ggn=geometry_curvature/curvature if curvature>0 else None,
                geometry_to_ridge=geometry_curvature/ridge_curvature if ridge_curvature>0 else None,
                geometry_fraction=geometry_curvature/(curvature+geometry_curvature+ridge_curvature) if curvature+geometry_curvature+ridge_curvature>0 else None,
                solves=deepcopy(A.solves),first_order_gains=gain.tolist(),
                gradient_norms=gradients.norm(dim=1).tolist(),gradient_interference=float(gradients[0]@gradients[1]) if len(gradients)>1 else None,
                pure_candidate_gain=float(g@pure),z_kernel_z=float(z@kernel_target) if kernel_target is not None else None,
                cross_interference=None if cross is None else cross.tolist(),
                cross_interference_scope='z_i.T K_candidate z_j at shared metric; approximate-solve diagnostic',
                quadratic_gain_at_unit_step=float(g@d)-curvature/2,curvature=curvature,
                constraint_seconds=time.perf_counter()-constraint_started,
                kkt=None if qp is None else dict(slack=qp.slack.tolist(),multipliers=qp.multipliers.tolist(),
                    stationarity=qp.stationarity,complementarity=qp.complementarity,status=qp.status,
                    geometry='opposing_gradients_no_strict_joint_descent' if opposed else 'zero_constraint' if bool((norms==0).any()) else 'single_current_task_descent' if len(gradients)==1 else 'two_halfspaces',
                    strict_progress=[bool(v>config.cg_tolerance*n*d.norm()) for v,n in zip(gain,norms)]),
                kernel=kernel,candidate_strategy_requested=config.candidate_strategy,
                candidate_strategy_applied=strategy_applied,
                candidate_strategy_diagnostics=strategy_info,
                incoming_directions_per_site=16,active_sites=len(active),
                candidate_generated=bool(active) if kernel in ('tiny_V','random') else True,
                candidate_reason='NO_CANDIDATE' if kernel in ('tiny_V','random') and not active else None,
                kernel_seconds=construction,projection_seconds=time.perf_counter()-started-construction,
                forward_mode='train_batch_statistics',random_trajectories=1 if kernel=='random' else 0,
                kernel_trace_normalization='exact_per_site_against_tiny_V' if kernel=='random' else None)
            diagnostic['selection_direct_backward_calls']=len(sites) if kernel in ('tiny_V','random') else 0
            counts['jvp']+=p.jvp_calls+p.extension_jvp_calls;counts['vjp']+=p.vjp_calls+p.extension_vjp_calls
            diagnostic.update(counts)
            # A is valid only inside this products context; retain the needed scalar diagnostics.
            return CandidateUpdate(d.detach(),gradients.detach(),None,diagnostic)

    def step(self,model,optimizer,inputs,losses,sites,step_size,*,penalty=None,damping=None,
             update_norm=None,compute_seconds=None,trial_guard=None):
        update_norm=self.config.physical_update_norm if update_norm is None else update_norm
        if self.config.physical_budget_policy=='matched' and update_norm is None:raise ValueError('matched physical budget requires an update norm')
        if not math.isfinite(step_size) or step_size<=0:raise ValueError('positive finite physical step required')
        for name,value in (('damping',damping),('update_norm',update_norm),('compute_seconds',compute_seconds)):
            if value is not None and (not math.isfinite(value) or value<=0):raise ValueError(f'{name} must be positive finite or None')
        if any(isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.p for m in model.modules()):
            raise ValueError('stochastic dropout is not certified by this BN pilot')
        device=next(model.parameters()).device
        if device.type=='cuda':torch.cuda.synchronize(device)
        started=time.perf_counter();state=State(model,optimizer);initial=vector(model);staged_optimizer=None
        row=dict(application=self.applications,mode=self.config.mode,kernel=self.config.kernel,
            constrained=self.config.constrained,loss_components=[c.name for c in losses.components],
            damping_source='shared_diagnostic_lock' if damping is not None else 'window_trace',
            update_norm_budget=update_norm,physical_budget_policy=self.config.physical_budget_policy,compute_budget_seconds=compute_seconds,
            acceptance='componentwise_nonincrease' if self.config.constrained else 'total_nonincrease')
        token=_compute_deadline.set(started+compute_seconds if compute_seconds is not None else _compute_deadline.get())
        try:
            self.boundary()
            eval_before=predict(model,inputs);eval_losses_before=evaluate(model,inputs,losses.components,'eval')
            base=None
            if self.config.mode=='unified':
                if any(group['lr']!=step_size for group in optimizer.param_groups):raise ValueError('unified mode requires a common optimizer LR')
                optimizer.step();base=(initial-vector(model))/step_size
                staged_optimizer=deepcopy(optimizer.state_dict());state.restore()
            # Retry the *whole* construction: z, zR, candidates and KKT solves
            # must all use the same damped A. Explicit damping is a locked protocol.
            retries=0 if damping is not None else self.config.damping_retries
            attempts=[];candidate_damping=damping
            for attempt in range(retries+1):
                if attempt:
                    self.boundary()
                    state.restore()
                try:
                    proposal=self.propose(model,inputs,losses,sites,base,penalty,damping=candidate_damping)
                    attempts.append(dict(attempt=attempt,mu=proposal.diagnostics.get('mu'),status='CERTIFIED',
                                         solves=deepcopy(proposal.diagnostics.get('solves',[]))))
                    break
                except Uncertified as exc:
                    diagnostics=exc.diagnostics if isinstance(exc.diagnostics,list) else []
                    last=diagnostics[-1] if diagnostics else {}
                    mu=last.get('mu',candidate_damping)
                    attempts.append(dict(attempt=attempt,mu=mu,status='UNCERTIFIED',reason=str(exc),
                                         solves=deepcopy(diagnostics),last_relative_residual=last.get('relative_residual'),
                                         last_relative_solution_error_bound=last.get('relative_solution_error_bound')))
                    row['damping_attempts']=attempts
                    if attempt==retries or not str(exc).startswith('linear solve failed') or mu is None:
                        raise
                    candidate_damping=mu*self.config.damping_multiplier
            row['damping_attempts']=attempts
            row.update(proposal.diagnostics)
            row['certified']=True
            row.setdefault('candidate_generated',bool(proposal.direction.norm()))
            row['eval_losses_before']=eval_losses_before
            before=evaluate(model,inputs,losses.components);trials=[];accepted=False;after=before;norm=0.;used=0.
            direction_norm=float(proposal.direction.norm())
            lr_limited_step=min(step_size,update_norm/direction_norm) if update_norm is not None and direction_norm>0 else step_size
            maximum_step=lr_limited_step
            if self.config.physical_budget_policy=='matched' and direction_norm>0:
                maximum_step=(update_norm/direction_norm)*(1-16*torch.finfo(next(model.parameters()).dtype).eps)
            row.update(maximum_step=maximum_step,proposed_direction_norm=direction_norm,
                first_order_gain_per_unit_norm=sum(row.get('first_order_gains',[0.]))/direction_norm if direction_norm else None)
            if self.config.physical_budget_policy=='matched':row['lr_limited_reference_step']=lr_limited_step
            for trial in range(self.config.max_trials):
                self.boundary();state.restore();s=maximum_step*(.5**trial)
                if trial and self.config.physical_budget_policy=='matched':
                    s=min(s,lr_limited_step*(.5**(trial-1)))
                subtract(model,proposal.direction,s)
                physical=(initial-vector(model))/s
                gains=proposal.gradients@physical
                scale=proposal.gradients.norm(dim=1)*physical.norm()
                feasible=not self.config.constrained or bool((gains>=-self.config.cg_tolerance*scale).all())
                after=evaluate(model,inputs,losses.components)
                finite=all(math.isfinite(x) for x in (*before,*after))
                train_gate=loss_gate(before,after,next(model.parameters()).dtype,self.config.constrained)
                nonincrease=train_gate['passed']
                # Improvement classification is separate from the numerical gate.
                if self.config.constrained:
                    thresholds=[self.config.improvement_relative_tolerance*max(1.,abs(b)) for b in before]
                    meaningful=any(b-a>threshold for a,b,threshold in zip(after,before,thresholds))
                else:
                    thresholds=[self.config.improvement_relative_tolerance*max(1.,abs(sum(before)))]
                    meaningful=sum(before)-sum(after)>thresholds[0]
                eval_after=evaluate(model,inputs,losses.components,'eval')
                eval_gate=loss_gate(eval_losses_before,eval_after,next(model.parameters()).dtype,self.config.constrained)
                eval_ok=eval_gate['passed']
                extra_gate=trial_guard(model) if trial_guard is not None else {'passed':True}
                if not isinstance(extra_gate,dict) or not isinstance(extra_gate.get('passed'),bool):
                    raise ValueError('trial_guard must return a dict with boolean passed')
                extra_ok=extra_gate['passed']
                physical_norm=float((initial-vector(model)).norm())
                in_budget=update_norm is None or physical_norm<=update_norm
                trials.append(dict(step=s,before=before,after=after,physical_gains=gains.tolist(),feasible=feasible,
                    physical_norm=physical_norm,in_update_budget=in_budget,
                    nominal_update_norm=s*direction_norm,
                    physical_budget_fill_ratio=physical_norm/update_norm if update_norm is not None else None,
                    meaningful_improvement=meaningful,improvement_thresholds=thresholds,
                    eval_after=eval_after,eval_nonincrease=eval_ok,train_gate=train_gate,eval_gate=eval_gate,
                    additional_gate=extra_gate,
                    failed_gates=[name for name,ok in (('finite',finite),('constraints',feasible),('train',nonincrease),('eval',eval_ok or not self.config.eval_acceptance),('budget',in_budget),('nonzero',physical_norm>0),('additional',extra_ok)) if not ok],
                    normalized_constraint_violation=float((-gains/scale.clamp_min(1e-30)).clamp_min(0).max())))
                if finite and feasible and nonincrease and in_budget and physical_norm>0 and extra_ok and (eval_ok or not self.config.eval_acceptance):
                    accepted=True;norm=physical_norm;used=s;break
            if not accepted:
                row['rejection_reason']=row.get('candidate_reason') or ('ZERO_DIRECTION' if direction_norm==0 else 'ZERO_PHYSICAL_UPDATE' if physical_norm==0 else 'CONSTRAINT_VIOLATION' if not feasible else 'TRAIN_LOSS_INCREASE' if not nonincrease else 'EVAL_LOSS_INCREASE' if self.config.eval_acceptance and not eval_ok else 'UPDATE_BUDGET' if not in_budget else 'ADDITIONAL_TRIAL_GATE' if not extra_ok else 'NONFINITE')
                state.restore()
            elif staged_optimizer is not None:optimizer.load_state_dict(staged_optimizer)
            candidate_after=after
            if not accepted:after=before
            # Keep status='ACCEPTED' for existing runners, but distinguish neutral updates in diagnostics.
            acceptance_detail=('ACCEPTED_WITH_IMPROVEMENT' if trials[-1]['meaningful_improvement'] else 'ACCEPTED_NEUTRAL') if accepted else 'REJECTED'
            certificate=AcceptanceCertificate(accepted,'ACCEPTED' if accepted else 'REJECTED',before,after,used,norm,trials)
            row.update(asdict(certificate),acceptance_detail=acceptance_detail,
                improvement_relative_tolerance=self.config.improvement_relative_tolerance,
                certificate_scope='window_parameter_intervention' if self.config.mode=='intervention' else 'window_entire_parameter_update',
                total_before=sum(before),total_candidate=sum(candidate_after),candidate_after=candidate_after,
                actual_loss_change=[a-b for a,b in zip(after,before)],
                constraint_violation=max(0.,-min(trials[-1]['physical_gains'])) if trials else 0.)
            row.update(first_order_predicted_gain=used*sum(row.get('first_order_gains',[0.])),
                quadratic_predicted_gain=used*sum(row.get('first_order_gains',[0.]))-.5*used**2*row.get('curvature',0.),
                prediction_scope='nominal direction at the pre-update linearization; physical gains separately logged')
            row.update(eval_losses_before=eval_losses_before,eval_losses_after=evaluate(model,inputs,losses.components,'eval'),
                eval_functional_drift=float((predict(model,inputs).double()-eval_before.double()).norm()),
                bn_buffer_drift=float(sum((b.double()-state.buffers[n].double()).square().sum() for n,b in model.named_buffers()).sqrt())
                    if any(True for _ in model.buffers()) else 0.)
            if initial.is_cuda:torch.cuda.synchronize(initial.device)
            self.boundary()
        except ComputeBudgetExhausted as exc:
            state.restore();row.update(accepted=False,status='SKIPPED_RESOURCE_BUDGET',reason=str(exc),step=0.,step_norm=0.,
                first_order_predicted_gain=0.,quadratic_predicted_gain=0.,constraint_violation=0.)
            if 'before' in row:row.update(after=row['before'],actual_loss_change=[0.]*len(row['before']))
            if 'eval_losses_before' in row:
                row.update(candidate_eval_losses_after=row.get('eval_losses_after'),eval_losses_after=row['eval_losses_before'],
                    eval_functional_drift=0.,bn_buffer_drift=0.)
        except Uncertified as exc:
            state.restore();row.update(certified=False,accepted=False,status='UNCERTIFIED',reason=str(exc),solver_failure=exc.diagnostics)
        except BaseException:
            state.restore();raise
        finally:
            self.applications+=1
            _compute_deadline.reset(token)
        if initial.is_cuda:torch.cuda.synchronize(initial.device)
        row.update(seconds=time.perf_counter()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(initial.device) if initial.is_cuda else 0,
            peak_vram_scope='process high-water mark; reset by experiment runner, not per candidate')
        row['cost_scope']='wall time includes all retries; jvp/vjp counters cover final successful construction'
        row['compute_overshoot_seconds']=max(0.,row['seconds']-compute_seconds) if compute_seconds is not None else None
        self.events.append(row)
        return row
