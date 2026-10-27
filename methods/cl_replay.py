"""H1: replay-aware correction, independent of the historical V3 policy."""
from copy import deepcopy
from dataclasses import dataclass,asdict
from contextlib import contextmanager
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


@dataclass
class LossComponent:
    name: str
    evaluate: object
    def __call__(self,logits):return self.evaluate(logits)


class _ReplayMetric(ICaRLPyCIL):
    def metric_apply(self,z,v,ctx):
        result=CE.metric_apply(self,z,v,ctx)
        known=ctx.get('known_classes',self.known_classes)
        if known:
            p=(z[:,:known].double()/self.T).softmax(-1);w=v.reshape_as(z)[:,:known].double()
            # The fixed float32 teacher weights can sum to slightly more/less than one.
            mass=(ctx['teacher_logits'][:,:known]/self.T).softmax(-1).double().sum(-1,keepdim=True)
            result[:,:known]+=mass*p*(w-(p*w).sum(-1,keepdim=True))/(len(z)*self.T**2)
        return result

    def metric_blocks(self,z,ctx):
        blocks=CE.metric_blocks(self,z,ctx);known=ctx.get('known_classes',self.known_classes)
        if known:
            p=(z[:,:known].double()/self.T).softmax(-1)
            mass=(ctx['teacher_logits'][:,:known]/self.T).softmax(-1).double().sum(-1)
            blocks[:,:known,:known]+=(torch.diag_embed(p)-p[:,:,None]*p[:,None,:])*mass[:,None,None]/(len(z)*self.T**2)
        return blocks


class ReplayLoss:
    """Partition mean CE without changing its batch weights; KD is old-only."""
    def __init__(self,targets,replay,*,kind='er',known=0,teacher=None,reference='replay_ce_kd',temperature=2.):
        if kind not in ('er','icarl') or reference not in ('replay_ce_kd','kd'):raise ValueError('unknown replay objective')
        if targets.shape!=replay.shape or targets.numel()==0:raise ValueError('nonempty aligned targets/replay mask required')
        if kind=='er' and reference=='kd':raise ValueError('ER has no KD reference')
        if known and kind=='icarl' and teacher is None:raise ValueError('old-class teacher logits required')
        self.targets,self.replay=targets,replay.bool()
        self.kind,self.known,self.teacher,self.reference,self.temperature=kind,known,teacher,reference,temperature
        self.spec=_ReplayMetric(T=temperature,known_classes=known) if kind=='icarl' else CE()
        self.ctx=dict(targets=targets,known_classes=known,teacher_logits=teacher)
        self.incoming=LossComponent('all_ce' if reference=='kd' else 'incoming_ce',self.new_loss)
        self.reference_loss=LossComponent('old_kd' if reference=='kd' else ('replay_ce_plus_old_kd' if kind=='icarl' else 'replay_ce'),self.old_loss)
        self.components=(self.incoming,self.reference_loss)

    def ce(self,z,mask):
        return F.cross_entropy(z,self.targets,reduction='none').masked_fill(~mask,0).sum()/len(z)

    def kd(self,z):
        if self.kind!='icarl' or not self.known:return z.sum()*0
        soft=(self.teacher[:,:self.known]/self.temperature).softmax(-1).to(z)
        return -(soft*(z[:,:self.known]/self.temperature).log_softmax(-1)).sum()/len(z)

    def new_loss(self,z):
        return F.cross_entropy(z,self.targets) if self.reference=='kd' else self.ce(z,~self.replay)

    def old_loss(self,z):
        return self.kd(z) if self.reference=='kd' else self.ce(z,self.replay)+self.kd(z)

    def total(self,z):return self.spec.loss(z,self.ctx)


@dataclass(frozen=True)
class CorrectionConfig:
    kernel: str = 'tiny_V'
    constrained: bool = True
    mode: str = 'intervention'
    damping_relative: float = .01
    cg_tolerance: float = 1e-6
    cg_iterations: int = 200
    trace_probes: int = 8
    max_trials: int = 12
    seed: int = 0

    def __post_init__(self):
        if self.kernel not in ('tiny_V','random','native','gn','gradient'):raise ValueError('unknown candidate kernel')
        if self.mode not in ('intervention','unified'):raise ValueError('unknown integration mode')
        if self.damping_relative<=0 or self.max_trials<1 or self.trace_probes<1:raise ValueError('positive numerical controls required')


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

    def _operator(self,p,op,penalty=None):
        cfg=GGNConfig(hutchinson_probes=self.config.trace_probes,seed=self.config.seed,damping_norm='params')
        mu=max(self.config.damping_relative*op.trace(cfg)/max(op.size,1),1e-12)
        return ProjectionOperator(lambda x:p.Jv(x).double(),lambda x:p.JT(x).double(),op.H,mu,
            tolerance=self.config.cg_tolerance,max_iterations=self.config.cg_iterations,
            penalty=penalty or (lambda x:torch.zeros_like(x)),boundary=self.boundary)

    def propose(self,model,inputs,losses,sites,optimizer_candidate=None,penalty=None):
        config=self.config;started=time.perf_counter();construction=0.;counts=dict(jvp=0,vjp=0)
        with training_products(model,[],{},inputs) as p:
            op=GGNProducts(p,losses.targets,losses.spec,losses.ctx)
            A=self._operator(p,op,penalty)
            residuals=[torch.autograd.functional.jacobian(lambda z:component(z),op.logits).detach().flatten()
                       for component in losses.components]
            gradients=torch.stack([p.JT(e).double() for e in residuals]);g=gradients.sum(0)
            z=A.split(g);zR=A.split(gradients[1]);mu=A.mu;solves=A.solves
            counts['jvp']+=p.jvp_calls;counts['vjp']+=p.vjp_calls
        selector=_ChunkedSelector('vna',seed=config.seed);selector.applications=self.applications
        proposals={};kernel=config.kernel
        if kernel in ('tiny_V','random'):
            stamp=time.perf_counter()
            from methods import vna_v3
            original=vna_v3.training_products
            @contextmanager
            def counted(*args,**kwargs):
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
                penalty=penalty or (lambda x:torch.zeros_like(x)),boundary=self.boundary)
            kernel_target=None;cross=None
            if kernel=='gn':d0=A.solve(g)
            elif kernel=='gradient':d0=g.clone()
            elif kernel=='native' or active:
                action=p.K if kernel=='native' else p.KE
                kernel_target=action(z).double();reference_target=action(zR).double();new_z=z-zR;new_target=kernel_target-reference_target
                cross=torch.stack((torch.stack((new_z@new_target,new_z@reference_target)),
                    torch.stack((zR@new_target,zR@reference_target)))).double()
                d0=A.solve(p.JT(op.H(kernel_target)).double())
            else:d0=torch.zeros_like(g)
            pure=d0.clone()
            if optimizer_candidate is not None:d0=d0+optimizer_candidate
            constraint_started=time.perf_counter()
            qp=ConstraintSolver(A,config.cg_tolerance*10).solve(d0,gradients) if config.constrained else None
            d=qp.direction if qp else d0
            curvature=float(d@A.curvature(d));gain=gradients@d
            diagnostic=dict(mu=mu,solves=deepcopy(A.solves),first_order_gains=gain.tolist(),
                gradient_norms=gradients.norm(dim=1).tolist(),gradient_interference=float(gradients[0]@gradients[1]),
                pure_candidate_gain=float(g@pure),z_kernel_z=float(z@kernel_target) if kernel_target is not None else None,
                cross_interference=None if cross is None else cross.tolist(),
                cross_interference_scope='z_i.T K_candidate z_j at shared metric; approximate-solve diagnostic',
                quadratic_gain_at_unit_step=float(g@d)-curvature/2,curvature=curvature,
                constraint_seconds=time.perf_counter()-constraint_started,
                kkt=None if qp is None else dict(slack=qp.slack.tolist(),multipliers=qp.multipliers.tolist(),
                    stationarity=qp.stationarity,complementarity=qp.complementarity,status=qp.status),
                kernel=kernel,incoming_directions_per_site=16,active_sites=len(active),
                kernel_seconds=construction,projection_seconds=time.perf_counter()-started-construction,
                forward_mode='train_batch_statistics',random_trajectories=1 if kernel=='random' else 0,
                kernel_trace_normalization='exact_per_site_against_tiny_V' if kernel=='random' else None)
            diagnostic['selection_direct_backward_calls']=len(sites) if kernel in ('tiny_V','random') else 0
            counts['jvp']+=p.jvp_calls+p.extension_jvp_calls;counts['vjp']+=p.vjp_calls+p.extension_vjp_calls
            diagnostic.update(counts)
            # A is valid only inside this products context; retain the needed scalar diagnostics.
            return CandidateUpdate(d.detach(),gradients.detach(),None,diagnostic)

    def step(self,model,optimizer,inputs,losses,sites,step_size,*,penalty=None):
        if not math.isfinite(step_size) or step_size<=0:raise ValueError('positive finite physical step required')
        if any(isinstance(m,torch.nn.modules.dropout._DropoutNd) and m.p for m in model.modules()):
            raise ValueError('stochastic dropout is not certified by this BN pilot')
        started=time.perf_counter();state=State(model,optimizer);initial=vector(model);staged_optimizer=None
        row=dict(application=self.applications,mode=self.config.mode,kernel=self.config.kernel,
            constrained=self.config.constrained,loss_components=[c.name for c in losses.components])
        try:
            eval_before=predict(model,inputs);eval_losses_before=evaluate(model,inputs,losses.components,'eval')
            base=None
            if self.config.mode=='unified':
                if any(group['lr']!=step_size for group in optimizer.param_groups):raise ValueError('unified mode requires a common optimizer LR')
                optimizer.step();base=(initial-vector(model))/step_size
                staged_optimizer=deepcopy(optimizer.state_dict());state.restore()
            proposal=self.propose(model,inputs,losses,sites,base,penalty)
            row.update(proposal.diagnostics)
            before=evaluate(model,inputs,losses.components);trials=[];accepted=False;after=before;norm=0.;used=0.
            for trial in range(self.config.max_trials):
                self.boundary();state.restore();s=step_size*(.5**trial)
                subtract(model,proposal.direction,s)
                physical=(initial-vector(model))/s
                gains=proposal.gradients@physical
                scale=proposal.gradients.norm(dim=1)*physical.norm()
                feasible=not self.config.constrained or bool((gains>=-self.config.cg_tolerance*scale).all())
                after=evaluate(model,inputs,losses.components)
                finite=all(math.isfinite(x) for x in (*before,*after))
                nonincrease=all(a<=b for a,b in zip(after,before)) if self.config.constrained else sum(after)<=sum(before)
                trials.append(dict(step=s,before=before,after=after,physical_gains=gains.tolist(),feasible=feasible))
                if finite and feasible and nonincrease:
                    accepted=True;norm=float((initial-vector(model)).norm());used=s;break
            if not accepted:state.restore()
            elif staged_optimizer is not None:optimizer.load_state_dict(staged_optimizer)
            candidate_after=after
            if not accepted:after=before
            certificate=AcceptanceCertificate(accepted,'ACCEPTED' if accepted else 'REJECTED',before,after,used,norm,trials)
            row.update(asdict(certificate),certificate_scope='window_parameter_intervention' if self.config.mode=='intervention' else 'window_entire_parameter_update',
                total_before=sum(before),total_candidate=sum(candidate_after),candidate_after=candidate_after,
                actual_loss_change=[a-b for a,b in zip(after,before)],
                constraint_violation=max(0.,-min(trials[-1]['physical_gains'])) if trials else 0.)
            row.update(first_order_predicted_gain=used*sum(row.get('first_order_gains',[0.])),
                quadratic_predicted_gain=used*sum(row.get('first_order_gains',[0.]))-.5*used**2*row.get('curvature',0.),
                prediction_scope='nominal direction at the pre-update linearization; physical gains separately logged')
            row.update(eval_losses_before=eval_losses_before,eval_losses_after=evaluate(model,inputs,losses.components,'eval'),
                eval_functional_drift=float((predict(model,inputs).double()-eval_before.double()).norm()),
                bn_buffer_drift=float(sum((b.double()-state.values[n].double()).square().sum() for n,b in model.named_buffers()).sqrt())
                    if any(True for _ in model.buffers()) else 0.)
        except Uncertified as exc:
            state.restore();row.update(accepted=False,status='UNCERTIFIED',reason=str(exc),solver_failure=exc.diagnostics)
        except BaseException:
            state.restore();raise
        finally:
            self.applications+=1
        if initial.is_cuda:torch.cuda.synchronize(initial.device)
        row.update(seconds=time.perf_counter()-started,peak_vram_bytes=torch.cuda.max_memory_allocated(initial.device) if initial.is_cuda else 0,
            peak_vram_scope='process high-water mark; reset by experiment runner, not per candidate')
        self.events.append(row)
        return row
