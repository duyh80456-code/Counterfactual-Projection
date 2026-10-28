"""Independent H3 finite-candidate transactions and small-system Schur audits."""
from dataclasses import dataclass
import math
import time
import torch
from methods.cl_state import State,evaluate,predict,isolated,vector
from methods.cl_projection import Uncertified
from methods.cl_replay import ReplayCorrection,CorrectionConfig,_ChunkedSelector,loss_gate
from methods.vna_v3 import training_products,V3Engine
from methods.vna_ggn import GGNProducts


def vna_column_derivative(K,H,KE,e,v,mu,alpha=0.):
    """Valid also when H and K do not commute; audit matrices only."""
    current=K+alpha*torch.outer(v,v)
    system=H@current+mu*torch.eye(len(e),device=e.device,dtype=e.dtype)
    residual=mu*torch.linalg.solve(system,e);z=e-residual
    return 2*(v@residual)*(z@KE@torch.linalg.solve(system,H@v))


@dataclass
class SwapCandidate:
    name: str
    slot_kind: str
    apply: object
    reset_optimizer: object


def preliminary_score(site,proposal,unit,gradients,mu):
    """Negative first-order loss jump minus isotropic LM displacement cost."""
    first,second=site.first_layer.layer,site.second_layer.layer
    changes=[(first.weight,unit,proposal['A']['weight'][0]-first.weight[unit]),
             (second.weight,(slice(None),unit),-second.weight[:,unit])]
    if first.bias is not None:changes.append((first.bias,unit,proposal['A']['bias'][0]-first.bias[unit]))
    for path in site.between:
        module=site.model.get_submodule(path)
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm) and module.affine:
            changes.extend([(module.weight,unit,proposal['A'].get('bn_weight',module.weight.new_ones(1))[0]-module.weight[unit]),
                            (module.bias,unit,proposal['A'].get('bn_bias',module.bias.new_zeros(1))[0]-module.bias[unit])])
    linear=norm2=0.
    for parameter,index,delta in changes:
        delta=delta.detach().double()
        linear+=float((gradients[id(parameter)][index]*delta).sum());norm2+=float(delta.square().sum())
    return -linear-.5*mu*norm2


def reset_sgd_coordinates(optimizer,masks):
    if not isinstance(optimizer,torch.optim.SGD):raise ValueError('this reset contract is SGD-only')
    for parameter,mask in masks:
        if mask.shape!=parameter.shape or mask.dtype!=torch.bool:raise ValueError('aligned boolean coordinate mask required')
        momentum=optimizer.state.get(parameter,{}).get('momentum_buffer')
        if momentum is not None:momentum.masked_fill_(mask,0)


def choose_swap(model,optimizer,inputs,components,candidates,*,surrogate=None,eval_acceptance=True):
    """Minimize a specified surrogate over feasible swaps plus identity."""
    state=State(model,optimizer);before=evaluate(model,inputs,components)
    eval_before=evaluate(model,inputs,components,'eval');eval_output=predict(model,inputs)
    initial=vector(model)
    output_before=predict(model,inputs,'train')
    shapes={n:tuple(p.shape) for n,p in model.named_parameters()}
    identities={n:id(p) for n,p in model.named_parameters()}
    count=sum(p.numel() for p in model.parameters());rows=[];winner=None;winner_state=None
    def score(losses):
        if surrogate is None:return sum(losses)
        with isolated(model,optimizer):return float(surrogate(model))
    best=score(before)
    if not math.isfinite(best):raise ValueError('finite identity objective required')
    try:
        for candidate in candidates:
            started=time.perf_counter()
            if candidate.slot_kind not in ('live','dead'):raise ValueError('slot kind must be measured explicitly')
            try:
                state.restore();candidate.apply(model)
                if (shapes!={n:tuple(p.shape) for n,p in model.named_parameters()} or count!=sum(p.numel() for p in model.parameters())
                        or identities!={n:id(p) for n,p in model.named_parameters()}):
                    raise ValueError('H3 requires fixed capacity and parameter structure')
                candidate.reset_optimizer(optimizer,model)
                after=evaluate(model,inputs,components)
                train_gate=loss_gate(before,after,next(model.parameters()).dtype)
                admissible=train_gate['passed']
                eval_after=evaluate(model,inputs,components,'eval')
                eval_gate=loss_gate(eval_before,eval_after,next(model.parameters()).dtype)
                eval_ok=eval_gate['passed']
                changed=float((vector(model)-initial).norm())
                admissible=admissible and changed>0 and (eval_ok or not eval_acceptance)
                objective=score(after)
                jump=predict(model,inputs,'train')-output_before
                rows.append(dict(candidate=candidate.name,slot_kind=candidate.slot_kind,before=before,after=after,
                    functional_loss_jump=[a-b for a,b in zip(after,before)],admissible=admissible,
                    objective=objective,output_jump_norm=float(jump.double().norm()),
                    window_output_preserved=not bool(jump.any()),step_norm=changed,
                    eval_before=eval_before,eval_after=eval_after,eval_nonincrease=eval_ok,train_gate=train_gate,eval_gate=eval_gate,
                    failed_gates=[name for name,ok in (('train',train_gate['passed']),('eval',eval_ok or not eval_acceptance),('nonzero',changed>0)) if not ok],
                    rejection_reason='TRAIN_LOSS_INCREASE' if not train_gate['passed'] else 'EVAL_LOSS_INCREASE' if eval_acceptance and not eval_ok else 'ZERO_PHYSICAL_UPDATE' if changed==0 else None,
                    eval_functional_drift=float((predict(model,inputs)-eval_output).double().norm())))
                rows[-1]['seconds']=time.perf_counter()-started
                if admissible and math.isfinite(objective) and objective<best:
                    winner=candidate;best=objective;winner_state=State(model,optimizer)
            except Uncertified as exc:
                state.restore()
                rows.append(dict(candidate=candidate.name,slot_kind=candidate.slot_kind,
                    admissible=False,status='UNCERTIFIED',solver_failure=exc.diagnostics,
                    rejection_reason='CANDIDATE_SOLVER_FAILURE',seconds=time.perf_counter()-started))
                continue
        state.restore()
        # Commit the tested state; a callback may consume external RNG or have side effects.
        if winner_state is not None:winner_state.restore()
    except Uncertified as exc:
        exc.candidate_trials=rows;exc.failed_candidate=candidate.name
        state.restore();raise
    except BaseException:
        state.restore();raise
    return dict(choice='identity' if winner is None else winner.name,trials=rows,
        candidate_solver_failures=sum('solver_failure' in r for r in rows),
        certified_candidates=sum('solver_failure' not in r for r in rows),
        criterion='minimum supplied surrogate' if surrogate is not None else 'minimum measured total window loss',
        search_scope='provided finite candidates only',dead_slot_preservation_claim=False,
        final_losses=evaluate(model,inputs,components))


class TangentReconfiguration:
    """Serial tiny_V candidates, fixed-capacity swaps, net GN recovery objective."""
    def __init__(self,config=CorrectionConfig(),*,max_candidates=None):
        if max_candidates is not None and (not isinstance(max_candidates,int) or max_candidates<1):raise ValueError('positive candidate limit required')
        self.config=config;self.events=[];self.applications=0;self.max_candidates=max_candidates

    def state_dict(self):
        from dataclasses import asdict
        from copy import deepcopy
        result=dict(config=asdict(self.config),events=deepcopy(self.events),applications=self.applications)
        if self.max_candidates is not None:result['max_candidates']=self.max_candidates
        return result

    def load_state_dict(self,state):
        from dataclasses import asdict
        from copy import deepcopy
        if state['config']!=asdict(self.config):raise ValueError('H3 resume configuration mismatch')
        if state.get('max_candidates')!=self.max_candidates:raise ValueError('H3 candidate search resume mismatch')
        self.events=deepcopy(state['events']);self.applications=state['applications']

    def step(self,model,optimizer,inputs,losses,sites):
        from copy import deepcopy
        started=time.perf_counter();snapshot=State(model,optimizer);attempts=[];damping=None
        for attempt in range(self.config.damping_retries+1):
            snapshot.restore();ReplayCorrection.boundary()
            result=self._attempt(model,optimizer,inputs,losses,sites,damping=damping)
            attempts.append(dict(attempt=attempt,status=result['status'],mu=result.get('mu'),
                solves=deepcopy(result.get('solves',result.get('solver_failure',[]))),reason=result.get('reason')))
            if result['status']!='UNCERTIFIED':break
            failure=result.get('solver_failure') or []
            mu=failure[-1].get('mu') if isinstance(failure,list) and failure else result.get('mu')
            if mu is None or not result.get('reason','').startswith('linear solve failed'):break
            damping=mu*self.config.damping_multiplier
        result.update(damping_attempts=attempts,application=self.applications,seconds=time.perf_counter()-started)
        result['certified']=result['status']!='UNCERTIFIED'
        result['candidate_generated']=bool(result.get('generated_candidates') or result.get('trials'))
        if result['status']=='ACCEPTED':
            trial=next(r for r in result['trials'] if r['candidate']==result['choice'])
            result['step_norm']=trial['step_norm']
            improved=any(b-a>self.config.improvement_relative_tolerance*max(1.,abs(b)) for a,b in zip(trial['after'],trial['before']))
            result['acceptance_detail']='ACCEPTED_WITH_IMPROVEMENT' if improved else 'ACCEPTED_NEUTRAL'
        else:result['rejection_reason']=result.get('reason','ALL_CANDIDATES_UNCERTIFIED' if result.get('trials') and all('solver_failure' in r for r in result['trials']) else 'NO_FEASIBLE_IMPROVING_SWAP' if result['candidate_generated'] else 'NO_CANDIDATE')
        self.applications+=1;self.events.append(result)
        return result

    def _attempt(self,model,optimizer,inputs,losses,sites,*,damping=None):
        started=time.perf_counter();snapshot=State(model,optimizer)
        helper=ReplayCorrection(self.config);counts=dict(jvp=0,vjp=0);solves=[];scores=[]
        ranking=[];candidates=[];generated=0;objective_calls=0
        try:
            with training_products(model,[],{},inputs) as p:
                op=GGNProducts(p,losses.targets,losses.spec,losses.ctx)
                A=helper._operator(p,op,damping=damping);mu=A.mu;z=A.split(op.gradient)
                if self.max_candidates is not None:
                    refs=[parameter for _,parameter in p.parameter_refs]
                    parts=op.gradient.split([parameter.numel() for parameter in refs])
                    gradients={id(parameter):part.reshape_as(parameter) for parameter,part in zip(
                        (v for v in model.parameters() if v.requires_grad),parts)}
                solves.extend(A.solves);counts['jvp']+=p.jvp_calls;counts['vjp']+=p.vjp_calls
            sites=[site for site in sites if V3Engine.valid_slot(site)]
            selector=_ChunkedSelector('vna',seed=self.config.seed)
            from contextlib import contextmanager
            from methods import vna_v3
            original=vna_v3.training_products;generation_started=time.perf_counter()
            @contextmanager
            def counted(*args,**kwargs):
                helper.boundary()
                with original(*args,**kwargs) as p:
                    try:yield p
                    finally:
                        counts['jvp']+=p.jvp_calls+p.extension_jvp_calls
                        counts['vjp']+=p.vjp_calls+p.extension_vjp_calls
            try:
                vna_v3.training_products=counted
                proposals=selector._select(model,sites,(inputs,losses.targets),dict(z=z))
            finally:vna_v3.training_products=original
            generation_seconds=time.perf_counter()-generation_started
            ranking_started=time.perf_counter()
            dead=V3Engine.dead_slots(model,sites,(inputs,losses.targets));candidates=[];ranking=[]
            for site in sites:
                if site.module_name not in proposals:continue
                entries=proposals[site.module_name]
                for row in range(len(entries['weight'])):
                    proposal={'A':{k:None if v is None else v[row:row+1].detach().clone() for k,v in entries.items()}}
                    for unit in range(site.first_layer.layer.weight.shape[0]):
                        def install(net,site=site,proposal=proposal,unit=unit):
                            # install_gir zeroes outgoing weights and affected SGD/BN state atomically.
                            V3Engine('vna',seed=self.config.seed).install_gir(net,site,[proposal],[unit],optimizer)
                        name=f'{site.module_name}:{unit}:{row}'
                        candidates.append(SwapCandidate(name,'dead' if unit in dead.get(site.module_name,[]) else 'live',install,lambda *args:None))
                        if self.max_candidates is not None:
                            score=preliminary_score(site,proposal,unit,gradients,mu)
                            ranking.append(dict(candidate=name,score=score if math.isfinite(score) else None))
            generated=len(candidates)
            if self.max_candidates is not None:
                ranked=sorted(zip(candidates,ranking),key=lambda pair:(-(pair[1]['score'] if pair[1]['score'] is not None else -math.inf),pair[0].name))
                candidates=[candidate for candidate,item in ranked if item['score'] is not None][:self.max_candidates]
                selected={candidate.name for candidate in candidates}
                ranking=[dict(item,selected=candidate.name in selected,
                    reason='shortlisted' if candidate.name in selected else 'nonfinite_surrogate' if item['score'] is None else 'outside_shortlist') for candidate,item in ranked]
            ranking_seconds=time.perf_counter()-ranking_started
            active=[site for site in sites if site.module_name in proposals]
            def objective(net):
                nonlocal objective_calls
                objective_calls+=1
                helper.boundary()
                with training_products(net,active,proposals,inputs) as p:
                    op=GGNProducts(p,losses.targets,losses.spec,losses.ctx)
                    A=helper._operator(p,op,damping=mu)
                    inverse=A.solve(op.gradient);recovery=float(op.gradient@inverse)/2
                    split=op.H(p.Jv(inverse));virtual_gain=float(split@p.KE(split).double()) if active else 0.
                    loss=sum(evaluate(net,inputs,losses.components))
                    scores.append(dict(loss=loss,predicted_gn_recovery=recovery,vna_gain=virtual_gain,
                                       net_surrogate=loss-recovery))
                    solves.extend(A.solves);counts['jvp']+=p.jvp_calls+p.extension_jvp_calls
                    counts['vjp']+=p.vjp_calls+p.extension_vjp_calls
                    return loss-recovery
            result=choose_swap(model,optimizer,inputs,losses.components,candidates,surrogate=objective,eval_acceptance=self.config.eval_acceptance)
            for row,score in zip((r for r in result['trials'] if 'solver_failure' not in r),scores[1:]):row.update(score)
            if self.max_candidates is not None:
                for trial in result['trials']:
                    trial['selected_winner']=trial['candidate']==result['choice']
                    trial['rejection_reason']=trial.get('rejection_reason') or (None if trial['selected_winner'] else 'component_loss_increase' if not trial['admissible'] else 'not_better_than_identity_or_winner')
                result.update(generated_candidates=generated,shortlist_count=len(candidates),exact_candidate_evaluations=len(result['trials']),
                    preliminary_ranking_seconds=ranking_seconds,
                    identity_evaluations=1,max_candidates=self.max_candidates,ranking=ranking,
                    ranking_surrogate='-g^T delta_theta_reset - mu/2 ||delta_theta_reset||^2; parameter-only first-order LM proxy',
                    search_scope='best feasible candidate among deterministic shortlist plus identity; no exhaustive/global guarantee')
            result.update(status='ACCEPTED' if result['choice']!='identity' else 'IDENTITY',identity=scores[0],generated_candidates=generated,
                actual_replacements=int(result['choice']!='identity'),mu=mu,solves=solves,**counts,
                candidate_generation_seconds=generation_seconds,candidate_evaluation_seconds=sum(r['seconds'] for r in result['trials']),
                selection_direct_backward_calls=len(sites),
                swap_kind=next((r['slot_kind'] for r in result['trials'] if r['candidate']==result['choice']),None),
                criterion='minimum post-removal loss minus damped GN recovery; componentwise loss gates with dtype-scaled numerical tolerance',
                gain_scope='net fixed-capacity surrogate; NOT added-column Schur theorem',
                vna_gain_role='diagnostic only; GN improvement does not imply VNA improvement',
                candidate_generation='existing tiny_V generalized eigenvectors with real-ReLU sign selection; every slot')
        except Uncertified as exc:
            snapshot.restore();result=dict(status='UNCERTIFIED',reason=str(exc),solver_failure=exc.diagnostics,actual_replacements=0)
            if self.max_candidates is not None:
                result.update(generated_candidates=generated,shortlist_count=len(candidates),max_candidates=self.max_candidates,
                    ranking=ranking,exact_candidate_evaluations=max(0,objective_calls-1),identity_evaluations=min(1,objective_calls),
                    trials=getattr(exc,'candidate_trials',[]),failed_candidate=getattr(exc,'failed_candidate','identity'),
                    search_scope='bounded shortlist; solver failure rejected the intervention')
        except BaseException:
            snapshot.restore();raise
        result.update(application=self.applications,seconds=time.perf_counter()-started,
            peak_vram_bytes=torch.cuda.max_memory_allocated(inputs.device) if inputs.is_cuda else 0)
        return result
