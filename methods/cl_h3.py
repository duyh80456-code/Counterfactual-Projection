"""Independent H3 finite-candidate transactions and small-system Schur audits."""
from dataclasses import dataclass
import math
import time
import torch
from methods.cl_state import State,evaluate,predict,isolated
from methods.cl_projection import Uncertified
from methods.cl_replay import ReplayCorrection,CorrectionConfig,_ChunkedSelector
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


def reset_sgd_coordinates(optimizer,masks):
    if not isinstance(optimizer,torch.optim.SGD):raise ValueError('this reset contract is SGD-only')
    for parameter,mask in masks:
        if mask.shape!=parameter.shape or mask.dtype!=torch.bool:raise ValueError('aligned boolean coordinate mask required')
        momentum=optimizer.state.get(parameter,{}).get('momentum_buffer')
        if momentum is not None:momentum.masked_fill_(mask,0)


def choose_swap(model,optimizer,inputs,components,candidates,*,surrogate=None):
    """Minimize a specified surrogate over feasible swaps plus identity."""
    state=State(model,optimizer);before=evaluate(model,inputs,components)
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
            state.restore();candidate.apply(model)
            if (shapes!={n:tuple(p.shape) for n,p in model.named_parameters()} or count!=sum(p.numel() for p in model.parameters())
                    or identities!={n:id(p) for n,p in model.named_parameters()}):
                raise ValueError('H3 requires fixed capacity and parameter structure')
            candidate.reset_optimizer(optimizer,model)
            after=evaluate(model,inputs,components)
            admissible=all(math.isfinite(a) and a<=b for a,b in zip(after,before))
            objective=score(after)
            jump=predict(model,inputs,'train')-output_before
            rows.append(dict(candidate=candidate.name,slot_kind=candidate.slot_kind,before=before,after=after,
                functional_loss_jump=[a-b for a,b in zip(after,before)],admissible=admissible,
                objective=objective,output_jump_norm=float(jump.double().norm()),
                window_output_preserved=not bool(jump.any())))
            rows[-1]['seconds']=time.perf_counter()-started
            if admissible and math.isfinite(objective) and objective<best:
                winner=candidate;best=objective;winner_state=State(model,optimizer)
        state.restore()
        # Commit the tested state; a callback may consume external RNG or have side effects.
        if winner_state is not None:winner_state.restore()
    except BaseException:
        state.restore();raise
    return dict(choice='identity' if winner is None else winner.name,trials=rows,
        criterion='minimum supplied surrogate' if surrogate is not None else 'minimum measured total window loss',
        search_scope='provided finite candidates only',dead_slot_preservation_claim=False,
        final_losses=evaluate(model,inputs,components))


class TangentReconfiguration:
    """Serial tiny_V candidates, fixed-capacity swaps, net GN recovery objective."""
    def __init__(self,config=CorrectionConfig()):
        self.config=config;self.events=[];self.applications=0

    def state_dict(self):
        from dataclasses import asdict
        from copy import deepcopy
        return dict(config=asdict(self.config),events=deepcopy(self.events),applications=self.applications)

    def load_state_dict(self,state):
        from dataclasses import asdict
        from copy import deepcopy
        if state['config']!=asdict(self.config):raise ValueError('H3 resume configuration mismatch')
        self.events=deepcopy(state['events']);self.applications=state['applications']

    def step(self,model,optimizer,inputs,losses,sites):
        started=time.perf_counter();snapshot=State(model,optimizer)
        helper=ReplayCorrection(self.config);counts=dict(jvp=0,vjp=0);solves=[];scores=[]
        try:
            with training_products(model,[],{},inputs) as p:
                op=GGNProducts(p,losses.targets,losses.spec,losses.ctx)
                A=helper._operator(p,op);mu=A.mu;z=A.split(op.gradient)
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
            dead=V3Engine.dead_slots(model,sites,(inputs,losses.targets));candidates=[]
            for site in sites:
                if site.module_name not in proposals:continue
                entries=proposals[site.module_name]
                for row in range(len(entries['weight'])):
                    proposal={'A':{k:None if v is None else v[row:row+1].detach().clone() for k,v in entries.items()}}
                    for unit in range(site.first_layer.layer.weight.shape[0]):
                        def install(net,site=site,proposal=proposal,unit=unit):
                            # install_gir zeroes outgoing weights and affected SGD/BN state atomically.
                            V3Engine('vna',seed=self.config.seed).install_gir(net,site,[proposal],[unit],optimizer)
                        candidates.append(SwapCandidate(f'{site.module_name}:{unit}:{row}',
                            'dead' if unit in dead.get(site.module_name,[]) else 'live',install,lambda *args:None))
            active=[site for site in sites if site.module_name in proposals]
            def objective(net):
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
            result=choose_swap(model,optimizer,inputs,losses.components,candidates,surrogate=objective)
            for row,score in zip(result['trials'],scores[1:]):row.update(score)
            result.update(status='ACCEPTED' if result['choice']!='identity' else 'IDENTITY',identity=scores[0],
                actual_replacements=int(result['choice']!='identity'),mu=mu,solves=solves,**counts,
                candidate_generation_seconds=generation_seconds,candidate_evaluation_seconds=sum(r['seconds'] for r in result['trials']),
                selection_direct_backward_calls=len(sites),
                swap_kind=next((r['slot_kind'] for r in result['trials'] if r['candidate']==result['choice']),None),
                criterion='minimum post-removal loss minus damped GN recovery; componentwise finite-loss nonincrease',
                gain_scope='net fixed-capacity surrogate; NOT added-column Schur theorem',
                vna_gain_role='diagnostic only; GN improvement does not imply VNA improvement',
                candidate_generation='existing tiny_V generalized eigenvectors with real-ReLU sign selection; every slot')
        except Uncertified as exc:
            snapshot.restore();result=dict(status='UNCERTIFIED',reason=str(exc),solver_failure=exc.diagnostics,actual_replacements=0)
        except BaseException:
            snapshot.restore();raise
        result.update(application=self.applications,seconds=time.perf_counter()-started,
            peak_vram_bytes=torch.cuda.max_memory_allocated(inputs.device) if inputs.is_cuda else 0)
        self.applications+=1;self.events.append(result)
        return result
