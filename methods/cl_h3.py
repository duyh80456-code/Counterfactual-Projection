"""Independent H3 finite-candidate transactions and small-system Schur audits."""
from dataclasses import dataclass
import math
import time
import torch
from methods.cl_state import State,evaluate,predict,isolated


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
    count=sum(p.numel() for p in model.parameters());rows=[];winner=None
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
            if admissible and math.isfinite(objective) and objective<best:winner=candidate;best=objective
        state.restore()
        if winner is not None:winner.apply(model);winner.reset_optimizer(optimizer,model)
    except BaseException:
        state.restore();raise
    return dict(choice='identity' if winner is None else winner.name,trials=rows,
        criterion='minimum supplied surrogate' if surrogate is not None else 'minimum measured total window loss',
        search_scope='provided finite candidates only',dead_slot_preservation_claim=False,
        final_losses=evaluate(model,inputs,components))
