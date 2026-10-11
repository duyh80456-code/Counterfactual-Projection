"""Temporary DeiT neuron proposals; original weights, modes, gradients and RNG survive."""
from contextlib import contextmanager
from dataclasses import replace
import math
import torch
import torch.nn.functional as F
from adapters.deit_mlp_growth import MLPExpansionCandidate, DeitMLPGrowthAdapter
from experiments.shared_protocol import rng_state, restore_rng


@contextmanager
def frozen_original(model):
    modes={module:module.training for module in model.modules()}
    flags={p:p.requires_grad for p in model.parameters()}
    gradients={p:p.grad for p in model.parameters()}
    state=rng_state()
    try:
        model.eval()
        for parameter in flags:parameter.requires_grad_(False)
        yield
    finally:
        for parameter,flag in flags.items():
            parameter.requires_grad_(flag);parameter.grad=gradients[parameter]
        for module,mode in modes.items():module.training=mode
        restore_rng(state)


def auxiliary_initialization(model, site, rank, seed, *, zero_B=True):
    mlp=DeitMLPGrowthAdapter.resolve_site(model,site);weight=mlp.fc1.weight
    generator=torch.Generator(device=weight.device).manual_seed(seed)
    A=torch.randn(rank,mlp.fc1.in_features,device=weight.device,dtype=weight.dtype,generator=generator)/math.sqrt(mlp.fc1.in_features)
    a=weight.new_zeros(rank)
    B=(weight.new_zeros(mlp.fc2.out_features,rank) if zero_B else
       torch.randn(mlp.fc2.out_features,rank,device=weight.device,dtype=weight.dtype,generator=generator)/math.sqrt(rank))
    return MLPExpansionCandidate(site,model,A,a,B,0.,{'source':'auxiliary_gaussian','seed':seed,'requested_rank':rank,'effective_rank':rank})


def gradient_tuned(model, site, tuning_batch, *, rank=4, steps=20, learning_rate=1e-3, seed=0):
    if not 1<=steps<=20 or not learning_rate>0:raise ValueError('auxiliary tuning needs1..20 steps and positive LR')
    initial=auxiliary_initialization(model,site,rank,seed)
    candidate=replace(initial,**{name:getattr(initial,name).clone().requires_grad_() for name in ('A','a','B')})
    optimizer=torch.optim.Adam([candidate.A,candidate.a,candidate.B],lr=learning_rate)
    losses=[]
    with frozen_original(model):
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with candidate.virtual_direction(1.):
                loss=F.cross_entropy(model(tuning_batch[0]).float(),tuning_batch[1])
            if not torch.isfinite(loss):raise ValueError('nonfinite auxiliary tuning loss')
            losses.append(float(loss.detach()));loss.backward();optimizer.step()
        with torch.no_grad(),candidate.virtual_direction(1.):
            final=float(F.cross_entropy(model(tuning_batch[0]).float(),tuning_batch[1]))
    detached=replace(candidate,**{name:getattr(candidate,name).detach().clone() for name in ('A','a','B')})
    return detached,{'source':'aux_gradient_tuned','seed':seed,'tuning_steps':steps,'tuning_learning_rate':learning_rate,
        'tuning_losses':losses+[final],'tuning_gain':losses[0]-final,
        'proposal_forward_calls':steps+1,'proposal_backward_calls':steps,
        'initialization':'nonzero_A_zero_B_zero_bias','tuning_role':'opt_fit_only_not_WHERE_projection_gate_or_validation',
        'evaluation_gate':1.,'compute_matched_to_TINY':False}


@torch.no_grad()
def random_structural(model, site, projection_inputs, *, norm_target, rank=4, gate=.05, seed=0, iterations=24, tolerance=1e-3):
    """Match a real auxiliary branch's whole-logit norm on projection inputs only."""
    if norm_target<0 or not math.isfinite(norm_target):raise ValueError('invalid random norm target')
    candidate=auxiliary_initialization(model,site,rank,seed,zero_B=False)
    base=model(projection_inputs).detach();calls=1
    def norm(scale):
        nonlocal calls
        with candidate.virtual_direction(gate*scale):value=model(projection_inputs)-base
        calls+=1
        return float(value.norm())
    if norm_target<=1e-12:
        coefficient=0.;actual=0.;converged=True
    else:
        lo=0.;hi=1.;value=norm(hi)
        for _ in range(12):
            if value>=norm_target:break
            hi*=2.;value=norm(hi)
        coefficient=hi;actual=value
        if value>=norm_target:
            for _ in range(iterations):
                coefficient=(lo+hi)/2.;actual=norm(coefficient)
                if abs(actual-norm_target)<=tolerance*norm_target:break
                if actual<norm_target:lo=coefficient
                else:hi=coefficient
        converged=abs(actual-norm_target)<=tolerance*norm_target
    result=replace(candidate,B=candidate.B*coefficient)
    return result,{'source':'aux_random_direction','seed':seed,'norm_target':norm_target,'norm_actual':actual,
        'norm_match_converged':converged,'norm_match_relative_error':abs(actual-norm_target)/max(norm_target,1e-12),
        'normalization_role':'match_TINY_projection_logits_norm_using_projection_inputs_only',
        'normalization_coefficient':coefficient,'proposal_forward_calls':calls,'proposal_backward_calls':0,
        'degenerate_random_control':norm_target<=1e-12,'evaluation_gate':gate}
