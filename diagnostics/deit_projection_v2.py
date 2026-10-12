"""Same frozen E, paired damping/scopes, independent heldout and nonlinear/margin diagnostics."""
import math
import time
import torch
import torch.nn.functional as F
from adapters.deit_cp_adapter import make_projector
from adapters.deit_mlp_growth import DeitMLPGrowthAdapter
from projection.margin_diagnostics import margin_transfer
from experiments.run_gromo_pilot import preview_projected_gain
from experiments.run_plateau_comparison import finite_projection
from diagnostics.deit_expansion_suite import private_rng


@torch.no_grad()
def preview_logits(model,projection,scale,inputs):
    parameters=dict(model.named_parameters());before={name:parameters[name].detach().clone() for name in projection.parameter_delta}
    try:
        projection.apply_(model,scale);return model(inputs).detach()
    finally:
        for name,value in before.items():parameters[name].copy_(value)


def analyze(model,candidate,batches,config,*,scope=None,seed=5):
    started=time.perf_counter();names=tuple(name for site in (scope or [candidate.site])
        for name in DeitMLPGrowthAdapter.original_mlp_parameters(model,site))
    fit=batches['projection_batch'];gate=batches['gate_batch'];held=batches['heldout']
    target=candidate.delta_logits(fit);gate_target=candidate.delta_logits(gate);held_target=candidate.delta_logits(held)
    projector=make_projector(config)
    with private_rng(seed):
        projection=projector.project(model,fit[0],target,block=candidate.site,parameter_names=names)
    if set(projection.parameter_delta)!=set(names):raise AssertionError('projection escaped declared scope')
    gains={str(scale):preview_projected_gain(model,projection,scale,gate) for scale in config.scales}
    finite=[(scale,gains[str(scale)]) for scale in config.scales if math.isfinite(gains[str(scale)])]
    scale,gate_gain=max(finite,key=lambda pair:pair[1]) if finite else (0.,None)
    accepted=bool(finite_projection(projection) and gate_gain is not None and gate_gain>0 and target.norm()>1e-12)
    evaluation=projector.evaluate_direction(model,held[0],held_target,projection.parameter_delta)
    base=model(held[0]).detach();expanded=base+held_target
    actual=preview_logits(model,projection,scale,held[0]);linear=scale*evaluation.fitted_delta
    margins=margin_transfer(base,expanded,linear,actual,held[1])
    gate_base=model(gate[0]).detach()
    gate_linear=projector.evaluate_direction(model,gate[0],gate_target,projection.parameter_delta).fitted_delta
    gate_actual=preview_logits(model,projection,scale,gate[0])
    gate_margins=margin_transfer(gate_base,gate_base+gate_target,scale*gate_linear,gate_actual,gate[1])
    where=batches['where_batches'];egains=[];emargins=[]
    from projection.margin_diagnostics import margins as class_margins
    with torch.no_grad():
        for batch in where:
            original=model(batch[0]);extended=original+candidate.delta_logits(batch)
            egains.append(float(F.cross_entropy(original.double(),batch[1])-F.cross_entropy(extended.double(),batch[1])))
            emargins.append(float((class_margins(extended,batch[1])-class_margins(original,batch[1])).mean()))
    record={'generator':candidate.generator,'selected_site':candidate.site,'rank':candidate.branch.A.shape[0],
        'e_gain':sum(egains)/len(egains),'e_margin_gain':sum(emargins)/len(emargins),'delta_f_norm':float(target.norm()),
        'projection_scope':list(scope or [candidate.site]),'damping_mode':config.damping_mode,'damping_used':projection.damping_used,
        'gram_trace_mean':projection.gram_trace_mean,'trace_probes':projection.trace_probes,
        'cg_converged':projection.cg.converged,'cg_relative_residual':projection.cg.relative_residual,
        'projection_relative_residual':projection.relative_residual,'projection_cosine':projection.cosine_alignment,
        'heldout_projection_residual':evaluation.relative_residual,'heldout_cosine':evaluation.cosine_alignment,
        'gate_margin_gain_after_projection':gate_margins['margin_gain_after_projection'],
        'margin_gain_after_projection':margins['margin_gain_after_projection'],
        'corrected_samples':margins['corrected_samples'],'damaged_samples':margins['damaged_samples'],
        'gate_gain':gate_gain,'correction_applied':False,'gate_accepted':accepted,
        'selected_scale':scale if accepted else None,'preview_scale':scale,'line_search_gains':gains,
        'search_time_seconds':candidate.budget['search_time_seconds'],'projection_time_seconds':time.perf_counter()-started,
        'generator_budget':candidate.budget,'heldout_margin_diagnostics':margins,'projection_parameter_names':list(names),
        'heldout_role':'independent_never_selects_generator_site_damping_or_scale',
        'heldout_samples':len(held[1]),
        'range_limit_known':False,'minimum_possible_residual':None,
        'interpretation':'CG residual does not establish O range limitation; dense SVD is only a small-problem reference'}
    return projection,record


def choose_projected_candidate(records,metric='gate_gain'):
    """Experimental gate-only selection; never read E gain or heldout/validation scores."""
    if metric not in ('gate_gain','gate_margin_gain_after_projection'):raise ValueError('unsupported gate selector')
    valid=[row for row in records if row['gate_accepted'] and len(row['projection_scope'])==1
        and row['gate_gain'] is not None and row['gate_gain']>0 and math.isfinite(row[metric])]
    return max(valid,key=lambda row:row[metric]) if valid else None
