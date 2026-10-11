"""E0..E4 diagnostics only: same frozen fork, independent probes, no committed O update."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import json
import math
import time
import torch
import torch.nn.functional as F
from adapters.deit_mlp_growth import MLPExpansionCandidate, DeitMLPGrowthAdapter
from adapters.deit_ablation import project_target
from adapters.deit_projection_aware import tensor_hash, projection_aware_search
from experiments.run_plateau_comparison import select_by_expansion_gain
from experiments.shared_protocol import rng_state, restore_rng, seed_everything, atomic_json_save, atomic_torch_save
from experiments.run_deit_projection_aware import write_csv
from diagnostics.deit_expansion_proposals import gradient_tuned, random_structural

@dataclass(frozen=True)
class ExpansionConfig:
    ranks: tuple = (2,4,8,16)
    probe_epsilons: tuple = (.01,.05,.1,.2)
    expansion_scales: tuple = (.25,.5,1.,2.)
    auxiliary_steps: int = 20
    auxiliary_learning_rate: float = .001
    all_blocks: bool = False
    verify_forward_fp64: bool = True
    strength_mode: str = 'exact_expanded_forward'
    proposal_seed: int = 2

    def __post_init__(self):
        if (not self.ranks or any(r<1 for r in self.ranks) or not self.probe_epsilons
                or any(not 0<e<=1 for e in self.probe_epsilons) or not self.expansion_scales
                or any(not math.isfinite(g) or g<=0 for g in self.expansion_scales)
                or not 1<=self.auxiliary_steps<=20 or not math.isfinite(self.auxiliary_learning_rate)
                or self.auxiliary_learning_rate<=0 or self.strength_mode not in ('exact_expanded_forward','linear_logits')):
            raise ValueError('invalid expansion diagnostics configuration')


@contextmanager
def private_rng(seed=None):
    before=rng_state()
    try:
        if seed is not None:seed_everything(seed)
        yield
    finally:restore_rng(before)


def case_seed(config, *parts):
    data=json.dumps([config.proposal_seed,*parts]).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4],'big') % (2**31-1)


def gain(base, expanded, labels, dtype):
    return float(F.cross_entropy(base.to(dtype),labels)-F.cross_entropy(expanded.to(dtype),labels))


@torch.no_grad()
def direction(model,candidate,batch,epsilon,scale,mode):
    base=model(batch[0]).detach()
    with candidate.virtual_direction(epsilon):reference=model(batch[0]).detach()
    linear=base+scale*(reference-base)
    if mode=='linear_logits':expanded=linear
    elif scale==1.:expanded=reference
    else:
        with candidate.virtual_direction(epsilon*scale):expanded=model(batch[0]).detach()
    return {'base':base,'expanded':expanded,'delta':expanded-base,
        'e_gain_fp32':gain(base,expanded,batch[1],torch.float32),
        'e_gain_fp64':gain(base,expanded,batch[1],torch.float64),
        'linear_logits_gain_fp64':gain(base,linear,batch[1],torch.float64),
        'delta_f_norm':float((expanded-base).double().norm()),
        'native_delta_f_norm':float((expanded-base).norm())}


def select_blocks(diagnostics, *, all_blocks=False):
    ranked=sorted(diagnostics,key=lambda row:row['tiny_rank'])
    if all_blocks:return [row['block_name'] for row in ranked]
    valid=[row for row in ranked if row.get('projection_residual') is not None and math.isfinite(row['projection_residual'])]
    if not valid:raise ValueError('D0 has no finite projection residual; cannot choose the declared diagnostic blocks')
    picks=[ranked[0]['block_name'],min(valid,key=lambda row:row['projection_residual'])['block_name'],ranked[len(ranked)//2]['block_name']]
    # Keep three distinct sites when the declared top1/residual/median sites overlap.
    for row in ranked:
        if len(set(picks))>=min(3,len(ranked)):break
        picks.append(row['block_name'])
    return list(dict.fromkeys(picks))[:min(3,len(ranked))]


def verify_baseline(where, d0, *, atol=1e-7, rtol=1e-4):
    old={row['block_name']:row for row in d0['diagnostics']}
    new=where['site_evaluations']
    if set(old)!=set(new):raise ValueError('D0 and reproduced TINY sites differ')
    errors=[]
    for site,row in new.items():
        expected=old[site]
        if row['rank']!=expected['tiny_rank']:errors.append(site+':rank')
        for key,field in [('mean_e_gain','e_gain'),('mean_functional_delta_norm','functional_delta_norm')]:
            if not math.isclose(row[key],expected[field],abs_tol=atol,rel_tol=rtol):errors.append(site+':'+field)
    if errors:raise ValueError('TINY baseline does not reproduce D0: '+', '.join(errors))
    return {'matched':True,'scope':'all_sites_TINY_WHERE_gain_norm_rank','absolute_tolerance':atol,
        'relative_tolerance':rtol,'projection_rng_note':'new transfer solves use recorded private seeds; old D0 solver path is not claimed identical'}


def classify(e_gain,residual,gate_gain,accepted):
    # Descriptive bins, never used to change solver, scale grid or acceptance.
    if e_gain<=1e-8:return 'no_meaningful_positive_E_gain_measured'
    if gate_gain is not None and gate_gain>0 and accepted:return 'valid_E_to_O_correction_found'
    if residual is None:return 'transfer_unavailable'
    return 'poor_transfer_to_O' if residual>.5 else 'projection_fit_but_gate_rejected'


def serialize_candidate(candidate):
    return {name:getattr(candidate,name).detach().cpu() for name in ('A','a','B')} | {
        'site':candidate.module_name,'proposal_score':candidate.proposal_score,'payload':candidate.payload}


def deserialize_candidate(saved,model):
    reference=next(model.parameters())
    return MLPExpansionCandidate(saved['site'],model,**{key:saved[key].to(reference) for key in ('A','a','B')},
        proposal_score=saved['proposal_score'],payload=saved['payload'])


def run_suite(model,batches,cp,config,output,*,fork_hash,protocol,probe_indices,d0=None,expected_sites=12,optimizer=None):
    output.mkdir(parents=True,exist_ok=True)
    modes={module:module.training for module in model.modules()};model.eval()
    original=tensor_hash(model.state_dict());parameter_ids={name:id(p) for name,p in model.named_parameters()}
    original_optimizer=copy.deepcopy(optimizer.state_dict()) if optimizer else None
    source_rng=rng_state()
    try:
        return _run(model,batches,cp,config,output,fork_hash,protocol,probe_indices,d0,expected_sites,original)
    finally:
        restore_rng(source_rng)
        for module,mode in modes.items():module.training=mode
        if tensor_hash(model.state_dict())!=original or parameter_ids!={name:id(p) for name,p in model.named_parameters()}:
            raise AssertionError('diagnostics changed original model or Parameter objects')
        if optimizer is not None:
            if not tree_equal(original_optimizer,optimizer.state_dict()):raise AssertionError('diagnostics changed original optimizer')


def tree_equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def validate_indices(indices):
    groups={**{name:indices[name] for name in ('statistics','projection','gate','opt_fit','opt_val')},
        **{f'where_{i}':group for i,group in enumerate(indices['where'])}}
    seen=set()
    for name,group in groups.items():
        if not group or len(group)!=len(set(group)) or seen.intersection(group):
            raise ValueError(f'probe roles overlap or contain duplicate/empty IDs: {name}')
        seen.update(group)


def _run(model,batches,cp,config,output,fork_hash,protocol,indices,d0,expected_sites,original):
    validate_indices(indices)
    reference_cp=replace(cp,rank=4,probe_epsilon=.05)
    if d0 is None:
        with private_rng():
            _,report,_=projection_aware_search(model,batches,reference_cp,diagnostic_all=True,expected_sites=expected_sites)
        d0={'method':'all12_transfer_diagnostic','diagnostics':report['candidates'],'where':report['where'],
            'reference_source':'fresh_D0_recomputation','protocol':protocol,
            'run_identity':{'fork_hash':fork_hash,'cp_config':json.loads(json.dumps(asdict(reference_cp)))}}
        atomic_json_save(d0,output/'d0_recomputed.json')
    with private_rng():
        _,where,proposals=select_by_expansion_gain(model,batches['statistics'],batches['where_batches'],reference_cp,
            next(model.parameters()).device,return_candidates=True)
    if len(proposals)!=expected_sites:raise ValueError('incorrect number of TINY sites')
    verification=verify_baseline(where,d0)
    atomic_json_save(d0,output/'d0_export/result.json')
    selected=select_blocks(d0['diagnostics'],all_blocks=config.all_blocks)
    identity={'version':1,'fork_hash':fork_hash,'protocol':protocol,'cp_config':json.loads(json.dumps(asdict(cp))),
        'expansion_config':json.loads(json.dumps(asdict(config))),'selected_blocks':selected,'probe_indices':indices,
        'baseline_where':where['site_evaluations'],'training_epochs':0,'committed_corrections':0,
        'probe_epsilon_role':'measurement_gate_only_native_TINY_generation_uses_rank_statistics',
        'FP64_metric_role':'CE_recomputed_on_identical_FP32_forward_logits',
        'FP64_forward_role':'whole_model_double_with_same_fixed_FP32_generated_proposal' if config.verify_forward_fp64 else 'not_run'}
    tensors={}
    for name,value in batches.items():
        group=value if name in ('statistics','where_batches') else [value]
        for index,batch in enumerate(group):
            tensors[f'{name}.{index}.inputs']=batch[0]
            tensors[f'{name}.{index}.labels']=batch[1]
    identity['probe_data_hash']=tensor_hash(tensors)
    latest=output/'diagnostic_progress.pt'
    saved=torch.load(latest,map_location='cpu',weights_only=False) if latest.exists() else None
    if saved and saved['run_identity']!=identity:raise ValueError('diagnostic resume identity mismatch')
    atomic_torch_save({'run_identity':identity,'probe_indices':indices,
        'tensors':{name:value.detach().cpu() for name,value in tensors.items()}},output/'probe_batches.pt')
    rows=saved['rows'] if saved else [];cache=saved['proposals'] if saved else {};ledger=saved['proposal_budget'] if saved else {}
    atomic_json_save({'run_identity':identity,'baseline_reproduction':verification,
        'd0_reference_source':d0.get('reference_source','provided_prior_D0'),
        'block_selection_role':'prior_D0_diagnostics_not_validation','interpretation_thresholds':{'E_gain':1e-8,'residual':.5},
        'compute_matched_comparison':False},output/'configuration.json')
    double_model=copy.deepcopy(model).double().eval() if config.verify_forward_fp64 else None
    done={row['case_id'] for row in rows}
    baseline_rows=[]
    for item in proposals:
        metrics=[direction(model,item['candidate'],batch,.05,1.,'exact_expanded_forward') for batch in batches['where_batches']]
        baseline_rows.append({'block_name':item['site'],'tiny_rank':item['rank'],
            'e_gain_fp32':sum(m['e_gain_fp32'] for m in metrics)/len(metrics),
            'e_gain_fp64':sum(m['e_gain_fp64'] for m in metrics)/len(metrics),
            'delta_f_norm':sum(m['delta_f_norm'] for m in metrics)/len(metrics),
            'gain_per_delta_norm':sum(m['e_gain_fp64'] for m in metrics)/max(sum(m['delta_f_norm'] for m in metrics),1e-12)})
        if double_model is not None:
            high=replace(item['candidate'],model=double_model,**{name:getattr(item['candidate'],name).double() for name in ('A','a','B')})
            precise=[direction(double_model,high,(batch[0].double(),batch[1]),.05,1.,'exact_expanded_forward') for batch in batches['where_batches']]
            baseline_rows[-1]['e_gain_forward_fp64']=sum(row['e_gain_fp64'] for row in precise)/len(precise)
        key=f"tiny_original:{item['site']}:4"
        if key not in cache:
            cache[key]=serialize_candidate(item['candidate'])
            ledger[key]={'proposal_method':'tiny_original','site':item['site'],'rank':4,
                'proposal_forward_calls':len(batches['statistics']),'proposal_backward_calls':len(batches['statistics']),
                'initial_all12_WHERE_forward_calls':2*len(batches['where_batches']),
                'search_time_seconds':where['where_seconds'],'shared_budget_id':'baseline_all_sites',
                'proposal_rng_role':'original_checkpoint_rng','seed':None,
                'cost_note':'shared all-site generation/WHERE wall time, not a per-site additive cost'}
    write_csv(baseline_rows,output/'existing_tiny_diagnostics.csv')
    def save():
        atomic_torch_save({'kind':'deit_expansion_diagnostic_progress','run_identity':identity,
            'rows':rows,'proposals':cache,'proposal_budget':ledger},latest)
        write_csv(rows,output/'expansion_diagnostics.csv');atomic_json_save(ledger,output/'proposal_budget.json')
    def proposal(method,site,rank):
        key=f'{method}:{site}:{rank}'
        if key in cache:return (None if cache[key] is None else deserialize_candidate(cache[key],model)),ledger[key]
        seed=case_seed(config,method,site,rank);start=time.perf_counter();counter=[0]
        handle=model.register_forward_pre_hook(lambda *_a:counter.__setitem__(0,counter[0]+1))
        candidate=None
        try:
            with private_rng(seed):
                if method=='tiny_original':
                    candidate=DeitMLPGrowthAdapter().propose_auxiliary_growth(model=model,site=site,batches=batches['statistics'],rank=rank)
                    record={'proposal_backward_calls':len(batches['statistics'])}
                elif method=='aux_gradient_tuned':
                    candidate,record=gradient_tuned(model,site,batches['opt_fit'],rank=rank,steps=config.auxiliary_steps,
                        learning_rate=config.auxiliary_learning_rate,seed=seed)
                else:
                    tiny=deserialize_candidate(cache[f'tiny_original:{site}:4'],model)
                    target=direction(model,tiny,batches['projection_batch'],.05,1.,'exact_expanded_forward')['native_delta_f_norm']
                    candidate,record=random_structural(model,site,batches['projection_batch'][0],norm_target=target,rank=rank,seed=seed)
        except RuntimeError as error:
            if not (str(error).startswith('native Linear TINY produced no extension') or
                    str(error).startswith('nonfinite TINY extension')):raise
            record={'proposal_backward_calls':len(batches['statistics']),'proposal_failure':str(error)}
        finally:handle.remove()
        record.update(proposal_method=method,site=site,rank=rank,seed=seed,search_time_seconds=time.perf_counter()-start,
            proposal_forward_calls=counter[0])
        cache[key]=serialize_candidate(candidate) if candidate is not None else None;ledger[key]=record;save()
        return candidate,record
    cases={}
    def add(stage,method,site,rank,epsilon,scale):
        key=json.dumps([method,site,rank,epsilon,scale,config.strength_mode])
        if key not in cases:cases[key]={'stage_ids':[],'method':method,'site':site,'rank':rank,'epsilon':epsilon,'scale':scale}
        if stage not in cases[key]['stage_ids']:cases[key]['stage_ids'].append(stage)
    for site in selected:
        add('E0','tiny_original',site,4,.05,1.)
        add('E3','tiny_original',site,4,.05,1.)
        for method in ('aux_gradient_tuned','aux_random_direction'):add('E3',method,site,4,.05,1.)
    for site in selected:
        for rank in config.ranks:add('E1','tiny_original',site,rank,.05,1.)
        for epsilon in config.probe_epsilons:
            for scale in config.expansion_scales:add('E2','tiny_original',site,4,epsilon,scale)
    for key,case in cases.items():
        if key in done:continue
        method,site,rank=case['method'],case['site'],case['rank'];epsilon,scale=case['epsilon'],case['scale']
        candidate,budget=proposal(method,site,rank);evaluation_epsilon=1. if method=='aux_gradient_tuned' else epsilon
        started=time.perf_counter();seed=case_seed(config,'projection',key)
        forward_calls=[0]
        handle=model.register_forward_pre_hook(lambda *_a:forward_calls.__setitem__(0,forward_calls[0]+1))
        try:
            with private_rng(seed):
                if candidate is None:
                    record={'e_gain_fp32':None,'e_gain_fp64':None,'e_gain_forward_fp64':None,'delta_f_norm':None,
                        'projection_residual':None,'projection_cosine':None,'gate_gain':None,'accepted':False,
                        'solver_status':'not_run_no_E_proposal','damping':None,'proposal_failure':budget['proposal_failure']}
                else:
                    record=evaluate_transfer(model,double_model,candidate,batches,replace(cp,rank=rank,probe_epsilon=epsilon),
                        evaluation_epsilon,scale,config.strength_mode)
        finally:handle.remove()
        record.update(case_id=key,stage_ids=case['stage_ids']+['E4'],proposal_method=method,block_name=site,rank=rank,
            effective_rank=candidate.A.shape[0] if candidate else 0,probe_epsilon=epsilon,expansion_scale=scale,
            evaluation_gate=evaluation_epsilon,proposal_seed=budget.get('seed',None),projection_seed=seed,
            proposal_rng_role=budget.get('proposal_rng_role','private_seed'),
            search_time_seconds=budget['search_time_seconds'],evaluation_time_seconds=time.perf_counter()-started,
            tuning_gain=budget.get('tuning_gain'),proposal_generation_id=f'{method}:{site}:{rank}',
            tuning_role='disjoint_opt_fit_only',E_gain_role='independent_WHERE_batches',gate_role='scale_selection_batch',
            forward_precision=str(next(model.parameters()).dtype),metric_fp64_role=identity['FP64_metric_role'],
            forward_fp64_role=identity['FP64_forward_role'],strength_mode=config.strength_mode,
            norm_match_converged=budget.get('norm_match_converged'),norm_target=budget.get('norm_target'),
            proposal_forward_calls=budget['proposal_forward_calls'],proposal_backward_calls=budget['proposal_backward_calls'],
            evaluation_model_forward_calls=forward_calls[0],evaluation_forward_counter_scope='base_model_calls_including_JVP_VJP_not_double_clone')
        record.update(norm_actual=budget.get('norm_actual'),norm_match_relative_error=budget.get('norm_match_relative_error'),
            control_comparison_eligible=(budget.get('norm_match_converged',False) and not budget.get('degenerate_random_control',False))
                if method=='aux_random_direction' else None)
        if tensor_hash(model.state_dict())!=original:raise AssertionError('diagnostic proposal mutated fork')
        rows.append(record);save()
        print(json.dumps({'expansion_diagnostic':{k:record[k] for k in ('proposal_method','block_name','rank','probe_epsilon','expansion_scale','e_gain_fp64','gate_gain','accepted')}}),flush=True)
    atomic_json_save({'run_identity':identity,'baseline_reproduction':verification,'rows':len(rows),
        'd0_reference_source':d0.get('reference_source','provided_prior_D0'),
        'accepted':sum(row['accepted'] for row in rows),'training_epochs':0,'committed_corrections':0,
        'compute_matched_comparison':False,'degenerate_random_controls':sum(bool(row.get('degenerate_target')) for row in rows if row['proposal_method']=='aux_random_direction')},output/'summary.json')
    return rows


def evaluate_transfer(model,double_model,candidate,batches,cp,epsilon,scale,mode):
    where=[direction(model,candidate,batch,epsilon,scale,mode) for batch in batches['where_batches']]
    fit=direction(model,candidate,batches['projection_batch'],epsilon,scale,mode)
    gate=direction(model,candidate,batches['gate_batch'],epsilon,scale,mode)
    n=len(where);norm=sum(row['delta_f_norm'] for row in where)/n
    record={'e_gain_fp32':sum(row['e_gain_fp32'] for row in where)/n,
        'e_gain_fp64':sum(row['e_gain_fp64'] for row in where)/n,'delta_f_norm':norm,
        'per_batch_e_gain_fp32':[row['e_gain_fp32'] for row in where],
        'per_batch_e_gain_fp64':[row['e_gain_fp64'] for row in where],
        'gain_per_delta_norm':sum(row['e_gain_fp64'] for row in where)/max(n*norm,1e-12),
        'projection_target_norm':fit['delta_f_norm'],'degenerate_target':fit['delta_f_norm']<=1e-12,
        'linear_logits_e_gain_fp64':sum(row['linear_logits_gain_fp64'] for row in where)/n}
    record.update(e_gain_zero_fp32=record['e_gain_fp32']==0.,
        positive_gain_only_in_fp64_metric=record['e_gain_fp32']<=0<record['e_gain_fp64'])
    if double_model is not None:
        high=replace(candidate,model=double_model,**{name:getattr(candidate,name).double() for name in ('A','a','B')})
        precise=[direction(double_model,high,(batch[0].double(),batch[1]),epsilon,scale,mode) for batch in batches['where_batches']]
        record.update(e_gain_forward_fp64=sum(row['e_gain_fp64'] for row in precise)/n,
            delta_f_norm_forward_fp64=sum(row['delta_f_norm'] for row in precise)/n)
    else:record.update(e_gain_forward_fp64=None,delta_f_norm_forward_fp64=None)
    record['forward_rounding_suspected']=bool(double_model is not None and record['delta_f_norm']<=1e-12
        and record['delta_f_norm_forward_fp64']>1e-12)
    try:
        projection,metrics=project_target(model,candidate.module_name,fit['delta'],gate['delta'],
            batches['projection_batch'],batches['gate_batch'],cp)
    except RuntimeError as error:
        if str(error)!='all projection attempts produced non-finite solutions':raise
        record.update(projection_residual=None,projection_cosine=None,gate_gain=None,accepted=False,
            selected_scale=None,solver_status='nonfinite_solution',damping=None,solver_error=str(error))
        return record
    names=DeitMLPGrowthAdapter.original_mlp_parameters(model,candidate.module_name)
    if set(projection.parameter_delta)!=set(names):raise AssertionError('cross-block projection')
    finite=[(float(alpha),value) for alpha,value in metrics['line_search_gains'].items() if math.isfinite(value)]
    best_alpha,best_gain=max(finite,key=lambda pair:pair[1]) if finite else (None,None)
    accepted=bool(metrics['correction_applied'] and best_gain is not None and best_gain>0
        and metrics['selected_scale']==best_alpha and not record['degenerate_target'])
    record.update(projection_residual=projection.relative_residual,projection_cosine=projection.cosine_alignment,
        gate_gain=best_gain,accepted=accepted,selected_scale=metrics['selected_scale'] if accepted else None,
        line_search_gains=metrics['line_search_gains'],projection_parameter_names=list(names),
        solver_status='converged' if projection.cg.converged else 'iteration_limit_or_breakdown',
        cg_relative_residual=projection.cg.relative_residual,damping=projection.damping_used,
        gate_gain_numerically_tiny=best_gain is not None and 0<best_gain<=1e-8)
    record['transfer_description']=classify(record['e_gain_fp64'],record['projection_residual'],best_gain,accepted)
    return record
