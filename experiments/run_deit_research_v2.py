"""Opt-in v2 diagnostics and fixed-site one-shot continuations; old baselines unchanged."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import torch
from torch.utils.data import DataLoader,Subset
from adapters.deit_ablation import res18_cp_config,candidate_from_reference
from adapters.deit_cp_adapter import ResearchCPConfig
from adapters.deit_expansion_generators import GeneratorConfig,ExpansionCandidate,generate
from adapters.deit_projection_aware import tensor_hash
from diagnostics.deit_expansion_suite import private_rng,serialize_candidate,deserialize_candidate,tree_equal
from diagnostics.deit_projection_v2 import analyze,choose_projected_candidate
from experiments.deit_protocol import checked_source,load_training_context,materialize_probe_batches
from experiments.run_deit_projection_aware import run,write_csv
from experiments.run_deit_all_arms import gpu_slots,run_jobs_parallel,run_jobs
from experiments.shared_protocol import sha256_file,atomic_json_save,atomic_torch_save,rng_state,restore_rng
from projection.exact_reference import svd_reference,solver_error

METHODS=('tiny','gradmax','optimized_e')


def research_batches(evaluation,train_ids,recipe,cp,device,heldout_samples=256):
    batches,indices=materialize_probe_batches(evaluation,train_ids,recipe,cp,device,include_opt=True)
    reserved=set(indices['statistics']+indices['projection']+indices['gate']+indices['opt_fit']+indices['opt_val']+
                 sum(indices['where'],[]))
    generator=torch.Generator().manual_seed(911731+recipe.seed*10000)
    order=torch.randperm(len(train_ids),generator=generator).tolist()
    heldout=[train_ids[i] for i in order if train_ids[i] not in reserved][:heldout_samples]
    if len(heldout)!=heldout_samples:raise ValueError('insufficient independent heldout samples')
    loader=DataLoader(Subset(evaluation,heldout),batch_size=heldout_samples,num_workers=0,
        generator=torch.Generator().manual_seed(recipe.seed))
    batches['heldout']=tuple(t.to(device) for t in next(iter(loader)));indices['heldout']=heldout
    return batches,indices


def small_reference_report(output):
    # A bounded rank-deficient example: demonstrates identifiability, not full-DeiT floor.
    J=torch.tensor([[1.,0.,0.],[0.,2.,0.],[0.,0.,0.]],dtype=torch.float64)
    target=torch.tensor([1.,1.,1.],dtype=torch.float64)
    from projection.cg import conjugate_gradient
    rows=[]
    for mode,mu in [('absolute',.001),('relative_gram',.001*float(J.square().sum()/3))]:
        ref=svd_reference(J,target,mu)
        dual=conjugate_gradient(lambda x:J@J.T@x+mu*x,target,max_iter=30,tolerance=1e-10)
        error=solver_error(J,target,J.T@dual.solution,ref)
        rows.append({'damping_mode':mode,'damping_used':mu,
            **{k:ref[k] for k in ('minimum_residual','ridge_residual','regularization_error','numerical_rank')},**error})
    atomic_json_save({'problem':'synthetic3x3_rank2_not_full_DeiT','rows':rows},output/'small_svd_reference.json')


def identity(fork_hash,raw_hash,protocol,config,indices):
    return {'version':2,'fork_hash':fork_hash,'raw_reference_hash':raw_hash,'protocol':protocol,
        'generator_config':asdict(config),'base_cp_config':json.loads(json.dumps(asdict(res18_cp_config()))),
        'probe_indices':indices,'site_policy':'fixed_TINY_top1_reference_site',
        'damping_modes':['absolute','relative_gram'],'trace_probes':8,'trace_seed':2028,
        'scope_policy':'single_site_plus_nearest_neighbor_diagnostic_only','official_test_used':False,
        'SSD_status':'not_registered_GELU_splitting_matrix_not_audited','stationarity_verified':False}


def prepare(args):
    fork,recipe=checked_source(args.plateau_checkpoint,{'deit_plateau_fork'})
    raw=torch.load(args.raw_reference,map_location='cpu',weights_only=False);h=sha256_file(args.plateau_checkpoint)
    cp=res18_cp_config()
    if raw['fork_hash']!=h or raw['cp_config']!=asdict(cp) or raw['seed']!=recipe.seed:raise ValueError('v2 raw/fork/config mismatch')
    model,opt,_,_,_,evaluation,train_ids,*_=load_training_context(args.data_root,recipe,torch.device(args.device),fork)
    model.eval();live=rng_state()
    batches,indices=research_batches(evaluation,train_ids,recipe,cp,torch.device(args.device));restore_rng(live)
    if {k:indices[k] for k in raw['probe_indices']}!=raw['probe_indices']:raise ValueError('original probe partition changed')
    cfg=GeneratorConfig(seed=recipe.seed)
    ident=identity(h,sha256_file(args.raw_reference),fork['protocol'],cfg,indices)
    root=args.output;root.mkdir(parents=True,exist_ok=True);cache=root/'generator_reference.pt'
    if cache.exists():
        state=torch.load(cache,map_location='cpu',weights_only=False)
        if state['run_identity']!=ident:raise ValueError('v2 reference belongs to another fork/config')
        return fork,recipe,raw,model,opt,batches,ident,state['candidates']
    before=tensor_hash(model.state_dict());optimizer_before=copy.deepcopy(opt.state_dict());candidates={}
    for name in METHODS:
        if name=='tiny':
            candidate=ExpansionCandidate(candidate_from_reference(model,raw),name,.05,
                {'source':'cached_original_TINY_reference','search_time_seconds':0.,'forward_calls':0,'backward_calls':0,
                 'cost_note':'initial generation already charged to raw reference creation'})
        else:candidate=generate(model,raw['site'],batches['statistics'],[batches['opt_fit']],cfg,name)
        candidates[name]={'branch':serialize_candidate(candidate.branch),'gate':candidate.gate,'budget':candidate.budget}
        if tensor_hash(model.state_dict())!=before:raise AssertionError('generator modified O')
    if not tree_equal(optimizer_before,opt.state_dict()):raise AssertionError('generator changed original AdamW')
    atomic_torch_save({'kind':'deit_v2_generator_reference','run_identity':ident,'candidates':candidates,
        'heldout_batch':tuple(t.cpu() for t in batches['heldout'])},cache)
    atomic_torch_save({'indices':indices,'batches':{k:([(x.cpu(),y.cpu()) for x,y in v] if isinstance(v,list)
        else tuple(t.cpu() for t in v)) for k,v in batches.items()}},root/'probe_batches.pt')
    return fork,recipe,raw,model,opt,batches,ident,candidates


def diagnose(args):
    fork,recipe,raw,model,opt,batches,ident,candidates=prepare(args);rows=[];before=tensor_hash(model.state_dict())
    old=copy.deepcopy(opt.state_dict());index=int(raw['site'].split('.')[1])
    neighbor=(index+1)%len(model.blocks)
    scopes=[[raw['site']],[raw['site'],f'blocks.{neighbor}.mlp']] if len(model.blocks)>1 else [[raw['site']]]
    for name,saved in candidates.items():
        candidate=ExpansionCandidate(deserialize_candidate(saved['branch'],model),name,saved['gate'],saved['budget'])
        for mode in ('absolute','relative_gram'):
            cp=ResearchCPConfig(**asdict(res18_cp_config()),damping_mode=mode)
            for scope in scopes:
                print(json.dumps({'v2_diagnostic_start':{'generator':name,'damping_mode':mode,'scope':scope}}),flush=True)
                _,row=analyze(model,candidate,batches,cp,scope=scope,seed=recipe.seed)
                row.update(seed=recipe.seed,checkpoint_hash=ident['fork_hash'],
                    raw_reference_hash=ident['raw_reference_hash'],technical_check_passed=True)
                rows.append(row)
                print(json.dumps({'v2_diagnostic_complete':{key:row[key] for key in
                    ('generator','selected_site','projection_scope','damping_mode','damping_used',
                     'cg_converged','projection_relative_residual','heldout_cosine','gate_gain','gate_accepted')}}),flush=True)
                write_csv(rows,args.output/'expansion_projection_diagnostics.csv')
                atomic_json_save({'run_identity':ident,'rows':rows,'technical_check_passed':False},args.output/'diagnostics.json')
                if tensor_hash(model.state_dict())!=before:raise AssertionError('preview changed O')
    if not tree_equal(old,opt.state_dict()):raise AssertionError('projection changed AdamW')
    small_reference_report(args.output)
    selections={metric:choose_projected_candidate(rows,metric) for metric in
        ('gate_gain','gate_margin_gain_after_projection')}
    atomic_json_save({'scope':'experimental_gate_only_diagnostic_no_commit',
        'selected':{metric:({key:row[key] for key in ('generator','selected_site','damping_mode','gate_gain','gate_margin_gain_after_projection')}
            if row else None) for metric,row in selections.items()}},args.output/'experimental_selection.json')
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except (OSError,subprocess.CalledProcessError):commit=None
    atomic_json_save({'run_identity':ident,'rows':rows,'technical_check_passed':True,
        'training_epochs':0,'committed_corrections':0,'full_DeiT_range_limit_known':False,'source_commit':commit},args.output/'diagnostics.json')
    print('v2 technical diagnostics complete; no accuracy improvement established',flush=True)


def continuation(args):
    fork,recipe=checked_source(args.plateau_checkpoint,{'deit_plateau_fork'})
    reference=torch.load(args.output.parent/'generator_reference.pt',map_location='cpu',weights_only=False)
    report=json.loads((args.output.parent/'diagnostics.json').read_text());ident=reference['run_identity']
    if not report['technical_check_passed'] or report['run_identity']!=ident:raise ValueError('technical diagnostics not complete/matched')
    if ident['fork_hash']!=sha256_file(args.plateau_checkpoint) or ident['raw_reference_hash']!=sha256_file(args.raw_reference):
        raise ValueError('continuation fork/reference mismatch')
    name=args.generator;mode=args.damping_mode;method=f'cp_v2_{name}_{mode}'
    config=ResearchCPConfig(**asdict(res18_cp_config()),damping_mode=mode)
    run_identity={'version':2,'fork_hash':ident['fork_hash'],'method':method,'cp_config':json.loads(json.dumps(asdict(config))),
        'generator_reference_hash':sha256_file(args.output.parent/'generator_reference.pt'),
        'generator':name,'site_policy':ident['site_policy'],'selected_site':reference['candidates'][name]['branch']['site'],
        'post_fork_epochs':150,'algorithm_patience':10,'retrigger':False,'anchor_policy':'accuracy_then_lower_loss_on_exact_tie',
        'report_best_scope':'postfork_training_validation_only','projection_scope':'original_selected_MLP_only'}
    saved=reference['candidates'][name]
    def search(model,batches,_config,**kwargs):
        # Exactly the cached proposal that passed diagnostics, same fixed site.
        candidate=ExpansionCandidate(deserialize_candidate(saved['branch'],model),name,saved['gate'],saved['budget'])
        # The fit/gate partition is unchanged; heldout is diagnostic only.
        heldout=tuple(t.to(batches['projection_batch'][0].device) for t in reference['heldout_batch'])
        batches={**batches,'heldout':heldout}
        projection,row=analyze(model,candidate,batches,config,seed=recipe.seed)
        accepted=row['gate_accepted']
        chosen={'candidate':candidate.branch,'projection':projection,'record':row} if accepted else None
        with torch.no_grad():direction=candidate.delta_logits((kwargs['comparison_inputs'],torch.zeros(len(kwargs['comparison_inputs']),dtype=torch.long,device=kwargs['comparison_inputs'].device))).cpu() if accepted else None
        record={'where':{'selection_candidate_count':1},'candidates':[row],
            'candidate_blocks':[candidate.site],'selected_block':candidate.site if accepted else None,
            'selected_gate_gain':row['gate_gain'],'accepted_count':int(accepted),'rejected_count':int(not accepted),
            'no_valid_intervention':not accepted,'anchor_hash':tensor_hash(model.state_dict()),'search_seconds':row['projection_time_seconds']}
        return chosen,record,direction
    args.method=method;args.post_fork_epochs=150;args.algorithm_patience=10
    latest=args.output/'checkpoint_latest.pt';args.resume=latest if latest.exists() else None
    return run(args,fork,recipe,config,identity_override=run_identity,search_fn=search)


def compare_results(args,statuses):
    fork,_=checked_source(args.plateau_checkpoint,{'deit_plateau_fork'});h=sha256_file(args.plateau_checkpoint)
    baseline_path=args.output.parent/'arm_status.json'
    baseline_status=json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    rows=[]
    for method,status in {**baseline_status,**statuses}.items():
        if method not in ('e_driven_o_raw','vanilla_rollback') and not method.startswith('cp_v2_'):continue
        root=args.output if method.startswith('cp_v2_') else args.output.parent/'arms'
        path=root/method/'result.json'
        if status.get('status')!='completed':
            rows.append({'method':method,'available':False,'status':status.get('status')});continue
        try:
            result=json.loads(path.read_text())
            if (result['theta_best_hash']!=h or result['protocol']!=fork['protocol']
                    or result['run_identity']['post_fork_epochs']!=150 or result['history'][-1]['post_fork_epoch']!=150
                    or result['run_identity']['algorithm_patience']!=10):raise ValueError('fork/protocol/budget mismatch')
            rows.append({'method':method,'available':True,'fork_hash':h,
                **{key:result[key] for key in ('report_best_accuracy','report_best_loss','report_best_epoch','scientific_escape','delta_vs_historical_best')},
                'rollback_count':len(result['rollback_events'])})
        except Exception as error:rows.append({'method':method,'available':False,'comparison_error':repr(error)})
    vanilla=next((row['report_best_accuracy'] for row in rows if row['method']=='vanilla_rollback' and row['available']),None)
    for row in rows:
        row['delta_vs_vanilla_rollback']=row['report_best_accuracy']-vanilla if row['available'] and vanilla is not None else None
    write_csv(rows,args.output/'continuation_comparison.csv')
    atomic_json_save({'rows':rows,'selection_policy':'fixed_original_TINY_top1_site_for_alternative_generators',
        'compute_budget_equal':False,'SSD_implemented':False},args.output/'continuation_comparison.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','plateau-checkpoint','raw-reference','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--phase',choices=('diagnostics','continuation','all-continuations'),default='diagnostics')
    parser.add_argument('--generator',choices=METHODS);parser.add_argument('--damping-mode',choices=('absolute','relative_gram'))
    parser.add_argument('--gpu-devices',default='auto')
    parser.add_argument('--input-root',type=Path,action='append',default=[])
    args=parser.parse_args()
    if args.phase!='continuation':
        from experiments.deit_research_resume import import_research
        fork,_=checked_source(args.plateau_checkpoint,{'deit_plateau_fork'})
        import_research(args.input_root,args.output,fork,sha256_file(args.raw_reference))
    if args.phase=='diagnostics':return diagnose(args)
    if args.phase=='continuation':
        if not args.generator or not args.damping_mode:parser.error('continuation needs generator and damping mode')
        return continuation(args)
    commands={}
    for name in METHODS:
        for mode in ('absolute','relative_gram'):
            if name=='tiny' and mode=='absolute':continue # existing raw baseline, reused separately
            method=f'cp_v2_{name}_{mode}'
            commands[method]=[sys.executable,'-m','experiments.run_deit_research_v2','--phase','continuation',
                '--generator',name,'--damping-mode',mode,'--data-root',str(args.data_root),
                '--plateau-checkpoint',str(args.plateau_checkpoint),'--raw-reference',str(args.raw_reference),
                '--output',str(args.output/method),'--device',args.device]
    slots=gpu_slots(args.gpu_devices,args.device)
    statuses=run_jobs_parallel(commands,args.output/'v2_arm_status.json',slots) if slots else run_jobs(commands,args.output/'v2_arm_status.json')
    compare_results(args,statuses)
    if any(row['status']=='failed' for row in statuses.values()):raise RuntimeError('Some v2 arms failed; other arms were attempted')

if __name__=='__main__':main()
