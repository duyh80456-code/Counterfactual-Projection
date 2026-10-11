"""Seed2 E0..E4: reuse plateau, diagnose E generators, no long training arm."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from adapters.deit_ablation import res18_cp_config
from diagnostics.deit_expansion_suite import ExpansionConfig, run_suite
from experiments.deit_protocol import checked_source, load_training_context, materialize_probe_batches
from experiments.run_deit_projection_aware import validate_seed2_fork
from experiments.deit_projection_resume import iter_results
from experiments.deit_resume import FULL_STATE, unique
from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.shared_protocol import sha256_file, rng_state, restore_rng, atomic_json_save


def prepare_inputs(roots,output,*,d0_path=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);found=[]
    for i,root in enumerate(dict.fromkeys(map(str,[output,*roots]))):
        if not Path(root).exists():continue
        items,_=discover_checkpoints(root,output/'_fork_cache'/str(i),kind={'deit_plateau_fork'},
            payload_filter=lambda p:FULL_STATE.issubset(p) and p.get('protocol',{}).get('seed')==2
                and p.get('protocol',{}).get('protocol_version')==5 and p.get('epoch')==375
                and p.get('historical_best_accuracy')==.562,
            payload_transform=lambda p:{k:v for k,v in p.items() if k not in FULL_STATE})
        found.extend(items)
    selected=unique(found,'seed2 diagnostic fork375 /56.20%')
    if not selected:raise FileNotFoundError('Attach original full seed2 plateau_checkpoint.pt (epoch375 /56.20%) or output archive')
    target=output/'plateau_checkpoint.pt'
    if selected['path'].resolve()!=target.resolve():
        temporary=target.with_suffix('.importing')
        try:shutil.copy2(selected['path'],temporary);temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
    fork,recipe=checked_source(target,{'deit_plateau_fork'});validate_seed2_fork(fork,recipe);h=sha256_file(target)
    choices=[]
    if d0_path:
        value=json.loads(Path(d0_path).read_text())
        if value.get('method')!='all12_transfer_diagnostic':raise ValueError('explicit D0 file is not all12_transfer_diagnostic')
        candidates=[(str(d0_path),value)]
    else:
        local=[(str(path),json.loads(path.read_text())) for path in
               (output/'d0_reference.json',output/'d0_recomputed.json') if path.exists()]
        candidates=iter([*local,*iter_results(roots)])
    for path,result in candidates:
        if result.get('method')!='all12_transfer_diagnostic':continue
        identity=result.get('run_identity',{})
        if identity.get('fork_hash')!=h or result.get('protocol')!=fork['protocol']:
            if d0_path:raise ValueError('explicit D0 belongs to another fork/protocol')
            continue
        from dataclasses import asdict
        if identity.get('cp_config')!=json.loads(json.dumps(asdict(res18_cp_config()))):
            raise ValueError('D0 configuration differs from original rank4/projection32 protocol')
        if len(result.get('diagnostics',[]))!=12:raise ValueError('D0 does not contain all12 sites')
        choices.append((path,result))
    if choices and any(result['diagnostics']!=choices[0][1]['diagnostics'] for _,result in choices):
        raise ValueError('multiple distinct D0 references; choose explicitly with --d0-result')
    d0=choices[0][1] if choices else None
    if d0:atomic_json_save(d0,output/'d0_reference.json')
    atomic_json_save({'fork_path':str(target),'fork_hash':h,'d0_source':choices[0][0] if choices else None,
        'without_prior_D0':'recompute all12 D0 first; no claim of matching a missing historical file',
        'training_epochs':0},output/'input_plan.json')
    return target,fork,recipe,d0


def import_progress(roots,output,fork_hash,protocol,config,cp):
    from dataclasses import asdict
    items=[];output=Path(output)
    for i,root in enumerate(dict.fromkeys(map(str,[output,*roots]))):
        if not Path(root).exists():continue
        found,_=discover_checkpoints(root,output/'_progress_cache'/str(i),kind={'deit_expansion_diagnostic_progress'},
            payload_filter=lambda p:p.get('run_identity',{}).get('protocol')==protocol,
            payload_transform=lambda p:{'run_identity':p['run_identity'],'rows':len(p['rows']),'proposals':len(p['proposals'])})
        items.extend(found)
    for item in items:
        identity=item['payload']['run_identity']
        if (identity['fork_hash']!=fork_hash or identity['cp_config']!=json.loads(json.dumps(asdict(cp)))
                or identity['expansion_config']!=json.loads(json.dumps(asdict(config)))):
            raise ValueError('diagnostic import fork/config mismatch; use a separate output/attachment set')
    if not items:return None
    progress=lambda item:(item['payload']['rows'],item['payload']['proposals'])
    maximum=max(map(progress,items))
    selected=unique([item for item in items if progress(item)==maximum],'diagnostic progress')
    target=output/'diagnostic_progress.pt'
    if selected['path'].resolve()!=target.resolve():
        temporary=target.with_suffix('.importing')
        try:shutil.copy2(selected['path'],temporary);temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-root',type=Path,action='append',default=[]);parser.add_argument('--d0-result',type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--all-blocks',action='store_true')
    parser.add_argument('--auxiliary-steps',type=int,default=20);parser.add_argument('--auxiliary-learning-rate',type=float,default=.001)
    parser.add_argument('--no-forward-fp64',action='store_true')
    parser.add_argument('--strength-mode',choices=('exact_expanded_forward','linear_logits'),default='exact_expanded_forward')
    args=parser.parse_args();path,fork,recipe,d0=prepare_inputs(args.input_root,args.output,d0_path=args.d0_result)
    model,optimizer,_,_,_,evaluation,train_ids,*_=load_training_context(args.data_root,recipe,torch.device(args.device),fork)
    model.eval();cp=res18_cp_config();live=rng_state()
    batches,indices=materialize_probe_batches(evaluation,train_ids,recipe,cp,torch.device(args.device),include_opt=True)
    restore_rng(live)
    config=ExpansionConfig(auxiliary_steps=args.auxiliary_steps,auxiliary_learning_rate=args.auxiliary_learning_rate,
        all_blocks=args.all_blocks,verify_forward_fp64=not args.no_forward_fp64,strength_mode=args.strength_mode)
    import_progress(args.input_root,args.output,sha256_file(path),fork['protocol'],config,cp)
    run_suite(model,batches,cp,config,args.output,fork_hash=sha256_file(path),protocol=fork['protocol'],
        probe_indices=indices,d0=d0,optimizer=optimizer)

if __name__=='__main__':main()
