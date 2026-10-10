"""Matched-budget relaxed GELU Bypass from the same historical-best DeiT fork."""
from __future__ import annotations
import argparse
import copy
import math
import time
from pathlib import Path
import torch
from adapters.deit_bypass import BypassConfig, embed, add_to_optimizer, contraction_norm, contract
from experiments.deit_protocol import checked_source, historical_best, load_training_context, save_state, evaluate_without_rng
from experiments.shared_protocol import atomic_json_save, atomic_torch_save, seed_everything, restore_rng, sha256_file, train_epoch
from experiments.deit_logging import emit_event, emit_epoch
from experiments.run_deit_projection_aware import write_csv

KIND = 'deit_bypass_latest'
METHOD = 'deit_bypass'


def identity_for(fork_hash, config, horizon):
    config.validate(horizon)
    return {'version': 1, 'method': METHOD, 'fork_hash': fork_hash,
            'post_fork_epochs': horizon, 'bypass': config.identity()}


def validate_progress(state, identity, fork):
    count = state['completed_epochs']; phase = state['phase']
    cfg = identity['bypass']
    if (state.get('kind') != KIND or state.get('run_identity') != identity
            or state.get('protocol') != fork['protocol'] or not 0 <= count <= identity['post_fork_epochs']
            or state['epoch'] != fork['epoch'] + count or len(state['history']) != count + 1
            or phase not in ('opt1', 'opt2', 'train3')
            or sum(state[k] for k in ('opt1_done','opt2_done','train3_done')) != count
            or not 0 <= state['opt1_done'] <= cfg['opt1_epochs']
            or not 0 <= state['opt2_done'] <= cfg['max_opt2_epochs']
            or (phase == 'opt1' and state['opt1_done'] >= cfg['opt1_epochs'])
            or (phase != 'opt1' and state['opt1_done'] != cfg['opt1_epochs'])
            or (phase == 'train3' and state['contraction_at_projection'] is None)):
        raise ValueError('inconsistent Bypass resume state/config/fork/phase')
    expanded = any(name.endswith('.mlp.act.d') for name in state['model'])
    if expanded != (phase != 'train3'):
        raise ValueError('Bypass resume phase and model extension geometry disagree')


def finish(state, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    original = [row for row in state['history'] if row.get('accuracy_comparison_eligible')]
    expanded = [row for row in state['history'] if row.get('post_fork_epoch', 0)>0]
    # Strict accuracy ties retain the loss and epoch of the first best observation.
    best = max(original, key=lambda row: row['validation_accuracy']) if original else None
    diagnostic = max(expanded, key=lambda row: row['validation_accuracy']) if expanded else None
    accuracy = best['validation_accuracy'] if best else None
    eligible = state['phase'] == 'train3' and best is not None
    result = {'method': METHOD, 'protocol': state['protocol'], 'run_identity': state['run_identity'],
        'theta_best_hash': state['run_identity']['fork_hash'], 'fork_epoch': state['historical_best_epoch'],
        'historical_best_accuracy': state['historical_best_accuracy'], 'historical_best_loss': state['historical_best_loss'],
        'completed_epochs': state['completed_epochs'], 'post_fork_epochs': state['run_identity']['post_fork_epochs'],
        'report_best_accuracy': accuracy, 'report_best_loss': best['validation_loss'] if best else None,
        'report_best_epoch': best['epoch'] if best else None,
        'delta_vs_historical_best': accuracy-state['historical_best_accuracy'] if best else None,
        'scientific_escape': bool(best and accuracy>state['historical_best_accuracy']),
        'best_expanded_or_original_diagnostic': diagnostic, 'report_best_scope': 'original_space_after_contraction',
        'bypass_completed': state['phase']=='train3', 'accuracy_comparison_eligible': eligible,
        'comparison_status': 'completed_original_space' if eligible else 'budget_exhausted_before_contraction',
        'phase': state['phase'], 'opt1_epochs': state['opt1_done'], 'opt2_epochs': state['opt2_done'],
        'train3_epochs': state['train3_done'], 'opt2_steps': state['opt2_steps'],
        'projection_loss_jump': state['projection_loss_jump'],
        'contraction_norm_at_projection': state['contraction_at_projection'],
        'history': state['history'], 'rollback_events': [], 'retrigger': False,
        'base_parameter_count': state['base_parameter_count'],
        'extension_parameter_count': state['extension_parameter_count'],
        'peak_train_parameter_count': state['base_parameter_count']+state['extension_parameter_count'],
        'source_paper': state['run_identity']['bypass']['paper']}
    atomic_json_save(result, output/'result.json')
    atomic_json_save({'protocol':state['protocol'],'run_identity':state['run_identity']}, output/'run_metadata.json')
    write_csv(state['history'], output/'epoch_history.csv')
    if state.get('best_original'):
        atomic_torch_save(state['best_original'], output/'best_checkpoint.pt')
    else:
        (output/'best_checkpoint.pt').unlink(missing_ok=True)
    emit_event('deit_arm_complete', {k:result[k] for k in ('method','report_best_accuracy','bypass_completed','accuracy_comparison_eligible')}, output)
    return result


def run(args, fork, recipe, config, *, expected_sites=12):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    identity=identity_for(sha256_file(args.plateau_checkpoint),config,args.post_fork_epochs)
    saved=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if saved:
        validate_progress(saved,identity,fork)
        if saved['completed_epochs']==args.post_fork_epochs:return finish(saved,output)
    seed_everything(recipe.seed);device=torch.device(args.device)
    # Build the base geometry and full fork optimizer before adding D coordinates.
    context=load_training_context(args.data_root,recipe,device,fork)
    model,optimizer,scheduler,loader,evaluation_loader,_,train_ids,val_ids,tuning_ids,reserved_ids=context
    base_parameter_count=sum(p.numel() for p in model.parameters())
    extension_parameter_count=sum(block.mlp.fc1.out_features for block in model.blocks)
    phase=saved['phase'] if saved else 'opt1'
    if phase!='train3':
        embed(model,expected_sites=expected_sites);add_to_optimizer(model,optimizer)
    if saved:
        model.load_state_dict(saved['model'],strict=True)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        scheduler.load_state_dict(copy.deepcopy(saved['scheduler']))
        loader.generator.set_state(saved['train_loader_generator_state'].cpu());restore_rng(saved['rng'])
    accuracy,loss,epoch0=historical_best(fork)
    history=saved['history'] if saved else [{'epoch':epoch0,'post_fork_epoch':0,
        'validation_accuracy':accuracy,'validation_loss':loss,'accuracy_comparison_eligible':False,'phase':'fork'}]
    completed=saved['completed_epochs'] if saved else 0
    opt1=saved['opt1_done'] if saved else 0;opt2=saved['opt2_done'] if saved else 0
    train3=saved['train3_done'] if saved else 0;steps=saved['opt2_steps'] if saved else 0
    contraction=saved['contraction_at_projection'] if saved else None
    jump=saved['projection_loss_jump'] if saved else None
    best=saved.get('best_original') if saved else None
    best_accuracy=max((r['validation_accuracy'] for r in history if r.get('accuracy_comparison_eligible')),default=float('-inf'))
    latest=output/'checkpoint_latest.pt'
    def save():
        save_state(latest,model=model,optimizer=optimizer,scheduler=scheduler,loader=loader,
            epoch=epoch0+completed,history=history,train_indices=train_ids,evaluation_indices=val_ids,
            source_tuning_indices=tuning_ids,trigger_indices=reserved_ids,run_protocol=fork['protocol'],kind=KIND,
            run_identity=identity,completed_epochs=completed,phase=phase,opt1_done=opt1,opt2_done=opt2,
            train3_done=train3,opt2_steps=steps,contraction_at_projection=contraction,projection_loss_jump=jump,
            historical_best_epoch=epoch0,historical_best_accuracy=accuracy,historical_best_loss=loss,best_original=best,
            base_parameter_count=base_parameter_count,extension_parameter_count=extension_parameter_count)
    emit_event('deit_run_start',{'method':METHOD,'seed':recipe.seed,'fork_epoch':epoch0,
        'resume_offset':completed,'phase':phase,'post_fork_epochs':args.post_fork_epochs,'bypass':config.identity(),
        'base_parameter_count':base_parameter_count,'extension_parameter_count':extension_parameter_count},output)
    save()
    for offset in range(completed+1,args.post_fork_epochs+1):
        started=time.perf_counter()
        epoch_phase=phase;gamma=0.;rates=[group['lr'] for group in optimizer.param_groups]
        if phase=='opt1':
            training=train_epoch(model,loader,optimizer,device);opt1+=1
            if opt1==config.opt1_epochs:phase='opt2'
        elif phase=='opt2':
            def penalty():
                nonlocal steps,gamma
                steps+=1;gamma=config.gamma_slope*steps
                return gamma*contraction_norm(model)
            training=train_epoch(model,loader,optimizer,device,loss_extra=penalty);opt2+=1
            norm=float(contraction_norm(model).detach())
            if norm<config.contraction_epsilon:
                # Validation observes a committed transition; it does not choose it.
                before=evaluate_without_rng(model,evaluation_loader,device)
                if not contract(model,optimizer,config.contraction_epsilon):raise AssertionError('contraction criterion changed')
                after=evaluate_without_rng(model,evaluation_loader,device)
                contraction=norm;jump=after['loss']-before['loss'];phase='train3'
                emit_event('bypass_projection',{'seed':recipe.seed,'epoch':epoch0+offset,
                    'contraction_norm':norm,'validation_loss_jump':jump,'original_moments_reset':False},output)
        else:
            training=train_epoch(model,loader,optimizer,device);train3+=1
        scheduler.step();validation=evaluate_without_rng(model,evaluation_loader,device)
        if not all(math.isfinite(float(validation[key])) for key in ('accuracy','loss')):
            raise ValueError('Bypass validation is non-finite')
        eligible=phase=='train3'
        row={'method':METHOD,'seed':recipe.seed,'epoch':epoch0+offset,'post_fork_epoch':offset,
            'phase':epoch_phase,'phase_after_epoch':phase,'train_loss':training['task_loss'],
            'optimized_loss':training['loss'],'train_accuracy':training['accuracy'],
            'validation_accuracy':validation['accuracy'],'validation_loss':validation['loss'],
            'learning_rates':rates,'next_learning_rates':[group['lr'] for group in optimizer.param_groups],
            'gamma':gamma,'opt2_steps':steps,'contraction_norm':float(contraction_norm(model).detach()),
            'accuracy_comparison_eligible':eligible,'projection_loss_jump':jump,
            'rollback_applied':False,'retriggered':False}
        history.append(row);completed=offset
        improved=eligible and validation['accuracy']>best_accuracy
        if improved:
            best_accuracy=validation['accuracy']
            # Store a full original-space best checkpoint, never an expanded best.
            best_path=output/'best_checkpoint.pt'
            save_state(best_path,model=model,optimizer=optimizer,scheduler=scheduler,loader=loader,
                epoch=epoch0+offset,history=history,train_indices=train_ids,evaluation_indices=val_ids,
                source_tuning_indices=tuning_ids,trigger_indices=reserved_ids,run_protocol=fork['protocol'],
                kind='deit_bypass_original_best',run_identity=identity,validation=validation)
            best=torch.load(best_path,map_location='cpu',weights_only=False)
        row.update(report_best_accuracy=best_accuracy if eligible else None,
            report_best_loss=best['validation']['loss'] if best else None,
            report_best_epoch=best['epoch'] if best else None, report_best_improved=improved,
            historical_best_accuracy=accuracy,fork_epoch=epoch0,
            scientific_escape=bool(eligible and best_accuracy>accuracy),
            delta_vs_historical_best=best_accuracy-accuracy if eligible else None,
            opt1_epochs=opt1,opt2_epochs=opt2,train3_epochs=train3,bypass_completed=eligible)
        save();write_csv(history,output/'epoch_history.csv');emit_epoch(METHOD,row,device,started,output)
    return finish(torch.load(latest,map_location='cpu',weights_only=False),output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','plateau-checkpoint','plateau-checkpoint-hash','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--resume')
    parser.add_argument('--post-fork-epochs',type=int,default=150)
    parser.add_argument('--bypass-opt1-epochs',type=int,default=100)
    parser.add_argument('--bypass-max-opt2-epochs',type=int,default=50)
    parser.add_argument('--bypass-contraction-epsilon',type=float,default=.002)
    parser.add_argument('--bypass-gamma-slope',type=float,default=3e-6)
    args=parser.parse_args()
    if sha256_file(args.plateau_checkpoint)!=args.plateau_checkpoint_hash:raise ValueError('fork hash mismatch')
    fork,recipe=checked_source(args.plateau_checkpoint,{'deit_plateau_fork'})
    config=BypassConfig(args.bypass_opt1_epochs,args.bypass_max_opt2_epochs,args.bypass_contraction_epsilon,args.bypass_gamma_slope)
    run(args,fork,recipe,config)

if __name__=='__main__':main()
