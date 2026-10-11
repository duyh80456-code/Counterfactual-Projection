"""Precision separation, temporary generators, native diagnostics and resume."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from adapters.deit_mlp_growth import MLPExpansionCandidate
from adapters.deit_ablation import res18_cp_config
from adapters.deit_projection_aware import tensor_hash, projection_aware_search
from diagnostics.deit_expansion_proposals import auxiliary_initialization, gradient_tuned, random_structural
from diagnostics.deit_expansion_suite import ExpansionConfig, direction, gain, select_blocks, verify_baseline, run_suite, tree_equal, validate_indices


def test_fp64_metrics_do_not_recover_fp32_forward_rounding():
    class MLP(nn.Module):
        def __init__(self):
            super().__init__();self.fc1=nn.Linear(1,1);self.fc2=nn.Linear(1,2)
            with torch.no_grad():
                self.fc1.weight.zero_();self.fc1.bias.zero_();self.fc2.weight.zero_();self.fc2.bias.fill_(1e8)
        def forward(self,x):return self.fc2(torch.nn.functional.gelu(self.fc1(x)))
    class Model(nn.Module):
        def __init__(self):
            super().__init__();block=nn.Module();block.mlp=MLP();self.blocks=nn.ModuleList([block])
        def forward(self,x):return self.blocks[0].mlp(x)
    model=Model().eval();batch=(torch.ones(2,1),torch.zeros(2,dtype=torch.long))
    candidate=MLPExpansionCandidate('blocks.0.mlp',model,torch.ones(1,1),torch.zeros(1),torch.tensor([[.1],[0.]]),0.,{})
    low=direction(model,candidate,batch,.05,1.,'exact_expanded_forward')
    assert low['delta_f_norm']==0 and low['e_gain_fp64']==0
    high_model=copy.deepcopy(model).double()
    high=replace(candidate,model=high_model,A=candidate.A.double(),a=candidate.a.double(),B=candidate.B.double())
    precise=direction(high_model,high,(batch[0].double(),batch[1]),.05,1.,'exact_expanded_forward')
    assert precise['delta_f_norm']>0 and precise['e_gain_fp64']>0
    base=torch.zeros(1,2);expanded=torch.tensor([[1e-8,0.]])
    assert gain(base,expanded,torch.tensor([0]),torch.float32)==0
    assert gain(base,expanded,torch.tensor([0]),torch.float64)>0


def test_auxiliary_tuning_is_function_preserving_and_freezes_original(deit_small,deit_batches):
    model=deit_small.eval();site='blocks.0.mlp';before=tensor_hash(model.state_dict())
    params={name:(id(p),p.requires_grad,p.grad) for name,p in model.named_parameters()}
    initial=auxiliary_initialization(model,site,4,123)
    with torch.no_grad():
        base=model(deit_batches[0][0])
        with initial.virtual_direction(1.):assert torch.equal(base,model(deit_batches[0][0]))
    candidate,record=gradient_tuned(model,site,deit_batches[0],steps=10,learning_rate=.01,seed=123)
    assert record['tuning_gain']>0 and candidate.B.norm()>0
    assert record['proposal_backward_calls']==10 and record['proposal_forward_calls']==11
    assert tensor_hash(model.state_dict())==before
    assert params=={name:(id(p),p.requires_grad,p.grad) for name,p in model.named_parameters()}
    assert not any(module._forward_hooks for module in model.modules())


def test_random_branch_norm_match_is_structural_and_repeatable(deit_small,deit_batches):
    model=deit_small.eval();before=tensor_hash(model.state_dict());inputs=deit_batches[0][0]
    candidate,record=random_structural(model,'blocks.0.mlp',inputs,norm_target=.05,seed=77)
    second,again=random_structural(model,'blocks.0.mlp',inputs,norm_target=.05,seed=77)
    assert record['norm_match_converged'] and record['norm_actual']==pytest.approx(.05,rel=1e-3)
    assert torch.equal(candidate.B,second.B) and record==again
    assert tensor_hash(model.state_dict())==before
    zero,record=random_structural(model,'blocks.0.mlp',inputs,norm_target=0.,seed=77)
    assert record['degenerate_random_control'] and zero.B.count_nonzero()==0


def test_block_selection_and_baseline_validation():
    rows=[{'block_name':f'blocks.{i}.mlp','tiny_rank':i+1,'projection_residual':i/12,'e_gain':i*.01,'functional_delta_norm':1.} for i in range(12)]
    assert select_blocks(rows)==['blocks.0.mlp','blocks.6.mlp','blocks.1.mlp']
    assert len(select_blocks(rows,all_blocks=True))==12
    where={'site_evaluations':{row['block_name']:{'rank':row['tiny_rank'],'mean_e_gain':row['e_gain'],'mean_functional_delta_norm':1.} for row in rows}}
    assert verify_baseline(where,{'diagnostics':rows})['matched']
    rows[0]['e_gain']=1.
    with pytest.raises(ValueError,match='does not reproduce'):verify_baseline(where,{'diagnostics':rows})
    with pytest.raises(ValueError,match='overlap'):validate_indices({'statistics':[1],'where':[[2]],'projection':[3],'gate':[4],'opt_fit':[1],'opt_val':[6]})


@pytest.mark.parametrize('failed_rank',[False,True])
def test_native_short_suite_all_methods_same_fork_and_resume(deit_small,deit_batches,tmp_path,monkeypatch,failed_rank):
    import diagnostics.deit_expansion_suite as suite
    model=deit_small.eval();optimizer=torch.optim.AdamW(model.parameters())
    for p in model.parameters():p.grad=torch.zeros_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    before=tensor_hash(model.state_dict());old_optimizer=copy.deepcopy(optimizer.state_dict())
    batches={'statistics':deit_batches[:2],'where_batches':deit_batches[1:],
        'projection_batch':deit_batches[2],'gate_batch':deit_batches[3],
        'opt_fit':deit_batches[0],'opt_val':deit_batches[1]}
    indices={'statistics':list(range(16)),'where':[list(range(16+i*8,24+i*8)) for i in range(3)],
        'projection':list(range(40,48)),'gate':list(range(48,56)),'opt_fit':list(range(56,64)),'opt_val':list(range(64,72))}
    cp=replace(res18_cp_config(),cg_iterations=3,cg_preconditioner_probes=0)
    _,report,_=projection_aware_search(model,batches,cp,diagnostic_all=True,expected_sites=2)
    d0={'diagnostics':report['candidates']}
    if failed_rank:
        original=suite.DeitMLPGrowthAdapter.propose_auxiliary_growth
        def propose(self,**kwargs):
            if kwargs['rank']==2:raise RuntimeError('native Linear TINY produced no extension at '+kwargs['site'])
            return original(self,**kwargs)
        monkeypatch.setattr(suite.DeitMLPGrowthAdapter,'propose_auxiliary_growth',propose)
    config=ExpansionConfig(ranks=(2,4) if failed_rank else (4,),probe_epsilons=(.05,),expansion_scales=(1.,),auxiliary_steps=2)
    rows=run_suite(model,batches,cp,config,tmp_path,fork_hash='fixture',protocol={'seed':2},probe_indices=indices,
        d0=d0,expected_sites=2,optimizer=optimizer)
    assert len(rows)==(8 if failed_rank else 6) and {row['proposal_method'] for row in rows}=={'tiny_original','aux_gradient_tuned','aux_random_direction'}
    assert tensor_hash(model.state_dict())==before and tree_equal(old_optimizer,optimizer.state_dict())
    for row in rows:
        if row['solver_status']=='not_run_no_E_proposal':
            assert failed_rank and row['rank']==2 and not row['accepted']
            continue
        assert all(name.startswith(row['block_name']+'.') for name in row['projection_parameter_names'])
        assert row['e_gain_forward_fp64'] is not None
        assert row['accepted']==(row['gate_gain']>0 and row['selected_scale'] is not None and not row['degenerate_target'])
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())
    assert (tmp_path/'probe_batches.pt').exists()
    summary=json.loads((tmp_path/'summary.json').read_text());assert summary['training_epochs']==0 and summary['committed_corrections']==0
    monkeypatch.setattr(suite,'evaluate_transfer',lambda *_a:pytest.fail('resume repeated a completed solve'))
    repeated=run_suite(model,batches,cp,config,tmp_path,fork_hash='fixture',protocol={'seed':2},probe_indices=indices,
        d0=d0,expected_sites=2,optimizer=optimizer)
    assert rows==repeated


def test_diagnostic_archive_inputs_and_progress_are_hash_bound(tmp_path):
    import tarfile
    from dataclasses import asdict
    from experiments.deit_protocol import DeitRecipe, protocol, canonical_model_config
    from experiments.deit_resume import FULL_STATE
    from experiments.run_deit_expansion_diagnostics import prepare_inputs, import_progress
    from experiments.shared_protocol import sha256_file
    recipe=replace(DeitRecipe(),seed=2,batch_size=64,schedule_epochs=300,max_epoch=800)
    declared=protocol(recipe,canonical_model_config());cp=res18_cp_config();config=ExpansionConfig()
    fork={**{key:{} for key in FULL_STATE},'kind':'deit_plateau_fork','protocol':declared,'epoch':375,
        'historical_best_epoch':375,'historical_best_accuracy':.562,'historical_best_loss':3.,
        'history':[{'epoch':375,'validation_accuracy':.562,'validation_loss':3.}]}
    inputs=tmp_path/'input';inputs.mkdir();source=tmp_path/'fork.pt';torch.save(fork,source);h=sha256_file(source)
    d0={'method':'all12_transfer_diagnostic','protocol':declared,
        'run_identity':{'fork_hash':h,'cp_config':json.loads(json.dumps(asdict(cp)))},
        'diagnostics':[{'block_name':f'blocks.{i}.mlp'} for i in range(12)]}
    progress={'kind':'deit_expansion_diagnostic_progress','run_identity':{'fork_hash':h,'protocol':declared,
        'cp_config':json.loads(json.dumps(asdict(cp))),'expansion_config':json.loads(json.dumps(asdict(config)))},
        'rows':[{},{}],'proposals':{},'proposal_budget':{}}
    saved=tmp_path/'progress.pt';torch.save(progress,saved)
    d0_path=tmp_path/'result.json';d0_path.write_text(json.dumps(d0))
    with tarfile.open(inputs/'output.tar.gz','w:gz') as archive:
        archive.add(source,arcname='run/renamed_fork.pt');archive.add(saved,arcname='run/renamed_progress.pt')
        archive.add(d0_path,arcname='run/d0_export/result.json')
    output=tmp_path/'output'
    path,loaded,actual,prior=prepare_inputs([inputs],output)
    assert sha256_file(path)==h and prior==d0 and actual==recipe
    imported=import_progress([inputs],output,h,declared,config,cp)
    assert sha256_file(imported)==sha256_file(saved)
    with pytest.raises(ValueError,match='config mismatch'):
        import_progress([inputs],tmp_path/'wrong',h,declared,replace(config,all_blocks=True),cp)
    bad=tmp_path/'bad.json';bad.write_text(json.dumps({**d0,'run_identity':{**d0['run_identity'],'fork_hash':'wrong'}}))
    with pytest.raises(ValueError,match='another fork'):prepare_inputs([inputs],tmp_path/'badoutput',d0_path=bad)


def test_diagnostic_notebook_has_no_long_training_arm():
    notebook=json.loads(Path('notebooks/kaggle_deit_tiny_seed2_expansion_diagnostics.ipynb').read_text())
    code='\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code')
    assert 'experiments.run_deit_expansion_diagnostics' in code
    assert 'experiments.run_deit_projection_aware_suite' not in code
    assert 'experiments.run_deit_all_arms' not in code
    assert 'AUXILIARY_STEPS = 20' in code
    assert 'ALL_BLOCKS = False' in code
