"""GELU paper extension, optimizer preservation, phase resume and eligibility."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from adapters.deit_bypass import BypassConfig, embed, extensions, add_to_optimizer, contract, contraction_norm
from experiments.deit_protocol import DeitRecipe, build_optimizer_scheduler, protocol, save_state
from experiments.shared_protocol import restore_rng, sha256_file
from models import DeiTTinyCifar
from tests.test_deit_e_rollback import assert_equal


def populated_optimizer(model,recipe):
    optimizer,scheduler=build_optimizer_scheduler(model,recipe)
    for p in model.parameters():p.grad=torch.ones_like(p)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    return optimizer,scheduler


def test_all12_gelu_sites_are_function_preserving():
    model=DeiTTinyCifar(num_classes=3,image_size=8,patch_size=4,embed_dim=12,
        depth=12,num_heads=3,mlp_ratio=2).double().eval()
    x=torch.randn(2,3,8,8,dtype=torch.float64)
    before=model(x).detach();parameters={name:id(p) for name,p in model.named_parameters()}
    assert len(embed(model))==12
    assert torch.equal(model(x),before)
    assert sum(module.d.numel() for module in extensions(model))==12*24
    assert all(id(dict(model.named_parameters())[name])==identity for name,identity in parameters.items())


def test_gelu_embedding_and_contraction_preserve_original_moments(deit_small,deit_batches):
    model=deit_small.eval();recipe=DeitRecipe()
    optimizer,scheduler=populated_optimizer(model,recipe)
    params=dict(model.named_parameters());states={name:copy.deepcopy(optimizer.state[p]) for name,p in params.items()}
    x=deit_batches[0][0];before=model(x).detach()
    paths=embed(model,expected_sites=2);assert len(paths)==2
    assert torch.equal(model(x),before)
    assert all(module.d.count_nonzero()==0 for module in extensions(model))
    add_to_optimizer(model,optimizer)
    assert len(optimizer.param_groups)==2
    assert all(any(p is module.d for p in optimizer.param_groups[1]['params']) for module in extensions(model))
    for module in extensions(model):module.d.grad=torch.ones_like(module.d)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert float(contraction_norm(model))>0
    assert not contract(model,optimizer,1e-20)
    for name,p in params.items():assert_equal(states[name],optimizer.state[p])
    for module in extensions(model):
        with torch.no_grad():module.d.zero_()
    extended=model(x).detach()
    assert contract(model,optimizer,.002)
    assert not extensions(model) and torch.equal(model(x),extended)
    assert set(dict(model.named_parameters()))==set(params)
    for name,p in model.named_parameters():
        assert p is params[name];assert_equal(states[name],optimizer.state[p])
    scheduler.step() # no extra param groups or scheduler reset


@pytest.mark.parametrize('epsilon,completed',[(1e6,True),(1e-20,False)])
def test_bypass_main_loop_full_phase_resume(deit_small,deit_batches,tmp_path,monkeypatch,epsilon,completed):
    import experiments.run_deit_bypass as runner
    model=deit_small;recipe=replace(DeitRecipe(),seed=3,workers=0,batch_size=8)
    optimizer,scheduler=populated_optimizer(model,recipe)
    data=TensorDataset(*deit_batches[0]);loader=DataLoader(data,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(3))
    fork_path=tmp_path/'fork.pt';history=[{'epoch':7,'validation_accuracy':.5,'validation_loss':2.}]
    save_state(fork_path,model=model,optimizer=optimizer,scheduler=scheduler,loader=loader,epoch=7,
        history=history,train_indices=list(range(8)),evaluation_indices=[],source_tuning_indices=[],trigger_indices=[],
        run_protocol=protocol(recipe,model),kind='deit_plateau_fork',historical_best_accuracy=.5,
        historical_best_loss=2.,historical_best_epoch=7)
    fork=torch.load(fork_path,weights_only=False)
    def context(_root,_recipe,device,source):
        current=DeiTTinyCifar(**model.config).double();opt,sched=build_optimizer_scheduler(current,recipe)
        current.load_state_dict(source['model']);opt.load_state_dict(copy.deepcopy(source['optimizer']));sched.load_state_dict(copy.deepcopy(source['scheduler']))
        train=DataLoader(data,batch_size=8,shuffle=True,generator=torch.Generator().set_state(source['train_loader_generator_state']))
        restore_rng(source['rng'])
        return current,opt,sched,train,loader,data,list(range(8)),[],[],[]
    monkeypatch.setattr(runner,'load_training_context',context)
    monkeypatch.setattr(runner,'evaluate_without_rng',lambda *_a:{'accuracy':.6,'loss':1.9})
    config=BypassConfig(2,4,epsilon,3e-6)
    args=SimpleNamespace(output=tmp_path/'arm',plateau_checkpoint=fork_path,data_root='fixture',device='cpu',post_fork_epochs=6,resume=None)
    original=runner.save_state;snapshots={}
    def capture(path,**kwargs):
        original(path,**kwargs)
        if kwargs['kind']==runner.KIND and kwargs['completed_epochs'] in (1,2,3,4):
            checkpoint=tmp_path/f"snapshot{kwargs['completed_epochs']}.pt"
            torch.save(torch.load(path,weights_only=False),checkpoint);snapshots[kwargs['completed_epochs']]=checkpoint
    monkeypatch.setattr(runner,'save_state',capture)
    result=runner.run(args,fork,recipe,config,expected_sites=2)
    final=torch.load(args.output/'checkpoint_latest.pt',weights_only=False)
    assert result['bypass_completed']==completed
    assert result['accuracy_comparison_eligible']==completed
    assert result['report_best_accuracy']==(.6 if completed else None)
    assert result['best_expanded_or_original_diagnostic']['validation_accuracy']==.6
    assert final['opt1_done']==2 and final['completed_epochs']==6
    assert bool(any(name.endswith('.act.d') for name in final['model']))== (not completed)
    if completed:
        assert result['train3_epochs']==3 and result['opt2_epochs']==1
        assert result['report_best_epoch']==10 # ties retain first original-space best
    else:
        assert result['train3_epochs']==0 and result['opt2_epochs']==4
        assert not (args.output/'best_checkpoint.pt').exists()
    # Freeze captured snapshots: resumes also save progress to the same working output.
    paths=list(snapshots.values())
    monkeypatch.setattr(runner,'save_state',original)
    for path in paths:
        args.resume=path;runner.run(args,fork,recipe,config,expected_sites=2)
        resumed=torch.load(args.output/'checkpoint_latest.pt',weights_only=False)
        for key in ('model','optimizer','scheduler','rng','train_loader_generator_state','history','phase','opt2_steps','best_original'):
            assert_equal(final[key],resumed[key])
    args.resume=args.output/'checkpoint_latest.pt'
    monkeypatch.setattr(runner,'load_training_context',lambda *_a:pytest.fail('completed Bypass rebuilt model'))
    assert runner.run(args,fork,recipe,config,expected_sites=2)['bypass_completed']==completed
    # Mounted output archives preserve original bytes and skip completed training.
    import tarfile
    from experiments.deit_bypass_resume import prepare_bypass_resume
    attachment=tmp_path/'attachment';attachment.mkdir()
    source=args.output/'checkpoint_latest.pt'
    with tarfile.open(attachment/'output.tar.gz','w:gz') as archive:
        archive.add(source,arcname='run/arms/deit_bypass/renamed.pt')
    imported=tmp_path/'imported'
    plan=prepare_bypass_resume([attachment],imported,fork_path,fork,config,6)
    assert plan['action']=='completed'
    assert sha256_file(source)==sha256_file(imported/'arms/deit_bypass/checkpoint_latest.pt')
    with pytest.raises(ValueError,match='fork/config mismatch'):
        prepare_bypass_resume([attachment],tmp_path/'wrong',fork_path,fork,replace(config,gamma_slope=1e-5),6)


def test_bypass_comparison_never_uses_expanded_accuracy(tmp_path):
    from experiments.deit_bypass_comparison import compare
    declared={'seed':3};h='fork'
    for method,accuracy,eligible in [('e_driven_o_raw',.55,True),('deit_bypass',None,False)]:
        path=tmp_path/'arms'/method;path.mkdir(parents=True)
        result={'method':method,'protocol':declared,'theta_best_hash':h,'run_identity':{'post_fork_epochs':150},
            'history':[{'post_fork_epoch':150}],'completed_epochs':150,'report_best_accuracy':accuracy,
            'report_best_loss':2. if eligible else None,'report_best_epoch':200 if eligible else None,
            'delta_vs_historical_best':.05 if eligible else None,'scientific_escape':eligible,
            'accuracy_comparison_eligible':eligible,'best_expanded_or_original_diagnostic':{'validation_accuracy':.99}}
        (path/'result.json').write_text(json.dumps(result))
    result=compare(tmp_path,h,declared,150)
    assert result['e2o_minus_bypass_best_accuracy'] is None and not result['comparison_eligible']
    with pytest.raises(ValueError,match='identical fork'):compare(tmp_path,'other',declared,150)


def test_new_seed_notebooks_are_isolated_and_only_raw_vs_bypass():
    for seed in (3,4,5):
        notebook=json.loads(open(f'notebooks/kaggle_deit_tiny_seed{seed}_raw_vs_bypass.ipynb').read())
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        assert f'SEED = {seed}' in code
        assert "ARMS = ('e_driven_o_raw', 'deit_bypass')" in code
        assert "'--arms', ','.join(ARMS)" in code
        assert "'--bypass-opt1-epochs'" in code
        assert 'tests/test_deit_bypass.py' in code


def test_suite_dispatches_only_raw_and_bypass_with_matching_budget(tmp_path,monkeypatch):
    import sys
    from dataclasses import asdict
    import experiments.run_deit_all_arms as suite
    import experiments.deit_resume as resume
    import experiments.deit_bypass_resume as bypass_resume
    import experiments.deit_bypass_comparison as comparison
    from adapters.deit_ablation import res18_cp_config,SUITE_VERSION
    from experiments.deit_protocol import canonical_model_config
    recipe=DeitRecipe(seed=3,batch_size=64,schedule_epochs=300,max_epoch=800,reference_epochs=150)
    fork={'epoch':37,'historical_best_accuracy':.55,'protocol':protocol(recipe,canonical_model_config())}
    phase1=tmp_path/'vanilla_stall';phase1.mkdir()
    fork_path=phase1/'plateau_checkpoint.pt';torch.save(fork,fork_path);h=sha256_file(fork_path)
    torch.save({'fork_hash':h,'cp_config':asdict(res18_cp_config()),'suite_version':SUITE_VERSION,'seed':3},tmp_path/'raw_intervention_reference.pt')
    monkeypatch.setattr(suite,'checked_source',lambda *_a:(fork,recipe))
    monkeypatch.setattr(resume,'prepare_resume',lambda *_a,**_kw:{'arms':{}})
    monkeypatch.setattr(bypass_resume,'prepare_bypass_resume',lambda *_a,**_kw:{'action':'start'})
    monkeypatch.setattr(comparison,'compare',lambda *_a:None)
    def jobs(commands,*_a):
        assert set(commands)=={'e_driven_o_raw','deit_bypass'}
        for method,command in commands.items():
            assert command[command.index('--post-fork-epochs')+1]=='150'
            assert command[command.index('--plateau-checkpoint-hash')+1]==h
            if method=='deit_bypass':
                assert 'experiments.run_deit_bypass' in command
                assert command[command.index('--bypass-opt1-epochs')+1]=='100'
                assert command[command.index('--bypass-max-opt2-epochs')+1]=='50'
                assert '--raw-reference' not in command
            else:assert '--raw-reference' in command
            directory=tmp_path/'arms'/method;directory.mkdir(parents=True)
            (directory/'result.json').write_text(json.dumps({'theta_best_hash':h,'report_best_accuracy':.56,
                'report_best_loss':2.,'delta_vs_historical_best':.01,'scientific_escape':True}))
        return {method:{'status':'completed'} for method in commands}
    monkeypatch.setattr(suite,'run_jobs',jobs)
    monkeypatch.setattr(sys,'argv',['suite','--data-root','fixture','--output',str(tmp_path),'--seed','3',
        '--device','cpu','--arms','e_driven_o_raw,deit_bypass'])
    suite.main()
