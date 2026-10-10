"""Read-only all-site geometry, gate selection, and recurrent/resume main loops."""
from dataclasses import replace
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from adapters.deit_ablation import res18_cp_config
from adapters.deit_projection_aware import projection_aware_search, choose_best_gate, tensor_hash, commit_choice
from experiments.deit_protocol import DeitRecipe, protocol, build_optimizer_scheduler, save_state
from experiments.shared_protocol import restore_rng
from models import DeiTTinyCifar


def test_gate_selector_rejects_top1_and_uses_positive_top3():
    rows = [{'record': {'tiny_rank': rank, 'gate_gain': gain, 'accepted': accepted}}
            for rank, gain, accepted in [(1,-.2,False),(2,.1,True),(3,.3,True)]]
    assert choose_best_gate(rows) is rows[2]
    rows[1]['record']['gate_gain'] = .3
    assert choose_best_gate(rows) is rows[1]  # stable TINY rank tie-break
    for row in rows: row['record']['accepted'] = False
    assert choose_best_gate(rows) is None


def test_all12_native_tiny_and_same_block_projection_without_commit(deit_batches):
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        torch.manual_seed(123)
        model = DeiTTinyCifar(num_classes=3,image_size=8,patch_size=4,embed_dim=12,
            depth=12,num_heads=3,mlp_ratio=2).double().eval()
        cfg = replace(res18_cp_config(),rank=2,cg_iterations=4,cg_preconditioner_probes=0)
        batches = {'statistics': deit_batches[:2], 'where_batches': deit_batches[1:],
            'projection_batch': deit_batches[2], 'gate_batch': deit_batches[3]}
        before = tensor_hash(model.state_dict()); params = {name:id(p) for name,p in model.named_parameters()}
        chosen, report, _ = projection_aware_search(model,batches,cfg,diagnostic_all=True)
        assert len(report['candidates']) == 12
        assert sorted(row['tiny_rank'] for row in report['candidates']) == list(range(1,13))
        assert {row['block_name'] for row in report['candidates']} == {f'blocks.{i}.mlp' for i in range(12)}
        for row in report['candidates']:
            assert row['gradient_damping_used'] == row['damping_used']
            assert all(name.startswith(row['block_name']+'.') for name in row['projection_parameter_names'])
        assert tensor_hash(model.state_dict()) == before
        assert params == {name:id(p) for name,p in model.named_parameters()}
        assert not any(module._forward_hooks for module in model.modules())
        assert commit_choice(model, torch.optim.AdamW(model.parameters()), None)['momentum_states_reset'] == 0
    finally:
        torch.set_num_threads(previous)


def assert_state_equal(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_state_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_state_equal(x,y)
    else:assert a==b


@pytest.mark.parametrize('solver_failed', [False, True])
def test_top3_previews_same_anchor_then_commits_only_valid_scope(deit_small,deit_batches,monkeypatch,solver_failed):
    import adapters.deit_projection_aware as search
    model=deit_small
    optimizer=torch.optim.AdamW(model.parameters())
    for parameter in model.parameters():parameter.grad=torch.zeros_like(parameter)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    ids={name:id(p) for name,p in model.named_parameters()}
    anchor=tensor_hash(model.state_dict()); previews=[]
    proposals=[{'candidate':SimpleNamespace(module_name=f'blocks.{i%2}.mlp',index=i),
                'site':f'blocks.{i%2}.mlp','rank':i+1,'mean_e_gain':3-i,'mean_delta_f_norm':1.}
               for i in range(3)]
    monkeypatch.setattr(search,'select_by_expansion_gain',lambda *_a,**_kw:(None,{},proposals))
    class Probe:
        def __call__(self,_model,*,candidate,batch,gate):
            return SimpleNamespace(delta_logits=torch.full((8,3),float(candidate.index>0),dtype=torch.float64))
    monkeypatch.setattr(search,'CandidateExpansionProbe',Probe)
    def project(current,site,fit,*_a):
        assert tensor_hash(current.state_dict())==anchor
        previews.append(site)
        index=len(previews)-1
        if index==2 and solver_failed:
            raise RuntimeError('all projection attempts produced non-finite solutions')
        names=search.DeitMLPGrowthAdapter.original_mlp_parameters(model,site)
        delta={name:torch.full_like(dict(model.named_parameters())[name],.01) for name in names}
        projection=SimpleNamespace(parameter_delta=delta,relative_residual=.1,cosine_alignment=.9,
            damping_used=.001,cg=SimpleNamespace(converged=True,relative_residual=.001))
        return projection,{'line_search_gains':{'.05':float('inf') if index==2 else .2},
            'correction_applied':True,'selected_scale':.05,'projection_parameter_names':list(names)}
    monkeypatch.setattr(search,'project_target',project)
    batches={'statistics':[],'where_batches':[], 'projection_batch':deit_batches[0],'gate_batch':deit_batches[1]}
    chosen,report,_=search.projection_aware_search(model,batches,res18_cp_config(),expected_sites=3)
    assert len(previews)==3 and tensor_hash(model.state_dict())==anchor
    assert report['candidates'][0]['degenerate_target'] and not report['candidates'][0]['accepted']
    assert not report['candidates'][2]['accepted']
    assert chosen['record']['tiny_rank']==2
    record=commit_choice(model,optimizer,chosen)
    assert set(record['projection_changed_parameters'])==set(record['adam_moments_reset_parameters'])
    assert len(record['projection_changed_parameters'])==4
    for name,p in model.named_parameters():
        assert id(p)==ids[name]
        if name.startswith('blocks.1.mlp.'):
            assert not torch.equal(p,before[name])
            assert optimizer.state[p]['exp_avg'].count_nonzero()==0
        else:assert torch.equal(p,before[name])
    assert not any(module._forward_hooks for module in model.modules())


@pytest.mark.parametrize('method,rounds', [('e2o_top3_best_gate',1),('e2o_top3_recurrent',3)])
@pytest.mark.parametrize('accept', [False, True])
def test_main_loop_rollbacks_and_pending_resume(deit_small,deit_batches,tmp_path,monkeypatch,method,rounds,accept):
    import experiments.run_deit_projection_aware as runner
    model = deit_small
    recipe=replace(DeitRecipe(), seed=2, workers=0,batch_size=8)
    optimizer,scheduler=build_optimizer_scheduler(model,recipe)
    # A real epoch375 fork has populated Adam state, unlike a cold optimizer.
    for parameter in model.parameters(): parameter.grad=torch.zeros_like(parameter)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    data=TensorDataset(*deit_batches[0]); loader=DataLoader(data,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(2))
    history=[{'epoch':375,'validation_accuracy':.562,'validation_loss':3.1}]
    fork_path=tmp_path/'fork.pt'
    save_state(fork_path,model=model,optimizer=optimizer,scheduler=scheduler,loader=loader,epoch=375,
        history=history,train_indices=list(range(8)),evaluation_indices=[],source_tuning_indices=[],trigger_indices=[],
        run_protocol=protocol(recipe,model),kind='deit_plateau_fork',historical_best_accuracy=.562,
        historical_best_loss=3.1,historical_best_epoch=375)
    fork=torch.load(fork_path,weights_only=False)
    def context(_root,_recipe,device,source):
        current=DeiTTinyCifar(**model.config).double(); opt,sched=build_optimizer_scheduler(current,recipe)
        current.load_state_dict(source['model']);opt.load_state_dict(copy.deepcopy(source['optimizer']));sched.load_state_dict(copy.deepcopy(source['scheduler']))
        train=DataLoader(data,batch_size=8,shuffle=True,generator=torch.Generator().set_state(source['train_loader_generator_state']))
        restore_rng(source['rng'])
        return current,opt,sched,train,loader,data,list(range(8)),[],[],[]
    monkeypatch.setattr(runner,'load_training_context',context)
    batches={'projection_batch':deit_batches[0]}
    tokens=[]; hashes=[]
    def materialize(*_a,**kwargs):
        token=kwargs.get('probe_index',0);tokens.append(token)
        return batches,{'projection':[1,2],'token':token}
    monkeypatch.setattr(runner,'materialize_probe_batches',materialize)
    def search(current,*_a,**_kw):
        hashes.append(tensor_hash(current.state_dict()))
        chosen=None; direction=None
        if accept:
            delta={name:torch.full_like(p, .01) for name,p in current.named_parameters()
                   if name.startswith('blocks.0.mlp.')}
            chosen={'projection':SimpleNamespace(parameter_delta=delta), 'record':{'selected_scale':.05}}
            direction=torch.ones_like(deit_batches[0][0])
        return chosen,{'candidates':[{'accepted':accept,'gate_gain':.1 if accept else -.1}]*3,'candidate_blocks':['a','b','c'],
            'selected_block':'blocks.0.mlp' if accept else None,'selected_gate_gain':.1 if accept else None,
            'accepted_count':3 if accept else 0,'rejected_count':0 if accept else 3,
            'no_valid_intervention':not accept,'anchor_hash':hashes[-1],'search_seconds':0.,'where':{}},direction
    monkeypatch.setattr(runner,'projection_aware_search',search)
    def evaluate(current,*_a):
        return {'accuracy':.562,'loss':3.1} if tensor_hash(current.state_dict())==tensor_hash(fork['model']) else {'accuracy':.55,'loss':3.2}
    monkeypatch.setattr(runner,'evaluate_without_rng',evaluate)
    args=SimpleNamespace(output=tmp_path/method,plateau_checkpoint=fork_path,method=method,
        post_fork_epochs=25,algorithm_patience=10,resume=None,data_root='fixture',device='cpu')
    pending_path=tmp_path/'pending.pt'; midpoint_path=tmp_path/'midpoint.pt'
    original_save=runner.save_state
    def capture(path,**kwargs):
        original_save(path,**kwargs)
        if kwargs['completed_epochs']==10 and kwargs['pending_search']:
            torch.save(torch.load(path,weights_only=False),pending_path)
        if kwargs['completed_epochs']==11 and not kwargs['pending_search']:
            torch.save(torch.load(path,weights_only=False),midpoint_path)
    monkeypatch.setattr(runner,'save_state',capture)
    cfg=res18_cp_config()
    result=runner.run(args,fork,recipe,cfg,expected_sites=2)
    final=torch.load(args.output/'checkpoint_latest.pt',weights_only=False)
    assert_state_equal(final['e_controller']['anchor']['optimizer'],fork['optimizer'])
    assert result['search_rounds']==rounds and result['corrections_accepted']==rounds*int(accept)
    for event in result['interventions']:
        changed=event['projection_changed_parameters']
        assert set(changed)==set(event['adam_moments_reset_parameters'])
        assert all(name.startswith('blocks.0.mlp.') for name in changed)
        assert len(changed)==(4 if accept else 0)
    if accept and rounds>1:
        assert result['interventions'][1]['functional_delta_cosine_to_previous']==pytest.approx(1.)
        assert result['interventions'][1]['functional_delta_difference_norm']==0.
    assert result['proposals_tried']==rounds*3 and len(result['rollback_events'])==2
    assert result['interventions'][0]['epochs_until_rollback']==10
    assert result['best_postfork_observed']['accuracy']==.55
    assert result['best_anchor_stored']['accuracy']==.562
    assert hashes==[tensor_hash(fork['model'])]*rounds
    if rounds>1:
        assert len(set(tokens[1:]))==rounds-1
        assert all(token > 0 for token in tokens[1:])
    args.resume=midpoint_path
    runner.run(args,fork,recipe,cfg,expected_sites=2)
    resumed=torch.load(args.output/'checkpoint_latest.pt',weights_only=False)
    for key in ['model','optimizer','scheduler','rng','train_loader_generator_state','history','interventions','e_controller']:
        assert_state_equal(final[key],resumed[key])
    if method=='e2o_top3_recurrent':
        args.resume=pending_path;runner.run(args,fork,recipe,cfg,expected_sites=2)
        resumed=torch.load(args.output/'checkpoint_latest.pt',weights_only=False)
        for key in ['model','optimizer','scheduler','rng','train_loader_generator_state','history','interventions','e_controller']:
            assert_state_equal(final[key],resumed[key])
    args.resume=args.output/'checkpoint_latest.pt'
    monkeypatch.setattr(runner,'load_training_context',lambda *_a: pytest.fail('completed resume rebuilt model'))
    assert runner.run(args,fork,recipe,cfg,expected_sites=2)['search_rounds']==rounds


def test_seed2_archive_discovery_and_readonly_comparison(tmp_path):
    import tarfile
    import zipfile
    from experiments.deit_protocol import canonical_model_config
    from experiments.deit_resume import FULL_STATE
    from experiments.deit_projection_resume import prepare_projection_resume, comparison_table
    from experiments.run_deit_projection_aware import identity_for
    from experiments.shared_protocol import sha256_file
    recipe=replace(DeitRecipe(),seed=2,batch_size=64,max_epoch=800,schedule_epochs=300)
    fork={**{key:{} for key in FULL_STATE},'kind':'deit_plateau_fork',
        'protocol':protocol(recipe,canonical_model_config()),'epoch':375,
        'historical_best_epoch':375,'historical_best_accuracy':.562,'historical_best_loss':3.1,
        'history':[{'epoch':375,'validation_accuracy':.562,'validation_loss':3.1}]}
    source=tmp_path/'original.pt';torch.save(fork,source)
    inputs=tmp_path/'input';inputs.mkdir();output=tmp_path/'out'
    with tarfile.open(inputs/'run.tar.gz','w:gz') as archive:archive.add(source,arcname='seed2/renamed.pt')
    plan=prepare_projection_resume([inputs],output)
    assert plan['fork_hash']==sha256_file(source)
    assert all(row['action']=='start' for row in plan['arms'].values())
    def result(method,accuracy,fork_hash=plan['fork_hash']):
        return {'method':method,'protocol':fork['protocol'],'theta_best_hash':fork_hash,
            'run_identity':identity_for(fork_hash,method,res18_cp_config(),150,10),
            'completed_epochs':150,'report_best_accuracy':accuracy,'report_best_loss':3.,'report_best_epoch':400}
    with zipfile.ZipFile(inputs/'old_results.zip','w') as archive:
        archive.writestr('raw/result.json',json.dumps(result('e_driven_o_raw',.56)))
        archive.writestr('other/result.json',json.dumps(result('o_projection_only_rollback',.9,'wrong')))
    method='e2o_top3_best_gate';(output/method).mkdir()
    (output/method/'result.json').write_text(json.dumps(result(method,.563)))
    rows=comparison_table([inputs],output,plan['fork_hash'])
    assert sum(row['available'] for row in rows)==2
    report=json.loads((output/'comparison.json').read_text())
    assert report['effects']['selection']==pytest.approx(.003)
    assert report['effects']['structural_A3'] is None
    assert len(report['excluded'])==1
    # Matching but distinct old trajectories are excluded, never picked by score.
    (inputs/'result.json').write_text(json.dumps(result('e_driven_o_raw',.57)))
    rows=comparison_table([inputs],output,plan['fork_hash'])
    assert not next(row for row in rows if row['method']=='e_driven_o_raw')['available']
    from experiments.deit_e_rollback import rollback_protocol
    recurrent='e2o_top3_recurrent'
    pending={**fork,'kind':'deit_projection_aware_latest',
        'run_identity':identity_for(plan['fork_hash'],recurrent,res18_cp_config(),150,10),
        'completed_epochs':10,'epoch':385,'history':[{}]*11,'search_count':1,
        'interventions':[{}],'pending_search':True,'e_controller':{
            'protocol':rollback_protocol(10,stall_on_anchor=True),'anchor':{'post_fork_epoch':0},
            'accuracy_stall_counter':0,'rollback_events':[]}}
    torch.save(pending,inputs/'pending.pt')
    committed={**pending,'search_count':2,'interventions':[{},{}],'pending_search':False}
    torch.save(committed,inputs/'committed.pt')
    plan2=prepare_projection_resume([inputs],output)
    imported=torch.load(output/recurrent/'checkpoint_latest.pt',weights_only=False)
    assert imported['search_count']==2 and not imported['pending_search']
    assert plan2['arms'][recurrent]['action']=='resume'
    with pytest.raises(FileNotFoundError,match='Attach seed2'):
        prepare_projection_resume([],tmp_path/'missing')
