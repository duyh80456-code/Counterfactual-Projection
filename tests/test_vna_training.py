"""Stage 2a CPU trajectories and notebook/restore contracts."""
import ast
import json
from pathlib import Path
import pytest
import numpy as np
import torch
from experiments.run_vna_training import parser, run, training_protocol


def arguments(output, *extra):
    return parser().parse_args(['--fake-data','--device','cpu','--output',str(output),
        '--epochs','2','--limit-batches','1','--vna-every','1','--vna-batch','3',
        '--stat-batch','8','--stat-batches','1','--refresh-epochs','1','--rank','2','--hutchinson-probes','2',
        '--cg-iters','100','--cg-tol','1e-4',*extra])


def load(path): return torch.load(path,map_location='cpu',weights_only=False)


def equal(a,b):
    if torch.is_tensor(a): assert torch.equal(a,b)
    elif isinstance(a,np.ndarray): assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    else: assert a==b


@pytest.mark.parametrize('solver',['exact','kfac','lite'])
def test_training_resume_bitwise(tmp_path,solver):
    full=load(run(arguments(tmp_path/'full','--arm','vna','--solver',solver)))
    first=run(arguments(tmp_path/'resume','--arm','vna','--solver',solver,'--max-epochs','1'))
    resumed=load(run(arguments(tmp_path/'resume','--arm','vna','--solver',solver,'--resume',str(first))))
    for key in ('model','optimizer','scheduler','rng','train_loader_generator_state','numpy_rng','vna','corrections','step'):
        equal(full[key],resumed[key])
    assert full['final']['params']==resumed['final']['params']
    assert full['final']['test']==resumed['final']['test']
    assert full['final']['test_evaluations']==1
    assert [p.name for p in (tmp_path/'resume').glob('*.pt')]==['latest.pt']
    # A finished resume cannot touch the test set.
    import experiments.run_vna_training as runner
    from unittest.mock import patch
    with patch.object(runner,'official_test',side_effect=AssertionError('second test evaluation')):
        run(arguments(tmp_path/'resume','--arm','vna','--solver',solver,'--resume',str(first)))


@pytest.mark.parametrize('solver',['exact','kfac','lite'])
def test_training_lambda_zero_sgd_identity_and_params(tmp_path,solver):
    sgd=load(run(arguments(tmp_path/'sgd')))
    vna=load(run(arguments(tmp_path/'vna','--arm','vna','--solver',solver,'--lambda-vna','0')))
    for key in ('model','optimizer','scheduler','train_loader_generator_state','rng'):
        equal(sgd[key],vna[key])
    assert sgd['final']['params']==vna['final']['params']
    for a,b in zip(sgd['history'],vna['history']):
        assert a['train']==b['train'] and a['eval']==b['eval']
    assert len(vna['corrections'])==2


@pytest.mark.parametrize('solver',['exact','kfac','lite'])
def test_training_ngd_norm_matching(tmp_path,solver):
    cp=load(run(arguments(tmp_path,'--arm','ngd','--solver',solver)))
    assert cp['corrections']
    for row in cp['corrections']:
        assert row['correction_norm']==pytest.approx(row['vna_norm'],rel=1e-10)
        assert row['state']=='active'
    assert cp['final']['params']==cp['final']['sgd_params']


def test_training_protocol_and_scaled_schedule(tmp_path):
    args=arguments(tmp_path,'--arm','sgd_cm')
    args.epochs=200
    protocol=training_protocol(args)
    assert protocol['total_epochs']==260 and protocol['milestones']==[130,195]
    assert parser().parse_args(['--arm','sgd_cm']).max_epochs>=260
    other=arguments(tmp_path,'--lambda-vna','0')
    run(arguments(tmp_path,'--max-epochs','1'))
    with pytest.raises(ValueError,match='protocol mismatch'):
        run(arguments(tmp_path,'--resume',str(tmp_path/'latest.pt'),'--lambda-vna','0'))


def test_training_no_forbidden_paths():
    root=Path(__file__).resolve().parents[1]
    for file in ['experiments/run_vna_training.py','experiments/vna_training_support.py',
                 'experiments/vna_training_kaggle.py','scripts/build_kaggle_vna_training_notebooks.py']:
        source=(root/file).read_text()
        for forbidden in ['.apply_(', 'reset_projected_momentum','plateau_checkpoint','controller_anchor_',
                          'vanilla_rollback','report_best','scientific_escape','select_by_expansion_gain']:
            assert forbidden not in source


def test_training_notebooks_and_solver_decision():
    from scripts.build_kaggle_vna_training_notebooks import build
    from experiments.vna_training_kaggle import choose_solver
    for seed in (0,1):
        notebook=build('a'*40,seed)
        for cell in notebook['cells']:
            if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
        source=''.join(''.join(c['source']) for c in notebook['cells'])
        assert f'SEED = {seed}' in source and 'github_token' in source
        assert 'cifar-100-python/test' in source and 'archive_stop.wait(3600)' in source
    rows=[{'variant':'exact','cg_converged':True,'sgd_units':20},
          {'variant':'kfac','cosine':.8}]
    assert choose_solver(rows)=='kfac'
    rows[1]['cosine']=.2; assert choose_solver(rows)=='exact'
    rows[0]['sgd_units']=31; assert choose_solver(rows)=='lite'
    assert choose_solver(rows,'exact')=='exact'
    rows[0]['cg_converged']=False
    with pytest.raises(ValueError,match='unconverged'): choose_solver(rows)


def test_training_archive_restore(tmp_path):
    from experiments.vna_training_kaggle import archive_training,restore_training
    source=tmp_path/'source'; args=arguments(source/'sgd'); path=run(args)
    archive_training(source)
    restored=tmp_path/'restored'
    restore_training(tmp_path,restored,{'sgd':training_protocol(args)})
    equal(load(path)['model'],load(restored/'sgd/latest.pt')['model'])
    assert json.loads((restored/'sgd/final.json').read_text())['test_evaluations']==1
    with pytest.raises(RuntimeError,match='exceeds'): archive_training(source,budget=1)


def test_recipe_validation_checkpoint_mode(tmp_path):
    from diagnostics.vna_solver_validation import run_checkpoint
    from argparse import Namespace
    checkpoint=run(arguments(tmp_path/'training','--max-epochs','1'))
    summary=run_checkpoint(Namespace(checkpoint=str(checkpoint),output=str(tmp_path/'validation'),
        device='cpu',seed=0,fake_data=True,data_root='data',rank=2,stat_batch=8,
        stat_batches=1,vna_batch=3,cg_iters=1000,cg_tol=1e-5))
    assert summary['exact_converged'] and summary['sites']
    assert len(summary['rows'])==20
    assert {r['variant'] for r in summary['rows']}=={'exact','kfac','lite','euclidean','ngd'}
    for row in summary['rows']:
        assert row['sgd_units']>0
        if row['variant']=='exact':
            assert row['cg'] and all(r['relative_residual']<=1e-5 for r in row['cg'])


def test_checkpoint_disk_budget_preserves_latest(tmp_path):
    checkpoint=run(arguments(tmp_path,'--max-epochs','1'))
    before=checkpoint.read_bytes()
    with pytest.raises(RuntimeError,match='previous latest preserved'):
        run(arguments(tmp_path,'--resume',str(checkpoint),'--disk-budget','1'))
    assert checkpoint.read_bytes()==before


@pytest.mark.parametrize('arm,selection',[('vna','tiny_U'),('vna','greedy_V'),('vna_random','tiny_V')])
def test_training_other_selection_and_random_arms(tmp_path,arm,selection):
    checkpoint=load(run(arguments(tmp_path,'--arm',arm,'--selection',selection,'--solver','lite')))
    assert checkpoint['final']['params']==checkpoint['final']['sgd_params']
    assert checkpoint['vna']['controller']['refreshes']==2
    if arm=='vna_random':
        for entry in checkpoint['vna']['cache'].values():
            assert entry['random_trace']['trace_target']>=0
