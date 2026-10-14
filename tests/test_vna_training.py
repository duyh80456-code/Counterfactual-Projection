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
        '--epochs','2','--vna-start-epoch','1','--limit-batches','1','--vna-every','1','--vna-batch','3',
        '--stat-batch','8','--stat-batches','1','--refresh-epochs','1','--rank','2','--hutchinson-probes','2',
        '--cg-iters','100','--cg-tol','1e-4',*extra])


def read_corrections(path):
    return [json.loads(line) for line in (path/'corrections.jsonl').read_text().splitlines()]

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
    for key in ('model','optimizer','scheduler','rng','train_loader_generator_state','numpy_rng','vna','correction_count','step'):
        equal(full[key],resumed[key])
    equal(read_corrections(tmp_path/'full'),read_corrections(tmp_path/'resume'))
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
    assert 'corrections' not in vna
    assert len(read_corrections(tmp_path/'vna'))==2


@pytest.mark.parametrize('solver',['exact','kfac','lite'])
def test_training_ngd_norm_matching(tmp_path,solver):
    cp=load(run(arguments(tmp_path,'--arm','ngd','--solver',solver)))
    assert read_corrections(tmp_path)
    for row in read_corrections(tmp_path):
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
    rows=[{'variant':name,'relative_mu':.01,'site':'a','epoch':40,'cg_converged':True,
           'sgd_units':cost,'cosine':cos,'realized_gain':1.} for name,cost,cos in
          [('exact',8,1.),('kfac',3,.95),('lite',1,.2)]]
    assert choose_solver(rows)=='kfac'
    rows[1]['cosine']=.8; assert choose_solver(rows)=='exact'
    rows[0]['sgd_units']=31; assert choose_solver(rows)=='kfac'
    rows[1]['sgd_units']=11; assert choose_solver(rows)=='lite'
    rows[2]['sgd_units']=11
    with pytest.raises(RuntimeError,match='Table'): choose_solver(rows)
    assert choose_solver(rows,'exact')=='exact'
    rows[0]['cg_converged']=False
    with pytest.raises(ValueError,match='unconverged'): choose_solver(rows)
    with pytest.raises(RuntimeError,match='SOLVER_OVERRIDE'): choose_solver([])
    assert choose_solver([],'lite')=='lite'


def test_training_archive_restore(tmp_path):
    from experiments.vna_training_kaggle import archive_training,restore_training
    source=tmp_path/'source'; args=arguments(source/'sgd'); path=run(args)
    archive_training(source)
    restored=tmp_path/'restored'
    restore_training(tmp_path,restored,{'sgd':training_protocol(args)})
    equal(load(path)['model'],load(restored/'sgd/latest.pt')['model'])
    assert json.loads((restored/'sgd/final.json').read_text())['test_evaluations']==1
    with pytest.raises(RuntimeError,match='exceeds'): archive_training(source,budget=1)


def test_recipe_validation_checkpoint_mode(tmp_path,monkeypatch):
    from diagnostics.vna_solver_validation import run_checkpoint
    checkpoint=run(arguments(tmp_path/'training','--max-epochs','1'))
    args=arguments(tmp_path/'validation','--rank','2','--max-sites','3','--dorfler-theta','1',
                   '--cg-iters','1000','--cg-tol','1e-5')
    args.checkpoint=str(checkpoint)
    import diagnostics.vna_solver_validation as validation
    original=validation.vna_exact_ggn
    first=True
    def nontraining_failure(*a,**kw):
        nonlocal first
        d,diag=original(*a,**kw)
        if first:
            diag['cg'][0]['converged']=False
            first=False
        return d,diag
    monkeypatch.setattr(validation,'vna_exact_ggn',nontraining_failure)
    summary=run_checkpoint(args)
    assert not summary['exact_converged']  # sweep failure is descriptive
    for key in ('rank','vna_batch','damping_rel','cg_iters','cg_tol','selection','max_sites'):
        assert summary['training_config'][key]==getattr(args,key)

    assert summary['exact_converged_at_train_mu'] and summary['sites']
    assert summary['batch_checks']
    assert len(summary['rows'])==20
    assert {r['variant'] for r in summary['rows']}=={'exact','kfac','lite','euclidean','ngd'}
    for row in summary['rows']:
        assert row['sgd_units']>0
        if row['variant']=='exact' and row['relative_mu']==args.damping_rel:
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


@pytest.mark.parametrize('solver',['exact','kfac','lite'])
@pytest.mark.parametrize('cached',[False,True])
@pytest.mark.parametrize('mixed_modes',[False,True])
def test_train_mode_buffers_and_modes_preserved(tmp_path,solver,cached,mixed_modes):
    from experiments.vna_support import build_model,setup_data,augmented_batch
    from experiments.vna_training_support import TrainingVNA
    args=arguments(tmp_path,'--solver',solver)
    args.cache_block_input=cached
    device=torch.device('cpu')
    train,_,indices,*_=setup_data('data',True,0)
    model=build_model(device,True); model.train()
    # Mixed child modes also must survive exactly.
    if mixed_modes: next(m for m in model.modules() if isinstance(m,torch.nn.BatchNorm2d)).eval()
    buffers={n:b.clone() for n,b in model.named_buffers()}
    modes={n:m.training for n,m in model.named_modules()}
    def unchanged():
        assert buffers.keys()==dict(model.named_buffers()).keys()
        for n,b in model.named_buffers(): assert torch.equal(b,buffers[n]),n
        assert modes=={n:m.training for n,m in model.named_modules()}
    engine=TrainingVNA(args,device)
    engine.refresh(model,train,indices,1); unchanged()
    batch=augmented_batch(train,indices,3,engine.generators['correction'],device)
    for site in engine.cache:
        engine.correction(model,site,batch); unchanged()
    A={s:{n:None if v is None else v.clone() for n,v in e['A'].items()} for s,e in engine.cache.items()}
    engine.refresh_factors(model,train,indices,2); unchanged()
    for site in A: equal(A[site],engine.cache[site]['A'])


def test_start_epoch_strength_and_compact_logs(tmp_path):
    cp=load(run(arguments(tmp_path,'--arm','ngd','--solver','lite','--epochs','3',
        '--vna-start-epoch','2','--refresh-epochs','5','--vna-every','2','--lambda-vna','.25')))
    assert cp['history'][0]['selected_sites']=={}
    assert cp['history'][0]['corrections_applied']==0
    assert cp['vna']['controller']['refreshes']==1
    assert cp['vna']['controller']['factor_refresh_epoch']==3
    rows=read_corrections(tmp_path)
    assert rows and all(r['epoch']>=2 and r['coefficient']==.5 for r in rows)
    assert all('raw_gain' in r and 'realized_gain' in r and 'applied_delta_ratio' in r for r in rows)
    assert 'corrections' not in cp and cp['correction_count']==len(rows)
    assert all('residuals' not in c for r in rows for c in r['cg'])
    from experiments.vna_training_support import TrainingVNA
    args=arguments(tmp_path,'--lambda-mode','per_step','--lambda-vna','.25','--vna-every','2')
    assert TrainingVNA(args,torch.device('cpu')).coefficient==.25


def test_solver_training_mu_worst_site_and_warning(capsys):
    from experiments.vna_training_kaggle import choose_solver
    rows=[]
    for epoch in (40,100):
        for site in ('a','b'):
            for variant,cost,cos in [('exact',8,1),('kfac',3,.95),('lite',1,.2)]:
                rows.append(dict(epoch=epoch,site=site,variant=variant,relative_mu=.01,
                    cg_converged=True,sgd_units=cost,cosine=cos,realized_gain=1,cos_disjoint_batches=.2))
    rows.append(dict(variant='exact',relative_mu=.0001,cg_converged=False))
    assert choose_solver(rows)=='kfac'
    assert 'VNA_BATCH=64' in capsys.readouterr().out
    rows[1]['cosine']=.85
    assert choose_solver(rows)=='kfac'  # exact exceeds total overhead; kfac affordable
    rows[1]['sgd_units']=6
    assert choose_solver(rows)=='lite'


def test_validation_shared_defaults():
    from diagnostics.vna_solver_validation import parser as validation_parser
    from experiments.vna_training_config import add_vna_arguments
    import argparse
    training=parser().parse_args([])
    validation=validation_parser().parse_args([])
    for key in vars(add_vna_arguments(argparse.ArgumentParser()).parse_args([])):
        assert getattr(validation,key)==getattr(training,key)
    assert validation.batch_size==training.batch_size

@pytest.mark.parametrize('allow_over_budget',[False,True])
def test_notebook_smoke_budget_and_queue(tmp_path,allow_over_budget):
    from scripts.build_kaggle_vna_training_notebooks import build
    import shutil
    import sys
    from types import SimpleNamespace
    notebook=build('a'*40,0)
    source=next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('# Cell B:'))
    source=source[:source.index('restore_training(Path(')]
    launched=[]
    def smoke(command,**kwargs):
        args=parser().parse_args(command[4:])
        launched.append(args)
        assert args.resume is None and args.max_epochs==2 and args.vna_start_epoch==1
        output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
        (output/'history.json').write_text(json.dumps([{'epoch':1,'epoch_seconds':999},
                                                     {'epoch':2,'epoch_seconds':36}]))
    scope=dict(RUN_ROOT=tmp_path/'real',DATA_ROOT=tmp_path,REPO=tmp_path,Path=Path,sys=sys,
        subprocess=SimpleNamespace(run=smoke),json=json,shutil=shutil,training_parser=parser,
        training_protocol=training_protocol,SEED=0,SOLVER='lite',GPU_DATA=False,
        CHANNELS_LAST=False,CUDNN_BENCHMARK=False,AMP=False,shared_arguments=[],test_env={},
        SMOKE_EPOCHS=2,GPU_HOURS_BUDGET=1,ALLOW_OVER_BUDGET=allow_over_budget)
    if allow_over_budget: exec(compile(source,'smoke','exec'),scope)
    else:
        with pytest.raises(RuntimeError,match='GPU-hours exceeds'): exec(compile(source,'smoke','exec'),scope)
    assert [a.arm for a in launched]==['sgd','vna','ngd','vna_random','vna','sgd_cm']
    assert not (tmp_path/'real').exists()
    assert scope['projected_total']==pytest.approx(12.6)
    assert all(Path(a.output).is_relative_to(scope['SMOKE_ROOT']) for a in launched)
