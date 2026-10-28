"""Only the notebook plan and new official-LPR/full-H3 paths; no GPU training."""
import ast
from copy import deepcopy
import json
import pytest
import torch
from diagnostics.vna_align import _global_state,_assert_bitwise


@pytest.fixture(autouse=True)
def isolate():
    threads=torch.get_num_threads();torch.set_num_threads(1)
    with _global_state():yield
    torch.set_num_threads(threads)


def test_four_notebooks_pin_source_and_exact_pairs():
    from scripts.build_kaggle_standard_compare import build,jobs
    for branch in ('H1','H2','H3','H4'):
        nb=build(branch,'a'*40)
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        text=''.join(''.join(c['source']) for c in nb['cells'])
        assert 'SOURCE_DIGEST' not in text and "'--fake-data'" not in text
        assert 'restore_stage_archives' in text and 'prefix=RUN_TAG' in text
        assert 'launch_logged' in text and 'max_workers=2' in text
        assert '--tasks' not in text if branch=='H3' else True
    assert [j['args'][-1] for lane in jobs('H1') for j in lane]==['baseline','h1']
    assert [j['args'][-1] for lane in jobs('H2') for j in lane]==['lpr','h2']
    assert [j['args'][-1] for lane in jobs('H3') for j in lane]==['cbp','vna_gir']
    assert all('--tasks' in j['args'] and j['args'][j['args'].index('--tasks')+1]=='20' for lane in jobs('H1') for j in lane)
    assert all(j['module']=='experiments.standard_online' for lane in jobs('H2') for j in lane)
    assert all(j['module']=='experiments.standard_imagenet' for lane in jobs('H3') for j in lane)
    assert len(jobs('H4')[0])==len(jobs('H4')[1])==1
    assert json.load(open('configs/standard_cl/h4_rotated_mnist.json'))['domains']==20


def test_lpr_unchanged_source_and_head_view_are_real():
    from methods.cl_lpr import official_module,prepare_head
    from experiments.runners.pycil import upstream_imports
    from experiments.vna_resume import rng_state
    upstream_imports()
    from convs.linears import SimpleLinear
    from types import SimpleNamespace
    module=official_module();net=SimpleNamespace(fc=SimpleLinear(2,3).double())
    x=torch.randn(4,2,dtype=torch.double);before=net.fc(x)['logits'].detach();ids=[id(p) for p in net.fc.parameters()];rng=rng_state()
    prepare_head(net)
    assert isinstance(net.fc,torch.nn.Linear) and ids==[id(p) for p in net.fc.parameters()]
    _assert_bitwise(before,net.fc(x)['logits'].detach(),'head');_assert_bitwise(rng,rng_state(),'head.rng')
    linear=torch.nn.Linear(2,3).double();linear(x).square().mean().backward()
    grad=torch.cat((linear.weight.grad,linear.bias.grad[:,None]),1).clone();matrix=torch.eye(3,dtype=torch.double)*.5
    module.precondition_gradient(linear,{'':module.PreconditionerInv(matrix,True)})
    torch.testing.assert_close(torch.cat((linear.weight.grad,linear.bias.grad[:,None]),1),grad@matrix)


def test_official_lpr_two_tasks_and_epoch_resume(tmp_path,monkeypatch):
    from experiments import standard_cl as runner
    def args(out):return runner.parser().parse_args(['--data-dir',str(tmp_path),'--out',str(out),'--tasks','2','--device','cpu',
        '--branch','H2','--baseline','er','--arm','lpr','--config','configs/standard_cl/h2_cifar100_bn.json','--fake-data'])
    result=runner.run(args(tmp_path/'complete'));assert result['status']=='COMPLETED'
    calls=0
    def stop():
        nonlocal calls
        calls+=1;return calls==9
    monkeypatch.setattr(runner,'deadline_due',stop)
    with pytest.raises(SystemExit):runner.run(args(tmp_path/'paused'))
    monkeypatch.setattr(runner,'deadline_due',lambda:False)
    runner.run(args(tmp_path/'paused'))
    a=torch.load(tmp_path/'complete/latest.pt',weights_only=False);b=torch.load(tmp_path/'paused/latest.pt',weights_only=False)
    assert 'fc' in a['learner']['_lpr_state']['inverses']
    for key in ('_lpr_state','_data_memory','_targets_memory'):_assert_bitwise(a['learner'][key],b['learner'][key],key)
    _assert_bitwise(a['learner']['_network'].state_dict(),b['learner']['_network'].state_dict(),'lpr.net')
    _assert_bitwise(a['rng'],b['rng'],'lpr.rng')
    full=args(tmp_path/'geometry');full.arm='h2';runner.run(full)
    from scripts.collect_standard_cl import collect
    paired=collect([tmp_path/'complete',tmp_path/'geometry'])
    assert len(paired)==2 and all(r['paired_reference_arm']=='lpr' for r in paired)
    assert all(r['metrics']['final_accuracy']['paired_seeds']==1 for r in paired)


def test_h3_full_vna_then_gir_and_transaction_resume(tmp_path,monkeypatch):
    from experiments import standard_plasticity as runner
    from methods.cl_replay import ReplayCorrection
    def args(out):return runner.parser().parse_args(['--out',str(out),'--arm','vna_gir','--fake-data'])
    result=runner.run(args(tmp_path/'complete'));assert result['status']=='COMPLETED'
    events=json.loads((tmp_path/'complete/interventions.json').read_text())
    assert len(events)==2 and all('vna' in e and e['vna']['kernel']=='tiny_V' for e in events)
    original=ReplayCorrection.boundary;calls=0
    def stop():
        nonlocal calls
        calls+=1
        if calls==7:raise SystemExit(75)
        original()
    monkeypatch.setattr(ReplayCorrection,'boundary',staticmethod(stop))
    with pytest.raises(SystemExit):runner.run(args(tmp_path/'paused'))
    monkeypatch.setattr(ReplayCorrection,'boundary',staticmethod(original))
    runner.run(args(tmp_path/'paused'))
    a=torch.load(tmp_path/'complete/latest.pt',weights_only=False);b=torch.load(tmp_path/'paused/latest.pt',weights_only=False)
    for key in ('model','optimizer','rng','rows','curves'):_assert_bitwise(a[key],b[key],key)
    assert a['vna_engine']['applications']==b['vna_engine']['applications']==2
