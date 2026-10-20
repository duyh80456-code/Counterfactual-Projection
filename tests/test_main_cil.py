"""Tiny local checks for the integrated main setup and real Phase 2 engine."""
import copy
import json
import pickle
import numpy as np
import pytest
import torch
from torch import nn
from adapters.cil_setup import class_order, FINE_TO_COARSE, COARSE_NAMES, resnet32_nobn
from experiments import main_cil as main


class Tiny(nn.Module):
    out_dim = 2
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3,2,1)

    def forward(self,x):
        x = self.stem(nn.functional.adaptive_avg_pool2d(x.to(self.stem.weight), 2))
        return dict(features=x.mean((2,3)))


def test_superclass_partition_and_pickle(tmp_path):
    order = class_order(0)
    assert sorted(order) == list(range(100))
    assert len(order) == 20*5
    assert all(len({FINE_TO_COARSE[f] for f in order[i:i+5]}) == 1 for i in range(0,100,5))
    assert class_order(1) != order
    assert class_order(0, 'random') == np.random.RandomState(0).permutation(100).tolist()
    (tmp_path/'train').write_bytes(pickle.dumps(dict(fine_labels=list(range(100)), coarse_labels=FINE_TO_COARSE)))
    (tmp_path/'meta').write_bytes(pickle.dumps(dict(coarse_label_names=COARSE_NAMES)))
    assert class_order(0, data_dir=tmp_path) == order
    (tmp_path/'train').write_bytes(pickle.dumps(dict(fine_labels=list(range(100)))))
    assert class_order(0, data_dir=tmp_path) == order
    (tmp_path/'meta').write_bytes(pickle.dumps(dict(coarse_label_names=['wrong'])))
    with pytest.raises(ValueError, match='coarse label names'): class_order(0, data_dir=tmp_path)


def test_nobn_forward_backward_and_recipe():
    main.cil.upstream_imports()
    torch.set_num_threads(1)
    model = resnet32_nobn().double()
    output = model(torch.randn(2,3,32,32, dtype=torch.float64))['features']
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert all(m.bias is not None for m in model.modules() if isinstance(m, nn.Conv2d))
    alphas = [p for n,p in model.named_parameters() if n.endswith('alpha')]
    assert len(alphas) == 15 and all(float(p) == 0 for p in alphas)
    assert sum(p.numel() for p in model.parameters()) > 460000
    params = main.recipe()
    assert (params['init_cls'],params['increment'],params['memory_size'],params['fixed_memory']) == (5,5,2000,False)
    from models import icarl
    args = main.parser().parse_args(['--out','unused','--nobn-lr-scale','.5'])
    old_epochs,old_lr = icarl.epochs,icarl.lrate
    with main.runtime(args, class_order(0)):
        assert (icarl.init_epoch,icarl.epochs)==(100,50)
        assert icarl.init_milestones==[30,60,85] and icarl.milestones==[24,35]
        assert icarl.init_lr==icarl.lrate==.05
    assert icarl.epochs==old_epochs and icarl.lrate==old_lr


def test_per_class_forgetting_and_linear_probe():
    torch.set_num_threads(1)
    targets = np.array([0,0,1,1])
    first = main.accuracy_metrics([], np.array([[0],[0],[1],[1]]), targets, 0, 'cnn')
    row = main.accuracy_metrics([first], np.array([[0],[1],[1],[1]]), targets, 1, 'cnn')
    assert row['cnn_forgetting'] == 25.
    assert row['cnn_old_top1'] == 50. and row['cnn_new_top1'] == 100.
    improved = main.accuracy_metrics([row], np.array([[0],[0],[1],[1]]), targets, 1, 'cnn')
    assert improved['cnn_forgetting'] == -25.
    from types import SimpleNamespace
    from torch.utils.data import TensorDataset
    class Features(nn.Module):
        def extract_vector(self, x): return x
    x = torch.tensor([[2.,0.],[3.,0.],[0.,2.],[0.,3.]])
    y = torch.tensor([0,0,1,1])
    manager = SimpleNamespace(get_dataset=lambda *a, **kw: TensorDataset(torch.arange(4),x,y))
    learner = SimpleNamespace(_network=Features(),_total_classes=2,_device='cpu')
    state = torch.get_rng_state().clone()
    assert main.feature_accuracy(learner,manager) == 100.
    assert torch.equal(state,torch.get_rng_state()) and learner._network.training


def test_collector_and_paired_table(tmp_path):
    from experiments.main_cil_results import table, stability_plasticity
    from scripts.collect_parallel import validate_result, collect, validate_shared_schedules
    commit = 'a'*40
    jobs = {}
    for arm in main.ARMS:
        for seed in (0,1):
            tasks=[]
            for task in range(1,21):
                row=dict(task=task, feature_accuracy=70.)
                for label in ('cnn','nme'):
                    row.update({label+'_'+key:60.+(arm=='icarl_full') for key in
                        ('top1','old_top1','new_top1','learning_accuracy','forgetting','average_accuracy','last_accuracy')})
                    row[label+'_per_class']={str(c):60. for c in range(task*5)}
                tasks.append(row)
            protocol=dict(setting='icarl',algorithm_version='v4',arm=arm,seed=seed,
                code_commit=commit, upstream_commit=main.cil.UPSTREAM_PIN, expected_tasks=20,
                cfg=dict(init_cls=5,increment=5,memory_size=2000,fixed_memory=False),
                r=16,backbone='resnet32_nobn',class_order_kind='superclass',class_order=class_order(seed),
                training=dict(first_epochs=100,later_epochs=50,first_milestones=[30,60,85],later_milestones=[24,35]),
                model_config=dict(backbone='resnet32_nobn',class_order='superclass'),
                dataset_fingerprint={k:'b'*64 for k in ('train','test','meta')})
            summary=dict(identity=copy.deepcopy(protocol),tasks=tasks,status='COMPLETED',gpu_hours=1.)
            summary.update({f'{label}_{key}':tasks[-1][f'{label}_{key}'] for label in ('cnn','nme') for key in
                ('average_accuracy','last_accuracy','learning_accuracy','forgetting')})
            job=f'{arm}__nobn__superclass__s{seed}'
            record=dict(protocol=protocol,summary=summary,tasks=tasks)
            jobs[job]=record
            validate_result(job,summary,protocol,tasks,commit)
            with pytest.raises(ValueError,match='dynamic memory 2000'):
                validate_result(job,summary,{**protocol,'cfg':dict(init_cls=5,increment=5,memory_size=40,fixed_memory=False)},tasks,commit)
            folder=tmp_path/'outputs'/job; folder.mkdir(parents=True)
            for name in main.RESULT_FILES:
                value=record.get(name[:-5],dict(status='COMPLETED'))
                if name=='diagnostics.json': value=[dict(task=i,B=1.,mu_diag=.1,norm_w2=2.) for i in range(1,21)]
                (folder/name).write_text('task\n' if name.endswith('.csv') else json.dumps(value))
            if arm=='icarl_full': (folder/'interventions.json').write_text('[]')
    rows=table(jobs.values())
    row=next(r for r in rows if r['arm']=='icarl_full' and r['metric']=='nme_average_accuracy')
    assert row['paired_delta']==dict(mean=1.,std=0.) and row['seeds_won']==2
    assert len(collect([tmp_path],'outputs'))==4
    stability_plasticity(jobs.values(),tmp_path/'stability.svg')
    assert (tmp_path/'stability.svg').stat().st_size>0
    protocol_file=tmp_path/'outputs'/'icarl_full__nobn__superclass__s0'/'protocol.json'
    original=protocol_file.read_text()
    altered=json.loads(original); altered['nobn_lr_scale']=.5
    protocol_file.write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='mixed noBN learning-rate scales'): collect([tmp_path],'outputs')
    protocol_file.write_text(original)
    victim=tmp_path/'outputs'/'icarl_full__nobn__superclass__s0'/'diagnostics.json'
    victim.unlink()
    with pytest.raises(ValueError,match='missing main CIL files'): collect([tmp_path],'outputs')
    for r in jobs.values(): r['protocol']['schedule']=dict(K0=3,seconds_per_application=1.)
    validate_shared_schedules(jobs)
    jobs[next(iter(jobs))]['protocol']['schedule']['K0']=4
    with pytest.raises(ValueError,match='inconsistent shared authority'): validate_shared_schedules(jobs)


def test_schedule_skips_and_roundtrip():
    class Engine:
        def __init__(self): self.state = {}; self.calls = []
        def state_dict(self): return copy.deepcopy(self.state)
        def load_state_dict(self, state): self.state = copy.deepcopy(state)
        def begin_task(self, task): self.state.update(task=task, A={})
        coefficient = 1.
        def apply(self, *a, **kw):
            self.calls.append(a[6]); return dict(seconds=0., step=a[6])
    from types import SimpleNamespace
    from unittest.mock import patch
    engine = Engine(); intervention = main.Intervention(engine, 3)
    learner = SimpleNamespace(_cur_task=0, _total_classes=5, _device='cpu', _network=nn.Linear(1,1))
    hook = SimpleNamespace(steps=1, current_batch=(torch.ones(1,1),torch.zeros(1,dtype=torch.long)))
    intervention.start_task(learner)
    with patch.object(main,'sites_for',return_value=[]), patch.object(main,'loss_context',return_value=(None,{})):
        for step in (1,2,5,6,12,24):
            hook.steps=step; intervention.after_step(hook,learner,None)
    assert engine.calls == [1,5,12,24]
    state = intervention.state_dict()
    restored = main.Intervention(Engine(), 1); restored.load_state_dict(state)
    assert restored.state_dict() == state
    learner._cur_task = 1; restored.start_task(learner)
    assert restored.last_iteration is None and restored.interval == 3


def test_two_task_smoke_identity_and_resume(tmp_path, monkeypatch):
    main.cil.upstream_imports()
    monkeypatch.setattr(main, 'resnet32_nobn', Tiny)
    old_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        results = []
        for name, coefficient in (('icarl',1.), ('off',0.), ('full',1.)):
            args = main.parser().parse_args(['--arm','icarl' if name=='icarl' else 'icarl_full',
                '--fake-data','--workers','0','--device','cpu','--coefficient',str(coefficient),
                '--out',str(tmp_path/name)])
            result = main.run(args); results.append(result)
            assert result['status'] == 'COMPLETED' and result['tasks_run'] == 2
            for file in main.RESULT_FILES: assert (tmp_path/name/file).exists()
            assert (tmp_path/name/'interventions.json').exists() == (name != 'icarl')
            protocol = json.loads((tmp_path/name/'protocol.json').read_text())
            assert protocol['parameter_count'] == result['parameter_count'] > 0
            assert protocol['r'] == 16 and protocol['nobn_lr_scale'] == 1.
            assert main.run(args) == result
        def state(name): return torch.load(tmp_path/name/'latest.pt',weights_only=False)
        baseline, off = state('icarl'), state('off')
        a,b = baseline['learner']['_network'].state_dict(), off['learner']['_network'].state_dict()
        assert all(torch.equal(a[k],b[k]) for k in a)
        assert results[0]['tasks'] == results[1]['tasks']
        assert all(torch.equal(baseline['rng'][k],off['rng'][k]) for k in ('torch',))
        assert json.loads((tmp_path/'full'/'interventions.json').read_text())
        for name in ('icarl','off','full'):
            rows = json.loads((tmp_path/name/'diagnostics.json').read_text())
            assert len(rows) == 2 and all(r['B'] >= 0 and r['mu_diag'] > 0 for r in rows)
            assert [len(r['probe_indices']) for r in rows] == [4,4]
        # Pause after a complete first task; native prefix replay must be exact.
        original_checkpoint = main.ICaRLHook.checkpoint
        def pause(hook, **kw):
            original_checkpoint(hook, **kw)
            if not kw.get('mid', True) and hook.completed == 1:
                raise SystemExit(75)
        args.arm='icarl'; args.out=str(tmp_path/'resumed')
        with monkeypatch.context() as patch:
            patch.setattr(main.ICaRLHook, 'checkpoint', pause)
            with pytest.raises(SystemExit) as exc: main.run(args)
            assert exc.value.code == 75
        resumed = main.run(args)
        actual = state('resumed')['learner']['_network'].state_dict()
        assert all(torch.equal(a[k],actual[k]) for k in a)
        assert resumed['tasks'] == results[0]['tasks']
        # A deadline during evaluation must retain the last safe training checkpoint.
        original_measurement_boundary = main.MainHook.measurement_boundary
        def pause_measurement(hook):
            if hook.learner._cur_task == 1:
                with monkeypatch.context() as patch:
                    patch.setattr('experiments.vna_resume.deadline_due', lambda *a: True)
                    original_measurement_boundary(hook)
            else: original_measurement_boundary(hook)
        args.out=str(tmp_path/'measurement_pause')
        with monkeypatch.context() as patch:
            patch.setattr(main.MainHook,'measurement_boundary',pause_measurement)
            with pytest.raises(SystemExit) as exc: main.run(args)
            assert exc.value.code == 75
        saved=state('measurement_pause')
        assert saved['completed']==1 and saved['learner']['_cur_task']==0
        result=main.run(args)
        actual=state('measurement_pause')
        assert result['tasks']==results[0]['tasks']
        assert all(torch.equal(a[k],actual['learner']['_network'].state_dict()[k]) for k in a)
        assert torch.equal(baseline['rng']['torch'],actual['rng']['torch'])
    finally: torch.set_default_dtype(old_dtype)


def test_main_authority_preflight_real_engine(tmp_path, monkeypatch):
    main.cil.upstream_imports()
    from models import icarl
    monkeypatch.setattr(main, 'resnet32_nobn', Tiny)
    monkeypatch.setattr(icarl, 'batch_size', 8)
    args = main.parser().parse_args(['--arm','icarl_full','--fake-data','--workers','0',
        '--device','cpu','--preflight-only','--schedule-output',str(tmp_path/'schedule.json'),
        '--out',str(tmp_path/'calibration')])
    result = main.run(args)
    schedule = json.loads((tmp_path/'schedule.json').read_text())
    assert result['status'] == 'PREFLIGHT'
    assert schedule['authoritative'] and schedule['K0'] >= 1
    assert [p['task'] for p in schedule['calibration']] == [1]*3+[2]*3
    assert schedule['seconds_per_application'] > 0
    assert all(p['iteration_seconds'] > 0 for p in schedule['calibration'])


def test_competitor_and_main_share_table_and_probe_setup():
    from experiments.main_cil_results import table
    from experiments.v4_competitors import recipe
    base = main.recipe(0,'cpu','resnet32')
    competitor,_,_ = recipe('icarl_redo','bn',0,'cpu')
    for key in ('init_cls','increment','memory_size','fixed_memory','convnet_type'):
        assert base[key] == competitor[key]
    def record(arm, competitor=False):
        summary = {f'{label}_{metric}': 50.+(arm!='icarl') for label in ('cnn','nme')
                   for metric in ('average_accuracy','last_accuracy','learning_accuracy','forgetting')}
        summary['gpu_hours'] = 1.+(arm!='icarl')
        protocol = dict(arm=arm,seed=0,backbone='bn' if competitor else 'resnet32',
                        order='superclass',class_order_kind='superclass',setting='icarl',algorithm_version='v4')
        if competitor:
            protocol.pop('setting'); protocol.pop('algorithm_version')
            protocol['benchmark']='v4_competitor'
        return dict(summary=summary,protocol=protocol)
    rows = table([record('icarl'),record('icarl_redo',True)])
    row = next(r for r in rows if r['arm']=='icarl_redo' and r['metric']=='cnn_average_accuracy')
    assert row['backbone']=='resnet32' and row['paired_delta']['mean']==1. and row['seeds_won']==1
    reproduction=record('icarl_cflat',True); reproduction['protocol']['reproduction']=True
    rows=table([record('icarl'),reproduction])
    row=next(r for r in rows if r['arm']=='icarl_cflat')
    assert row['class_order']=='superclass/reproduction' and row['paired_seeds']==0


def test_rebuilt_notebooks_keep_checkpoint_runtime_and_placeholder_pin(tmp_path, monkeypatch):
    import ast
    from scripts.build_kaggle_vna_parallel import main as build_notebooks, PRESETS
    monkeypatch.setattr('sys.argv', ['build_kaggle_vna_parallel', '--commit', 'PLACEHOLDER',
        '--variants', '--output', str(tmp_path/'vna4_M1.ipynb')])
    build_notebooks()
    assert {p.name for p in tmp_path.glob('*.ipynb')} == {f'vna4_{name}.ipynb' for name in PRESETS} | {'vna4_exact_diagnostics.ipynb'}
    for path in tmp_path.glob('*.ipynb'):
        notebook = json.loads(path.read_text())
        source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'])
        assert "REPO_COMMIT = 'PLACEHOLDER'" in source
        assert 'launch_logged' in source
        assert ('--branch' in source if path.stem == 'vna4_exact_diagnostics' else 'validate_result' in source)
        assert 'restore_stage_archives' in source and 'VNA_DEADLINE' in source
        assert 'One-Shot-TAS-CCIL' not in source and 'GROMO' not in source
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code': ast.parse(''.join(cell['source']))


@pytest.mark.parametrize('arm', [a for a in main.ENGINE_ARMS if a != 'icarl_full'])
def test_round3_ablation_arms_run_with_engine_variant(tmp_path, monkeypatch, arm):
    main.cil.upstream_imports()
    monkeypatch.setattr(main, 'resnet32_nobn', Tiny)
    old_dtype = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        args = main.parser().parse_args(['--arm', arm, '--fake-data', '--workers', '0',
                                         '--device', 'cpu', '--out', str(tmp_path/arm)])
        result = main.run(args)
        assert result['status'] == 'COMPLETED' and result['tasks_run'] == 2
        events = json.loads((tmp_path/arm/'interventions.json').read_text())
        assert events and all(e.get('beta', 0.) == 0. for e in events if arm == 'icarl_full_beta0' and 'beta' in e)
    finally:
        torch.set_default_dtype(old_dtype)
