"""Split notebook delivery and cross-part comparison contracts."""
import ast
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from experiments.vna_compare import compare, collect_results, hash_config, save_run_config
from experiments.vna_training_kaggle import apply_budget_policy
from scripts.build_kaggle_vna_training_notebooks import PART_ARMS, build


def test_six_notebooks(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for seed in (0, 1):
        assert not (root/f'notebooks/kaggle_vna_training_seed{seed}.ipynb').exists()
        for part in (1, 2, 3):
            path = tmp_path/f'kaggle_vna_seed{seed}_part{part}.ipynb'
            path.write_text(json.dumps(build('a'*40, seed=seed, part=part)))
            notebook = json.loads(path.read_text())
            assert notebook == build('a'*40, seed=seed, part=part)
            source = ''.join(''.join(c['source']) for c in notebook['cells'])
            assert f'/kaggle/working/vna_seed{seed}_part{part}' in source
            assert 'GPU_HOURS_BUDGET = 15' in source
            assert 'config_hash=config_hash' in source
            for c in notebook['cells']:
                if c['cell_type'] == 'code': ast.parse(''.join(c['source']))
            cell_b = next(''.join(c['source']) for c in notebook['cells'] if '# Cell B:' in ''.join(c['source']))
            scope = {'PART': part}
            exec(cell_b[:cell_b.index('flags =')], scope)
            assert scope['arms'] == PART_ARMS[part]
            assert scope['gpu_by_arm'] == {a[0]: gpu for gpu, a in enumerate(PART_ARMS[part])}
    assert [[a[0] for a in PART_ARMS[p]] for p in (1,2,3)] == [
        ['sgd','vna_tiny_V'], ['ngd','vna_random'], ['vna_tiny_U','sgd_cm']]


@pytest.mark.parametrize('part', [1,2])
def test_required_parts_never_drop(part):
    arms = PART_ARMS[part]; projections = {a[0]: 8 for a in arms}
    with pytest.raises(RuntimeError, match='Table:'):
        apply_budget_policy(arms, projections, part=part)
    assert apply_budget_policy(arms, projections, allow_over_budget=True, part=part) == (arms, 16)


def test_part3_drops_controls():
    arms = PART_ARMS[3]; projections = {a[0]: 8 for a in arms}
    assert apply_budget_policy(arms, projections, part=3) == ([arms[0]], 8)
    assert apply_budget_policy(arms, projections, budget=7, part=3) == ([], 0)


def finished(root, arm, *, seed=0, config=None, recipe=None):
    from experiments.run_vna_training import parser, training_protocol
    config = config or {'seed':seed,'solver':'lite','rank':16,'vna_batch':128,'vna_every':20,'code_commit':'a'*40}
    args = parser().parse_args(['--arm', 'vna' if arm.startswith('vna_tiny') else arm,
                                 '--selection', 'tiny_U' if arm == 'vna_tiny_U' else 'tiny_V'])
    protocol = training_protocol(args)
    if recipe: protocol.update(recipe)
    record = dict(arm=args.arm, selection=args.selection, seed=seed, epoch=protocol['total_epochs'],
                  protocol=protocol, config_hash=hash_config(config), run_config=config,
                  test={'accuracy':.6,'loss':1}, params=10, gpu_hours=2, wall_hours=2,
                  code_history=[{'commit':'a'*40,'from_epoch':1}])
    folder = root/arm; folder.mkdir(parents=True)
    (folder/'final.json').write_text(json.dumps(record))
    (folder/'history.json').write_text(json.dumps([dict(epoch=protocol['total_epochs'],
        train={'loss':1},eval={'accuracy':.5},mean_rho=.1)]))
    return folder


def test_compare_partial_without_sgd(tmp_path):
    finished(tmp_path/'part2', 'ngd')
    frame, missing, fig = compare(tmp_path, 0, input_root=tmp_path/'absent', output=tmp_path/'report', plot=False)
    assert frame.arm.tolist() == ['ngd']
    assert frame.delta_vs_sgd_pp.isna().all() and frame.cost_vs_sgd.isna().all()
    assert missing == ['sgd','vna_tiny_V','vna_random','vna_tiny_U','sgd_cm']
    assert (tmp_path/'report/results.csv').exists()
    assert fig is None


@pytest.mark.parametrize('archive_type', ['tar','zip'])
def test_compare_across_part_archives(tmp_path, archive_type):
    finished(tmp_path/'local', 'sgd')
    folder = finished(tmp_path/'source', 'vna_tiny_V')
    inputs = tmp_path/'inputs'; inputs.mkdir()
    if archive_type == 'tar':
        with tarfile.open(inputs/'part1.tar.gz', 'w:gz') as a: a.add(folder, arcname='./vna_tiny_V')
    else:
        with zipfile.ZipFile(inputs/'part1.zip', 'w') as a:
            for path in folder.iterdir(): a.write(path, f'part1/vna_tiny_V/{path.name}')
    frame, missing, fig = compare(tmp_path/'local', 0, input_root=inputs, plot=False)
    assert frame.arm.tolist() == ['sgd','vna_tiny_V']
    assert frame.cost_vs_sgd.tolist() == [1,1]
    assert 'vna_tiny_V' not in missing
    assert fig is None


def test_compare_config_hash_mismatch(tmp_path):
    finished(tmp_path/'p1','vna_tiny_V')
    finished(tmp_path/'p2','ngd',config={'seed':0,'solver':'exact','rank':16})
    with pytest.raises(ValueError, match='config_hash mismatch.*diff=.*solver'):
        collect_results(tmp_path, 0, tmp_path/'absent')


def test_compare_recipe_mismatch(tmp_path):
    finished(tmp_path/'p1','sgd')
    finished(tmp_path/'p2','vna_tiny_U',recipe={'batch_size':17})
    with pytest.raises(ValueError, match='Recipe protocol mismatch.*batch_size'):
        collect_results(tmp_path, 0, tmp_path/'absent')


def test_hash_order_and_existing_config_guard(tmp_path):
    assert hash_config({'a':1,'b':2}) == hash_config({'b':2,'a':1})
    path = tmp_path/'run_config.json'
    save_run_config(path, {'a':1})
    with pytest.raises(ValueError, match='fresh output'):
        save_run_config(path, {'a':2})


def test_resume_ignores_other_parts(tmp_path):
    from experiments.vna_training_kaggle import restore_training, archive_training
    from experiments.run_vna_training import run, training_protocol
    from tests.test_vna_training import arguments, load
    source = tmp_path/'attached'
    a = arguments(source/'sgd')
    b = arguments(source/'ngd', '--arm','ngd','--solver','lite')
    run(a); run(b); archive_training(source)
    output = tmp_path/'part1'
    restore_training(tmp_path, output, {'sgd':training_protocol(a)})
    assert (output/'sgd/final.json').exists()
    assert not (output/'ngd').exists()
    assert load(output/'sgd/latest.pt')['protocol'] == training_protocol(a)


@pytest.mark.parametrize('part', [1,2,3])
def test_execute_part_smoke_and_parallel_launch(tmp_path, part):
    import shutil
    import sys
    from types import SimpleNamespace
    from experiments.run_vna_training import parser, training_protocol
    from experiments.vna_training_kaggle import smoke_diagnostics
    notebook = build('a'*40, 0, part)
    source = next(''.join(c['source']) for c in notebook['cells'] if '# Cell B:' in ''.join(c['source']))
    smokes, launches, restored = [], [], []
    def smoke(command, **kwargs):
        args = parser().parse_args(command[4:]); smokes.append(args)
        assert args.max_epochs == 2 and args.vna_start_epoch == 1 and args.run_config is None
        output = Path(args.output); output.mkdir(parents=True)
        (output/'history.json').write_text(json.dumps([{'epoch':2,'epoch_seconds':36}]))
    class Process:
        def __init__(self, command, **kwargs):
            launches.append(parser().parse_args(command[4:]))
        def poll(self): return 0
    def restore(inputs, output, protocols, **kwargs):
        restored.extend(protocols)
        assert kwargs['config_hash'] == json.loads((output/'run_config.json').read_text())['config_hash']
    scope = dict(Path=Path, RUN_ROOT=tmp_path/f'part{part}', DATA_ROOT=tmp_path, REPO=tmp_path,
        sys=sys, json=json, shutil=shutil, subprocess=SimpleNamespace(run=smoke, Popen=Process, STDOUT=-2),
        training_parser=parser, training_protocol=training_protocol, PART=part, SEED=0, REPO_COMMIT='a'*40,
        SOLVER='lite', GPU_DATA=False, CHANNELS_LAST=False, CUDNN_BENCHMARK=False, AMP=False,
        shared_defaults={}, shared_arguments=[], test_env={}, SMOKE_EPOCHS=2,
        GPU_HOURS_BUDGET=15, ALLOW_OVER_BUDGET=False, apply_budget_policy=apply_budget_policy,
        smoke_diagnostics=smoke_diagnostics, restore_training=restore, archive_training=lambda p: p)
    exec(compile(source, 'Cell B', 'exec'), scope)
    expected = PART_ARMS[part]
    assert [a.arm for a in smokes] == [a[1] for a in expected]
    assert [a.arm for a in launches] == [a[1] for a in expected]
    assert [a.device for a in launches] == ['cuda:0','cuda:1']
    assert restored == [a[0] for a in expected]
    assert all(a.run_config == str(scope['RUN_ROOT']/'run_config.json') for a in launches)
    assert not scope['SMOKE_ROOT'].exists()


def test_render_comparison_curves(tmp_path):
    plt = pytest.importorskip('matplotlib.pyplot', reason='optional diagnostics dependency unavailable')
    finished(tmp_path/'part1', 'sgd')
    _, _, fig = compare(tmp_path/'part1', 0, input_root=tmp_path/'absent', output=tmp_path/'report')
    assert len(fig.axes) == 3 and all(len(ax.lines) == 1 for ax in fig.axes)
    assert (tmp_path/'report/curves.png').exists()
    plt.close(fig)


def test_runner_persists_and_checks_shared_config(tmp_path):
    from experiments.run_vna_training import run
    from tests.test_vna_training import arguments, load
    config = {'seed':0, 'solver':'lite', 'lambda_vna':1., 'rank':2,
              'damping_rel':.01, 'vna_batch':3, 'vna_every':1, 'code_commit':'a'*40}
    path = tmp_path/'run_config.json'; digest = save_run_config(path, config)
    checkpoint = run(arguments(tmp_path/'vna','--arm','vna','--solver','lite','--run-config',str(path)))
    payload = load(checkpoint)
    assert payload['config_hash'] == payload['final']['config_hash'] == digest
    assert payload['final']['run_config'] == config
    assert payload['final']['protocol'] == payload['protocol']
    other = tmp_path/'other_config.json'; save_run_config(other, dict(config, code_commit='b'*40))
    with pytest.raises(ValueError, match='resume config_hash mismatch'):
        run(arguments(tmp_path/'vna','--arm','vna','--solver','lite','--run-config',str(other),'--resume',str(checkpoint)))
    incompatible = tmp_path/'bad_config.json'; save_run_config(incompatible, dict(config, rank=9))
    with pytest.raises(ValueError, match='training arguments: rank'):
        run(arguments(tmp_path/'vna','--arm','vna','--solver','lite','--run-config',str(incompatible)))


def test_curve_series_for_partial_results(tmp_path, monkeypatch):
    """Check plotted data independently of the optional raster backend."""
    import sys
    from types import ModuleType, SimpleNamespace
    calls = [[], [], []]
    axes = [SimpleNamespace(plot=lambda x, y, i=i, **k: calls[i].append((x, y, k)),
                            set=lambda **k: None, grid=lambda **k: None, legend=lambda **k: None)
            for i in range(3)]
    figure = SimpleNamespace(tight_layout=lambda: None)
    package = ModuleType('matplotlib'); pyplot = ModuleType('matplotlib.pyplot')
    pyplot.subplots = lambda *a, **k: (figure, axes)
    package.pyplot = pyplot
    monkeypatch.setitem(sys.modules, 'matplotlib', package)
    monkeypatch.setitem(sys.modules, 'matplotlib.pyplot', pyplot)
    finished(tmp_path/'p2', 'ngd')
    frame, missing, fig = compare(tmp_path, 0, input_root=tmp_path/'absent')
    assert fig is figure and frame.arm.tolist() == ['ngd']
    assert [series[0][1] for series in calls] == [[1], [.5], [.1]]
    assert all(series[0][2]['label'] == 'ngd' for series in calls)
    assert 'sgd' in missing
