"""Actual pinned loops: alignment, stream ordering, task resume and recipes."""
import copy
import pickle
import random
import numpy as np
import pytest
import torch
from adapters.lop_hooks import LearnHook, Prediction, sites_for, learner_state
from diagnostics.vna_align import _assert_bitwise, MeasurementHook, _global_state
from experiments.stage_b5 import upstream_imports, recipe, prepare_data, run_native, baseline_gate, RATES
from methods.vna import block_products
from methods.vna_ggn import GGNConfig
from methods.vna_select import tiny_V
from tests.test_vna_stage_b_upstream import pinned


@pytest.fixture(autouse=True)
def runtime():
    pinned('loss-of-plasticity'); upstream_imports()
    old = torch.get_num_threads(); torch.set_num_threads(1)
    with _global_state(): yield
    torch.set_num_threads(old)


def fixture(benchmark, arm, path, tasks=2):
    params, meta = recipe(benchmark, arm, 1e-3 if benchmark == 'regression' and arm == 'cbp' else None)
    params['use_gpu'] = 0
    path.mkdir(parents=True, exist_ok=True)
    params['data_file'] = str(path/'logs.pkl'); params['run_idx'] = 0
    params['num_features'] = 5 if benchmark == 'regression' else 4
    if benchmark == 'regression': params.update(flip_after=160, num_data_points=tasks*160)
    elif benchmark == 'mnist': params.update(num_tasks=tasks, change_after=160, fixture_rank_padding=True)
    else: params.update(num_tasks=tasks, num_showings=1, mini_batch_size=100,
                        fake_train_images_per_class=100, fake_test_images_per_class=50, examples_per_epoch=200)
    prepare_data(benchmark, params, path, fake=True, smoke=True)
    return params


def reset(seed=13):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def actual_measure(model, batch, spec, ctx, *, generator):
    # The alignment probe uses the real selection and full-network GGN solve,
    # then deliberately consumes every global stream to test isolation.
    from methods.vna_joint import joint_correction
    site = sites_for(model)[-1]
    with torch.no_grad(): logits = model(batch[0])
    g = spec.residual(logits, ctx)
    A, _ = tiny_V(model, site, [batch], 1, cotangents=[g], score_signs=False)
    record = joint_correction(model, [site], {site: A}, batch,
                              GGNConfig(cg_iters=2, hutchinson_probes=1, project_bn=False),
                              loss_spec=spec, ctx=ctx)[3]
    random.random(); np.random.rand(); torch.rand(3); torch.rand(2, generator=generator)
    return [dict(task=ctx['task'], step=ctx['step'], rho=record['rho'],
                 point=ctx['point'], within_task_step=ctx['within_task_step'],
                 buffer_task_purity=ctx['buffer_task_purity'])]


@pytest.mark.parametrize('benchmark', ['regression', 'mnist', 'imagenet'])
@pytest.mark.parametrize('arm', ['bp', 'cbp'])
def test_native_loops_bit_identical_params_optimizer_cbp_logs(benchmark, arm, tmp_path):
    params = fixture(benchmark, arm, tmp_path)
    states = []
    for enabled in (False, True):
        reset()
        if not enabled:
            # Off really leaves learn untouched. Capture construction only to
            # inspect the otherwise-local learner after the original function.
            from lop.algos.bp import Backprop
            from lop.algos.cbp import ContinualBackprop
            from lop.algos.convCBP import ConvCBP
            captured = []
            with pytest.MonkeyPatch.context() as patch:
                for cls in (Backprop, ContinualBackprop, ConvCBP):
                    init = cls.__init__
                    def capture(learner, *args, _init=init, **kwargs):
                        _init(learner, *args, **kwargs); captured.append(learner)
                    patch.setattr(cls, '__init__', capture)
                logs = run_native(benchmark, params, tmp_path, None)
            states.append((learner_state(captured[-1]), logs))
            continue
        hook = LearnHook(benchmark, arm, params, actual_measure if enabled and arm == 'bp' else None)
        if enabled and arm == 'cbp':
            isolation = MeasurementHook(actual_measure)
            def observe(learner, x, target, task, step):
                if (step-1)%hook.task_steps: return
                from methods.vna_losses import CE, MSE
                spec = MSE(reduction='mean') if benchmark == 'regression' else CE()
                isolation(Prediction(learner.net), (x, target), spec,
                          {'targets': target, 'task': task, 'step': step, 'point': 'probe',
                           'within_task_step': (step-1)%hook.task_steps+1, 'buffer_task_purity': 1.0,
                           'loss_fn': lambda z, c: learner.loss_func(z, c['targets'].to(z) if benchmark == 'regression' else c['targets'])})
            hook.observe = observe
        logs = run_native(benchmark, params, tmp_path, hook)
        states.append((learner_state(hook.last_learner), logs))
    _assert_bitwise(*states)


@pytest.mark.parametrize('benchmark', ['regression', 'mnist', 'imagenet'])
@pytest.mark.parametrize('arm', ['bp', 'cbp'])
def test_resume_task_boundary_restores_rng_logs_and_learner(benchmark, arm, tmp_path):
    params = fixture(benchmark, arm, tmp_path, tasks=3)
    reset()
    full = LearnHook(benchmark, arm, params, actual_measure if arm == 'bp' else None)
    logs = run_native(benchmark, params, tmp_path, full)
    expected = learner_state(full.last_learner)
    checkpoint = {}
    class Stop(Exception): pass
    def save(state):
        if state['step'] == full.task_steps:
            checkpoint.update(copy.deepcopy(state)); raise Stop()
    reset()
    partial = LearnHook(benchmark, arm, params, actual_measure if arm == 'bp' else None, checkpoint=save)
    with pytest.raises(Stop): run_native(benchmark, params, tmp_path, partial)
    # Scramble all streams before restart, then reproduce native initialization.
    reset(991); reset()
    resumed = LearnHook(benchmark, arm, params, actual_measure if arm == 'bp' else None, resume=checkpoint)
    actual = run_native(benchmark, params, tmp_path, resumed)
    _assert_bitwise(expected, learner_state(resumed.last_learner))
    _assert_bitwise(logs, actual)
    _assert_bitwise(full.rows, resumed.rows)
    # MNIST checkpoints at the next task's pre-training SVD, before learn clears the previous task buffer.
    assert len(checkpoint['buffer']) == (64 if benchmark == 'mnist' else 0)
    if arm == 'bp':
        # The production schedule measures tasks 1, 2, 5, ...; task 3 is
        # deliberately unmeasured. ImageNet fills its 100-example fixture
        # buffer on step 1, where its two-step first/mid points coincide.
        expected_points = [('first', 1)] if benchmark == 'imagenet' else [('first', 64), ('mid', 80)]
        assert [(r['task'], r['point'], r['within_task_step']) for r in full.rows] == [
            (task, point, step) for task in (1, 2) for point, step in expected_points]
        assert all(r['buffer_task_purity'] == 1.0 for r in full.rows)
    _assert_bitwise(full.results.state(), resumed.results.state())


@pytest.mark.parametrize('benchmark', ['regression', 'mnist', 'imagenet'])
def test_first_three_tasks_data_and_permutations_native_stream(benchmark, tmp_path):
    params = fixture(benchmark, 'bp', tmp_path, tasks=3)
    streams = []
    for enabled in (False, True):
        reset()
        stream = []
        def observe(learner, x, target, task, step):
            stream.append((task, x.detach().clone(), target.detach().clone()))
        def measure(model, batch, spec, ctx, *, generator):
            current = [entry for entry in stream if entry[0] == ctx['task']]
            expected_x = torch.cat([entry[1] for entry in current])[-64 if benchmark != 'imagenet' else -100:]
            expected_y = torch.cat([entry[2] for entry in current])[-64 if benchmark != 'imagenet' else -100:]
            _assert_bitwise(batch, (expected_x, expected_y))
            assert ctx['buffer_task_purity'] == 1.0
            return actual_measure(model, batch, spec, ctx, generator=generator)
        hook = LearnHook(benchmark, 'bp', params, measure if enabled else None, observe=observe)
        run_native(benchmark, params, tmp_path, hook)
        streams.append(stream)
    _assert_bitwise(*streams)
    assert {row[0] for row in streams[0]} == {1, 2, 3}
    if benchmark == 'mnist':
        # Independent oracle applies upstream's cumulative pixel AND sample
        # permutations, including the full 60k permutation even in tiny runs.
        with open(tmp_path/'data/mnist_', 'rb') as f: x, y, _, _ = pickle.load(f)
        np.random.seed(13)
        expected = []
        for task in range(1, 4):
            x = x[:, np.random.permutation(784)]
            order = np.random.permutation(60000); x, y = x[order], y[order]
            expected.extend((task, x[i:i+1], y[i:i+1]) for i in range(160))
        _assert_bitwise(streams[0], expected)


@pytest.mark.parametrize('index', range(5))
def test_native_cnn_all_sites_pool_conv_fc_fd_adjoint_zero_B(index):
    from lop.nets.conv_net import ConvNet
    reset()
    model = Prediction(ConvNet()).double()
    site = sites_for(model)[index]
    x = torch.randn(1, 3, 32, 32, dtype=torch.float64)
    first = site.first_layer.layer
    A = {'weight': first.weight[:1].detach(), 'bias': first.bias[:1].detach()}
    with block_products(model, site, A, x) as p:
        torch.testing.assert_close(p._call({}), model(x).flatten(), rtol=0, atol=0)
        v = torch.randn(sum(t.numel() for _, t in p.extension_refs), dtype=torch.float64)
        u = torch.randn(2, dtype=torch.float64)
        torch.testing.assert_close(p.JEv(v)@u, v@p.JET(u), rtol=1e-9, atol=1e-9)
        pieces = v.split([t.numel() for _, t in p.extension_refs]); eps = 1e-5
        plus = p._call({n: t+eps*a.reshape_as(t) for (n, t), a in zip(p.extension_refs, pieces)})
        minus = p._call({n: t-eps*a.reshape_as(t) for (n, t), a in zip(p.extension_refs, pieces)})
        torch.testing.assert_close(p.JEv(v), (plus-minus)/(2*eps), rtol=1e-5, atol=1e-8)
    A, captured = tiny_V(model, site, [(x, torch.tensor([0]))], 1, cotangents=[torch.tensor([[1., -1.]])])
    assert torch.isfinite(A['weight']).all()


def test_recipes_gate_locked_and_hook_restored(tmp_path):
    for benchmark, lr in [('regression', .01), ('mnist', .003), ('imagenet', .01)]:
        for arm in ['bp', 'cbp']:
            params, meta = recipe(benchmark, arm, RATES[0] if benchmark == 'regression' and arm == 'cbp' else None)
            assert params['step_size'] == lr and meta['cfg_source'].endswith('.json')
    def rows(values, key): return [dict(task=i+1, **{key: v}) for i, v in enumerate(values)]
    bp = rows([.9]*10+[.5]*10, 'online_accuracy')
    cbp = rows([.9]*10+[.89]*10, 'online_accuracy')
    assert baseline_gate('mnist', bp, {'cbp': cbp})['status'] == 'PASS'
    reg = rows([1.]*10+[2.]*10, 'online_error')
    arms = {str(rate): rows([1.]*20, 'online_error') for rate in RATES}
    arms['best'] = rows([.5]*20, 'online_error')
    gate = baseline_gate('regression', reg, arms)
    assert gate['status'] == 'PASS' and gate['cbp_selected_arm'] == 'best'
    from lop.algos.bp import Backprop
    original = Backprop.learn
    params = fixture('regression', 'bp', tmp_path)
    with pytest.raises(RuntimeError):
        with LearnHook('regression', 'bp', params).install(): raise RuntimeError('injected')
    assert Backprop.learn is original


def test_full_classification_measurement_rows_and_baseline_invalidation(tmp_path):
    from lop.nets.deep_ffnn import DeepFFNN
    from lop.algos.bp import Backprop
    from diagnostics.lop_measure import B5Measurement
    from experiments.stage_b5 import finalize, write_json
    reset()
    native = DeepFFNN(input_size=7, num_features=5, num_outputs=3, num_hidden_layers=2)
    learner = Backprop(native, loss='nll', step_size=.003)
    params, meta = recipe('mnist', 'bp')
    params.update(num_tasks=2, change_after=160)
    measurement = B5Measurement('mnist', {'schema_version': 'stage1d-v1', 'benchmark': 'mnist',
                               'arm': 'bp', 'code_commit': 'test', 'seed': 13, **meta}, ranks=[1], cg_iters=100)
    hook = LearnHook('mnist', 'bp', params, measurement)
    from diagnostics.vna_align import MeasurementBuffer
    hook.buffer = MeasurementBuffer(64)
    x, target = torch.randn(2, 7), torch.tensor([0, 1])
    with hook.install():
        for _ in range(64): learner.learn(x=x[:1], target=target[:1])
    rows = hook.rows
    assert len(rows) == 4
    for row in rows:
        assert row['schema_version'] == 'stage1d-v1' and row['step_size'] == .003
        assert row['measurement_m'] == 64 and row['buffer_m'] == 64
        assert row['point'] == 'first' and row['within_task_step'] == 64
        assert row['buffer_task_purity'] == 1.0
        assert len(__import__('json').loads(row['null_lambda_N1'])) == 8
        assert len(__import__('json').loads(row['null_lambda_N2'])) == 8
        assert row['null_kind'] == 'N2' and row['n2_centered'] is True
        assert row['damping_norm'] == 'outputs' and 'K_top_eigenvalues' in row
        assert row['rho_joint'] == next(r['rho_joint'] for r in rows if r['mu_rel'] == row['mu_rel'])
        assert row['cg_converged']
    path = tmp_path/'b5/mnist/bp'; path.mkdir(parents=True)
    tasks = [dict(task=i+1, online_accuracy=.9 if i < 10 else .5, online_error=1.) for i in range(20)]
    write_json(path/'summary.json', {'arm': 'bp', 'tasks': tasks, 'fake_data': True})
    write_json(path/'measurements.json', rows)
    cbp = path.parent/'cbp'; cbp.mkdir()
    write_json(cbp/'summary.json', {'arm': 'cbp', 'tasks': [dict(t, online_accuracy=.9) for t in tasks], 'fake_data': True})
    gate = finalize(tmp_path, 'mnist')
    assert gate['status'] == 'INVALID' and not gate['measurements_valid']


def test_notebook_pin_full_suite_archive_resume_budget_and_sources(tmp_path):
    from scripts.build_kaggle_vna_stage_b_nb1 import build
    commit = 'a'*40
    notebook = build(commit)
    scope = {}
    exec(''.join(notebook['cells'][1]['source']), scope)
    assert scope['REPO_COMMIT'] == commit
    sources = '\n'.join(''.join(c['source']) for c in notebook['cells'])
    # NB1 now includes Stage A's CIFAR diagnostics; ImageNet was dropped from its lanes.
    assert 'DATA_ROOT' in sources and 'cifar-100-python' in sources
    assert 'ImageNet-32 is dropped' in sources
    for text in ['requirements-upstream.txt', 'fetch_third_party.sh', 'REQUIRE_UPSTREAM',
                 'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', '--resume', 'GPU_HOURS_BUDGET',
                 'ALLOW_OVER_BUDGET', 'STAGE_A / "summary.json"', 'json.loads(path.read_text())["mu_rel_default"]',
                 '--mu-rel-default-source', 'WARNING: Stage A failed', 'archive_stop.wait(900)', 'finalize(STAGE_B, benchmark']:
        assert text in sources
    assert '"-m", "pytest", "-q"], cwd=REPO' in sources
    for i, c in enumerate(notebook['cells']):
        if c['cell_type'] == 'code': compile(''.join(c['source']), f'cell-{i}', 'exec')
    # All three Stage B notebooks must stop on a failing full-suite setup.
    from scripts.build_kaggle_vna_stage_b_rest import build as build_rest
    for number in (2, 3):
        other = build_rest(commit, number)
        assert '"-m", "pytest", "-q"], cwd=REPO' in ''.join(other['cells'][3]['source'])
        for i, cell in enumerate(other['cells']):
            if cell['cell_type'] == 'code': compile(''.join(cell['source']), f'nb{number}-cell-{i}', 'exec')
    from experiments.stage_b5 import restore_archives
    import tarfile
    import io
    archive = tmp_path/'b5.tar.gz'
    with tarfile.open(archive, 'w:gz') as f:
        data = b'resume'; info = tarfile.TarInfo('outputs/stageB/b5/mnist/bp/resume.pkl'); info.size = len(data)
        f.addfile(info, io.BytesIO(data))
    restore_archives(tmp_path, tmp_path/'restored')
    assert (tmp_path/'restored/b5/mnist/bp/resume.pkl').read_bytes() == b'resume'


@pytest.mark.parametrize('spec_name, centered', [('CE', True), ('KDRepDistiller', True),
                                                  ('ICaRLPyCIL', True), ('MSE', False)])
def test_N2_loss_space_norm_and_zero_sum(spec_name, centered):
    from methods import vna_losses
    from diagnostics.stage_a import null_residual
    spec = getattr(vna_losses, spec_name)()
    assert spec.residual_is_zero_sum is centered
    generator = torch.Generator().manual_seed(42)
    g = torch.randn(64, 5, dtype=torch.float64, generator=generator)
    if centered: g -= g.mean(-1, keepdim=True)
    g[0] = 0
    null = null_residual(g, 'N2', generator, residual_is_zero_sum=spec.residual_is_zero_sum)
    torch.testing.assert_close(null.norm(dim=-1), g.norm(dim=-1), rtol=1e-12, atol=1e-12)
    if centered:
        torch.testing.assert_close(null.sum(-1), torch.zeros(64, dtype=g.dtype), rtol=0, atol=1e-12)
    else:
        assert (null.sum(-1)[1:].abs() > 1e-12).all()
        scalar = torch.tensor([[0.], [1.], [-2.], [3.]] * 16, dtype=torch.float64)
        signs = null_residual(scalar, 'N2', generator, residual_is_zero_sum=False)
        torch.testing.assert_close(signs.abs(), scalar.abs(), rtol=1e-12, atol=1e-12)
        assert (signs[scalar != 0] != 0).all()
        assert (signs > 0).any() and (signs < 0).any()


def test_scalar_regression_full_measurement_alignment_and_resume(tmp_path):
    from diagnostics.lop_measure import B5Measurement
    params = fixture('regression', 'bp', tmp_path)
    _, meta = recipe('regression', 'bp')
    measurement = B5Measurement('regression', {'schema_version': 'stage1d-v1',
        'benchmark': 'regression', 'arm': 'bp', 'code_commit': 'test', 'seed': 13, **meta}, cg_iters=100)
    reset()
    plain = LearnHook('regression', 'bp', params)
    expected_logs = run_native('regression', params, tmp_path, plain)
    expected_state = learner_state(plain.last_learner)
    reset()
    full = LearnHook('regression', 'bp', params, measurement, seed=13)
    from diagnostics.vna_align import MeasurementBuffer
    full.buffer = MeasurementBuffer(64)
    logs = run_native('regression', params, tmp_path, full)
    _assert_bitwise(expected_logs, logs)
    _assert_bitwise(expected_state, learner_state(full.last_learner))
    assert {(r['task'], r['point']) for r in full.rows} == {(1, 'first'), (1, 'mid'), (2, 'first'), (2, 'mid')}
    assert {r['rank'] for r in full.rows} == {1, 2, 4}
    assert len(full.rows) == 24
    import json
    for row in full.rows:
        assert row['within_task_step'] == (64 if row['point'] == 'first' else 80)
        assert row['measurement_m'] == 64 and row['buffer_task_purity'] == 1.0
        assert row['null_kind'] == 'N2' and row['n2_centered'] is False
        assert len(json.loads(row['null_lambda_N2'])) == 8
        assert len(json.loads(row['null_rho_N2'])) == 8
        assert all(np.isfinite(row[key]) for key in ('U', 'V', 'eta', 'rho', 'rho_joint', 'lambda1'))
        assert all(f'cbp_rr_{rate:g}_m' in row for rate in RATES)
    checkpoint = {}
    class Stop(Exception): pass
    def save(state):
        if state['step'] == full.task_steps:
            checkpoint.update(copy.deepcopy(state)); raise Stop()
    reset()
    partial = LearnHook('regression', 'bp', params, measurement, seed=13, checkpoint=save)
    partial.buffer = MeasurementBuffer(64)
    with pytest.raises(Stop): run_native('regression', params, tmp_path, partial)
    reset()
    resumed = LearnHook('regression', 'bp', params, measurement, seed=13, resume=checkpoint)
    resumed.buffer = MeasurementBuffer(64)
    resumed_logs = run_native('regression', params, tmp_path, resumed)
    _assert_bitwise(logs, resumed_logs)
    _assert_bitwise(expected_state, learner_state(resumed.last_learner))
    # Timing is observational, all numerical rows must replay identically.
    numerical = lambda rows: [{k: v for k, v in row.items() if k != 'wall_seconds'} for row in rows]
    _assert_bitwise(numerical(full.rows), numerical(resumed.rows))


@pytest.mark.parametrize('k', [0, 1, 2, 4])
def test_gir_swap_fills_missing_tiny_with_shared_random_rows(k):
    from diagnostics.lop_measure import swap_directions
    from methods.vna_select import neuron_norms
    random = {'weight': torch.tensor([[1., 2.], [2., 1.], [-1., 2.], [2., -1.]]),
              'bias': torch.zeros(4)}
    tiny = {'weight': torch.tensor([[-1., -2.], [-2., -1.], [1., -2.], [-2., 1.]])[:k],
            'bias': torch.zeros(k)}
    swap, random_arm, selected = swap_directions(tiny, random, 3.)
    assert selected == k
    assert len(torch.unique(swap['weight'], dim=0)) == 4
    torch.testing.assert_close(neuron_norms(swap), torch.full((4,), 3.))
    torch.testing.assert_close(neuron_norms(random_arm), torch.full((4,), 3.))
    if k < 4:
        for name in swap:
            torch.testing.assert_close(swap[name][k:], random_arm[name][:4-k], rtol=0, atol=0)
    if k == 0:
        _assert_bitwise(swap, random_arm)
