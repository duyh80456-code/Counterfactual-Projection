"""Stage-A contracts: residual scale, matrix-free math, gates and resume."""
import ast
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time

import pytest
import torch
from experiments.vna_support import tiny_model, tiny_candidate, isolated_rng
from methods.vna import dedicated_generator, extract_tiny_A, _resolve_site, VNAConfig, vna_block_correction, block_products
from methods.vna_ggn import GGNConfig, vna_exact_ggn
from methods.vna_joint import joint_correction, joint_products, _inverse
from methods.vna_select import tiny_U_from_cotangent, prepare_A, subspace_sim, random_A_like, extension_trace
from diagnostics.stage_a import (null_residual, empirical_p, invariant, residual, diagnostic_ratio,
                                 parser, run, load_json, read_csv, summarize)
from diagnostics import vna_gate as gate


@pytest.fixture
def toy():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    gen = dedicated_generator(125, 'stage-a-test')
    with isolated_rng(gen):
        model = tiny_model()
    # Prime upstream's lazy float32 convolution identity before conversion;
    # this lets the unchanged native tiny_candidate be the float64 oracle.
    site = model.growing_blocks()[0].name
    tiny_candidate(model, site, (torch.randn(4, 3, 8, 8, generator=gen), torch.tensor([0, 1, 2, 0])), 2)
    model.double()
    batch = (torch.randn(4, 3, 8, 8, generator=gen, dtype=torch.float64), torch.tensor([0, 1, 2, 0]))
    site = model.growing_blocks()[0].name
    yield model, site, batch
    torch.set_num_threads(threads)


def test_nulls_and_p_values():
    gen = dedicated_generator(3, 'null-test')
    g = torch.randn(20, 5, generator=gen, dtype=torch.float64)
    g -= g.mean(1, keepdim=True)
    n1 = null_residual(g, 'N1', gen)
    assert torch.equal(g.norm(dim=1).sort().values, n1.norm(dim=1).sort().values)
    assert n1.norm() == pytest.approx(float(g.norm()))
    n2 = null_residual(g, 'N2', gen)
    assert torch.allclose(n2.sum(1), torch.zeros(20, dtype=g.dtype), atol=1e-14)
    assert torch.allclose(n2.norm(dim=1), g.norm(dim=1), atol=1e-14)
    assert empirical_p(10, range(8)) == 1/9
    assert empirical_p(0, range(8)) == 1
    with pytest.raises(ValueError): empirical_p(0, [float('nan')])


def test_cotangent_tiny_parity_and_normalization(toy):
    m, s, b = toy
    m.eval()
    candidate = tiny_candidate(m, s, b, 2)
    expected, _ = extract_tiny_A(m, candidate)
    with invariant(m):
        A, values = tiny_U_from_cotangent(m, s, b[0], residual(m, b), 2)
    assert torch.allclose(values, torch.tensor(candidate.payload['tiny_eigenvalues'], dtype=values.dtype), rtol=1e-5, atol=1e-12)
    assert subspace_sim(A, expected) >= 1-1e-6
    ref, _ = prepare_A(m, s, A, 'tiny_U')
    for kind in ('tiny_U', 'tiny_V', 'greedy_V', 'random_trace', 'random_frob'):
        raw = random_A_like(A, generator=dedicated_generator(5, kind))
        with invariant(m):
            a, record = prepare_A(m, s, raw, kind, reference=ref, inputs=b[0])
        assert len(record['normalization_constant']) == len(raw['weight'])
        if kind == 'random_frob':
            assert a['weight'].norm() == pytest.approx(float(ref['weight'].norm()), rel=1e-14)
        if kind == 'random_trace':
            assert extension_trace(m, s, a, b[0]) == pytest.approx(extension_trace(m, s, ref, b[0]), rel=1e-10)


def test_subspace_overlap():
    gen = dedicated_generator(8, 'subspace')
    a = torch.randn(8, 1000, generator=gen, dtype=torch.float64)
    b = torch.randn(8, 1000, generator=gen, dtype=torch.float64)
    assert subspace_sim(a, a) == pytest.approx(1, abs=1e-12)
    assert subspace_sim(a, b) == pytest.approx(8/1000, abs=.003)


@pytest.mark.parametrize('norm', ['params', 'outputs'])
@pytest.mark.parametrize('cache', [False, True])
def test_joint_single_site_ggn_and_legacy(toy, cache, norm):
    m, s, b = toy
    A, _ = tiny_U_from_cotangent(m, s, b[0], residual(m, b), 2)
    A, _ = prepare_A(m, s, A, 'tiny_U')
    names = _resolve_site(m, s)[2]
    cfg = GGNConfig(project_bn=False, damping_norm=norm, cg_iters=1000, cg_tol=1e-10, cache_block_input=cache)
    expected, diag = vna_exact_ggn(m, s, A, b, cfg)
    with invariant(m):
        actual, V, rho, record = joint_correction(m, [s], {s: A}, b, cfg, o_params=names)
    actual_flat = torch.cat([actual[n].flatten() for n in names])
    expected_flat = torch.cat([expected[n].flatten() for n in names])
    assert float((actual_flat-expected_flat).norm()/expected_flat.norm()) <= 1e-6
    assert V == pytest.approx(diag['V'], rel=1e-6)
    assert record['mu'] == diag['mu']
    assert record['mu'] == pytest.approx(cfg.damping_rel * record['trace_F'] / (record['p'] if norm == 'params' else record['n_out']))
    assert record['mu_rel_equiv_params'] == pytest.approx(record['mu'] / (record['trace_F']/record['p']))
    legacy = VNAConfig(cg_iters=1000, cg_tol=1e-10, cache_block_input=cache)
    expected, diag = vna_block_correction(m, s, A, b, legacy, apply_gain_guard=False)
    actual, V, rho, record = joint_correction(m, [s], {s: A}, b, legacy, o_params=names)
    actual_flat = torch.cat([actual[n].flatten() for n in names])
    expected_flat = torch.cat([expected[n].flatten() for n in names])
    assert float((actual_flat-expected_flat).norm()/expected_flat.norm()) <= 1e-6
    assert V == pytest.approx(diag['V'], rel=1e-6)
    assert rho == pytest.approx(diag['rho'], rel=1e-6)


def test_joint_full_parameter_dense_oracle_and_invariance(toy):
    m, s, b = toy
    A, _ = tiny_U_from_cotangent(m, s, b[0], residual(m, b), 2)
    cfg = GGNConfig(project_bn=False, cg_iters=1000, cg_tol=1e-10)
    m.train(); list(m.modules())[-1].eval()
    for p in m.parameters(): p.grad = torch.ones_like(p)
    with invariant(m):
        d, V, rho, record = joint_correction(m, [s], {s: A}, b, cfg, mu=.03)
        with joint_products(m, [s], {s: A}, b[0]) as p:
            from methods.vna_ggn import GGNProducts
            op = GGNProducts(p, b[1])
            eye = torch.eye(op.g.numel(), dtype=torch.float64)
            J = torch.stack([p.JT(e) for e in eye])
            E = torch.stack([p.JET(e) for e in eye])
            H = torch.stack([op.H(e) for e in eye], 1)
            F = J.T @ H @ J
            M = F + .03 * torch.eye(len(F), dtype=F.dtype)
            u = torch.linalg.solve(M, J.T @ op.g)
            oracle = torch.linalg.solve(M, J.T @ H @ E @ E.T @ H @ J @ u)
            actual = torch.cat([d[n].flatten() for n, _ in p.parameter_refs])
            assert float((actual-oracle).norm()/oracle.norm()) < 1e-6
            assert V == pytest.approx(float((E.T @ H @ J @ u).square().sum()), rel=1e-6)


def gate_rows():
    return [dict(epoch=i*20, rank=16, mu_rel=.01, cg_converged=True,
                 rho_joint=.3, significant_N1=True, V_tiny_V=2., V_random_frob=1., train_acc=.98)
            for i in range(1, 11)]


def test_gate_constants_windows_and_failures():
    assert (gate.RHO_MIN, gate.RHO_FRAC, gate.N1_FRAC, gate.TINYV_FRAC, gate.TRAIN_ACC_MAX, gate.GATE_RANK) == (.3, .5, .5, .7, .99, 16)
    rows = gate_rows()
    cfg = gate.setting_config('B1')
    assert gate.evaluate_gate(rows, cfg)['status'] == 'PASS'
    rows[-1]['train_acc'] = .99
    assert gate.evaluate_gate(rows, cfg)['status'] == 'FAIL'
    rows = gate_rows()
    for r in rows[:4]: r['V_tiny_V'] = 0
    assert gate.evaluate_gate(rows, cfg)['tiny_V_fraction'] == .6
    assert gate.evaluate_gate(rows, cfg)['status'] == 'FAIL'
    assert gate.evaluate_gate([], cfg)['status'] == 'FAIL'
    assert not gate.setting_config('B4')['window']({'epoch': 150})
    assert gate.setting_config('B3')['window']({'task': 2, 'end_of_task': False})
    assert not gate.setting_config('B3')['window']({'task': 2, 'end_of_task': True})
    cfg = gate.setting_config('B5', tasks_run=200)
    assert not cfg['window']({'task': 100})
    rows = [dict(r, task=150) for r in gate_rows()]
    result = gate.evaluate_gate(rows, cfg)
    assert result['status'] == 'PASS' and result['conditions']['train_acc'] == 'n/a'
    rows = [dict(r, cg_converged=False) for r in rows]
    assert gate.evaluate_gate(rows, cfg)['status'] == 'FAIL'


def test_a5_tiny_residual_cg_and_guarded_ratio():
    # Old CG's absolute denominator threshold erases a nonzero tiny RHS.
    from projection.cg import conjugate_gradient
    rhs = torch.tensor([1e-20, 2e-20], dtype=torch.float64)
    old = conjugate_gradient(lambda v: 2*v, rhs, eps=1e-30)
    assert old.iterations == 0 and not old.converged
    d, record = _inverse(lambda v: 2*v, rhs, GGNConfig())
    assert record['converged'] and torch.allclose(d, rhs/2, rtol=1e-12, atol=0)
    assert diagnostic_ratio(1e-30, 2e-30) == .5
    assert diagnostic_ratio(1, float('inf')) is None
    assert diagnostic_ratio(1, 0) is None


def test_stage_a_smoke_resume_schema_and_identity(tmp_path):
    start = time.perf_counter()
    args = parser().parse_args(['all', '--smoke', '--fake-data', '--device', 'cpu', '--output', str(tmp_path)])
    result = run(args)
    assert time.perf_counter()-start < 120
    for name in ('a5.csv', 'null.csv', 'mu_sweep.csv', 'selection.csv', 'joint.csv', 'solver.csv'):
        rows = read_csv(tmp_path / name)
        assert rows and all(r['schema_version'] == 'stage1d-v1' for r in rows)
        assert all({'code_commit', 'seed', 'device', 'wall_seconds'} <= r.keys() for r in rows)
    markers = list((tmp_path/'completed').rglob('*.json'))
    assert len(markers) == 6
    before = {p: p.read_bytes() for p in markers}
    run(args)
    assert all(p.read_bytes() == before[p] for p in markers)
    payload = torch.load(tmp_path/'smoke_recipe.pt', weights_only=False)
    payload['extra'] = True
    torch.save(payload, tmp_path/'smoke_recipe.pt')
    with pytest.raises(ValueError, match='resume identity'): run(args)
    marker = markers[0]
    data = json.loads(marker.read_text()); data['schema_version'] = 'old'
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='schema_version'): load_json(marker)


def test_summary_preregistered_rules(tmp_path):
    from diagnostics.stage_a import write_csv
    meta = dict(schema_version='stage1d-v1', code_commit='test', seed=0, device='cpu', wall_seconds=0)
    rows = [dict(meta, epoch=40, site='joint', mu_rel=mu, rho=rho, cg_converged=True)
            for mu, rho in ((.01, .1), (.001, .3))]
    write_csv(tmp_path/'mu_sweep.csv', rows)
    args = parser().parse_args(['summary', '--output', str(tmp_path)])
    assert summarize(args)['mu_rel_default'] == .001
    rows[1]['cg_converged'] = False
    write_csv(tmp_path/'mu_sweep.csv', rows)
    assert summarize(args)['mu_rel_default'] == .01


def test_notebook_pin_order_tests_and_python_cells():
    import importlib.util
    path = Path('scripts/build_kaggle_vna_stage_a_notebook.py')
    import sys
    sys.path.insert(0, str(path.parent.resolve()))
    try:
        spec = importlib.util.spec_from_file_location('stage_a_builder', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        notebook = module.build('a'*40)
    finally:
        sys.path.pop(0)
    sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    for source in sources: ast.parse(source)
    joined = '\n'.join(sources)
    assert 'REPO_COMMIT = "'+'a'*40+'"' in joined
    assert 'worker, 0, ["null", "selection"]' in joined
    assert 'worker, 1, ["joint", "mu", "solver"]' in joined
    assert 'glob("test_vna*.py")' in joined
    assert 'archive_stop.wait(3600)' in joined
    assert 'PHASE0_DIR' in joined


def test_archive_contains_stage_a_and_safe_restore(tmp_path):
    import tarfile
    import io
    from experiments.vna_kaggle import archive_outputs
    from diagnostics.stage_a import restore_archives
    run_root = tmp_path/'run'
    marker = run_root/'outputs/stageA/completed/null/epoch_020.json'
    marker.parent.mkdir(parents=True)
    marker.write_text('{"schema_version":"stage1d-v1"}')
    archive = archive_outputs(run_root, folders=('outputs/stageA',))
    restored = tmp_path/'restored'
    attachments = tmp_path/'attachments'; attachments.mkdir()
    import shutil
    shutil.copy2(archive, attachments/'previous.tar.gz')
    restore_archives(attachments, restored)
    assert (restored/'completed/null/epoch_020.json').read_bytes() == marker.read_bytes()
    with tarfile.open(attachments/'bad.tar.gz', 'w:gz') as tar:
        info = tarfile.TarInfo('outputs/stageA/../escaped.txt'); info.size = 1
        tar.addfile(info, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='unsafe archive'): restore_archives(attachments, restored)


def test_matrix_free_greedy_matches_dense_objective(toy):
    from methods.vna_select import greedy_V
    m, s, b = toy
    A, _ = tiny_U_from_cotangent(m, s, b[0], residual(m, b), 2)
    z = residual(m, b)
    kwargs = dict(random_count=1, r_max=2, theta=1., nu=.01, seed=55)
    dense, dd = greedy_V(m, s, A, b[0], z, **kwargs)
    with invariant(m):
        free, fd = greedy_V(m, s, A, b[0], z, matrix_free=True, **kwargs)
    assert fd['cg_converged']
    assert fd['pool_gain'] == pytest.approx(dd['pool_gain'], rel=1e-5)
    assert fd['gains'][-1] == pytest.approx(dd['gains'][-1], rel=1e-5)
    assert fd['selected'] == dd['selected']


def test_joint_multiple_sites_sum_virtual_kernels(toy):
    from gromo.containers.resnet import init_full_resnet_structure
    m, _, b = toy
    with isolated_rng(dedicated_generator(51, 'two-site-model')):
        m.core = init_full_resnet_structure(input_shape=(3, 8, 8), out_features=3,
            reduction_factor=1., number_of_blocks_per_stage=2, inplanes=3,
            nb_stages=1, small_inputs=True, use_preactivation=False, device='cpu').double()
    sites = [r.name for r in m.growing_blocks()]
    assert len(sites) == 2
    A = {s: {'weight': m.block(s).first_layer.layer.weight.detach()[:2].clone(), 'bias': None} for s in sites}
    z = torch.randn(12, dtype=torch.float64, generator=dedicated_generator(51, 'two-site-z'))
    with invariant(m):
        images = []
        for s in sites:
            with block_products(m, s, A[s], b[0]) as p:
                images.append(p.KE(z))
        with joint_products(m, sites, A, b[0]) as p:
            assert torch.allclose(p.KE(z), sum(images), rtol=1e-10, atol=1e-12)
            assert len(p.parameter_refs) == len(list(m.parameters())) - len(p.extension_refs) - sum(1 for n, _ in m.named_parameters() if 'extended_output_layer' in n)


def test_a5_summary_evidence():
    from diagnostics.stage_a import a5_decision
    assert a5_decision([]) == 'unresolved'
    row = dict(epoch=100, old_ratio=0., fixed_ratio=.1)
    assert a5_decision([row]) == 'resolved'
    for change in (dict(epoch=90), dict(old_ratio=.1), dict(fixed_ratio=0.), dict(fixed_ratio=None)):
        assert a5_decision([{**row, **change}]) == 'unresolved'


def test_trace_estimator_common_random_numbers(toy):
    from methods.vna_select import extension_trace_estimate, trace_matched_random_A
    m, site, batch = toy
    A = {'weight': m.block(site).first_layer.layer.weight.detach()[:2], 'bias': None}
    with invariant(m):
        exact = extension_trace(m, site, A, batch[0])
        estimate, record = extension_trace_estimate(m, site, A, batch[0], seed=42, probes=256)
        _, matched = trace_matched_random_A(m, site, A, batch[0], A=A, seed=42, estimator='hutchinson')
    assert abs(estimate-exact) <= 5*record['standard_error']
    assert matched['scale'] == 1.
    assert matched['trace_target_record'] == matched['trace_random_record']
    with pytest.raises(ValueError): extension_trace_estimate(m, site, A, batch[0], seed=42, probes=63)


def test_damping_default_and_validation():
    assert GGNConfig().damping_norm == 'params'
    assert gate.setting_config('B1')['damping_norm'] == 'outputs'
    with pytest.raises(ValueError): GGNConfig(damping_norm='other')


def test_phase0_normalized_cg_regression():
    from methods.vna import _phase0_cg
    rhs = torch.tensor([1e-20, 2e-20], dtype=torch.float64)
    kwargs = dict(eps=1e-30, tolerance=1e-8, preconditioner=lambda v: v/2)
    old = _phase0_cg(lambda v: 2*v, rhs, **kwargs)
    fixed = _phase0_cg(lambda v: 2*v, rhs, normalize_rhs=True, **kwargs)
    assert old.breakdown and not old.converged
    assert fixed.converged and not fixed.breakdown
    assert torch.allclose(fixed.solution, rhs/2, rtol=1e-12, atol=0.)


def test_a5_phase0_oracle_and_state_missing():
    from diagnostics.stage_a import Measurement, recipe_scheduler_state
    from experiments.vna_support import setup_data, augmented_batch
    from methods.vna import random_A_like as legacy_random
    with isolated_rng(dedicated_generator(0, 'model')):
        model = tiny_model()
    data = setup_data('', True, 0)
    args = parser().parse_args(['a5', '--smoke', '--fake-data', '--device', 'cpu'])
    measurement = Measurement(model, data, args, 100)
    with invariant(model): rows = measurement.measure_a5({'epoch': 100})
    cfg = VNAConfig(seed=0, stat_samples=2, vna_batch=1, r_max=1, cg_iters=3, hutchinson_probes=1)
    stat = augmented_batch(data[0], data[2], 2, dedicated_generator(0, 'statistics'), torch.device('cpu'))
    batch = augmented_batch(data[0], data[2], 1, dedicated_generator(0, 'vna'), torch.device('cpu'))
    site = measurement.sites[0]
    candidate = tiny_candidate(model, site, stat, 1, 0)
    A, _ = extract_tiny_A(model, candidate)
    random = legacy_random(A, generator=dedicated_generator(0, f'random-A:{site}:1'))
    _, old = vna_block_correction(model, site, A, batch, cfg)
    _, control = vna_block_correction(model, site, random, batch, cfg,
                                     damping_cache=old['damping_cache'], apply_gain_guard=False)
    row = rows[0]
    assert row['old_V'] == old['V']
    assert row['old_random_V'] == control['V']
    assert row['old_ratio'] == diagnostic_ratio(old['V'], control['V'])
    assert row['checkpoint_scheduler_state'] is None
    assert row['recipe_lr_reconstructed'] == pytest.approx([.01])
    assert recipe_scheduler_state(90)['_last_lr'] == [.1]
    for path in ('old_', 'fixed_'):
        for control_prefix in ('', 'random_'):
            for solve in ('cg_z', 'cg_q'):
                assert all(path+control_prefix+solve+'_'+key in row for key in
                           ('iterations', 'converged', 'relative_residual', 'breakdown'))


def test_notebook_cost_budget_and_a5_first(tmp_path):
    from scripts.build_kaggle_vna_stage_a_notebook import build
    from types import SimpleNamespace
    notebook = build('a'*40)
    source = next(''.join(c['source']) for c in notebook['cells']
                  if 'cost_projection.json' in ''.join(c['source']))
    recipe = tmp_path/'recipe'; recipe.mkdir()
    for epoch in (90, 100): (recipe/f'epoch_{epoch:03d}.pt').touch()
    stage = tmp_path/'stageA'; stage.mkdir()
    (stage/'cost_projection.json').write_text(json.dumps(dict(
        cost_projection={'selection': {'projected_gpu_hours': 3.1}},
        total_projected_gpu_hours=3.1, smoke_gpu_hours=.1)))
    commands = []
    context = dict(sys=SimpleNamespace(executable='python'),
        subprocess=SimpleNamespace(run=lambda command, **kwargs: commands.append(command)),
        common=[], RECIPE=recipe, REPO=tmp_path, ENV={}, STAGE_A=stage,
        json=json, archive_outputs=lambda root: None, RUN_ROOT=tmp_path,
        GPU_HOURS_BUDGET=3, ALLOW_OVER_BUDGET=False)
    with pytest.raises(RuntimeError, match='GPU_HOURS_BUDGET'): exec(source, context)
    assert commands[0][4] == 'a5' and commands[0][-1] == 'cuda:0'
    assert commands[1][4] == 'cost'
    context['ALLOW_OVER_BUDGET'] = True
    exec(source, context)


def test_params_default_bitwise_and_gate_convention(toy):
    m, site, batch = toy
    A = {'weight': m.block(site).first_layer.layer.weight.detach()[:1], 'bias': None}
    cfg = GGNConfig(project_bn=False, cg_iters=100)
    default, before = vna_exact_ggn(m, site, A, batch, cfg)
    explicit, after = vna_exact_ggn(m, site, A, batch, replace(cfg, damping_norm='params'))
    assert all(torch.equal(default[n], explicit[n]) for n in default)
    assert all(before[k] == after[k] for k in ('mu', 'V', 'raw_gain', 'realized_gain', 'cg'))
    rows = [dict(r, damping_norm='params') for r in gate_rows()]
    assert gate.evaluate_gate(rows, gate.setting_config('B1'))['eligible_points'] == 0
