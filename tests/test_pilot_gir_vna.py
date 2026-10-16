"""Native identity checks, exact correction geometry, pool lifecycle and resume."""
import copy
from contextlib import nullcontext
import pickle
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from adapters.lop_hooks import LearnHook, Prediction, sites_for, learner_state
from adapters.pilot_lop import LoPPilot
from diagnostics.vna_align import MeasurementBuffer, _global_state, _assert_bitwise
from experiments.pilot_gir_vna import PilotEngine, exact_cache, scale_rows, choose_interval
from experiments.stage_b5 import upstream_imports, run_native
from experiments.vna_resume import rng_state
from methods.vna_ggn import GGNConfig, ngd_control
from methods.vna_joint import joint_correction, joint_products
from methods.vna_losses import MSE
from methods.vna_select import tiny_V, neuron_norms
from tests.test_vna_stage_b5 import fixture, reset


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    with _global_state(): yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('base,options', [('bp', dict(coefficient=0.)),
    ('cbp', dict(gir=True, upstream_random=True))])
def test_native_pilot_bitwise_logs_weights_optimizer_rng(base, options, tmp_path, monkeypatch):
    upstream_imports()
    params = fixture('regression', base, tmp_path)
    if base=='cbp': params['replacement_rate'] = .1  # force genuine reset events
    results = []
    for enabled in (False, True):
        reset()
        intervention = LoPPilot(PilotEngine(**options))
        hook = LearnHook('regression', base, params, intervention=intervention if enabled else None)
        hook.buffer = MeasurementBuffer(128)
        with intervention.install() if enabled else nullcontext():
            logs = run_native('regression', params, tmp_path, hook)
        results.append(dict(logs=logs, learner=learner_state(hook.last_learner), rng=rng_state(),
                            logged=list(hook.results.tasks())))
    _assert_bitwise(*results)


def small_problem():
    upstream_imports()
    from lop.nets.ffnn import FFNN
    torch.manual_seed(12)
    model = Prediction(FFNN(input_size=3, num_features=3)).double().eval()
    batch = torch.randn(6,3,dtype=torch.float64), torch.randn(6,1,dtype=torch.float64)
    return model, sites_for(model), batch, MSE(reduction='mean'), {'targets': batch[1]}


def test_exact_ngd_matches_vna_norm_and_positive_gradient():
    model, sites, batch, spec, ctx = small_problem()
    cfg = GGNConfig(project_bn=False)
    cache, factor = exact_cache(model, sites, batch, spec, ctx, cfg)
    with torch.no_grad(): e = spec.residual(model(batch[0]), ctx)
    a, _ = tiny_V(model, sites[0], [batch], 1, cotangents=[factor.z(e, cache.mu)], cfg=cfg)
    d, _, _, record = joint_correction(model, sites, {sites[0]:a}, batch, cfg,
        mu=cache.mu, trace_F=cache.trace, loss_spec=spec, ctx=ctx, output_factor=factor)
    ngd, control = ngd_control(model, sites[0], a, batch, cfg, mu=cache.mu,
        trace_F=cache.trace, loss_spec=spec, ctx=ctx, reference=d, norm_match=True, output_factor=factor)
    norm = lambda delta: sum(float(v.double().square().sum()) for v in delta.values())**.5
    assert norm(d) > 0
    assert norm(d) == pytest.approx(norm(ngd), rel=1e-12)
    assert record['realized_gain'] >= -1e-12
    assert control['realized_gain'] >= -1e-12
    raw, _ = ngd_control(model, sites[0], a, batch, cfg, mu=cache.mu,
        trace_F=cache.trace, loss_spec=spec, ctx=ctx, output_factor=factor)
    with joint_products(model, sites, {sites[0]:a}, batch[0]) as products:
        from methods.vna_ggn import GGNProducts
        op = GGNProducts(products, batch[1], spec, ctx)
        matrix = torch.stack([op.F(v) for v in torch.eye(op.size, dtype=torch.float64)], 1)
        expected = torch.linalg.solve(matrix+cache.mu*torch.eye(op.size), op.gradient)
        flat = torch.cat([raw[n].flatten() for n,_ in products.parameter_refs])
        torch.testing.assert_close(flat, expected, rtol=1e-8, atol=1e-9)


def test_pool_consumes_once_norm_positive_refresh_and_resume(monkeypatch):
    model, sites, batch, spec, ctx = small_problem()
    site = sites[0]; engine = PilotEngine(gir=True, pool=3)
    engine.begin_task(1)
    # Supply distinguishable positive rows. The third row is nonpositive and excluded.
    rows = torch.eye(3, dtype=torch.float64)
    def select(*a, **kw):
        return {'weight':rows.clone(), 'bias':torch.zeros(3,dtype=torch.float64)}, {
            'generalized_eigenvalues':torch.tensor([3.,2.,0.])}
    monkeypatch.setattr('experiments.pilot_gir_vna.tiny_V', select)
    first = engine.take(model, site, batch, spec, ctx, 1, 2., 1)
    state = copy.deepcopy(engine.state_dict())
    restored = PilotEngine(); restored.load_state_dict(state)
    second = restored.take(model, site, batch, spec, ctx, 2, 2., 2)
    assert len(second['weight'])==1
    assert not torch.equal(first['weight'], second['weight'])
    assert len([r for r in restored.events if r['kind']=='gir_refresh'])==1
    torch.testing.assert_close(neuron_norms(first), torch.tensor([2.], dtype=torch.float64))
    assert restored.events[-1]['gir_random_fill']==1
    restored.take(model, site, batch, spec, ctx, 1, 2., 3)
    assert len([r for r in restored.events if r['kind']=='gir_refresh'])==2
    restored.begin_task(2); assert not restored.pools


def test_gir_exact_selected_count_outgoing_zero_and_neutral_incoming(tmp_path, monkeypatch):
    upstream_imports()
    from lop.algos.cbp import ContinualBackprop
    from lop.nets.ffnn import FFNN
    learner = ContinualBackprop(FFNN(input_size=3,num_features=3).double(), replacement_rate=.1,
                               maturity_threshold=0, device='cpu')
    params = dict(flip_after=10,num_data_points=10)
    engine = PilotEngine(gir=True)
    intervention = LoPPilot(engine)
    hook = LearnHook('regression','cbp',params,intervention=intervention)
    hook.buffer = MeasurementBuffer(128)
    x, y = torch.randn(4,3,dtype=torch.float64), torch.zeros(4,1,dtype=torch.float64)
    intervention.before(hook,learner,x,y)
    # Removing an already-zero outgoing channel is exactly function preserving.
    units = torch.tensor([1,2]); learner.gnt.ages[0][units] = 10
    with torch.no_grad(): learner.net.layers[2].weight[:, units] = 0
    before = Prediction(learner.net)(x).detach().clone()
    untouched = learner.net.layers[0].weight[0].detach().clone()
    with intervention.install(): learner.gnt.gen_new_features([units],[2])
    after = Prediction(learner.net)(x).detach()
    torch.testing.assert_close(before,after,rtol=0,atol=1e-12)
    assert torch.equal(untouched,learner.net.layers[0].weight[0])
    assert torch.count_nonzero(learner.net.layers[2].weight[:,units])==0
    assert engine.events[-1]['replaced']==2
    assert torch.count_nonzero(learner.gnt.ages[0][units])==0


def test_correction_rng_and_modes_and_rho_gate():
    model, sites, batch, spec, ctx = small_problem()
    model.train(); model.native.layers[0].eval()
    modes = [m.training for m in model.modules()]
    before = copy.deepcopy(model.state_dict()); rng = rng_state()
    engine = PilotEngine(coefficient=1., every=1, rho_min=float('inf'))
    row = engine.correction(model, sites, batch, spec, ctx, 1, .01)
    assert row['skipped'] and row['rho_gate_skip']
    assert modes == [m.training for m in model.modules()]
    _assert_bitwise(before, model.state_dict()); _assert_bitwise(rng, rng_state())


def test_interval_rule_is_fixed_before_outcomes():
    points = [dict(task=1,n_out=1280,vna_seconds=10.,iteration_seconds=1.),
              dict(task=2,n_out=2560,vna_seconds=20.,iteration_seconds=1.)]
    assert choose_interval(points, 60)['K']==100
    assert choose_interval(points, 150)['K']==100
    with pytest.raises(ValueError): choose_interval([dict(n_out=1,vna_seconds=100.,iteration_seconds=.001)],10)


def test_active_interventions_resume_exactly_mid_task(tmp_path):
    upstream_imports()
    params = fixture('regression','cbp',tmp_path)
    params['replacement_rate'] = .01
    def make(resume=None, checkpoint=None, observe=None):
        pilot = LoPPilot(PilotEngine(seed=13,gir=True,random=True,coefficient=1.,every=40))
        hook = LearnHook('regression','cbp',params, intervention=pilot, resume=resume,
                         checkpoint=checkpoint,observe=observe)
        hook.buffer=MeasurementBuffer(128)
        return hook,pilot
    reset(); full,pilot = make()
    with pilot.install(): full_logs=run_native('regression',params,tmp_path,full)
    state={}
    def checkpoint(s):
        if s['step']==200:
            state.clear(); state.update(copy.deepcopy(s)); raise Pause
        return 0
    class Pause(Exception): pass
    def observe(learner,x,y,task,step):
        if step==200:
            # -inf (not 0): time.monotonic() counts from boot, so a fresh container
            # can be < checkpoint_seconds and a 0 reference never triggers.
            partial.last_checkpoint = float('-inf')  # next native pre-learn checkpoint is a safe boundary
    reset(); partial,pp = make(checkpoint=checkpoint,observe=observe)
    with pp.install(), pytest.raises(Pause): run_native('regression',params,tmp_path,partial)
    reset(); resumed,rp = make(resume=state)
    with rp.install(): resumed_logs=run_native('regression',params,tmp_path,resumed)
    _assert_bitwise(full_logs,resumed_logs)
    _assert_bitwise(learner_state(full.last_learner),learner_state(resumed.last_learner))
    assert pilot.engine.positions==rp.engine.positions
    _assert_bitwise(pilot.engine.generators['cpu'].get_state(),rp.engine.generators['cpu'].get_state())


def test_notebook_builders_pin_code_and_required_lanes():
    import ast
    from scripts.build_kaggle_vna_pilot_b5 import build as b5
    from scripts.build_kaggle_vna_stage_b_nb2 import build as b3
    for builder in (b5,b3):
        notebook = builder('a'*40)
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        for cell in notebook['cells']:
            if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
        assert 'a'*40 in code and 'archive_outputs' in code and 'VNA_DEADLINE' in code
        assert 'launch_logged' in code and 'ALLOW_OVER_BUDGET = True' in code
    cpu='\n'.join(''.join(c['source']) for c in b5('a'*40)['cells'])
    gpu='\n'.join(''.join(c['source']) for c in b3('a'*40)['cells'])
    assert 'max_workers=4' in cpu and '(0,1,2)' in cpu
    assert 'icarl_gir_vna' in gpu and 'icarl_gir_randdir' in gpu
    assert 'estimate["sessions"] >= 3' in gpu and 'schedule' in gpu
