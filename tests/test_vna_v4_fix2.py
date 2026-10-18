"""Tiny dense-reference checks for prompt 7.1–7.4, 7.8 and FIX2."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.optimize import minimize
from torch import nn
from diagnostics.vna_align import _assert_bitwise, _global_state
from experiments.vna_resume import rng_state, restore_rng
from methods.vna_losses import CE, MSE, ICaRLPyCIL, KDRepDistiller
from methods.vna_v3 import (V3Engine, training_products, seed_proposals, diagnose,
                            trust_region_step, net_gain, statistics)
from methods.vna_ggn import GGNProducts
from tests.test_vna_v3 import problem


@pytest.fixture(autouse=True)
def isolated():
    threads = torch.get_num_threads()
    with _global_state():
        torch.manual_seed(40)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def dense(p, op):
    J = torch.stack([p.Jv(v) for v in torch.eye(op.size, dtype=torch.float64)], 1)
    H = torch.block_diag(*op.loss_spec.metric_blocks(op.logits, op.ctx))
    return J, H


def test_7_1a_gn_matches_constrained_slsqp():
    model, site, batch, spec, ctx, _ = problem()
    with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
        op = GGNProducts(p, batch[1], spec, ctx)
        J, H = dense(p, op)
        F, g = (J.T@H@J).numpy(), op.gradient.numpy()
        x = torch.linalg.solve(torch.from_numpy(F)+.2*torch.eye(op.size, dtype=torch.float64), op.gradient).numpy()
        result = minimize(lambda v: -g@v+.5*v@F@v, np.zeros_like(g), jac=lambda v: F@v-g,
            constraints=[dict(type='ineq', fun=lambda v: x@x-v@v, jac=lambda v: -2*v)],
            method='SLSQP', options=dict(ftol=1e-14, maxiter=500))
        assert result.success
        np.testing.assert_allclose(result.x, x, atol=1e-6, rtol=0)


def test_7_1b_trust_region_bound():
    g, d = torch.randn(2, 9, dtype=torch.float64)
    if g@d < 0: d = -d
    for delta in (.001, 1., 100.):
        s, _, _ = trust_region_step(g, d, 2*d, .2, delta)
        assert float((s*d).norm()) <= delta+1e-12


def reject(engine, model, site, batch, spec, ctx, optimizer, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(engine, '_measure_ratio', lambda *a: ((1., None), (0., None)))
        patch.setattr(engine, '_evaluate_current', lambda *a: (2., None))
        row = engine.apply(model, [site], batch, spec, ctx, optimizer, 1, force=True)
    assert row['rejected'] and engine.nu == 2.
    assert engine.mu == 2*row['mu']


def test_7_1c_rejected_step_restores_parameters_bn(monkeypatch):
    model, site, batch, spec, ctx, opt = problem(True)
    engine = V3Engine('kernel_self'); engine.begin_task(1)
    before = deepcopy(model.state_dict()), deepcopy(opt.state_dict()), rng_state()
    reject(engine, model, site, batch, spec, ctx, opt, monkeypatch)
    _assert_bitwise(before, (model.state_dict(), opt.state_dict(), rng_state()))


def protected():
    model, site, (x, _), _, _, opt = problem(outputs=3)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    ctx = dict(targets=y, known_classes=2, teacher_logits=torch.randn(6, 2, dtype=torch.float64))
    return model, site, (x, y), ICaRLPyCIL(known_classes=2), ctx, opt


def test_7_2a_protected_residual_matches_dense():
    model, site, batch, spec, ctx, _ = protected()
    engine = V3Engine(); engine.tol = 1e-12
    with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
        op = GGNProducts(p, batch[1], spec, ctx); J, H = dense(p, op)
        values = engine._direction(p, op, .2)
        # Exercise the actual protected residual branch even when BetaRule selects zero.
        beta = 3.
        delta = torch.zeros_like(op.logits)
        delta[:, :2] = -p.Jv(values['xO']).reshape_as(delta)[:, :2]
        eb = op.g-beta*values['H_old'](delta.flatten())
        Hb = H+beta*torch.stack([values['H_old'](v) for v in torch.eye(len(op.g), dtype=torch.float64)], 1)
        x = values['solve']('gir_residual', p.JT(eb).double(), beta)
        actual = eb-Hb@p.Jv(x)
        expected = .2*torch.linalg.solve(Hb@J@J.T+.2*torch.eye(len(eb), dtype=torch.float64), eb)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=0)


def schur(J, H, e, Jv, mu):
    A = J.T@H@J+mu*torch.eye(J.shape[1], dtype=J.dtype)
    rp = e-H@J@torch.linalg.solve(A, J.T@e)
    Hp = H-H@J@torch.linalg.solve(A, J.T@H)
    s = Jv.T@rp
    return .5*s@torch.linalg.solve(Jv.T@Hp@Jv+mu*torch.eye(Jv.shape[1], dtype=J.dtype), s)


def test_7_2b_schur_gain_equals_dense_difference():
    model, site, batch, spec, ctx, _ = protected()
    with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
        op = GGNProducts(p, batch[1], spec, ctx); J, H = dense(p, op)
        Jv = torch.stack([p.JEv(v) for v in torch.eye(3, dtype=torch.float64)], 1)
        e, mu = op.g, .2
        q = lambda J: -.5*(J.T@e)@torch.linalg.solve(J.T@H@J+mu*torch.eye(J.shape[1], dtype=J.dtype), J.T@e)
        gain = schur(J, H, e, Jv, mu)
        assert gain >= 0
        torch.testing.assert_close(gain, q(J)-q(torch.cat((J, Jv), 1)), atol=1e-10, rtol=1e-8)


def test_7_2c_beta_zero_residual_bitwise(monkeypatch):
    model, site, batch, spec, ctx, opt = problem()
    engine = V3Engine('vna_gir'); engine.begin_task(1)
    seen = []
    original = engine._direction
    def direction(*a, **kw):
        result = original(*a, **kw)
        if not kw.get('split_only'): seen.append((a[1].g-result['z']).clone())
        return result
    monkeypatch.setattr(engine, '_direction', direction)
    def gir(*args):
        assert torch.equal(args[-2]['rperp_beta'], seen[-1])
        return []
    monkeypatch.setattr(engine, '_run_gir', gir)
    engine.apply(model, [site], batch, spec, ctx, opt, 1)
    assert seen


def test_7_3a_net_gain_matches_bfgs():
    model, site, batch, spec, ctx, _ = problem()
    with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
        op = GGNProducts(p, batch[1], spec, ctx); J, H = dense(p, op)
        Jp = J[:, :5]; fu = torch.randn(len(op.g), dtype=torch.float64)*.1
        e, mu = op.g, .2
        objective = lambda v: float(e@(-fu+Jp@torch.from_numpy(v))+.5*(-fu+Jp@torch.from_numpy(v))@H@(-fu+Jp@torch.from_numpy(v))+.5*mu*np.dot(v,v))
        result = minimize(objective, np.zeros(5), method='BFGS', options=dict(gtol=1e-9))
        q0 = -.5*(J.T@e)@torch.linalg.solve(J.T@H@J+mu*torch.eye(J.shape[1], dtype=J.dtype), J.T@e)
        assert net_gain(e,H,J,Jp,fu,mu) == pytest.approx(float(q0)-result.fun, abs=1e-6)


def test_7_3b_dead_slot_equals_schur():
    model, site, batch, spec, ctx, _ = problem()
    with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
        op = GGNProducts(p, batch[1], spec, ctx); J, H = dense(p, op)
        Jv = torch.randn(len(op.g), 2, dtype=torch.float64)
        assert net_gain(op.g,H,J,torch.cat((J,Jv),1),torch.zeros_like(op.g),.2) == pytest.approx(float(schur(J,H,op.g,Jv,.2)), abs=1e-10)


def gir_problem():
    model, site, batch, spec, ctx, opt = problem()
    with torch.no_grad():
        model[0].weight[:2].zero_(); model[0].bias[:2].fill_(-1)
        model[3].weight[:, :2].zero_()
    engine = V3Engine('vna_gir'); engine.begin_task(1); engine.mu=.2; engine.tol=1e-12
    return engine, model, site, batch, spec, ctx, opt


def run_gir(engine, model, site, batch, spec, ctx, opt, units=(0,1)):
    A = seed_proposals([site])
    with training_products(model,[site],A,batch[0]) as p:
        op = GGNProducts(p,batch[1],spec,ctx)
        J, H = dense(p, op)
        x = torch.linalg.solve(J.T@H@J+engine.mu*torch.eye(op.size, dtype=torch.float64), op.gradient)
        values = dict(e_beta=op.g,H_beta=op.H,rperp_beta=op.g-op.H(p.Jv(x)),gir_x=x)
    return engine._run_gir(model,[site],[site],A,batch,spec,ctx,opt,{site.module_name:list(units)},2,values,engine.mu)


def test_two_dead_slots_get_distinct_sx_orthogonal_rows(monkeypatch):
    engine, model, site, batch, spec, ctx, opt = gir_problem()
    original = engine.prepare_gir
    residuals = []
    def prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        residuals.append(kwargs['residual'].clone())
        S, _ = statistics(model, site, batch, [kwargs['residual']])
        for proposal in result:
            for used in kwargs['used_directions']:
                assert abs(float(used @ S @ proposal['direction'])) < 1e-10
        return result
    monkeypatch.setattr(engine, 'prepare_gir', prepare)
    before = deepcopy(model.state_dict())
    records = run_gir(engine, model, site, batch, spec, ctx, opt)
    after = deepcopy(model.state_dict())
    assert len(records) == 2
    assert len(residuals) == 2 and torch.equal(*residuals)
    rows = torch.cat((model[0].weight[:2],model[0].bias[:2,None]),1)
    assert not torch.equal(rows[0], rows[1])
    # Bias quantile shifts are separate from the S_x-orthogonal selected directions.
    assert engine.install_ages.keys() == {(site.module_name,0),(site.module_name,1)}
    assert not engine.warm
    # Check the real temporary-network CG gain against a dense dead-slot Schur gain.
    A = seed_proposals([site])
    model.load_state_dict(before)
    with training_products(model,[site],A,batch[0]) as p:
        op = GGNProducts(p,batch[1],spec,ctx)
        J, H = dense(p,op)
        e = op.g
    with torch.no_grad():
        model[0].weight[0].copy_(after['0.weight'][0])
        model[0].bias[0].copy_(after['0.bias'][0])
    with training_products(model,[site],A,batch[0]) as p:
        Jp, _ = dense(p,GGNProducts(p,batch[1],spec,ctx))
    assert records[0]['Gamma'] == pytest.approx(net_gain(e,H,J,Jp,torch.zeros_like(e),engine.mu),abs=1e-9)


def test_7_3c_rejected_swap_restores_all_state(monkeypatch):
    engine, model, site, batch, spec, ctx, opt = gir_problem()
    original = engine.prepare_gir
    def proposals(*a, **kw):
        result = original(*a, **kw)
        return result[:1]
    monkeypatch.setattr(engine,'prepare_gir',proposals)
    import methods.vna_v3 as module
    original_cg = module.conjugate_gradient
    def cg(*a, **kw):
        result = original_cg(*a, **kw)
        # Zero recovery predicts a loss relative to the current fitted model.
        if any(engine.install_ages): result.solution.zero_()
        return result
    monkeypatch.setattr(module,'conjugate_gradient',cg)
    before = deepcopy(model.state_dict()),deepcopy(opt.state_dict()),rng_state(),deepcopy(engine.warm),deepcopy(engine.install_ages)
    assert not run_gir(engine,model,site,batch,spec,ctx,opt)
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state(),engine.warm,engine.install_ages))


def test_7_4_t25_bound_and_dense_reach():
    model, site, batch, spec, ctx, opt = problem()
    engine = V3Engine('vna'); engine.begin_task(1); engine.tol=1e-12
    row = engine.apply(model,[site],batch,spec,ctx,opt,1,apply=False)
    assert row['V'] <= row['lambda_E_B']+1e-10
    blocks = spec.metric_blocks(model(batch[0]),ctx)
    e = spec.residual(model(batch[0]),ctx).flatten()
    expected = e@torch.linalg.pinv(torch.block_diag(*blocks),hermitian=True)@e
    assert row['norm_w2'] == pytest.approx(float(expected),abs=1e-10)


@pytest.mark.parametrize('spec',[CE(),MSE(reduction='mean'),KDRepDistiller(),ICaRLPyCIL(known_classes=2)])
def test_loss_blocks_equal_dense_metric(spec):
    z = torch.randn(3,4,dtype=torch.float64)
    ctx = dict(targets=torch.randn_like(z) if isinstance(spec, MSE) else torch.tensor([0,1,2]),
               teacher_logits=torch.randn_like(z),known_classes=2)
    H = torch.stack([spec.metric_apply(z,v,ctx).flatten() for v in torch.eye(z.numel(),dtype=z.dtype)],1)
    blocks = torch.block_diag(*spec.metric_blocks(z,ctx))
    torch.testing.assert_close(blocks,H,atol=1e-14,rtol=1e-14)
    e = spec.residual(z,ctx).flatten()
    assert float(e@torch.linalg.pinv(H,hermitian=True)@e) == pytest.approx(float(e@torch.linalg.pinv(blocks,hermitian=True)@e),abs=1e-10)


def test_7_8_diagnose_read_only_parameters_buffers_optimizer_rng():
    model,site,batch,spec,ctx,opt = problem(True)
    model(batch[0]).sum().backward();opt.step()
    model.eval()
    before = deepcopy(model.state_dict()),deepcopy(opt.state_dict()),rng_state(),[p.grad.clone() for p in model.parameters()]
    result = diagnose(model,batch,sites=[site],spec=spec,ctx=ctx)
    _assert_bitwise(before,(model.state_dict(),opt.state_dict(),rng_state(),[p.grad for p in model.parameters()]))
    assert not model.training and 0 <= result['reach'] <= 1
    assert set(result['sites'][site.module_name]) == {'dead','dormant'}


def test_mid_task_resume_after_rejection_and_swap_bitwise(monkeypatch):
    engine,model,site,batch,spec,ctx,opt = gir_problem()
    engine.arm='kernel_self'
    reject(engine,model,site,batch,spec,ctx,opt,monkeypatch)
    engine.arm='vna_gir'
    assert run_gir(engine,model,site,batch,spec,ctx,opt)
    checkpoint = deepcopy((model.state_dict(),opt.state_dict(),engine.state_dict(),rng_state()))
    def advance():
        opt.zero_grad();spec.loss(model(batch[0]),ctx).backward();opt.step()
        with training_products(model,[site],seed_proposals([site]),batch[0]) as p:
            engine._direction(p,GGNProducts(p,batch[1],spec,ctx),engine.mu)
        torch.rand(4)
        return deepcopy((model.state_dict(),opt.state_dict(),engine.mu,engine.nu,engine.install_ages,engine.warm,rng_state()))
    expected = advance()
    model,site,_,_,_,opt = problem()
    engine = V3Engine()
    model.load_state_dict(checkpoint[0]);opt.load_state_dict(checkpoint[1])
    engine.load_state_dict(checkpoint[2]);restore_rng(checkpoint[3])
    _assert_bitwise(expected,advance())


def test_slot_jump_computed_once_per_live_unit(monkeypatch):
    engine,model,site,batch,spec,ctx,opt = gir_problem()
    calls = []
    def jump(*args):
        calls.append(args[-1])
        return 1., None
    monkeypatch.setattr(engine, '_slot_jump', jump)
    assert not run_gir(engine,model,site,batch,spec,ctx,opt,units=(2,))
    assert calls == [2]


@pytest.mark.parametrize('benchmark', ['regression', 'mnist'])
@pytest.mark.parametrize('arm', ['baseline','cbp','vna_gir','cbp_gradmax'])
def test_task_end_probe_every_arm_fixed_seeded_read_only(arm, benchmark, tmp_path):
    from adapters.vna_v3_hooks import LoPV3
    from diagnostics.vna_align import MeasurementBuffer
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 2))
        def predict(self, x):
            features = self.layers[:2](x)
            return self.layers[2](features), [features]
    net = Net().double()
    loss = nn.MSELoss() if benchmark == 'regression' else nn.CrossEntropyLoss()
    learner = SimpleNamespace(net=net, opt=torch.optim.SGD(net.parameters(),lr=.01), loss_func=loss)
    engine = V3Engine(arm, coefficient=0.)
    adapter = LoPV3(engine, benchmark=benchmark)
    hook = SimpleNamespace(step=0,task_steps=2,buffer=MeasurementBuffer(128),status=None,
        results=SimpleNamespace(losses=torch.ones(4),accuracies=None))
    x = torch.randn(4,3,dtype=torch.float64)
    y = torch.randn(4,2,dtype=torch.float64) if benchmark == 'regression' else torch.arange(4)%2
    for step in range(4):
        adapter.before(hook,learner,x+step,y)
        probe = deepcopy(adapter.probe_batch)
        hook.step += 1
        before = deepcopy(net.state_dict()),deepcopy(learner.opt.state_dict()),rng_state()
        adapter.after(hook,learner)
        _assert_bitwise(before,(net.state_dict(),learner.opt.state_dict(),rng_state()))
        _assert_bitwise(probe,adapter.probe_batch)
    assert [r['task'] for r in engine.activities] == [1,2]
    assert len(adapter.probe_batch[0]) == 8
    assert all({'B','reach','unreach','sites'} <= r.keys() for r in engine.activities)
    restored = LoPV3(V3Engine())
    restored.load_state_dict(adapter.state_dict())
    _assert_bitwise(adapter.probe_batch,restored.probe_batch)
    from experiments.pilot_v3 import write_events
    import json
    write_events(tmp_path, engine)
    assert json.loads((tmp_path/'diagnostics.json').read_text()) == engine.activities


def test_7_9_bad_warm_start_resets_to_zero_bitwise():
    from projection.cg import conjugate_gradient
    matrix = torch.diag(torch.tensor([1.,2.,4.],dtype=torch.float64))
    rhs = torch.tensor([1.,-2.,3.],dtype=torch.float64)
    zero = conjugate_gradient(lambda x: matrix@x,rhs,max_iter=200,tolerance=1e-3)
    warm = conjugate_gradient(lambda x: matrix@x,rhs,x0=100*rhs,max_iter=200,tolerance=1e-3)
    assert warm.warm_reset and warm.converged
    assert torch.equal(warm.solution,zero.solution)
    assert warm.residual_history == zero.residual_history


def test_authority_measurement_preserves_training_state():
    model, site, batch, spec, ctx, optimizer = problem(True)
    engine = V3Engine('vna_gir'); engine.begin_task(1)
    for p in model.parameters(): p.grad = torch.ones_like(p)
    before = deepcopy(model.state_dict()),deepcopy(optimizer.state_dict()),rng_state(),[p.grad.clone() for p in model.parameters()]
    state = engine.state_dict()
    report = engine.preflight(model,[site],batch,spec,ctx,optimizer)
    _assert_bitwise(before,(model.state_dict(),optimizer.state_dict(),rng_state(),[p.grad for p in model.parameters()]))
    after = engine.state_dict()
    after['protocol'].pop('preflight_points')
    _assert_bitwise(state,after)
    assert report['trials'][0]['vna_seconds'] > 0
