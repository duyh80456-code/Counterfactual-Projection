"""Small float64 checks of the actual v3 training implementation."""
from copy import deepcopy
import pytest
import torch
from torch import nn
from methods.vna_losses import MSE, ICaRLPyCIL
from methods.vna_ggn import GGNProducts, GGNConfig
from methods.vna_sites import HookSite
from methods.vna_joint import joint_correction
from methods.vna_v3 import (V3Engine, training_products, seed_proposals, beta_rule,
    correction_step, generalized_rows, schedule_from_points, statistics)


@pytest.fixture(autouse=True)
def isolated():
    previous = torch.get_num_threads()
    with torch.random.fork_rng():
        torch.manual_seed(40); torch.set_num_threads(1)
        yield
    torch.set_num_threads(previous)


def problem(bn=False, outputs=1):
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4) if bn else nn.Identity(),
                          nn.ReLU(), nn.Linear(4, outputs)).double().train()
    site = HookSite(model, '0', '3', 'linear', between=('1', '2'))
    batch = torch.randn(6, 3, dtype=torch.float64), torch.randn(6, outputs, dtype=torch.float64)
    spec, ctx = MSE(reduction='mean'), dict(targets=batch[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    return model, site, batch, spec, ctx, optimizer


def flat(delta): return torch.cat([v.flatten() for v in delta.values()])


def test_joint_same_full_scope_is_sum_of_separate_sites():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 1)).double().eval()
    sites = [HookSite(model, '0', '2', 'linear', between=('1',)), HookSite(model, '2', '4', 'linear', between=('3',))]
    A = {s: dict(weight=torch.randn(1, s.first_layer.layer.in_features, dtype=torch.float64), bias=torch.zeros(1, dtype=torch.float64)) for s in sites}
    batch = torch.randn(4, 2, dtype=torch.float64), torch.randn(4, 1, dtype=torch.float64)
    cfg = GGNConfig(project_bn=False, cg_tol=1e-12, cg_iters=500)
    joint = joint_correction(model, sites, A, batch, cfg, mu=.2, loss_spec=MSE())[0]
    separate = [joint_correction(model, [s], {s: A[s]}, batch, cfg, mu=.2, loss_spec=MSE())[0] for s in sites]
    expected = {name: sum(d[name] for d in separate) for name in joint}
    assert float((flat(joint)-flat(expected)).norm()/flat(expected).norm()) < 1e-10


@pytest.mark.parametrize('bn', [False, True])
def test_kernel_self_cg_and_warm_and_read_only_bn(bn):
    model, site, batch, spec, ctx, optimizer = problem(bn)
    before = deepcopy(model.state_dict())
    engine = V3Engine('kernel_self'); engine.begin_task(1); engine.tol = 1e-12
    with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
        op = GGNProducts(p, batch[1], spec, ctx)
        mu = .1
        actual = engine._direction(p, op, mu)
        eye = torch.eye(op.size, dtype=torch.float64)
        F = torch.stack([op.F(v) for v in eye], 1)
        G = torch.linalg.inv(F+mu*eye)
        expected = F@G@F@G@op.gradient
        torch.testing.assert_close(actual['d'], expected, rtol=1e-8, atol=1e-10)
        again = engine._direction(p, op, mu)
        assert again['records'][0]['iterations'] == 0
        true_residual = float((op.gradient-(F+mu*eye)@again['x']).norm()/op.gradient.norm())
        assert true_residual == pytest.approx(again['records'][0]['relative_residual'], abs=1e-10)
        engine.tol = 1e-3; engine.warm.clear()
        approximate = engine._direction(p, op, mu)
        assert torch.nn.functional.cosine_similarity(expected, approximate['d'], dim=0) >= .99
    for name, value in before.items(): assert torch.equal(value, model.state_dict()[name])


def test_kernel_self_protects_old_classes_but_keeps_unprotected_beta_zero_case():
    model, site, (x, _), _, _, optimizer = problem(outputs=4)
    with torch.no_grad(): teacher = model(x)[:, :2].clone(); teacher[:, 0] += 1
    engine = V3Engine('kernel_self'); engine.begin_task(2)
    with training_products(model, [site], seed_proposals([site]), x) as p:
        op = GGNProducts(p, torch.tensor([0, 1, 2, 3, 0, 2]),
            ICaRLPyCIL(T=2, known_classes=2),
            dict(targets=torch.tensor([0, 1, 2, 3, 0, 2]), known_classes=2,
                 teacher_logits=teacher))
        values = engine._direction(p, op, .1)
        assert values['beta_branch'] != 'unprotected'

    model, site, batch, _, _, _ = problem(outputs=1)
    engine = V3Engine('kernel_self'); engine.begin_task(1); engine.tol = 1e-12
    with training_products(model, [site], seed_proposals([site]), batch[0]) as p:
        op = GGNProducts(p, batch[1], MSE(), dict(targets=batch[1]))
        values = engine._direction(p, op, .1)
        eye = torch.eye(op.size, dtype=torch.float64)
        F = torch.stack([op.F(v) for v in eye], 1)
        expected = F @ torch.linalg.inv(F+.1*eye) @ F @ torch.linalg.inv(F+.1*eye) @ op.gradient
        torch.testing.assert_close(values['d'], expected, rtol=1e-8, atol=1e-10)


def test_beta_all_branches_verified_upper_and_monotone_leak():
    F = torch.tensor([[2., .2], [.2, 1.]], dtype=torch.float64)
    B = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
    rhs = torch.tensor([2., 1.], dtype=torch.float64)
    solve = lambda name, b, beta: torch.linalg.solve(F+beta*B, b)
    leak = lambda d: float(d@B@d)
    args = (solve, rhs, lambda v: B@v, lambda v: v, rhs)
    _, a = beta_rule(*args, 1., 0., torch.ones(2)); assert a['beta_branch'] == 'a'
    _, b = beta_rule(*args, 0., 1., torch.ones(2)); assert b['beta_branch'] == 'b'
    d, c = beta_rule(*args, .01, 1., torch.ones(2))
    assert c['beta_branch'] == 'c_bisect' and leak(d) <= c['epsilon_star']
    leaks = [leak(solve('', rhs, beta)) for beta in (0., .01, 1., 10., 100.)]
    assert all(a >= b for a, b in zip(leaks, leaks[1:]))
    _, z = beta_rule(*args, 100., 1., torch.ones(2)); assert z['beta'] == 0.


def test_step_and_faithful_and_nonpositive_gain():
    g = torch.tensor([2., 3.], dtype=torch.float64); d = g*2
    s, optimum, _ = correction_step(g, d, d, .1, .01, 50)
    assert 0 < s <= optimum and s <= .01*float(g.norm()/d.norm())
    faithful, optimum, _ = correction_step(g, d, d, .1, .001, 10, .9, True)
    assert faithful == min(optimum, 10*.001/(1-.9))
    assert correction_step(g, -d, -d, .1, .01, 50)[0] == 0.


def test_zero_lambda_is_bitwise_and_readonly_preflight_state():
    model, site, batch, spec, ctx, optimizer = problem(True)
    engine = V3Engine('vna', coefficient=0.); engine.begin_task(1)
    before, rng = deepcopy(model.state_dict()), torch.random.get_rng_state().clone()
    assert engine.apply(model, [site], batch, spec, ctx, optimizer, 1) is None
    assert not engine.events and torch.equal(rng, torch.random.get_rng_state())
    for n, v in before.items(): assert torch.equal(v, model.state_dict()[n])
    engine.coefficient = 1.
    state = engine.state_dict()
    row = engine.apply(model, [site], batch, spec, ctx, optimizer, 1, apply=False)
    assert row and not engine.warm and engine.last_iteration is None and not engine.events
    assert engine.A == state['A']
    for n, v in before.items(): assert torch.equal(v, model.state_dict()[n])


def test_exact_check_is_apply_only_and_timed_separately(monkeypatch):
    model, site, batch, spec, ctx, optimizer = problem()
    engine = V3Engine('vna'); engine.begin_task(1)
    calls = []
    original = engine._direction
    def direction(*args, **kwargs):
        calls.append(kwargs.get('exact', False))
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, '_direction', direction)
    row = engine.apply(model, [site], batch, spec, ctx, optimizer, 1, apply=False)
    assert True not in calls
    assert row['exact_check_seconds'] == 0.
    row = engine.apply(model, [site], batch, spec, ctx, optimizer, 1)
    assert True in calls
    assert row['exact_check_seconds'] > 0.


def test_kernel_self_uses_all_valid_sites_without_selection(monkeypatch):
    model, site, batch, spec, ctx, optimizer = problem()
    engine = V3Engine('kernel_self'); engine.begin_task(1)
    monkeypatch.setattr(engine, '_select', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('selection')))
    row = engine.apply(model, [site], batch, spec, ctx, optimizer, 1)
    assert row['skip_reason'] != 'no_positive_directions'


def test_adaptive_schedule_iterations_skips_and_budget_exceeded():
    points = [dict(task=1, classes=10, vna_seconds=100., iteration_seconds=.01),
              dict(task=2, classes=20, vna_seconds=200., iteration_seconds=.01)]
    schedule = schedule_from_points(points, 55)
    assert schedule['K_base'] == 800 and schedule['budget_exceeded']
    assert schedule['projected_vna_seconds'] == 1000.
    model, site, batch, spec, ctx, optimizer = problem()
    engine = V3Engine('vna', K_base=10); engine.begin_task(1)
    row = engine.apply(model, [site], batch, spec, ctx, optimizer, 1, remaining=100)
    assert row is not None
    assert row['K_next'] == max(1, int(10*min(4., max(.25, .3/row['rho']))+.5))
    assert engine.apply(model, [site], batch, spec, ctx, optimizer, 2) is None
    assert engine.apply(model, [site], batch, spec, ctx, optimizer, 1+row['K_next']) is not None
    engine.begin_task(2)
    assert engine.apply(model, [site], batch, spec, ctx, optimizer, 1234) is not None


def test_trace_matched_random_per_site_with_bn():
    model, site, batch, spec, ctx, optimizer = problem(True)
    engine = V3Engine('vna_rand'); engine.begin_task(1)
    A = seed_proposals([site])
    randomized = engine._randomize(model, [site], batch, A)
    with training_products(model, [site], A, batch[0]) as p: target = engine._trace_extension(p)
    with training_products(model, [site], randomized, batch[0]) as p: actual = engine._trace_extension(p)
    assert actual == pytest.approx(target, rel=1e-8, abs=1e-12)


def test_gir_dead_slot_preserves_output_and_zeros_optimizer_state():
    model, site, batch, spec, ctx, optimizer = problem()
    for p in model.parameters(): optimizer.state[p]['momentum_buffer'] = torch.ones_like(p)
    with torch.no_grad(): model[0].bias[0] = -100.
    before = model(batch[0]).detach().clone()
    engine = V3Engine('gir'); engine.begin_task(1)
    proposals = engine.prepare_gir(model, site, batch, spec, ctx, optimizer, [0], 1)
    assert proposals
    assert engine.install_gir(model, site, proposals, [0], optimizer) == 1
    torch.testing.assert_close(model(batch[0]), before, rtol=0, atol=1e-12)
    for p, index in ((model[0].weight, 0), (model[0].bias, 0), (model[3].weight, (slice(None), 0))):
        assert torch.count_nonzero(optimizer.state[p]['momentum_buffer'][index]) == 0
    assert model[3].weight[:, 0].eq(0).all()


def test_gradmax_is_euclidean_and_differs_from_generalized():
    S = torch.diag(torch.tensor([100., 1.], dtype=torch.float64))
    N = torch.tensor([[2., 0.], [0., 1.]], dtype=torch.float64)
    euclid = generalized_rows(N@N.T, S, 1, euclidean=True)[0]
    generalized = generalized_rows(N@N.T, S, 1)[0]
    assert abs(float(euclid[0])) > .99 and abs(float(generalized[0])) < 1e-10
