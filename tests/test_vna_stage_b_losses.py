"""Upstream formulas and true output-Hessian finite differences (float64)."""
import importlib.util
import os
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from methods.vna_losses import CE, MSE, KDRepDistiller, ICaRLPyCIL
from methods.vna import dedicated_generator, block_products
from methods.vna_sites import HookSite
from methods.vna_ggn import GGNProducts, GGNConfig, vna_exact_ggn
from methods.vna_joint import joint_correction


def upstream_formula(spec, z, ctx):
    if isinstance(spec, KDRepDistiller):
        # RepDistiller/distiller_zoo/KD.py DistillKL.forward, pinned b84f547.
        T = spec.T
        y_s, y_t = z, ctx['teacher_logits']
        p_s = F.log_softmax(y_s/T, dim=1)
        p_t = F.softmax(y_t/T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (T**2) / y_s.shape[0]
        return spec.alpha * F.cross_entropy(z, ctx['targets']) + spec.beta * loss
    if isinstance(spec, ICaRLPyCIL):
        loss = F.cross_entropy(z, ctx['targets'])
        if spec.known_classes:
            # PyCIL/models/icarl.py _KD_loss, pinned f3509b8.
            pred, soft = z[:, :spec.known_classes], ctx['teacher_logits'][:, :spec.known_classes]
            T = spec.T
            pred = torch.log_softmax(pred / T, dim=1)
            soft = torch.softmax(soft / T, dim=1)
            loss = loss + -1 * torch.mul(soft, pred).sum() / pred.shape[0]
        return loss
    if isinstance(spec, MSE):
        return ((z-ctx['targets']).square().sum()/len(z) if spec.reduction == 'batchmean'
                else F.mse_loss(z, ctx['targets']))
    return F.cross_entropy(z, ctx['targets'])


@pytest.mark.parametrize('spec', [CE(output_dim=5), MSE(output_dim=5), MSE(reduction='mean'),
                                 KDRepDistiller(output_dim=5), ICaRLPyCIL(known_classes=3),
                                 ICaRLPyCIL(known_classes=0)])
def test_loss_autograd_analytic_and_fd(spec):
    gen = dedicated_generator(12, spec.name)
    z = torch.randn(4, 5, generator=gen, dtype=torch.float64)
    v = torch.randn(4, 5, generator=gen, dtype=torch.float64)
    ctx = {'targets': torch.tensor([0, 1, 3, 4]),
           'teacher_logits': torch.randn(4, 5, generator=gen, dtype=torch.float64)}
    if isinstance(spec, MSE):
        ctx['targets'] = torch.randn(4, 5, generator=gen, dtype=torch.float64)
    called = []
    def loss_fn(output, context):
        called.append(True)
        return upstream_formula(spec, output, context)
    ctx['loss_fn'] = loss_fn
    g = spec.residual(z, ctx)
    assert called
    if isinstance(spec, MSE):
        analytic = 2 * (z-ctx['targets']) / (len(z) if spec.reduction == 'batchmean' else z.numel())
    else:
        analytic = (z.softmax(-1) - F.one_hot(ctx['targets'], 5)) / len(z)
        if isinstance(spec, KDRepDistiller):
            analytic = spec.alpha * analytic + spec.beta * spec.T * ((z/spec.T).softmax(-1)-(ctx['teacher_logits']/spec.T).softmax(-1))/len(z)
        if isinstance(spec, ICaRLPyCIL) and spec.known_classes:
            k = spec.known_classes
            analytic[:, :k] += ((z[:, :k]/spec.T).softmax(-1)-(ctx['teacher_logits'][:, :k]/spec.T).softmax(-1))/(spec.T*len(z))
    torch.testing.assert_close(g, analytic, rtol=1e-12, atol=1e-12)
    eps = 1e-5
    finite_difference = (spec.residual(z+eps*v, ctx)-spec.residual(z-eps*v, ctx))/(2*eps)
    torch.testing.assert_close(spec.metric_apply(z, v, ctx), finite_difference, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(spec.loss(z, ctx), upstream_formula(spec, z, ctx))


def test_mse_upstream_mean_reduction_and_generic_ggn():
    torch.set_num_threads(1)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).double()
    site = HookSite(model, '0', '2', 'linear', between=('1',))
    gen = dedicated_generator(6, 'regression')
    batch = (torch.randn(3, 3, generator=gen, dtype=torch.float64), torch.randn(3, 2, generator=gen, dtype=torch.float64))
    ctx = {'loss_fn': lambda z, c: F.mse_loss(z, c['targets'])}
    spec = MSE()
    A = model[0].weight[:2].detach()
    with block_products(model, site, A, batch[0]) as p:
        op = GGNProducts(p, batch[1], spec, ctx)
        assert op.g.numel() == 6
        torch.testing.assert_close(op.H(torch.ones(6)), torch.full((6,), 1/3, dtype=torch.float64))
        eye = torch.eye(op.size, dtype=torch.float64)
        J = torch.stack([p.Jv(e) for e in eye], 1)
        vector = torch.randn(op.size, generator=gen, dtype=torch.float64)
        torch.testing.assert_close(op.F(vector), J.T @ J @ vector / 3)
    cfg = GGNConfig(cg_iters=100, cg_tol=1e-10, damping_norm='outputs')
    d, diag = vna_exact_ggn(model, site, A, batch, cfg, loss_spec=spec, ctx=ctx, mu=.1)
    joint, V, rho, record = joint_correction(model, [site], {site: A}, batch, cfg, o_params=site.parameter_names,
                                            loss_spec=spec, ctx=ctx, mu=.1)
    for name in d:
        torch.testing.assert_close(d[name], joint[name], rtol=1e-8, atol=1e-10)
    assert diag['V'] == pytest.approx(V, rel=1e-8)


@pytest.mark.parametrize('repo', ['RepDistiller', 'PyCIL'])
def test_imported_upstream_loss_equality(repo):
    root = Path(__file__).resolve().parents[1] / 'third_party' / repo
    if not (root / ".git").is_dir():
        message = f'{repo} pinned clone absent; run scripts/fetch_third_party.sh'
        if os.environ.get('REQUIRE_UPSTREAM') == '1':
            pytest.fail(message)
        pytest.skip(message)
    gen = dedicated_generator(8, repo)
    z = torch.randn(4, 5, generator=gen, dtype=torch.float64)
    teacher = torch.randn(4, 5, generator=gen, dtype=torch.float64)
    if repo == 'RepDistiller':
        module_spec = importlib.util.spec_from_file_location('upstream_kd', root/'distiller_zoo/KD.py')
        module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
        loss = module.DistillKL(4)(z, teacher)
        spec = KDRepDistiller(alpha=0, beta=1)
        ctx = {'targets': torch.zeros(4, dtype=torch.long), 'teacher_logits': teacher}
        torch.testing.assert_close(spec.loss(z, ctx), loss, rtol=0, atol=0)
        ctx['loss_fn'] = lambda output, c: module.DistillKL(4)(output, c['teacher_logits'])
    else:
        # Import the real entry point with its native package dependencies.
        import sys
        import subprocess
        code = "from models.icarl import _KD_loss; import torch; print(_KD_loss(torch.zeros(2,3),torch.ones(2,3),2).item())"
        subprocess.run([sys.executable, '-c', code], cwd=root, check=True, capture_output=True)
        # Isolate PyCIL's generic 'models'/'utils' names from our process.
        code = """import torch
from models.icarl import _KD_loss
z, t = torch.load(__import__('sys').argv[1], weights_only=True)
torch.save(_KD_loss(z,t,2), __import__('sys').argv[2])
"""
        import tempfile
        with tempfile.TemporaryDirectory(prefix='vna-upstream-') as directory:
            inp, out = Path(directory)/'inputs.pt', Path(directory)/'output.pt'
            torch.save((z[:, :3], teacher[:, :3]), inp)
            subprocess.run([sys.executable, '-c', code, str(inp), str(out)], cwd=root, check=True, capture_output=True)
            loss = torch.load(out, weights_only=True)
        spec = ICaRLPyCIL(known_classes=3)
        ctx = {'targets': torch.zeros(4, dtype=torch.long), 'teacher_logits': teacher}
        torch.testing.assert_close(spec.loss(z, ctx)-F.cross_entropy(z, ctx['targets']), loss)


def test_default_and_explicit_ce_keep_legacy_bits():
    from methods.vna import _mean_ce_logit_grad
    gen = dedicated_generator(40, 'ce-bits')
    with torch.random.fork_rng():
        torch.manual_seed(4)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 3)).double()
    site = HookSite(model, '0', '2', 'linear', between=('1',))
    x = torch.randn(4, 3, generator=gen, dtype=torch.float64)
    y = torch.tensor([0, 1, 2, 0])
    with block_products(model, site, model[0].weight[:2].detach(), x) as p:
        logits = p._call({}).reshape(4, 3)
        probs = logits.softmax(-1)
        vector = torch.randn(12, generator=gen, dtype=torch.float64).reshape(4, 3)
        old_H = (probs * (vector-(probs*vector).sum(-1, keepdim=True))/4).flatten()
        for spec in (None, CE()):
            op = GGNProducts(p, y, spec)
            assert torch.equal(op.g, _mean_ce_logit_grad(logits, y))
            assert torch.equal(op.H(vector), old_H)
            assert torch.equal(op.gradient, p.JT(op.g))


def test_generic_loss_flows_through_selection_and_diagnostics():
    from methods.vna_select import tiny_V
    from diagnostics.stage_a import residual
    from diagnostics.signal_curve import heldout
    from methods.vna import VNAConfig, vna_lite_correction
    gen = dedicated_generator(7, 'loss-plumbing')
    with torch.random.fork_rng():
        torch.manual_seed(4)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).double()
    site = HookSite(model, '0', '2', 'linear', between=('1',))
    batch = (torch.randn(4, 3, generator=gen, dtype=torch.float64), torch.randn(4, 2, generator=gen, dtype=torch.float64))
    called = []
    def upstream(z, ctx):
        called.append(True)
        return F.mse_loss(z, ctx['targets'])
    spec, ctx = MSE(), {'loss_fn': upstream}
    g = residual(model, batch, spec, ctx)
    with torch.no_grad(): expected = (model(batch[0])-batch[1])/4
    torch.testing.assert_close(g, expected)
    A, record = tiny_V(model, site, [batch], 2, loss_spec=spec, contexts=[ctx],
                       cfg=GGNConfig(cg_iters=20, hutchinson_probes=1), score_signs=False)
    d, diag = vna_lite_correction(model, site, A, batch, VNAConfig(hutchinson_probes=1), loss_spec=spec, ctx=ctx)
    measured = heldout(model, site, A, batch, {'vna': d}, VNAConfig(), loss_spec=spec, ctx=ctx)
    assert measured['loss_name'] == 'MSE' and measured['heldout_loss'] > 0 and called
