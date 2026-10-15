"""Tiny float64 checks for the output-space Cholesky path."""
import pytest
import torch
from torch import nn
from methods.vna_output import OutputFactor
from methods.vna_ggn import GGNProducts, GGNConfig, solve_ggn
from methods.vna_joint import joint_products
from methods.vna_losses import CE, KDRepDistiller, MSE
from methods.vna_sites import HookSite


def relative(actual, expected):
    assert float((actual-expected).norm()/expected.norm().clamp_min(1e-30)) <= 1e-8


@pytest.mark.parametrize('spec', [CE(), KDRepDistiller(), MSE()])
def test_cholesky_eigh_cg_and_principal_group(spec):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(48)
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 3)).double().eval()
        site = HookSite(model, '0', '2', 'linear', between=('1',))
        x = torch.randn(6, 3, dtype=torch.float64)
        y = torch.randn(6, 3, dtype=torch.float64) if isinstance(spec, MSE) else torch.arange(6)%3
        teacher = torch.randn(6, 3, dtype=torch.float64)
        ctx = dict(targets=y, teacher_logits=teacher)
        a = {'weight': model[0].weight[:2].detach().clone(), 'bias': model[0].bias[:2].detach().clone()}
        cfg = GGNConfig(cg_iters=2000, cg_tol=1e-12, project_bn=False)
        with joint_products(model, [site], {site: a}, x) as p:
            op = GGNProducts(p, y, spec, ctx)
            chol = OutputFactor(op, factor_mode='cholesky')
            eigen = OutputFactor(op, factor_mode='eigh')
            w = p.JET(op.g)
            for mu in (.001, .01):
                u = solve_ggn(op, p.JT(op.g), mu, cfg)
                assert u.converged
                z = op.H(p.Jv(u.solution))
                d = solve_ggn(op, p.JT(op.H(p.JEv(w))), mu, cfg)
                assert d.converged
                for factor in (chol, eigen):
                    relative(factor.z(op.g, mu), z)
                    relative(factor.d(p, w, mu), d.solution)
        mask = torch.tensor([True, False, True, False, True, False])
        indices = mask[:, None].expand(-1, 3).flatten().nonzero().flatten()
        matrix = chol.matrix[indices[:, None], indices[None, :]] * 2
        subctx = dict(targets=y[mask], teacher_logits=teacher[mask])
        with joint_products(model, [site], {site: a}, x[mask]) as p:
            op = GGNProducts(p, y[mask], spec, subctx)
            group = OutputFactor(op, matrix=matrix, factor_mode='cholesky')
            fresh = OutputFactor(op, factor_mode='cholesky')
            torch.testing.assert_close(group.matrix, fresh.matrix, rtol=1e-10, atol=1e-14)
            w = p.JET(op.g)
            for mu in (.001, .01):
                relative(group.z(op.g, mu), fresh.z(op.g, mu))
                relative(group.d(p, w, mu), fresh.d(p, w, mu))
    finally:
        torch.set_num_threads(old)
