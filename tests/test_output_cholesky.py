"""Cholesky (n_out > OUTPUT_EXACT_MAX) path of the output-space exact solve.

The threshold is lowered so the small problem takes the large-n path:
CPU float64 Cholesky per damping value, block symmetrization, disk parking
of inactive factors, and principal submatrices for groups. Every quantity is
compared with matrix-free CG at tolerance 1e-12.
"""
import types
import pytest
import torch
import methods.vna_output as vo
from diagnostics.measure_efficiency import SolveCache
from diagnostics.stage_a import null_residual
from methods.vna_ggn import GGNConfig
from methods.vna_joint import joint_correction
from methods.vna_losses import CE, KDRepDistiller, MSE
from methods.vna_select import tiny_V
from tests.test_measure_round2 import problem

CFG = GGNConfig(cg_iters=2000, cg_tol=1e-12, project_bn=False)


@pytest.fixture(autouse=True)
def large_path(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(48)
    monkeypatch.setattr(vo, 'OUTPUT_EXACT_MAX', 4)
    yield
    torch.set_num_threads(old)


def relative(a, b):
    return float((a-b).norm()/b.norm().clamp_min(1e-300))


@pytest.mark.parametrize('spec', [CE(), KDRepDistiller(), MSE()])
@pytest.mark.parametrize('park', [False, True])
def test_cholesky_matches_cg(spec, park, monkeypatch):
    if park:  # force every inactive factor to be parked on disk and restored
        import psutil
        monkeypatch.setattr(psutil, 'virtual_memory', lambda: types.SimpleNamespace(available=0))
    model, site, data = problem()
    if isinstance(spec, MSE): data = data[0], torch.randn(6, 3, dtype=torch.float64)
    context = {'targets': data[1], 'teacher_logits': torch.randn(6, 3, dtype=torch.float64)}
    other = (data[0].flip(0), data[1].flip(0)); other_ctx = {**context, 'targets': other[1]}
    factors = {}
    for mu in (1e-3, 1e-2):
        exact = SolveCache(model, site, data, spec, context, CFG, mu=mu, solver='output_exact', factors=factors)
        cg = SolveCache(model, site, data, spec, context, CFG, mu=mu, solver='cg')
        e = spec.residual(model(data[0]), context)
        e_other = spec.residual(model(other[0]), other_ctx)
        residuals = [e] + [null_residual(e, kind, torch.Generator().manual_seed(i),
                                         residual_is_zero_sum=spec.residual_is_zero_sum)
                           for kind in ('N1', 'N2') for i in range(2)]
        for index, residual in enumerate(residuals):
            z = exact.solve(str(index), residual, data, context)
            # Alternate datasets so parking/restoring happens between solves.
            z_other = exact.solve(f'other{index}', e_other, other, other_ctx)
            assert exact.factor(data, context).factor_mode == 'cholesky'
            assert relative(z, cg.solve(str(index), residual, data, context)) <= 1e-8
            assert relative(z_other, cg.solve(f'other{index}', e_other, other, other_ctx)) <= 1e-8
            A, _ = tiny_V(model, site, [data], 2, cotangents=[z], cfg=CFG)
            exact_d = joint_correction(model, [site], {site: A}, data, CFG, mu=mu, loss_spec=spec, ctx=context,
                                       cotangent=residual, output_factor=exact.factor(data, context))
            cg_d = joint_correction(model, [site], {site: A}, data, CFG, mu=mu, loss_spec=spec, ctx=context,
                                    cotangent=residual)
            flat = lambda d: torch.cat([v.flatten() for v in d.values()])
            assert relative(flat(exact_d[0]), flat(cg_d[0])) <= 1e-8
            for key in ('V', 'rho'):
                assert exact_d[3][key] == pytest.approx(cg_d[3][key], rel=1e-8, abs=1e-25)
        factor = exact.factor(data, context)
        assert factor.factor_error is None and factor.range_ok


def test_cholesky_matches_eigh(monkeypatch):
    model, site, data = problem(); context = {'targets': data[1]}
    e = CE().residual(model(data[0]), context)
    for mu in (1e-3, 1e-2):
        chol = SolveCache(model, site, data, CE(), context, CFG, mu=mu, solver='output_exact').solve('e', e, data, context)
        monkeypatch.setattr(vo, 'OUTPUT_EXACT_MAX', 4096)
        eigh = SolveCache(model, site, data, CE(), context, CFG, mu=mu, solver='output_exact').solve('e', e, data, context)
        monkeypatch.setattr(vo, 'OUTPUT_EXACT_MAX', 4)
        assert relative(chol, eigh) <= 1e-10


def test_cholesky_group_submatrix():
    model, site, data = problem(); context = {'targets': data[1]}
    for mu in (1e-3, 1e-2):
        exact = SolveCache(model, site, data, CE(), context, CFG, mu=mu, solver='output_exact')
        parent = exact.factor(data, context)
        mask = torch.tensor([True, False, True, False, True, False])
        subset = tuple(v[mask] for v in data); sub_ctx = {'targets': subset[1]}
        exact.factor(subset, sub_ctx, parent=parent, mask=mask)
        residual = CE().residual(model(subset[0]), sub_ctx)
        z = exact.solve('group', residual, subset, sub_ctx)
        reference = SolveCache(model, site, subset, CE(), sub_ctx, CFG, mu=mu, solver='cg').solve('group', residual, subset, sub_ctx)
        assert relative(z, reference) <= 1e-8
