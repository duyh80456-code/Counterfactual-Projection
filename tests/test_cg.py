import torch

from projection import conjugate_gradient


def test_conjugate_gradient_solves_spd_system():
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    rhs = torch.tensor([1.0, 2.0], requires_grad=True)
    result = conjugate_gradient(lambda value: matrix @ value, rhs,
                                max_iter=10, tolerance=1e-7, eps=1e-30)
    assert result.converged
    assert result.relative_residual <= 1e-7
    assert not result.solution.requires_grad
    assert len(result.residual_history) == result.iterations + 1
    assert torch.allclose(result.solution, torch.linalg.solve(matrix, rhs),
                          atol=1e-6)


def test_preconditioned_cg_handles_a_badly_scaled_diagonal_system():
    diagonal = torch.tensor([1e-4, 1.0, 1e4], dtype=torch.double)
    rhs = torch.tensor([1.0, -2.0, 3.0], dtype=torch.double)
    result = conjugate_gradient(
        lambda value: diagonal * value, rhs,
        preconditioner=lambda value: value / diagonal,
        max_iter=2, tolerance=1e-10)
    assert result.converged
    assert result.relative_residual <= 1e-10
    assert torch.allclose(result.solution, rhs / diagonal)


def test_default_cg_normalizes_small_rhs_and_rescales_diagnostics():
    rhs = torch.tensor([1e-20, 2e-20], dtype=torch.float64)
    kwargs = dict(tolerance=1e-10, preconditioner=lambda v: v/2)
    result = conjugate_gradient(lambda v: 2*v, rhs, **kwargs)
    unit = conjugate_gradient(lambda v: 2*v, rhs/rhs.norm(), normalize_rhs=False, **kwargs)
    assert result.converged and not result.breakdown
    torch.testing.assert_close(result.solution, rhs/2, rtol=1e-12, atol=0)
    assert result.residual_norm == unit.residual_norm * float(rhs.norm())
    assert result.residual_history == tuple(v * float(rhs.norm()) for v in unit.residual_history)
    assert result.relative_residual == unit.relative_residual


def test_normalized_cg_matches_original_scale_and_warm_start():
    matrix = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64)
    rhs = torch.tensor([1., 2.], dtype=torch.float64)
    x0 = torch.tensor([.1, -.2], dtype=torch.float64)
    kwargs = dict(x0=x0, max_iter=10, tolerance=1e-13)
    old = conjugate_gradient(lambda v: matrix @ v, rhs, normalize_rhs=False, **kwargs)
    result = conjugate_gradient(lambda v: matrix @ v, rhs, **kwargs)
    assert old.converged and result.converged
    assert float((result.solution-old.solution).norm()/old.solution.norm()) <= 1e-12
