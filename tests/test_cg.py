import torch

from projection import conjugate_gradient


def test_conjugate_gradient_solves_spd_system():
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    rhs = torch.tensor([1.0, 2.0], requires_grad=True)
    result = conjugate_gradient(lambda value: matrix @ value, rhs,
                                max_iter=10, tolerance=1e-7)
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
