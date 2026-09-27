import torch

from projection import conjugate_gradient


def test_conjugate_gradient_solves_spd_system():
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    rhs = torch.tensor([1.0, 2.0])
    result = conjugate_gradient(lambda value: matrix @ value, rhs,
                                max_iter=10, tolerance=1e-7)
    assert result.converged
    assert torch.allclose(result.solution, torch.linalg.solve(matrix, rhs),
                          atol=1e-6)

