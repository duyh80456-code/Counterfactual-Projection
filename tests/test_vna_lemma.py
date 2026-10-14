"""Nested Gauss-Newton projections: equality at zero damping, O(mu) gap."""
import torch
from methods.vna import dedicated_generator


def test_nested_gn_velocity_projection():
    gen = dedicated_generator(15, 'nested-GN')
    J = torch.randn(30, 8, dtype=torch.float64, generator=gen)
    E = torch.randn(30, 7, dtype=torch.float64, generator=gen)
    residual = torch.randn(30, dtype=torch.float64, generator=gen)
    grown = torch.cat((J, E), 1)
    projection = J @ torch.linalg.pinv(J)
    def velocity(matrix, mu):
        return -matrix @ torch.linalg.solve(matrix.T @ matrix + mu*torch.eye(matrix.shape[1], dtype=matrix.dtype), matrix.T @ residual)
    fixed = velocity(J, 1e-12)
    assert float((projection @ velocity(grown, 1e-12)-fixed).norm()/fixed.norm()) <= 1e-6
    gaps = [float((projection @ velocity(grown, mu)-velocity(J, mu)).norm()) for mu in (1e-2, 1e-3, 1e-4)]
    assert .09 < gaps[1]/gaps[0] < .11
    assert .09 < gaps[2]/gaps[1] < .11
