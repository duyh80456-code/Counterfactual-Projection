import torch
from torch import nn

from methods.e_repopt import RepOptGradientHandler


def test_repopt_transform_matches_formula():
    parameter = nn.Parameter(torch.zeros(2, 3))
    direction = torch.tensor([[1.0, 2.0, 0.0]])
    gradient = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    handler = RepOptGradientHandler(parameter, direction, strength=0.5)
    expected = gradient + 0.5 * (gradient @ direction.T) @ direction
    assert torch.equal(handler.transform(gradient), expected)

