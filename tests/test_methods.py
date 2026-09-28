import torch
from torch import nn

import pytest

from methods.e_repopt import ERepOpt, RepOptGradientHandler
from probe import ProbeSignal


def test_repopt_transform_matches_formula():
    parameter = nn.Parameter(torch.zeros(2, 3))
    direction = torch.tensor([[1.0, 2.0, 0.0]])
    gradient = torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
    handler = RepOptGradientHandler(parameter, direction, strength=0.5)
    expected = gradient + 0.5 * (gradient @ direction.T) @ direction
    assert torch.equal(handler.transform(gradient), expected)


def test_structural_repopt_is_explicitly_excluded_without_rank_coordinates():
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    signal = ProbeSignal(
        block="logical.block.name", A_E=None, B_E=None, delta_feature=None,
        delta_logits=torch.ones(1, 2), predicted_gain=0.0,
        singular_values=None, source="tiny_gromo_structural",
        is_structural_expansion=True)
    with pytest.raises(NotImplementedError, match="excluded"):
        ERepOpt.from_signal(optimizer, model, signal)
