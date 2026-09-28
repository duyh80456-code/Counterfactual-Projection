import torch
from torch import nn
from torch.func import functional_call, jvp

from projection import FunctionalProjector


def test_functional_projection_recovers_tangent_direction():
    torch.manual_seed(5)
    model = nn.Sequential(nn.Linear(3, 2, bias=False)).eval()
    inputs = torch.randn(4, 3)
    parameter = model[0].weight.detach()
    known = torch.randn_like(parameter) * 0.01

    def function(weight):
        return functional_call(model, {"0.weight": weight}, (inputs,))

    target = jvp(function, (parameter,), (known,))[1]
    result = FunctionalProjector(
        damping=1e-8, max_iter=30, tolerance=1e-7).project(
            model, inputs, target, block="0")
    assert result.relative_residual < 1e-4
    assert abs(result.fitted_norm_ratio - 1.0) < 1e-4
