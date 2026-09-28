import torch
from torch import nn
from torch.func import functional_call, jvp

from projection import FunctionalProjector
from experiments.run_gromo_pilot import reset_projected_momentum


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
    assert result.jvp_calls >= 2
    assert result.vjp_calls >= 2


def test_direct_projection_clears_only_touched_momentum():
    torch.manual_seed(6)
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = model(torch.randn(4, 3)).sum()
    loss.backward()
    optimizer.step()
    assert all(parameter in optimizer.state for parameter in model.parameters())
    inputs = torch.randn(4, 3)
    target = torch.randn(4, 2) * 0.01
    result = FunctionalProjector(max_iter=2).project(
        model, inputs, target, block="0")
    reset = reset_projected_momentum(optimizer, model, result)
    assert reset == 2
    assert all(parameter not in optimizer.state for parameter in model[0].parameters())
    assert all(parameter in optimizer.state for parameter in model[1].parameters())
