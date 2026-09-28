import torch
from torch import nn
from torch.func import functional_call, jvp

from projection import FunctionalProjector
from projection.cg import CGResult
from experiments.run_gromo_pilot import (
    actual_update_metrics, eval_logits, reset_projected_momentum,
    reset_residual_path_momentum,
    sign_randomized_parameter_delta)


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
    assert not result.cg.solution.requires_grad
    assert not result.fitted_delta.requires_grad
    assert len(result.cg.residual_history) == result.cg.iterations + 1
    heldout_inputs = torch.randn(5, 3)

    def heldout_function(weight):
        return functional_call(model, {"0.weight": weight}, (heldout_inputs,))

    heldout_target = jvp(
        heldout_function, (parameter,), (known,))[1]
    heldout = FunctionalProjector().evaluate_direction(
        model, heldout_inputs, heldout_target, result.parameter_delta)
    assert heldout.relative_residual < 1e-4
    assert heldout.cosine_alignment > 0.999


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


def test_momentum_reset_control_matches_residual_path_projection_scope():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Linear(3, 3)
            self.conv2 = nn.Linear(3, 3)
            self.bn1 = nn.BatchNorm1d(3)
            self.bn2 = nn.BatchNorm1d(3)
            self.downsample = nn.Linear(3, 3)

        def forward(self, inputs):
            return self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(inputs))))) + \
                self.downsample(inputs)

    block = Block().train()
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1, momentum=0.9)
    block(torch.randn(5, 3)).sum().backward()
    optimizer.step()
    assert all(parameter in optimizer.state for parameter in block.parameters())
    model = nn.Module()
    model.block = block
    reset = reset_residual_path_momentum(optimizer, model, "block")
    assert reset == sum(1 for module in (
        block.conv1, block.bn1, block.conv2, block.bn2)
                        for _ in module.parameters())
    assert all(parameter not in optimizer.state
               for module in (block.conv1, block.bn1, block.conv2, block.bn2)
               for parameter in module.parameters())
    assert all(parameter in optimizer.state for parameter in block.downsample.parameters())


def test_sign_randomized_control_preserves_each_tensor_norm():
    delta = {
        "conv.weight": torch.randn(4, 3, 3, 3),
        "bn.weight": torch.randn(4) * 0.01,
        "bn.bias": torch.randn(4) * 10,
    }
    randomized = sign_randomized_parameter_delta(
        delta, generator=torch.Generator().manual_seed(91))
    assert randomized.keys() == delta.keys()
    assert all(torch.equal(randomized[name].abs(), value.abs())
               for name, value in delta.items())
    assert any(not torch.equal(randomized[name], value)
               for name, value in delta.items())


def test_actual_update_metrics_measure_realized_function_change():
    model = nn.Linear(3, 2, bias=False).eval()
    inputs = torch.randn(5, 3)
    update = torch.randn_like(model.weight) * 0.01
    scale = 0.05
    baseline = eval_logits(model, inputs)
    target = inputs @ update.t()
    with torch.no_grad():
        model.weight.add_(update, alpha=scale)
    metrics = actual_update_metrics(model, inputs, baseline, target, scale)
    assert metrics["actual_heldout_relative_residual"] < 1e-4
    assert metrics["actual_heldout_cosine_alignment"] > 0.999


def test_projector_retries_with_stronger_damping(monkeypatch):
    import projection.functional as functional_module

    calls = []

    def fake_cg(matvec, rhs, **_kwargs):
        calls.append(matvec)
        converged = len(calls) == 2
        return CGResult(
            solution=torch.zeros_like(rhs), iterations=1,
            residual_norm=0.0 if converged else 1.0,
            converged=converged, residual_history=(1.0,))

    monkeypatch.setattr(functional_module, "conjugate_gradient", fake_cg)
    model = nn.Linear(2, 2, bias=False)
    inputs = torch.randn(2, 2)
    result = FunctionalProjector(
        damping=1e-3, max_iter=1, max_damping_retries=2).project(
            model, inputs, torch.randn(2, 2), block="")
    assert result.cg.converged
    assert result.damping_requested == 1e-3
    assert result.damping_used == 1e-2
    assert [attempt.damping for attempt in result.cg_attempts] == [1e-3, 1e-2]
