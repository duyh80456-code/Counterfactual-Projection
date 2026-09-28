"""Real Gromo/TINY integration; required by the Kaggle notebook gate."""

import os

import pytest
import torch


def _optional_imports():
    required = os.environ.get("REQUIRE_GROMO_INTEGRATION") == "1"
    try:
        from dual_growth.adapters import TinyAdapter
        from dual_growth.controller import GrowthBudget
        import gromo  # noqa: F401
    except Exception as error:
        if required:
            pytest.fail(f"required Gromo integration imports failed: {error}")
        pytest.skip(f"optional Gromo/reference checkout unavailable: {error}")
    if not torch.cuda.is_available():
        if required:
            pytest.fail("required Gromo integration test needs CUDA")
        pytest.skip("real Gromo integration runs in the Kaggle CUDA gate")
    return TinyAdapter, GrowthBudget


@pytest.mark.gromo_integration
def test_full_model_tiny_overexpansion_to_functional_projection():
    TinyAdapter, GrowthBudget = _optional_imports()
    from methods import EProjection
    from baselines import RealEOracle
    from probe import CounterfactualTinyProbe, build_pretrained_gromo_resnet18
    from projection import FunctionalProjector

    torch.manual_seed(31415)
    device = torch.device("cuda:0")
    model = build_pretrained_gromo_resnet18(100, device=device).eval()
    site = "stages.2.blocks.0"
    block = model.block(site)
    base_width = int(block.second_layer.in_neurons)
    assert base_width == 256
    assert int(block.second_layer.target_in_neurons) == base_width
    inputs = torch.randn(2, 3, 64, 64, device=device)
    targets = torch.tensor([3, 17], device=device)
    adapter = TinyAdapter(
        quantum_params=10**9, max_statistics_batches=1,
        numerical_threshold=0.0, statistical_threshold=0.0)
    candidate = CounterfactualTinyProbe(1, site).propose(
        adapter, model, [(inputs, targets)], GrowthBudget(10**9),
        sample_inputs=inputs)
    assert candidate.payload["counterfactual_over_expansion"]
    assert candidate.payload["base_hidden_width"] == 256
    assert candidate.payload["counterfactual_target_width"] == 257
    assert int(block.second_layer.in_neurons) == 256
    assert int(block.second_layer.target_in_neurons) == 256
    assert candidate.payload["actual_extension_flops"] > 0

    step = EProjection(projector=FunctionalProjector(
        damping=1e-3, max_iter=1, tolerance=1e-4)).discover_candidate(
            model, candidate, (inputs, targets), gate=0.05)
    assert step.signal.is_structural_expansion
    assert step.signal.delta_logits.norm() > 0
    assert torch.isfinite(step.projection.fitted_delta).all()
    assert step.projection.jvp_calls >= 2
    assert step.projection.vjp_calls >= 2
    assert int(block.second_layer.in_neurons) == 256
    assert int(block.second_layer.target_in_neurons) == 256
    committed = RealEOracle.commit_(model, candidate).committed_module
    assert int(committed.second_layer.in_neurons) == 257
    assert int(committed.second_layer.target_in_neurons) == 257
