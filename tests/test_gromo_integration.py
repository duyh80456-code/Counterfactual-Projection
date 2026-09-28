"""Real Gromo/TINY integration; required by the Kaggle notebook gate."""

import copy
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
    from baselines import RealEGrowth
    from probe import (
        CandidateExpansionProbe, CounterfactualTinyProbe,
        build_pretrained_gromo_resnet18)
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
        damping=1e-3, max_iter=200, tolerance=1e-2)).discover_candidate(
            model, candidate, (inputs, targets), gate=0.05)
    assert step.signal.is_structural_expansion
    assert step.signal.delta_logits.norm() > 0
    assert torch.isfinite(step.projection.fitted_delta).all()
    assert step.projection.jvp_calls >= 2
    assert step.projection.vjp_calls >= 2
    assert step.projection.cg.iterations <= 200
    assert step.projection.solver_space == "dual_output"
    assert step.projection.linear_system_dimension == inputs.shape[0] * 100
    assert step.projection.solver_dtype == "float64"
    assert step.projection.preconditioner == "hutchinson_jacobi"
    assert step.projection.damping_used >= step.projection.damping_requested
    assert len(step.projection.cg_attempts) <= 5
    # torch.func must not leave GradTrackingTensor wrappers in Gromo's cached
    # attributes or registered BN parameters; real growth deep-copies the block.
    copy.deepcopy(model.block(site))
    assert all("downsample" not in name
               for name in step.projection.parameter_delta)
    assert any("post_layer_function" in name
               for name in step.projection.parameter_delta)
    heldout_inputs = torch.randn(2, 3, 64, 64, device=device)
    heldout_targets = torch.tensor([11, 29], device=device)
    heldout_signal = CandidateExpansionProbe()(
        model, candidate=candidate,
        batch=(heldout_inputs, heldout_targets), gate=0.05)
    heldout = FunctionalProjector().evaluate_direction(
        model, heldout_inputs, heldout_signal.delta_logits,
        step.projection.parameter_delta)
    assert torch.isfinite(torch.tensor(heldout.relative_residual))
    assert torch.isfinite(torch.tensor(heldout.cosine_alignment))
    with torch.no_grad():
        before_correction = model(heldout_inputs).clone()
    step.projection.apply_(model, scale=0.05)
    with torch.no_grad():
        after_correction = model(heldout_inputs)
    assert not torch.equal(before_correction, after_correction)
    assert int(block.second_layer.in_neurons) == 256
    assert int(block.second_layer.target_in_neurons) == 256
    committed = RealEGrowth.commit_(model, candidate).committed_module
    assert int(committed.second_layer.in_neurons) == 257
    assert int(committed.second_layer.target_in_neurons) == 257
