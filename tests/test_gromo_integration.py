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


@pytest.mark.gromo_integration
def test_full_cifar_resnet_relaxed_bypass_embed_and_contract():
    _optional_imports()
    from baselines.bypass import (
        activations, embed_relaxed_bypass, project_relaxed_bypass_)
    from experiments.shared_protocol import build_cifar_gromo_resnet18

    device = torch.device("cuda:0")
    model = build_cifar_gromo_resnet18(device).eval()
    inputs = torch.randn(2, 3, 32, 32, device=device)
    with torch.no_grad():
        expected = model(inputs).clone()
    paths = embed_relaxed_bypass(model)
    modules = activations(model)
    assert paths and len(paths) == len(modules)
    assert all(module.d.is_cuda and module.d.numel() in {64, 128, 256, 512}
               for module in modules)
    with torch.no_grad():
        assert torch.equal(model(inputs), expected)
    assert project_relaxed_bypass_(model) == len(paths)
    with torch.no_grad():
        assert torch.equal(model(inputs), expected)


@pytest.mark.gromo_integration
def test_full_cifar_gromo_resnet34_has_canonical_blocks_and_forward():
    _optional_imports()
    from experiments.shared_protocol import build_cifar_gromo_resnet34

    device = torch.device("cuda:0")
    model = build_cifar_gromo_resnet34(device).eval()
    refs = model.growing_blocks()
    assert len(refs) == 16
    assert [int(ref.module.hidden_neurons) for ref in refs] == (
        [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3)
    assert refs[0].name == "stages.0.blocks.0"
    assert refs[-1].name == "stages.3.blocks.2"
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 32, 32, device=device))
    assert logits.shape == (2, 100)
    assert torch.isfinite(logits).all()


@pytest.mark.gromo_integration
def test_full_cifar_gromo_vgg16_tiny_candidate_and_projection():
    _, GrowthBudget = _optional_imports()
    from experiments.shared_protocol import build_cifar_gromo_vgg16
    from methods import EProjection
    from probe import CandidateExpansionProbe, CounterfactualTinyProbe
    from probe.vgg_tiny_adapter import VggTinyAdapter
    from projection import FunctionalProjector

    torch.manual_seed(31415)
    device = torch.device("cuda:0")
    model = build_cifar_gromo_vgg16(device).eval()
    refs = model.growing_blocks()
    assert len(refs) == 12
    assert refs[0].name == "stages.0.links.0"
    assert [ref.name for ref in refs if ref.is_boundary] == [
        "stages.0.boundary_to_1", "stages.1.boundary_to_2",
        "stages.2.boundary_to_3", "stages.3.boundary_to_4"]
    assert refs[-1].name == "stages.4.links.1"
    # A 2-image probe can leave a rank-1 deep ReLU feature inactive for the
    # entire batch, making a valid TINY candidate appear to have zero delta-f.
    # Match the actual experiment's larger functional batches instead.
    inputs = torch.randn(16, 3, 32, 32, device=device)
    targets = torch.arange(16, device=device, dtype=torch.long) % 100
    with torch.no_grad():
        logits = model(inputs)
    assert logits.shape == (16, 100)
    adapter = VggTinyAdapter(10**9, max_statistics_batches=1)
    projector = FunctionalProjector(
        damping=1e-3, max_iter=20, tolerance=1e-2,
        preconditioner_probes=1)
    for ref in refs:
        pair = ref.module
        before_state = {name: value.detach().clone()
                        for name, value in model.state_dict().items()}
        before_targets = [layer.target_in_neurons
                          for layer in model.core._growable_layers]
        before_index = model.core.layer_to_grow_index
        before_rng = torch.random.get_rng_state().clone()
        before_cuda_rng = torch.cuda.get_rng_state(device).clone()
        before_virtual_state = (
            pair.first_layer.extended_output_layer,
            pair.second_layer.extended_input_layer,
            pair.first_layer.output_extension_scaling.detach().clone(),
            pair.second_layer.input_extension_scaling.detach().clone(),
            pair.second_layer.optimal_delta_scaling.detach().clone(),
            model._extended_forward)
        candidate = CounterfactualTinyProbe(1, ref.name).propose(
            adapter, model, [(inputs, targets)], GrowthBudget(10**9),
            sample_inputs=inputs)
        assert all(torch.equal(value, before_state[name])
                   for name, value in model.state_dict().items())
        assert [layer.target_in_neurons
                for layer in model.core._growable_layers] == before_targets
        assert model.core.layer_to_grow_index == before_index
        if ref.is_boundary:
            assert torch.equal(torch.random.get_rng_state(), before_rng)
            assert torch.equal(torch.cuda.get_rng_state(device), before_cuda_rng)
        assert pair.first_layer.extended_output_layer is before_virtual_state[0]
        assert pair.second_layer.extended_input_layer is before_virtual_state[1]
        assert torch.equal(pair.first_layer.output_extension_scaling,
                           before_virtual_state[2])
        assert torch.equal(pair.second_layer.input_extension_scaling,
                           before_virtual_state[3])
        assert torch.equal(pair.second_layer.optimal_delta_scaling,
                           before_virtual_state[4])
        assert model._extended_forward is before_virtual_state[5]
        assert candidate.payload["effective_rank"] == 1
        if ref.is_boundary:
            assert candidate.payload["bridge_statistics_finite"] is True
            assert candidate.payload["bridge"] == (
                "actual_maxpool_forward_with_destination_gradient")
            assert candidate.payload["solver"] == (
                "damped_pool_aware_least_squares")
            assert candidate.payload["novel_source_feature_direction"] is False
            assert candidate.payload["statistics_samples"] == len(targets)
            assert candidate.payload["deployed_fit_feature_relative_error"] < 1e-4
            assert (candidate.payload["least_squares_residual_after"] <=
                    candidate.payload["least_squares_residual_before"] * (1 + 1e-8))
        else:
            assert all(torch.isfinite(torch.tensor(value))
                       for value in candidate.payload["tiny_eigenvalues"])

        with torch.no_grad():
            base = model(inputs).clone()
            with candidate.virtual_direction(0.0):
                zero_gate = model(inputs)
        assert torch.equal(base, zero_gate), f"E(0) != O at {ref.name}"
        try:
            signal = CandidateExpansionProbe()(
                model, candidate=candidate, batch=(inputs, targets), gate=0.05)
        except RuntimeError as exc:
            raise RuntimeError(
                f"VGG functional probe failed at site {ref.name}: {exc}") from exc
        assert torch.isfinite(signal.delta_logits).all()
        assert signal.delta_logits.norm() > 0

        if ref.is_boundary:
            # Observe the auxiliary tensor at the actual pool and at the
            # consumer extension input; shape alone is not enough.
            pool_branch = []
            consumer_branch = []

            def pool_hook(_module, module_inputs, output):
                if module_inputs[0].shape[1] == candidate.payload["effective_rank"]:
                    pool_branch.append((module_inputs[0].detach().clone(),
                                        output.detach().clone()))

            bridge_handle = pair.bridge.register_forward_hook(pool_hook)
            with candidate.virtual_direction(0.05):
                extension_input = pair.second_layer.extended_input_layer
                assert extension_input is not None
                input_handle = extension_input.register_forward_pre_hook(
                    lambda _module, args: consumer_branch.append(
                        args[0].detach().clone()))
                with torch.no_grad():
                    model(inputs)
                input_handle.remove()
            bridge_handle.remove()
            assert len(pool_branch) == 1
            before_pool, after_pool = pool_branch[0]
            assert before_pool.shape[1] == candidate.payload["effective_rank"]
            assert after_pool.shape[1] == candidate.payload["effective_rank"]
            assert after_pool.shape[-2] * 2 == before_pool.shape[-2]
            assert after_pool.shape[-1] * 2 == before_pool.shape[-1]
            assert len(consumer_branch) == 1
            assert torch.equal(after_pool, consumer_branch[0])

            # Repartition identical samples into separate minibatches. Sum-CE
            # sufficient statistics should yield the same boundary proposal.
            split_adapter = VggTinyAdapter(10**9, max_statistics_batches=4)
            split_candidate = CounterfactualTinyProbe(1, ref.name).propose(
                split_adapter, model,
                [(inputs[i:i + 4], targets[i:i + 4])
                 for i in range(0, len(targets), 4)],
                GrowthBudget(10**9), sample_inputs=inputs)
            with torch.no_grad():
                with candidate.virtual_direction(0.05):
                    candidate_logits = model(inputs).clone()
                with split_candidate.virtual_direction(0.05):
                    split_logits = model(inputs).clone()
            assert split_candidate.payload["statistics_samples"] == len(targets)
            assert torch.allclose(candidate_logits, split_logits,
                                  rtol=1e-4, atol=1e-5)
            assert (candidate.payload["selected_source_channels"] ==
                    split_candidate.payload["selected_source_channels"])

            if ref.name == "stages.0.boundary_to_1":
                # Requested scale/fairness check at the actual statistics
                # budget: identical 256 examples partitioned as 4x64 vs 2x128.
                partition_inputs = torch.randn(256, 3, 32, 32, device=device)
                partition_targets = torch.randint(0, 100, (256,), device=device)
                batches_64 = [
                    (partition_inputs[i:i + 64], partition_targets[i:i + 64])
                    for i in range(0, 256, 64)]
                batches_128 = [
                    (partition_inputs[i:i + 128], partition_targets[i:i + 128])
                    for i in range(0, 256, 128)]
                candidate_4x64 = CounterfactualTinyProbe(1, ref.name).propose(
                    VggTinyAdapter(10**9, max_statistics_batches=4), model,
                    batches_64, GrowthBudget(10**9),
                    sample_inputs=partition_inputs[:2])
                candidate_2x128 = CounterfactualTinyProbe(1, ref.name).propose(
                    VggTinyAdapter(10**9, max_statistics_batches=2), model,
                    batches_128, GrowthBudget(10**9),
                    sample_inputs=partition_inputs[:2])
                assert candidate_4x64.payload["statistics_samples"] == 256
                assert candidate_2x128.payload["statistics_samples"] == 256
                assert (candidate_4x64.payload["selected_source_channels"] ==
                        candidate_2x128.payload["selected_source_channels"])
                with torch.no_grad():
                    with candidate_4x64.virtual_direction(0.05):
                        logits_4x64 = model(inputs).clone()
                    with candidate_2x128.virtual_direction(0.05):
                        logits_2x128 = model(inputs).clone()
                assert torch.allclose(logits_4x64, logits_2x128,
                                      rtol=1e-4, atol=1e-5)

            if ref.is_boundary:
                # Verify the bridge-aware finite difference against autograd of
                # the exact Conv→MaxPool→Conv path, not a resized surrogate.
                # Native intra-stage TINY candidates use a different operator
                # and are not covered by this MaxPool-specific sanity check.
                probe = torch.randn_like(base)
                gate = torch.tensor(1e-3, device=device, requires_grad=True)
                with candidate.virtual_direction(gate):
                    score = (model(inputs) * probe).sum()
                autograd_direction = torch.autograd.grad(score, gate)[0]
                epsilon = 1e-4
                gate_value = float(gate.detach())
                with torch.no_grad():
                    with candidate.virtual_direction(gate_value):
                        finite_base = model(inputs).clone()
                    with candidate.virtual_direction(gate_value + epsilon):
                        finite_logits = model(inputs).clone()
                finite_direction = (
                    ((finite_logits - finite_base) / epsilon) * probe).sum()
                assert torch.isfinite(autograd_direction), ref.name
                assert torch.isfinite(finite_direction), ref.name
                assert torch.allclose(
                    autograd_direction, finite_direction,
                    rtol=0.1, atol=1e-2), ref.name

        step = EProjection(projector=projector).discover_candidate(
            model, candidate, (inputs, targets), gate=0.05)
        assert step.projection.parameter_delta
        assert torch.isfinite(step.projection.fitted_delta).all()
        allowed = {id(parameter)
                   for module in model.projection_parameter_modules(ref.name)
                   for parameter in module.parameters()}
        named_ids = {id(parameter): name
                     for name, parameter in model.named_parameters()}
        allowed_names = {named_ids[parameter_id] for parameter_id in allowed
                         if parameter_id in named_ids}
        assert set(step.projection.parameter_delta).issubset(allowed_names)
