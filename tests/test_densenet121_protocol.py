import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from methods.e_projection import candidate_projection_parameter_names
from probe import CandidateExpansionProbe
from probe.densenet_auxiliary import (
    CifarDenseNet121, propose_denseblock_candidates)


SITES = tuple(
    f"core.features.denseblock{index}" for index in range(1, 5))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(7)
    return CifarDenseNet121(num_classes=100, device="cpu").eval()


def notebook(seed):
    return json.loads(Path(
        f"notebooks/kaggle_densenet121_seed{seed}_end_to_end_t4x2.ipynb"
    ).read_text())


def notebook_source(seed):
    payload = notebook(seed)
    return payload, "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"])


def test_densenet_exposes_four_architecture_native_boundary_sites(model):
    assert tuple(ref.name for ref in model.growing_blocks()) == SITES
    assert model.boundary_shapes(SITES[0]) == (256, 128, False)
    assert model.boundary_shapes(SITES[1]) == (512, 256, False)
    assert model.boundary_shapes(SITES[2]) == (1024, 512, False)
    assert model.boundary_shapes(SITES[3]) == (1024, 100, True)


def test_denseblock_auxiliary_is_exactly_anchored_at_gate_zero(model):
    inputs = torch.randn(1, 3, 32, 32)
    auxiliary = model.make_auxiliary(SITES[0], rank=2).eval()
    with torch.no_grad():
        baseline = model(inputs)
        with model.auxiliary_context(SITES[0], auxiliary, gate=0.0):
            anchored = model(inputs)
        with model.auxiliary_context(SITES[0], auxiliary, gate=0.05):
            expanded = model(inputs)
    assert torch.equal(baseline, anchored)
    assert not torch.equal(baseline, expanded)


def test_denseblock_candidate_produces_a_real_functional_direction(model):
    statistics = [(torch.randn(2, 3, 32, 32), torch.tensor([1, 2]))]
    before = {name: value.detach().clone()
              for name, value in model.state_dict().items()}
    candidate = propose_denseblock_candidates(
        model, statistics, rank=2, sites=[SITES[0]], steps=1)[0]
    signal = CandidateExpansionProbe()(
        model, candidate=candidate, batch=statistics[0], gate=0.05)
    assert candidate.payload["source"] == "densenet_block_boundary_auxiliary"
    assert candidate.payload["function_preserving_at_gate_zero"] is True
    assert signal.source == "densenet_block_boundary_auxiliary"
    assert float(signal.delta_logits.norm()) > 0
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_projection_scope_is_selected_block_plus_direct_consumer(model):
    candidate = SimpleNamespace(module_name=SITES[1])
    names = candidate_projection_parameter_names(
        model, candidate, scope="residual_path")
    assert any(name.startswith("core.features.denseblock2.") for name in names)
    assert any(name.startswith("core.features.transition2.") for name in names)
    assert not any(name.startswith("core.features.denseblock1.") for name in names)
    assert not any(name.startswith("core.features.denseblock3.") for name in names)

    final_names = candidate_projection_parameter_names(
        model, SimpleNamespace(module_name=SITES[3]),
        scope="residual_path")
    assert any(name.startswith("core.features.denseblock4.")
               for name in final_names)
    assert any(name.startswith("core.features.norm5.") for name in final_names)
    assert any(name.startswith("core.classifier.") for name in final_names)


def test_densenet_notebooks_are_three_seed_matched_runs():
    normalized = []
    for seed in (0, 1, 2):
        payload, source = notebook_source(seed)
        assert f"SEED = {seed}" in source
        assert f"densenet121_seed{seed}_stall150_v1" in source
        assert '"--architecture", "densenet121"' in source
        assert '"--site", "core.features.denseblock3"' in source
        assert '"--stall-patience", "150"' in source
        assert '"--post-fork-epochs", "150"' in source
        assert '"ours_e_driven_o": 10' in source
        assert '"o_projection_only": 10' in source
        assert ('"--retrigger-patience", str(retrigger_patience[name])' in
                source)
        assert '"o_projection_only": "single_initial_intervention"' in source
        assert "all 4 DenseBlock boundaries" in source
        assert 'GPU0: E-driven O; GPU1: O-only' in source
        assert 'launch(1, "bypass")' not in source
        for index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]),
                        f"densenet121-seed{seed}-cell{index}", "exec")
        normalized.append(
            source.replace(f"seed {seed}", "seed SEED")
                  .replace(f"seed{seed}", "seedSEED")
                  .replace(f"SEED = {seed}", "SEED = SEED")
                  .replace(f'"--seed", "{seed}"', '"--seed", "SEED"'))
    assert normalized[0] == normalized[1] == normalized[2]


def test_densenet_reuses_verified_kaggle_plumbing_without_touching_vgg():
    for seed in (0, 1, 2):
        dense = notebook(seed)
        vgg = json.loads(Path(
            f"notebooks/kaggle_vgg16_seed{seed}_end_to_end_t4x2.ipynb"
        ).read_text())
        dense_bootstrap = "".join(dense["cells"][1]["source"]).replace(
            f"densenet121_seed{seed}_stall150_v1", "ARCH_OUTPUT")
        vgg_bootstrap = "".join(vgg["cells"][1]["source"]).replace(
            f"vgg16_seed{seed}_stall150_v1", "ARCH_OUTPUT")
        assert dense_bootstrap == vgg_bootstrap
        assert ("".join(dense["cells"][2]["source"]) ==
                "".join(vgg["cells"][2]["source"]))


def test_densenet_is_wired_without_changing_legacy_defaults():
    phase1 = Path("experiments/run_unified_vanilla_to_stall.py").read_text()
    phase2 = Path("experiments/run_plateau_fork.py").read_text()
    shared = Path("experiments/shared_protocol.py").read_text()
    selector = Path("experiments/run_shared_comparison.py").read_text()
    choices = 'choices=("resnet18", "resnet34", "vgg16", "densenet121")'
    assert choices in phase1
    assert choices in phase2
    assert 'default="resnet18"' in phase1
    assert 'default="resnet18"' in phase2
    assert '"densenet121": build_cifar_densenet121' in shared
    assert '"densenet121": "CIFAR-DenseNet121"' in shared
    assert "propose_denseblock_candidates" in selector
    assert "mean_observed_structural_E_gain" in Path(
        "experiments/run_plateau_comparison.py").read_text()
