import json
from pathlib import Path
from types import SimpleNamespace

import torch


def notebook_source(seed):
    notebook = json.loads(Path(
        f"notebooks/kaggle_vgg16_seed{seed}_end_to_end_t4x2.ipynb"
    ).read_text())
    return notebook, "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])


def test_vgg16_exposes_twelve_native_and_pool_bridge_conv_links():
    source = Path("probe/vgg_tiny_adapter.py").read_text()
    assert "from gromo.containers.vgg import VGG" in source
    assert "class GromoVGG16" in source
    assert "class VggTinyAdapter" in source
    assert 'f"stages.{stage_index}.links.{link_index}"' in source
    assert 'f"stages.{stage_index}.boundary_to_{stage_index + 1}"' in source
    assert '"vgg_pool_bridge_closed_form_autograd"' in source
    assert '"actual_maxpool_forward_with_destination_gradient"' in source
    assert '"damped_pool_aware_least_squares"' in source
    assert 'reduction="sum"' in source
    assert "optimizer.step()" not in source[source.index("def _propose_pool_bridge"):]
    assert "len(refs) != 12" in source
    assert "pair.second_layer" in source
    assert "model.core.compute_optimal_updates" in source


def test_vgg16_notebooks_are_three_seed_matched_runs():
    normalized = []
    for seed in (0, 1, 2):
        notebook, source = notebook_source(seed)
        assert f"SEED = {seed}" in source
        assert f"vgg16_seed{seed}_stall150_v1" in source
        assert '"--architecture", "vgg16"' in source
        # O-only needs a concrete projection scope. E-driven O's own
        # plateau selector scans all 12 VGG sites independently of this flag.
        assert '"--site", "stages.2.links.0"' in source
        assert '"--site-selection-mode", "all_functional_gain"' in source
        assert "all 12 adjacent VGG16 conv interfaces (8 native + 4 operator-aware MaxPool-bridge)" in source
        assert '"--stall-patience", "150"' in source
        assert '"--post-fork-epochs", "150"' in source
        assert '"ours_e_driven_o": 25' in source
        assert '"o_projection_only": 10' in source
        assert ('"--retrigger-patience", str(retrigger_patience[name])' in
                source)
        assert '"o_projection_only": "single_initial_intervention"' in source
        assert 'name != "ours_e_driven_o" or' in source
        assert 'GPU0: E-driven O; GPU1: O-only' in source
        assert 'launch(0, "ours_e_driven_o")' in source
        assert 'launch(1, "o_projection_only")' in source
        assert 'launch(1, "bypass")' not in source
        assert "CIFAR-ResNet34" not in source
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]),
                        f"vgg16-seed{seed}-cell{index}", "exec")
        normalized.append(
            source.replace(f"seed {seed}", "seed SEED")
                  .replace(f"seed{seed}", "seedSEED")
                  .replace(f"SEED = {seed}", "SEED = SEED")
                  .replace(f'"--seed", "{seed}"', '"--seed", "SEED"'))
    assert normalized[0] == normalized[1] == normalized[2]


def test_vgg16_reuses_the_verified_resnet_kaggle_bootstrap():
    for seed in (0, 1, 2):
        vgg = json.loads(Path(
            f"notebooks/kaggle_vgg16_seed{seed}_end_to_end_t4x2.ipynb"
        ).read_text())
        resnet = json.loads(Path(
            f"notebooks/kaggle_resnet34_seed{seed}_end_to_end_t4x2.ipynb"
        ).read_text())
        vgg_clone = "".join(vgg["cells"][1]["source"]).replace(
            f"vgg16_seed{seed}_stall150_v1", "ARCH_OUTPUT")
        resnet_clone = "".join(resnet["cells"][1]["source"]).replace(
            f"resnet34_seed{seed}_stall150_v1", "ARCH_OUTPUT")
        assert vgg_clone == resnet_clone
        assert ("".join(vgg["cells"][2]["source"]) ==
                "".join(resnet["cells"][2]["source"]))


def test_vgg16_is_supported_by_both_phase_runners():
    phase1 = Path("experiments/run_unified_vanilla_to_stall.py").read_text()
    phase2 = Path("experiments/run_plateau_fork.py").read_text()
    shared = Path("experiments/shared_protocol.py").read_text()
    selector = Path("experiments/run_shared_comparison.py").read_text()
    choices = 'choices=("resnet18", "resnet34", "vgg16", "densenet121")'
    assert choices in phase1
    assert choices in phase2
    assert '"vgg16": build_cifar_gromo_vgg16' in shared
    assert '"vgg16": "CIFAR-VGG16-BN"' in shared
    assert "adapter_type = VggTinyAdapter" in selector
    assert "def select_functional_gain_candidate(" in selector
    assert '"observed_functional_loss_gain"' in selector
    assert '"functional_delta_norm"' in selector


def test_vgg_where_uses_shared_observed_functional_gain(monkeypatch):
    from experiments import run_shared_comparison as runner

    candidates = [
        SimpleNamespace(module_name="native", proposal_score=100.0, payload={}),
        SimpleNamespace(module_name="boundary", proposal_score=0.01, payload={
            "solver": "damped_pool_aware_least_squares"}),
    ]

    class FakeProbe:
        def __call__(self, _model, *, candidate, batch, gate):
            assert batch == "same-selection-batch"
            assert gate == 0.05
            gain = 0.02 if candidate.module_name == "native" else 0.2
            magnitude = 3.0 if candidate.module_name == "native" else 7.0
            return SimpleNamespace(
                observed_loss_gain=gain,
                delta_logits=torch.full((1, 2), magnitude),
                source="fake")

    monkeypatch.setattr(runner, "propose_structural_candidates",
                        lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(runner, "CandidateExpansionProbe", FakeProbe)
    selected, diagnostics = runner.select_functional_gain_candidate(
        model=object(), statistics=[], selection_batch="same-selection-batch",
        rank=1, candidate_sites="", device=torch.device("cpu"), gate=0.05)

    assert selected.module_name == "boundary"
    assert diagnostics["site_scores"] == {"native": 0.02, "boundary": 0.2}
    assert diagnostics["selected_site_score"] == 0.2
    assert diagnostics["selected_functional_delta_norm"] > 0
    assert diagnostics["selection_candidate_count"] == 2


def test_vgg16_seed1_method_only_notebook_skips_vanilla():
    path = Path(
        "notebooks/kaggle_vgg16_seed1_methods_from_plateau_t4x2.ipynb")
    notebook = json.loads(path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'kind="plateau_fork_checkpoint"' in source
    assert 'protocol.get("architecture") == "CIFAR-VGG16-BN"' in source
    assert 'protocol.get("seed") == SEED' in source
    assert 'payload.get("vanilla_baseline_complete") is True' in source
    assert '"ours_e_driven_o": 25' in source
    assert '"o_projection_only": 10' in source
    assert '"o_projection_only": "single_initial_intervention"' in source
    assert "experiments.run_unified_vanilla_to_stall" not in source
    assert "No Phase-1 command is executed below" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]),
                    f"vgg16-method-only-cell{index}", "exec")
