import json
from pathlib import Path


def notebook_source(seed):
    notebook = json.loads(Path(
        f"notebooks/kaggle_vgg16_seed{seed}_end_to_end_t4x2.ipynb"
    ).read_text())
    return notebook, "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])


def test_vgg16_uses_native_gromo_container_and_eight_conv_links():
    source = Path("probe/vgg_tiny_adapter.py").read_text()
    assert "from gromo.containers.vgg import VGG" in source
    assert "class GromoVGG16" in source
    assert "class VggTinyAdapter" in source
    assert 'f"stages.{stage_index}.links.{link_index}"' in source
    assert "pair.second_layer" in source
    assert "model.core.compute_optimal_updates" in source


def test_vgg16_notebooks_are_three_seed_matched_runs():
    normalized = []
    for seed in (0, 1, 2):
        notebook, source = notebook_source(seed)
        assert f"SEED = {seed}" in source
        assert f"vgg16_seed{seed}_stall150_v1" in source
        assert '"--architecture", "vgg16"' in source
        assert '"--site", "stages.2.links.0"' in source
        assert '"--stall-patience", "150"' in source
        assert '"--post-fork-epochs", "150"' in source
        assert '"--retrigger-patience", "10"' in source
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
