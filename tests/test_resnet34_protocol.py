import json
from pathlib import Path

from experiments.plateau_protocol import validation_stall_plan


def test_resnet34_builder_uses_canonical_basic_block_layout():
    source = Path("experiments/shared_protocol.py").read_text()
    assert "def build_cifar_gromo_resnet34" in source
    assert "number_of_blocks_per_stage=(3, 4, 6, 3)" in source
    assert '"cifar_gromo_resnet34_full_random_v1"' in source
    assert "[64] * 3 + [128] * 4 + [256] * 6 + [512] * 3" in source


def test_validation_stall_plan_supports_exact_150_epoch_patience():
    history = [
        {"epoch": epoch,
         "validation_accuracy": 0.75 if epoch == 225 else 0.74}
        for epoch in range(200, 375)]
    waiting = validation_stall_plan(history, patience=150, min_epoch=200)
    assert waiting["validation_best_epoch"] == 225
    assert waiting["target_epoch"] == 375
    assert waiting["minimum_additional_epochs_if_no_new_best"] == 1
    assert waiting["has_post_best_patience_epochs"] is False
    history.append({"epoch": 375, "validation_accuracy": 0.74})
    ready = validation_stall_plan(history, patience=150, min_epoch=200)
    assert ready["has_post_best_patience_epochs"] is True


def _resnet34_notebook_source(seed):
    notebook = json.loads(Path(
        f"notebooks/kaggle_resnet34_seed{seed}_end_to_end_t4x2.ipynb"
    ).read_text())
    return "\n".join("".join(cell["source"])
                      for cell in notebook["cells"])


def test_resnet34_notebooks_run_only_three_matched_arms():
    for seed in (0, 1, 2):
        source = _resnet34_notebook_source(seed)
        assert f"SEED = {seed}" in source
        assert f"/kaggle/working/resnet34_seed{seed}_stall150_v1" in source
        assert "def load_checkout_module" in source
        assert '"experiments/kaggle_checkpoint_discovery.py"' in source
        assert "from experiments.kaggle_checkpoint_discovery" not in source
        assert '"--architecture", "resnet34"' in source
        assert '"--stall-patience", "150"' in source
        assert '"--post-fork-epochs", "150"' in source
        assert '"--retrigger-patience", "10"' in source
        assert '"matched_validation_best_to_150_epoch_window"' in source
        assert 'GPU0: E-driven O; GPU1: O-only' in source
        assert 'launch(0, "ours_e_driven_o")' in source
        assert 'launch(1, "o_projection_only")' in source
        assert 'launch(1, "bypass")' not in source
        assert 'for name in ("vanilla", "ours_e_driven_o", "o_projection_only")' in source


def test_resnet34_seed_notebooks_differ_only_by_seed_and_output_path():
    normalized = []
    for seed in (0, 1, 2):
        source = _resnet34_notebook_source(seed)
        source = source.replace(f"seed {seed}", "seed SEED")
        source = source.replace(f"seed{seed}", "seedSEED")
        source = source.replace(f"SEED = {seed}", "SEED = SEED")
        source = source.replace(f'"--seed", "{seed}"', '"--seed", "SEED"')
        normalized.append(source)
    assert normalized[0] == normalized[1] == normalized[2]


def test_resnet34_runner_arguments_preserve_resnet18_defaults():
    phase1 = Path("experiments/run_unified_vanilla_to_stall.py").read_text()
    fork = Path("experiments/run_plateau_fork.py").read_text()
    assert 'choices=("resnet18", "resnet34")' in phase1
    assert 'choices=("resnet18", "resnet34")' in fork
    assert 'default="resnet18"' in phase1
    assert 'default="resnet18"' in fork
    assert '"architecture": expected_architecture' in fork
    assert "plateau checkpoint architecture mismatch" in fork
