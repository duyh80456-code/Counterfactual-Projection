import json
from pathlib import Path


def test_one_file_notebook_connects_meaningful_stall_to_three_methods():
    notebook = json.loads(Path(
        "notebooks/kaggle_plateau_end_to_end_t4x2.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", []))
                       for cell in notebook["cells"])

    assert "experiments.run_vanilla_to_plateau" in source
    assert "phase1_result[\"plateau_found\"]" in source
    assert "checkpoint_best.pt" in source
    assert "no-new-best stall" in source
    assert "Phase 2 was intentionally skipped" in source
    assert "plateau_fork_checkpoint" in source
    assert "hashlib.sha256(PLATEAU_CHECKPOINT.read_bytes())" in source
    assert "experiments.run_plateau_fork" in source
    assert '"ours_e_driven_o"' in source
    assert '"o_projection_only"' in source
    assert '"bypass"' in source
    assert '"vanilla"' in source
    assert 'results["vanilla"] = VANILLA_CONTROL' in source
    assert '"theta_P_validation_accuracy"' in source
    assert '"meaningful_best_validation_accuracy"' in source
    assert 'VANILLA_CONTROL = dict(PLATEAU_PAYLOAD["vanilla_control"])' in source
    assert 'VANILLA_CONTROL["post_fork_epochs"] != 100' in source
    assert '"--post-fork-epochs", "100"' in source
    assert 'ours_job = launch(0, "ours_e_driven_o")' in source
    assert 'bypass_job = launch(1, "bypass")' in source
    assert 'o_only_job = launch(1, "o_projection_only")' in source
    assert "pending.pop(0)" not in source
    assert 'result["theta_best_hash"] != PLATEAU_HASH' in source
    assert '"--best-min-gain", "0.0"' in source
    assert '"--significant-min-gain", "0.001"' in source
    assert '"--gamma-post-increase-multiplier", "2.0"' in source
    assert '"--gamma-post-increase-multiplier", "10.0"' not in source
    assert "torch.cuda.device_count() != 2" in source
    assert "import threading" in source
    assert source.index("import threading") < source.index("threading.Thread")


def test_one_file_notebook_does_not_scan_input_for_plateau_checkpoint():
    notebook = json.loads(Path(
        "notebooks/kaggle_plateau_end_to_end_t4x2.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", []))
                       for cell in notebook["cells"])

    assert 'rglob("plateau_checkpoint.pt")' not in source
    assert 'rglob("shared_seed1_epoch300.pt")' in source
