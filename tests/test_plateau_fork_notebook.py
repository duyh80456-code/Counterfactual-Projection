import json
from pathlib import Path


def test_plateau_fork_notebook_runs_three_methods_and_reuses_matched_vanilla():
    notebook = json.loads(Path(
        "notebooks/kaggle_plateau_fork_t4x2.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", []))
                       for cell in notebook["cells"])
    assert "discover_checkpoints(" in source
    assert "sys.path.insert(0, str(REPO))" in source
    assert (source.index("sys.path.insert(0, str(REPO))") <
            source.index("from experiments.kaggle_checkpoint_discovery"))
    assert '"/kaggle/input", OUTPUT, kind="plateau_fork_checkpoint"' in source
    assert 'rglob("plateau_checkpoint.pt")' not in source
    assert "followlinks=True" in Path(
        "experiments/kaggle_checkpoint_discovery.py").read_text()
    assert '"matched_significant_best_to_stall_window"' in source
    assert 'results["vanilla"] = VANILLA_CONTROL' in source
    assert 'ours_job = launch(0, "ours_e_driven_o")' in source
    assert 'bypass_job = launch(1, "bypass")' in source
    assert 'o_only_job = launch(1, "o_projection_only")' in source
    assert "pending" not in source
    assert '"--post-fork-epochs", "100"' in source
    assert '"--opt1-epochs", "70"' in source
    assert '"--max-opt2-epochs", "30"' in source
    assert '"--retrigger-patience", "10"' in source
    assert '"--significant-improvement", "0.001"' in source
    assert '"--gamma-increase-opt2-epoch", "15"' in source
    assert '"--gamma-post-increase-multiplier", "2.0"' in source
    assert '"--gamma-post-increase-multiplier", "10.0"' not in source
    assert 'VANILLA_CONTROL["post_fork_epochs"] != 100' in source
    assert 'Path(result["best_checkpoint"]).is_file()' in source
    assert "best_acc" in source
    assert "final_acc" in source
    assert 'result["plateau_checkpoint_hash"] != PLATEAU_HASH' in source
    assert 'result["theta_best_hash"] != PLATEAU_HASH' in source
    assert "budget_exhausted_before_contraction_accuracy_is_diagnostic" in source
    assert "time_spent_expanded_seconds" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"fork-cell-{index}", "exec")
