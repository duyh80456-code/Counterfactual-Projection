import json
from pathlib import Path


def test_plateau_fork_notebook_runs_three_jobs_and_reuses_vanilla_control():
    notebook = json.loads(Path(
        "notebooks/kaggle_plateau_fork_t4x2.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", []))
                       for cell in notebook["cells"])
    assert 'payload.get("kind") == "plateau_fork_checkpoint"' in source
    assert 'results["vanilla"] = VANILLA_CONTROL' in source
    assert 'ours_job = launch(0, "ours_e_driven_o")' in source
    assert 'bypass_job = launch(1, "bypass")' in source
    assert 'o_only_job = launch(1, "o_projection_only")' in source
    assert 'bypass_status = finish(bypass_job)' in source
    assert "pending" not in source
    assert '"--post-fork-epochs", "100"' in source
    assert '"--opt1-epochs", "70"' in source
    assert '"--max-opt2-epochs", "30"' in source
    assert '"--retrigger-patience", "10"' in source
    assert '"--significant-improvement", "0.001"' in source
    assert '"--gamma-increase-opt2-epoch", "15"' in source
    assert '"--gamma-post-increase-multiplier", "10.0"' in source
    assert '"--gamma-post-increase-multiplier", "2.0"' not in source
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
