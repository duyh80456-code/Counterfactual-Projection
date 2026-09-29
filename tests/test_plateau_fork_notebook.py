import json
from pathlib import Path


def test_plateau_fork_notebook_runs_three_jobs_and_reuses_vanilla_control():
    notebook = json.loads(Path(
        "notebooks/kaggle_plateau_fork_t4x2.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", []))
                       for cell in notebook["cells"])
    assert 'payload.get("kind") == "plateau_fork_checkpoint"' in source
    assert 'pending = ["ours_e_driven_o", "bypass", "o_projection_only"]' in source
    assert 'results["vanilla"] = VANILLA_CONTROL' in source
    assert "running = {gpu: launch(gpu, pending.pop(0)) for gpu in (0, 1)}" in source
    assert "process.poll()" in source
    assert "running[gpu] = launch(gpu, pending.pop(0))" in source
    assert '"--post-fork-epochs", "100"' in source
    assert '"--max-opt2-epochs", "60"' in source
    assert '"--retrigger-patience", "10"' in source
    assert 'VANILLA_CONTROL["post_fork_epochs"] != 100' in source
    assert 'Path(result["best_checkpoint"]).is_file()' in source
    assert "best_acc" in source
    assert "final_acc" in source
    assert 'result["plateau_checkpoint_hash"] != PLATEAU_HASH' in source
    assert "budget_exhausted_before_contraction_accuracy_is_diagnostic" in source
    assert "time_spent_expanded_seconds" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"fork-cell-{index}", "exec")
