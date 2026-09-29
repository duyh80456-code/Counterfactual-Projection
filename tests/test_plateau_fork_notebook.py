import json
from pathlib import Path


def test_plateau_fork_notebook_runs_three_arms_from_one_hash():
    notebook = json.loads(Path(
        "notebooks/kaggle_plateau_fork_t4x2.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", []))
                       for cell in notebook["cells"])
    assert 'payload.get("kind") == "plateau_fork_checkpoint"' in source
    assert '(0, "ours_e_driven_o"), (1, "bypass")' in source
    assert '(0, "vanilla")' in source
    assert '"--post-fork-epochs", "60"' in source
    assert 'result["plateau_checkpoint_hash"] != PLATEAU_HASH' in source
    assert "budget_exhausted_before_contraction_accuracy_is_diagnostic" in source
    assert "time_spent_expanded_seconds" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"fork-cell-{index}", "exec")

