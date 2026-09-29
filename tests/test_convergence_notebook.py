import json
from pathlib import Path


def test_convergence_notebook_is_review_and_resume_safe():
    notebook = json.loads(Path(
        "notebooks/kaggle_vanilla_to_plateau.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", []))
                       for cell in notebook["cells"])
    assert "MAX_EPOCH = 500" in source
    assert '"--required-plateau-windows", "2"' in source
    assert '"--plateau-window", "20"' in source
    assert '"--trigger-samples", "2000"' in source
    assert 'payload.get("kind") == "vanilla_convergence_progress"' in source
    assert "NO CONVERGENCE CLAIM" in source
    assert "plateau_checkpoint" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"convergence-cell-{index}", "exec")

