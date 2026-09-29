import json
from pathlib import Path


NOTEBOOK = Path("notebooks/kaggle_adaptive_e_driven_o.ipynb")


def notebook_source():
    notebook = json.loads(NOTEBOOK.read_text())
    return notebook, "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])


def test_adaptive_notebook_only_runs_main_arm_from_theta300():
    _notebook, source = notebook_source()
    assert '"--method", "ours_e_driven_o"' in source
    assert '"--site", "auto"' in source
    assert '"--candidate-sites", ""' in source
    assert '"--site-selection-mode", "all_projected_utility"' in source
    assert '"--selection-cg-iterations", "25"' in source
    assert '"--selection-min-projectability"' not in source
    assert '"--rank", "4"' in source
    assert "shared_seed1_epoch300.json" in source
    assert "shared_seed1_epoch300.pt" in source
    assert "materialize_checkpoint" in source
    assert "repacked_from_kaggle_directory" in source
    assert "payload.get(\"protocol\") == manifest.get(\"protocol\")" in source
    assert '"--shared-checkpoint-hash", SHARED_HASH' in source
    assert 'result["post_fork_epochs"] != 60' in source
    assert "epochs 301-360" in source
    assert '"--method", "prepare_shared"' not in source
    assert '"--method", "vanilla_continue"' not in source
    assert '"--method", "o_projection_only"' not in source
    assert '"--method", "bypass"' not in source


def test_adaptive_notebook_preserves_resume_and_selection_outputs():
    _notebook, source = notebook_source()
    assert "checkpoint_latest.pt" in source
    assert "site_selection_history" in source
    assert "site_counts" in source
    assert "dominant_site_fraction" in source
    assert "when_gate_pass_rate" in source
    assert "run.log" in source
    assert "stdout=subprocess.PIPE" in source


def test_adaptive_notebook_code_cells_parse_as_python():
    notebook, _source = notebook_source()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"adaptive-cell-{index}", "exec")
