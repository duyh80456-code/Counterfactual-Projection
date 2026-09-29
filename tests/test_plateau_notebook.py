import json
from pathlib import Path


def source():
    notebook = json.loads(Path(
        "notebooks/kaggle_plateau_eo_t4x2.ipynb").read_text())
    return "\n".join("".join(cell.get("source", []))
                      for cell in notebook["cells"])


def test_plateau_notebook_forks_two_arms_from_shared_theta300():
    text = source()
    assert 'payload.get("kind") == "shared_fork_checkpoint"' in text
    assert 'int(payload.get("epoch", -1)) == 300' in text
    assert 'int(history[-1].get("epoch", -1)) == 300' in text
    assert '(0, "vanilla_continue", vanilla)' in text
    assert '(1, "plateau_e_driven_o", plateau)' in text
    assert 'result["final_epoch"] != EXPECTED_FINAL_EPOCH' in text
    assert 'result["continuation_epochs"] != EXPECTED_CONTINUATION_EPOCHS' in text
    assert 'SCHEDULER_STATE["T_max"]' in text
    assert 'SCHEDULER_STATE["last_epoch"]' in text
    assert '"--trigger-samples", "2000"' in text
    assert '"--minimum-sgd-epochs", "15"' in text
    assert '"--plateau-accuracy-min-gain", "0.001"' in text
    assert "--continuation-lr" not in text
    assert 'result["scheduler_state_restored"] is not True' in text
    assert 'result["scheduler_restarted"] is not False' in text
    assert '"--line-search-scales", "0.0125,0.025,0.05"' in text


def test_plateau_notebook_cells_parse():
    notebook = json.loads(Path(
        "notebooks/kaggle_plateau_eo_t4x2.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"plateau-cell-{index}", "exec")
