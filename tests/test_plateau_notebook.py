import json
from pathlib import Path


def source():
    notebook = json.loads(Path(
        "notebooks/kaggle_plateau_eo_t4x2.ipynb").read_text())
    return "\n".join("".join(cell.get("source", []))
                      for cell in notebook["cells"])


def test_plateau_notebook_forks_two_arms_from_vanilla_theta360():
    text = source()
    assert 'protocol.get("method") == "vanilla_continue"' in text
    assert 'int(payload.get("post_epoch", -1)) == 60' in text
    assert 'int(history[-1].get("epoch", -1)) == 360' in text
    assert '(0, "vanilla_continue", vanilla)' in text
    assert '(1, "plateau_e_driven_o", plateau)' in text
    assert '"final_epoch"] != 500' in text
    assert '"continuation_epochs"] != 140' in text
    assert '"--continuation-lr", "0.01"' in text
    assert '"--line-search-scales", "0.0125,0.025,0.05"' in text


def test_plateau_notebook_cells_parse():
    notebook = json.loads(Path(
        "notebooks/kaggle_plateau_eo_t4x2.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"plateau-cell-{index}", "exec")

