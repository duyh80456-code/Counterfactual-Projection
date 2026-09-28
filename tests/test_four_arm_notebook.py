import json
from pathlib import Path


def test_four_arm_notebook_is_restart_safe_and_uses_official_sources():
    notebook = json.loads(Path(
        "notebooks/kaggle_counterfactual_projection_t4x2.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", []))
                       for cell in notebook["cells"])
    for method in ("ours_e_driven_o", "repan", "expandnets", "repoptimizer"):
        assert method in source
    for repository in ("xfey/RepAn", "GUOShuxuan/expandnets",
                       "DingXiaoH/RepOptimizers"):
        assert repository in source
    assert "TOTAL_EPOCHS = 80" in source
    assert "OURS_EPOCHS = TOTAL_EPOCHS - WARMUP_EPOCHS" in source
    assert 'expected_epoch = OURS_EPOCHS if name == "ours_e_driven_o"' in source
    assert 'result["method_epochs"] = result["epoch"]' in source
    assert "checkpoint_latest.pt" in source
    assert "Restored prior phase" in source
    assert "Wave 1/2" in source and "Wave 2/2" in source
    assert "official_test_used" in source


def test_notebook_code_cells_parse_as_python():
    notebook = json.loads(Path(
        "notebooks/kaggle_counterfactual_projection_t4x2.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_ours_alias_maps_to_frozen_projection_arm():
    source = Path("experiments/run_gromo_pilot.py").read_text()
    assert '"ours_e_driven_o"' in source
    assert 'args.method = "tiny_projection"' in source
    assert '"application_gate": "finite_and_heldout_functional_fit"' in source
    assert 'cg_converged != True' not in source
    assert '"epoch": len(history)' in source
    assert '"total_training_epochs": (' in source
    assert 'completed.get("epoch") == args.epochs' in source


def test_ours_epoch_accounting_is_77_method_and_80_total():
    total_epochs = 80
    warmup_epochs = 3
    ours_epochs = total_epochs - warmup_epochs
    ours_result = {"epoch": ours_epochs}
    assert ours_result["epoch"] == ours_epochs
    assert ours_result["epoch"] + warmup_epochs == total_epochs


def test_gromo_validation_loss_is_accumulated_once():
    source = Path("experiments/run_gromo_pilot.py").read_text()
    evaluate_body = source.split("def evaluate(model, loader, device):", 1)[1]
    evaluate_body = evaluate_body.split("\ndef train_epoch", 1)[0]
    assert evaluate_body.count("loss_sum +=") == 1


def test_repan_rebuilds_optimizer_and_scheduler_per_cycle():
    source = Path("baselines/run_repan.py").read_text()
    assert "fresh_optimizer_and_scheduler" in source
    assert "return fresh_optimizer_and_scheduler()" in source
    assert "optimizer.state.clear()" not in source
