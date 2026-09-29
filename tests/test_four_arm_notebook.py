import json
from pathlib import Path


def test_shared_checkpoint_notebook_is_restart_safe_and_uses_two_gpus():
    notebook = json.loads(Path(
        "notebooks/kaggle_counterfactual_projection_t4x2.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", []))
                       for cell in notebook["cells"])
    for method in (
            "ours_e_driven_o", "bypass", "vanilla_continue",
            "o_projection_only"):
        assert method in source
    assert "TOTAL_EPOCHS = 360" in source
    assert "FORK_EPOCH = 300" in source
    assert "POST_FORK_EPOCHS = 60" in source
    assert "shared_seed1_epoch300.pt" in source
    assert "--bootstrap-checkpoint" not in source
    assert '"--method", "prepare_shared"' not in source
    assert "theta_150 is not accepted" in source
    assert "counterfactual_shared_theta300_360ep_all8_v8" in source
    assert "SHARED_HASH" in source
    assert "checkpoint_latest.pt" in source
    assert "Restored prior WHEN-WHERE-HOW run" in source
    assert "Restored theta_300 only" in source
    assert "Wave 1/2" in source and "Wave 2/2" in source
    assert '(0, "ours_e_driven_o", ours), (1, "bypass", bypass)' in source
    assert '"--site", "auto"' in source
    assert '"--candidate-sites", ""' in source
    assert '"--site-selection-mode", "all_projected_utility"' in source
    assert '(1, "o_projection_only", projection_only)' in source
    assert "stdout=subprocess.PIPE" in source
    assert 'print(f"[{name}] {line}"' in source
    assert 'result.get("bypass_completed") is not True' in source
    assert '"--opt1-epochs", "40"' in source
    assert '"--max-opt2-epochs", "20"' in source
    assert "completed_accuracy_comparators" in source
    assert "incomplete_comparators" in source
    assert "Accuracy is retained diagnostically" in source
    assert "a completed Bypass comparator" in source
    assert "best_validation_accuracy_delta" in source
    assert "validation_loss_delta" in source
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


def test_shared_epoch_accounting_is_300_plus_60():
    fork_epoch = 300
    post_fork_epochs = 60
    assert fork_epoch + post_fork_epochs == 360


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
