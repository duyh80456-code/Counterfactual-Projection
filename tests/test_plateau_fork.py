from pathlib import Path

import pytest

from experiments.run_plateau_fork import (
    next_controller_stall_counter, validation_state_updates)


@pytest.mark.parametrize(
    ("accuracy", "loss", "expected_report", "expected_anchor", "reason",
     "expected_stall"),
    [
        (0.76, 1.5, True, True, "accuracy_increase", 0),
        (0.75, 1.39115, False, True, "same_accuracy_lower_loss", 0),
        (0.75, 1.6, False, False, None, 6),
        (0.74, 1.0, False, False, None, 6),
    ],
    ids=("higher-accuracy", "same-accuracy-lower-loss",
         "same-accuracy-higher-loss", "lower-accuracy"))
def test_report_anchor_and_stall_updates_cover_all_four_cases(
        accuracy, loss, expected_report, expected_anchor, reason,
        expected_stall):
    report_improved, anchor_improved, anchor_reason = validation_state_updates(
        accuracy=accuracy, loss=loss,
        report_best_accuracy=0.75,
        anchor_accuracy=0.75, anchor_loss=1.41308)

    assert report_improved is expected_report
    assert anchor_improved is expected_anchor
    assert anchor_reason == reason
    assert next_controller_stall_counter(5, anchor_improved) == expected_stall


def test_epoch_302_tie_resets_stall_without_changing_report_best():
    report_improved, anchor_improved, reason = validation_state_updates(
        accuracy=73.1333, loss=1.39115,
        report_best_accuracy=73.1333,
        anchor_accuracy=73.1333, anchor_loss=1.41308)

    assert (report_improved, anchor_improved, reason) == (
        False, True, "same_accuracy_lower_loss")
    assert next_controller_stall_counter(14, anchor_improved) == 0


def test_fifteen_epochs_without_anchor_improvement_reaches_retrigger():
    counter = 0
    for _ in range(15):
        counter = next_controller_stall_counter(counter, anchor_improved=False)

    assert counter == 15
    assert counter >= 15


def test_plateau_fork_has_recurrent_e_and_single_shot_o_control():
    source = Path("experiments/run_plateau_fork.py").read_text()
    assert 'METHODS = ("vanilla", "bypass", "ours_e_driven_o", "o_projection_only")' in source
    assert '"o_projection_only")' in source
    assert 'source.get("kind") != "plateau_fork_checkpoint"' in source
    assert "optimizer.load_state_dict(source[\"optimizer\"])" in source
    assert 'scheduler_from_state(optimizer, source["scheduler"])' in source
    assert "scheduler.sync_optimizer_groups()" in source
    assert "train_indices = list(source[\"train_indices\"])" in source
    assert '"mode": "recurrent_best_rollback"' in source
    assert 'choices=("all_functional_gain",)' in source
    assert 'parser.add_argument(\n        "--o-only-site"' in source
    assert "o_only_site = args.o_only_site or args.site" in source
    assert '"site_selection_mode": args.site_selection_mode' in source
    assert '"mode": "single_initial_intervention"' in source
    assert '"patience": args.retrigger_patience' in source
    assert "run_intervention(" in source
    assert "run_o_only_intervention(" in source
    assert "supervised_functional_descent_direction" in source
    assert '"raw_validation_stall"' in source
    assert 'args.method == "ours_e_driven_o" and' in source
    assert ('args.method in {"ours_e_driven_o", "o_projection_only"} and' not
            in source)
    assert "live_rng = rng_state()" in source
    assert "live_loader_state = train_loader.generator.get_state().clone()" in source
    assert 'best_state["model"]' in source
    assert 'optimizer.load_state_dict(best_state["optimizer"])' in source
    assert 'scheduler.load_state_dict(best_state["scheduler"])' in source
    assert 'restore_rng(live_rng)' in source
    assert 'train_loader.generator.set_state(live_loader_state.cpu())' in source
    assert "perform_intervention(" in source
    assert '"intervention_count"' in source
    assert '"rollback_count"' in source
    assert "pre_probe_rng = rng_state()" in source
    assert "restore_rng(pre_probe_rng)" in source
    assert "embed_relaxed_bypass" in source
    assert "transition_from_opt2_" in source
    assert '"time_spent_expanded_seconds"' in source
    assert '"peak_train_params"' in source
    assert '"epochs_to_best"' in source
    assert 'best_checkpoint = output / "checkpoint_best.pt"' in source
    assert '"plateau_fork_arm_best"' in source
    assert '"--post-fork-epochs", type=int, default=150' in source
    assert '"--retrigger-patience", type=int, default=10' in source
    assert '"--opt1-epochs", type=int, default=70' in source
    assert '"--max-opt2-epochs", type=int, default=30' in source
    assert '"--gamma-increase-opt2-epoch", type=int, default=15' in source
    assert '"--gamma-post-increase-multiplier", type=float, default=2.0' in source
    assert '"plateau_checkpoint_hash": fork_hash' in source
    assert '"train_indices": train_indices' in source
    assert '"trigger_indices": trigger_indices' in source
    assert '"evaluation_indices": evaluation_indices' in source
    assert '"theta_best_hash": fork_hash' in source
    assert '"opt1_epochs": opt1_done if args.method == "bypass" else None' in source
    assert 'validation_state_updates(' in source
    assert 'loss < anchor_loss' in source
    assert '"controller_anchor_rule": "accuracy, then lower loss on exact accuracy tie"' in source
    assert '"report_stall_counter": report_stall_counter' in source
    assert '"controller_stall_counter": controller_stall_counter' in source
    assert '"report_best_loss": report_best_loss' in source
    assert '"best_validation_loss": report_best_loss' in source
    assert 'controller_stall_counter = next_controller_stall_counter(' in source
    assert 'significant_improved = (' not in source
    assert 'row["controller_anchor_reason"] = controller_anchor_reason' in source
    assert 'row["exact_best_improved"]' not in source
    assert 'row["stall_counter"]' not in source
    assert 'scheduler.step(trigger["accuracy"])' not in source
    assert 'phase = "incomplete"' in source
    assert '"budget_exhausted_before_contraction"' in source
    assert '"compact_best_validation_accuracy"' in source
    assert "max(compact_rows," in source
    assert '"post_fork_epoch": 0' not in source
    assert 'accuracy > anchor_accuracy' in source
    assert "epochs_since_best" not in source
