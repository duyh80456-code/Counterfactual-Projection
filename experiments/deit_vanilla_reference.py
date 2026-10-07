"""Reuse the uninterrupted Phase 1 plateau window as the Vanilla comparison."""
from __future__ import annotations

from pathlib import Path

from experiments.deit_protocol import checked_source, historical_best
from experiments.shared_protocol import atomic_json_save, atomic_torch_save, sha256_file

TRAJECTORY_SOURCE = "phase1_plateau_window"
REFERENCE_VERSION = 1


def create_vanilla_reference(fork_path, terminal_path, output_path):
    """Export observed metrics and the actual terminal state; never train/evaluate."""
    fork, recipe = checked_source(fork_path, {"deit_plateau_fork"})
    terminal, terminal_recipe = checked_source(terminal_path, {"deit_vanilla_latest"})
    accuracy, loss, epoch = historical_best(fork)
    horizon = recipe.reference_epochs or recipe.stall_patience
    if (terminal_recipe != recipe or terminal["protocol"] != fork["protocol"] or
            not terminal.get("plateau_detected") or terminal["epoch"] != epoch + horizon or
            terminal.get("historical_best_epoch") != epoch or
            terminal.get("historical_best_accuracy") != accuracy or
            terminal.get("historical_best_loss") != loss or
            terminal["history"][:len(fork["history"])] != fork["history"] or
            terminal["history"][len(fork["history"]):] != fork.get("vanilla_history", fork.get("stall_history"))):
        raise ValueError("Phase 1 terminal checkpoint does not match the selected fork trajectory")
    for key in ("train_indices", "evaluation_indices", "source_tuning_indices", "trigger_indices"):
        if terminal[key] != fork[key]:
            raise ValueError("Phase 1 terminal checkpoint and fork data splits differ")
    rows = fork.get("vanilla_history", fork["stall_history"])
    if len(rows) != horizon or [row["epoch"] for row in rows] != list(range(epoch + 1, epoch + horizon + 1)):
        raise ValueError("Phase 1 plateau window is not a complete contiguous horizon")
    before = {"accuracy": accuracy, "loss": loss}
    history = [{"epoch": epoch, "post_fork_epoch": 0,
                "validation_accuracy": accuracy, "validation_loss": loss,
                "metric_timing": "before_SGD_no_projection"}]
    history += [{**row, "post_fork_epoch": row["epoch"] - epoch} for row in rows]
    # max is stable on accuracy ties, preserving loss at the first strict best.
    best = max(history[1:], key=lambda row: row["validation_accuracy"])
    reference = {**terminal, "kind": "deit_vanilla_reference",
        "vanilla_reference_version": REFERENCE_VERSION, "trajectory_source": TRAJECTORY_SOURCE,
        "source_phase1_checkpoint_hash": sha256_file(terminal_path),
        "theta_best_hash": sha256_file(fork_path), "history": history,
        "method": "vanilla_continue", "completed_epochs": horizon, "post_fork_epochs": horizon,
        "interventions": [], "validation_before": before,
        "validation_immediately_after_projection": dict(before),
        "report_best_accuracy": best["validation_accuracy"],
        "report_best_loss": best["validation_loss"], "report_best_epoch": best["epoch"]}
    atomic_torch_save(reference, Path(output_path))
    return reference


def export_reused_vanilla_arm(fork, reference, output, identity):
    """Write a matched arm from its reference, retaining every terminal state tensor."""
    accuracy, loss, epoch = historical_best(fork)
    horizon = identity["post_fork_epochs"]
    history = reference["history"]
    expected_rows = [{**row, "post_fork_epoch": row["epoch"] - epoch} for row in fork.get("vanilla_history", fork["stall_history"])]
    if (reference.get("vanilla_reference_version") != REFERENCE_VERSION or
            reference.get("trajectory_source") != TRAJECTORY_SOURCE or
            reference.get("theta_best_hash") != identity["fork_hash"] or
            reference["protocol"] != fork["protocol"] or
            reference.get("method") != "vanilla_continue" or reference.get("interventions") != [] or
            reference.get("completed_epochs") != horizon or horizon != (fork["protocol"]["recipe"].get("reference_epochs") or fork["protocol"]["recipe"]["stall_patience"]) or
            reference["epoch"] != epoch + horizon or len(history) != horizon + 1 or
            history[1:] != expected_rows or
            not history or history[0]["epoch"] != epoch or history[0].get("post_fork_epoch") != 0 or
            history[0]["validation_accuracy"] != accuracy or history[0]["validation_loss"] != loss or
            [row["epoch"] for row in history[1:]] != list(range(epoch + 1, epoch + horizon + 1))):
        raise ValueError("Vanilla reference fork/horizon/protocol/history mismatch; Vanilla is not retrained")
    before = {"accuracy": accuracy, "loss": loss}
    for key in ("historical_best_accuracy", "historical_best_loss", "historical_best_epoch",
                "train_indices", "evaluation_indices", "source_tuning_indices", "trigger_indices"):
        if reference[key] != fork[key]:
            raise ValueError("Vanilla reference historical best or data split mismatch")
    if (reference["validation_before"] != before or
            reference["validation_immediately_after_projection"] != before):
        raise ValueError("Vanilla reference initial metric does not match historical best")
    best = max(history[1:], key=lambda row: row["validation_accuracy"])
    if any(row["validation_accuracy"] > accuracy for row in history[1:1 + fork["protocol"]["recipe"]["stall_patience"]]):
        raise ValueError("Vanilla confirmation window exceeds its declared historical best")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {"report_best_accuracy": best["validation_accuracy"],
               "report_best_loss": best["validation_loss"], "report_best_epoch": best["epoch"]}
    state = {**reference, **metrics, "kind": "deit_fork_arm_latest", "run_identity": identity}
    atomic_torch_save(state, output / "checkpoint_latest.pt")
    result = {"method": "vanilla_continue", "protocol": fork["protocol"],
        "run_id": str(output.resolve()), "theta_best_hash": identity["fork_hash"],
        "fork_epoch": epoch, "fork_validation_accuracy": accuracy,
        "historical_best_accuracy": accuracy, "historical_best_loss": loss, "historical_best_epoch": epoch,
        "validation_before": before, "validation_immediately_after_projection": dict(before),
        "validation_1_to_5_epochs_after": history[1:6], "interventions": [], "history": history,
        **metrics, "delta_vs_historical_best": metrics["report_best_accuracy"] - accuracy,
        "scientific_escape": metrics["report_best_accuracy"] > accuracy, "report_best_scope": "epochs1_to_K_epoch0_reported_separately",
        "final_validation_accuracy": history[-1]["validation_accuracy"],
        "final_validation_loss": history[-1]["validation_loss"],
        "post_fork_epochs": horizon, "horizon": horizon, "run_identity": identity,
        "controller": "observed_vanilla_no_rollback_no_retrigger", "trajectory_source": TRAJECTORY_SOURCE,
        "source_phase1_checkpoint_hash": reference["source_phase1_checkpoint_hash"],
        "additional_training_epochs": 0}
    atomic_json_save(result, output / "result.json")
    return result
