"""Matched Vanilla / O-only / one-shot E-to-O AdamW continuations from theta_P."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from adapters.deit_cp_adapter import CPConfig, one_shot_intervention
from experiments.deit_protocol import (checked_source, load_training_context,
    materialize_probe_batches, save_state, evaluate_without_rng, historical_best)
from experiments.shared_protocol import atomic_json_save, seed_everything, sha256_file, train_epoch

METHODS = ("vanilla_continue", "o_projection_only", "ours_e_driven_o")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--plateau-checkpoint", type=Path, required=True)
    parser.add_argument("--plateau-checkpoint-hash", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--post-fork-epochs", "--horizon", dest="post_fork_epochs", type=int, default=150)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    for name, default in asdict(CPConfig()).items():
        if name == "scales":
            parser.add_argument("--scales", default=",".join(map(str, default)))
        else:
            parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args()
    config = CPConfig(**{name: getattr(args, name) for name in asdict(CPConfig()) if name != "scales"},
                      scales=tuple(float(value) for value in args.scales.split(",")))
    if args.post_fork_epochs < 1:
        parser.error("post-fork epochs must be positive")
    fork_hash = sha256_file(args.plateau_checkpoint)
    if fork_hash != args.plateau_checkpoint_hash:
        raise ValueError("DeiT fork SHA256 mismatch")
    fork, recipe = checked_source(args.plateau_checkpoint, {"deit_plateau_fork"})
    historical_accuracy, historical_loss, historical_epoch = historical_best(fork)
    if int(fork["epoch"]) + args.post_fork_epochs > recipe.schedule_epochs:
        raise ValueError("fork + horizon exceeds the declared global scheduler budget")
    identity = {"fork_hash": fork_hash, "method": args.method,
                "post_fork_epochs": args.post_fork_epochs, "cp_config": json.loads(json.dumps(asdict(config)))}
    saved = None
    if args.resume:
        saved, _ = checked_source(args.resume, {"deit_fork_arm_latest"})
        if saved.get("run_identity") != identity or saved["protocol"] != fork["protocol"]:
            raise ValueError("DeiT resume fork/method/recipe/CP config mismatch")
        completed = int(saved["completed_epochs"])
        if (not 0 <= completed <= args.post_fork_epochs or saved["epoch"] != fork["epoch"] + completed or
                len(saved["interventions"]) != (0 if args.method == "vanilla_continue" else 1) or
                len(saved["history"]) != completed + 1 or
                saved["validation_immediately_after_projection"] is None):
            raise ValueError("incomplete or inconsistent DeiT arm resume state")
    seed_everything(recipe.seed)
    device = torch.device(args.device)
    context = load_training_context(args.data_root, recipe, device, saved or fork)
    model, optimizer, scheduler, loader, eval_loader, eval_set, train_ids, val_ids, tuning_ids, trigger_ids = context
    args.output.mkdir(parents=True, exist_ok=True)
    history = list(saved["history"]) if saved else []
    interventions = list(saved["interventions"]) if saved else []
    before = saved["validation_before"] if saved else evaluate_without_rng(model, eval_loader, device)
    immediate = saved["validation_immediately_after_projection"] if saved else None
    start = int(saved["completed_epochs"]) if saved else 0
    best_accuracy = float(saved["report_best_accuracy"]) if saved else float("-inf")
    best_loss = float(saved["report_best_loss"]) if saved else float("inf")
    best_epoch = int(saved["report_best_epoch"]) if saved else int(fork["epoch"])

    def save(epoch, completed):
        save_state(args.output / "checkpoint_latest.pt", model=model, optimizer=optimizer,
            scheduler=scheduler, loader=loader, epoch=epoch, history=history,
            train_indices=train_ids, evaluation_indices=val_ids, source_tuning_indices=tuning_ids,
            trigger_indices=trigger_ids,
            run_protocol=fork["protocol"], kind="deit_fork_arm_latest", run_identity=identity,
            method=args.method, completed_epochs=completed, interventions=interventions,
            validation_before=before, validation_immediately_after_projection=immediate,
            historical_best_accuracy=historical_accuracy, historical_best_loss=historical_loss,
            historical_best_epoch=historical_epoch,
            report_best_accuracy=best_accuracy, report_best_loss=best_loss, report_best_epoch=best_epoch)

    if saved is None:
        if args.method != "vanilla_continue":
            batches, indices = materialize_probe_batches(eval_set, train_ids, recipe, config, device)
            record = one_shot_intervention(model, optimizer, config=config, method=args.method, **batches)
            immediate = evaluate_without_rng(model, eval_loader, device)
            record.update(epoch=int(fork["epoch"]), probe_index=0,
                          probe_indices=indices, validation_before=before,
                          validation_immediately_after_projection=immediate,
                          validation_1_to_5_epochs_after=[])
            interventions.append(record)
            print(json.dumps({"intervention": record}), flush=True)
        else:
            immediate = dict(before)
        history.append({"epoch": int(fork["epoch"]), "post_fork_epoch": 0,
                        "validation_accuracy": immediate["accuracy"],
                        "validation_loss": immediate["loss"],
                        "metric_timing": ("before_SGD_no_projection" if args.method == "vanilla_continue"
                                          else "after_initial_projection_before_SGD")})
        save(int(fork["epoch"]), 0)  # Resume does not reapply the one-shot jump.
    for offset in range(start + 1, args.post_fork_epochs + 1):
        epoch = int(fork["epoch"]) + offset
        train = train_epoch(model, loader, optimizer, device)
        validation = evaluate_without_rng(model, eval_loader, device)
        scheduler.step()  # Same inherited scheduler; no rebase after projection.
        history.append({"epoch": epoch, "post_fork_epoch": offset,
                        "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                        "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
                        "learning_rates": [group["lr"] for group in optimizer.param_groups]})
        if interventions:
            interventions[0]["validation_1_to_5_epochs_after"] = [
                row for row in history if 1 <= row["post_fork_epoch"] <= 5]
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_loss, best_epoch = validation["accuracy"], validation["loss"], epoch
        save(epoch, offset)
        print(json.dumps(history[-1]), flush=True)
    first_five = [row for row in history if 1 <= row["post_fork_epoch"] <= 5]
    atomic_json_save({"method": args.method, "protocol": fork["protocol"],
        "run_id": str(args.output.resolve()), "theta_best_hash": fork_hash,
        "fork_epoch": int(fork["epoch"]), "fork_validation_accuracy": before["accuracy"],
        "historical_best_accuracy": historical_accuracy, "historical_best_loss": historical_loss,
        "historical_best_epoch": historical_epoch,
        "validation_before": before, "validation_immediately_after_projection": immediate,
        "validation_1_to_5_epochs_after": first_five, "interventions": interventions,
        "history": history, "report_best_accuracy": best_accuracy,
        "report_best_loss": best_loss, "report_best_epoch": best_epoch,
        "delta_vs_historical_best": best_accuracy - historical_accuracy,
        "scientific_escape": best_accuracy > historical_accuracy,
        "report_best_scope": "epochs1_to_K_epoch0_reported_separately",
        "final_validation_accuracy": history[-1]["validation_accuracy"],
        "final_validation_loss": history[-1]["validation_loss"],
        "post_fork_epochs": args.post_fork_epochs,
        "horizon": args.post_fork_epochs, "run_identity": identity,
        "controller": "one_shot_no_rollback_no_retrigger"}, args.output / "result.json")


if __name__ == "__main__":
    main()
