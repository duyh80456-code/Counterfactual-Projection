"""Train Vanilla to validation plateau; export its historical-best checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import time
from pathlib import Path

import torch

from experiments.deit_protocol import (DeitRecipe, checked_source, load_training_context,
    protocol, save_state, evaluate_without_rng)
from experiments.shared_protocol import atomic_json_save, seed_everything, train_epoch
from experiments.deit_logging import emit_event, emit_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", type=Path)
    for name, value in asdict(DeitRecipe()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    recipe = DeitRecipe(**{name: getattr(args, name) for name in asdict(DeitRecipe())})
    recipe.validate()
    source = None
    if args.resume:
        source, previous_recipe = checked_source(args.resume, {"deit_vanilla_latest"})
        if previous_recipe != recipe:
            raise ValueError("resume recipe must match exactly")
    seed_everything(recipe.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    context = load_training_context(args.data_root, recipe, device, source)
    model, optimizer, scheduler, loader, eval_loader, _, train_ids, val_ids, tuning_ids, trigger_ids = context
    declared = protocol(recipe, model)
    emit_event("deit_run_start", {"phase": "vanilla", "architecture": declared["architecture"],
        "seed": recipe.seed, "resume_epoch": source["epoch"] if source else 0,
        "stall_patience": recipe.stall_patience, "stall_start_epoch": recipe.stall_start_epoch,
        "schedule_epochs": recipe.schedule_epochs}, args.output)
    history = list(source["history"]) if source else []
    best_accuracy = source["report_best_accuracy"] if source else -1.
    best_loss = source["report_best_loss"] if source else float("inf")
    best_epoch = source["report_best_epoch"] if source else 0
    best_path = args.output / "checkpoint_best.pt"
    if source and not best_path.is_file():
        raise FileNotFoundError("Resume also requires checkpoint_best.pt in the output directory")
    if source:
        selected_best, selected_recipe = checked_source(best_path, {"deit_vanilla_best"})
        if (selected_recipe != recipe or selected_best["epoch"] != best_epoch or
                selected_best["report_best_accuracy"] != best_accuracy or
                selected_best["report_best_loss"] != best_loss):
            raise ValueError("resume latest and best checkpoints do not match")
    start_epoch = int(source["epoch"]) if source else 0
    detected = bool(source and source.get("plateau_detected"))
    detected_epoch = source.get("stall_detected_epoch", source["epoch"] if detected else None) if source else None
    reference_epochs = recipe.reference_epochs or recipe.stall_patience

    def save(path, epoch, kind, **extra):
        save_state(path, model=model, optimizer=optimizer, scheduler=scheduler, loader=loader,
            epoch=epoch, history=history, train_indices=train_ids, evaluation_indices=val_ids,
            source_tuning_indices=tuning_ids, run_protocol=declared, kind=kind,
            trigger_indices=trigger_ids,
            report_best_accuracy=best_accuracy, report_best_loss=best_loss,
            report_best_epoch=best_epoch, historical_best_accuracy=best_accuracy,
            historical_best_loss=best_loss, historical_best_epoch=best_epoch,
            epochs_since_best=epoch - best_epoch, **extra)

    if source is None:
        initial = evaluate_without_rng(model, eval_loader, device)
        best_accuracy = initial["accuracy"]
        best_loss = initial["loss"]
        history.append({"epoch": 0, "validation_accuracy": initial["accuracy"],
                        "validation_loss": initial["loss"]})
        save(best_path, 0, "deit_vanilla_best")
    for epoch in range(start_epoch + 1, recipe.max_epoch + 1):
        if detected and start_epoch >= best_epoch + reference_epochs:
            break
        epoch_started = time.perf_counter()
        training_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        train = train_epoch(model, loader, optimizer, device)
        validation = evaluate_without_rng(model, eval_loader, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train["loss"],
                        "train_accuracy": train["accuracy"],
                        "validation_accuracy": validation["accuracy"],
                        "validation_loss": validation["loss"],
                        "learning_rates": training_lrs,
                        "next_learning_rates": [float(group["lr"]) for group in optimizer.param_groups]})
        report_improved = not detected and (epoch == recipe.stall_start_epoch or validation["accuracy"] > best_accuracy)
        if report_improved:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            best_loss = validation["loss"]
        if not detected and best_epoch >= recipe.stall_start_epoch and epoch - best_epoch >= recipe.stall_patience:
            detected = True
            detected_epoch = epoch
        history[-1].update(phase="vanilla", architecture=declared["architecture"], seed=recipe.seed,
            report_best_improved=report_improved, report_best_accuracy=best_accuracy,
            report_best_loss=best_loss, report_best_epoch=best_epoch,
            report_stall_counter=epoch - best_epoch, epochs_since_best=epoch - best_epoch,
            stall_patience=recipe.stall_patience, plateau_detected=detected,
            stall_detected_epoch=detected_epoch,
            best_checkpoint_statistics={"metric": float(validation["accuracy"]),
                "best_metric": best_accuracy, "best_epoch": best_epoch,
                "improved": report_improved, "epochs_since_best": epoch - best_epoch,
                "stalled": detected})
        if report_improved:
            save(best_path, epoch, "deit_vanilla_best")
        save(args.output / "checkpoint_latest.pt", epoch, "deit_vanilla_latest",
             plateau_detected=detected, stall_detected_epoch=detected_epoch,
             vanilla_reference_complete=detected and epoch >= best_epoch + reference_epochs)
        emit_epoch("deit_vanilla", history[-1], device, epoch_started, args.output)
        if detected and epoch >= best_epoch + reference_epochs:
            break
    if source and source.get("plateau_detected"):
        detected = True
    if not detected or history[-1]["epoch"] < best_epoch + reference_epochs:
        atomic_json_save({"protocol": declared, "history": history,
                          "status": "plateau_detected_vanilla_incomplete" if detected else "plateau_not_detected"},
                         args.output / "result.json")
        raise RuntimeError("No complete plateau/Vanilla reference within max_epoch; no theta_P exported")
    best, selected_recipe = checked_source(best_path, {"deit_vanilla_best"})
    if (selected_recipe != recipe or best["epoch"] != best_epoch or
            best["report_best_accuracy"] != best_accuracy or best["report_best_loss"] != best_loss):
        raise ValueError("saved best does not match historical validation-best state")
    emit_event("deit_plateau_confirmed", {"fork_epoch": best_epoch, "historical_best_accuracy": best_accuracy,
        "historical_best_loss": best_loss, "stall_detected_epoch": detected_epoch,
        "epochs_since_best": detected_epoch - best_epoch, "vanilla_reference_epochs": reference_epochs}, args.output)
    best.update(kind="deit_plateau_fork", stall_detected_epoch=detected_epoch,
                stall_history=[row for row in history if best_epoch < row["epoch"] <= detected_epoch],
                vanilla_history=[row for row in history if row["epoch"] > best_epoch])
    from experiments.shared_protocol import atomic_torch_save
    atomic_torch_save(best, args.output / "plateau_checkpoint.pt")
    from experiments.deit_vanilla_reference import create_vanilla_reference
    create_vanilla_reference(args.output / "plateau_checkpoint.pt",
                            args.output / "checkpoint_latest.pt", args.output / "vanilla_reference.pt")
    atomic_json_save({"protocol": declared, "history": history,
                      "fork_epoch": best_epoch, "report_best_accuracy": best_accuracy,
                      "report_best_loss": best_loss, "report_best_epoch": best_epoch,
                      "historical_best_accuracy": best_accuracy, "historical_best_loss": best_loss,
                      "historical_best_epoch": best_epoch,
                      "stall_detected_epoch": detected_epoch, "status": "plateau_detected"},
                     args.output / "result.json")


if __name__ == "__main__":
    main()
