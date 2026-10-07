"""Train Vanilla to validation plateau; export its historical-best checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from experiments.deit_protocol import (DeitRecipe, checked_source, load_training_context,
    protocol, save_state, evaluate_without_rng)
from experiments.shared_protocol import atomic_json_save, seed_everything, train_epoch


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
    detected = False

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
        if source and source.get("plateau_detected"):
            detected = True
            break
        train = train_epoch(model, loader, optimizer, device)
        validation = evaluate_without_rng(model, eval_loader, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train["loss"],
                        "train_accuracy": train["accuracy"],
                        "validation_accuracy": validation["accuracy"],
                        "validation_loss": validation["loss"],
                        "learning_rates": [group["lr"] for group in optimizer.param_groups]})
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            best_loss = validation["loss"]
            save(best_path, epoch, "deit_vanilla_best")
        detected = epoch - best_epoch >= recipe.stall_patience
        save(args.output / "checkpoint_latest.pt", epoch, "deit_vanilla_latest",
             plateau_detected=detected)
        print(json.dumps(history[-1]), flush=True)
        if detected:
            break
    if source and source.get("plateau_detected"):
        detected = True
    if not detected:
        atomic_json_save({"protocol": declared, "history": history,
                          "status": "plateau_not_detected"}, args.output / "result.json")
        raise RuntimeError("No plateau within max_epoch; no theta_P exported")
    best, selected_recipe = checked_source(best_path, {"deit_vanilla_best"})
    if (selected_recipe != recipe or best["epoch"] != best_epoch or
            best["report_best_accuracy"] != best_accuracy or best["report_best_loss"] != best_loss):
        raise ValueError("saved best does not match historical validation-best state")
    best.update(kind="deit_plateau_fork", stall_detected_epoch=history[-1]["epoch"],
                stall_history=[row for row in history if row["epoch"] > best_epoch])
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
                      "stall_detected_epoch": history[-1]["epoch"], "status": "plateau_detected"},
                     args.output / "result.json")


if __name__ == "__main__":
    main()
