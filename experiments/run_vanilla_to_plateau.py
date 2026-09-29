"""Train Vanilla from shared theta300 until robust convergence plateau."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.plateau_protocol import (
    BestCheckpointStallDetector, ConstantCheckpointScheduler)
from experiments.shared_protocol import (
    atomic_json_save, atomic_torch_save, build_cifar_gromo_resnet18,
    build_optimizer_scheduler, datasets_and_indices, evaluate,
    make_eval_loader, make_train_loader, restore_rng, rng_state,
    seed_everything, sha256_file, train_epoch)


START_EPOCH = 300


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fork-checkpoint", required=True)
    parser.add_argument("--fork-checkpoint-hash", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-epoch", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--trigger-samples", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--no-new-best-patience", type=int, default=100)
    parser.add_argument("--significant-min-gain", type=float, default=1e-3)
    return parser.parse_args()


def save(path, payload):
    atomic_torch_save({"format_version": 1, **payload}, path)


def finalize_best_stall(best_path, plateau_path, detector, history,
                        protocol, deploy_params):
    """Fork exact theta_best and attach its already-observed Vanilla control."""
    if not best_path.is_file():
        raise RuntimeError("stall detected but checkpoint_best.pt is missing")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    best_epoch = int(best["epoch"])
    stall_start = int(detector.last_meaningful_improvement_epoch) + 1
    stall_end = int(detector.observations[-1]["epoch"])
    control = [dict(row) for row in history
               if stall_start <= int(row["epoch"]) <= stall_end]
    if len(control) != detector.patience:
        raise RuntimeError(
            "significant-best patience must provide 100 Vanilla control epochs")
    fork_row = next(row for row in history if int(row["epoch"]) == best_epoch)
    candidates = [fork_row, *control]
    best_accuracy = max(row["validation_accuracy"] for row in candidates)
    best_row = next(row for row in candidates
                    if row["validation_accuracy"] == best_accuracy)
    best["kind"] = "plateau_fork_checkpoint"
    best["stall_evidence"] = detector.state_dict()
    best["stall_detected_epoch"] = detector.observations[-1]["epoch"]
    best["vanilla_control_history"] = control
    best["vanilla_control"] = {
        "method": "vanilla_reused_from_phase1",
        "fork_epoch": best_epoch,
        "stall_window_start_epoch": stall_start,
        "stall_detected_epoch": best["stall_detected_epoch"],
        "post_fork_epochs": len(control),
        "fork_validation_accuracy": fork_row["validation_accuracy"],
        "fork_validation_loss": fork_row["validation_loss"],
        "final_validation_accuracy": control[-1]["validation_accuracy"],
        "best_validation_accuracy": best_accuracy,
        "final_validation_loss": control[-1]["validation_loss"],
        "best_validation_loss": min(
            row["validation_loss"] for row in candidates),
        "validation_accuracy_delta": (
            control[-1]["validation_accuracy"] -
            fork_row["validation_accuracy"]),
        "best_validation_accuracy_delta": (
            best_accuracy - fork_row["validation_accuracy"]),
        "epochs_to_best": int(best_row["epoch"]) - stall_start + 1,
        "training_seconds": sum(
            float(row.get("epoch_seconds", 0.0)) for row in control),
        "peak_gpu_memory": max(
            int(row.get("peak_gpu_memory", 0)) for row in control),
        "peak_train_params": int(deploy_params),
        "deploy_params": int(deploy_params),
        "time_spent_expanded_seconds": 0.0,
        "bypass_completed": None,
        "intervention": None,
        "best_checkpoint": str(plateau_path),
    }
    save(plateau_path, best)
    return best


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    if args.max_epoch <= START_EPOCH:
        raise ValueError("max_epoch must exceed 300")
    import sys
    reference_root = Path(args.reference_root).resolve()
    sys.path.insert(0, str(reference_root))
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    fork_path = Path(args.fork_checkpoint)
    fork_hash = sha256_file(fork_path)
    if fork_hash != args.fork_checkpoint_hash:
        raise RuntimeError("theta300 checkpoint hash mismatch")
    source = torch.load(fork_path, map_location=device, weights_only=False)
    if (source.get("kind") != "shared_fork_checkpoint" or
            int(source.get("epoch", -1)) != START_EPOCH):
        raise RuntimeError("fork checkpoint must be shared theta300")

    source_tuning = list(source["tuning_indices"])
    (train_set, eval_set, generated_train, generated_validation,
     generated_tuning) = datasets_and_indices(
        args.data_root, args.validation_samples, len(source_tuning))
    train_indices = list(source["train_indices"])
    validation_pool = list(source["validation_indices"])
    if (train_indices != generated_train or
            validation_pool != generated_validation or
            source_tuning != generated_tuning):
        raise RuntimeError("theta300 data split mismatch")
    if not 0 < args.trigger_samples < len(validation_pool):
        raise ValueError("invalid held-out trigger split")
    trigger_indices = validation_pool[:args.trigger_samples]
    evaluation_indices = validation_pool[args.trigger_samples:]

    model = build_cifar_gromo_resnet18(device)
    deploy_params = sum(parameter.numel() for parameter in model.parameters())
    optimizer, _ = build_optimizer_scheduler(
        model, float(source["protocol"].get("learning_rate", 0.1)),
        args.weight_decay)
    model.load_state_dict(source["model"], strict=True)
    optimizer.load_state_dict(source["optimizer"])
    fork_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if not all(lr > 0 for lr in fork_lrs):
        raise RuntimeError("theta300 must have positive LR")
    # Phase 1 is explicitly a matched extended-convergence protocol, not the
    # old finite cosine. Keep theta300 LR constant so plateau is not caused by
    # the scheduler reaching zero. This schedule is then inherited by theta_P.
    scheduler = ConstantCheckpointScheduler(optimizer)
    restore_rng(source["rng"])
    train_loader = make_train_loader(
        train_set, train_indices, args.batch_size, args.workers,
        source["train_loader_generator_state"], args.seed)
    trigger_loader = make_eval_loader(
        eval_set, trigger_indices, args.batch_size * 2, args.workers)
    evaluation_loader = make_eval_loader(
        eval_set, evaluation_indices, args.batch_size * 2, args.workers)

    detector = BestCheckpointStallDetector(
        args.no_new_best_patience, args.significant_min_gain)
    protocol = {
        "phase": "vanilla_best_checkpoint_search", "source_epoch": 300,
        "source_checkpoint_hash": fork_hash,
        "schedule": "constant theta300 LR; no LR optimization or restart",
        "schedule_id": "constant-theta300-lr-best-stall-v1",
        "selection_metric": "trigger accuracy (2,000 held-out samples)",
        "evaluation_role": "report-only (3,000 held-out samples)",
        "no_new_best_patience": args.no_new_best_patience,
        "exact_best_min_gain": 0.0,
        "significant_min_gain": args.significant_min_gain,
        "training_indices_unchanged": True,
        "trigger_samples": len(trigger_indices),
        "evaluation_samples": len(evaluation_indices),
        "official_test_used": False,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoint_latest.pt"
    best_path = output / "checkpoint_best.pt"
    history = []
    start_epoch = START_EPOCH
    prior_seconds = 0.0
    prior_peak = 0
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["protocol"] != protocol:
            raise RuntimeError("convergence-search resume protocol mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        train_loader.generator.set_state(
            saved["train_loader_generator_state"].cpu())
        detector.load_state_dict(saved["best_stall_detector"])
        history = saved["history"]
        start_epoch = int(saved["epoch"])
        prior_seconds = float(saved.get("training_seconds", 0.0))
        prior_peak = int(saved.get("peak_gpu_memory", 0))
        if not best_path.is_file():
            raise RuntimeError(
                "resume requires checkpoint_best.pt beside checkpoint_latest.pt")

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    plateau_path = output / "plateau_checkpoint.pt"

    # Epoch 300 itself is eligible to be the best model. Save it before any
    # continuation step so a later stall can roll back weights, optimizer,
    # momentum, scheduler, RNG, and loader order to exactly theta_best.
    if not history:
        trigger = evaluate(model, trigger_loader, device)
        evaluation = evaluate(model, evaluation_loader, device)
        selection = detector.update(START_EPOCH, trigger["accuracy"])
        baseline = {
            "epoch": START_EPOCH, "train_loss": None,
            "train_accuracy": None,
            "trigger_accuracy": trigger["accuracy"],
            "trigger_loss": trigger["loss"],
            "validation_accuracy": evaluation["accuracy"],
            "validation_loss": evaluation["loss"],
            "learning_rates": fork_lrs,
            "best_checkpoint_statistics": selection,
        }
        history.append(baseline)
        initial_payload = {
            "kind": "vanilla_best_checkpoint", "epoch": START_EPOCH,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": rng_state(),
            "train_loader_generator_state": train_loader.generator.get_state(),
            "train_indices": train_indices, "trigger_indices": trigger_indices,
            "evaluation_indices": evaluation_indices,
            "source_tuning_indices": source_tuning,
            "history": list(history),
            "best_stall_detector": detector.state_dict(),
            "training_seconds": 0.0, "peak_gpu_memory": 0,
            "protocol": protocol,
        }
        save(best_path, initial_payload)
        save(latest, {**initial_payload, "kind": "vanilla_convergence_progress"})

    if (detector.observations[-1]["stalled"] and not plateau_path.is_file()):
        recovered = finalize_best_stall(
            best_path, plateau_path, detector, history, protocol,
            deploy_params)
        digest = sha256_file(plateau_path)
        atomic_json_save({
            "checkpoint": str(plateau_path), "sha256": digest,
            "epoch": recovered["epoch"],
            "stall_detected_epoch": recovered["stall_detected_epoch"],
            "protocol": protocol,
        }, plateau_path.with_suffix(".json"))
    for epoch in range(start_epoch + 1, args.max_epoch + 1):
        if plateau_path.is_file():
            break
        epoch_started = time.perf_counter()
        train = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        trigger = evaluate(model, trigger_loader, device)
        evaluation = evaluate(model, evaluation_loader, device)
        selection = detector.update(epoch, trigger["accuracy"])
        row = {
            "epoch": epoch, "train_loss": train["task_loss"],
            "train_accuracy": train["accuracy"],
            "trigger_accuracy": trigger["accuracy"],
            "trigger_loss": trigger["loss"],
            "validation_accuracy": evaluation["accuracy"],
            "validation_loss": evaluation["loss"],
            "learning_rates": [float(group["lr"])
                               for group in optimizer.param_groups],
            "best_checkpoint_statistics": selection,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "peak_gpu_memory": int(torch.cuda.max_memory_allocated(device)),
        }
        history.append(row)
        elapsed = prior_seconds + time.perf_counter() - started
        peak = max(prior_peak, int(torch.cuda.max_memory_allocated(device)))
        payload = {
            "kind": "vanilla_convergence_progress", "epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": rng_state(),
            "train_loader_generator_state": train_loader.generator.get_state(),
            "train_indices": train_indices, "trigger_indices": trigger_indices,
            "evaluation_indices": evaluation_indices,
            "source_tuning_indices": source_tuning,
            "history": history,
            "best_stall_detector": detector.state_dict(),
            "training_seconds": elapsed, "peak_gpu_memory": peak,
            "protocol": protocol,
        }
        save(latest, payload)
        if selection["improved"]:
            save(best_path, {**payload, "kind": "vanilla_best_checkpoint"})
        atomic_json_save({"latest": row}, output / "progress.json")
        print(json.dumps({"vanilla_convergence": row}, sort_keys=True), flush=True)
        if selection["stalled"]:
            best = finalize_best_stall(
                best_path, plateau_path, detector, history, protocol,
                deploy_params)
            digest = sha256_file(plateau_path)
            atomic_json_save({
                "checkpoint": str(plateau_path), "sha256": digest,
                "epoch": best["epoch"], "stall_detected_epoch": epoch,
                "protocol": protocol,
            }, plateau_path.with_suffix(".json"))
            break

    plateau_found = plateau_path.is_file()
    last = history[-1]
    result = {
        "phase": "vanilla_best_checkpoint_search",
        "plateau_found": plateau_found,
        "plateau_epoch": detector.best_epoch if plateau_found else None,
        "best_epoch": detector.best_epoch,
        "stall_detected_epoch": (
            detector.observations[-1]["epoch"] if plateau_found else None),
        "epochs_without_improvement": (
            detector.observations[-1]["epochs_without_improvement"]),
        "review_epoch_reached": last["epoch"],
        "status": ("best_checkpoint_ready_after_stall" if plateau_found else
                   "review_horizon_reached_no_plateau"),
        "final_validation_accuracy": last["validation_accuracy"],
        "best_validation_accuracy": max(
            row["validation_accuracy"] for row in history),
        "exact_best_trigger_accuracy": detector.best_metric,
        "significant_best_trigger_accuracy":
            detector.patience_reference_metric,
        "final_validation_loss": last["validation_loss"],
        "fork_learning_rates": fork_lrs,
        "training_seconds": prior_seconds + time.perf_counter() - started,
        "checkpoint_latest": str(latest),
        "checkpoint_best": str(best_path),
        "plateau_checkpoint": str(plateau_path) if plateau_found else None,
        "history": history, "protocol": protocol,
    }
    atomic_json_save(result, output / "result.json")


if __name__ == "__main__":
    main()
