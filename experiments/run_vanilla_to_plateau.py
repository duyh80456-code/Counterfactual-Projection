"""Train Vanilla from shared theta300 until robust convergence plateau."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.plateau_protocol import (
    ConsecutiveWindowPlateauDetector, ConstantCheckpointScheduler)
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
    parser.add_argument("--plateau-window", type=int, default=20)
    parser.add_argument("--required-plateau-windows", type=int, default=2)
    parser.add_argument("--plateau-accuracy-min-gain", type=float, default=1e-3)
    parser.add_argument("--plateau-loss-ema-min-drop", type=float, default=1e-3)
    parser.add_argument("--plateau-ema-alpha", type=float, default=0.3)
    return parser.parse_args()


def save(path, payload):
    atomic_torch_save({"format_version": 1, **payload}, path)


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

    detector = ConsecutiveWindowPlateauDetector(
        args.plateau_window, args.required_plateau_windows,
        args.plateau_accuracy_min_gain, args.plateau_loss_ema_min_drop,
        args.plateau_ema_alpha)
    protocol = {
        "phase": "vanilla_convergence_search", "source_epoch": 300,
        "source_checkpoint_hash": fork_hash,
        "schedule": "constant theta300 LR until robust plateau",
        "schedule_status": "matched extended-convergence protocol",
        "training_indices_unchanged": True,
        "trigger_samples": len(trigger_indices),
        "evaluation_samples": len(evaluation_indices),
        "plateau_window": args.plateau_window,
        "required_plateau_windows": args.required_plateau_windows,
        "official_test_used": False,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoint_latest.pt"
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
        detector.load_state_dict(saved["plateau_detector"])
        history = saved["history"]
        start_epoch = int(saved["epoch"])
        prior_seconds = float(saved.get("training_seconds", 0.0))
        prior_peak = int(saved.get("peak_gpu_memory", 0))

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    plateau_path = output / "plateau_checkpoint.pt"
    if (detector.consecutive_plateau_windows >= detector.required_windows and
            not plateau_path.is_file()):
        recovered = torch.load(latest, map_location="cpu", weights_only=False)
        recovered["kind"] = "plateau_fork_checkpoint"
        save(plateau_path, recovered)
        digest = sha256_file(plateau_path)
        atomic_json_save({
            "checkpoint": str(plateau_path), "sha256": digest,
            "epoch": recovered["epoch"], "protocol": protocol,
        }, plateau_path.with_suffix(".json"))
    for epoch in range(start_epoch + 1, args.max_epoch + 1):
        if plateau_path.is_file():
            break
        train = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        trigger = evaluate(model, trigger_loader, device)
        plateau = detector.update(epoch, trigger["accuracy"], trigger["loss"])
        evaluation = evaluate(model, evaluation_loader, device)
        row = {
            "epoch": epoch, "train_loss": train["task_loss"],
            "train_accuracy": train["accuracy"],
            "trigger_accuracy": trigger["accuracy"],
            "trigger_loss": trigger["loss"],
            "validation_accuracy": evaluation["accuracy"],
            "validation_loss": evaluation["loss"],
            "learning_rates": [float(group["lr"])
                               for group in optimizer.param_groups],
            "plateau_statistics": plateau,
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
            "history": history, "plateau_detector": detector.state_dict(),
            "training_seconds": elapsed, "peak_gpu_memory": peak,
            "protocol": protocol,
        }
        save(latest, payload)
        atomic_json_save({"latest": row}, output / "progress.json")
        print(json.dumps({"vanilla_convergence": row}, sort_keys=True), flush=True)
        if plateau["plateau"]:
            payload["kind"] = "plateau_fork_checkpoint"
            save(plateau_path, payload)
            digest = sha256_file(plateau_path)
            atomic_json_save({
                "checkpoint": str(plateau_path), "sha256": digest,
                "epoch": epoch, "protocol": protocol,
            }, plateau_path.with_suffix(".json"))
            break

    plateau_found = plateau_path.is_file()
    last = history[-1]
    result = {
        "phase": "vanilla_convergence_search",
        "plateau_found": plateau_found,
        "plateau_epoch": last["epoch"] if plateau_found else None,
        "review_epoch_reached": last["epoch"],
        "status": ("plateau_checkpoint_ready" if plateau_found else
                   "review_horizon_reached_no_plateau"),
        "final_validation_accuracy": last["validation_accuracy"],
        "best_validation_accuracy": max(
            row["validation_accuracy"] for row in history),
        "final_validation_loss": last["validation_loss"],
        "fork_learning_rates": fork_lrs,
        "training_seconds": prior_seconds + time.perf_counter() - started,
        "checkpoint_latest": str(latest),
        "plateau_checkpoint": str(plateau_path) if plateau_found else None,
        "history": history, "protocol": protocol,
    }
    atomic_json_save(result, output / "result.json")


if __name__ == "__main__":
    main()
