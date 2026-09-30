"""Train Vanilla continuously from initialization until a confirmed stall."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.plateau_protocol import (
    BestCheckpointStallDetector, StandardMultiStepScheduler)
from experiments.run_vanilla_to_plateau import finalize_best_stall
from experiments.shared_protocol import (
    atomic_json_save, atomic_torch_save, build_cifar_gromo_resnet18,
    datasets_and_indices, evaluate, index_sha256, make_eval_loader,
    make_train_loader, restore_rng, rng_state, seed_everything, sha256_file,
    train_epoch)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-epoch", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--trigger-samples", type=int, default=2000)
    parser.add_argument("--tuning-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--recipe-epochs", type=int, default=200)
    parser.add_argument("--lr-milestones", default="100,150")
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--stall-patience", type=int, default=100)
    parser.add_argument("--best-min-gain", type=float, default=0.0)
    parser.add_argument("--significant-min-gain", type=float, default=1e-3)
    return parser.parse_args()


def save(path, payload):
    atomic_torch_save({"format_version": 1, **payload}, path)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    if args.best_min_gain != 0.0:
        raise ValueError("best_min_gain must be zero")
    if args.significant_min_gain <= 0 or args.stall_patience < 1:
        raise ValueError("invalid stall configuration")
    import sys
    reference_root = Path(args.reference_root).resolve()
    sys.path.insert(0, str(reference_root))
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    (train_set, eval_set, train_indices, validation_indices,
     tuning_indices) = datasets_and_indices(
        args.data_root, args.validation_samples, args.tuning_samples)
    if not 0 < args.trigger_samples < len(validation_indices):
        raise ValueError("invalid trigger/evaluation split")
    trigger_indices = validation_indices[:args.trigger_samples]
    evaluation_indices = validation_indices[args.trigger_samples:]
    model = build_cifar_gromo_resnet18(device)
    deploy_params = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay)
    milestones = tuple(int(value) for value in args.lr_milestones.split(","))
    scheduler = StandardMultiStepScheduler(
        optimizer, milestones=milestones, gamma=args.lr_gamma,
        recipe_epochs=args.recipe_epochs)
    detector = BestCheckpointStallDetector(
        args.stall_patience, args.significant_min_gain, require_arm=True)
    protocol = {
        "phase": "unified_vanilla_from_initialization", "seed": args.seed,
        "dataset": "CIFAR-100", "architecture": "CIFAR-ResNet18",
        "input_size": 32, "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "schedule_id": "cifar-resnet18-sgd-multistep-200-v2-post-arm-best",
        "schedule": (
            f"base recipe: {args.recipe_epochs} epochs, milestones="
            f"{milestones}, gamma={args.lr_gamma}; metric-independent"),
        "scheduler_restart_count": 0, "trigger_samples": len(trigger_indices),
        "evaluation_samples": len(evaluation_indices),
        "tuning_samples": len(tuning_indices),
        "selection_metric": "trigger accuracy",
        "evaluation_role": "report-only",
        "stall_patience": args.stall_patience,
        "stall_gate": "base backbone recipe complete",
        "theta_P_scope": "exact trigger best at or after stall arm",
        "exact_best_min_gain": args.best_min_gain,
        "significant_min_gain": args.significant_min_gain,
        "train_indices_sha256": index_sha256(train_indices),
        "validation_indices_sha256": index_sha256(validation_indices),
        "tuning_indices_sha256": index_sha256(tuning_indices),
        "official_test_used": False,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoint_latest.pt"
    best_path = output / "checkpoint_best.pt"
    plateau_path = output / "plateau_checkpoint.pt"
    history, start_epoch, elapsed_before, peak_before = [], 0, 0.0, 0
    generator_state = None
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["protocol"] != protocol:
            raise RuntimeError("unified Vanilla resume protocol mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        generator_state = saved["train_loader_generator_state"]
        detector.load_state_dict(saved["best_stall_detector"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"])
        elapsed_before = float(saved.get("training_seconds", 0.0))
        peak_before = int(saved.get("peak_gpu_memory", 0))
        if not best_path.is_file():
            raise RuntimeError("resume requires checkpoint_best.pt")
    train_loader = make_train_loader(
        train_set, train_indices, args.batch_size, args.workers,
        generator_state, args.seed)
    trigger_loader = make_eval_loader(
        eval_set, trigger_indices, args.batch_size * 2, args.workers)
    evaluation_loader = make_eval_loader(
        eval_set, evaluation_indices, args.batch_size * 2, args.workers)

    def payload(kind, epoch, elapsed, peak):
        return {
            "kind": kind, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "rng": rng_state(),
            "train_loader_generator_state": train_loader.generator.get_state(),
            "train_indices": train_indices, "trigger_indices": trigger_indices,
            "evaluation_indices": evaluation_indices,
            "source_tuning_indices": tuning_indices, "history": history,
            "best_stall_detector": detector.state_dict(),
            "training_seconds": elapsed, "peak_gpu_memory": peak,
            "protocol": protocol,
        }

    if not history:
        trigger = evaluate(model, trigger_loader, device)
        evaluation = evaluate(model, evaluation_loader, device)
        selection = detector.update(0, trigger["accuracy"])
        history.append({
            "epoch": 0, "train_loss": None, "train_accuracy": None,
            "trigger_accuracy": trigger["accuracy"],
            "trigger_loss": trigger["loss"],
            "validation_accuracy": evaluation["accuracy"],
            "validation_loss": evaluation["loss"],
            "learning_rates": [args.lr],
            "next_learning_rates": [float(group["lr"])
                                    for group in optimizer.param_groups],
            "best_checkpoint_statistics": selection,
        })
        save(best_path, payload("vanilla_best_checkpoint", 0, 0.0, 0))
        save(latest, payload("unified_vanilla_progress", 0, 0.0, 0))
    if detector.observations[-1]["stalled"] and not plateau_path.is_file():
        recovered = finalize_best_stall(
            best_path, plateau_path, detector, history, protocol, deploy_params)
        atomic_json_save({
            "checkpoint": str(plateau_path),
            "sha256": sha256_file(plateau_path),
            "epoch": recovered["epoch"],
            "stall_detected_epoch": recovered["stall_detected_epoch"],
            "seed": args.seed, "protocol": protocol,
        }, plateau_path.with_suffix(".json"))

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch + 1, args.max_epoch + 1):
        if plateau_path.is_file():
            break
        epoch_started = time.perf_counter()
        training_lrs = [float(group["lr"])
                        for group in optimizer.param_groups]
        train = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        trigger = evaluate(model, trigger_loader, device)
        evaluation = evaluate(model, evaluation_loader, device)
        selection = detector.update(epoch, trigger["accuracy"])
        if scheduler.recipe_complete() and not detector.stall_armed:
            detector.arm_stall(epoch, trigger["accuracy"])
        row = {
            "epoch": epoch, "train_loss": train["task_loss"],
            "train_accuracy": train["accuracy"],
            "trigger_accuracy": trigger["accuracy"],
            "trigger_loss": trigger["loss"],
            "validation_accuracy": evaluation["accuracy"],
            "validation_loss": evaluation["loss"],
            "learning_rates": training_lrs,
            "next_learning_rates": [float(group["lr"])
                                    for group in optimizer.param_groups],
            "best_checkpoint_statistics": selection,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "peak_gpu_memory": int(torch.cuda.max_memory_allocated(device)),
        }
        history.append(row)
        elapsed = elapsed_before + time.perf_counter() - started
        peak = max(peak_before, int(torch.cuda.max_memory_allocated(device)))
        current = payload("unified_vanilla_progress", epoch, elapsed, peak)
        save(latest, current)
        if selection["improved"]:
            save(best_path, {**current, "kind": "vanilla_best_checkpoint"})
        atomic_json_save({"latest": row}, output / "progress.json")
        print(json.dumps({"unified_vanilla": row}, sort_keys=True), flush=True)
        if selection["stalled"]:
            best = finalize_best_stall(
                best_path, plateau_path, detector, history, protocol,
                deploy_params)
            atomic_json_save({
                "checkpoint": str(plateau_path),
                "sha256": sha256_file(plateau_path), "epoch": best["epoch"],
                "stall_detected_epoch": epoch, "seed": args.seed,
                "protocol": protocol,
            }, plateau_path.with_suffix(".json"))
            break
    plateau_found = plateau_path.is_file()
    last = history[-1]
    atomic_json_save({
        "phase": "unified_vanilla_from_initialization",
        "seed": args.seed,
        "status": ("stall_detected" if plateau_found else
                   "no_stall_detected"),
        "plateau_found": plateau_found,
        "base_recipe_complete": scheduler.recipe_complete(),
        "stall_armed_epoch": detector.stall_armed_epoch,
        "pre_arm_global_best_epoch": detector.pre_arm_best_epoch,
        "pre_arm_global_best_accuracy": detector.pre_arm_best_metric,
        "best_epoch": detector.best_epoch,
        "stall_detected_epoch": (
            detector.observations[-1]["epoch"] if plateau_found else None),
        "review_epoch_reached": last["epoch"],
        "final_validation_accuracy": last["validation_accuracy"],
        "best_validation_accuracy": max(
            row["validation_accuracy"] for row in history),
        "exact_best_trigger_accuracy": detector.best_metric,
        "significant_best_trigger_accuracy":
            detector.patience_reference_metric,
        "plateau_checkpoint": str(plateau_path) if plateau_found else None,
        "checkpoint_latest": str(latest), "checkpoint_best": str(best_path),
        "history": history, "protocol": protocol,
    }, output / "result.json")


if __name__ == "__main__":
    main()
