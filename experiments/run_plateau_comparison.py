"""Exact-schedule Vanilla versus plateau-triggered E-to-O from theta300."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from experiments.plateau_protocol import PlateauDetector
from experiments.run_gromo_pilot import (
    actual_update_metrics, batch_loss, cg_diagnostics, eval_logits,
    evaluate_heldout_direction, heldout_metrics, parameter_delta_norm,
    preview_projected_gain, reset_projected_momentum, synchronize)
from experiments.run_shared_comparison import propose_structural_candidates
from experiments.shared_protocol import (
    atomic_json_save, atomic_torch_save, build_cifar_gromo_resnet18,
    build_optimizer_scheduler, datasets_and_indices, evaluate,
    make_eval_loader, make_train_loader, restore_rng, rng_state,
    seed_everything, sha256_file, train_epoch)
from methods import EProjection
from probe import CandidateExpansionProbe
from projection import FunctionalProjector


METHODS = ("vanilla_continue", "plateau_e_driven_o")
START_EPOCH = 300


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--fork-checkpoint", required=True)
    parser.add_argument("--fork-checkpoint-hash", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--trigger-samples", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--plateau-window", type=int, default=15)
    parser.add_argument("--plateau-accuracy-min-gain", type=float, default=1e-3)
    parser.add_argument("--plateau-loss-ema-min-drop", type=float, default=1e-3)
    parser.add_argument("--plateau-ema-alpha", type=float, default=0.3)
    parser.add_argument("--minimum-sgd-epochs", type=int, default=15)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--probe-epsilon", type=float, default=0.05)
    parser.add_argument("--statistics-samples", type=int, default=256)
    parser.add_argument("--where-batches", type=int, default=3)
    parser.add_argument("--where-samples", type=int, default=32)
    parser.add_argument("--projection-samples", type=int, default=32)
    parser.add_argument("--gate-samples", type=int, default=32)
    parser.add_argument("--cg-iterations", type=int, default=200)
    parser.add_argument("--cg-relative-tolerance", type=float, default=1e-2)
    parser.add_argument("--cg-preconditioner-probes", type=int, default=8)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--line-search-scales", default="0.0125,0.025,0.05")
    return parser.parse_args()


def intervention_batches(eval_set, train_indices, args, probe_index, device):
    generator = torch.Generator().manual_seed(
        911_731 + args.seed * 10_000 + probe_index)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    total = (args.statistics_samples + args.where_batches * args.where_samples +
             args.projection_samples + args.gate_samples)
    selected = [train_indices[index] for index in order[:total]]
    cursor = 0
    statistics_indices = selected[cursor:cursor + args.statistics_samples]
    cursor += args.statistics_samples
    where_indices = []
    for _ in range(args.where_batches):
        where_indices.append(selected[cursor:cursor + args.where_samples])
        cursor += args.where_samples
    projection_indices = selected[cursor:cursor + args.projection_samples]
    cursor += args.projection_samples
    gate_indices = selected[cursor:cursor + args.gate_samples]

    statistics = list(DataLoader(
        Subset(eval_set, statistics_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True))

    def one_batch(indices):
        loader = DataLoader(
            Subset(eval_set, indices), batch_size=len(indices), shuffle=False,
            num_workers=args.workers, pin_memory=True)
        return tuple(value.to(device, non_blocking=True)
                     for value in next(iter(loader)))

    return (statistics, [one_batch(indices) for indices in where_indices],
            one_batch(projection_indices), one_batch(gate_indices))


def select_by_expansion_gain(model, statistics, where_batches, args, device):
    synchronize(device)
    started = time.perf_counter()
    candidates = propose_structural_candidates(
        model, statistics, rank=args.rank, site="auto", candidate_sites="")
    rows = []
    for candidate in candidates:
        gains = []
        tiny_score = float(candidate.proposal_score)
        for batch in where_batches:
            signal = CandidateExpansionProbe()(
                model, candidate=candidate, batch=batch,
                gate=args.probe_epsilon)
            gains.append(float(signal.observed_loss_gain))
        rows.append({
            "candidate": candidate, "site": str(candidate.module_name),
            "tiny_score": tiny_score, "e_gains": gains,
            "mean_e_gain": sum(gains) / len(gains),
        })
    ranked = sorted(rows, key=lambda row: row["mean_e_gain"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    selected = ranked[0]
    batch_winners = []
    for batch_index in range(len(where_batches)):
        batch_winners.append(max(
            rows, key=lambda row: row["e_gains"][batch_index])["site"])
    top_gap = (ranked[0]["mean_e_gain"] - ranked[1]["mean_e_gain"]
               if len(ranked) > 1 else None)
    synchronize(device)
    diagnostics = {
        "where_selector": "mean_observed_structural_E_gain",
        "selected_site": selected["site"],
        "selected_e_gain": selected["mean_e_gain"],
        "top1_top2_e_gain_gap": top_gap,
        "where_batch_winners": batch_winners,
        "where_stability": (Counter(batch_winners)[selected["site"]] /
                            len(batch_winners)),
        "site_evaluations": {
            row["site"]: {
                "rank": row["rank"], "mean_e_gain": row["mean_e_gain"],
                "per_batch_e_gain": row["e_gains"],
                "tiny_score": row["tiny_score"],
            } for row in ranked
        },
        "where_seconds": time.perf_counter() - started,
    }
    return selected["candidate"], diagnostics


def finite_projection(projection) -> bool:
    norm = float(parameter_delta_norm(projection.parameter_delta))
    return bool(norm > 0 and math.isfinite(norm) and all(
        torch.isfinite(value).all()
        for value in projection.parameter_delta.values()))


def run_intervention(model, optimizer, eval_set, train_indices, args,
                     device, probe_index):
    statistics, where_batches, projection_batch, gate_batch = intervention_batches(
        eval_set, train_indices, args, probe_index, device)
    candidate, where = select_by_expansion_gain(
        model, statistics, where_batches, args, device)
    projector = FunctionalProjector(
        args.damping, args.cg_iterations,
        tolerance=args.cg_relative_tolerance,
        preconditioner_probes=args.cg_preconditioner_probes)
    step = EProjection(projector=projector).discover_candidate(
        model, candidate, projection_batch, gate=args.probe_epsilon,
        projection_scope="residual_path")
    gate_signal = CandidateExpansionProbe()(
        model, candidate=candidate, batch=gate_batch,
        gate=args.probe_epsilon)
    heldout, heldout_seconds = evaluate_heldout_direction(
        projector, model, gate_batch, gate_signal.delta_logits,
        step.projection.parameter_delta, device)
    scales = [float(value) for value in args.line_search_scales.split(",")]
    gains = {str(scale): preview_projected_gain(
        model, step.projection, scale, gate_batch) for scale in scales}
    best_scale = max(scales, key=lambda scale: gains[str(scale)])
    best_gain = gains[str(best_scale)]
    loss_before = batch_loss(model, gate_batch)
    baseline_logits = eval_logits(model, gate_batch[0])
    applied = finite_projection(step.projection) and best_gain > 0
    actual = {}
    momentum_resets = 0
    if applied:
        step.projection.apply_(model, best_scale)
        momentum_resets = reset_projected_momentum(
            optimizer, model, step.projection)
        actual = actual_update_metrics(
            model, gate_batch[0], baseline_logits,
            gate_signal.delta_logits, best_scale)
    return {
        **where, "correction_applied": applied,
        "selected_scale": best_scale if applied else None,
        "line_search_gains": gains,
        "parameter_delta_norm": float(parameter_delta_norm(
            step.projection.parameter_delta)),
        "loss_before": loss_before, "loss_after": batch_loss(model, gate_batch),
        "actual_loss_improvement": (
            loss_before - batch_loss(model, gate_batch)),
        "actual_cosine_alignment": actual.get("actual_cosine_alignment"),
        "actual_relative_residual": actual.get("actual_relative_residual"),
        "momentum_states_reset": momentum_resets,
        "heldout_evaluation_seconds": heldout_seconds,
        **heldout_metrics(heldout), **cg_diagnostics(step.projection),
    }


def save_checkpoint(path, **payload):
    atomic_torch_save({"format_version": 1, **payload}, path)


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    if args.where_batches < 2:
        raise ValueError("WHERE stability requires at least two batches")
    scales = [float(value) for value in args.line_search_scales.split(",")]
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("line-search scales must be positive")
    reference_root = Path(args.reference_root).resolve()
    if not (reference_root / "dual_growth").is_dir():
        raise FileNotFoundError(f"invalid One-Shot-TAS checkout: {reference_root}")
    import sys
    sys.path.insert(0, str(reference_root))
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    fork_path = Path(args.fork_checkpoint)
    actual_hash = sha256_file(fork_path)
    if actual_hash != args.fork_checkpoint_hash:
        raise RuntimeError("theta300 checkpoint hash mismatch")
    source = torch.load(fork_path, map_location=device, weights_only=False)
    source_protocol = source.get("protocol", {})
    if (source.get("kind") != "shared_fork_checkpoint" or
            int(source.get("epoch", -1)) != START_EPOCH or
            not source.get("history") or
            int(source["history"][-1].get("epoch", -1)) != START_EPOCH):
        raise RuntimeError("fork checkpoint must be shared theta300")
    if not source.get("optimizer", {}).get("state"):
        raise RuntimeError("theta300 checkpoint has no SGD optimizer state")

    source_tuning = list(source["tuning_indices"])
    (train_set, eval_set, generated_train, generated_validation,
     generated_tuning) = datasets_and_indices(
        args.data_root, args.validation_samples, len(source_tuning))
    train_indices = list(source["train_indices"])
    validation_indices = source["validation_indices"]
    if (train_indices != generated_train or
            validation_indices != generated_validation or
            source_tuning != generated_tuning):
        raise RuntimeError("theta300 data split differs from requested protocol")
    if not 0 < args.trigger_samples < len(validation_indices):
        raise ValueError("trigger_samples must split the held-out validation pool")
    # Preserve the original training pool exactly. The 5k set that was already
    # held out before theta300 is split into trigger and evaluation partitions.
    trigger_indices = list(validation_indices[:args.trigger_samples])
    evaluation_indices = list(validation_indices[args.trigger_samples:])
    if (set(trigger_indices) & set(train_indices) or
            set(evaluation_indices) & set(train_indices) or
            set(trigger_indices) & set(evaluation_indices)):
        raise RuntimeError("train/trigger/evaluation pools overlap")

    model = build_cifar_gromo_resnet18(device)
    optimizer, scheduler = build_optimizer_scheduler(
        model, float(source_protocol.get("learning_rate", 0.1)),
        args.weight_decay)
    model.load_state_dict(source["model"], strict=True)
    optimizer.load_state_dict(source["optimizer"])
    scheduler.load_state_dict(source["scheduler"])
    fork_learning_rates = [float(group["lr"])
                           for group in optimizer.param_groups]
    if not all(rate > 0 for rate in fork_learning_rates):
        raise RuntimeError("theta300 LR must be positive for the 300-to-500 schedule")
    if not all(math.isclose(float(group["weight_decay"]), args.weight_decay)
               for group in optimizer.param_groups):
        raise RuntimeError("theta300 weight decay differs from requested protocol")
    # Continue the exact scheduler state stored at theta300. Stepping beyond a
    # CosineAnnealingLR T_max would start another half-cycle, so exact-original
    # comparison ends precisely at the stored scheduler horizon.
    if not hasattr(scheduler, "T_max"):
        raise RuntimeError("theta300 scheduler has no finite T_max")
    continuation_epochs = int(scheduler.T_max) - int(scheduler.last_epoch)
    if continuation_epochs <= 0:
        raise RuntimeError("theta300 scheduler has already reached its horizon")
    total_epochs = START_EPOCH + continuation_epochs
    restore_rng(source["rng"])
    train_loader = make_train_loader(
        train_set, train_indices, args.batch_size, args.workers,
        source["train_loader_generator_state"], args.seed)
    validation_loader = make_eval_loader(
        eval_set, evaluation_indices, args.batch_size * 2, args.workers)
    trigger_loader = make_eval_loader(
        eval_set, trigger_indices, args.batch_size * 2, args.workers)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoint_latest.pt"
    protocol = {
        "method": args.method, "dataset": "CIFAR-100",
        "architecture": "CIFAR-ResNet18", "seed": args.seed,
        "source_epoch": START_EPOCH, "total_epochs": total_epochs,
        "continuation_epochs": continuation_epochs,
        "source_checkpoint_hash": actual_hash,
        "optimizer_state_preserved": True,
        "trigger_samples": len(trigger_indices),
        "evaluation_samples": len(evaluation_indices),
        "training_indices_unchanged": True,
        "trigger_disjoint_from_training": True,
        "lr_schedule": (
            "exact optimizer and scheduler state restored from theta300; "
            "continued only to the stored scheduler horizon"),
        "scheduler_state_restored": True,
        "scheduler_restarted": False,
        "official_test_used": False,
    }
    detector = PlateauDetector(
        args.plateau_window, args.plateau_accuracy_min_gain,
        args.plateau_loss_ema_min_drop, args.plateau_ema_alpha,
        args.minimum_sgd_epochs)
    history = []
    interventions = []
    start_offset = 0
    elapsed_before = 0.0
    peak_before = 0
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["protocol"] != protocol:
            raise RuntimeError("plateau-run resume protocol mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        train_loader.generator.set_state(
            saved["train_loader_generator_state"].cpu())
        detector.load_state_dict(saved.get("plateau_detector", {}))
        history = saved["history"]
        interventions = saved["interventions"]
        start_offset = int(saved["continuation_epoch"])
        elapsed_before = float(saved.get("training_seconds", 0.0))
        peak_before = int(saved.get("peak_gpu_memory", 0))

    fork_validation = evaluate(model, validation_loader, device) if not history else None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for offset in range(start_offset, continuation_epochs):
        epoch = START_EPOCH + offset + 1
        train = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        trigger = evaluate(model, trigger_loader, device)
        plateau = detector.update(epoch, trigger["accuracy"], trigger["loss"])
        intervention = None
        if args.method == "plateau_e_driven_o" and plateau["plateau"]:
            pre_intervention = output / f"checkpoint_pre_intervention_epoch{epoch}.pt"
            save_checkpoint(
                pre_intervention, model=model.state_dict(),
                optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                rng=rng_state(),
                train_loader_generator_state=train_loader.generator.get_state(),
                history=history, interventions=interventions,
                plateau_detector=detector.state_dict(),
                continuation_epoch=offset + 1, global_epoch=epoch,
                protocol=protocol, source_checkpoint_hash=actual_hash,
                snapshot_kind="pre_intervention_plateau")
            # TINY/projection may consume RNG internally. Restore it afterward
            # so future SGD shuffling and augmentation follow the same random
            # stream as Vanilla; only the accepted parameter correction differs.
            pre_probe_rng = rng_state()
            try:
                intervention = run_intervention(
                    model, optimizer, eval_set, train_indices, args, device,
                    len(interventions))
            finally:
                restore_rng(pre_probe_rng)
            intervention.update(
                epoch=epoch, trigger=plateau,
                pre_intervention_checkpoint=str(pre_intervention))
            interventions.append(intervention)
            detector.reset()
        validation = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch, "continuation_epoch": offset + 1,
            "train_loss": train["task_loss"],
            "train_accuracy": train["accuracy"],
            "trigger_loss": trigger["loss"],
            "trigger_accuracy": trigger["accuracy"],
            "plateau_diagnostics": plateau,
            "intervention": intervention,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "learning_rates": [float(group["lr"])
                               for group in optimizer.param_groups],
        }
        history.append(row)
        elapsed = elapsed_before + time.perf_counter() - started
        peak = max(peak_before, int(torch.cuda.max_memory_allocated(device)))
        save_checkpoint(
            latest, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), rng=rng_state(),
            train_loader_generator_state=train_loader.generator.get_state(),
            history=history, interventions=interventions,
            plateau_detector=detector.state_dict(),
            continuation_epoch=offset + 1, global_epoch=epoch,
            protocol=protocol, training_seconds=elapsed,
            peak_gpu_memory=peak, train_indices=train_indices,
            evaluation_indices=evaluation_indices,
            trigger_indices=trigger_indices)
        atomic_json_save({"latest": row}, output / "progress.json")
        print(json.dumps({args.method: row}, sort_keys=True), flush=True)

    elapsed = elapsed_before + time.perf_counter() - started
    final = history[-1]
    baseline_validation = (fork_validation or {
        "accuracy": source["history"][-1]["validation_accuracy"],
        "loss": source["history"][-1]["validation_loss"],
    })
    applied = [item for item in interventions
               if item["correction_applied"]]
    result = {
        "method": args.method, "source_epoch": START_EPOCH,
        "final_epoch": total_epochs,
        "continuation_epochs": len(history),
        "source_checkpoint_hash": actual_hash,
        "fork_learning_rates": fork_learning_rates,
        "source_optimizer_state_entries": len(source["optimizer"]["state"]),
        "optimizer_state_preserved": True,
        "scheduler_state_restored": True,
        "scheduler_restarted": False,
        "fork_validation_accuracy": baseline_validation["accuracy"],
        "fork_validation_loss": baseline_validation["loss"],
        "final_validation_accuracy": final["validation_accuracy"],
        "best_validation_accuracy": max(
            row["validation_accuracy"] for row in history),
        "final_validation_loss": final["validation_loss"],
        "best_validation_loss": min(
            row["validation_loss"] for row in history),
        "validation_accuracy_delta": (
            final["validation_accuracy"] - baseline_validation["accuracy"]),
        "intervention_count": len(interventions),
        "correction_application_count": len(applied),
        "correction_application_rate": (
            len(applied) / len(interventions) if interventions else 0.0),
        "interventions": interventions, "history": history,
        "training_seconds": elapsed,
        "peak_gpu_memory": max(
            peak_before, int(torch.cuda.max_memory_allocated(device))),
        "deploy_params": sum(parameter.numel()
                             for parameter in model.parameters()),
        "checkpoint": str(latest), "protocol": protocol,
    }
    atomic_json_save(result, output / "result.json")


if __name__ == "__main__":
    main()
