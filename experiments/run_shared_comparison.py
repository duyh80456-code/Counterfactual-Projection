"""Shared-checkpoint CIFAR-100 comparison for Vanilla and projection arms."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from experiments.run_gromo_pilot import (
    actual_update_metrics, batch_loss, cg_diagnostics,
    evaluate_heldout_direction, eval_logits, heldout_metrics,
    projection_application_gate, reset_projected_momentum, synchronize)
from experiments.shared_protocol import (
    BOOTSTRAP_EPOCH, FORK_EPOCH, POST_FORK_EPOCHS,
    atomic_json_save, atomic_torch_save,
    build_cifar_gromo_resnet18, build_optimizer_scheduler,
    datasets_and_indices, evaluate, load_shared_checkpoint, make_eval_loader,
    make_train_loader, protocol, restore_rng, rng_state,
    rebase_scheduler_from_theta150, save_shared_checkpoint, seed_everything,
    sha256_file, train_epoch)
from methods import EProjection
from methods.e_projection import (
    candidate_projection_block, candidate_projection_parameter_names)
from probe import CandidateExpansionProbe, CounterfactualTinyProbe
from projection import FunctionalProjector


METHODS = (
    "prepare_shared", "vanilla_continue", "ours_e_driven_o",
    "o_projection_only")


@torch.no_grad()
def supervised_functional_descent_direction(model, batch):
    """Negative summed-CE logit gradient: one_hot(y) - softmax(f(x))."""
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        logits = model(batch[0]).detach()
        target = torch.zeros_like(logits).scatter_(
            1, batch[1].view(-1, 1), 1.0)
        return target - logits.softmax(dim=1)
    finally:
        for module, training in modes.items():
            module.training = training


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shared-checkpoint", required=True)
    parser.add_argument("--shared-checkpoint-hash", default="")
    parser.add_argument("--bootstrap-checkpoint", default="")
    parser.add_argument("--bootstrap-checkpoint-hash", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--tuning-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--site", default="stages.2.blocks.0")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--probe-epsilon", type=float, default=0.05)
    parser.add_argument("--statistics-samples", type=int, default=256)
    parser.add_argument("--projection-samples", type=int, default=32)
    parser.add_argument("--cg-iterations", type=int, default=200)
    parser.add_argument("--cg-relative-tolerance", type=float, default=1e-2)
    parser.add_argument("--cg-preconditioner-probes", type=int, default=8)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--application-max-heldout-residual", type=float,
                        default=1.0)
    parser.add_argument("--application-min-heldout-cosine", type=float,
                        default=0.0)
    return parser.parse_args()


def intervention_batches(eval_set, train_indices, args, epoch, device):
    generator = torch.Generator().manual_seed(
        81_337 + args.seed * 10_000 + epoch)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    count = args.statistics_samples + args.projection_samples
    selected = [train_indices[index] for index in order[:count]]
    statistics_indices = selected[:args.statistics_samples]
    projection_indices = selected[args.statistics_samples:]
    statistics_loader = DataLoader(
        Subset(eval_set, statistics_indices), args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    projection_loader = DataLoader(
        Subset(eval_set, projection_indices), args.projection_samples,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    projection_batch = tuple(value.to(device, non_blocking=True)
                             for value in next(iter(projection_loader)))
    return list(statistics_loader), projection_batch


def prepare_shared(args, device, model, optimizer, scheduler, train_set,
                   eval_set, train_indices, validation_indices, tuning_indices,
                   run_protocol):
    target = Path(args.shared_checkpoint)
    manifest = target.with_suffix(".json")
    if target.is_file() and manifest.is_file():
        recorded = json.loads(manifest.read_text())
        actual = sha256_file(target)
        if actual != recorded["sha256"]:
            raise RuntimeError("existing shared checkpoint hash is invalid")
        if (int(recorded.get("epoch", -1)) != FORK_EPOCH or
                recorded.get("protocol") != run_protocol):
            raise RuntimeError("existing shared checkpoint protocol mismatch")
        print(json.dumps(recorded, sort_keys=True), flush=True)
        return
    progress = target.with_name("shared_seed1_progress.pt")
    history = []
    start_epoch = 0
    generator_state = None
    if progress.is_file():
        saved = torch.load(progress, map_location=device)
        if saved["protocol"] != run_protocol:
            raise RuntimeError("shared burn-in progress protocol mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        history = saved["history"]
        start_epoch = int(saved["epoch"])
        generator_state = saved["train_loader_generator_state"]
    else:
        if not args.bootstrap_checkpoint or not args.bootstrap_checkpoint_hash:
            raise RuntimeError("theta_150 bootstrap checkpoint is required")
        bootstrap_path = Path(args.bootstrap_checkpoint)
        actual_hash = sha256_file(bootstrap_path)
        if actual_hash != args.bootstrap_checkpoint_hash:
            raise RuntimeError("theta_150 bootstrap checkpoint hash mismatch")
        saved = torch.load(bootstrap_path, map_location=device)
        if (saved.get("kind") != "shared_fork_checkpoint" or
                int(saved.get("epoch", -1)) != BOOTSTRAP_EPOCH):
            raise RuntimeError("bootstrap checkpoint is not shared theta_150")
        if (saved["train_indices"] != train_indices or
                saved["validation_indices"] != validation_indices or
                saved["tuning_indices"] != tuning_indices):
            raise RuntimeError("theta_150 bootstrap data split mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        history = saved["history"]
        start_epoch = BOOTSTRAP_EPOCH
        generator_state = saved["train_loader_generator_state"]
        scheduler = rebase_scheduler_from_theta150(optimizer)
    train_loader = make_train_loader(
        train_set, train_indices, args.batch_size, args.workers,
        generator_state, args.seed)
    validation_loader = make_eval_loader(
        eval_set, validation_indices, args.batch_size * 2, args.workers)
    for epoch in range(start_epoch, FORK_EPOCH):
        train = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        validation = evaluate(model, validation_loader, device)
        row = {"epoch": epoch + 1, "train_loss": train["task_loss"],
               "train_accuracy": train["accuracy"],
               "validation_loss": validation["loss"],
               "validation_accuracy": validation["accuracy"]}
        history.append(row)
        atomic_torch_save({
            "format_version": 1, "kind": "shared_burnin_progress",
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "epoch": epoch + 1,
            "history": history, "protocol": run_protocol,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "tuning_indices": tuning_indices,
            "rng": rng_state(),
            "train_loader_generator_state": train_loader.generator.get_state(),
        }, progress)
        print(json.dumps({"shared_burn_in": row}, sort_keys=True), flush=True)
    digest = save_shared_checkpoint(
        target, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=FORK_EPOCH, train_indices=train_indices,
        validation_indices=validation_indices, tuning_indices=tuning_indices,
        loader=train_loader, history=history, run_protocol=run_protocol)
    print(json.dumps({"shared_checkpoint": str(target), "sha256": digest},
                     sort_keys=True), flush=True)


def save_arm_checkpoint(path, *, model, optimizer, scheduler, history,
                        post_epoch, shared_hash, train_loader, run_protocol,
                        elapsed, peak_gpu_memory, train_indices,
                        validation_indices, tuning_indices):
    atomic_torch_save({
        "format_version": 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "history": history, "post_epoch": post_epoch,
        "shared_checkpoint_hash": shared_hash, "protocol": run_protocol,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "tuning_indices": tuning_indices,
        "rng": rng_state(),
        "train_loader_generator_state": train_loader.generator.get_state(),
        "training_seconds": elapsed,
        "peak_gpu_memory": peak_gpu_memory,
    }, path)


def run_arm(args, device, model, optimizer, scheduler, train_set, eval_set,
            train_indices, validation_indices, tuning_indices, run_protocol):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, shared_hash = load_shared_checkpoint(
        Path(args.shared_checkpoint), args.shared_checkpoint_hash,
        device=device, model=model, optimizer=optimizer, scheduler=scheduler)
    if checkpoint["protocol"] != run_protocol:
        raise RuntimeError("shared checkpoint protocol differs from arm protocol")
    arm_protocol = {**run_protocol, "method": args.method}
    if args.method in {"ours_e_driven_o", "o_projection_only"}:
        arm_protocol["functional_projection"] = {
            "site": args.site, "rank": args.rank,
            "probe_epsilon": args.probe_epsilon,
            "projection_scope": "residual_path",
            "statistics_samples": args.statistics_samples,
            "projection_samples": args.projection_samples,
            "cg_iterations": args.cg_iterations,
            "cg_relative_tolerance": args.cg_relative_tolerance,
            "cg_preconditioner_probes": args.cg_preconditioner_probes,
            "damping": args.damping,
            "application_max_heldout_residual":
                args.application_max_heldout_residual,
            "application_min_heldout_cosine":
                args.application_min_heldout_cosine,
            "functional_target": (
                "structural_TINY_delta" if args.method == "ours_e_driven_o"
                else "negative_summed_CE_logit_gradient"),
        }
    train_loader = make_train_loader(
        train_set, train_indices, args.batch_size, args.workers,
        checkpoint["train_loader_generator_state"], args.seed)
    validation_loader = make_eval_loader(
        eval_set, validation_indices, args.batch_size * 2, args.workers)
    tuning_loader = make_eval_loader(
        eval_set, tuning_indices, len(tuning_indices), args.workers)
    tuning_batch = tuple(value.to(device, non_blocking=True)
                         for value in next(iter(tuning_loader)))
    arm_checkpoint = output / "checkpoint_latest.pt"
    history = []
    start_epoch = 0
    prior_seconds = 0.0
    prior_peak_gpu_memory = 0
    if arm_checkpoint.is_file():
        saved = torch.load(arm_checkpoint, map_location=device)
        if saved["shared_checkpoint_hash"] != shared_hash:
            raise RuntimeError("arm checkpoint came from another fork checkpoint")
        if saved["protocol"] != arm_protocol:
            raise RuntimeError("arm resume protocol mismatch")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        train_loader.generator.set_state(saved["train_loader_generator_state"].cpu())
        history = saved["history"]
        start_epoch = int(saved["post_epoch"])
        prior_seconds = float(saved.get("training_seconds", 0.0))
        prior_peak_gpu_memory = int(saved.get("peak_gpu_memory", 0))
    initial_params = sum(parameter.numel() for parameter in model.parameters())
    projector = FunctionalProjector(
        args.damping, args.cg_iterations,
        tolerance=args.cg_relative_tolerance,
        preconditioner_probes=args.cg_preconditioner_probes)
    e_projection = EProjection(projector=projector)
    from dual_growth.adapters import TinyAdapter
    from dual_growth.controller import GrowthBudget
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for post_epoch in range(start_epoch, POST_FORK_EPOCHS):
        diagnostics = None
        global_epoch = FORK_EPOCH + post_epoch
        if args.method == "ours_e_driven_o":
            statistics, projection_batch = intervention_batches(
                eval_set, train_indices, args, global_epoch, device)
            adapter = TinyAdapter(
                quantum_params=10**9, max_statistics_batches=len(statistics))
            candidate = CounterfactualTinyProbe(args.rank, args.site).propose(
                adapter, model, statistics, GrowthBudget(10**9),
                sample_inputs=statistics[0][0])
            tuning_signal = CandidateExpansionProbe()(
                model, candidate=candidate, batch=tuning_batch,
                gate=args.probe_epsilon)
            loss_before = batch_loss(model, tuning_batch)
            synchronize(device)
            projection_started = time.perf_counter()
            step = e_projection.discover_candidate(
                model, candidate, projection_batch, gate=args.probe_epsilon,
                projection_scope="residual_path")
            heldout, heldout_seconds = evaluate_heldout_direction(
                projector, model, tuning_batch, tuning_signal.delta_logits,
                step.projection.parameter_delta, device)
            application = projection_application_gate(
                step.projection.parameter_delta, heldout,
                max_relative_residual=args.application_max_heldout_residual,
                min_cosine_alignment=args.application_min_heldout_cosine)
            baseline_logits = eval_logits(model, tuning_batch[0])
            actual = {}
            momentum_resets = 0
            if application["apply"]:
                step.projection.apply_(model, args.probe_epsilon)
                momentum_resets = reset_projected_momentum(
                    optimizer, model, step.projection)
                actual = actual_update_metrics(
                    model, tuning_batch[0], baseline_logits,
                    tuning_signal.delta_logits, args.probe_epsilon)
            loss_after = batch_loss(model, tuning_batch)
            diagnostics = {
                "source": "structural_TINY_E",
                "correction_applied": application["apply"],
                "parameter_delta_norm": application["parameter_delta_norm"],
                "actual_cosine_alignment": actual.get(
                    "actual_cosine_alignment"),
                "actual_relative_residual": actual.get(
                    "actual_relative_residual"),
                "loss_before": loss_before, "loss_after": loss_after,
                "momentum_states_reset": momentum_resets,
                "projection_seconds": time.perf_counter() - projection_started,
                "heldout_evaluation_seconds": heldout_seconds,
                **heldout_metrics(heldout), **cg_diagnostics(step.projection),
            }
        elif args.method == "o_projection_only":
            _, projection_batch = intervention_batches(
                eval_set, train_indices, args, global_epoch, device)
            marker = SimpleNamespace(module_name=args.site)
            block = candidate_projection_block(model, marker)
            parameter_names = candidate_projection_parameter_names(
                model, marker, "residual_path")
            fit_target = supervised_functional_descent_direction(
                model, projection_batch)
            heldout_target = supervised_functional_descent_direction(
                model, tuning_batch)
            loss_before = batch_loss(model, tuning_batch)
            synchronize(device)
            projection_started = time.perf_counter()
            projection = projector.project(
                model, projection_batch[0], fit_target, block=block,
                parameter_names=parameter_names)
            heldout, heldout_seconds = evaluate_heldout_direction(
                projector, model, tuning_batch, heldout_target,
                projection.parameter_delta, device)
            application = projection_application_gate(
                projection.parameter_delta, heldout,
                max_relative_residual=args.application_max_heldout_residual,
                min_cosine_alignment=args.application_min_heldout_cosine)
            baseline_logits = eval_logits(model, tuning_batch[0])
            actual = {}
            momentum_resets = 0
            if application["apply"]:
                projection.apply_(model, args.probe_epsilon)
                momentum_resets = reset_projected_momentum(
                    optimizer, model, projection)
                actual = actual_update_metrics(
                    model, tuning_batch[0], baseline_logits,
                    heldout_target, args.probe_epsilon)
            diagnostics = {
                "source": "original_space_supervised_projection_only",
                "correction_applied": application["apply"],
                "parameter_delta_norm": application["parameter_delta_norm"],
                "actual_cosine_alignment": actual.get(
                    "actual_cosine_alignment"),
                "actual_relative_residual": actual.get(
                    "actual_relative_residual"),
                "loss_before": loss_before,
                "loss_after": batch_loss(model, tuning_batch),
                "momentum_states_reset": momentum_resets,
                "projection_seconds": time.perf_counter() - projection_started,
                "heldout_evaluation_seconds": heldout_seconds,
                **heldout_metrics(heldout), **cg_diagnostics(projection),
            }
        train = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        validation = evaluate(model, validation_loader, device)
        row = {
            "epoch": global_epoch + 1, "post_fork_epoch": post_epoch + 1,
            "train_loss": train["task_loss"],
            "train_accuracy": train["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "diagnostics": diagnostics,
        }
        history.append(row)
        elapsed = prior_seconds + time.perf_counter() - started
        peak_gpu_memory = max(
            prior_peak_gpu_memory,
            int(torch.cuda.max_memory_allocated(device)))
        save_arm_checkpoint(
            arm_checkpoint, model=model, optimizer=optimizer,
            scheduler=scheduler, history=history, post_epoch=post_epoch + 1,
            shared_hash=shared_hash, train_loader=train_loader,
            run_protocol=arm_protocol, elapsed=elapsed,
            peak_gpu_memory=peak_gpu_memory, train_indices=train_indices,
            validation_indices=validation_indices,
            tuning_indices=tuning_indices)
        atomic_json_save({"method": args.method, "completed_post_fork_epochs":
                          post_epoch + 1, "latest": row},
                         output / "progress.json")
        print(json.dumps({args.method: row}, sort_keys=True), flush=True)
    elapsed = (prior_seconds if start_epoch >= POST_FORK_EPOCHS else
               prior_seconds + time.perf_counter() - started)
    last = history[-1]
    applied = [row["diagnostics"] for row in history
               if row["diagnostics"] and row["diagnostics"]["correction_applied"]]
    result = {
        "method": args.method, "shared_checkpoint_hash": shared_hash,
        "fork_epoch": FORK_EPOCH, "post_fork_epochs": len(history),
        "final_validation_accuracy": last["validation_accuracy"],
        "best_validation_accuracy": max(
            row["validation_accuracy"] for row in history),
        "final_validation_loss": last["validation_loss"],
        "training_seconds": elapsed,
        "peak_gpu_memory": max(
            prior_peak_gpu_memory,
            int(torch.cuda.max_memory_allocated(device))),
        "deploy_params": sum(parameter.numel() for parameter in model.parameters()),
        "initial_deploy_params": initial_params, "history": history,
        "checkpoint": str(arm_checkpoint), "protocol": arm_protocol,
    }
    if args.method in {"ours_e_driven_o", "o_projection_only"}:
        result.update({
            "correction_application_rate": len(applied) / len(history),
            "actual_cosine_alignment": (applied[-1].get(
                "actual_cosine_alignment") if applied else None),
            "actual_relative_residual": (applied[-1].get(
                "actual_relative_residual") if applied else None),
        })
    atomic_json_save(result, output / "result.json")


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    reference_root = Path(args.reference_root).resolve()
    if not (reference_root / "dual_growth").is_dir():
        raise FileNotFoundError(f"invalid One-Shot-TAS checkout: {reference_root}")
    import sys
    sys.path.insert(0, str(reference_root))
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    (train_set, eval_set, train_indices, validation_indices,
     tuning_indices) = datasets_and_indices(
         args.data_root, args.validation_samples, args.tuning_samples)
    run_protocol = protocol(
        args.seed, train_indices, validation_indices, tuning_indices,
        args.batch_size, args.lr, args.weight_decay)
    model = build_cifar_gromo_resnet18(device)
    optimizer, scheduler = build_optimizer_scheduler(
        model, args.lr, args.weight_decay)
    if args.method == "prepare_shared":
        prepare_shared(
            args, device, model, optimizer, scheduler, train_set, eval_set,
            train_indices, validation_indices, tuning_indices, run_protocol)
    else:
        if not args.shared_checkpoint_hash:
            raise ValueError("comparison arms require --shared-checkpoint-hash")
        run_arm(
            args, device, model, optimizer, scheduler, train_set, eval_set,
            train_indices, validation_indices, tuning_indices, run_protocol)


if __name__ == "__main__":
    main()
