"""Fork one converged theta_P into Vanilla, Bypass, and one-shot E-to-O."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

from baselines.bypass import (
    add_extension_parameters_, contraction_norm, embed_relaxed_bypass,
    extension_parameters, project_ready_activations, transition_from_opt2_)
from experiments.plateau_protocol import ConstantCheckpointScheduler
from experiments.run_gromo_pilot import (
    actual_update_metrics, batch_loss, cg_diagnostics, eval_logits,
    evaluate_heldout_direction, heldout_metrics, parameter_delta_norm,
    preview_projected_gain, reset_projected_momentum)
from experiments.run_plateau_comparison import finite_projection, run_intervention
from experiments.run_shared_comparison import (
    supervised_functional_descent_direction)
from experiments.shared_protocol import (
    atomic_json_save, atomic_torch_save, build_cifar_gromo_resnet18,
    build_optimizer_scheduler, datasets_and_indices, evaluate,
    make_eval_loader, make_train_loader, restore_rng, rng_state,
    seed_everything, sha256_file, train_epoch)
from methods.e_projection import (
    candidate_projection_block, candidate_projection_parameter_names)
from projection import FunctionalProjector


METHODS = ("vanilla", "bypass", "ours_e_driven_o", "o_projection_only")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--plateau-checkpoint", required=True)
    parser.add_argument("--plateau-checkpoint-hash", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--post-fork-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--site", default="stages.2.blocks.0")
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
    parser.add_argument("--opt1-epochs", type=int, default=40)
    parser.add_argument("--max-opt2-epochs", type=int, default=20)
    parser.add_argument("--contraction-epsilon", type=float, default=0.002)
    parser.add_argument("--gamma-slope", type=float, default=3e-6)
    return parser.parse_args()


def save_checkpoint(path, payload):
    atomic_torch_save({"format_version": 1, **payload}, path)


def run_o_only_intervention(model, optimizer, eval_set, train_indices,
                            args, device):
    """One supervised projection-only control at theta_P."""
    generator = torch.Generator().manual_seed(911_731 + args.seed * 10_000)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    count = args.projection_samples + args.gate_samples
    selected = [train_indices[index] for index in order[:count]]

    def one_batch(indices):
        loader = DataLoader(
            Subset(eval_set, indices), batch_size=len(indices), shuffle=False,
            num_workers=args.workers, pin_memory=True)
        return tuple(value.to(device, non_blocking=True)
                     for value in next(iter(loader)))

    projection_batch = one_batch(selected[:args.projection_samples])
    gate_batch = one_batch(selected[args.projection_samples:])
    marker = SimpleNamespace(module_name=args.site)
    block = candidate_projection_block(model, marker)
    parameter_names = candidate_projection_parameter_names(
        model, marker, "residual_path")
    projector = FunctionalProjector(
        args.damping, args.cg_iterations,
        tolerance=args.cg_relative_tolerance,
        preconditioner_probes=args.cg_preconditioner_probes)
    fit_target = supervised_functional_descent_direction(
        model, projection_batch)
    projection = projector.project(
        model, projection_batch[0], fit_target, block=block,
        parameter_names=parameter_names)
    gate_target = supervised_functional_descent_direction(model, gate_batch)
    heldout, heldout_seconds = evaluate_heldout_direction(
        projector, model, gate_batch, gate_target,
        projection.parameter_delta, device)
    scales = [float(value) for value in args.line_search_scales.split(",")]
    gains = {str(scale): preview_projected_gain(
        model, projection, scale, gate_batch) for scale in scales}
    best_scale = max(scales, key=lambda scale: gains[str(scale)])
    best_gain = gains[str(best_scale)]
    loss_before = batch_loss(model, gate_batch)
    baseline_logits = eval_logits(model, gate_batch[0])
    applied = finite_projection(projection) and best_gain > 0
    actual = {}
    momentum_resets = 0
    if applied:
        projection.apply_(model, best_scale)
        momentum_resets = reset_projected_momentum(
            optimizer, model, projection)
        actual = actual_update_metrics(
            model, gate_batch[0], baseline_logits, gate_target, best_scale)
    loss_after = batch_loss(model, gate_batch)
    return {
        "source": "supervised_projection_only_control",
        "functional_target": "one_hot_minus_softmax",
        "uses_structural_E": False, "selected_site": args.site,
        "correction_applied": applied,
        "selected_scale": best_scale if applied else None,
        "line_search_gains": gains,
        "parameter_delta_norm": float(parameter_delta_norm(
            projection.parameter_delta)),
        "loss_before": loss_before, "loss_after": loss_after,
        "actual_loss_improvement": loss_before - loss_after,
        "actual_cosine_alignment": actual.get("actual_cosine_alignment"),
        "actual_relative_residual": actual.get("actual_relative_residual"),
        "momentum_states_reset": momentum_resets,
        "heldout_evaluation_seconds": heldout_seconds,
        **heldout_metrics(heldout), **cg_diagnostics(projection),
    }


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    if args.post_fork_epochs < 1:
        raise ValueError("post_fork_epochs must be positive")
    import sys
    reference_root = Path(args.reference_root).resolve()
    sys.path.insert(0, str(reference_root))
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    fork_path = Path(args.plateau_checkpoint)
    fork_hash = sha256_file(fork_path)
    if fork_hash != args.plateau_checkpoint_hash:
        raise RuntimeError("plateau checkpoint hash mismatch")
    source = torch.load(fork_path, map_location=device, weights_only=False)
    if source.get("kind") != "plateau_fork_checkpoint":
        raise RuntimeError("input is not a plateau fork checkpoint")
    fork_epoch = int(source["epoch"])

    source_tuning = list(source["source_tuning_indices"])
    validation_samples = (len(source["trigger_indices"]) +
                          len(source["evaluation_indices"]))
    (train_set, eval_set, generated_train, generated_validation,
     generated_tuning) = datasets_and_indices(
        args.data_root, validation_samples, len(source_tuning))
    train_indices = list(source["train_indices"])
    trigger_indices = list(source["trigger_indices"])
    evaluation_indices = list(source["evaluation_indices"])
    if (train_indices != generated_train or
            trigger_indices + evaluation_indices != generated_validation or
            source_tuning != generated_tuning):
        raise RuntimeError("plateau checkpoint data split mismatch")

    model = build_cifar_gromo_resnet18(device)
    optimizer, _ = build_optimizer_scheduler(model, 0.1, args.weight_decay)
    model.load_state_dict(source["model"], strict=True)
    optimizer.load_state_dict(source["optimizer"])
    scheduler = ConstantCheckpointScheduler(optimizer)
    scheduler.load_state_dict(source["scheduler"])
    fork_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    restore_rng(source["rng"])
    train_loader = make_train_loader(
        train_set, train_indices, args.batch_size, args.workers,
        source["train_loader_generator_state"], args.seed)
    evaluation_loader = make_eval_loader(
        eval_set, evaluation_indices, args.batch_size * 2, args.workers)
    fork_evaluation = evaluate(model, evaluation_loader, device)

    protocol = {
        "phase": "plateau_fork_comparison", "method": args.method,
        "fork_epoch": fork_epoch, "post_fork_epochs": args.post_fork_epochs,
        "stall_detected_epoch": source.get("stall_detected_epoch"),
        "plateau_checkpoint_hash": fork_hash,
        "optimizer_state_preserved": True,
        "scheduler_state_preserved": True,
        "training_indices_unchanged": True,
        "official_test_used": False,
        "structural_E_interventions": (
            1 if args.method == "ours_e_driven_o" else 0),
        "supervised_O_interventions": (
            1 if args.method == "o_projection_only" else 0),
        "o_projection_site": (
            args.site if args.method == "o_projection_only" else None),
        "bypass_schedule": ({
            "opt1_epochs": args.opt1_epochs,
            "max_opt2_epochs": args.max_opt2_epochs,
            "force_projection": False,
        } if args.method == "bypass" else None),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoint_latest.pt"
    saved = (torch.load(latest, map_location=device, weights_only=False)
             if latest.is_file() else None)
    if saved is not None and saved["protocol"] != protocol:
        raise RuntimeError("plateau-fork resume protocol mismatch")

    phase = "train"
    extension_paths = []
    opt1_done = opt2_done = train3_done = opt2_steps = 0
    contraction_at_projection = projection_loss_jump = None
    expanded_seconds = 0.0
    peak_train_params = sum(parameter.numel() for parameter in model.parameters())
    if args.method == "bypass":
        phase = saved["phase"] if saved is not None else "opt1"
        if phase in {"opt1", "opt2"}:
            embed_started = time.perf_counter()
            extension_paths = embed_relaxed_bypass(model)
            add_extension_parameters_(optimizer, extension_parameters(model))
            expanded_seconds += time.perf_counter() - embed_started
            scheduler = ConstantCheckpointScheduler(optimizer)
            # Extensions share the existing optimizer group and therefore the
            # same constant base LR; preserve elapsed schedule steps explicitly.
            scheduler.steps = int(source["scheduler"]["steps"])
            scheduler.learning_rates = tuple(
                float(group["lr"]) for group in optimizer.param_groups)

    history = []
    intervention = None
    start_offset = 0
    elapsed_before = 0.0
    peak_before = 0
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        train_loader.generator.set_state(
            saved["train_loader_generator_state"].cpu())
        history = saved["history"]
        intervention = saved.get("intervention")
        start_offset = int(saved["post_fork_epoch"])
        phase = saved["phase"]
        opt1_done = int(saved.get("opt1_epochs", 0))
        opt2_done = int(saved.get("opt2_epochs", 0))
        train3_done = int(saved.get("train3_epochs", 0))
        opt2_steps = int(saved.get("opt2_steps", 0))
        contraction_at_projection = saved.get("contraction_at_projection")
        projection_loss_jump = saved.get("projection_loss_jump")
        extension_paths = saved.get("extension_paths", extension_paths)
        expanded_seconds = float(saved.get("time_spent_expanded_seconds", 0.0))
        peak_train_params = int(saved.get("peak_train_params", peak_train_params))
        elapsed_before = float(saved.get("training_seconds", 0.0))
        peak_before = int(saved.get("peak_gpu_memory", 0))

    if args.method in {"ours_e_driven_o", "o_projection_only"} and intervention is None:
        pre_probe_rng = rng_state()
        intervention_started = time.perf_counter()
        try:
            if args.method == "ours_e_driven_o":
                intervention = run_intervention(
                    model, optimizer, eval_set, train_indices, args, device, 0)
            else:
                intervention = run_o_only_intervention(
                    model, optimizer, eval_set, train_indices, args, device)
        finally:
            restore_rng(pre_probe_rng)
        intervention["epoch"] = fork_epoch
        intervention["intervention_seconds"] = (
            time.perf_counter() - intervention_started)
        post_path = output / "checkpoint_post_intervention.pt"
        save_checkpoint(post_path, {
            "kind": "plateau_post_intervention", "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": rng_state(), "train_loader_generator_state":
                train_loader.generator.get_state(),
            "epoch": fork_epoch, "train_indices": train_indices,
            "trigger_indices": trigger_indices,
            "evaluation_indices": evaluation_indices,
            "source_tuning_indices": source_tuning,
            "intervention": intervention, "protocol": protocol,
        })

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for offset in range(start_offset, args.post_fork_epochs):
        epoch = fork_epoch + offset + 1
        epoch_phase = phase
        epoch_started = time.perf_counter()
        gamma = 0.0
        criterion_met = None
        if args.method != "bypass" or phase == "train3":
            train = train_epoch(model, train_loader, optimizer, device)
            if args.method == "bypass": train3_done += 1
        elif phase == "opt1":
            train = train_epoch(model, train_loader, optimizer, device)
            opt1_done += 1
            if opt1_done >= args.opt1_epochs: phase = "opt2"
        else:
            def penalty():
                nonlocal opt2_steps, gamma
                opt2_steps += 1
                gamma = args.gamma_slope * opt2_steps
                return contraction_norm(model) * gamma

            train = train_epoch(
                model, train_loader, optimizer, device, loss_extra=penalty,
                post_step=lambda: project_ready_activations(
                    model, args.contraction_epsilon))
            opt2_done += 1
            before = evaluate(model, evaluation_loader, device)["loss"]
            transition = transition_from_opt2_(
                model, optimizer, epsilon=args.contraction_epsilon,
                opt2_done=opt2_done, soft_cap=args.max_opt2_epochs)
            criterion_met = transition.criterion_met
            if transition.phase == "train3":
                contraction_at_projection = transition.contraction_norm
                projection_loss_jump = (
                    evaluate(model, evaluation_loader, device)["loss"] - before)
                phase = "train3"
        scheduler.step()
        if args.method == "bypass" and epoch_phase in {"opt1", "opt2"}:
            expanded_seconds += time.perf_counter() - epoch_started
        peak_train_params = max(
            peak_train_params,
            sum(parameter.numel() for parameter in model.parameters()))
        validation = evaluate(model, evaluation_loader, device)
        row = {
            "epoch": epoch, "post_fork_epoch": offset + 1,
            "phase": epoch_phase, "train_loss": train["task_loss"],
            "train_accuracy": train["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "learning_rates": [float(group["lr"])
                               for group in optimizer.param_groups],
            "gamma": gamma, "contraction_criterion_met": criterion_met,
            "contraction_norm": (float(contraction_norm(model).detach())
                                 if args.method == "bypass" and
                                 phase in {"opt1", "opt2"} else 0.0),
        }
        history.append(row)
        elapsed = elapsed_before + time.perf_counter() - started
        peak = max(peak_before, int(torch.cuda.max_memory_allocated(device)))
        save_checkpoint(latest, {
            "kind": "plateau_fork_arm_progress", "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": rng_state(), "train_loader_generator_state":
                train_loader.generator.get_state(),
            "plateau_checkpoint_hash": fork_hash,
            "train_indices": train_indices,
            "trigger_indices": trigger_indices,
            "evaluation_indices": evaluation_indices,
            "source_tuning_indices": source_tuning,
            "history": history, "intervention": intervention,
            "post_fork_epoch": offset + 1, "phase": phase,
            "opt1_epochs": opt1_done, "opt2_epochs": opt2_done,
            "train3_epochs": train3_done, "opt2_steps": opt2_steps,
            "contraction_at_projection": contraction_at_projection,
            "projection_loss_jump": projection_loss_jump,
            "extension_paths": extension_paths,
            "time_spent_expanded_seconds": expanded_seconds,
            "peak_train_params": peak_train_params,
            "training_seconds": elapsed, "peak_gpu_memory": peak,
            "protocol": protocol,
        })
        print(json.dumps({args.method: row}, sort_keys=True), flush=True)

    last = history[-1]
    best_accuracy = max(row["validation_accuracy"] for row in history)
    result = {
        "method": args.method, "fork_epoch": fork_epoch,
        "stall_detected_epoch": source.get("stall_detected_epoch"),
        "post_fork_epochs": len(history),
        "plateau_checkpoint_hash": fork_hash,
        "fork_validation_accuracy": fork_evaluation["accuracy"],
        "fork_validation_loss": fork_evaluation["loss"],
        "final_validation_accuracy": last["validation_accuracy"],
        "best_validation_accuracy": best_accuracy,
        "final_validation_loss": last["validation_loss"],
        "best_validation_loss": min(row["validation_loss"] for row in history),
        "validation_accuracy_delta": (
            last["validation_accuracy"] - fork_evaluation["accuracy"]),
        "best_validation_accuracy_delta": (
            best_accuracy - fork_evaluation["accuracy"]),
        "epochs_to_best": next(
            row["post_fork_epoch"] for row in history
            if row["validation_accuracy"] == best_accuracy),
        "training_seconds": elapsed_before + time.perf_counter() - started,
        "peak_gpu_memory": max(
            peak_before, int(torch.cuda.max_memory_allocated(device))),
        "peak_train_params": peak_train_params,
        "deploy_params": sum(parameter.numel() for parameter in model.parameters()),
        "time_spent_expanded_seconds": expanded_seconds,
        "intervention": intervention,
        "bypass_completed": (phase == "train3" if args.method == "bypass" else None),
        "contraction_at_projection": contraction_at_projection,
        "projection_loss_jump": projection_loss_jump,
        "history": history, "checkpoint": str(latest), "protocol": protocol,
    }
    atomic_json_save(result, output / "result.json")


if __name__ == "__main__":
    main()
