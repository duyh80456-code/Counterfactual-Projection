"""Run projected controls or Bypass from a stalled Vanilla best."""

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
from experiments.plateau_protocol import scheduler_from_state
from experiments.run_gromo_pilot import (
    actual_update_metrics, batch_loss, cg_diagnostics, eval_logits,
    evaluate_heldout_direction, heldout_metrics, parameter_delta_norm,
    preview_projected_gain, reset_projected_momentum)
from experiments.run_plateau_comparison import finite_projection, run_intervention
from experiments.run_shared_comparison import (
    supervised_functional_descent_direction)
from experiments.shared_protocol import (
    architecture_label, atomic_json_save, atomic_torch_save,
    build_cifar_gromo_resnet, build_optimizer_scheduler,
    datasets_and_indices, evaluate,
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
    parser.add_argument("--post-fork-epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--architecture",
        choices=("resnet18", "resnet34", "vgg16", "densenet121"),
        default="resnet18")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--site", default="stages.2.blocks.0")
    parser.add_argument(
        "--site-selection-mode", choices=("all_functional_gain",),
        default="all_functional_gain")
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
    parser.add_argument("--retrigger-patience", type=int, default=10)
    parser.add_argument("--opt1-epochs", type=int, default=70)
    parser.add_argument("--max-opt2-epochs", type=int, default=30)
    parser.add_argument("--contraction-epsilon", type=float, default=0.002)
    parser.add_argument("--gamma-slope", type=float, default=3e-6)
    parser.add_argument("--gamma-increase-opt2-epoch", type=int, default=15)
    parser.add_argument("--gamma-post-increase-multiplier", type=float, default=2.0)
    return parser.parse_args()


def save_checkpoint(path, payload):
    atomic_torch_save({"format_version": 1, **payload}, path)


def run_o_only_intervention(model, optimizer, eval_set, train_indices,
                            args, device, probe_index=0):
    """One supervised projection-only control at theta_P."""
    generator = torch.Generator().manual_seed(
        911_731 + args.seed * 10_000 + int(probe_index) * 1_009)
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
    if args.retrigger_patience < 1:
        raise ValueError("retrigger_patience must be positive")
    if (args.method == "bypass" and
            args.opt1_epochs + args.max_opt2_epochs > args.post_fork_epochs):
        raise ValueError("Bypass opt1/opt2 exceed the post-fork budget")
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

    expected_architecture = architecture_label(args.architecture)
    source_architecture = source.get("protocol", {}).get("architecture")
    if source_architecture != expected_architecture:
        raise RuntimeError(
            "plateau checkpoint architecture mismatch: "
            f"{source_architecture!r} != {expected_architecture!r}")
    model = build_cifar_gromo_resnet(args.architecture, device)
    fork_deploy_params = sum(
        parameter.numel() for parameter in model.parameters())
    optimizer, _ = build_optimizer_scheduler(model, 0.1, args.weight_decay)
    model.load_state_dict(source["model"], strict=True)
    optimizer.load_state_dict(source["optimizer"])
    scheduler = scheduler_from_state(optimizer, source["scheduler"])
    fork_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    restore_rng(source["rng"])
    train_loader = make_train_loader(
        train_set, train_indices, args.batch_size, args.workers,
        source["train_loader_generator_state"], args.seed)
    trigger_loader = make_eval_loader(
        eval_set, trigger_indices, args.batch_size * 2, args.workers)
    evaluation_loader = make_eval_loader(
        eval_set, evaluation_indices, args.batch_size * 2, args.workers)
    fork_trigger = evaluate(model, trigger_loader, device)
    fork_evaluation = evaluate(model, evaluation_loader, device)

    protocol = {
        "phase": "plateau_fork_comparison", "method": args.method,
        "architecture": expected_architecture,
        "fork_epoch": fork_epoch, "post_fork_epochs": args.post_fork_epochs,
        "stall_detected_epoch": source.get("stall_detected_epoch"),
        "plateau_checkpoint_hash": fork_hash,
        "theta_best_hash": fork_hash,
        "selection_metric": "validation accuracy (3,000-sample split)",
        "evaluation_role": "model selection and reporting (3,000 samples)",
        "optimizer_state_preserved": True,
        "scheduler_state_preserved": True,
        "training_indices_unchanged": True,
        "official_test_used": False,
        "intervention_schedule": (
            {"mode": "recurrent_best_rollback",
             "patience": args.retrigger_patience,
             "metric": "strict raw validation best",
             "site_selection_mode": args.site_selection_mode,
             "sgd_epoch_budget": args.post_fork_epochs,
             "rollback_rng": False,
             "rollback_loader_stream": False}
            if args.method == "ours_e_driven_o" else
            {"mode": "single_initial_intervention",
             "patience": None,
             "metric": None,
             "sgd_epoch_budget": args.post_fork_epochs,
             "rollback_rng": False,
             "rollback_loader_stream": False}
            if args.method == "o_projection_only" else None),
        "o_projection_site": (
            args.site if args.method == "o_projection_only" else None),
        "bypass_schedule": ({
            "opt1_epochs": args.opt1_epochs,
            "max_opt2_epochs": args.max_opt2_epochs,
            "force_projection": False,
            "variant": "scaled_matched_horizon_not_exact_reproduction",
            "gamma_increase_opt2_epoch": args.gamma_increase_opt2_epoch,
            "gamma_post_increase_multiplier":
                args.gamma_post_increase_multiplier,
        } if args.method == "bypass" else None),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "checkpoint_latest.pt"
    best_checkpoint = output / "checkpoint_best.pt"
    saved = (torch.load(latest, map_location=device, weights_only=False)
             if latest.is_file() else None)
    saved_best = (torch.load(
        best_checkpoint, map_location=device, weights_only=False)
        if best_checkpoint.is_file() else None)
    if (saved_best is not None and
            (saved is None or int(saved_best.get("post_fork_epoch", -1)) >
             int(saved.get("post_fork_epoch", -1)))):
        # Best is written before latest. If a crash lands between the two
        # atomic writes, the newer best payload is itself a complete progress
        # checkpoint and can safely repair latest.
        saved = saved_best
        save_checkpoint(latest, saved)
    if saved is not None and saved["protocol"] != protocol:
        raise RuntimeError("plateau-fork resume protocol mismatch")
    if saved is not None and not best_checkpoint.is_file():
        raise RuntimeError(
            "resume requires checkpoint_best.pt beside checkpoint_latest.pt")

    phase = "train"
    extension_paths = []
    opt1_done = opt2_done = train3_done = opt2_steps = 0
    contraction_at_projection = projection_loss_jump = None
    budget_exhausted_before_contraction = False
    expanded_seconds = 0.0
    peak_train_params = sum(parameter.numel() for parameter in model.parameters())
    if args.method == "bypass":
        phase = saved["phase"] if saved is not None else "opt1"
        if phase in {"opt1", "opt2"}:
            embed_started = time.perf_counter()
            extension_paths = embed_relaxed_bypass(model)
            add_extension_parameters_(optimizer, extension_parameters(model))
            scheduler.sync_optimizer_groups()
            expanded_seconds += time.perf_counter() - embed_started
            # Extension coordinates join the existing optimizer group, so the
            # inherited scheduler continues unchanged without a restart.

    history = []
    intervention = None
    interventions = []
    rollback_count = 0
    stall_counter = 0
    start_offset = 0
    elapsed_before = 0.0
    peak_before = 0
    best_accuracy = None
    best_loss = None
    best_offset = None
    exact_best_validation_accuracy = None
    if saved is not None:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        restore_rng(saved["rng"])
        train_loader.generator.set_state(
            saved["train_loader_generator_state"].cpu())
        history = saved["history"]
        intervention = saved.get("intervention")
        interventions = list(saved.get(
            "interventions", [intervention] if intervention else []))
        rollback_count = int(saved.get("rollback_count", 0))
        stall_counter = int(saved.get("stall_counter", 0))
        start_offset = int(saved["post_fork_epoch"])
        phase = saved["phase"]
        opt1_done = int(saved.get("opt1_epochs", 0))
        opt2_done = int(saved.get("opt2_epochs", 0))
        train3_done = int(saved.get("train3_epochs", 0))
        opt2_steps = int(saved.get("opt2_steps", 0))
        contraction_at_projection = saved.get("contraction_at_projection")
        projection_loss_jump = saved.get("projection_loss_jump")
        budget_exhausted_before_contraction = bool(
            saved.get("budget_exhausted_before_contraction", False))
        extension_paths = saved.get("extension_paths", extension_paths)
        expanded_seconds = float(saved.get("time_spent_expanded_seconds", 0.0))
        peak_train_params = int(saved.get("peak_train_params", peak_train_params))
        elapsed_before = float(saved.get("training_seconds", 0.0))
        peak_before = int(saved.get("peak_gpu_memory", 0))
        best_accuracy = float(saved["best_validation_accuracy"])
        best_loss = float(saved["best_validation_loss"])
        best_offset = int(saved["best_post_fork_epoch"])
        exact_best_validation_accuracy = float(
            saved["exact_best_validation_accuracy"])

    def state_payload(kind, post_offset, training_seconds, peak_memory):
        return {
            "kind": kind, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": rng_state(), "train_loader_generator_state":
                train_loader.generator.get_state(),
            "plateau_checkpoint_hash": fork_hash,
            "theta_best_hash": fork_hash,
            "train_indices": train_indices,
            "trigger_indices": trigger_indices,
            "evaluation_indices": evaluation_indices,
            "source_tuning_indices": source_tuning,
            "history": history, "intervention": intervention,
            "interventions": interventions,
            "rollback_count": rollback_count,
            "stall_counter": stall_counter,
            "post_fork_epoch": post_offset, "phase": phase,
            "opt1_epochs": opt1_done, "opt2_epochs": opt2_done,
            "train3_epochs": train3_done, "opt2_steps": opt2_steps,
            "contraction_at_projection": contraction_at_projection,
            "projection_loss_jump": projection_loss_jump,
            "budget_exhausted_before_contraction":
                budget_exhausted_before_contraction,
            "extension_paths": extension_paths,
            "time_spent_expanded_seconds": expanded_seconds,
            "peak_train_params": peak_train_params,
            "training_seconds": training_seconds,
            "peak_gpu_memory": peak_memory,
            "best_validation_accuracy": best_accuracy,
            "best_validation_loss": best_loss,
            "best_post_fork_epoch": best_offset,
            "exact_best_validation_accuracy": exact_best_validation_accuracy,
            "protocol": protocol,
        }

    def perform_intervention(probe_index, post_offset, reason):
        nonlocal intervention
        pre_probe_rng = rng_state()
        intervention_started = time.perf_counter()
        try:
            if args.method == "ours_e_driven_o":
                intervention = run_intervention(
                    model, optimizer, eval_set, train_indices, args, device,
                    probe_index)
            else:
                intervention = run_o_only_intervention(
                    model, optimizer, eval_set, train_indices, args, device,
                    probe_index)
        finally:
            restore_rng(pre_probe_rng)
        intervention["epoch"] = fork_epoch + post_offset
        intervention["post_fork_epoch"] = post_offset
        intervention["probe_index"] = probe_index
        intervention["trigger_reason"] = reason
        intervention["intervention_seconds"] = (
            time.perf_counter() - intervention_started)
        interventions.append(dict(intervention))
        if args.method == "ours_e_driven_o":
            print(json.dumps({"e_driven_o_intervention": intervention},
                             sort_keys=True), flush=True)
        post_path = output / f"checkpoint_post_intervention_{probe_index:03d}.pt"
        save_checkpoint(post_path, state_payload(
            "plateau_post_intervention", post_offset,
            elapsed_before, peak_before))

    # The pre-intervention theta_P is a valid global best. Save it first so a
    # harmful intervention can roll back to the actual fork rather than to its
    # perturbed state.
    if saved is None:
        best_accuracy = float(fork_evaluation["accuracy"])
        best_loss = float(fork_evaluation["loss"])
        best_offset = 0
        exact_best_validation_accuracy = float(fork_evaluation["accuracy"])
        save_checkpoint(best_checkpoint, state_payload(
            "plateau_fork_arm_best", 0, 0.0, 0))
        if args.method in {"ours_e_driven_o", "o_projection_only"}:
            perform_intervention(0, 0, "initial_theta_P")
            immediate_trigger = evaluate(model, trigger_loader, device)
            immediate = evaluate(model, evaluation_loader, device)
            if immediate["accuracy"] > best_accuracy:
                best_accuracy = float(immediate["accuracy"])
                best_offset = 0
            best_loss = min(best_loss, float(immediate["loss"]))
            immediate_exact = (
                immediate["accuracy"] > exact_best_validation_accuracy)
            if immediate_exact:
                exact_best_validation_accuracy = float(immediate["accuracy"])
                stall_counter = 0
            if immediate_exact:
                save_checkpoint(best_checkpoint, state_payload(
                    "plateau_fork_arm_best", 0, 0.0, 0))

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for offset in range(start_offset, args.post_fork_epochs):
        if args.method == "bypass" and phase == "incomplete":
            break
        epoch = fork_epoch + offset + 1
        epoch_phase = phase
        epoch_started = time.perf_counter()
        gamma = 0.0
        gamma_multiplier = 1.0
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
                nonlocal opt2_steps, gamma, gamma_multiplier
                opt2_steps += 1
                opt2_epoch = opt2_done + 1
                gamma_multiplier = (
                    args.gamma_post_increase_multiplier
                    if opt2_epoch >= args.gamma_increase_opt2_epoch else 1.0)
                gamma = args.gamma_slope * opt2_steps * gamma_multiplier
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
            elif opt2_done >= args.max_opt2_epochs:
                # The method is not allowed to spend the train3 allocation in
                # opt2 or to force a projection. Without contraction there is
                # no valid compact train3 state, so this arm ends as an
                # explicitly incomplete diagnostic comparator.
                budget_exhausted_before_contraction = True
                phase = "incomplete"
        if args.method == "bypass" and epoch_phase in {"opt1", "opt2"}:
            expanded_seconds += time.perf_counter() - epoch_started
        peak_train_params = max(
            peak_train_params,
            sum(parameter.numel() for parameter in model.parameters()))
        trigger = evaluate(model, trigger_loader, device)
        validation = evaluate(model, evaluation_loader, device)
        # The inherited base scheduler is metric-independent. Trigger metrics
        # are diagnostic only and must never alter the optimization path.
        scheduler.step()
        deploy_compact = args.method != "bypass" or phase == "train3"
        row = {
            "epoch": epoch, "post_fork_epoch": offset + 1,
            "phase": epoch_phase, "train_loss": train["task_loss"],
            "train_accuracy": train["accuracy"],
            "trigger_loss": trigger["loss"],
            "trigger_accuracy": trigger["accuracy"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "learning_rates": [float(group["lr"])
                               for group in optimizer.param_groups],
            "gamma": gamma, "contraction_criterion_met": criterion_met,
            "deploy_compact": deploy_compact,
            "gamma_multiplier": gamma_multiplier,
            "contraction_norm": (float(contraction_norm(model).detach())
                                 if args.method == "bypass" and
                                 phase in {"opt1", "opt2"} else 0.0),
        }
        history.append(row)
        elapsed = elapsed_before + time.perf_counter() - started
        peak = max(peak_before, int(torch.cuda.max_memory_allocated(device)))
        report_improved = validation["accuracy"] > best_accuracy
        exact_improved = (
            validation["accuracy"] > exact_best_validation_accuracy)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = float(validation["accuracy"])
            best_offset = offset + 1
        best_loss = min(best_loss, float(validation["loss"]))
        if exact_improved and args.method == "ours_e_driven_o":
            exact_best_validation_accuracy = float(validation["accuracy"])
            stall_counter = 0
        elif args.method == "ours_e_driven_o":
            stall_counter += 1
        elif exact_improved:
            exact_best_validation_accuracy = float(validation["accuracy"])
            stall_counter = 0
        if exact_improved:
            save_checkpoint(best_checkpoint, state_payload(
                "plateau_fork_arm_best", offset + 1, elapsed, peak))

        retriggered = False
        if (args.method == "ours_e_driven_o" and
                stall_counter >= args.retrigger_patience and
                offset + 1 < args.post_fork_epochs):
            # Roll back trainable state but deliberately keep the consumed RNG
            # and loader streams. Restoring them would replay the same E probe
            # and the same ten SGD epochs forever.
            live_rng = rng_state()
            live_loader_state = train_loader.generator.get_state().clone()
            best_state = torch.load(
                best_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(best_state["model"], strict=True)
            optimizer.load_state_dict(best_state["optimizer"])
            scheduler.load_state_dict(best_state["scheduler"])
            restore_rng(live_rng)
            train_loader.generator.set_state(live_loader_state.cpu())
            rollback_count += 1
            stall_counter = 0
            perform_intervention(
                len(interventions), offset + 1,
                "raw_validation_stall")
            immediate_trigger = evaluate(model, trigger_loader, device)
            immediate = evaluate(model, evaluation_loader, device)
            if immediate["accuracy"] > best_accuracy:
                best_accuracy = float(immediate["accuracy"])
                best_offset = offset + 1
            best_loss = min(best_loss, float(immediate["loss"]))
            immediate_exact = (
                immediate["accuracy"] > exact_best_validation_accuracy)
            if immediate_exact:
                exact_best_validation_accuracy = float(immediate["accuracy"])
                stall_counter = 0
            if immediate_exact:
                save_checkpoint(best_checkpoint, state_payload(
                    "plateau_fork_arm_best", offset + 1, elapsed, peak))
            retriggered = True
        row["report_best_improved"] = report_improved
        row["exact_best_improved"] = exact_improved
        row["raw_validation_best_improved"] = exact_improved
        row["stall_counter"] = stall_counter
        row["rollback_triggered"] = retriggered
        row["intervention_count"] = len(interventions)
        save_checkpoint(latest, state_payload(
            "plateau_fork_arm_progress", offset + 1, elapsed, peak))
        print(json.dumps({args.method: row}, sort_keys=True), flush=True)
        if args.method == "bypass" and phase == "incomplete":
            break

    last = history[-1]
    expanded_rows = ([row for row in history
                      if not bool(row.get("deploy_compact", True))]
                     if args.method == "bypass" else [])
    compact_rows = ([row for row in history
                     if bool(row.get("deploy_compact", False))]
                    if args.method == "bypass" else history)
    bypass_completed = phase == "train3" if args.method == "bypass" else None
    compact_best_row = (
        max(compact_rows,
            key=lambda row: float(row["validation_accuracy"]))
        if args.method == "bypass" and compact_rows else None)
    compact_best_loss = (
        min(float(row["validation_loss"]) for row in compact_rows)
        if args.method == "bypass" and compact_rows else None)
    reported_best_accuracy = (
        (float(compact_best_row["validation_accuracy"])
         if compact_best_row is not None else None)
        if args.method == "bypass" else best_accuracy)
    reported_best_loss = (
        compact_best_loss if args.method == "bypass" else best_loss)
    reported_best_offset = (
        (int(compact_best_row["post_fork_epoch"])
         if compact_best_row is not None else None)
        if args.method == "bypass" else best_offset)
    result = {
        "method": args.method, "fork_epoch": fork_epoch,
        "stall_detected_epoch": source.get("stall_detected_epoch"),
        "post_fork_epochs": len(history),
        "plateau_checkpoint_hash": fork_hash,
        "theta_best_hash": fork_hash,
        "fork_trigger_accuracy": fork_trigger["accuracy"],
        "fork_trigger_loss": fork_trigger["loss"],
        "exact_best_validation_accuracy": exact_best_validation_accuracy,
        "fork_validation_accuracy": fork_evaluation["accuracy"],
        "fork_validation_loss": fork_evaluation["loss"],
        "final_validation_accuracy": last["validation_accuracy"],
        "final_validation_accuracy_space": (
            "expanded_diagnostic"
            if args.method == "bypass" and not bypass_completed
            else "compact_deploy"),
        "best_validation_accuracy": reported_best_accuracy,
        "best_validation_accuracy_space": "compact_deploy",
        "final_validation_loss": last["validation_loss"],
        "best_validation_loss": reported_best_loss,
        "validation_accuracy_delta": (
            last["validation_accuracy"] - fork_evaluation["accuracy"]),
        "best_validation_accuracy_delta": (
            reported_best_accuracy - fork_evaluation["accuracy"]
            if reported_best_accuracy is not None else None),
        "epochs_to_best": reported_best_offset,
        "training_seconds": elapsed_before + time.perf_counter() - started,
        "peak_gpu_memory": max(
            peak_before, int(torch.cuda.max_memory_allocated(device))),
        "peak_train_params": peak_train_params,
        "deploy_params": (
            sum(parameter.numel() for parameter in model.parameters())
            if args.method != "bypass" or bypass_completed else None),
        "current_params": sum(
            parameter.numel() for parameter in model.parameters()),
        "fork_deploy_params": fork_deploy_params,
        "time_spent_expanded_seconds": expanded_seconds,
        "intervention": intervention, "interventions": interventions,
        "intervention_count": len(interventions),
        "correction_application_count": sum(
            bool(item.get("correction_applied")) for item in interventions),
        "correction_application_rate": (
            sum(bool(item.get("correction_applied")) for item in interventions) /
            len(interventions) if interventions else None),
        "intervention_seconds": sum(
            float(item.get("intervention_seconds", 0.0))
            for item in interventions),
        "rollback_count": rollback_count,
        "retrigger_patience": (
            args.retrigger_patience
            if args.method == "ours_e_driven_o" else None),
        "retrigger_metric": (
            "strict raw validation best"
            if args.method == "ours_e_driven_o" else None),
        "bypass_completed": bypass_completed,
        "bypass_comparison_eligible": (
            bypass_completed if args.method == "bypass" else None),
        "budget_exhausted_before_contraction": (
            budget_exhausted_before_contraction
            if args.method == "bypass" else None),
        "unused_post_fork_epoch_budget": (
            args.post_fork_epochs - len(history)
            if args.method == "bypass" and not bypass_completed else 0),
        "expanded_best_validation_accuracy": (
            max(float(row["validation_accuracy"]) for row in expanded_rows)
            if expanded_rows else None),
        "expanded_final_validation_accuracy": (
            float(expanded_rows[-1]["validation_accuracy"])
            if expanded_rows else None),
        "compact_best_validation_accuracy": (
            float(compact_best_row["validation_accuracy"])
            if args.method == "bypass" and compact_best_row is not None
            else None),
        "compact_final_validation_accuracy": (
            float(last["validation_accuracy"])
            if args.method == "bypass" and bypass_completed else None),
        "opt1_epochs": opt1_done if args.method == "bypass" else None,
        "opt2_epochs": opt2_done if args.method == "bypass" else None,
        "train3_epochs": train3_done if args.method == "bypass" else None,
        "gamma_increase_opt2_epoch": (
            args.gamma_increase_opt2_epoch if args.method == "bypass" else None),
        "gamma_post_increase_multiplier": (
            args.gamma_post_increase_multiplier
            if args.method == "bypass" else None),
        "contraction_at_projection": contraction_at_projection,
        "projection_loss_jump": projection_loss_jump,
        "history": history, "checkpoint": str(latest),
        "best_checkpoint": str(best_checkpoint), "protocol": protocol,
    }
    atomic_json_save(result, output / "result.json")


if __name__ == "__main__":
    main()
