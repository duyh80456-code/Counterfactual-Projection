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
    parser.add_argument("--site", default="auto")
    parser.add_argument(
        "--candidate-sites", default="",
        help="comma-separated TINY sites for --site=auto; empty scans all blocks")
    parser.add_argument(
        "--site-selection-mode",
        choices=("all_projected_utility", "fast_topk_projectability",
                 "tiny_score_argmax"),
        default="all_projected_utility")
    parser.add_argument("--selection-top-k", type=int, default=3)
    parser.add_argument("--selection-samples", type=int, default=16)
    parser.add_argument("--selection-cg-iterations", type=int, default=25)
    parser.add_argument("--selection-cg-relative-tolerance", type=float,
                        default=5e-2)
    parser.add_argument("--selection-preconditioner-probes", type=int,
                        default=2)
    parser.add_argument("--selection-damping", type=float, default=1e-3)
    parser.add_argument("--selection-min-utility", type=float, default=0.0)
    parser.add_argument("--selection-min-projectability", type=float,
                        default=0.05)
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
    count = (args.statistics_samples + args.selection_samples +
             args.projection_samples)
    selected = [train_indices[index] for index in order[:count]]
    statistics_indices = selected[:args.statistics_samples]
    selection_end = args.statistics_samples + args.selection_samples
    selection_indices = selected[args.statistics_samples:selection_end]
    projection_indices = selected[selection_end:]
    statistics_loader = DataLoader(
        Subset(eval_set, statistics_indices), args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    projection_loader = DataLoader(
        Subset(eval_set, projection_indices), args.projection_samples,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    selection_loader = DataLoader(
        Subset(eval_set, selection_indices), args.selection_samples,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    selection_batch = tuple(value.to(device, non_blocking=True)
                            for value in next(iter(selection_loader)))
    projection_batch = tuple(value.to(device, non_blocking=True)
                             for value in next(iter(projection_loader)))
    return list(statistics_loader), selection_batch, projection_batch


def structural_candidate_sites(model, site, candidate_sites=""):
    """Resolve the structural-E search space without sampling new data."""
    if site != "auto":
        return [site]
    requested = [value.strip() for value in candidate_sites.split(",")
                 if value.strip()]
    return requested or [ref.name for ref in model.growing_blocks()]


def propose_structural_candidates(model, statistics, *, rank, site,
                                  candidate_sites):
    """Run TINY at every requested site on one shared statistics batch set."""
    from dual_growth.adapters import TinyAdapter
    from dual_growth.controller import GrowthBudget

    sites = structural_candidate_sites(model, site, candidate_sites)
    if not sites:
        raise RuntimeError("structural-E candidate site set is empty")
    candidates = []
    for candidate_site in sites:
        adapter = TinyAdapter(
            quantum_params=10**9,
            max_statistics_batches=len(statistics))
        candidate = CounterfactualTinyProbe(rank, candidate_site).propose(
            adapter, model, statistics, GrowthBudget(10**9),
            sample_inputs=statistics[0][0])
        candidates.append(candidate)
    return candidates


def select_structural_candidate(model, statistics, *, rank, site,
                                candidate_sites, device):
    """Raw-TINY selector retained for fixed-site and argmax ablations."""
    synchronize(device)
    started = time.perf_counter()
    candidates = propose_structural_candidates(
        model, statistics, rank=rank, site=site,
        candidate_sites=candidate_sites)
    selected = max(
        candidates, key=lambda candidate: float(candidate.proposal_score))
    synchronize(device)
    selection = {
        "site_selection_mode": (
            "tiny_score_argmax" if site == "auto" else "fixed_site"),
        "selected_site": str(selected.module_name),
        "site_scores": {
            str(candidate.module_name): float(candidate.proposal_score)
            for candidate in candidates
        },
        "selected_site_score": float(selected.proposal_score),
        "site_selection_seconds": time.perf_counter() - started,
        "when_gate_passed": True,
        "when_gate_reason": "not_used_by_raw_tiny_ablation",
    }
    return selected, selection


@torch.no_grad()
def functional_loss_utility(descent_direction, functional_direction):
    """Mean negative-CE directional derivative in logit space."""
    if descent_direction.shape != functional_direction.shape:
        raise ValueError("utility directions must have the same shape")
    per_sample = (descent_direction.detach() *
                  functional_direction.detach()).flatten(1).sum(1)
    return float(per_sample.mean())


def select_projectability_aware_candidate(
        model, statistics, selection_batch, *, rank, site, candidate_sites,
        device, gate, top_k, cheap_projector, min_utility,
        min_projectability):
    """Rank candidates by projected loss utility, with an explicit WHEN gate.

    ``top_k=None`` evaluates every structural site and makes positive utility
    the only scientific intervention criterion. A finite ``top_k`` plus a
    projectability threshold is retained only as the fast approximation.
    """
    synchronize(device)
    started = time.perf_counter()
    candidates = propose_structural_candidates(
        model, statistics, rank=rank, site=site,
        candidate_sites=candidate_sites)
    ranked = sorted(
        candidates, key=lambda candidate: float(candidate.proposal_score),
        reverse=True)
    evaluated = (ranked if top_k is None
                 else ranked[:min(top_k, len(ranked))])
    if not evaluated:
        raise RuntimeError("structural selector produced no candidate")
    descent = supervised_functional_descent_direction(model, selection_batch)
    evaluations = {}
    viable = []
    for candidate in evaluated:
        site_name = str(candidate.module_name)
        signal = CandidateExpansionProbe()(
            model, candidate=candidate, batch=selection_batch, gate=gate)
        expansion_utility = functional_loss_utility(
            descent, signal.delta_logits)
        block = candidate_projection_block(model, candidate)
        parameter_names = candidate_projection_parameter_names(
            model, candidate, "residual_path")
        try:
            cheap = cheap_projector.project(
                model, selection_batch[0], signal.delta_logits, block=block,
                parameter_names=parameter_names)
            projected_utility = functional_loss_utility(
                descent, cheap.fitted_delta)
            projectability = float(cheap.fitted_norm_ratio)
            finite = bool(
                math.isfinite(projected_utility) and
                math.isfinite(projectability))
            evaluation = {
                "tiny_score": float(candidate.proposal_score),
                "expansion_utility": expansion_utility,
                "projectability_rho": projectability,
                "projected_utility": projected_utility,
                "cheap_relative_residual": float(cheap.relative_residual),
                "cheap_cosine_alignment": float(cheap.cosine_alignment),
                "cheap_cg_iterations": int(cheap.cg.iterations),
                "cheap_cg_converged": bool(cheap.cg.converged),
                "cheap_projection_finite": finite,
            }
            if finite:
                viable.append((projected_utility, candidate, evaluation))
        except RuntimeError as error:
            evaluation = {
                "tiny_score": float(candidate.proposal_score),
                "expansion_utility": expansion_utility,
                "projectability_rho": 0.0,
                "projected_utility": None,
                "cheap_projection_finite": False,
                "cheap_projection_error": str(error),
            }
        evaluations[site_name] = evaluation
    if viable:
        projected_utility, selected, selected_evaluation = max(
            viable, key=lambda item: item[0])
        projectability = selected_evaluation["projectability_rho"]
        utility_passed = projected_utility > min_utility
        projectability_passed = (
            True if min_projectability is None
            else projectability >= min_projectability)
        when_passed = utility_passed and projectability_passed
        if not utility_passed:
            reason = "projected_utility_not_above_threshold"
        elif not projectability_passed:
            reason = "projectability_below_threshold"
        else:
            reason = "usable_candidate_found"
    else:
        selected = evaluated[0]
        selected_evaluation = evaluations[str(selected.module_name)]
        projected_utility = None
        projectability = 0.0
        when_passed = False
        reason = "no_finite_cheap_projection"
    synchronize(device)
    selection = {
        "site_selection_mode": (
            "all_sites_projected_utility" if top_k is None
            else "tiny_topk_projectability_utility"),
        "selected_site": str(selected.module_name),
        "site_scores": {
            str(candidate.module_name): float(candidate.proposal_score)
            for candidate in candidates
        },
        "selected_site_score": float(selected.proposal_score),
        "prescreen_top_k_sites": (
            None if top_k is None else
            [str(candidate.module_name) for candidate in evaluated]),
        "cheap_projection_sites": [
            str(candidate.module_name) for candidate in evaluated],
        "site_functional_evaluations": evaluations,
        "selected_expansion_utility":
            selected_evaluation["expansion_utility"],
        "selected_projectability_rho": projectability,
        "selected_projected_utility": projected_utility,
        "when_min_utility": float(min_utility),
        "when_min_projectability": (
            None if min_projectability is None
            else float(min_projectability)),
        "when_gate_passed": when_passed,
        "when_gate_reason": reason,
        "site_selection_seconds": time.perf_counter() - started,
    }
    return selected, selection


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
    if args.method == "o_projection_only" and args.site == "auto":
        raise ValueError("o_projection_only requires one concrete --site")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint, shared_hash = load_shared_checkpoint(
        Path(args.shared_checkpoint), args.shared_checkpoint_hash,
        device=device, model=model, optimizer=optimizer, scheduler=scheduler)
    if checkpoint["protocol"] != run_protocol:
        raise RuntimeError("shared checkpoint protocol differs from arm protocol")
    arm_protocol = {**run_protocol, "method": args.method}
    if args.method in {"ours_e_driven_o", "o_projection_only"}:
        selection_mode = (
            "fixed_site" if args.site != "auto" else
            ({
                "all_projected_utility": "all_sites_projected_utility",
                "fast_topk_projectability":
                    "tiny_topk_projectability_utility",
                "tiny_score_argmax": "tiny_score_argmax",
            }[args.site_selection_mode]))
        all_sites_main = selection_mode == "all_sites_projected_utility"
        arm_protocol["functional_projection"] = {
            "site": args.site, "rank": args.rank,
            "candidate_sites": args.candidate_sites,
            "site_selection_mode": selection_mode,
            "selection_top_k": (
                None if all_sites_main else args.selection_top_k),
            "selection_samples": args.selection_samples,
            "selection_cg_iterations": args.selection_cg_iterations,
            "selection_cg_relative_tolerance":
                args.selection_cg_relative_tolerance,
            "selection_preconditioner_probes":
                args.selection_preconditioner_probes,
            "selection_damping": args.selection_damping,
            "selection_min_utility": args.selection_min_utility,
            "selection_min_projectability": (
                None if all_sites_main else
                args.selection_min_projectability),
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
    cheap_projector = FunctionalProjector(
        args.selection_damping, args.selection_cg_iterations,
        tolerance=args.selection_cg_relative_tolerance,
        max_damping_retries=0,
        preconditioner_probes=args.selection_preconditioner_probes)
    e_projection = EProjection(projector=projector)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for post_epoch in range(start_epoch, POST_FORK_EPOCHS):
        diagnostics = None
        global_epoch = FORK_EPOCH + post_epoch
        if args.method == "ours_e_driven_o":
            statistics, selection_batch, projection_batch = intervention_batches(
                eval_set, train_indices, args, global_epoch, device)
            if (args.site == "auto" and args.site_selection_mode in {
                    "all_projected_utility", "fast_topk_projectability"}):
                main_all_sites = (
                    args.site_selection_mode == "all_projected_utility")
                candidate, site_selection = select_projectability_aware_candidate(
                    model, statistics, selection_batch, rank=args.rank,
                    site=args.site, candidate_sites=args.candidate_sites,
                    device=device, gate=args.probe_epsilon,
                    top_k=None if main_all_sites else args.selection_top_k,
                    cheap_projector=cheap_projector,
                    min_utility=args.selection_min_utility,
                    min_projectability=(
                        None if main_all_sites else
                        args.selection_min_projectability))
            else:
                candidate, site_selection = select_structural_candidate(
                    model, statistics, rank=args.rank, site=args.site,
                    candidate_sites=args.candidate_sites, device=device)
            if not site_selection["when_gate_passed"]:
                loss = batch_loss(model, tuning_batch)
                diagnostics = {
                    "source": "structural_TINY_E_when_gate_skipped",
                    "correction_applied": False,
                    "parameter_delta_norm": 0.0,
                    "actual_cosine_alignment": None,
                    "actual_relative_residual": None,
                    "loss_before": loss,
                    "loss_after": loss,
                    "momentum_states_reset": 0,
                    "projection_seconds": 0.0,
                    "heldout_evaluation_seconds": 0.0,
                    **site_selection,
                }
            else:
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
                    "projection_seconds":
                        time.perf_counter() - projection_started,
                    "heldout_evaluation_seconds": heldout_seconds,
                    **site_selection,
                    **heldout_metrics(heldout),
                    **cg_diagnostics(step.projection),
                }
        elif args.method == "o_projection_only":
            _, _, projection_batch = intervention_batches(
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
    if args.method == "o_projection_only":
        result.update({
            "method_label": "projection_only_supervised_control",
            "control_type": "supervised_functional_projection_control",
            "functional_target": (
                "negative summed-CE logit gradient: "
                "one_hot(y) - softmax(f_theta(x))"),
            "uses_structural_E": False,
        })
    if args.method == "ours_e_driven_o":
        result_selection_mode = (
            "fixed_site" if args.site != "auto" else
            ({
                "all_projected_utility": "all_sites_projected_utility",
                "fast_topk_projectability":
                    "tiny_topk_projectability_utility",
                "tiny_score_argmax": "tiny_score_argmax",
            }[args.site_selection_mode]))
        result.update({
            "method_label": "e_driven_o_when_where_how",
            "site_selection_mode": result_selection_mode,
            "when_gate_pass_rate": sum(
                bool(row["diagnostics"].get("when_gate_passed"))
                for row in history) / len(history),
            "full_projection_attempt_rate": sum(
                bool(row["diagnostics"].get("when_gate_passed"))
                for row in history) / len(history),
            "site_selection_history": [
                {
                    "epoch": row["epoch"],
                    "selected_site": row["diagnostics"]["selected_site"],
                    "selected_site_score":
                        row["diagnostics"]["selected_site_score"],
                    "site_scores": row["diagnostics"]["site_scores"],
                    "site_selection_seconds":
                        row["diagnostics"]["site_selection_seconds"],
                    "prescreen_top_k_sites":
                        row["diagnostics"].get("prescreen_top_k_sites"),
                    "cheap_projection_sites":
                        row["diagnostics"].get("cheap_projection_sites"),
                    "site_functional_evaluations": row["diagnostics"].get(
                        "site_functional_evaluations"),
                    "selected_expansion_utility": row["diagnostics"].get(
                        "selected_expansion_utility"),
                    "selected_projectability_rho": row["diagnostics"].get(
                        "selected_projectability_rho"),
                    "selected_projected_utility": row["diagnostics"].get(
                        "selected_projected_utility"),
                    "when_gate_passed":
                        row["diagnostics"].get("when_gate_passed"),
                    "when_gate_reason":
                        row["diagnostics"].get("when_gate_reason"),
                }
                for row in history
            ],
        })
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
    if (args.selection_top_k < 1 or args.selection_samples < 1 or
            args.selection_cg_iterations < 1 or
            args.selection_preconditioner_probes < 0 or
            args.selection_damping < 0 or
            not 0 < args.selection_cg_relative_tolerance < 1 or
            args.selection_min_projectability < 0):
        raise ValueError("invalid projectability-aware selection configuration")
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
