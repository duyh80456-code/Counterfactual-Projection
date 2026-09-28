"""Fair full-width structural-E CIFAR-100 pilot on one visible GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from baselines import ExpandedTrainProject, RealEGrowth
from methods import EProjection
from methods.e_projection import candidate_projection_parameter_names
from probe import (
    CandidateExpansionProbe, CounterfactualTinyProbe,
    build_pretrained_gromo_resnet18)
from projection import (
    FunctionalProjector, ProjectionResult, cosine_alignment,
    fitted_norm_ratio, relative_residual)


METHODS = (
    "vanilla", "vanilla_matched_compute", "vanilla_extra_sgd",
    "vanilla_momentum_reset", "random_projection", "tiny_projection",
    "sign_randomized_projection",
    "tiny_projection_conv_only", "tiny_projection_whole_block",
    "expand_train_project", "real_e_growth", "ours_e_driven_o")
FULL_WIDTHS = [64, 64, 128, 128, 256, 256, 512, 512]
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 5e-4


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def index_sha256(indices) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(int(index).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3,
                        help="post-warm-up intervention/training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--warmup-checkpoint", default="")
    parser.add_argument("--prepare-warmup", action="store_true")
    parser.add_argument("--evaluate-official-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--statistics-samples", type=int, default=256)
    parser.add_argument("--projection-samples", type=int, default=32)
    parser.add_argument("--tuning-samples", "--check-samples",
                        dest="tuning_samples", type=int, default=128)
    parser.add_argument("--site", default="stages.2.blocks.0")
    parser.add_argument(
        "--candidate-sites", default="",
        help="comma-separated sites used when --site=auto; empty probes all")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--probe-epsilon", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--cg-iterations", type=int, default=200)
    parser.add_argument("--cg-relative-tolerance", type=float, default=1e-2)
    parser.add_argument("--cg-preconditioner-probes", type=int, default=8)
    parser.add_argument(
        "--solver-revision",
        default="e-driven-o-best-functional-fit-v1")
    parser.add_argument("--application-max-heldout-residual", type=float,
                        default=1.0)
    parser.add_argument("--application-min-heldout-cosine", type=float,
                        default=0.0)
    parser.add_argument("--projection-scale", type=float, default=0.0,
                        help="0 applies the fitted direction at probe epsilon")
    parser.add_argument("--expanded-train-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    return parser.parse_args()


def validate_arguments(args) -> None:
    if min(args.statistics_samples, args.projection_samples,
           args.tuning_samples, args.validation_samples) < 1:
        raise ValueError("validation/statistics/projection/tuning sizes must be positive")
    if args.validation_samples + args.tuning_samples >= 50000:
        raise ValueError("held-out splits leave no CIFAR-100 training examples")
    available = 50000 - args.validation_samples - args.tuning_samples
    used = available if args.train_samples == 0 else min(args.train_samples, available)
    if args.statistics_samples + args.projection_samples > used:
        raise ValueError("one intervention needs more examples than the training pool")
    if not 0 < args.probe_epsilon <= 1:
        raise ValueError("probe epsilon must be in (0, 1]")
    if args.projection_scale < 0 or args.warmup_epochs < 0:
        raise ValueError("projection scale and warm-up epochs must be non-negative")
    if not 0 < args.cg_relative_tolerance < 1:
        raise ValueError("CG relative tolerance must be in (0, 1)")
    if args.cg_preconditioner_probes < 0:
        raise ValueError("CG preconditioner probes must be non-negative")
    if args.application_max_heldout_residual <= 0:
        raise ValueError("application held-out residual threshold must be positive")
    if not -1 <= args.application_min_heldout_cosine <= 1:
        raise ValueError("application held-out cosine threshold must be in [-1, 1]")
    if args.prepare_warmup and not args.warmup_checkpoint:
        raise ValueError("--prepare-warmup requires --warmup-checkpoint")


def datasets_and_indices(args):
    from torchvision import datasets, transforms

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_transform = transforms.Compose([
        transforms.Resize(args.image_size + 16),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR100(
        args.data_root, train=True, transform=train_transform, download=False)
    eval_set = datasets.CIFAR100(
        args.data_root, train=True, transform=eval_transform, download=False)
    test_set = (datasets.CIFAR100(
        args.data_root, train=False, transform=eval_transform, download=False)
        if args.evaluate_official_test else None)
    order = torch.randperm(
        len(train_set), generator=torch.Generator().manual_seed(20260928)).tolist()
    validation_indices = order[:args.validation_samples]
    tuning_start = args.validation_samples
    tuning_indices = order[tuning_start:tuning_start + args.tuning_samples]
    train_indices = order[tuning_start + args.tuning_samples:]
    if args.train_samples:
        train_indices = train_indices[:args.train_samples]
    return (train_set, eval_set, test_set, train_indices,
            validation_indices, tuning_indices)


def make_train_loader(dataset, indices, args):
    return DataLoader(
        Subset(dataset, indices), args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers, pin_memory=True, persistent_workers=False)


def make_eval_loader(dataset, indices, batch_size, args):
    return DataLoader(
        Subset(dataset, indices), batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)


def intervention_batches(eval_set, train_indices, args, intervention: int,
                         device):
    """Draw fresh disjoint B_stats/B_projection from the shared train pool."""
    generator = torch.Generator().manual_seed(
        81_337 + args.seed * 10_000 + intervention)
    order = torch.randperm(len(train_indices), generator=generator).tolist()
    selected = [train_indices[index] for index in order[
        :args.statistics_samples + args.projection_samples]]
    statistics_indices = selected[:args.statistics_samples]
    projection_indices = selected[args.statistics_samples:]
    statistics_loader = DataLoader(
        Subset(eval_set, statistics_indices), args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    projection_loader = DataLoader(
        Subset(eval_set, projection_indices), args.projection_samples,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    projection_batch = tuple(
        value.to(device, non_blocking=True)
        for value in next(iter(projection_loader)))
    audit = {
        "statistics_indices_sha256": index_sha256(statistics_indices),
        "projection_indices_sha256": index_sha256(projection_indices),
        "statistics_projection_overlap": len(
            set(statistics_indices) & set(projection_indices)),
    }
    return statistics_loader, projection_batch, audit


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss_sum += float(F.cross_entropy(logits.float(), targets, reduction="sum"))
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return loss_sum / count, correct / count


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = correct = count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * targets.numel()
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return loss_sum / count, correct / count


def make_optimizer(model, args):
    return torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=SGD_MOMENTUM,
        weight_decay=SGD_WEIGHT_DECAY)


def reset_projected_momentum(optimizer, model, projection: ProjectionResult) -> int:
    """Clear stale SGD state only for coordinates changed by a direct jump."""
    parameters = dict(model.named_parameters())
    reset = 0
    for name in projection.parameter_delta:
        parameter = parameters[name]
        if parameter in optimizer.state:
            optimizer.state.pop(parameter)
            reset += 1
    return reset


def reset_residual_path_momentum(optimizer, model, block_name: str) -> int:
    """Match the main conv+BN residual-path projection coordinates."""
    candidate = type("Site", (), {"module_name": block_name})()
    names = candidate_projection_parameter_names(
        model, candidate, "residual_path")
    parameters = dict(model.named_parameters())
    reset = 0
    for name in names:
        parameter = parameters[name]
        if parameter in optimizer.state:
            optimizer.state.pop(parameter)
            reset += 1
    return reset


def rebuild_sgd_after_growth(model, optimizer, args):
    """Add grown parameters while preserving/migrating existing SGD momentum."""
    migrations = getattr(model, "consume_parameter_migrations", lambda: [])()
    old_for_new = {new: old for old, new in migrations}
    replacement = make_optimizer(model, args)
    for parameter in model.parameters():
        source = old_for_new.get(parameter, parameter)
        if source not in optimizer.state:
            continue
        copied = {}
        for key, value in optimizer.state[source].items():
            if (torch.is_tensor(value) and value.shape == source.shape and
                    parameter.shape != source.shape):
                expanded = torch.zeros_like(parameter)
                slices = tuple(slice(0, size) for size in source.shape)
                expanded[slices].copy_(value)
                copied[key] = expanded
            else:
                copied[key] = value.detach().clone() if torch.is_tensor(value) else value
        replacement.state[parameter] = copied
    return replacement, len(migrations)


@torch.no_grad()
def batch_loss(model, batch):
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        return float(F.cross_entropy(model(batch[0]).float(), batch[1]))
    finally:
        for module, training in modes.items():
            module.training = training


def preview_projected_gain(model, projection, scale, batch) -> float:
    parameters = dict(model.named_parameters())
    saved = {name: parameters[name].detach().clone()
             for name in projection.parameter_delta}
    before = batch_loss(model, batch)
    try:
        projection.apply_(model, scale)
        return before - batch_loss(model, batch)
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                parameters[name].copy_(value)


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def recovery_fraction(projected_gain: float | None,
                      structural_gain: float | None):
    if (projected_gain is None or structural_gain is None or
            abs(structural_gain) < 1e-12):
        return None
    return projected_gain / structural_gain


def cg_diagnostics(projection: ProjectionResult | None) -> dict:
    if projection is None:
        return {
            "cg_converged": None, "cg_residual_norm": None,
            "cg_relative_residual": None,
            "cg_residual_norm_at_12": None,
            "cg_residual_norm_at_25": None,
            "cg_residual_norm_at_50": None,
            "cg_residual_norm_at_100": None,
            "cg_residual_norm_at_200": None,
            "cg_damping_requested": None, "cg_damping_used": None,
            "cg_attempt_count": 0, "cg_attempts": [],
            "cg_target_scale": None,
            "cg_solver_space": None, "cg_system_dimension": None,
            "cg_solver_dtype": None,
            "cg_preconditioner": None,
            "cg_preconditioner_probes": None,
            "cg_selected_attempt": None, "cg_selection_rule": None,
        }
    history = projection.cg.residual_history

    def at(iteration: int):
        return history[iteration] if iteration < len(history) else None

    return {
        "cg_converged": projection.cg.converged,
        "cg_residual_norm": projection.cg.residual_norm,
        "cg_relative_residual": projection.cg.relative_residual,
        "cg_residual_norm_at_12": at(12),
        "cg_residual_norm_at_25": at(25),
        "cg_residual_norm_at_50": at(50),
        "cg_residual_norm_at_100": at(100),
        "cg_residual_norm_at_200": at(200),
        "cg_damping_requested": projection.damping_requested,
        "cg_damping_used": projection.damping_used,
        "cg_attempt_count": len(projection.cg_attempts),
        "cg_target_scale": projection.target_scale,
        "cg_solver_space": projection.solver_space,
        "cg_system_dimension": projection.linear_system_dimension,
        "cg_solver_dtype": projection.solver_dtype,
        "cg_preconditioner": projection.preconditioner,
        "cg_preconditioner_probes": projection.preconditioner_probes,
        "cg_selected_attempt": projection.selected_attempt,
        "cg_selection_rule": projection.selection_rule,
        "cg_attempts": [
            {"damping": attempt.damping,
             "iterations": attempt.iterations,
             "residual_norm": attempt.residual_norm,
             "relative_residual": attempt.relative_residual,
             "converged": attempt.converged,
             "solution_is_finite": attempt.solution_is_finite,
             "functional_relative_residual":
                 attempt.functional_relative_residual,
             "functional_cosine_alignment":
                 attempt.functional_cosine_alignment}
            for attempt in projection.cg_attempts],
    }


def parameter_delta_norm(parameter_delta: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(sum(torch.sum(value.square())
                          for value in parameter_delta.values()))


def projection_application_gate(parameter_delta, heldout, *,
                                max_relative_residual: float,
                                min_cosine_alignment: float) -> dict:
    """Gate an update by finite parameters and held-out functional transfer."""
    delta_norm = float(parameter_delta_norm(parameter_delta))
    solution_is_finite = bool(
        delta_norm > 0 and math.isfinite(delta_norm) and
        all(torch.isfinite(value).all() for value in parameter_delta.values()))
    functional_fit_accepted = bool(
        math.isfinite(heldout.relative_residual) and
        math.isfinite(heldout.cosine_alignment) and
        heldout.relative_residual <= max_relative_residual and
        heldout.cosine_alignment >= min_cosine_alignment)
    return {
        "apply": solution_is_finite and functional_fit_accepted,
        "parameter_delta_norm": delta_norm,
        "solution_is_finite": solution_is_finite,
        "functional_fit_accepted": functional_fit_accepted,
    }


def sign_randomized_parameter_delta(
        parameter_delta: dict[str, torch.Tensor], *,
        generator: torch.Generator) -> dict[str, torch.Tensor]:
    """Randomize coordinate signs while preserving every tensor's norm."""
    randomized = {}
    for name, value in parameter_delta.items():
        signs = torch.empty_like(value).bernoulli_(
            0.5, generator=generator).mul_(2).sub_(1)
        randomized[name] = value * signs
    return randomized


def heldout_metrics(evaluation) -> dict[str, float]:
    return {
        "heldout_fitted_norm_ratio": evaluation.fitted_norm_ratio,
        "heldout_relative_residual": evaluation.relative_residual,
        "heldout_cosine_alignment": evaluation.cosine_alignment,
    }


@torch.no_grad()
def eval_logits(model, inputs):
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        return model(inputs).detach()
    finally:
        for module, training in modes.items():
            module.training = training


def actual_update_metrics(model, inputs, baseline_logits, target_delta,
                          applied_scale: float) -> dict[str, float]:
    """Compare the realized finite parameter jump with the held-out E target."""
    if applied_scale == 0:
        raise ValueError("actual-update metrics require non-zero applied scale")
    realized = (eval_logits(model, inputs) - baseline_logits) / applied_scale
    target = target_delta.detach()
    return {
        "actual_heldout_fitted_norm_ratio": fitted_norm_ratio(realized, target),
        "actual_heldout_relative_residual": relative_residual(realized, target),
        "actual_heldout_cosine_alignment": cosine_alignment(realized, target),
        "actual_relative_residual": relative_residual(realized, target),
        "actual_cosine_alignment": cosine_alignment(realized, target),
        "actual_functional_delta_norm": float(realized.norm()),
    }


def evaluate_heldout_direction(projector, model, tuning_batch,
                               target_delta, parameter_delta, device):
    """Measure out-of-sample tangent fit and its cost separately."""
    synchronize(device)
    started = time.perf_counter()
    evaluation = projector.evaluate_direction(
        model, tuning_batch[0], target_delta, parameter_delta)
    synchronize(device)
    return evaluation, time.perf_counter() - started


def main():
    args = arguments()
    requested_method = args.method
    if requested_method == "ours_e_driven_o":
        # Public four-arm name; implementation remains the frozen main arm.
        args.method = "tiny_projection"
    validate_arguments(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    completion_file = output / ("warmup.json" if args.prepare_warmup else "result.json")
    warmup_target = Path(args.warmup_checkpoint) if args.warmup_checkpoint else None
    completion_is_valid = False
    if completion_file.is_file():
        if args.prepare_warmup:
            completion_is_valid = (
                warmup_target is not None and warmup_target.is_file())
        else:
            completed = json.loads(completion_file.read_text())
            completion_is_valid = (
                len(completed.get("history", [])) >= args.epochs and
                (output / "checkpoint_latest.pt").is_file())
    if completion_is_valid:
        print(f"completed {completion_file.name} exists; skipping", flush=True)
        return
    reference_root = Path(args.reference_root).resolve()
    if not (reference_root / "dual_growth").is_dir():
        raise FileNotFoundError(
            f"invalid One-Shot-TAS-CCIL checkout: {reference_root}")
    sys.path.insert(0, str(reference_root))
    from dual_growth.adapters import TinyAdapter
    from dual_growth.controller import GrowthBudget

    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    (train_set, eval_set, test_set, train_indices,
     validation_indices, tuning_indices) = datasets_and_indices(args)
    validation_loader = make_eval_loader(
        eval_set, validation_indices, args.batch_size * 2, args)
    tuning_loader = make_eval_loader(
        eval_set, tuning_indices, args.tuning_samples, args)
    tuning_batch = tuple(
        value.to(device, non_blocking=True) for value in next(iter(tuning_loader)))
    test_loader = (None if test_set is None else DataLoader(
        test_set, args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0))

    model = build_pretrained_gromo_resnet18(100, device=device).to(device)
    block_refs = model.growing_blocks()
    hidden_widths = [int(ref.module.hidden_neurons) for ref in block_refs]
    if hidden_widths != FULL_WIDTHS:
        raise RuntimeError(
            f"expected full ResNet-18 widths {FULL_WIDTHS}, got {hidden_widths}")
    missing = {ref.name: ref.module.second_layer.missing_neurons()
               for ref in block_refs}
    if any(value != 0 for value in missing.values()):
        raise RuntimeError(f"full-width model still has growth capacity: {missing}")
    pretrained_sha256 = state_sha256(model)
    optimizer = make_optimizer(model, args)
    checkpoint_path = (Path(args.warmup_checkpoint)
                       if args.warmup_checkpoint else None)
    protocol = {
        "seed": args.seed,
        "warmup_epochs": args.warmup_epochs,
        "train_indices_sha256": index_sha256(train_indices),
        "validation_indices_sha256": index_sha256(validation_indices),
        "tuning_indices_sha256": index_sha256(tuning_indices),
        "train_samples": len(train_indices),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "architecture": model.architecture_id,
    }

    if checkpoint_path is not None and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint["protocol"] != protocol:
            raise RuntimeError("warm-up checkpoint protocol does not match this arm")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        warmup_history = checkpoint["history"]
        if state_sha256(model) != checkpoint["model_sha256"]:
            raise RuntimeError("loaded warm-up checkpoint failed its model hash")
    else:
        if checkpoint_path is not None and not args.prepare_warmup:
            raise FileNotFoundError(
                f"shared warm-up checkpoint is missing: {checkpoint_path}")
        warmup_loader = make_train_loader(train_set, train_indices, args)
        warmup_history = []
        for epoch in range(args.warmup_epochs):
            loss, accuracy = train_epoch(model, warmup_loader, optimizer, device)
            validation_loss, validation_accuracy = evaluate(
                model, validation_loader, device)
            row = {
                "epoch": epoch + 1, "train_loss": loss,
                "train_accuracy": accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
            warmup_history.append(row)
            print(json.dumps({"warmup": row}, sort_keys=True), flush=True)
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "history": warmup_history, "protocol": protocol,
                "model_sha256": state_sha256(model)}, temporary)
            temporary.replace(checkpoint_path)

    warmup_sha256 = state_sha256(model)
    if args.prepare_warmup:
        completion_file.write_text(json.dumps({
            "checkpoint": str(checkpoint_path), "protocol": protocol,
            "pretrained_model_sha256": pretrained_sha256,
            "warmup_model_sha256": warmup_sha256,
            "history": warmup_history,
        }, indent=2, sort_keys=True))
        return

    # Every arm starts its post-warm-up sample order from the same state.
    train_loader = make_train_loader(train_set, train_indices, args)
    initial_parameters = sum(parameter.numel() for parameter in model.parameters())
    projector = FunctionalProjector(
        args.damping, args.cg_iterations,
        tolerance=args.cg_relative_tolerance,
        preconditioner_probes=args.cg_preconditioner_probes)
    e_projection = EProjection(projector=projector)
    history = []
    prior_elapsed_seconds = 0.0
    start_epoch = 0
    resume_path = output / "checkpoint_latest.pt"
    run_protocol = {
        **protocol, "method": requested_method, "site": args.site,
        "rank": args.rank, "probe_epsilon": args.probe_epsilon,
        "projection_scope": "residual_path",
        "solver_revision": args.solver_revision,
    }
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location=device)
        if resume["protocol"] != run_protocol:
            raise RuntimeError("post-warm-up resume checkpoint protocol mismatch")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        history = resume["history"]
        start_epoch = int(resume["epoch"])
        prior_elapsed_seconds = float(resume.get("elapsed_seconds", 0.0))
        random.setstate(resume["python_rng_state"])
        torch.set_rng_state(resume["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(resume["cuda_rng_states"])
        if "train_loader_generator_state" in resume:
            train_loader.generator.set_state(
                resume["train_loader_generator_state"].cpu())
        if start_epoch > args.epochs:
            raise RuntimeError(
                f"checkpoint epoch {start_epoch} exceeds target {args.epochs}")
    start = time.perf_counter()
    projection_scale = (args.probe_epsilon if args.projection_scale == 0
                        else args.projection_scale)

    def propose(statistics_loader):
        if args.site == "auto":
            requested = [site.strip() for site in args.candidate_sites.split(",")
                         if site.strip()]
            sites = requested or [ref.name for ref in model.growing_blocks()]
        else:
            sites = [args.site]
        candidates = []
        for site in sites:
            adapter = TinyAdapter(
                quantum_params=10**9,
                max_statistics_batches=len(statistics_loader))
            candidate = CounterfactualTinyProbe(args.rank, site).propose(
                adapter, model, statistics_loader, GrowthBudget(10**9),
                sample_inputs=statistics_loader[0][0])
            candidates.append(candidate)
        selected = max(candidates, key=lambda item: float(item.proposal_score))
        selection = {
            "site_selection_mode": "tiny_score_argmax" if args.site == "auto"
                                   else "fixed_pilot_site",
            "selected_site": str(selected.module_name),
            "site_scores": {str(item.module_name): float(item.proposal_score)
                            for item in candidates},
        }
        return selected, selection

    for epoch in range(start_epoch, args.epochs):
        diagnostics = None
        if args.method == "vanilla_momentum_reset":
            if args.site == "auto":
                raise ValueError(
                    "vanilla_momentum_reset requires a frozen concrete site")
            reset = reset_residual_path_momentum(optimizer, model, args.site)
            diagnostics = {
                "source": "vanilla_residual_path_momentum_reset_control",
                "momentum_reset_scope": "residual_path",
                "momentum_states_reset": reset,
                "uses_additional_labels": False,
            }
        elif args.method == "vanilla_extra_sgd":
            statistics_loader, projection_batch, batch_audit = intervention_batches(
                eval_set, train_indices, args, epoch, device)
            extra_losses = []
            model.train()
            for inputs, targets in list(statistics_loader) + [projection_batch]:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(inputs).float(), targets)
                loss.backward()
                optimizer.step()
                extra_losses.append(float(loss.detach()))
            diagnostics = {
                "source": "vanilla_extra_sgd_data_exposure_control",
                "extra_sgd_steps": len(extra_losses),
                "extra_sgd_mean_loss": sum(extra_losses) / len(extra_losses),
                "extra_supervised_samples":
                    args.statistics_samples + args.projection_samples,
                **batch_audit,
            }
        elif args.method != "vanilla":
            statistics_loader, projection_batch, batch_audit = intervention_batches(
                eval_set, train_indices, args, epoch, device)
            synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            memory_before = torch.cuda.memory_allocated(device)
            reserved_before = torch.cuda.memory_reserved(device)
            started = time.perf_counter()
            candidate, site_selection = propose(list(statistics_loader))
            synchronize(device)
            e_seconds = time.perf_counter() - started
            signal_started = time.perf_counter()
            tuning_signal = CandidateExpansionProbe()(
                model, candidate=candidate, batch=tuning_batch,
                gate=args.probe_epsilon)
            synchronize(device)
            signal_seconds = time.perf_counter() - signal_started
            projection_result = None
            heldout = None
            projection_seconds = 0.0
            heldout_evaluation_seconds = 0.0
            momentum_resets = 0
            projection_scope = (
                "whole_block" if args.method == "tiny_projection_whole_block"
                else "conv_only" if args.method == "tiny_projection_conv_only"
                else "residual_path")

            if args.method == "vanilla_matched_compute":
                started = time.perf_counter()
                matched_step = e_projection.discover_candidate(
                    model, candidate, projection_batch,
                    gate=args.probe_epsilon,
                    projection_scope=projection_scope)
                synchronize(device)
                projection_seconds = time.perf_counter() - started
                projection_result = matched_step.projection
                heldout, heldout_evaluation_seconds = evaluate_heldout_direction(
                    projector, model, tuning_batch, tuning_signal.delta_logits,
                    projection_result.parameter_delta, device)
                diagnostics = {
                    "source": "vanilla_exact_probe_compute_discarded",
                    "probe_gate": args.probe_epsilon,
                    "correction_applied": False,
                    "correction_was_attempted": False,
                    "structural_loss_gain": matched_step.structural_loss_gain,
                    "tuning_structural_loss_gain":
                        tuning_signal.observed_loss_gain,
                    "fitted_norm_ratio": projection_result.fitted_norm_ratio,
                    "relative_residual": projection_result.relative_residual,
                    "cosine_alignment": projection_result.cosine_alignment,
                    **heldout_metrics(heldout),
                }
            elif args.method in {"tiny_projection",
                                 "tiny_projection_conv_only",
                                 "tiny_projection_whole_block"}:
                tuning_loss_before = batch_loss(model, tuning_batch)
                started = time.perf_counter()
                step = e_projection.discover_candidate(
                    model, candidate, projection_batch,
                    gate=args.probe_epsilon,
                    projection_scope=projection_scope)
                synchronize(device)
                projection_seconds = time.perf_counter() - started
                projection_result = step.projection
                heldout, heldout_evaluation_seconds = evaluate_heldout_direction(
                    projector, model, tuning_batch, tuning_signal.delta_logits,
                    projection_result.parameter_delta, device)
                application = projection_application_gate(
                    projection_result.parameter_delta, heldout,
                    max_relative_residual=
                        args.application_max_heldout_residual,
                    min_cosine_alignment=
                        args.application_min_heldout_cosine)
                # CG convergence is a solver diagnostic, not the application
                # gate. A finite direction that demonstrably transfers to the
                # held-out functional batch is the relevant safety criterion.
                correction_applied = application["apply"]
                tuning_logits_before = eval_logits(model, tuning_batch[0])
                actual_metrics = {}
                if correction_applied:
                    projection_result.apply_(model, projection_scale)
                    momentum_resets = reset_projected_momentum(
                        optimizer, model, projection_result)
                    actual_metrics = actual_update_metrics(
                        model, tuning_batch[0], tuning_logits_before,
                        tuning_signal.delta_logits, projection_scale)
                    if (not math.isfinite(
                            actual_metrics["actual_functional_delta_norm"]) or
                            actual_metrics["actual_functional_delta_norm"] <= 0):
                        raise RuntimeError(
                            "projected correction had no finite functional effect")
                tuning_loss_after = batch_loss(model, tuning_batch)
                tuning_projected_gain = tuning_loss_before - tuning_loss_after
                diagnostics = {
                    "source": step.signal.source,
                    "projection_scope": projection_scope,
                    "probe_gate": args.probe_epsilon,
                    "applied_scale": projection_scale,
                    "correction_applied": correction_applied,
                    "correction_was_attempted": True,
                    "application_gate": "finite_and_heldout_functional_fit",
                    "application_gate_uses_functional_fit": True,
                    "solution_is_finite": application["solution_is_finite"],
                    "functional_fit_accepted":
                        application["functional_fit_accepted"],
                    "application_max_heldout_residual":
                        args.application_max_heldout_residual,
                    "application_min_heldout_cosine":
                        args.application_min_heldout_cosine,
                    "parameter_delta_norm":
                        application["parameter_delta_norm"],
                    "loss_before": tuning_loss_before,
                    "loss_after": tuning_loss_after,
                    "structural_loss_gain": step.structural_loss_gain,
                    "structural_directional_gain": step.structural_directional_gain,
                    "tuning_structural_loss_gain": tuning_signal.observed_loss_gain,
                    "tuning_structural_directional_gain": tuning_signal.predicted_gain,
                    "tuning_projected_loss_gain": tuning_projected_gain,
                    "tuning_local_recovery_fraction": recovery_fraction(
                        tuning_projected_gain, tuning_signal.observed_loss_gain),
                    "fitted_norm_ratio": projection_result.fitted_norm_ratio,
                    "relative_residual": projection_result.relative_residual,
                    "cosine_alignment": projection_result.cosine_alignment,
                    **heldout_metrics(heldout),
                    **actual_metrics,
                }
            elif args.method in {"random_projection",
                                 "sign_randomized_projection"}:
                started = time.perf_counter()
                e_step = e_projection.discover_candidate(
                    model, candidate, projection_batch,
                    gate=args.probe_epsilon,
                    projection_scope=projection_scope)
                e_delta = e_step.projection.parameter_delta
                e_norm = parameter_delta_norm(e_delta)
                if args.method == "sign_randomized_projection":
                    generator = torch.Generator(device=device).manual_seed(
                        970_001 + args.seed * 10_000 + epoch)
                    random_parameter_delta = sign_randomized_parameter_delta(
                        e_delta, generator=generator)
                    source = "sign_randomized_e_delta_tensor_norm_matched"
                else:
                    random_parameter_delta = {
                        name: torch.randn_like(value)
                        for name, value in e_delta.items()}
                    random_norm = parameter_delta_norm(random_parameter_delta)
                    scale = e_norm / random_norm.clamp_min(1e-12)
                    random_parameter_delta = {
                        name: value * scale
                        for name, value in random_parameter_delta.items()}
                    source = "random_parameter_delta_global_norm_matched_to_e"
                per_tensor_relative_norm_error = max(
                    float(abs(random_parameter_delta[name].norm() - value.norm()) /
                          value.norm().clamp_min(1e-12))
                    for name, value in e_delta.items())
                projection_result = replace(
                    e_step.projection, parameter_delta=random_parameter_delta)
                synchronize(device)
                projection_seconds = time.perf_counter() - started
                heldout, heldout_evaluation_seconds = evaluate_heldout_direction(
                    projector, model, tuning_batch, tuning_signal.delta_logits,
                    random_parameter_delta, device)
                tuning_loss_before = batch_loss(model, tuning_batch)
                application = projection_application_gate(
                    random_parameter_delta, heldout,
                    max_relative_residual=
                        args.application_max_heldout_residual,
                    min_cosine_alignment=
                        args.application_min_heldout_cosine)
                correction_applied = application["apply"]
                tuning_logits_before = eval_logits(model, tuning_batch[0])
                actual_metrics = {}
                if correction_applied:
                    projection_result.apply_(model, projection_scale)
                    momentum_resets = reset_projected_momentum(
                        optimizer, model, projection_result)
                    actual_metrics = actual_update_metrics(
                        model, tuning_batch[0], tuning_logits_before,
                        tuning_signal.delta_logits, projection_scale)
                diagnostics = {
                    "source": source,
                    "probe_gate": args.probe_epsilon,
                    "applied_scale": projection_scale,
                    "correction_applied": correction_applied,
                    "correction_was_attempted": True,
                    "tuning_projected_loss_gain":
                        tuning_loss_before - batch_loss(model, tuning_batch),
                    "e_parameter_delta_norm": float(e_norm),
                    "random_parameter_delta_norm": float(
                        parameter_delta_norm(random_parameter_delta)),
                    "max_per_tensor_relative_norm_error":
                        per_tensor_relative_norm_error,
                    **heldout_metrics(heldout),
                    **actual_metrics,
                }
            elif args.method == "expand_train_project":
                control = ExpandedTrainProject(
                    steps=args.expanded_train_steps,
                    learning_rate=args.lr, momentum=SGD_MOMENTUM,
                    weight_decay=SGD_WEIGHT_DECAY, projector=projector)
                started = time.perf_counter()
                control_result = control.discover(
                    model, candidate, projection_batch,
                    gate=1.0, projection_scope=projection_scope,
                    heldout_batch=tuning_batch, base_optimizer=optimizer)
                synchronize(device)
                projection_seconds = time.perf_counter() - started
                projection_result = control_result.projection
                if control_result.heldout_delta_logits is None:
                    raise RuntimeError("expanded control omitted held-out target")
                heldout, heldout_evaluation_seconds = evaluate_heldout_direction(
                    projector, model, tuning_batch,
                    control_result.heldout_delta_logits,
                    projection_result.parameter_delta, device)
                control_scale = (1.0 if args.projection_scale == 0
                                 else args.projection_scale)
                tuning_loss_before = batch_loss(model, tuning_batch)
                application = projection_application_gate(
                    projection_result.parameter_delta, heldout,
                    max_relative_residual=
                        args.application_max_heldout_residual,
                    min_cosine_alignment=
                        args.application_min_heldout_cosine)
                correction_applied = application["apply"]
                tuning_logits_before = eval_logits(model, tuning_batch[0])
                actual_metrics = {}
                if correction_applied:
                    projection_result.apply_(model, control_scale)
                    momentum_resets = reset_projected_momentum(
                        optimizer, model, projection_result)
                    actual_metrics = actual_update_metrics(
                        model, tuning_batch[0], tuning_logits_before,
                        control_result.heldout_delta_logits, control_scale)
                diagnostics = {
                    "source": "repan_bypass_like_control",
                    "signal_source": control_result.signal.source,
                    "expansion_gate": 1.0,
                    "applied_scale": control_scale,
                    "correction_applied": correction_applied,
                    "correction_was_attempted": True,
                    "expanded_train_losses": control_result.expansion_train_losses,
                    "expanded_train_steps": args.expanded_train_steps,
                    "expanded_train_samples_per_step":
                        int(projection_batch[1].numel()),
                    "expanded_train_mode": True,
                    "expanded_train_optimizer": {
                        "name": "SGD", "learning_rate": args.lr,
                        "momentum": SGD_MOMENTUM,
                        "weight_decay": SGD_WEIGHT_DECAY,
                        "inherited_optimizer_states":
                            control_result.inherited_optimizer_states,
                    },
                    "temporary_base_parameter_update_norm":
                        control_result.temporary_base_parameter_update_norm,
                    "tuning_expanded_trained_loss_gain":
                        control_result.heldout_loss_gain,
                    "heldout_target_source": "expanded_trained_e",
                    "tuning_projected_loss_gain":
                        tuning_loss_before - batch_loss(model, tuning_batch),
                    "fitted_norm_ratio": projection_result.fitted_norm_ratio,
                    "relative_residual": projection_result.relative_residual,
                    "cosine_alignment": projection_result.cosine_alignment,
                    **heldout_metrics(heldout),
                    **actual_metrics,
                }
            elif args.method == "real_e_growth":
                started = time.perf_counter()
                local_step = e_projection.discover_candidate(
                    model, candidate, projection_batch,
                    gate=args.probe_epsilon,
                    projection_scope=projection_scope)
                synchronize(device)
                projection_seconds = time.perf_counter() - started
                projection_result = local_step.projection
                heldout, heldout_evaluation_seconds = evaluate_heldout_direction(
                    projector, model, tuning_batch, tuning_signal.delta_logits,
                    projection_result.parameter_delta, device)
                application = projection_application_gate(
                    projection_result.parameter_delta, heldout,
                    max_relative_residual=
                        args.application_max_heldout_residual,
                    min_cosine_alignment=
                        args.application_min_heldout_cosine)
                local_projected_gain = (
                    preview_projected_gain(
                        model, projection_result, projection_scale, tuning_batch)
                    if application["apply"] else None)
                commit_started = time.perf_counter()
                commit = RealEGrowth.commit_(model, candidate)
                committed_second = commit.committed_module.second_layer
                growth_current_width = int(committed_second.in_neurons)
                growth_target_width = int(committed_second.target_in_neurons)
                if growth_current_width != growth_target_width:
                    raise RuntimeError(
                        "growth commit left current width different from target")
                optimizer, migrated = rebuild_sgd_after_growth(
                    model, optimizer, args)
                synchronize(device)
                commit_seconds = time.perf_counter() - commit_started
                diagnostics = {
                    "source": "tiny_gromo_committed_e_growth",
                    "correction_was_attempted": False,
                    "probe_gate": args.probe_epsilon,
                    "local_structural_loss_gain": local_step.structural_loss_gain,
                    "tuning_structural_loss_gain": tuning_signal.observed_loss_gain,
                    "tuning_local_projected_loss_gain": local_projected_gain,
                    "tuning_local_recovery_fraction": recovery_fraction(
                        local_projected_gain, tuning_signal.observed_loss_gain),
                    "fitted_norm_ratio": projection_result.fitted_norm_ratio,
                    "relative_residual": projection_result.relative_residual,
                    "cosine_alignment": projection_result.cosine_alignment,
                    **heldout_metrics(heldout),
                    "deploy_parameter_delta_this_intervention":
                        commit.deploy_parameter_delta,
                    "optimizer_migrations": migrated,
                    "growth_current_width": growth_current_width,
                    "growth_target_width": growth_target_width,
                    "commit_seconds": commit_seconds,
                }

            diagnostics.update(batch_audit)
            diagnostics.update({
                "statistics_from_training_pool": True,
                "tuning_batch_is_held_out_from_updates": True,
                "projection_scope": projection_scope,
                **site_selection,
                "candidate_extra_flops": float(candidate.extra_flops),
                "e_statistics_solve_seconds": e_seconds,
                "structural_signal_seconds": signal_seconds,
                "projection_seconds": projection_seconds,
                "heldout_evaluation_seconds": heldout_evaluation_seconds,
                "jvp_calls": (0 if projection_result is None
                              else projection_result.jvp_calls +
                              (0 if heldout is None else heldout.jvp_calls)),
                "vjp_calls": (0 if projection_result is None
                              else projection_result.vjp_calls),
                "cg_iterations": (0 if projection_result is None
                                  else projection_result.cg.iterations),
                "cg_max_iterations": args.cg_iterations,
                "cg_tolerance": projector.tolerance,
                **cg_diagnostics(projection_result),
                "momentum_states_reset": momentum_resets,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "gpu_peak_increment_bytes": max(
                    0, torch.cuda.max_memory_allocated(device) - memory_before),
                "gpu_peak_reserved_increment_bytes": max(
                    0, torch.cuda.max_memory_reserved(device) - reserved_before),
            })

        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, device)
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device)
        row = {
            "epoch": epoch + 1, "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "diagnostics": diagnostics,
        }
        if diagnostics is not None:
            diagnostics["validation_accuracy"] = validation_accuracy
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        elapsed = prior_elapsed_seconds + time.perf_counter() - start
        temporary = resume_path.with_suffix(".pt.tmp")
        torch.save({
            "format_version": 1, "epoch": epoch + 1,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "history": history, "protocol": run_protocol,
            "elapsed_seconds": elapsed,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "train_loader_generator_state": train_loader.generator.get_state(),
        }, temporary)
        temporary.replace(resume_path)
        progress = output / "progress.json"
        progress_tmp = output / "progress.json.tmp"
        progress_tmp.write_text(json.dumps({
            "method": requested_method, "target_epochs": args.epochs,
            "completed_epochs": epoch + 1, "latest": row,
            "checkpoint": str(resume_path),
        }, indent=2, sort_keys=True))
        progress_tmp.replace(progress)

    # Hyperparameter-sweep arms do not construct or iterate the official test
    # set. The selected configuration is rerun with this explicit flag.
    if test_loader is None:
        test_loss = test_accuracy = None
    else:
        test_loss, test_accuracy = evaluate(model, test_loader, device)
    final_parameters = sum(parameter.numel() for parameter in model.parameters())
    correction_attempts = sum(
        bool(row["diagnostics"] and
             row["diagnostics"].get("correction_was_attempted"))
        for row in history)
    corrections_applied = sum(
        bool(row["diagnostics"] and
             row["diagnostics"].get("correction_was_attempted") and
             row["diagnostics"].get("correction_applied"))
        for row in history)
    applied_diagnostics = [
        row["diagnostics"] for row in history
        if row["diagnostics"] and row["diagnostics"].get("correction_applied")]
    latest_applied = applied_diagnostics[-1] if applied_diagnostics else {}
    run_peak_gpu_memory = max([
        int(row["diagnostics"].get("gpu_peak_allocated_bytes", 0))
        for row in history if row["diagnostics"]
    ] + [int(torch.cuda.max_memory_allocated(device))])
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]),
             "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        source_commit = "unknown"
    result = {
        "method": requested_method, "seed": args.seed,
        "epoch": len(history),
        "train_accuracy": history[-1]["train_accuracy"],
        "pretrained_model_sha256": pretrained_sha256,
        "initial_model_sha256": warmup_sha256,
        "warmup_history": warmup_history,
        "validation_accuracy": history[-1]["validation_accuracy"],
        "validation_loss": history[-1]["validation_loss"],
        "best_validation_accuracy": max(x["validation_accuracy"] for x in history),
        "official_test_loss": test_loss,
        "official_test_accuracy": test_accuracy,
        "deploy_parameters_before": initial_parameters,
        "deploy_parameters_after": final_parameters,
        "deploy_parameter_delta": final_parameters - initial_parameters,
        "correction_attempts": correction_attempts,
        "corrections_applied": corrections_applied,
        "correction_application_rate": (
            corrections_applied / correction_attempts
            if correction_attempts else None),
        "actual_cosine_alignment": latest_applied.get(
            "actual_cosine_alignment"),
        "actual_relative_residual": latest_applied.get(
            "actual_relative_residual"),
        "architecture": model.architecture_id,
        "pretrained_backbone": True,
        "initial_hidden_widths": hidden_widths,
        "initial_missing_neurons": missing,
        "data_protocol": {
            **protocol,
            "probe_pool_equals_training_pool": True,
            "tuning_batch_never_used_for_updates": True,
            "official_test_evaluated_once_after_training":
                args.evaluate_official_test,
        },
        "training_seconds": prior_elapsed_seconds + time.perf_counter() - start,
        "elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - start,
        "peak_train_params": initial_parameters,
        "deploy_params": final_parameters,
        "peak_gpu_memory": run_peak_gpu_memory,
        "source_repo": "https://github.com/duyh80456-code/Counterfactual-Projection.git",
        "source_commit": source_commit,
        "checkpoint": str(resume_path),
        "history": history, "config": vars(args),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
