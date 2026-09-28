"""Fair full-width structural-E CIFAR-100 pilot on one visible GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from baselines import ExpandedTrainProject, RealEOracle
from methods import EProjection
from methods.e_projection import candidate_projection_block
from probe import (
    CandidateExpansionProbe, CounterfactualTinyProbe,
    build_pretrained_gromo_resnet18)
from projection import FunctionalProjector, ProjectionResult


METHODS = (
    "vanilla", "random_projection", "tiny_projection",
    "expand_train_project", "real_e_oracle")
FULL_WIDTHS = [64, 64, 128, 128, 256, 256, 512, 512]


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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--statistics-samples", type=int, default=256)
    parser.add_argument("--projection-samples", type=int, default=32)
    parser.add_argument("--check-samples", type=int, default=128)
    parser.add_argument("--site", default="stages.2.blocks.0")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--probe-epsilon", type=float, default=0.05)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--cg-iterations", type=int, default=12)
    parser.add_argument("--projection-scale", type=float, default=0.0,
                        help="0 applies the fitted direction at probe epsilon")
    parser.add_argument("--expanded-train-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    return parser.parse_args()


def validate_arguments(args) -> None:
    if min(args.statistics_samples, args.projection_samples,
           args.check_samples, args.validation_samples) < 1:
        raise ValueError("validation/statistics/projection/check sizes must be positive")
    if args.validation_samples + args.check_samples >= 50000:
        raise ValueError("held-out splits leave no CIFAR-100 training examples")
    available = 50000 - args.validation_samples - args.check_samples
    used = available if args.train_samples == 0 else min(args.train_samples, available)
    if args.statistics_samples + args.projection_samples > used:
        raise ValueError("one intervention needs more examples than the training pool")
    if not 0 < args.probe_epsilon <= 1:
        raise ValueError("probe epsilon must be in (0, 1]")
    if args.projection_scale < 0 or args.warmup_epochs < 0:
        raise ValueError("projection scale and warm-up epochs must be non-negative")
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
    order = torch.randperm(
        len(train_set), generator=torch.Generator().manual_seed(20260928)).tolist()
    validation_indices = order[:args.validation_samples]
    check_start = args.validation_samples
    check_indices = order[check_start:check_start + args.check_samples]
    train_indices = order[check_start + args.check_samples:]
    if args.train_samples:
        train_indices = train_indices[:args.train_samples]
    return train_set, eval_set, train_indices, validation_indices, check_indices


def make_train_loader(dataset, indices, args):
    return DataLoader(
        Subset(dataset, indices), args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)


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
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


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


def recovery_fraction(projected_gain: float, structural_gain: float | None):
    if structural_gain is None or abs(structural_gain) < 1e-12:
        return None
    return projected_gain / structural_gain


def main():
    args = arguments()
    validate_arguments(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    completion_file = output / ("warmup.json" if args.prepare_warmup else "result.json")
    warmup_target = Path(args.warmup_checkpoint) if args.warmup_checkpoint else None
    completion_is_valid = (completion_file.is_file() and
                           (not args.prepare_warmup or
                            (warmup_target is not None and warmup_target.is_file())))
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
    (train_set, eval_set, train_indices,
     validation_indices, check_indices) = datasets_and_indices(args)
    validation_loader = make_eval_loader(
        eval_set, validation_indices, args.batch_size * 2, args)
    check_loader = make_eval_loader(
        eval_set, check_indices, args.check_samples, args)
    check_batch = tuple(
        value.to(device, non_blocking=True) for value in next(iter(check_loader)))

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
        "check_indices_sha256": index_sha256(check_indices),
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
        args.damping, args.cg_iterations, tolerance=1e-5)
    e_projection = EProjection(projector=projector)
    history = []
    start = time.perf_counter()
    projection_scale = (args.probe_epsilon if args.projection_scale == 0
                        else args.projection_scale)

    def propose(statistics_loader):
        adapter = TinyAdapter(
            quantum_params=10**9,
            max_statistics_batches=len(statistics_loader))
        return CounterfactualTinyProbe(args.rank, args.site).propose(
            adapter, model, statistics_loader, GrowthBudget(10**9))

    for epoch in range(args.epochs):
        diagnostics = None
        if args.method != "vanilla":
            statistics_loader, projection_batch, batch_audit = intervention_batches(
                eval_set, train_indices, args, epoch, device)
            synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            memory_before = torch.cuda.memory_allocated(device)
            reserved_before = torch.cuda.memory_reserved(device)
            started = time.perf_counter()
            candidate = propose(list(statistics_loader))
            synchronize(device)
            e_seconds = time.perf_counter() - started
            signal_started = time.perf_counter()
            check_signal = CandidateExpansionProbe()(
                model, candidate=candidate, batch=check_batch,
                gate=args.probe_epsilon)
            synchronize(device)
            signal_seconds = time.perf_counter() - signal_started
            projection_result = None
            projection_seconds = 0.0
            momentum_resets = 0

            if args.method == "tiny_projection":
                check_loss_before = batch_loss(model, check_batch)
                started = time.perf_counter()
                step = e_projection.discover_candidate(
                    model, candidate, projection_batch,
                    gate=args.probe_epsilon)
                synchronize(device)
                projection_seconds = time.perf_counter() - started
                projection_result = step.projection
                projection_result.apply_(model, projection_scale)
                momentum_resets = reset_projected_momentum(
                    optimizer, model, projection_result)
                check_projected_gain = (
                    check_loss_before - batch_loss(model, check_batch))
                diagnostics = {
                    "source": step.signal.source,
                    "probe_gate": args.probe_epsilon,
                    "applied_scale": projection_scale,
                    "structural_loss_gain": step.structural_loss_gain,
                    "structural_directional_gain": step.structural_directional_gain,
                    "check_structural_loss_gain": check_signal.observed_loss_gain,
                    "check_structural_directional_gain": check_signal.predicted_gain,
                    "check_projected_loss_gain": check_projected_gain,
                    "check_local_recovery_fraction": recovery_fraction(
                        check_projected_gain, check_signal.observed_loss_gain),
                    "fitted_norm_ratio": projection_result.fitted_norm_ratio,
                    "relative_residual": projection_result.relative_residual,
                    "cosine_alignment": projection_result.cosine_alignment,
                }
            elif args.method == "random_projection":
                fit_signal = CandidateExpansionProbe()(
                    model, candidate=candidate, batch=projection_batch,
                    gate=args.probe_epsilon)
                random_delta = torch.randn_like(fit_signal.delta_logits)
                random_delta.mul_(fit_signal.delta_logits.norm() /
                                  random_delta.norm().clamp_min(1e-12))
                block = candidate_projection_block(model, candidate)
                started = time.perf_counter()
                projection_result = projector.project(
                    model, projection_batch[0], random_delta, block=block)
                synchronize(device)
                projection_seconds = time.perf_counter() - started
                check_loss_before = batch_loss(model, check_batch)
                projection_result.apply_(model, projection_scale)
                momentum_resets = reset_projected_momentum(
                    optimizer, model, projection_result)
                diagnostics = {
                    "source": "random_matched_to_structural_norm",
                    "probe_gate": args.probe_epsilon,
                    "applied_scale": projection_scale,
                    "check_projected_loss_gain":
                        check_loss_before - batch_loss(model, check_batch),
                    "fitted_norm_ratio": projection_result.fitted_norm_ratio,
                    "relative_residual": projection_result.relative_residual,
                    "cosine_alignment": projection_result.cosine_alignment,
                }
            elif args.method == "expand_train_project":
                control = ExpandedTrainProject(
                    args.expanded_train_steps, args.lr, projector)
                started = time.perf_counter()
                control_result = control.discover(
                    model, candidate, projection_batch)
                synchronize(device)
                projection_seconds = time.perf_counter() - started
                projection_result = control_result.projection
                control_scale = (1.0 if args.projection_scale == 0
                                 else args.projection_scale)
                check_loss_before = batch_loss(model, check_batch)
                projection_result.apply_(model, control_scale)
                momentum_resets = reset_projected_momentum(
                    optimizer, model, projection_result)
                diagnostics = {
                    "source": "repan_bypass_like_control",
                    "signal_source": control_result.signal.source,
                    "applied_scale": control_scale,
                    "expanded_train_losses": control_result.expansion_train_losses,
                    "check_projected_loss_gain":
                        check_loss_before - batch_loss(model, check_batch),
                    "fitted_norm_ratio": projection_result.fitted_norm_ratio,
                    "relative_residual": projection_result.relative_residual,
                    "cosine_alignment": projection_result.cosine_alignment,
                }
            elif args.method == "real_e_oracle":
                started = time.perf_counter()
                local_step = e_projection.discover_candidate(
                    model, candidate, projection_batch,
                    gate=args.probe_epsilon)
                synchronize(device)
                projection_seconds = time.perf_counter() - started
                projection_result = local_step.projection
                local_projected_gain = preview_projected_gain(
                    model, projection_result, projection_scale, check_batch)
                commit_started = time.perf_counter()
                commit = RealEOracle.commit_(model, candidate)
                optimizer, migrated = rebuild_sgd_after_growth(
                    model, optimizer, args)
                synchronize(device)
                commit_seconds = time.perf_counter() - commit_started
                diagnostics = {
                    "source": "tiny_gromo_committed_oracle",
                    "probe_gate": args.probe_epsilon,
                    "local_structural_loss_gain": local_step.structural_loss_gain,
                    "check_structural_loss_gain": check_signal.observed_loss_gain,
                    "check_local_projected_loss_gain": local_projected_gain,
                    "check_local_recovery_fraction": recovery_fraction(
                        local_projected_gain, check_signal.observed_loss_gain),
                    "fitted_norm_ratio": projection_result.fitted_norm_ratio,
                    "relative_residual": projection_result.relative_residual,
                    "cosine_alignment": projection_result.cosine_alignment,
                    "deploy_parameter_delta_this_intervention":
                        commit.deploy_parameter_delta,
                    "optimizer_migrations": migrated,
                    "commit_seconds": commit_seconds,
                }

            diagnostics.update(batch_audit)
            diagnostics.update({
                "statistics_from_training_pool": True,
                "check_is_held_out": True,
                "e_statistics_solve_seconds": e_seconds,
                "structural_signal_seconds": signal_seconds,
                "projection_seconds": projection_seconds,
                "jvp_calls": (0 if projection_result is None
                              else projection_result.jvp_calls),
                "vjp_calls": (0 if projection_result is None
                              else projection_result.vjp_calls),
                "cg_iterations": (0 if projection_result is None
                                  else projection_result.cg.iterations),
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
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    final_parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "method": args.method, "seed": args.seed,
        "pretrained_model_sha256": pretrained_sha256,
        "initial_model_sha256": warmup_sha256,
        "warmup_history": warmup_history,
        "validation_accuracy": history[-1]["validation_accuracy"],
        "best_validation_accuracy": max(x["validation_accuracy"] for x in history),
        "deploy_parameters_before": initial_parameters,
        "deploy_parameters_after": final_parameters,
        "deploy_parameter_delta": final_parameters - initial_parameters,
        "architecture": model.architecture_id,
        "pretrained_backbone": True,
        "initial_hidden_widths": hidden_widths,
        "initial_missing_neurons": missing,
        "data_protocol": {
            **protocol,
            "probe_pool_equals_training_pool": True,
            "check_never_used_for_updates": True,
        },
        "elapsed_seconds": time.perf_counter() - start,
        "history": history, "config": vars(args),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
