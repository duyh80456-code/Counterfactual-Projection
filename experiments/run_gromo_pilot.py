"""Real structural-E CIFAR-100 pilot using One-Shot-TAS-CCIL + Gromo."""

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
from torch.utils.data import DataLoader, Subset

from baselines import ExpandedTrainProject, RealEOracle
from methods import EProjection
from methods.e_projection import candidate_projection_block
from probe import (
    CandidateExpansionProbe, CounterfactualTinyProbe,
    build_pretrained_gromo_resnet18)
from projection import FunctionalProjector


METHODS = (
    "vanilla", "random_projection", "tiny_projection",
    "expand_train_project", "real_e_oracle")


def state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
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
    parser.add_argument("--statistics-batches", type=int, default=1)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--cg-iterations", type=int, default=12)
    parser.add_argument("--projection-scale", type=float, default=0.0,
                        help="0 applies the fitted direction at probe epsilon")
    parser.add_argument("--expanded-train-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    return parser.parse_args()


def loaders(args):
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
    validation_set = datasets.CIFAR100(
        args.data_root, train=True, transform=eval_transform, download=False)
    order = torch.randperm(
        len(train_set), generator=torch.Generator().manual_seed(20260928)).tolist()
    cursor = 0
    validation_indices = order[cursor:cursor + args.validation_samples]
    cursor += args.validation_samples
    check_indices = order[cursor:cursor + args.check_samples]
    cursor += args.check_samples
    projection_indices = order[cursor:cursor + args.projection_samples]
    cursor += args.projection_samples
    statistics_indices = order[cursor:cursor + args.statistics_samples]
    cursor += args.statistics_samples
    train_indices = order[cursor:]
    if args.train_samples:
        train_indices = train_indices[:args.train_samples]
    train = DataLoader(
        Subset(train_set, train_indices), args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    validation = DataLoader(
        Subset(validation_set, validation_indices), args.batch_size * 2,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    statistics = DataLoader(
        Subset(validation_set, statistics_indices), args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    projection = DataLoader(
        Subset(validation_set, projection_indices), args.projection_samples,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    check = DataLoader(
        Subset(validation_set, check_indices), args.check_samples,
        shuffle=False, num_workers=args.workers, pin_memory=True)
    return train, validation, statistics, next(iter(projection)), next(iter(check))


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


def main():
    args = arguments()
    reserved = (args.validation_samples + args.statistics_samples +
                args.projection_samples + args.check_samples)
    if min(args.statistics_samples, args.projection_samples,
           args.check_samples) < 1:
        raise ValueError("statistics/projection/check splits must be non-empty")
    if reserved >= 50000:
        raise ValueError("reserved CIFAR-100 splits leave no training examples")
    if not 0 < args.probe_epsilon <= 1:
        raise ValueError("probe epsilon must be in (0, 1]")
    if args.projection_scale < 0:
        raise ValueError("projection scale must be non-negative")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").is_file():
        print("completed result exists; skipping", flush=True)
        return
    reference = Path(args.reference_root).resolve()
    if not (reference / "dual_growth").is_dir():
        raise FileNotFoundError(f"invalid One-Shot-TAS-CCIL checkout: {reference}")
    sys.path.insert(0, str(reference))
    from dual_growth.adapters import TinyAdapter
    from dual_growth.controller import GrowthBudget

    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    (train_loader, validation_loader, statistics_loader,
     projection_batch, check_batch) = loaders(args)
    projection_batch = tuple(
        value.to(device, non_blocking=True) for value in projection_batch)
    check_batch = tuple(
        value.to(device, non_blocking=True) for value in check_batch)
    model = build_pretrained_gromo_resnet18(100, device=device).to(device)
    block_refs = model.growing_blocks()
    hidden_widths = [int(block_ref.module.hidden_neurons)
                     for block_ref in block_refs]
    expected_widths = [64, 64, 128, 128, 256, 256, 512, 512]
    if hidden_widths != expected_widths:
        raise RuntimeError(
            f"expected full ResNet-18 widths {expected_widths}, got {hidden_widths}")
    missing = {
        block_ref.name: block_ref.module.second_layer.missing_neurons()
        for block_ref in block_refs}
    if any(value != 0 for value in missing.values()):
        raise RuntimeError(f"full-width model still has growth capacity: {missing}")
    initial_sha256 = state_sha256(model)
    initial_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    projector = FunctionalProjector(
        args.damping, args.cg_iterations, tolerance=1e-5)
    e_projection = EProjection(projector=projector)
    history = []
    start = time.time()
    projection_scale = (args.probe_epsilon if args.projection_scale == 0
                        else args.projection_scale)

    def propose():
        adapter = TinyAdapter(
            quantum_params=10**9,
            max_statistics_batches=args.statistics_batches)
        return CounterfactualTinyProbe(args.rank, args.site).propose(
            adapter, model, statistics_loader, GrowthBudget(10**9))

    @torch.no_grad()
    def batch_loss(batch):
        modes = {module: module.training for module in model.modules()}
        try:
            model.eval()
            return float(F.cross_entropy(model(batch[0]).float(), batch[1]))
        finally:
            for module, training in modes.items():
                module.training = training

    for epoch in range(args.epochs):
        diagnostics = None
        if args.method != "vanilla" and not (
                args.method == "real_e_oracle" and epoch > 0):
            candidate = propose()
            if args.method == "tiny_projection":
                check_signal = CandidateExpansionProbe()(
                    model, candidate=candidate, batch=check_batch,
                    gate=args.probe_epsilon)
                check_loss_before = batch_loss(check_batch)
                step = e_projection.discover_candidate(
                    model, candidate, projection_batch,
                    gate=args.probe_epsilon)
                step.projection.apply_(model, projection_scale)
                check_loss_after = batch_loss(check_batch)
                diagnostics = {
                    "source": step.signal.source,
                    "probe_gate": args.probe_epsilon,
                    "applied_scale": projection_scale,
                    "structural_loss_gain": step.structural_loss_gain,
                    "structural_directional_gain":
                        step.structural_directional_gain,
                    "check_structural_loss_gain":
                        check_signal.observed_loss_gain,
                    "check_structural_directional_gain":
                        check_signal.predicted_gain,
                    "check_projected_loss_gain":
                        check_loss_before - check_loss_after,
                    "fitted_norm_ratio": step.projection.fitted_norm_ratio,
                    "relative_residual": step.projection.relative_residual,
                    "cosine_alignment": step.projection.cosine_alignment,
                }
            elif args.method == "random_projection":
                signal = CandidateExpansionProbe()(
                    model, candidate=candidate, batch=projection_batch,
                    gate=args.probe_epsilon)
                random_delta = torch.randn_like(signal.delta_logits)
                random_delta.mul_(signal.delta_logits.norm() /
                                  random_delta.norm().clamp_min(1e-12))
                block = candidate_projection_block(model, candidate)
                projected = projector.project(
                    model, projection_batch[0], random_delta, block=block)
                check_loss_before = batch_loss(check_batch)
                projected.apply_(model, projection_scale)
                check_loss_after = batch_loss(check_batch)
                diagnostics = {
                    "source": "random_matched_to_structural_norm",
                    "probe_gate": args.probe_epsilon,
                    "applied_scale": projection_scale,
                    "check_projected_loss_gain":
                        check_loss_before - check_loss_after,
                    "fitted_norm_ratio": projected.fitted_norm_ratio,
                    "relative_residual": projected.relative_residual,
                    "cosine_alignment": projected.cosine_alignment,
                }
            elif args.method == "expand_train_project":
                control = ExpandedTrainProject(
                    args.expanded_train_steps, args.lr, projector)
                result = control.discover(model, candidate, projection_batch)
                control_scale = (1.0 if args.projection_scale == 0
                                 else args.projection_scale)
                check_loss_before = batch_loss(check_batch)
                result.projection.apply_(model, control_scale)
                check_loss_after = batch_loss(check_batch)
                diagnostics = {
                    "source": "repan_bypass_like_control",
                    "signal_source": result.signal.source,
                    "applied_scale": control_scale,
                    "expanded_train_losses": result.expansion_train_losses,
                    "check_projected_loss_gain":
                        check_loss_before - check_loss_after,
                    "fitted_norm_ratio": result.projection.fitted_norm_ratio,
                    "relative_residual": result.projection.relative_residual,
                    "cosine_alignment": result.projection.cosine_alignment,
                }
            elif args.method == "real_e_oracle":
                commit = RealEOracle.commit_(model, candidate)
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=args.lr, momentum=0.9,
                    weight_decay=5e-4)
                diagnostics = {
                    "source": "tiny_gromo_committed_oracle",
                    "deploy_parameter_delta": commit.deploy_parameter_delta,
                }

        model.train()
        loss_sum = correct = count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits.float(), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * targets.numel()
            correct += int((logits.argmax(1) == targets).sum())
            count += targets.numel()
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device)
        row = {
            "epoch": epoch + 1, "train_loss": loss_sum / count,
            "train_accuracy": correct / count,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "diagnostics": diagnostics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    final_parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "method": args.method, "seed": args.seed,
        "initial_model_sha256": initial_sha256,
        "validation_accuracy": history[-1]["validation_accuracy"],
        "best_validation_accuracy": max(x["validation_accuracy"] for x in history),
        "deploy_parameters_before": initial_parameters,
        "deploy_parameters_after": final_parameters,
        "deploy_parameter_delta": final_parameters - initial_parameters,
        "architecture": model.architecture_id,
        "pretrained_backbone": True,
        "initial_hidden_widths": hidden_widths,
        "initial_missing_neurons": missing,
        "split_samples": {
            "statistics": args.statistics_samples,
            "projection": args.projection_samples,
            "check": args.check_samples,
        },
        "elapsed_seconds": time.time() - start,
        "history": history, "config": vars(args),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
