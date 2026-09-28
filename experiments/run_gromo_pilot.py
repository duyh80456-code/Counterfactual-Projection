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
from probe import CandidateExpansionProbe, TransactionalCandidateSource
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
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--site", default="stages.2.blocks.0")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--start-width", type=float, default=0.25)
    parser.add_argument("--statistics-batches", type=int, default=1)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--cg-iterations", type=int, default=12)
    parser.add_argument("--projection-scale", type=float, default=1.0)
    parser.add_argument("--expanded-train-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    return parser.parse_args()


def loaders(args):
    from torchvision import datasets, transforms

    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR100(
        args.data_root, train=True, transform=train_transform, download=False)
    validation_set = datasets.CIFAR100(
        args.data_root, train=True, transform=eval_transform, download=False)
    order = torch.randperm(
        len(train_set), generator=torch.Generator().manual_seed(20260928)).tolist()
    validation_indices = order[:args.validation_samples]
    train_indices = order[args.validation_samples:]
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
    return train, validation


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
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").is_file():
        print("completed result exists; skipping", flush=True)
        return
    reference = Path(args.reference_root).resolve()
    if not (reference / "dual_growth").is_dir():
        raise FileNotFoundError(f"invalid One-Shot-TAS-CCIL checkout: {reference}")
    sys.path.insert(0, str(reference))
    from dual_growth.adapters import GromoResNet18, TinyAdapter
    from dual_growth.controller import GrowthBudget

    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    train_loader, validation_loader = loaders(args)
    model = GromoResNet18(
        100, args.start_width, device=device, use_preactivation=True).to(device)
    initial_sha256 = state_sha256(model)
    initial_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    projector = FunctionalProjector(
        args.damping, args.cg_iterations, tolerance=1e-5)
    e_projection = EProjection(projector=projector)
    candidate_source = TransactionalCandidateSource()
    history = []
    start = time.time()

    def propose(batch):
        adapter = TinyAdapter(
            quantum_params=10**9,
            max_statistics_batches=args.statistics_batches)
        adapter.schedule_site(args.site, args.rank)
        return candidate_source.propose(
            adapter, model, [batch], GrowthBudget(10**9))[0]

    for epoch in range(args.epochs):
        probe_batch = next(iter(train_loader))
        probe_batch = tuple(value.to(device, non_blocking=True)
                            for value in probe_batch)
        diagnostics = None
        if args.method != "vanilla" and not (
                args.method == "real_e_oracle" and epoch > 0):
            candidate = propose(probe_batch)
            if args.method == "tiny_projection":
                step = e_projection.step_candidate_(
                    model, candidate, probe_batch, scale=args.projection_scale)
                diagnostics = {
                    "source": step.signal.source,
                    "structural_loss_gain": step.structural_loss_gain,
                    "projected_loss_gain": step.projected_loss_gain,
                    "fitted_norm_ratio": step.projection.fitted_norm_ratio,
                    "relative_residual": step.projection.relative_residual,
                    "cosine_alignment": step.projection.cosine_alignment,
                }
            elif args.method == "random_projection":
                signal = CandidateExpansionProbe()(
                    model, candidate=candidate, batch=probe_batch)
                random_delta = torch.randn_like(signal.delta_logits)
                random_delta.mul_(signal.delta_logits.norm() /
                                  random_delta.norm().clamp_min(1e-12))
                block = candidate_projection_block(model, candidate)
                projected = projector.project(
                    model, probe_batch[0], random_delta, block=block)
                projected.apply_(model, args.projection_scale)
                diagnostics = {
                    "source": "random_matched_to_structural_norm",
                    "fitted_norm_ratio": projected.fitted_norm_ratio,
                    "relative_residual": projected.relative_residual,
                    "cosine_alignment": projected.cosine_alignment,
                }
            elif args.method == "expand_train_project":
                control = ExpandedTrainProject(
                    args.expanded_train_steps, args.lr, projector)
                result = control.discover(model, candidate, probe_batch)
                result.projection.apply_(model, args.projection_scale)
                diagnostics = {
                    "source": result.signal.source,
                    "expanded_train_losses": result.expansion_train_losses,
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
        "elapsed_seconds": time.time() - start,
        "history": history, "config": vars(args),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
