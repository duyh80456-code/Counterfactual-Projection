"""Single-GPU CIFAR-100 pilot arm for Kaggle T4 workers."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from methods import EProjection, ERepOpt
from probe import VirtualExpansionProbe
from projection import FunctionalProjector


METHODS = (
    "vanilla", "gradient_random_projection", "gradient_repopt_control",
    "gradient_projection_control")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--block", default="layer3.1.conv2")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--probe-scale", type=float, default=0.01)
    parser.add_argument("--projection-interval", type=int, default=1)
    parser.add_argument("--projection-scale", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--cg-iterations", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--train-samples", type=int, default=0,
                        help="0 uses the complete training partition")
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def datasets(args):
    from torchvision import datasets as tv_datasets, transforms

    # Match the initialization: ImageNet-pretrained weights must receive the
    # normalization they were trained with. CIFAR statistics are used only
    # for the explicitly random-initialized control.
    mean, std = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) \
        if args.no_pretrained else ((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225))
    normalize = transforms.Normalize(mean, std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([
        transforms.Resize(args.image_size + 16),
        transforms.CenterCrop(args.image_size), transforms.ToTensor(), normalize])
    train_data = tv_datasets.CIFAR100(args.data_root, train=True,
                                     transform=train_transform, download=False)
    validation_data = tv_datasets.CIFAR100(
        args.data_root, train=True, transform=eval_transform, download=False)
    generator = torch.Generator().manual_seed(20260928)
    indices = torch.randperm(len(train_data), generator=generator).tolist()
    validation_indices = indices[:args.validation_samples]
    train_indices = indices[args.validation_samples:]
    if args.train_samples:
        train_indices = train_indices[:args.train_samples]
    return Subset(train_data, train_indices), Subset(
        validation_data, validation_indices), train_indices, validation_indices


def build_model(args, device):
    from torchvision.models import ResNet18_Weights, resnet18

    weights = None if args.no_pretrained else ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 100)
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = count = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss_sum += float(F.cross_entropy(logits, targets, reduction="sum"))
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return {"loss": loss_sum / count, "accuracy": correct / count}


def random_target(signal, generator):
    target = torch.randn(signal.delta_logits.shape, device=signal.delta_logits.device,
                         dtype=signal.delta_logits.dtype, generator=generator)
    return target * (signal.delta_logits.norm() / target.norm().clamp_min(1e-12))


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        print(f"completed result exists: {result_path}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("this pilot expects one visible CUDA device")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    train_data, validation_data, train_indices, validation_indices = datasets(args)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, args.batch_size, shuffle=True, generator=loader_generator,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    validation_loader = DataLoader(
        validation_data, args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    model = build_model(args, device)
    deploy_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    scaler = torch.cuda.amp.GradScaler()
    probe = VirtualExpansionProbe(args.probe_scale)
    projector = FunctionalProjector(
        args.damping, args.cg_iterations, tolerance=1e-5)
    e_projection = EProjection(probe, projector)
    history = []
    start = time.time()

    for epoch in range(args.epochs):
        projection_metrics = None
        probe_batch = next(iter(train_loader))
        probe_batch = tuple(item.to(device, non_blocking=True) for item in probe_batch)
        repopt_handler = None
        if epoch % args.projection_interval == 0 and args.method != "vanilla":
            signal = probe(model, block=args.block, rank=args.rank, batch=probe_batch)
            if args.method == "gradient_repopt_control":
                repopt_handler = ERepOpt.from_signal(
                    optimizer, model, signal).handler
                projection_metrics = {"predicted_gain": signal.predicted_gain}
            else:
                target = signal.delta_logits
                if args.method == "gradient_random_projection":
                    random_generator = torch.Generator(device=device).manual_seed(
                        100000 + args.seed * 1000 + epoch)
                    target = random_target(signal, random_generator)
                projection = projector.project(model, probe_batch[0], target,
                                               block=args.block)
                projection.apply_(model, args.projection_scale)
                projection_metrics = {
                    "predicted_gain": signal.predicted_gain,
                    "fitted_norm_ratio": projection.fitted_norm_ratio,
                    "cosine_alignment": projection.cosine_alignment,
                    "relative_residual": projection.relative_residual,
                    "cg_iterations": projection.cg.iterations,
                    "cg_converged": projection.cg.converged,
                }

        model.train()
        loss_sum = correct = count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if repopt_handler is not None:
                repopt_handler.apply()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * targets.numel()
            correct += int((logits.argmax(1) == targets).sum())
            count += targets.numel()
        scheduler.step()
        validation = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch + 1, "train_loss": loss_sum / count,
            "train_accuracy": correct / count,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "projection": projection_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1, "args": vars(args)}, output / "latest.pth")

    result = {
        "method": args.method, "seed": args.seed,
        "validation_accuracy": history[-1]["validation_accuracy"],
        "validation_loss": history[-1]["validation_loss"],
        "best_validation_accuracy": max(row["validation_accuracy"] for row in history),
        "deploy_parameters_before": deploy_parameters,
        "deploy_parameters_after": sum(p.numel() for p in model.parameters()),
        "deploy_parameter_delta": sum(p.numel() for p in model.parameters()) - deploy_parameters,
        "elapsed_seconds": time.time() - start,
        "train_indices": len(train_indices),
        "validation_indices": len(validation_indices),
        "history": history,
        "config": vars(args),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
