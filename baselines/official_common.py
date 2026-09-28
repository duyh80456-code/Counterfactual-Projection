"""Shared experiment plumbing for thin wrappers around official repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset


SPLIT_SEED = 20260928


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80,
                        help="target epoch; an existing checkpoint resumes to this value")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--tuning-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--license", required=True)
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def index_sha256(indices) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(int(index).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def cifar100_loaders(args):
    """Use the exact split and transforms of the main Gromo pilot."""
    from torchvision import datasets, transforms

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    eval_transform = transforms.Compose([
        transforms.Resize(args.image_size + 16),
        transforms.CenterCrop(args.image_size), transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR100(
        args.data_root, train=True, transform=train_transform, download=False)
    eval_set = datasets.CIFAR100(
        args.data_root, train=True, transform=eval_transform, download=False)
    order = torch.randperm(
        len(train_set), generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
    validation_indices = order[:args.validation_samples]
    tuning_end = args.validation_samples + args.tuning_samples
    train_indices = order[tuning_end:]
    if args.train_samples:
        train_indices = train_indices[:args.train_samples]
    train_loader = DataLoader(
        Subset(train_set, train_indices), args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.workers, pin_memory=True, persistent_workers=False)
    validation_loader = DataLoader(
        Subset(eval_set, validation_indices), args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    protocol = {
        "dataset": "CIFAR-100", "seed": args.seed,
        "split_seed": SPLIT_SEED, "batch_size": args.batch_size,
        "image_size": args.image_size, "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "tuning_samples_excluded": args.tuning_samples,
        "train_indices_sha256": index_sha256(train_indices),
        "validation_indices_sha256": index_sha256(validation_indices),
        "official_test_used": False,
        "training_schedule": "constant learning rate; phase-continuable",
    }
    return train_loader, validation_loader, protocol


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_epoch(model, loader, optimizer, device,
                input_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None,
                post_backward: Callable[[], None] | None = None):
    model.train()
    loss_sum = correct = count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        if input_adapter is not None:
            inputs = input_adapter(inputs)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        if post_backward is not None:
            post_backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * targets.numel()
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return loss_sum / count, correct / count


@torch.no_grad()
def evaluate(model, loader, device,
             input_adapter: Callable[[torch.Tensor], torch.Tensor] | None = None):
    model.eval()
    loss_sum = correct = count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        if input_adapter is not None:
            inputs = input_adapter(inputs)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += float(F.cross_entropy(logits.float(), targets, reduction="sum"))
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return loss_sum / count, correct / count


def rng_state() -> dict:
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def load_resume(checkpoint_path: Path, model, optimizer, scheduler, protocol,
                train_loader):
    if not checkpoint_path.is_file():
        return 0, [], 0.0, 0
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["protocol"] != protocol:
        raise RuntimeError("resume checkpoint protocol does not match this run")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng(checkpoint["rng"])
    if "train_loader_generator_state" in checkpoint:
        train_loader.generator.set_state(
            checkpoint["train_loader_generator_state"])
    return int(checkpoint["epoch"]), checkpoint["history"], float(
        checkpoint.get("training_seconds", 0.0)), int(
        checkpoint.get("peak_gpu_memory", 0))


def save_training_checkpoint(path: Path, epoch: int, model, optimizer,
                             scheduler, history, protocol, training_seconds,
                             train_loader, peak_gpu_memory: int,
                             extra: dict | None = None) -> None:
    payload = {
        "format_version": 1, "epoch": epoch,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "history": history,
        "protocol": protocol, "rng": rng_state(),
        "training_seconds": training_seconds,
        "train_loader_generator_state": train_loader.generator.get_state(),
        "peak_gpu_memory": peak_gpu_memory,
    }
    if extra:
        payload["extra"] = extra
    atomic_torch_save(payload, path)


def run_epochs(*, args, method: str, model, optimizer, scheduler,
               train_loader, validation_loader, protocol,
               input_adapter=None, result_extra=None, checkpoint_extra=None,
               pre_epoch=None, post_backward=None):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint_latest.pt"
    initial_epoch, history, prior_seconds, prior_peak_memory = load_resume(
        checkpoint_path, model, optimizer, scheduler, protocol, train_loader)
    if initial_epoch > args.epochs:
        raise RuntimeError(
            f"checkpoint epoch {initial_epoch} exceeds target {args.epochs}")
    device = next(model.parameters()).device
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(initial_epoch, args.epochs):
        if pre_epoch is not None:
            pre_epoch(epoch)
        train_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, device, input_adapter,
            None if post_backward is None else lambda: post_backward(epoch))
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, device, input_adapter)
        scheduler.step()
        row = {
            "epoch": epoch + 1, "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        elapsed = prior_seconds + time.perf_counter() - started
        peak_memory = max(
            prior_peak_memory, int(torch.cuda.max_memory_allocated(device)))
        save_training_checkpoint(
            checkpoint_path, epoch + 1, model, optimizer, scheduler, history,
            protocol, elapsed, train_loader, peak_memory, checkpoint_extra)
        atomic_json_save({"method": method, "target_epochs": args.epochs,
                          "completed_epochs": epoch + 1, "latest": row},
                         output / "progress.json")
        print(json.dumps({method: row}, sort_keys=True), flush=True)
    elapsed = prior_seconds + time.perf_counter() - started
    last = history[-1]
    result = {
        "method": method, "seed": args.seed, "epoch": initial_epoch
        if not history else int(last["epoch"]),
        "train_accuracy": last["train_accuracy"],
        "validation_accuracy": last["validation_accuracy"],
        "validation_loss": last["validation_loss"],
        "training_seconds": elapsed,
        "peak_gpu_memory": max(
            prior_peak_memory, int(torch.cuda.max_memory_allocated(device))),
        "source_repo": args.source_url, "source_commit": args.source_commit,
        "source_license": args.license, "protocol": protocol,
        "history": history, "checkpoint": str(checkpoint_path),
    }
    if result_extra:
        result.update(result_extra(model))
    atomic_json_save(result, output / "result.json")
    return result
