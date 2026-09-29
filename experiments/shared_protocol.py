"""Shared CIFAR-100 fork protocol for Vanilla, projection arms, and Bypass."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset


SPLIT_SEED = 20260928
BOOTSTRAP_EPOCH = 150
FORK_EPOCH = 300
TOTAL_EPOCHS = 360
POST_FORK_EPOCHS = TOTAL_EPOCHS - FORK_EPOCH


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_sha256(indices) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(int(index).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


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


def rng_state() -> dict:
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if not torch.cuda.is_available():
        return
    # ``torch.load(..., map_location=device)`` recursively moves checkpoint
    # tensors, including RNG snapshots, onto CUDA. PyTorch's RNG restoration
    # API explicitly requires CPU uint8 tensors. A shared checkpoint may also
    # have been written with a different number of visible GPUs than an arm.
    cuda_states = [value.detach().cpu() for value in state.get("cuda", [])]
    visible_devices = torch.cuda.device_count()
    if len(cuda_states) == visible_devices:
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states:
        for device_index in range(visible_devices):
            torch.cuda.set_rng_state(
                cuda_states[min(device_index, len(cuda_states) - 1)],
                device=device_index)


def datasets_and_indices(data_root: str, validation_samples: int = 5000,
                         tuning_samples: int = 128):
    from torchvision import datasets, transforms

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    eval_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR100(
        data_root, train=True, transform=train_transform, download=False)
    eval_set = datasets.CIFAR100(
        data_root, train=True, transform=eval_transform, download=False)
    order = torch.randperm(
        len(train_set), generator=torch.Generator().manual_seed(SPLIT_SEED)).tolist()
    validation_indices = order[:validation_samples]
    tuning_indices = order[validation_samples:validation_samples + tuning_samples]
    train_indices = order[validation_samples + tuning_samples:]
    return train_set, eval_set, train_indices, validation_indices, tuning_indices


def make_train_loader(dataset, indices, batch_size: int, workers: int,
                      generator_state=None, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    if generator_state is not None:
        generator.set_state(generator_state.cpu())
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=workers, pin_memory=True,
        persistent_workers=False)


def make_eval_loader(dataset, indices, batch_size: int, workers: int):
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def build_cifar_gromo_resnet18(device):
    from dual_growth.adapters import GromoResNet18

    model = GromoResNet18(
        num_classes=100, start_width=1.0, device=device,
        use_preactivation=False).to(device)
    model.architecture_id = "cifar_gromo_resnet18_full_random_v1"
    widths = [int(ref.module.hidden_neurons) for ref in model.growing_blocks()]
    if widths != [64, 64, 128, 128, 256, 256, 512, 512]:
        raise RuntimeError(f"unexpected full CIFAR-ResNet18 widths: {widths}")
    return model


def build_optimizer_scheduler(model, lr: float = 0.1,
                              weight_decay: float = 5e-4):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS)
    return optimizer, scheduler


def rebase_scheduler_from_theta150(optimizer):
    """Keep theta_150's LR continuous and anneal it to the configured end."""
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS - BOOTSTRAP_EPOCH)


def rebase_scheduler_from_fork(optimizer, post_fork_epochs=POST_FORK_EPOCHS):
    """Start a common cosine segment from the exact LR stored at the fork."""
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=post_fork_epochs)


def protocol(seed, train_indices, validation_indices, tuning_indices,
             batch_size, lr, weight_decay):
    return {
        "dataset": "CIFAR-100", "architecture": "CIFAR-ResNet18",
        "input_size": 32, "seed": seed, "split_seed": SPLIT_SEED,
        "fork_epoch": FORK_EPOCH, "total_epochs": TOTAL_EPOCHS,
        "post_fork_epochs": POST_FORK_EPOCHS, "batch_size": batch_size,
        "learning_rate": lr, "weight_decay": weight_decay,
        "scheduler": (
            "shared theta300 LR; post-fork CosineAnnealingLR(T_max=60) "
            "through epoch360"),
        "lr_schedule_status": (
            "post-fork scheduler rebased identically for every arm; not "
            "equivalent to a fresh CosineAnnealingLR(T_max=360) run"),
        "train_indices_sha256": index_sha256(train_indices),
        "validation_indices_sha256": index_sha256(validation_indices),
        "tuning_indices_sha256": index_sha256(tuning_indices),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "tuning_samples": len(tuning_indices), "official_test_used": False,
    }


def assert_fork_protocol_compatible(checkpoint_protocol, run_protocol):
    """Validate theta_300 while allowing a newly defined post-fork budget."""
    post_fork_fields = {
        "total_epochs", "post_fork_epochs", "scheduler",
        "lr_schedule_status",
    }
    expected = {
        key: value for key, value in run_protocol.items()
        if key not in post_fork_fields
    }
    actual = {
        key: value for key, value in checkpoint_protocol.items()
        if key not in post_fork_fields
    }
    if actual != expected:
        differing = sorted(
            key for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key))
        raise RuntimeError(
            f"shared checkpoint base protocol differs at {differing}")


def train_epoch(model, loader, optimizer, device, loss_extra=None,
                post_step=None):
    model.train()
    loss_sum = task_loss_sum = correct = count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        task_loss = F.cross_entropy(logits.float(), targets)
        extra = (task_loss.new_zeros(()) if loss_extra is None
                 else loss_extra())
        loss = task_loss + extra
        loss.backward()
        optimizer.step()
        if post_step is not None:
            post_step()
        loss_sum += float(loss.detach()) * targets.numel()
        task_loss_sum += float(task_loss.detach()) * targets.numel()
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return {
        "loss": loss_sum / count, "task_loss": task_loss_sum / count,
        "accuracy": correct / count,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss_sum += float(F.cross_entropy(
            logits.float(), targets, reduction="sum"))
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return {"loss": loss_sum / count, "accuracy": correct / count}


def save_shared_checkpoint(path: Path, *, model, optimizer, scheduler, epoch,
                           train_indices, validation_indices, tuning_indices,
                           loader, history, run_protocol) -> str:
    atomic_torch_save({
        "format_version": 1, "kind": "shared_fork_checkpoint",
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "history": history,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "tuning_indices": tuning_indices, "rng": rng_state(),
        "train_loader_generator_state": loader.generator.get_state(),
        "protocol": run_protocol,
    }, path)
    digest = sha256_file(path)
    atomic_json_save({"checkpoint": str(path), "sha256": digest,
                      "epoch": epoch, "protocol": run_protocol},
                     path.with_suffix(".json"))
    return digest


def load_shared_checkpoint(path: Path, expected_hash: str, *, device,
                           model, optimizer, scheduler):
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"shared checkpoint hash mismatch: {actual_hash} != {expected_hash}")
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("kind") != "shared_fork_checkpoint":
        raise RuntimeError("not a shared fork checkpoint")
    if int(checkpoint["epoch"]) != FORK_EPOCH:
        raise RuntimeError(f"shared checkpoint is not at epoch {FORK_EPOCH}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    restore_rng(checkpoint["rng"])
    return checkpoint, actual_hash
