"""AdamW, deterministic data streams and checkpoint protocol for DeiT forks."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from models import DeiTTinyCifar
from adapters.deit_cp_adapter import CPConfig, probe_indices
from experiments.shared_protocol import (atomic_torch_save, datasets_and_indices,
    evaluate, make_train_loader, make_eval_loader, rng_state, restore_rng)

ARCHITECTURE = "CIFAR-DeiT-Tiny-Patch4"
PROTOCOL_VERSION = 3


@dataclass(frozen=True)
class DeitRecipe:
    seed: int = 1
    batch_size: int = 128
    workers: int = 2
    learning_rate: float = 5e-4
    weight_decay: float = .05
    schedule_epochs: int = 400
    warmup_epochs: int = 5
    min_lr_ratio: float = .01
    validation_samples: int = 5000
    trigger_samples: int = 2000
    tuning_samples: int = 128
    min_plateau_epoch: int = 100
    patience: int = 20
    max_epoch: int = 300

    def validate(self):
        if not (self.batch_size > 0 and self.workers >= 0 and self.learning_rate > 0 and
                self.weight_decay >= 0 and 0 <= self.warmup_epochs < self.schedule_epochs and
                0 <= self.min_lr_ratio <= 1 and self.validation_samples > 0 and
                0 < self.trigger_samples < self.validation_samples and
                self.tuning_samples >= 0 and self.patience > 0 and
                self.schedule_epochs >= self.max_epoch >= self.min_plateau_epoch >= 1):
            raise ValueError("invalid DeiT baseline recipe")


def build_optimizer_scheduler(model, recipe):
    recipe.validate()
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim == 1 or name in {"pos_embed", "cls_token"}
         else decay).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": recipe.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}],
        lr=recipe.learning_rate, betas=(.9, .999))

    def multiplier(epoch):
        if epoch < recipe.warmup_epochs:
            return (epoch + 1) / max(1, recipe.warmup_epochs)
        progress = min(1., (epoch - recipe.warmup_epochs) /
                       (recipe.schedule_epochs - recipe.warmup_epochs))
        return recipe.min_lr_ratio + (1 - recipe.min_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def protocol(recipe, model):
    return {"protocol_version": PROTOCOL_VERSION, "architecture": ARCHITECTURE,
            "dataset": "CIFAR-100", "seed": recipe.seed, "recipe": asdict(recipe),
            "model_config": getattr(model, "config", model), "pretrained": False, "distilled": False,
            "drop_rate": 0., "drop_path_rate": 0.,
            "optimizer": "AdamW", "scheduler": "warmup_then_cosine_fixed_global_schedule",
            "theta_P_rule": "historical strict evaluation accuracy best after no-new-best patience",
            "selection_metric": "evaluation split accuracy",
            "evaluation_samples": recipe.validation_samples - recipe.trigger_samples,
            "report_validation_role": "historical_best_and_plateau_selection_then_fixed_horizon_reporting",
            "unused_trigger_split_role": "reserved_excluded_from_training_and_selection",
            "official_test_used": False}


def checked_source(path, kinds):
    source = torch.load(path, map_location="cpu", weights_only=False)
    declared = source.get("protocol", {})
    if (source.get("kind") not in kinds or declared.get("architecture") != ARCHITECTURE or
            declared.get("protocol_version") != PROTOCOL_VERSION):
        raise ValueError("incompatible DeiT checkpoint kind/architecture/protocol")
    recipe = DeitRecipe(**declared["recipe"])
    recipe.validate()
    if declared != protocol(recipe, declared["model_config"]):
        raise ValueError("incompatible DeiT recipe or architecture flags")
    if declared["model_config"] != canonical_model_config():
        raise ValueError("training requires canonical DeiT-Tiny CIFAR geometry")
    if source["kind"] == "deit_plateau_fork":
        historical_best(source)
    return source, recipe


def historical_best(source):
    """Require theta_P to be the selected historical validation-best state."""
    accuracy = float(source["historical_best_accuracy"])
    epoch = int(source["historical_best_epoch"])
    loss = float(source["historical_best_loss"])
    history = source["history"]
    if (not history or epoch != source["epoch"] or
            history[-1]["epoch"] != epoch or
            history[-1]["validation_accuracy"] != accuracy or
            history[-1]["validation_loss"] != loss or
            max(row["validation_accuracy"] for row in history) != accuracy or
            any(row["validation_accuracy"] >= accuracy for row in history[:-1]) or
            any(row["validation_accuracy"] > accuracy for row in source.get("stall_history", []))):
        raise ValueError("theta_P is not the historical validation-best checkpoint")
    return accuracy, loss, epoch


def canonical_model_config():
    return dict(num_classes=100, image_size=32, patch_size=4, embed_dim=192,
                depth=12, num_heads=3, mlp_ratio=4)


def load_training_context(data_root, recipe, device, source=None):
    train, evaluation, train_ids, validation_ids, tuning_ids = datasets_and_indices(
        data_root, recipe.validation_samples, recipe.tuning_samples)
    trigger_ids = validation_ids[:recipe.trigger_samples]
    validation_ids = validation_ids[recipe.trigger_samples:]
    if source is not None:
        if any(source[key] != value for key, value in (
                ("train_indices", train_ids), ("evaluation_indices", validation_ids),
                ("trigger_indices", trigger_ids),
                ("source_tuning_indices", tuning_ids))):
            raise ValueError("DeiT checkpoint data split mismatch")
    model = DeiTTinyCifar().to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, recipe)
    if source is not None:
        model.load_state_dict(source["model"], strict=True)
        optimizer.load_state_dict(source["optimizer"])
        scheduler.load_state_dict(source["scheduler"])
    train_loader = make_train_loader(train, train_ids, recipe.batch_size, recipe.workers,
        None if source is None else source["train_loader_generator_state"], recipe.seed)
    eval_loader = make_eval_loader(evaluation, validation_ids, recipe.batch_size, recipe.workers)
    if source is not None:
        restore_rng(source["rng"])
    return model, optimizer, scheduler, train_loader, eval_loader, evaluation, train_ids, validation_ids, tuning_ids, trigger_ids


def evaluate_without_rng(model, loader, device):
    rng = rng_state()
    try:
        return evaluate(model, loader, device)
    finally:
        restore_rng(rng)


def save_state(path, *, model, optimizer, scheduler, loader, epoch, history,
               train_indices, evaluation_indices, source_tuning_indices, trigger_indices, run_protocol,
               kind, **extra):
    atomic_torch_save({"format_version": 1, "kind": kind, "epoch": epoch,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "rng": rng_state(),
        "train_loader_generator_state": loader.generator.get_state(),
        "history": history, "train_indices": train_indices,
        "evaluation_indices": evaluation_indices,
        "trigger_indices": trigger_indices,
        "source_tuning_indices": source_tuning_indices, "protocol": run_protocol,
        **extra}, Path(path))


def materialize_probe_batches(evaluation, train_ids, recipe, config, device):
    indices = probe_indices(train_ids, recipe.seed, config)

    def load(ids, batch_size):
        loader = DataLoader(Subset(evaluation, ids), batch_size=batch_size,
            shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(recipe.seed))
        return [tuple(value.to(device) for value in batch) for batch in loader]

    return {"statistics": load(indices["statistics"], recipe.batch_size),
            "where_batches": [load(ids, len(ids))[0] for ids in indices["where"]],
            "projection_batch": load(indices["projection"], len(indices["projection"]))[0],
            "gate_batch": load(indices["gate"], len(indices["gate"]))[0]}, indices
