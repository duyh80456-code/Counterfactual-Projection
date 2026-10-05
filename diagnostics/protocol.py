"""Checkpoint loading and disjoint diagnostic data protocol."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Subset

from experiments.shared_protocol import (
    architecture_label, build_cifar_gromo_resnet, build_optimizer_scheduler,
    datasets_and_indices, sha256_file)
from experiments.plateau_protocol import scheduler_from_state


def load_context(checkpoint, data_root, reference_root, architecture, device):
    sys.path.insert(0, str(Path(reference_root).resolve()))
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if source.get("kind") != "plateau_fork_checkpoint":
        raise ValueError("requires original theta_P plateau_fork_checkpoint")
    if source["protocol"]["architecture"] != architecture_label(architecture):
        raise ValueError("checkpoint architecture mismatch")
    train, evaluation, train_ids, val_ids, tuning_ids = datasets_and_indices(
        data_root, len(source["trigger_indices"]) + len(source["evaluation_indices"]),
        len(source["source_tuning_indices"]))
    if (train_ids != source["train_indices"] or
        val_ids != source["trigger_indices"] + source["evaluation_indices"] or
        tuning_ids != source["source_tuning_indices"]):
        raise ValueError("checkpoint data split mismatch")
    model = build_cifar_gromo_resnet(architecture, torch.device(device))
    model.load_state_dict(source["model"], strict=True)
    optimizer, _ = build_optimizer_scheduler(model, 0.001, 0.0)
    optimizer.load_state_dict(source["optimizer"])
    scheduler = scheduler_from_state(optimizer, source["scheduler"])
    return SimpleNamespace(model=model, optimizer=optimizer, scheduler=scheduler,
        source=source, train_set=train, eval_set=evaluation, device=torch.device(device),
        checkpoint_hash=sha256_file(checkpoint))


def select_batches(context, *, seed=0, statistics_samples=256,
                   where_samples=96, projection_samples=64, heldout_samples=256):
    counts = [statistics_samples, where_samples, projection_samples, heldout_samples]
    if heldout_samples < 256 or any(count < 1 for count in counts):
        raise ValueError("positive split sizes and >=256 heldout samples required")
    train_ids = context.source["train_indices"]
    if len(train_ids) < sum(counts):
        raise ValueError("insufficient training samples for disjoint diagnostic splits")
    order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(seed)).tolist()
    batches, indices, cursor = {}, {}, 0
    for name, count in zip(("statistics", "where", "projection", "heldout"), counts):
        chosen = [train_ids[i] for i in order[cursor:cursor + count]]
        cursor += count
        indices[name] = chosen
        loader = DataLoader(Subset(context.eval_set, chosen), batch_size=(64 if name == "statistics" else count),
                            shuffle=False, num_workers=0)
        moved = [tuple(t.to(context.device) for t in batch) for batch in loader]
        batches[name] = moved if name == "statistics" else moved[0]
    return batches, indices


def propose(model, statistics, rank, site="auto"):
    from experiments.run_shared_comparison import propose_structural_candidates
    return propose_structural_candidates(model, statistics, rank=rank,
                                        site=site, candidate_sites="")
