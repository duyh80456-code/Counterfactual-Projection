"""Plain 200-epoch CIFAR-100 recipe with resumable periodic checkpoints."""
from __future__ import annotations
import argparse
from itertools import islice
from pathlib import Path
import time

import torch
from experiments.shared_protocol import (atomic_torch_save, atomic_json_save,
    make_train_loader, make_eval_loader, train_epoch, evaluate, seed_everything,
    rng_state, restore_rng)
from experiments.vna_support import setup_data, build_model


def run(args):
    device = torch.device(args.device)
    if args.fake_data:
        torch.set_num_threads(1)
    seed_everything(args.seed)
    train, evaluation, indices, trigger, eval_indices, tuning = setup_data(args.data_root, args.fake_data, args.seed)
    model = build_model(device, args.fake_data)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=.1)
    protocol = {"seed": args.seed, "fake_data": args.fake_data, "batch_size": args.batch_size,
                "workers": args.workers, "limit_batches": args.limit_batches,
                "epochs": 200, "architecture": "tiny" if args.fake_data else "resnet18"}
    epoch, history, loader_state = 0, [], None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["protocol"] != protocol:
            raise ValueError("resume protocol mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        epoch, history = checkpoint["epoch"], checkpoint["history"]
        loader_state = checkpoint["train_loader_generator_state"]
        if checkpoint["splits"] != [indices, trigger, eval_indices, tuning]:
            raise ValueError("resume split mismatch")
    loader = make_train_loader(train, indices, args.batch_size, args.workers, loader_state, args.seed + 1)
    eval_loader = make_eval_loader(evaluation, eval_indices, args.batch_size, args.workers)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for current in range(epoch + 1, min(200, args.max_epochs) + 1):
        start = time.perf_counter()
        batches = islice(loader, args.limit_batches) if args.limit_batches else loader
        metrics = train_epoch(model, batches, optimizer, device)
        eval_metrics = evaluate(model, eval_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({"epoch": current, "train": metrics, "eval": eval_metrics,
                        "lr": lr, "seconds": time.perf_counter() - start})
        payload = {"kind": "vna_recipe", "epoch": current, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "rng": rng_state(), "train_loader_generator_state": loader.generator.get_state(),
                   "splits": [indices, trigger, eval_indices, tuning], "protocol": protocol, "history": history}
        atomic_torch_save(payload, output / "latest.pt")
        if current % 10 == 0:
            atomic_torch_save(payload, output / f"epoch_{current:03d}.pt")
        atomic_json_save(history, output / "history.json")
    return output / "latest.pt"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data")
    p.add_argument("--output", default="outputs/vna_recipe")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--limit-batches", type=int, default=0)
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--resume")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
