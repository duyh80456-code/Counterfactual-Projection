"""Plain 200-epoch CIFAR-100 recipe with resumable periodic checkpoints."""
from __future__ import annotations
import argparse
from pathlib import Path
import time

from experiments.cifar_pilot import LimitedLoader
from experiments.vna_provenance import code_commit, resume_code_history
import torch
from experiments.shared_protocol import (atomic_torch_save, atomic_json_save,
    make_train_loader, make_eval_loader, evaluate, seed_everything,
    rng_state, restore_rng, validate_resume_protocol)
from experiments.vna_support import setup_data, build_model, DeviceCIFAR, DeviceTrainLoader, fast_train_epoch


def run(args):
    device = torch.device(args.device)
    if args.fake_data:
        torch.set_num_threads(1)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA fp16")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    seed_everything(args.seed)
    train, evaluation, indices, trigger, eval_indices, tuning = setup_data(args.data_root, args.fake_data, args.seed)
    architecture = getattr(args, 'architecture', 'resnet18')
    if architecture == 'resnet18-quarter':
        from experiments.stage_b4 import build_model as quarter_model
        model = quarter_model(device)
        if args.fake_data:
            from torch.utils.data import TensorDataset
            train = evaluation = TensorDataset(torch.nn.functional.interpolate(train.tensors[0], size=32), train.tensors[1])
    else:
        model = build_model(device, args.fake_data)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=.1)
    protocol = {"seed": args.seed, "fake_data": args.fake_data, "batch_size": args.batch_size,
                "workers": args.workers, "limit_batches": args.limit_batches,
                "gpu_data": args.gpu_data, "amp": args.amp, "channels_last": args.channels_last,
                "cudnn_benchmark": args.cudnn_benchmark, "epochs": 200, "architecture": architecture if architecture == "resnet18-quarter" else "tiny" if args.fake_data else "resnet18"}
    commit = code_commit(); code_history = [{"from_epoch": 1, "commit": commit}]
    epoch, history, loader_state = 0, [], None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_protocol(checkpoint["protocol"], protocol)
        code_history = resume_code_history(checkpoint, commit)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng"])
        scaler.load_state_dict(checkpoint.get("scaler", {})) if args.amp else None
        epoch, history = checkpoint["epoch"], checkpoint["history"]
        loader_state = checkpoint["train_loader_generator_state"]
        if checkpoint["splits"] != [indices, trigger, eval_indices, tuning]:
            raise ValueError("resume split mismatch")
    loader = make_train_loader(train, indices, args.batch_size, args.workers, loader_state, args.seed + 1)
    if args.gpu_data:
        if args.fake_data:
            raise ValueError("--gpu-data requires uint8 CIFAR, not fake tensors")
        loader = DeviceTrainLoader(DeviceCIFAR(train, device), indices, args.batch_size, args.seed + 1, loader_state)
    eval_loader = make_eval_loader(evaluation, eval_indices, args.batch_size, args.workers)
    if getattr(args, 'eval_limit_batches', None):
        eval_loader = LimitedLoader(eval_loader, args.eval_limit_batches)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for current in range(epoch + 1, min(200, args.max_epochs) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        batches = LimitedLoader(loader, args.limit_batches) if args.limit_batches else loader
        metrics = fast_train_epoch(model, batches, optimizer, device, amp=args.amp,
                                   scaler=scaler, channels_last=args.channels_last)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - start
        eval_metrics = evaluate(model, eval_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append({"code_commit": commit, "code_history": list(code_history), "epoch": current, "train": metrics, "eval": eval_metrics,
                        "lr": lr, "train_seconds": train_seconds, "seconds": time.perf_counter() - start})
        payload = {"kind": "vna_recipe", "epoch": current, "model": model.state_dict(),
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "scheduler": scheduler.state_dict(),
                   "rng": rng_state(), "train_loader_generator_state": loader.generator.get_state(),
                   "splits": [indices, trigger, eval_indices, tuning], "protocol": protocol, "history": history,
                   "code_commit": commit, "code_history": code_history}
        atomic_torch_save(payload, output / "latest.pt")
        if current % 10 == 0 or current in (98, 148) or architecture == "resnet18-quarter" and current == args.max_epochs:
            # Periodic diagnostics need only fp32 model weights, never momentum.
            atomic_torch_save({k: payload[k] for k in ("kind", "epoch", "model", "protocol", "code_commit", "code_history", "history")},
                              output / f"epoch_{current:03d}.pt")
        atomic_json_save(history, output / "history.json")
        print({"epoch": current, "train_seconds": train_seconds,
               "checkpoint_bytes": sum(p.stat().st_size for p in output.glob("*.pt"))}, flush=True)
    if epoch >= 200 or args.max_epochs >= 200:
        atomic_json_save({"epoch": 200}, output / "training_done.json")
    return output / "latest.pt"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=("resnet18", "resnet18-quarter"), default="resnet18")
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
    p.add_argument("--gpu-data", action="store_true", help="resident uint8 CIFAR and generator-only device augmentation; use identically in paired arms")
    p.add_argument("--amp", action="store_true", help="CUDA fp16 SGD only (default off)")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--cudnn-benchmark", action="store_true", help="faster CUDA algorithm selection; default off for deterministic comparisons")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
