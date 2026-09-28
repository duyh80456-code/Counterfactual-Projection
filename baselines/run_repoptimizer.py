"""Thin CIFAR-100 runner using the official RepOpt-VGG model and optimizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from baselines.official_common import (
    add_common_arguments, cifar100_loaders, parameter_count, run_epochs,
    seed_everything)


def arguments():
    parser = add_common_arguments(argparse.ArgumentParser())
    parser.add_argument("--scales-path", default="")
    return parser.parse_args()


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    seed_everything(args.seed)
    root = Path(args.official_root).resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "repoptimizer"))
    from repoptimizer.repoptvgg_impl import (
        build_RepOptVGG_and_SGD_optimizer_from_pth)

    scales_path = (Path(args.scales_path) if args.scales_path else
                   root / "RepOpt-VGG-B1-scales.pth")
    if not scales_path.is_file():
        raise FileNotFoundError(f"official RepOpt scale checkpoint missing: {scales_path}")
    device = torch.device("cuda:0")
    train_loader, validation_loader, protocol = cifar100_loaders(args)
    protocol.update({
        "architecture": "official RepOpt-VGG-B1 target",
        "num_blocks": [4, 6, 16, 1],
        "width_multiplier": [2, 2, 2, 4],
        "scales_file": scales_path.name,
        "comparison_scope": "adjacent architecture comparator",
    })
    model, optimizer = build_RepOptVGG_and_SGD_optimizer_from_pth(
        [4, 6, 16, 1], [2, 2, 2, 4], str(scales_path),
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
        num_classes=100)
    model = model.to(device)
    # The official builder creates CUDA multiplier tensors before model.to().
    # Kaggle exposes one process-local device, so all tensors already target cuda:0.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    params = parameter_count(model)

    run_epochs(
        args=args, method="repoptimizer", model=model, optimizer=optimizer,
        scheduler=scheduler, train_loader=train_loader,
        validation_loader=validation_loader, protocol=protocol,
        result_extra=lambda _: {
            "peak_train_params": params, "deploy_params": params,
            "grad_multiplier_tensors": len(optimizer.grad_mult_map),
        }, checkpoint_extra={"architecture": "RepOpt-VGG-B1 target"})


if __name__ == "__main__":
    main()
