"""Thin CIFAR-100 runner around the official RepAn RepVGG operators."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

from baselines.official_common import (
    add_common_arguments, cifar100_loaders, parameter_count, run_epochs,
    seed_everything)


def arguments():
    return add_common_arguments(argparse.ArgumentParser()).parse_args()


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("one visible CUDA GPU is required")
    seed_everything(args.seed)
    root = Path(args.official_root).resolve()
    sys.path.insert(0, str(root))
    from net.repvgg_cifar import RepVGG_A1

    device = torch.device("cuda:0")
    train_loader, validation_loader, protocol = cifar100_loaders(args)
    protocol.update({
        "architecture": "official RepAn RepVGG-A1 CIFAR",
        "operator": "_reparam(first=True) -> inverse_turn_all(1.0)",
        "official_warmup_epochs": 5,
        "annealing_cycle_epochs": 30,
        "criterion": "cross_entropy (official teacher checkpoint unavailable)",
    })
    model = RepVGG_A1(num_classes=100).to(device)

    # Official expansion/restoration smoke test before any training.
    model.eval()
    sample = torch.randn(2, 3, args.image_size, args.image_size, device=device)
    model._reparam(first=True)
    compact_logits = model(sample).detach()
    model.inverse_turn_all(1.0)
    expanded_logits = model(sample).detach()
    model._reparam(first=True)
    restored_logits = model(sample).detach()
    smoke_restoration_error = float((expanded_logits - restored_logits).abs().max())
    if not torch.isfinite(restored_logits).all() or smoke_restoration_error > 1e-4:
        raise RuntimeError(
            f"official RepAn expansion/restoration smoke test failed: {smoke_restoration_error}")
    del compact_logits, expanded_logits, restored_logits
    model._train()

    expanded_params = parameter_count(model)
    # Match the official cycle script's two parameter groups.
    optimizer = torch.optim.SGD([
        {"params": model.weights(rep=False), "weight_decay": args.weight_decay},
        {"params": model.weights(rep=True), "weight_decay": 0.0},
    ], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    def pre_epoch(epoch):
        # Match cycle_repvgg_cifar.py's branch-attachment schedule.
        cycle_epoch = epoch % 30
        if epoch and cycle_epoch == 0:
            model._reparam(first=True)
            model.inverse_turn_all(1.0)
            optimizer.state.clear()  # official script creates a fresh SGD per cycle
        if cycle_epoch > 0:
            model.set_attach_rate(min((cycle_epoch + 1.0) / 5.0, 1.0))

    def post_backward(epoch):
        if epoch % 30 < 5:
            model.freeze_conv3_grad()

    def final_metadata(trained_model):
        training_state = copy.deepcopy(trained_model.state_dict())
        trained_model.eval()
        inputs = torch.randn(2, 3, args.image_size, args.image_size, device=device)
        before = trained_model(inputs).detach()
        trained_model._reparam(first=True)
        after = trained_model(inputs).detach()
        error = float((before - after).abs().max())
        restored_params = sum(
            p.numel() for name, p in trained_model.named_parameters()
            if "rep_conv" in name or name.startswith("fc."))
        torch.save({"model": trained_model.state_dict(), "epoch": args.epochs,
                    "representation": "official reparameterized RepVGG"},
                   Path(args.output) / "restored_model.pt")
        trained_model.load_state_dict(training_state, strict=True)
        trained_model._train()
        return {
            "peak_train_params": expanded_params,
            "deploy_params": restored_params,
            "expanded_params": expanded_params,
            "restored_params": restored_params,
            "restoration_success": bool(error <= 1e-4),
            "restoration_max_abs_error": error,
            "smoke_restoration_max_abs_error": smoke_restoration_error,
        }

    run_epochs(
        args=args, method="repan", model=model, optimizer=optimizer,
        scheduler=scheduler, train_loader=train_loader,
        validation_loader=validation_loader, protocol=protocol,
        result_extra=final_metadata,
        checkpoint_extra={"representation": "expanded RepAn training state"},
        pre_epoch=pre_epoch, post_backward=post_backward)


if __name__ == "__main__":
    main()
