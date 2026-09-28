"""Thin CIFAR-100 runner around official ExpandNets CL and contraction code."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
    cifar_root = root / "exp_cifar"
    sys.path.insert(0, str(cifar_root))
    from models.cifar_tiny_3 import Cifar_Tiny, Cifar_Tiny_ExpandNet_cl
    from utils.compute_new_weights import from_expandnet_cl_to_snet

    device = torch.device("cuda:0")
    train_loader, validation_loader, protocol = cifar100_loaders(args)
    protocol.update({
        "architecture": "official ExpandNets Cifar_Tiny_ExpandNet_cl",
        "expansion": "ExpandNet-CL, expansion ratio 4",
        "model_input_size": 32,
        "input_adapter": "bilinear 128-to-32 required by official fc1 shape",
        "protocol_deviation": "official CIFAR model is intrinsically 32x32",
    })

    def adapt(inputs):
        return F.interpolate(inputs, size=(32, 32), mode="bilinear",
                             align_corners=False)

    model = Cifar_Tiny_ExpandNet_cl(num_classes=100).to(device)
    expanded_params = parameter_count(model)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    # The official contraction routine itself is the smoke test.
    model.eval()
    smoke_input = torch.randn(2, 3, 32, 32, device=device)
    smoke_before = model(smoke_input).detach()
    smoke_compact = from_expandnet_cl_to_snet(
        copy.deepcopy(model).cpu(), Cifar_Tiny(num_classes=100),
        exp_layer_names=["conv1", "conv2", "conv3"]).to(device).eval()
    smoke_after = smoke_compact(smoke_input).detach()
    smoke_error = float((smoke_before - smoke_after).abs().max())
    if smoke_error > 1e-4:
        raise RuntimeError(f"official ExpandNets contraction smoke test failed: {smoke_error}")

    def final_metadata(trained_model):
        trained_model.eval()
        inputs = torch.randn(2, 3, 32, 32, device=device)
        before = trained_model(inputs).detach()
        compact = from_expandnet_cl_to_snet(
            copy.deepcopy(trained_model).cpu(), Cifar_Tiny(num_classes=100),
            exp_layer_names=["conv1", "conv2", "conv3"]).to(device).eval()
        after = compact(inputs).detach()
        error = float((before - after).abs().max())
        compact_params = parameter_count(compact)
        torch.save({"model": compact.state_dict(), "epoch": args.epochs,
                    "representation": "official contracted Cifar_Tiny"},
                   Path(args.output) / "restored_model.pt")
        return {
            "peak_train_params": expanded_params,
            "deploy_params": compact_params,
            "expanded_params": expanded_params,
            "restored_params": compact_params,
            "restoration_success": bool(error <= 1e-4),
            "restoration_max_abs_error": error,
            "smoke_restoration_max_abs_error": smoke_error,
        }

    run_epochs(
        args=args, method="expandnets", model=model, optimizer=optimizer,
        scheduler=scheduler, train_loader=train_loader,
        validation_loader=validation_loader, protocol=protocol,
        input_adapter=adapt, result_extra=final_metadata,
        checkpoint_extra={"representation": "expanded ExpandNet-CL training state"})


if __name__ == "__main__":
    main()
