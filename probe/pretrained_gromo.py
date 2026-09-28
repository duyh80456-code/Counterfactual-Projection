"""Full-width classical Gromo ResNet-18 initialized from torchvision."""

from __future__ import annotations

import torch
from torch import nn


def _copy_module(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict(), strict=True)


def _first(module: nn.Module, kind):
    if isinstance(module, kind):
        return module
    return next(child for child in module.modules()
                if child is not module and isinstance(child, kind))


def _source_blocks(model: nn.Module):
    return [block for stage in (model.layer1, model.layer2,
                                model.layer3, model.layer4)
            for block in stage]


def build_pretrained_gromo_resnet18(
        num_classes: int, device, weights="IMAGENET1K_V1"):
    """Build a full 64/128/256/512 Gromo model and import torchvision weights.

    Imports are delayed so the core package remains usable without optional
    Gromo/torchvision dependencies.
    """
    from torchvision.models import ResNet18_Weights, resnet18
    from gromo.containers.resnet import init_full_resnet_structure
    from dual_growth.adapters import GromoResNet18

    class FullPretrainedGromoResNet18(GromoResNet18):
        architecture_id = "gromo_resnet18_full_imagenet_pretrained_v1"

        def __init__(self):
            # Initialize nn.Module and the helper fields owned by the audited
            # adapter, then replace its CIFAR core with the classical full core.
            super().__init__(1000, 1.0, device=device, use_preactivation=False)
            self.core = init_full_resnet_structure(
                input_shape=(3, 224, 224), out_features=1000,
                reduction_factor=1.0, number_of_blocks_per_stage=2,
                inplanes=64, nb_stages=4, small_inputs=False,
                skip_first_downsample=False, use_preactivation=False,
                device=device)
            self._extended_forward = False
            self._parameter_migrations = []

    if isinstance(weights, str):
        selected_weights = ResNet18_Weights[weights]
    else:
        selected_weights = weights
    source = resnet18(weights=selected_weights).to(device).eval()
    target = FullPretrainedGromoResNet18().to(device).eval()
    _copy_module(target.core.pre_net[0], source.conv1)
    _copy_module(target.core.pre_net[1], source.bn1)
    target_blocks = [reference.module for reference in target.growing_blocks()]
    source_blocks = _source_blocks(source)
    if len(target_blocks) != len(source_blocks):
        raise RuntimeError("Gromo/torchvision ResNet-18 block counts differ")
    for destination, original in zip(target_blocks, source_blocks):
        _copy_module(destination.first_layer.layer, original.conv1)
        _copy_module(_first(destination.first_layer.post_layer_function,
                            nn.BatchNorm2d), original.bn1)
        _copy_module(destination.second_layer.layer, original.conv2)
        _copy_module(_first(destination.second_layer.post_layer_function,
                            nn.BatchNorm2d), original.bn2)
        if original.downsample is not None:
            _copy_module(destination.downsample, original.downsample)
    _copy_module(target.core.post_net[-1], source.fc)

    # Verify the imported backbone before replacing the task head.
    generator = torch.Generator(device=device).manual_seed(1729)
    sample = torch.randn(1, 3, 64, 64, generator=generator, device=device)
    with torch.no_grad():
        reference_logits = source(sample)
        imported_logits = target(sample)
    if not torch.allclose(reference_logits, imported_logits, atol=2e-5, rtol=2e-5):
        error = float((reference_logits - imported_logits).abs().max())
        raise RuntimeError(f"pretrained Gromo import parity failed: max error {error}")

    if num_classes != 1000:
        target.core.post_net[-1] = nn.Linear(512, num_classes, device=device)
    target.num_classes = int(num_classes)
    return target
