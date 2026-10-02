"""DenseBlock-boundary counterfactual auxiliary spaces for DenseNet-121."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DenseBlockRef:
    stage: int
    block: int
    name: str
    module: nn.Module


class TransitionAuxiliary(nn.Module):
    def __init__(self, in_channels, out_channels, rank):
        super().__init__()
        self.branch = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, rank, kernel_size=1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(rank, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, inputs):
        return self.branch(inputs)


class ClassifierAuxiliary(nn.Module):
    def __init__(self, in_channels, num_classes, rank):
        super().__init__()
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, rank, kernel_size=1, bias=False),
            nn.ReLU(inplace=False))
        self.classifier = nn.Linear(rank, num_classes)

    def forward(self, inputs):
        features = self.features(inputs)
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return self.classifier(pooled)


class CifarDenseNet121(nn.Module):
    """CIFAR-adapted DenseNet-121 with four block-boundary E sites."""

    architecture_id = "cifar_densenet121_block_boundary_random_v1"

    def __init__(self, num_classes=100, device="cpu"):
        super().__init__()
        from torchvision.models.densenet import DenseNet

        self.core = DenseNet(
            growth_rate=32, block_config=(6, 12, 24, 16),
            num_init_features=64, bn_size=4, drop_rate=0,
            num_classes=num_classes, memory_efficient=False).to(device)
        self.core.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False,
            device=device)
        self.core.features.pool0 = nn.Identity()
        self._active_site = None
        self._active_gate = 0.0
        self._active_auxiliary = None
        self._block_names = tuple(
            f"core.features.denseblock{index}" for index in range(1, 5))

    def growing_blocks(self):
        return [DenseBlockRef(
            index - 1, index - 1, name,
            getattr(self.core.features, f"denseblock{index}"))
            for index, name in enumerate(self._block_names, start=1)]

    def block(self, name):
        if name not in self._block_names:
            raise KeyError(name)
        return dict(self.named_modules())[name]

    def boundary_shapes(self, name):
        index = self._block_names.index(name) + 1
        if index < 4:
            transition = getattr(self.core.features, f"transition{index}")
            in_channels = int(transition.norm.num_features)
            out_channels = int(transition.conv.out_channels)
            return in_channels, out_channels, False
        in_channels = int(self.core.features.norm5.num_features)
        return in_channels, int(self.core.classifier.out_features), True

    def make_auxiliary(self, name, rank):
        in_channels, out_features, final = self.boundary_shapes(name)
        auxiliary = (ClassifierAuxiliary(in_channels, out_features, rank)
                     if final else
                     TransitionAuxiliary(in_channels, out_features, rank))
        return auxiliary.to(next(self.parameters()).device)

    @contextmanager
    def auxiliary_context(self, name, auxiliary, gate):
        if name not in self._block_names:
            raise KeyError(name)
        old = (self._active_site, self._active_auxiliary, self._active_gate)
        object.__setattr__(self, "_active_site", name)
        object.__setattr__(self, "_active_auxiliary", auxiliary)
        object.__setattr__(self, "_active_gate", float(gate))
        try:
            yield
        finally:
            object.__setattr__(self, "_active_site", old[0])
            object.__setattr__(self, "_active_auxiliary", old[1])
            object.__setattr__(self, "_active_gate", old[2])

    def projection_parameter_modules(self, name):
        if name not in self._block_names:
            raise KeyError(name)
        index = self._block_names.index(name) + 1
        block = getattr(self.core.features, f"denseblock{index}")
        if index < 4:
            consumer = getattr(self.core.features, f"transition{index}")
            return {"conv_only": (block, consumer),
                    "residual_path": (block, consumer)}
        return {"conv_only": (block, self.core.features.norm5,
                               self.core.classifier),
                "residual_path": (block, self.core.features.norm5,
                                   self.core.classifier)}

    def forward(self, inputs):
        features = self.core.features
        x = features.pool0(features.relu0(features.norm0(features.conv0(inputs))))
        for index in range(1, 5):
            x = getattr(features, f"denseblock{index}")(x)
            site = self._block_names[index - 1]
            if index < 4:
                base = getattr(features, f"transition{index}")(x)
                if self._active_site == site:
                    auxiliary = self._active_auxiliary(x)
                    base = base + self._active_gate * auxiliary
                x = base
            else:
                normalized = F.relu(features.norm5(x), inplace=False)
                pooled = F.adaptive_avg_pool2d(normalized, (1, 1)).flatten(1)
                logits = self.core.classifier(pooled)
                if self._active_site == site:
                    logits = (logits + self._active_gate *
                              self._active_auxiliary(x))
                return logits
        raise RuntimeError("DenseNet forward did not reach its classifier")


@dataclass
class DenseBlockAuxiliaryCandidate:
    module_name: str
    auxiliary: nn.Module
    model: CifarDenseNet121
    proposal_score: float
    payload: dict
    extra_flops: float = 0.0

    @contextmanager
    def virtual_direction(self, gate):
        auxiliary = self.auxiliary
        training = auxiliary.training
        auxiliary.eval()
        try:
            with self.model.auxiliary_context(
                    self.module_name, auxiliary, gate):
                yield
        finally:
            auxiliary.train(training)


def _train_auxiliary(model, site, statistics, rank, steps, learning_rate):
    auxiliary = model.make_auxiliary(site, rank)
    optimizer = torch.optim.SGD(
        auxiliary.parameters(), lr=learning_rate, momentum=0.9)
    device = next(model.parameters()).device
    modes = {module: module.training for module in model.modules()}
    requires_grad = {parameter: parameter.requires_grad
                     for parameter in model.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    auxiliary.train()
    try:
        for step in range(steps):
            inputs, targets = statistics[step % len(statistics)]
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with model.auxiliary_context(site, auxiliary, 1.0):
                loss = F.cross_entropy(model(inputs).float(), targets)
            loss.backward()
            optimizer.step()
        gains = []
        auxiliary.eval()
        with torch.no_grad():
            for inputs, targets in statistics:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                baseline = F.cross_entropy(model(inputs).float(), targets)
                with model.auxiliary_context(site, auxiliary, 0.05):
                    expanded = F.cross_entropy(model(inputs).float(), targets)
                gains.append(float((baseline - expanded).item()))
    finally:
        for parameter, enabled in requires_grad.items():
            parameter.requires_grad_(enabled)
        for module, training in modes.items():
            module.training = training
    score = sum(gains) / len(gains)
    return copy.deepcopy(auxiliary).eval(), score


def propose_denseblock_candidates(model, statistics, rank, sites=None,
                                  steps=20, learning_rate=0.05):
    """Train temporary rank-r branches on one shared statistics batch set."""
    if not isinstance(model, CifarDenseNet121):
        raise TypeError("DenseBlock auxiliary proposals require CifarDenseNet121")
    if not statistics or rank < 1 or steps < 1:
        raise ValueError("statistics, rank, and auxiliary steps must be positive")
    requested = set(sites or [ref.name for ref in model.growing_blocks()])
    candidates = []
    for ref in model.growing_blocks():
        if ref.name not in requested:
            continue
        auxiliary, score = _train_auxiliary(
            model, ref.name, statistics, rank, steps, learning_rate)
        candidates.append(DenseBlockAuxiliaryCandidate(
            ref.name, auxiliary, model, score, {
                "effective_rank": rank,
                "auxiliary_operator": "denseblock_boundary_branch",
                "source": "densenet_block_boundary_auxiliary",
                "function_preserving_at_gate_zero": True,
                "auxiliary_training_steps": steps,
                "auxiliary_learning_rate": learning_rate,
            }))
    if len(candidates) != len(requested):
        found = {candidate.module_name for candidate in candidates}
        raise ValueError(f"unknown DenseBlock sites: {sorted(requested - found)}")
    return candidates
