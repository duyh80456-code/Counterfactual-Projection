"""Learnable-activation embedding for the paper's relaxed ResNet Bypass."""

from __future__ import annotations

import torch
from torch import nn


class LearnableBypassActivation(nn.Module):
    """psi(x) = ReLU(x) + D x with channel-diagonal D and embed D=0."""

    def __init__(self, device=None):
        super().__init__()
        # The exact channel count is materialized by the function-preserving
        # calibration forward in ``embed_relaxed_bypass``.
        self.d = nn.Parameter(torch.empty(0, device=device))
        self.register_buffer("projected", torch.tensor(False, device=device))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        activated = torch.relu(inputs)
        if bool(self.projected):
            return activated
        if inputs.ndim != 4:
            raise ValueError("relaxed ResNet Bypass expects NCHW activations")
        channels = inputs.shape[1]
        if self.d.numel() == 0:
            self.d = nn.Parameter(inputs.new_zeros(channels))
        if channels != self.d.numel():
            raise ValueError(
                f"activation has {channels} channels, D has {self.d.numel()}")
        diagonal = self.d.view(1, channels, 1, 1)
        return activated + inputs * diagonal

    def active_norm(self) -> torch.Tensor:
        return self.d.new_zeros(()) if bool(self.projected) else self.d.norm()

    @torch.no_grad()
    def project_(self) -> None:
        self.d.zero_()
        self.projected.fill_(True)


def _replace_relus(parent: nn.Module, prefix: str, replaced: list[str],
                   device: torch.device) -> None:
    for name, child in list(parent._modules.items()):
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.ReLU):
            parent._modules[name] = LearnableBypassActivation(device=device)
            replaced.append(path)
        else:
            _replace_relus(child, path, replaced, device)


def embed_relaxed_bypass(model: nn.Module) -> list[str]:
    """Replace every ReLU registered under residual stages, as in Sec. VI-C."""
    core = getattr(model, "core", None)
    stages = getattr(core, "stages", None)
    if stages is None:
        raise TypeError("expected CIFAR Gromo ResNet with core.stages")
    parameter = next(model.parameters())
    replaced: list[str] = []
    _replace_relus(stages, "core.stages", replaced, parameter.device)
    if not replaced:
        raise RuntimeError("relaxed Bypass embed found no residual ReLU")
    # Allocate each diagonal D at its actual activation width. D=0 makes this
    # calibration exactly function-preserving, and eval mode protects BN stats.
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 32, 32, device=parameter.device,
                          dtype=parameter.dtype))
    model.train(was_training)
    if any(module.d.numel() == 0 for module in activations(model)):
        raise RuntimeError("not every relaxed Bypass activation was calibrated")
    return replaced


def activations(model: nn.Module) -> list[LearnableBypassActivation]:
    return [module for module in model.modules()
            if isinstance(module, LearnableBypassActivation)]


def extension_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [module.d for module in activations(model)]


def contraction_norm(model: nn.Module) -> torch.Tensor:
    modules = activations(model)
    if not modules:
        parameter = next(model.parameters())
        return parameter.new_zeros(())
    return torch.stack([module.active_norm() for module in modules]).sum()


@torch.no_grad()
def project_ready_activations(model: nn.Module, epsilon: float) -> int:
    projected = 0
    for module in activations(model):
        if not bool(module.projected) and float(module.d.norm()) < epsilon:
            module.project_()
            projected += 1
    return projected
