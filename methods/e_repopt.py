"""Expansion-discovered RepOptimizer-style gradient transformation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from probe import ProbeSignal


@dataclass
class RepOptGradientHandler:
    parameter: nn.Parameter
    direction: Tensor
    strength: float = 1.0

    def __post_init__(self) -> None:
        width = self.parameter.reshape(self.parameter.shape[0], -1).shape[1]
        direction = torch.as_tensor(
            self.direction, device=self.parameter.device,
            dtype=self.parameter.dtype)
        if direction.ndim != 2 or direction.shape[1] != width:
            raise ValueError("RepOpt direction has the wrong flattened width")
        self.direction = direction.detach().clone()

    def transform(self, gradient: Tensor) -> Tensor:
        if gradient.shape != self.parameter.shape:
            raise ValueError("gradient and parameter shapes differ")
        flat = gradient.reshape(gradient.shape[0], -1)
        transformed = flat + self.strength * (
            (flat @ self.direction.T) @ self.direction)
        return transformed.reshape_as(gradient)

    def apply(self) -> None:
        if self.parameter.grad is not None:
            self.parameter.grad.copy_(self.transform(self.parameter.grad))


class ERepOpt:
    """Optimizer decorator whose geometry comes from the current E probe."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 handler: RepOptGradientHandler):
        self.optimizer = optimizer
        self.handler = handler

    @classmethod
    def from_signal(cls, optimizer: torch.optim.Optimizer, model: nn.Module,
                    signal: ProbeSignal, strength: float = 1.0) -> "ERepOpt":
        module = dict(model.named_modules())[signal.block]
        if not isinstance(module, nn.Conv2d):
            raise TypeError("E->RepOpt currently supports Conv2d probes")
        directions = signal.A_E.reshape(signal.rank, -1)
        return cls(optimizer, RepOptGradientHandler(
            module.weight, directions, strength))

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, closure=None):
        self.handler.apply()
        return self.optimizer.step(closure)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        return self.optimizer.load_state_dict(state)

