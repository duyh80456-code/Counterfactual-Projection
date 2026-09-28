"""Transfer operators from a measured structural E space to model space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from probe.virtual_expansion import ProbeSignal
from .functional import FunctionalProjector, ProjectionResult


@dataclass(frozen=True)
class TransferOperator:
    project: Callable[[Tensor], Tensor]
    lift: Callable[[Tensor], Tensor]

    def auxiliary_precondition(self, gradient: Tensor,
                               inverse_hessian: Callable[[Tensor], Tensor]) -> Tensor:
        return self.project(inverse_hessian(self.lift(gradient)))


@dataclass(frozen=True)
class StructuralExpansionTransfer:
    """Concrete Pi_E induced by one TINY/Gromo virtual direction.

    The auxiliary coordinate is the scalar gate of the structural candidate.
    Its functional lift is alpha * delta_f_E. Pi_E is realized by the same
    block-local tangent solve used by Functional Projection.
    """

    model: nn.Module
    inputs: Tensor
    signal: ProbeSignal
    block: str
    projector: FunctionalProjector

    def __post_init__(self) -> None:
        if not self.signal.is_structural_expansion:
            raise ValueError("transfer requires a real structural E signal")

    def auxiliary_to_function(self, coefficient: Tensor | float) -> Tensor:
        value = torch.as_tensor(
            coefficient, device=self.signal.delta_logits.device,
            dtype=self.signal.delta_logits.dtype)
        return value * self.signal.delta_logits

    def auxiliary_adjoint(self, output_cotangent: Tensor) -> Tensor:
        """J_E^T v for the one-dimensional auxiliary gate."""
        if output_cotangent.shape != self.signal.delta_logits.shape:
            raise ValueError("output cotangent shape differs from delta-f_E")
        return torch.sum(output_cotangent * self.signal.delta_logits)

    def to_model(self, coefficient: Tensor | float = 1.0) -> ProjectionResult:
        return self.projector.project(
            self.model, self.inputs,
            self.auxiliary_to_function(coefficient), block=self.block)
