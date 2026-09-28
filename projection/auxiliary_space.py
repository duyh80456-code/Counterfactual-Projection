"""General and structural auxiliary-space formulations."""

from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from .functional import ProjectionResult
from .transfer_operator import StructuralExpansionTransfer, TransferOperator


@dataclass(frozen=True)
class AuxiliarySpaceProjector:
    transfer: TransferOperator
    inverse_auxiliary_hessian: Callable[[Tensor], Tensor]

    def correction(self, gradient: Tensor) -> Tensor:
        return -self.transfer.auxiliary_precondition(
            gradient, self.inverse_auxiliary_hessian)


@dataclass(frozen=True)
class StructuralAuxiliarySpace:
    """One-dimensional B_E = Pi_E H_E^-1 Pi_E^T realization."""

    transfer: StructuralExpansionTransfer
    curvature: float = 1.0
    damping: float = 1e-3

    def __post_init__(self) -> None:
        if self.curvature < 0 or self.damping <= 0:
            raise ValueError("auxiliary curvature/damping must define an SPD solve")

    def correction(self, output_gradient: Tensor) -> ProjectionResult:
        auxiliary_gradient = self.transfer.auxiliary_adjoint(output_gradient)
        coefficient = -auxiliary_gradient / (self.curvature + self.damping)
        return self.transfer.to_model(coefficient)
