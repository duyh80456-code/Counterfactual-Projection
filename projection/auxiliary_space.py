"""General auxiliary-space formulation."""

from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from .transfer_operator import TransferOperator


@dataclass(frozen=True)
class AuxiliarySpaceProjector:
    transfer: TransferOperator
    inverse_auxiliary_hessian: Callable[[Tensor], Tensor]

    def correction(self, gradient: Tensor) -> Tensor:
        return -self.transfer.auxiliary_precondition(
            gradient, self.inverse_auxiliary_hessian)

