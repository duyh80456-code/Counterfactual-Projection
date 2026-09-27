"""Explicit transfer operators from an auxiliary space to model space."""

from dataclasses import dataclass
from typing import Callable

from torch import Tensor


@dataclass(frozen=True)
class TransferOperator:
    project: Callable[[Tensor], Tensor]
    lift: Callable[[Tensor], Tensor]

    def auxiliary_precondition(self, gradient: Tensor,
                               inverse_hessian: Callable[[Tensor], Tensor]) -> Tensor:
        return self.project(inverse_hessian(self.lift(gradient)))

