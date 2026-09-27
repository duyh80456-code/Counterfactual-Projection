"""Matrix-free conjugate gradient."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class CGResult:
    solution: Tensor
    iterations: int
    residual_norm: float
    converged: bool


def conjugate_gradient(matvec: Callable[[Tensor], Tensor], rhs: Tensor, *,
                       x0: Tensor | None = None, max_iter: int = 50,
                       tolerance: float = 1e-6,
                       eps: float = 1e-12) -> CGResult:
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    residual = rhs - matvec(x)
    direction = residual.clone()
    squared = torch.dot(residual, residual)
    threshold = tolerance * max(float(torch.linalg.vector_norm(rhs)), eps)
    if float(torch.sqrt(squared)) <= threshold:
        return CGResult(x, 0, float(torch.sqrt(squared)), True)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        image = matvec(direction)
        denominator = torch.dot(direction, image)
        if abs(float(denominator)) <= eps:
            break
        alpha = squared / denominator
        x = x + alpha * direction
        residual = residual - alpha * image
        new_squared = torch.dot(residual, residual)
        norm = float(torch.sqrt(new_squared))
        if norm <= threshold:
            return CGResult(x, iteration, norm, True)
        direction = residual + (new_squared / squared) * direction
        squared = new_squared
    return CGResult(x, iteration, float(torch.linalg.vector_norm(residual)), False)

