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
    residual_history: tuple[float, ...] = ()


def conjugate_gradient(matvec: Callable[[Tensor], Tensor], rhs: Tensor, *,
                       x0: Tensor | None = None, max_iter: int = 50,
                       tolerance: float = 1e-6,
                       eps: float = 1e-12) -> CGResult:
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    rhs = rhs.detach()
    x = x.detach()
    residual = (rhs - matvec(x)).detach()
    direction = residual.clone()
    squared = torch.dot(residual, residual)
    threshold = tolerance * max(float(torch.linalg.vector_norm(rhs)), eps)
    initial_norm = float(torch.sqrt(squared))
    history = [initial_norm]
    if initial_norm <= threshold:
        return CGResult(x, 0, initial_norm, True, tuple(history))
    iteration = 0
    completed_iterations = 0
    for iteration in range(1, max_iter + 1):
        image = matvec(direction).detach()
        denominator = torch.dot(direction, image)
        if abs(float(denominator)) <= eps:
            break
        alpha = squared / denominator
        x = (x + alpha * direction).detach()
        residual = (residual - alpha * image).detach()
        completed_iterations = iteration
        new_squared = torch.dot(residual, residual)
        norm = float(torch.sqrt(new_squared))
        history.append(norm)
        if norm <= threshold:
            return CGResult(x, iteration, norm, True, tuple(history))
        direction = (residual + (new_squared / squared) * direction).detach()
        squared = new_squared
    return CGResult(
        x, completed_iterations, float(torch.linalg.vector_norm(residual)), False,
        tuple(history))
