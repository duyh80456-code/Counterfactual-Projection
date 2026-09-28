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
    relative_residual: float = float("inf")


def conjugate_gradient(matvec: Callable[[Tensor], Tensor], rhs: Tensor, *,
                       x0: Tensor | None = None, max_iter: int = 50,
                       tolerance: float = 1e-6,
                       preconditioner: Callable[[Tensor], Tensor] | None = None,
                       eps: float = 1e-12) -> CGResult:
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    x = torch.zeros_like(rhs) if x0 is None else x0.clone()
    rhs = rhs.detach()
    x = x.detach()
    residual = (rhs - matvec(x)).detach()
    rhs_norm = max(float(torch.linalg.vector_norm(rhs)), eps)
    threshold = tolerance * rhs_norm
    initial_norm = float(torch.linalg.vector_norm(residual))
    history = [initial_norm]
    if initial_norm <= threshold:
        return CGResult(
            x, 0, initial_norm, True, tuple(history), initial_norm / rhs_norm)
    precondition = preconditioner or (lambda value: value)
    preconditioned = precondition(residual).detach()
    direction = preconditioned.clone()
    residual_product = torch.dot(residual, preconditioned)
    iteration = 0
    completed_iterations = 0
    for iteration in range(1, max_iter + 1):
        image = matvec(direction).detach()
        denominator = torch.dot(direction, image)
        if abs(float(denominator)) <= eps:
            break
        alpha = residual_product / denominator
        x = (x + alpha * direction).detach()
        residual = (residual - alpha * image).detach()
        completed_iterations = iteration
        norm = float(torch.linalg.vector_norm(residual))
        history.append(norm)
        if norm <= threshold:
            # The recursively updated residual can drift from b - A x in
            # finite precision. Verify convergence against the true residual;
            # if it fails, restart PCG from that residual.
            residual = (rhs - matvec(x)).detach()
            norm = float(torch.linalg.vector_norm(residual))
            history[-1] = norm
            if norm <= threshold:
                return CGResult(
                    x, iteration, norm, True, tuple(history), norm / rhs_norm)
            preconditioned = precondition(residual).detach()
            direction = preconditioned.clone()
            residual_product = torch.dot(residual, preconditioned)
            continue
        next_preconditioned = precondition(residual).detach()
        next_product = torch.dot(residual, next_preconditioned)
        if abs(float(residual_product)) <= eps:
            break
        direction = (next_preconditioned +
                     (next_product / residual_product) * direction).detach()
        preconditioned = next_preconditioned
        residual_product = next_product
    residual = (rhs - matvec(x)).detach()
    final_norm = float(torch.linalg.vector_norm(residual))
    history[-1] = final_norm
    return CGResult(
        x, completed_iterations, final_norm, False, tuple(history),
        final_norm / rhs_norm)
