"""Block-local functional projection using torch.func JVP/VJP products."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.func import functional_call, jvp, vjp

from .cg import CGResult, conjugate_gradient


@dataclass(frozen=True)
class ProjectionResult:
    block: str
    parameter_delta: dict[str, Tensor]
    fitted_delta: Tensor
    target_delta: Tensor
    projection_ratio: float
    relative_residual: float
    cg: CGResult

    @torch.no_grad()
    def apply_(self, model: nn.Module, scale: float = 1.0) -> None:
        parameters = dict(model.named_parameters())
        for name, delta in self.parameter_delta.items():
            if name not in parameters:
                raise KeyError(f"model no longer contains {name!r}")
            parameters[name].add_(delta.to(parameters[name]), alpha=float(scale))


class FunctionalProjector:
    def __init__(self, damping: float = 1e-3, max_iter: int = 50,
                 tolerance: float = 1e-6):
        if damping < 0:
            raise ValueError("damping must be non-negative")
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    def project(self, model: nn.Module, inputs: Tensor, target_delta: Tensor,
                *, block: str) -> ProjectionResult:
        modes = {module: module.training for module in model.modules()}
        model.eval()
        try:
            return self._project_eval(model, inputs, target_delta, block=block)
        finally:
            for module, training in modes.items():
                module.training = training

    def _project_eval(self, model: nn.Module, inputs: Tensor,
                      target_delta: Tensor, *, block: str) -> ProjectionResult:
        prefix = f"{block}." if block else ""
        selected = [(name, parameter) for name, parameter in model.named_parameters()
                    if name.startswith(prefix)]
        if not selected:
            raise KeyError(f"block {block!r} has no parameters")
        names = [name for name, _ in selected]
        shapes = [parameter.shape for _, parameter in selected]
        sizes = [parameter.numel() for _, parameter in selected]
        base = torch.cat([parameter.detach().reshape(-1) for _, parameter in selected])
        target = target_delta.detach().reshape(-1).to(base)

        def unpack(vector: Tensor) -> dict[str, Tensor]:
            return {name: piece.reshape(shape) for name, shape, piece in zip(
                names, shapes, vector.split(sizes))}

        def function(vector: Tensor) -> Tensor:
            return functional_call(
                model, unpack(vector), (inputs,), strict=False).reshape(-1)

        def jacobian_vector(vector: Tensor) -> Tensor:
            return jvp(function, (base,), (vector,))[1]

        _, pullback = vjp(function, base)

        def transpose_jacobian(vector: Tensor) -> Tensor:
            return pullback(vector)[0]

        rhs = transpose_jacobian(target)

        def normal_matrix(vector: Tensor) -> Tensor:
            return transpose_jacobian(jacobian_vector(vector)) + self.damping * vector

        cg = conjugate_gradient(normal_matrix, rhs, max_iter=self.max_iter,
                                tolerance=self.tolerance)
        fitted = jacobian_vector(cg.solution).reshape_as(target_delta).detach()
        target_norm = torch.linalg.vector_norm(target_delta.detach()).clamp_min(1e-12)
        ratio = float(torch.linalg.vector_norm(fitted) / target_norm)
        residual = float(torch.linalg.vector_norm(target_delta.detach() - fitted) / target_norm)
        return ProjectionResult(
            block=block,
            parameter_delta={name: value.detach()
                             for name, value in unpack(cg.solution).items()},
            fitted_delta=fitted, target_delta=target_delta.detach(),
            projection_ratio=ratio, relative_residual=residual, cg=cg)
