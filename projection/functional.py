"""Block-local functional projection using torch.func JVP/VJP products."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.func import functional_call, jvp, vjp

from .cg import CGResult, conjugate_gradient
from .metrics import cosine_alignment, fitted_norm_ratio, relative_residual


@dataclass(frozen=True)
class FunctionalEvaluation:
    fitted_delta: Tensor
    target_delta: Tensor
    fitted_norm_ratio: float
    relative_residual: float
    cosine_alignment: float
    jvp_calls: int = 1


@dataclass(frozen=True)
class ProjectionResult:
    block: str
    parameter_delta: dict[str, Tensor]
    fitted_delta: Tensor
    target_delta: Tensor
    fitted_norm_ratio: float
    relative_residual: float
    cosine_alignment: float
    cg: CGResult
    jvp_calls: int
    vjp_calls: int

    @property
    def projection_ratio(self) -> float:
        """Compatibility alias; this is only a norm ratio under damping."""
        return self.fitted_norm_ratio

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
                *, block: str,
                parameter_names: tuple[str, ...] | None = None) -> ProjectionResult:
        modes = {module: module.training for module in model.modules()}
        model.eval()
        try:
            return self._project_eval(
                model, inputs, target_delta, block=block,
                parameter_names=parameter_names)
        finally:
            for module, training in modes.items():
                module.training = training

    def _project_eval(self, model: nn.Module, inputs: Tensor,
                      target_delta: Tensor, *, block: str,
                      parameter_names: tuple[str, ...] | None) -> ProjectionResult:
        prefix = f"{block}." if block else ""
        requested = None if parameter_names is None else set(parameter_names)
        selected = [
            (name, parameter) for name, parameter in model.named_parameters()
            if ((requested is None and name.startswith(prefix)) or
                (requested is not None and name in requested))]
        if requested is not None and {name for name, _ in selected} != requested:
            missing = requested - {name for name, _ in selected}
            raise KeyError(f"unknown projection parameters: {sorted(missing)}")
        if not selected:
            raise KeyError(f"block {block!r} has no parameters")
        names = [name for name, _ in selected]
        shapes = [parameter.shape for _, parameter in selected]
        sizes = [parameter.numel() for _, parameter in selected]
        base = torch.cat([parameter.detach().reshape(-1) for _, parameter in selected])
        target = target_delta.detach().reshape(-1).to(base)
        jvp_calls = 0
        vjp_calls = 0

        def unpack(vector: Tensor) -> dict[str, Tensor]:
            return {name: piece.reshape(shape) for name, shape, piece in zip(
                names, shapes, vector.split(sizes))}

        def function(vector: Tensor) -> Tensor:
            return functional_call(
                model, unpack(vector), (inputs,), strict=False).reshape(-1)

        def jacobian_vector(vector: Tensor) -> Tensor:
            nonlocal jvp_calls
            jvp_calls += 1
            return jvp(function, (base,), (vector.detach(),))[1].detach()

        _, pullback = vjp(function, base)

        def transpose_jacobian(vector: Tensor) -> Tensor:
            nonlocal vjp_calls
            vjp_calls += 1
            return pullback(vector.detach())[0].detach()

        rhs = transpose_jacobian(target).detach()

        def normal_matrix(vector: Tensor) -> Tensor:
            return (transpose_jacobian(jacobian_vector(vector)) +
                    self.damping * vector.detach()).detach()

        cg = conjugate_gradient(normal_matrix, rhs, max_iter=self.max_iter,
                                tolerance=self.tolerance)
        fitted = jacobian_vector(cg.solution).reshape_as(target_delta).detach()
        return ProjectionResult(
            block=block,
            parameter_delta={name: value.detach()
                             for name, value in unpack(cg.solution).items()},
            fitted_delta=fitted, target_delta=target_delta.detach(),
            fitted_norm_ratio=fitted_norm_ratio(fitted, target_delta),
            relative_residual=relative_residual(fitted, target_delta),
            cosine_alignment=cosine_alignment(fitted, target_delta), cg=cg,
            jvp_calls=jvp_calls, vjp_calls=vjp_calls)

    def evaluate_direction(
            self, model: nn.Module, inputs: Tensor, target_delta: Tensor,
            parameter_delta: dict[str, Tensor]) -> FunctionalEvaluation:
        """Evaluate a fitted parameter direction on inputs never used to fit it."""
        if not parameter_delta:
            raise ValueError("parameter direction is empty")
        parameters = dict(model.named_parameters())
        names = tuple(parameter_delta)
        missing = set(names) - set(parameters)
        if missing:
            raise KeyError(f"model no longer contains {sorted(missing)}")
        modes = {module: module.training for module in model.modules()}
        try:
            model.eval()
            primals = tuple(parameters[name].detach() for name in names)
            tangents = tuple(parameter_delta[name].to(parameters[name])
                             for name in names)

            def function(*values):
                replacements = dict(zip(names, values))
                return functional_call(
                    model, replacements, (inputs,), strict=False)

            fitted = jvp(function, primals, tangents)[1].detach()
        finally:
            for module, training in modes.items():
                module.training = training
        target = target_delta.detach()
        return FunctionalEvaluation(
            fitted_delta=fitted, target_delta=target,
            fitted_norm_ratio=fitted_norm_ratio(fitted, target),
            relative_residual=relative_residual(fitted, target),
            cosine_alignment=cosine_alignment(fitted, target))
