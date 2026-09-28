"""Block-local functional projection using torch.func JVP/VJP products."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.func import functional_call, jvp, vjp

from .cg import CGResult, conjugate_gradient
from .metrics import cosine_alignment, fitted_norm_ratio, relative_residual


@contextmanager
def _preserve_functional_state(model: nn.Module):
    """Prevent torch.func wrappers from leaking into a stateful model.

    Gromo growing modules optionally retain ``input`` and ``activity`` as
    ordinary Python attributes. A forward executed inside JVP/VJP can otherwise
    leave a GradTrackingTensor there, which is not deepcopy-able and later
    breaks a real growth commit. Some custom module arrangements can also retain
    a tensor swapped temporarily by ``functional_call`` in a parameter/buffer
    slot. Restore both categories by object identity. Projection never consumes
    Gromo's cached statistics, so disabling their collection is safe and cheaper.
    """
    cache_names = ("input", "activity")
    control_names = ("store_input", "store_activity")
    caches = []
    controls = []
    registered = []
    for module in model.modules():
        registered.extend(
            (module, "_parameters", name, value)
            for name, value in module._parameters.items())
        registered.extend(
            (module, "_buffers", name, value)
            for name, value in module._buffers.items())
        for name in cache_names:
            if name in module.__dict__:
                caches.append((module, name, module.__dict__[name]))
        for name in control_names:
            if name in module.__dict__:
                controls.append((module, name, module.__dict__[name]))
                setattr(module, name, 0)
    try:
        yield
    finally:
        for module, collection, name, value in registered:
            getattr(module, collection)[name] = value
        for module, name, value in caches:
            setattr(module, name, value)
        for module, name, value in controls:
            setattr(module, name, value)


@dataclass(frozen=True)
class FunctionalEvaluation:
    fitted_delta: Tensor
    target_delta: Tensor
    fitted_norm_ratio: float
    relative_residual: float
    cosine_alignment: float
    jvp_calls: int = 1


@dataclass(frozen=True)
class CGAttempt:
    damping: float
    iterations: int
    residual_norm: float
    relative_residual: float
    converged: bool
    solution_is_finite: bool
    functional_relative_residual: float
    functional_cosine_alignment: float


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
    damping_requested: float
    damping_used: float
    cg_attempts: tuple[CGAttempt, ...]
    target_scale: float
    solver_space: str
    linear_system_dimension: int
    solver_dtype: str
    preconditioner: str
    preconditioner_probes: int
    selected_attempt: int
    selection_rule: str

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
    def __init__(self, damping: float = 1e-3, max_iter: int = 200,
                 tolerance: float = 1e-6, max_damping_retries: int = 4,
                 damping_multiplier: float = 10.0,
                 preconditioner_probes: int = 8):
        if (damping < 0 or max_damping_retries < 0 or
                damping_multiplier <= 1 or preconditioner_probes < 0):
            raise ValueError("invalid damping/retry configuration")
        self.damping = float(damping)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)
        self.max_damping_retries = int(max_damping_retries)
        self.damping_multiplier = float(damping_multiplier)
        self.preconditioner_probes = int(preconditioner_probes)

    def project(self, model: nn.Module, inputs: Tensor, target_delta: Tensor,
                *, block: str,
                parameter_names: tuple[str, ...] | None = None) -> ProjectionResult:
        modes = {module: module.training for module in model.modules()}
        model.eval()
        try:
            with _preserve_functional_state(model):
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
        target_scale = float(torch.linalg.vector_norm(target).clamp_min(1e-12))
        normalized_target = target / target_scale
        # Keep the expensive network derivatives in the model dtype, but use
        # float64 for the small output-space Krylov vectors. CG recurrence and
        # dot products are otherwise prone to losing conjugacy in float32 on
        # the ill-conditioned real ResNet/Gromo operator.
        solver_dtype = (torch.float64 if normalized_target.dtype in {
            torch.float16, torch.bfloat16, torch.float32
        } else normalized_target.dtype)
        solver_rhs = normalized_target.to(dtype=solver_dtype)
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

        def gram_matrix(vector: Tensor) -> Tensor:
            operator_vector = vector.detach().to(normalized_target)
            return jacobian_vector(
                transpose_jacobian(operator_vector)).to(vector).detach()

        # Hutchinson gives a cheap matrix-free estimate of diag(J J^T).
        # A positive diagonal floor keeps the PCG preconditioner SPD even when
        # a finite-probe estimate is noisy.
        gram_diagonal = None
        if self.preconditioner_probes:
            generator = torch.Generator(device=solver_rhs.device).manual_seed(2027)
            diagonal_sum = torch.zeros_like(solver_rhs)
            for _ in range(self.preconditioner_probes):
                signs = torch.empty_like(solver_rhs).bernoulli_(
                    0.5, generator=generator).mul_(2).sub_(1)
                diagonal_sum.add_(signs * gram_matrix(signs))
            estimate = diagonal_sum / self.preconditioner_probes
            scale = estimate.abs().mean().clamp_min(1e-12)
            gram_diagonal = estimate.clamp_min(scale * 1e-3).detach()

        cg_attempts = []
        candidate_solutions = []
        damping_used = self.damping
        for attempt in range(self.max_damping_retries + 1):
            damping_used = (self.damping * self.damping_multiplier ** attempt
                            if self.damping > 0 else 0.0)

            # Solve the equivalent output-space ridge system
            #
            #   (J J^T + mu I) u = target,  delta = J^T u.
            #
            # This has exactly the same parameter solution as the primal
            # normal equation for mu > 0, while avoiding a poorly scaled
            # J^T target RHS and a CG system with millions of coordinates.
            # The implementation remains matrix-free: every matvec is one
            # VJP followed by one JVP.
            def dual_matrix(vector: Tensor) -> Tensor:
                return (gram_matrix(vector) +
                        damping_used * vector.detach()).detach()

            preconditioner = None
            if gram_diagonal is not None:
                inverse_diagonal = (gram_diagonal + damping_used).reciprocal()

                def preconditioner(vector: Tensor) -> Tensor:
                    return (inverse_diagonal * vector.detach()).detach()

            dual_cg = conjugate_gradient(
                dual_matrix, solver_rhs, max_iter=self.max_iter,
                tolerance=self.tolerance, preconditioner=preconditioner)
            normalized_parameter_solution = transpose_jacobian(
                dual_cg.solution.to(normalized_target)).detach()
            cg = CGResult(
                solution=normalized_parameter_solution * target_scale,
                iterations=dual_cg.iterations,
                residual_norm=dual_cg.residual_norm * target_scale,
                converged=dual_cg.converged,
                residual_history=tuple(
                    value * target_scale
                    for value in dual_cg.residual_history),
                relative_residual=dual_cg.relative_residual)
            candidate_fitted = jacobian_vector(cg.solution).reshape_as(
                target_delta).detach()
            solution_is_finite = bool(
                torch.isfinite(cg.solution).all() and
                torch.isfinite(candidate_fitted).all())
            candidate_functional_residual = (
                relative_residual(candidate_fitted, target_delta)
                if solution_is_finite else float("inf"))
            candidate_functional_cosine = (
                cosine_alignment(candidate_fitted, target_delta)
                if solution_is_finite else float("-inf"))
            cg_attempts.append(CGAttempt(
                damping=damping_used, iterations=cg.iterations,
                residual_norm=cg.residual_norm,
                relative_residual=cg.relative_residual,
                converged=cg.converged,
                solution_is_finite=solution_is_finite,
                functional_relative_residual=candidate_functional_residual,
                functional_cosine_alignment=candidate_functional_cosine))
            if solution_is_finite:
                candidate_solutions.append((
                    candidate_functional_residual,
                    -candidate_functional_cosine,
                    0 if cg.converged else 1,
                    len(cg_attempts) - 1,
                    damping_used, cg, candidate_fitted))
            if cg.converged or self.damping == 0:
                break
        if not candidate_solutions:
            raise RuntimeError("all projection attempts produced non-finite solutions")
        (_, _, _, selected_attempt, damping_used,
         cg, fitted) = min(candidate_solutions, key=lambda item: item[:4])
        return ProjectionResult(
            block=block,
            parameter_delta={name: value.detach()
                             for name, value in unpack(cg.solution).items()},
            fitted_delta=fitted, target_delta=target_delta.detach(),
            fitted_norm_ratio=fitted_norm_ratio(fitted, target_delta),
            relative_residual=relative_residual(fitted, target_delta),
            cosine_alignment=cosine_alignment(fitted, target_delta), cg=cg,
            jvp_calls=jvp_calls, vjp_calls=vjp_calls,
            damping_requested=self.damping, damping_used=damping_used,
            cg_attempts=tuple(cg_attempts), target_scale=target_scale,
            solver_space="dual_output",
            linear_system_dimension=normalized_target.numel(),
            solver_dtype=str(solver_dtype).removeprefix("torch."),
            preconditioner=("hutchinson_jacobi" if gram_diagonal is not None
                            else "none"),
            preconditioner_probes=self.preconditioner_probes,
            selected_attempt=selected_attempt,
            selection_rule="minimum_finite_functional_relative_residual")

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
            with _preserve_functional_state(model):
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
