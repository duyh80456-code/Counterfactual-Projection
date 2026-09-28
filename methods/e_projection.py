"""Project a real TINY/Gromo structural delta-f into the original model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from probe import CandidateExpansionProbe, ProbeSignal
from projection import FunctionalProjector, ProjectionResult


def _module_path(model: nn.Module, target: nn.Module) -> str:
    matches = [name for name, module in model.named_modules() if module is target]
    if len(matches) != 1:
        raise RuntimeError("candidate block is not uniquely registered on model")
    return matches[0]


def candidate_projection_block(model: nn.Module, candidate) -> str:
    logical_name = str(candidate.module_name)
    if logical_name in dict(model.named_modules()):
        return logical_name
    resolver = getattr(model, "block", None)
    if resolver is None:
        raise KeyError(f"cannot resolve candidate block {logical_name!r}")
    return _module_path(model, resolver(logical_name))


def candidate_projection_parameter_names(
        model: nn.Module, candidate,
        scope: str = "residual_path") -> tuple[str, ...] | None:
    """Select residual-path trainable coordinates while excluding shortcuts."""
    if scope == "whole_block":
        return None
    if scope not in {"residual_path", "conv_only"}:
        raise ValueError(f"unknown projection scope {scope!r}")
    block_path = candidate_projection_block(model, candidate)
    block = dict(model.named_modules())[block_path]
    if hasattr(block, "first_layer") and hasattr(block, "second_layer"):
        convolution_modules = (
            block.first_layer.layer, block.second_layer.layer)
    elif hasattr(block, "conv1") and hasattr(block, "conv2"):
        convolution_modules = (block.conv1, block.conv2)
    else:
        raise TypeError("candidate block does not expose its two-convolution path")
    modules = list(convolution_modules)
    if scope == "residual_path":
        if hasattr(block, "first_layer") and hasattr(block, "second_layer"):
            post_functions = (block.first_layer.post_layer_function,
                              block.second_layer.post_layer_function)
            modules.extend(
                child for post in post_functions for child in post.modules()
                if isinstance(child, nn.modules.batchnorm._BatchNorm))
        else:
            modules.extend(
                module for name in ("bn1", "bn2")
                if isinstance(
                    (module := getattr(block, name, None)),
                    nn.modules.batchnorm._BatchNorm))
    parameter_ids = {
        id(parameter) for module in convolution_modules
        for parameter in module.parameters(recurse=False)} | {
        id(parameter) for module in modules
        for parameter in module.parameters(recurse=False)}
    names = tuple(name for name, parameter in model.named_parameters()
                  if id(parameter) in parameter_ids)
    if not names:
        raise RuntimeError("residual-path projection selected no parameters")
    return names


def _eval_loss(model: nn.Module, batch: tuple[Tensor, Tensor]) -> float:
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.no_grad():
            return float(F.cross_entropy(model(batch[0]).float(), batch[1]))
    finally:
        for module, training in modes.items():
            module.training = training


@dataclass(frozen=True)
class ProjectionStep:
    signal: ProbeSignal
    projection: ProjectionResult
    baseline_loss: float
    expanded_loss: float
    projected_loss: float | None = None

    @property
    def structural_loss_gain(self) -> float:
        return self.baseline_loss - self.expanded_loss

    @property
    def structural_directional_gain(self) -> float:
        return self.signal.predicted_gain

    @property
    def projected_loss_gain(self) -> float | None:
        return (None if self.projected_loss is None else
                self.baseline_loss - self.projected_loss)


class EProjection:
    """Functional projection whose primary source is a structural candidate.

    A non-structural control probe can still be passed explicitly, but there is
    intentionally no gradient-SVD default.
    """

    def __init__(self, probe=None, projector: FunctionalProjector | None = None):
        self.probe = probe
        self.structural_probe = CandidateExpansionProbe()
        self.projector = projector or FunctionalProjector()

    def discover_candidate(self, model: nn.Module, candidate,
                           batch: tuple[Tensor, Tensor], *, gate: float = 0.05,
                           block: str | None = None,
                           projection_scope: str = "residual_path") -> ProjectionStep:
        baseline_loss = _eval_loss(model, batch)
        signal = self.structural_probe(
            model, candidate=candidate, batch=batch, gate=gate)
        if not signal.is_structural_expansion:
            raise RuntimeError("main E-projection requires a structural E signal")
        projection_block = block or candidate_projection_block(model, candidate)
        parameter_names = candidate_projection_parameter_names(
            model, candidate, projection_scope)
        result = self.projector.project(
            model, batch[0], signal.delta_logits, block=projection_block,
            parameter_names=parameter_names)
        return ProjectionStep(
            signal, result, baseline_loss,
            baseline_loss - float(signal.observed_loss_gain))

    def step_candidate_(self, model: nn.Module, candidate,
                        batch: tuple[Tensor, Tensor], *, gate: float = 0.05,
                        block: str | None = None,
                        projection_scope: str = "residual_path",
                        scale: float = 1.0) -> ProjectionStep:
        step = self.discover_candidate(
            model, candidate, batch, gate=gate, block=block,
            projection_scope=projection_scope)
        step.projection.apply_(model, scale)
        return replace(step, projected_loss=_eval_loss(model, batch))

    def discover(self, model: nn.Module, batch: tuple[Tensor, Tensor], *,
                 block: str, rank: int) -> ProjectionStep:
        if self.probe is None:
            raise RuntimeError(
                "no control probe configured; use discover_candidate with "
                "a TINY/Gromo candidate")
        baseline_loss = _eval_loss(model, batch)
        signal = self.probe(model, block=block, rank=rank, batch=batch)
        result = self.projector.project(
            model, batch[0], signal.delta_logits, block=block)
        return ProjectionStep(
            signal, result, baseline_loss,
            baseline_loss - signal.predicted_gain)

    def step_(self, model: nn.Module, batch: tuple[Tensor, Tensor], *,
              block: str, rank: int, scale: float = 1.0) -> ProjectionStep:
        step = self.discover(model, batch, block=block, rank=rank)
        step.projection.apply_(model, scale)
        return replace(step, projected_loss=_eval_loss(model, batch))
