"""Controls where the real structural expansion participates in training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from methods.e_projection import (
    candidate_projection_block, candidate_projection_parameter_names)
from probe import ProbeSignal
from projection import FunctionalProjector, ProjectionResult


@dataclass(frozen=True)
class GrowthCommit:
    train_parameter_delta: int
    deploy_parameter_delta: int
    committed_module: nn.Module


class RealEGrowth:
    """Commit function-preserving TINY/Gromo E and train the grown model."""

    @staticmethod
    def commit_(model: nn.Module, candidate) -> GrowthCommit:
        before = sum(parameter.numel() for parameter in model.parameters())
        committed = candidate.commit()
        second = getattr(committed, "second_layer", None)
        if (second is not None and hasattr(second, "in_neurons") and
                hasattr(second, "target_in_neurons")):
            # A committed counterfactual over-expansion becomes the grown model's
            # new full-width baseline. Never leave current_width > target.
            second.target_in_neurons = int(second.in_neurons)
        after = sum(parameter.numel() for parameter in model.parameters())
        return GrowthCommit(
            train_parameter_delta=after - before,
            deploy_parameter_delta=after - before,
            committed_module=committed)


# Compatibility alias. Reports and experiment names use ``real_e_growth``;
# virtual structural gain at the same state is the actual local oracle metric.
RealEOracle = RealEGrowth
OracleCommit = GrowthCommit


@dataclass(frozen=True)
class ExpandedTrainProjectResult:
    signal: ProbeSignal
    projection: ProjectionResult
    expansion_train_losses: tuple[float, ...]


class ExpandedTrainProject:
    """RepAn/Bypass-like control: train E temporarily, then contract via Pi_E."""

    def __init__(self, steps: int = 1, learning_rate: float = 1e-2,
                 projector: FunctionalProjector | None = None):
        if steps < 1 or learning_rate <= 0:
            raise ValueError("invalid expanded training configuration")
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.projector = projector or FunctionalProjector()

    def discover(self, model: nn.Module, candidate,
                 batch: tuple[Tensor, Tensor], *, gate: float = 1.0,
                 block: str | None = None,
                 projection_scope: str = "conv_path") -> ExpandedTrainProjectResult:
        inputs, targets = batch
        modes = {module: module.training for module in model.modules()}
        base_parameters = tuple(model.parameters())
        base_ids = {id(parameter) for parameter in base_parameters}
        base_state = {name: value.detach().clone()
                      for name, value in model.state_dict().items()}
        old_requires_grad = {parameter: parameter.requires_grad
                             for parameter in base_parameters}
        try:
            model.eval()
            with torch.no_grad():
                baseline = model(inputs)
                baseline_loss = F.cross_entropy(baseline.float(), targets)
            with candidate.virtual_direction(gate):
                extension_parameters = [parameter for parameter in model.parameters()
                                        if id(parameter) not in base_ids]
                if not extension_parameters:
                    raise RuntimeError(
                        "candidate did not register trainable expansion parameters")
                for parameter in base_parameters:
                    parameter.requires_grad_(False)
                optimizer = torch.optim.SGD(
                    extension_parameters, lr=self.learning_rate)
                losses = []
                for _ in range(self.steps):
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.cross_entropy(model(inputs).float(), targets)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                with torch.no_grad():
                    expanded = model(inputs)
                    expanded_loss = F.cross_entropy(expanded.float(), targets)
            delta = (expanded - baseline).detach()
            signal = ProbeSignal(
                block=str(candidate.module_name), A_E=None, B_E=None,
                delta_feature=None, delta_logits=delta,
                predicted_gain=float((baseline_loss - expanded_loss).item()),
                singular_values=None, source="expanded_train_then_contract",
                is_structural_expansion=True)
            projection_block = block or candidate_projection_block(model, candidate)
            parameter_names = candidate_projection_parameter_names(
                model, candidate, projection_scope)
            projection = self.projector.project(
                model, inputs, delta, block=projection_block,
                parameter_names=parameter_names)
            return ExpandedTrainProjectResult(signal, projection, tuple(losses))
        finally:
            for parameter, requires_grad in old_requires_grad.items():
                parameter.requires_grad_(requires_grad)
            for module, training in modes.items():
                module.training = training
            after = model.state_dict()
            changed = [name for name, value in after.items()
                       if name not in base_state or not torch.equal(value, base_state[name])]
            if changed:
                raise RuntimeError(
                    f"temporary expanded training mutated base model: {changed[:3]}")
