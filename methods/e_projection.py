"""Probe, discover, and functionally project in one orchestration object."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from probe import ProbeSignal, VirtualExpansionProbe
from projection import FunctionalProjector, ProjectionResult


@dataclass(frozen=True)
class ProjectionStep:
    signal: ProbeSignal
    projection: ProjectionResult


class EProjection:
    def __init__(self, probe: VirtualExpansionProbe | None = None,
                 projector: FunctionalProjector | None = None):
        self.probe = probe or VirtualExpansionProbe()
        self.projector = projector or FunctionalProjector()

    def discover(self, model: nn.Module, batch: tuple[Tensor, Tensor], *,
                 block: str, rank: int) -> ProjectionStep:
        signal = self.probe(model, block=block, rank=rank, batch=batch)
        result = self.projector.project(
            model, batch[0], signal.delta_logits, block=block)
        return ProjectionStep(signal, result)

    def step_(self, model: nn.Module, batch: tuple[Tensor, Tensor], *,
              block: str, rank: int, scale: float = 1.0) -> ProjectionStep:
        result = self.discover(model, batch, block=block, rank=rank)
        result.projection.apply_(model, scale)
        return result

