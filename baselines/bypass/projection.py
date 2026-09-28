"""Trivial linear projection D -> 0 for the relaxed Bypass formulation."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .extension import (
    LearnableBypassActivation, activations, contraction_norm,
    extension_parameters)
from .optimizer import remove_extension_parameters_


@dataclass(frozen=True)
class Opt2Transition:
    phase: str
    contraction_norm: float
    criterion_met: bool
    soft_cap_exceeded: bool
    projected_count: int


def _contract(parent: nn.Module) -> int:
    count = 0
    for name, child in list(parent._modules.items()):
        if isinstance(child, LearnableBypassActivation):
            parent._modules[name] = nn.ReLU()
            count += 1
        else:
            count += _contract(child)
    return count


def project_relaxed_bypass_(model: nn.Module) -> int:
    """Drop every D coordinate while retaining all original-space weights."""
    count = _contract(model)
    if count < 1:
        raise RuntimeError("relaxed Bypass projection found no extension")
    return count


def transition_from_opt2_(model: nn.Module, optimizer, *, epsilon: float,
                          opt2_done: int, soft_cap: int) -> Opt2Transition:
    """Project only after contraction; a soft cap never authorizes projection."""
    current_norm = float(contraction_norm(model).detach())
    criterion_met = current_norm < epsilon
    soft_cap_exceeded = opt2_done >= soft_cap and not criterion_met
    if not criterion_met:
        return Opt2Transition(
            phase="opt2", contraction_norm=current_norm,
            criterion_met=False, soft_cap_exceeded=soft_cap_exceeded,
            projected_count=0)

    parameters = extension_parameters(model)
    expected = len(activations(model))
    for module in activations(model):
        module.project_()
    remove_extension_parameters_(optimizer, parameters)
    projected_count = project_relaxed_bypass_(model)
    if projected_count != expected:
        raise RuntimeError("Bypass projection did not remove every D")
    return Opt2Transition(
        phase="train3", contraction_norm=current_norm,
        criterion_met=True, soft_cap_exceeded=False,
        projected_count=projected_count)
