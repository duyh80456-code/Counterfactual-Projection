"""Trivial linear projection D -> 0 for the relaxed Bypass formulation."""

from __future__ import annotations

from torch import nn

from .extension import LearnableBypassActivation


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
