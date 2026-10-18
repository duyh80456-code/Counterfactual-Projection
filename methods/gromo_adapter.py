"""Bitwise state guards for virtual model operations."""
from typing import Mapping
import torch
from torch import Tensor, nn

def _snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}

def _assert_unchanged(model: nn.Module, before: Mapping[str, Tensor]) -> None:
    after = model.state_dict()
    if after.keys() != before.keys():
        raise RuntimeError("virtual expansion changed model state structure")
    changed = [name for name, value in after.items()
               if not torch.equal(value, before[name])]
    if changed:
        raise RuntimeError(f"virtual expansion mutated model state: {changed[:3]}")
