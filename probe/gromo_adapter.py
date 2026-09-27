"""Optional bridge for candidate objects produced by One-Shot-TAS-CCIL."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .virtual_expansion import ProbeSignal


@torch.no_grad()
def signal_from_candidate(model: nn.Module, candidate, inputs: Tensor,
                          gate: float = 1.0) -> ProbeSignal:
    """Extract a functional signal without importing or copying Gromo code."""
    baseline = model(inputs)
    with candidate.virtual_direction(gate):
        expanded = model(inputs)
    delta = expanded - baseline
    payload = candidate.payload
    A = payload.get("A_E")
    B = payload.get("B_E")
    if A is None or B is None:
        raise ValueError("candidate payload must expose A_E and B_E")
    return ProbeSignal(
        block=candidate.module_name,
        A_E=torch.as_tensor(A).detach(), B_E=torch.as_tensor(B).detach(),
        delta_feature=torch.empty(0, device=delta.device),
        delta_logits=delta.detach(),
        predicted_gain=float(candidate.verified_gain or candidate.proposal_score),
        singular_values=torch.as_tensor(
            payload.get("tiny_eigenvalues", []), device=delta.device))

