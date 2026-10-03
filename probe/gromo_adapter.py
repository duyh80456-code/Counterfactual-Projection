"""Transaction-only bridge to TINY/Gromo candidates.

No Gromo source is copied and no assumption is made about candidate payloads.
The main method needs only the public ``virtual_direction`` transaction.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .virtual_expansion import ProbeSignal


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


class CandidateExpansionProbe:
    """Measure delta-f from a real temporary structural expansion candidate."""

    def __call__(self, model: nn.Module, *, candidate, batch: tuple[Tensor, Tensor],
                 gate: float = 0.05, loss_fn=None) -> ProbeSignal:
        if not hasattr(candidate, "virtual_direction"):
            raise TypeError("candidate must provide virtual_direction(gate)")
        inputs, targets = batch
        gate = float(gate)
        if not 0 < gate <= 1:
            raise ValueError("finite-difference gate must be in (0, 1]")
        loss_fn = loss_fn or F.cross_entropy
        before = _snapshot(model)
        modes = {module: module.training for module in model.modules()}
        grads = {parameter: None if parameter.grad is None else parameter.grad.clone()
                 for parameter in model.parameters()}
        try:
            model.eval()
            with torch.no_grad():
                baseline = model(inputs)
                baseline_loss = loss_fn(baseline.float(), targets)
                with candidate.virtual_direction(gate):
                    expanded = model(inputs)
                    expanded_loss = loss_fn(expanded.float(), targets)
            raw_delta = expanded - baseline
            delta = raw_delta / gate
            if not torch.isfinite(delta).all() or float(delta.norm()) == 0:
                payload = getattr(candidate, "payload", {})
                source = payload.get("source", "unknown")
                raise RuntimeError(
                    "TINY/Gromo candidate produced zero/nonfinite delta-f: "
                    f"site={candidate.module_name}, source={source}, "
                    f"gate={gate}, delta_norm={float(delta.norm())}")
            payload = getattr(candidate, "payload", {})
            A_E, B_E = payload.get("A_E"), payload.get("B_E")
            return ProbeSignal(
                block=str(candidate.module_name),
                A_E=None if A_E is None else torch.as_tensor(A_E).detach(),
                B_E=None if B_E is None else torch.as_tensor(B_E).detach(),
                delta_feature=None, delta_logits=delta.detach(),
                predicted_gain=float((baseline_loss - expanded_loss).item() / gate),
                singular_values=(None if "tiny_eigenvalues" not in payload else
                                 torch.as_tensor(payload["tiny_eigenvalues"])),
                source=str(payload.get("source", "tiny_gromo_structural")),
                is_structural_expansion=True, probe_gate=gate,
                observed_loss_gain=float((baseline_loss - expanded_loss).item()))
        finally:
            for module, training in modes.items():
                module.training = training
            for parameter, gradient in grads.items():
                parameter.grad = gradient
            _assert_unchanged(model, before)


class TransactionalCandidateSource:
    """Run a TINY adapter's statistics/solve without leaking model state."""

    def propose(self, adapter, model: nn.Module, statistics_loader, budget):
        before = _snapshot(model)
        modes = {module: module.training for module in model.modules()}
        grads = {parameter: None if parameter.grad is None else parameter.grad.clone()
                 for parameter in model.parameters()}
        try:
            candidates = adapter.propose_all(model, statistics_loader, budget)
            if not candidates:
                raise RuntimeError("TINY/Gromo produced no structural candidate")
            return candidates
        finally:
            # Statistics collection may update BatchNorm buffers. Candidate
            # closures own deep copies of E, so restoring the base is safe.
            model.load_state_dict(before, strict=True)
            for module, training in modes.items():
                module.training = training
            for parameter, gradient in grads.items():
                parameter.grad = gradient
            _assert_unchanged(model, before)


def signal_from_candidate(model: nn.Module, candidate, inputs: Tensor,
                          targets: Tensor, gate: float = 0.05) -> ProbeSignal:
    return CandidateExpansionProbe()(
        model, candidate=candidate, batch=(inputs, targets), gate=gate)
