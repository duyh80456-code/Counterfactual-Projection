"""Counterfactual TINY proposal beyond a model's configured target width."""

from __future__ import annotations

from dataclasses import dataclass

from .gromo_adapter import TransactionalCandidateSource


@dataclass
class CounterfactualTinyProbe:
    """Ask TINY for current_width -> current_width + rank, then restore target.

    The target-width override exists only while Gromo computes its statistics
    and optimal extension. The returned candidate owns deep copies of the
    extension, so restoring ``target_in_neurons`` does not invalidate it.
    """

    rank: int
    module_name: str

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("counterfactual probe rank must be positive")

    def propose(self, adapter, model, statistics_loader, budget):
        block = model.block(self.module_name)
        second = block.second_layer
        current_width = int(second.in_neurons)
        old_target = second.target_in_neurons
        adapter.schedule_site(self.module_name, self.rank)
        second.target_in_neurons = current_width + self.rank
        try:
            candidates = TransactionalCandidateSource().propose(
                adapter, model, statistics_loader, budget)
        finally:
            second.target_in_neurons = old_target
        if len(candidates) != 1:
            raise RuntimeError("scheduled counterfactual TINY probe was not unique")
        candidate = candidates[0]
        effective = int(candidate.payload.get("effective_rank", 0))
        if not 0 < effective <= self.rank:
            raise RuntimeError(
                f"invalid counterfactual rank {effective}; requested {self.rank}")
        candidate.payload.update({
            "counterfactual_over_expansion": True,
            "base_hidden_width": current_width,
            "counterfactual_target_width": current_width + self.rank,
        })
        return candidate

