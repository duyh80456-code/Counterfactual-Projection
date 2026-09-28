"""Counterfactual TINY proposal beyond a model's configured target width."""

from __future__ import annotations

from dataclasses import dataclass

import torch

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

    @staticmethod
    def _measure_extension_flops(model, block, sample_inputs, rank: int) -> float:
        first, second = block.first_layer.layer, block.second_layer.layer
        spatial = {}

        def capture(name):
            def hook(_module, _inputs, output):
                spatial[name] = tuple(output.shape[-2:])
            return hook

        handles = [first.register_forward_hook(capture("first")),
                   second.register_forward_hook(capture("second"))]
        modes = {module: module.training for module in model.modules()}
        try:
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                model(sample_inputs[:1].to(device))
        finally:
            for handle in handles:
                handle.remove()
            for module, training in modes.items():
                module.training = training
        if spatial.keys() != {"first", "second"}:
            raise RuntimeError("could not measure the selected block's spatial sizes")
        first_kernel = first.kernel_size[0] * first.kernel_size[1]
        second_kernel = second.kernel_size[0] * second.kernel_size[1]
        first_macs = (spatial["first"][0] * spatial["first"][1] *
                      first.in_channels * first_kernel)
        second_macs = (spatial["second"][0] * spatial["second"][1] *
                       second.out_channels * second_kernel)
        return float(2 * rank * (first_macs + second_macs))

    def propose(self, adapter, model, statistics_loader, budget,
                sample_inputs=None):
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
        if sample_inputs is not None:
            actual_flops = self._measure_extension_flops(
                model, block, sample_inputs, effective)
            candidate.extra_flops = actual_flops
            candidate.payload.update({
                "actual_extension_flops": actual_flops,
                "flops_measured_from_runtime_spatial_shape": True,
                "probe_input_shape": list(sample_inputs.shape[1:]),
            })
        return candidate
