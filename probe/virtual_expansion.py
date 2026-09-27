"""One-shot, non-persistent low-rank expansion probes for convolutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class ProbeSignal:
    block: str
    A_E: Tensor
    B_E: Tensor
    delta_feature: Tensor
    delta_logits: Tensor
    predicted_gain: float
    singular_values: Tensor

    @property
    def rank(self) -> int:
        return int(self.A_E.shape[0])


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    modules: Mapping[str, nn.Module] = dict(model.named_modules())
    if path not in modules:
        raise KeyError(f"unknown module {path!r}")
    return modules[path]


def _snapshot_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_state_unchanged(model: nn.Module, before: Mapping[str, Tensor]) -> None:
    after = model.state_dict()
    if before.keys() != after.keys():
        raise RuntimeError("probe changed the model state structure")
    changed = [name for name, value in after.items()
               if not torch.equal(value, before[name])]
    if changed:
        raise RuntimeError(f"probe mutated model state: {changed[:3]}")


class VirtualExpansionProbe:
    """Discover a rank-r virtual Conv2d branch in one backward pass.

    The negative weight gradient is truncated by SVD and factorized as
    ``B_E @ A_E``. A hook previews this low-rank branch exactly once. No
    optimizer is created and no expansion tensor is installed on the model.
    """

    def __init__(self, scale: float = 1e-2, eps: float = 1e-12):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = float(scale)
        self.eps = float(eps)

    def __call__(self, model: nn.Module, *, block: str, rank: int,
                 batch: tuple[Tensor, Tensor], loss_fn=None) -> ProbeSignal:
        if rank < 1:
            raise ValueError("rank must be positive")
        module = _resolve_module(model, block)
        if not isinstance(module, nn.Conv2d):
            raise TypeError("the initial probe supports nn.Conv2d blocks only")
        if module.groups != 1:
            raise ValueError("grouped convolutions are not supported")
        inputs, targets = batch
        loss_fn = loss_fn or F.cross_entropy
        before = _snapshot_state(model)
        modes = {item: item.training for item in model.modules()}
        parameter_grads = {
            parameter: None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in model.parameters()
        }
        captured: dict[str, Tensor] = {}

        def capture_input(_module, args):
            captured["input"] = args[0].detach()

        handle = module.register_forward_pre_hook(capture_input)
        try:
            model.eval()
            with torch.enable_grad():
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                gradient, output_gradient = torch.autograd.grad(
                    loss, (module.weight, logits))
            handle.remove()

            flat = gradient.reshape(gradient.shape[0], -1)
            U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
            effective_rank = min(rank, int(S.numel()))
            if effective_rank < 1 or float(S[0]) <= self.eps:
                raise RuntimeError("the probed convolution has a zero expansion direction")
            roots = S[:effective_rank].clamp_min(0).sqrt()
            A_E = (roots[:, None] * Vh[:effective_rank]).reshape(
                effective_rank, module.in_channels, *module.kernel_size)
            B_E = (-U[:, :effective_rank] * roots[None, :])[:, :, None, None]
            A_E, B_E = A_E.detach(), B_E.detach()

            feature_delta = F.conv2d(
                captured["input"], A_E, None, module.stride, module.padding,
                module.dilation, 1)
            feature_delta = F.conv2d(feature_delta, B_E) * self.scale
            preview: dict[str, Tensor] = {}

            def add_branch(_module, _args, output):
                delta = feature_delta.to(device=output.device, dtype=output.dtype)
                preview["feature"] = delta
                return output + delta

            preview_handle = module.register_forward_hook(add_branch)
            try:
                with torch.no_grad():
                    expanded_logits = model(inputs)
            finally:
                preview_handle.remove()
            delta_logits = expanded_logits - logits.detach()
            predicted_gain = float(-(output_gradient.detach() * delta_logits).sum())
            return ProbeSignal(
                block=block, A_E=A_E, B_E=B_E,
                delta_feature=preview["feature"].detach(),
                delta_logits=delta_logits.detach(),
                predicted_gain=predicted_gain,
                singular_values=S[:effective_rank].detach())
        finally:
            handle.remove()
            for item, training in modes.items():
                item.training = training
            for parameter, old_grad in parameter_grads.items():
                parameter.grad = old_grad
            _assert_state_unchanged(model, before)

