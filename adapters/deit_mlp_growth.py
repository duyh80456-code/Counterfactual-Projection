"""Wrap an original DeiT MLP pair in the existing Gromo Linear TINY solver.

Gromo wrappers exist only while collecting statistics and solving. Candidates
own detached extension tensors; their hooks implement fc2(GELU(fc1(z))) +
gamma B GELU(A z + a), exactly the added-hidden-unit width expansion.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import types

import torch
from torch import nn
import torch.nn.functional as F

from probe.gromo_adapter import _snapshot, _assert_unchanged
from experiments.shared_protocol import rng_state, restore_rng

GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"


@dataclass
class MLPExpansionCandidate:
    module_name: str
    model: nn.Module
    A: torch.Tensor
    a: torch.Tensor
    B: torch.Tensor
    proposal_score: float
    payload: dict

    @contextmanager
    def virtual_direction(self, gate):
        mlp = DeitMLPGrowthAdapter.resolve_site(self.model, self.module_name)
        gamma = float(gate)
        if not torch.isfinite(torch.tensor(gamma)):
            raise ValueError("gate must be finite")

        def expanded(_module, inputs, output):
            if gamma == 0:
                return output  # bitwise identity, not merely zero times E
            z = inputs[0]
            extension = F.linear(F.gelu(F.linear(z, self.A, self.a)), self.B)
            return output + gamma * extension

        handle = mlp.register_forward_hook(expanded)
        try:
            yield
        finally:
            handle.remove()


class DeitMLPGrowthAdapter:
    @staticmethod
    def enumerate_sites(model):
        return tuple(f"blocks.{index}.mlp" for index in range(len(model.blocks)))

    @staticmethod
    def resolve_site(model, site):
        if site not in DeitMLPGrowthAdapter.enumerate_sites(model):
            raise KeyError(f"not a DeiT MLP expansion site: {site}")
        return dict(model.named_modules())[site]

    @staticmethod
    def original_mlp_parameters(model, site):
        DeitMLPGrowthAdapter.resolve_site(model, site)
        return tuple(f"{site}.{layer}.{field}" for layer in ("fc1", "fc2")
                     for field in ("weight", "bias"))

    def propose_auxiliary_growth(self, *, model, site, batches, rank=8):
        from gromo.modules.linear_growing_module import LinearGrowingModule

        if rank < 1 or not batches:
            raise ValueError("positive rank and nonempty statistics batches required")
        mlp = self.resolve_site(model, site)
        before = _snapshot(model)
        modes = {module: module.training for module in model.modules()}
        grads = {p: None if p.grad is None else p.grad.detach().clone() for p in model.parameters()}
        rng = rng_state()
        # Restore the exact instance dictionary, including absence of an
        # instance forward override, so no temporary wrapper escapes.
        old_forward = mlp.__dict__.get("forward")
        second = None
        try:
            device, dtype = mlp.fc1.weight.device, mlp.fc1.weight.dtype
            first = LinearGrowingModule(mlp.fc1.in_features, mlp.fc1.out_features,
                post_layer_function=nn.GELU(), device=device, name=site + ".fc1").to(dtype=dtype)
            second = LinearGrowingModule(mlp.fc2.in_features, mlp.fc2.out_features,
                previous_module=first, allow_growing=True,
                target_in_features=mlp.fc2.in_features + rank,
                device=device, name=site + ".fc2").to(dtype=dtype)
            first.layer.load_state_dict(mlp.fc1.state_dict())
            second.layer.load_state_dict(mlp.fc2.state_dict())
            first.eval()
            second.eval()
            mlp.forward = types.MethodType(lambda _self, z: second(first(z)), mlp)
            model.eval()
            second.init_computation()
            for inputs, labels in batches:
                model.zero_grad(set_to_none=True)
                first.zero_grad(set_to_none=True)
                second.zero_grad(set_to_none=True)
                logits = model(inputs.to(device))
                F.cross_entropy(logits.float(), labels.to(device), reduction="sum").backward()
                second.update_computation()
            second.compute_optimal_updates(compute_delta=True, use_covariance=True,
                alpha_zero=False, use_projection=True, maximum_added_neurons=rank,
                numerical_threshold=1e-6, statistical_threshold=1e-3)
            outgoing, incoming = first.extended_output_layer, second.extended_input_layer
            values = second.eigenvalues_extension
            if outgoing is None or incoming is None or values is None or values.numel() == 0:
                raise RuntimeError(f"native Linear TINY produced no extension at {site}")
            effective = min(rank, values.numel())
            A = outgoing.weight[:effective].detach().to(mlp.fc1.weight).clone()
            a = outgoing.bias[:effective].detach().to(mlp.fc1.weight).clone()
            B = incoming.weight[:, :effective].detach().to(mlp.fc2.weight).clone()
            if not all(torch.isfinite(value).all() for value in (A, a, B)):
                raise RuntimeError(f"nonfinite TINY extension at {site}")
            return MLPExpansionCandidate(site, model, A, a, B,
                float(values[:effective].square().sum()), {
                    "source": "native_gromo_linear_tiny", "gromo_commit": GROMO_COMMIT,
                    "requested_rank": rank, "effective_rank": int(effective),
                    "base_hidden_width": mlp.fc1.out_features,
                    "tiny_eigenvalues": values[:effective].detach().cpu().tolist(),
                    "auxiliary_training_steps": 0,
                    "function_preserving_at_gate_zero": True,
                    "statistics_loss_reduction": "sum",
                    "statistics_token_rule": "native_gromo_flatten_tokens_normalize_by_images"})
        finally:
            if old_forward is None:
                mlp.__dict__.pop("forward", None)
            else:
                mlp.forward = old_forward
            if second is not None:
                second.reset_computation()
                second.delete_update(include_previous=True)
            model.load_state_dict(before, strict=True)
            for p, grad in grads.items():
                p.grad = grad
            for module, mode in modes.items():
                module.training = mode
            restore_rng(rng)
            _assert_unchanged(model, before)
