"""Virtual-NTK Augmentation core products and diagnostics.

The implementation is deliberately controller-free: it builds the block-local
VNA correction for one selected Gromo site and returns tensors that can be
added to the matching parameter gradients by the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, jvp, vjp

from methods.gromo_adapter import _assert_unchanged, _snapshot
from projection.functional import _preserve_functional_state


def _module_path(model: nn.Module, target: nn.Module) -> str:
    matches = [name for name, module in model.named_modules() if module is target]
    if len(matches) != 1:
        raise RuntimeError("candidate block is not uniquely registered on model")
    return matches[0]








def _purpose_seed(seed: int, purpose: str) -> int:
    digest = hashlib.sha256(f"{seed}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def dedicated_generator(seed: int, purpose: str, device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(_purpose_seed(seed, purpose))


def _as_a_dict(A: Tensor | Mapping[str, Tensor]) -> dict[str, Tensor | None]:
    if isinstance(A, Tensor):
        return {"weight": A.detach(), "bias": None}
    if "weight" not in A:
        raise KeyError("A must contain a 'weight' tensor")
    return {
        "weight": torch.as_tensor(A["weight"]).detach(),
        "bias": None if A.get("bias") is None else torch.as_tensor(A["bias"]).detach(),
    }


def _rank_of_a(A: Tensor | Mapping[str, Tensor]) -> int:
    weight = _as_a_dict(A)["weight"]
    assert isinstance(weight, Tensor)
    return int(weight.shape[0])




def random_A_like(
    A: Tensor | Mapping[str, Tensor],
    *,
    generator: torch.Generator,
) -> dict[str, Tensor | None]:
    """Random-A control with matched rank and total Frobenius norm."""
    source = _as_a_dict(A)
    weight = source["weight"]
    bias = source["bias"]
    assert isinstance(weight, Tensor)
    random_weight = torch.randn(
        weight.shape, dtype=weight.dtype, device=weight.device, generator=generator)
    random_bias = (
        None if bias is None else torch.randn(
            bias.shape, dtype=bias.dtype, device=bias.device, generator=generator))
    source_norm_sq = weight.square().sum()
    random_norm_sq = random_weight.square().sum()
    if bias is not None and random_bias is not None:
        source_norm_sq = source_norm_sq + bias.square().sum()
        random_norm_sq = random_norm_sq + random_bias.square().sum()
    scale = (source_norm_sq / random_norm_sq.clamp_min(torch.finfo(weight.dtype).tiny)).sqrt()
    return {
        "weight": random_weight * scale,
        "bias": None if random_bias is None else random_bias * scale.to(random_bias),
    }


def candidate_projection_block(model: nn.Module, candidate) -> str:
    logical_name = str(candidate.module_name)
    if hasattr(model, "projection_parameter_modules"):
        return logical_name
    if logical_name in dict(model.named_modules()):
        return logical_name
    resolver = getattr(model, "block", None)
    if resolver is None:
        raise KeyError(f"cannot resolve candidate block {logical_name!r}")
    return _module_path(model, resolver(logical_name))


def candidate_projection_parameter_names(
        model: nn.Module, candidate,
        scope: str = "residual_path") -> tuple[str, ...] | None:
    """Select residual-path trainable coordinates while excluding shortcuts."""
    if scope == "whole_block":
        return None
    if scope not in {"residual_path", "conv_only"}:
        raise ValueError(f"unknown projection scope {scope!r}")
    custom_resolver = getattr(model, "projection_parameter_modules", None)
    if custom_resolver is not None:
        resolved = custom_resolver(str(candidate.module_name))
        if isinstance(resolved, dict):
            modules = list(resolved[scope])
        else:
            modules = list(resolved[:2] if scope == "conv_only" else resolved)
        parameter_ids = {
            id(parameter) for module in modules
            for parameter in module.parameters(recurse=True)}
        names = tuple(name for name, parameter in model.named_parameters()
                      if id(parameter) in parameter_ids)
        if not names:
            raise RuntimeError("custom projection scope selected no parameters")
        return names
    block_path = candidate_projection_block(model, candidate)
    block = dict(model.named_modules())[block_path]
    if hasattr(block, "first_layer") and hasattr(block, "second_layer"):
        convolution_modules = (
            block.first_layer.layer, block.second_layer.layer)
    elif hasattr(block, "conv1") and hasattr(block, "conv2"):
        convolution_modules = (block.conv1, block.conv2)
    else:
        raise TypeError("candidate block does not expose its two-convolution path")
    modules = list(convolution_modules)
    if scope == "residual_path":
        if hasattr(block, "first_layer") and hasattr(block, "second_layer"):
            post_functions = (block.first_layer.post_layer_function,
                              block.second_layer.post_layer_function)
            modules.extend(
                child for post in post_functions for child in post.modules()
                if isinstance(child, nn.modules.batchnorm._BatchNorm))
        else:
            modules.extend(
                module for name in ("bn1", "bn2")
                if isinstance(
                    (module := getattr(block, name, None)),
                    nn.modules.batchnorm._BatchNorm))
    parameter_ids = {
        id(parameter) for module in convolution_modules
        for parameter in module.parameters(recurse=False)} | {
        id(parameter) for module in modules
        for parameter in module.parameters(recurse=False)}
    names = tuple(name for name, parameter in model.named_parameters()
                  if id(parameter) in parameter_ids)
    if not names:
        raise RuntimeError("residual-path projection selected no parameters")
    return names


def _resolve_site(model: nn.Module, site) -> tuple[str, nn.Module, tuple[str, ...]]:
    if hasattr(site, "resolve"):
        return site.resolve(model)
    if isinstance(site, str):
        from types import SimpleNamespace
        site = SimpleNamespace(module_name=site)
    block_name = candidate_projection_block(model, site)
    block = dict(model.named_modules())[block_name]
    parameter_names = candidate_projection_parameter_names(model, site, "residual_path")
    if parameter_names is None:
        raise RuntimeError("VNA requires an explicit residual-path parameter scope")
    return block_name, block, tuple(parameter_names)


def _make_matching_b_layer(second_layer, rank: int, dtype: torch.dtype, device: torch.device) -> nn.Module:
    layer = second_layer.layer
    if isinstance(layer, nn.Conv2d):
        return nn.Conv2d(
            rank,
            layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            bias=layer.bias is not None,
            device=device,
            dtype=dtype,
        )
    if isinstance(layer, nn.Linear):
        return nn.Linear(rank, layer.out_features, bias=layer.bias is not None, device=device, dtype=dtype)
    raise TypeError(f"unsupported Gromo second layer type: {type(layer)!r}")


def _make_a_layer(first_layer, A: Mapping[str, Tensor | None]) -> nn.Module:
    base = first_layer.layer
    weight = A["weight"]
    bias = A["bias"]
    assert isinstance(weight, Tensor)
    if isinstance(base, nn.Conv2d):
        layer = nn.Conv2d(
            base.in_channels,
            weight.shape[0],
            kernel_size=base.kernel_size,
            stride=base.stride,
            padding=base.padding,
            dilation=base.dilation,
            bias=bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
    elif isinstance(base, nn.Linear):
        layer = nn.Linear(base.in_features, weight.shape[0], bias=bias is not None, device=weight.device, dtype=weight.dtype)
    else:
        raise TypeError(f"unsupported Gromo first layer type: {type(base)!r}")
    layer.weight = nn.Parameter(weight.detach().clone())
    if bias is not None:
        layer.bias = nn.Parameter(bias.detach().clone())
    return layer


@contextmanager
def _virtual_b_context(model: nn.Module, block: nn.Module, A: Mapping[str, Tensor | None]):
    rank = _rank_of_a(A)
    weight = A["weight"]
    assert isinstance(weight, Tensor)
    old = (
        block.first_layer.extended_output_layer,
        block.second_layer.extended_input_layer,
        block.first_layer.output_extension_scaling,
        block.second_layer.input_extension_scaling,
        block.second_layer.optimal_delta_scaling,
        getattr(model, "_extended_forward", False),
    )
    # Constructors initialize parameters; isolate that otherwise invisible RNG use.
    devices = [weight.device.index] if weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        a_layer = _make_a_layer(block.first_layer, A)
        b_layer = _make_matching_b_layer(block.second_layer, rank, weight.dtype, weight.device)
    with torch.no_grad():
        b_layer.weight.zero_()
        if getattr(b_layer, "bias", None) is not None:
            b_layer.bias.zero_()
    block.first_layer.extended_output_layer = a_layer
    block.second_layer.extended_input_layer = b_layer
    block.first_layer.output_extension_scaling = torch.ones(1, device=weight.device, dtype=weight.dtype)
    block.second_layer.input_extension_scaling = torch.ones(1, device=weight.device, dtype=weight.dtype)
    block.second_layer.optimal_delta_scaling = torch.zeros(1, device=weight.device, dtype=weight.dtype)
    if hasattr(model, "_extended_forward"):
        model._extended_forward = True
    try:
        yield b_layer
    finally:
        block.first_layer.extended_output_layer = old[0]
        block.second_layer.extended_input_layer = old[1]
        block.first_layer.output_extension_scaling = old[2]
        block.second_layer.input_extension_scaling = old[3]
        block.second_layer.optimal_delta_scaling = old[4]
        if hasattr(model, "_extended_forward"):
            model._extended_forward = old[5]


def _copy_buffers(model: nn.Module) -> dict[str, Tensor]:
    return {name: buffer.detach().clone() for name, buffer in model.named_buffers()}


def _flatten(values: tuple[Tensor, ...]) -> Tensor:
    return torch.cat([value.reshape(-1) for value in values])


def _unflatten(vector: Tensor, refs: tuple[tuple[str, Tensor], ...]) -> dict[str, Tensor]:
    sizes = [ref.numel() for _, ref in refs]
    pieces = vector.split(sizes)
    return {name: piece.reshape_as(ref) for (name, ref), piece in zip(refs, pieces)}


class _Suffix(nn.Module):
    """Exact native Gromo forward from a fixed selected block input onward."""
    def __init__(self, model, start, inputs):
        super().__init__()
        self.model, self.start, self.inputs = model, start, inputs

    def forward(self):
        x = self.inputs
        nodes = [node for stage in self.model.core.stages for node in stage]
        for node in nodes[self.start:]:
            x = node.extended_forward(x) if hasattr(node, "extended_forward") else node(x)
        return self.model.core.post_net(x)


class _Products:
    def __init__(
        self,
        model: nn.Module,
        inputs: Tensor,
        parameter_names: tuple[str, ...],
        extension_names: tuple[str, ...],
        *,
        bn_mode: str,
        cache_block_input: bool = False,
    ) -> None:
        if bn_mode not in {"eval", "train"}:
            raise ValueError("bn_mode must be 'eval' or 'train'")
        self.model = model
        self.inputs = inputs
        self.parameter_refs = tuple((name, dict(model.named_parameters())[name].detach()) for name in parameter_names)
        named_parameters = dict(model.named_parameters())
        self.extension_refs = tuple((name, named_parameters[name].detach()) for name in extension_names)
        self.bn_mode = bn_mode
        self.buffers = _copy_buffers(model) if bn_mode == "train" else {}
        self.forward_calls = 0
        self._pullbacks = {}
        self.suffix = None
        if cache_block_input and bn_mode == "eval" and hasattr(model, "core"):
            block_path = parameter_names[0].split(".first_layer")[0]
            target = dict(model.named_modules())[block_path]
            nodes = [node for stage in model.core.stages for node in stage]
            if target not in nodes:
                raise ValueError("cached products require a native Gromo block")
            with torch.no_grad():
                x = model.core.pre_net(inputs)
                for node in nodes[:nodes.index(target)]:
                    x = node.extended_forward(x) if hasattr(node, "extended_forward") else node(x)
            self.forward_calls += 1  # one prefix evaluation
            self.suffix = _Suffix(model, nodes.index(target), x.detach())
        self.jvp_calls = 0
        self.vjp_calls = 0
        self.extension_jvp_calls = 0
        self.extension_vjp_calls = 0

    def _call(self, replacements: Mapping[str, Tensor]) -> Tensor:
        self.forward_calls += 1
        merged: dict[str, Tensor] = dict(replacements)
        if self.bn_mode == "train":
            merged.update({name: value.clone() for name, value in self.buffers.items()})
        if self.suffix is not None:
            return functional_call(self.suffix, {"model." + k: v for k, v in merged.items()}, (), strict=False).reshape(-1)
        return functional_call(self.model, merged, (self.inputs,), strict=False).reshape(-1)

    def _function_for(self, refs: tuple[tuple[str, Tensor], ...]):
        names = tuple(name for name, _ in refs)

        def fn(*values: Tensor) -> Tensor:
            return self._call(dict(zip(names, values)))

        return fn

    def _pullback(self, key, refs):
        if key not in self._pullbacks:
            _, self._pullbacks[key] = vjp(self._function_for(refs), *(r for _, r in refs))
        return self._pullbacks[key]

    def Jv(self, vector: Tensor) -> Tensor:
        self.jvp_calls += 1
        primals = tuple(ref for _, ref in self.parameter_refs)
        tangents = tuple(piece.reshape_as(ref).to(ref) for (_, ref), piece in zip(self.parameter_refs, vector.split([r.numel() for _, r in self.parameter_refs])))
        return jvp(self._function_for(self.parameter_refs), primals, tangents)[1].detach()

    def JT(self, vector: Tensor) -> Tensor:
        self.vjp_calls += 1
        primals = tuple(ref for _, ref in self.parameter_refs)
        pullback = self._pullback("J", self.parameter_refs)
        return _flatten(tuple(value.detach() for value in pullback(vector.detach().to(primals[0]))))

    def JEv(self, vector: Tensor) -> Tensor:
        self.extension_jvp_calls += 1
        primals = tuple(ref for _, ref in self.extension_refs)
        sizes = [ref.numel() for _, ref in self.extension_refs]
        tangents = tuple(piece.reshape_as(ref).to(ref) for (_, ref), piece in zip(self.extension_refs, vector.split(sizes)))
        return jvp(self._function_for(self.extension_refs), primals, tangents)[1].detach()

    def JET(self, vector: Tensor) -> Tensor:
        self.extension_vjp_calls += 1
        primals = tuple(ref for _, ref in self.extension_refs)
        pullback = self._pullback("E", self.extension_refs)
        return _flatten(tuple(value.detach() for value in pullback(vector.detach().to(primals[0]))))

    def K(self, vector: Tensor) -> Tensor:
        return self.Jv(self.JT(vector.to(self.inputs).detach())).detach()

    def KE(self, vector: Tensor) -> Tensor:
        return self.JEv(self.JET(vector.to(self.inputs).detach())).detach()


def _mean_ce_logit_grad(logits: Tensor, targets: Tensor) -> Tensor:
    probs = torch.softmax(logits.double(), dim=1)
    one_hot = F.one_hot(targets, probs.shape[1]).to(dtype=probs.dtype, device=probs.device)
    return ((probs - one_hot) / targets.numel()).reshape(-1)



def loss_logit_grad(logits, targets, loss_spec=None, ctx=None):
    """Autograd on the supplied loss; legacy default CE keeps its exact rounding.

    Stage A/Phase 0 used softmax-minus-onehot in float64. Retain those bits
    for default/explicit CE when no upstream context is supplied.
    """
    from methods.vna_losses import CE
    residual = (CE() if loss_spec is None else loss_spec).residual(
        logits, {'targets': targets, **(ctx or {})}).flatten()
    legacy_ce = (loss_spec is None or type(loss_spec) is CE) and ctx is None
    return _mean_ce_logit_grad(logits, targets) if legacy_ce else residual














__all__ = ["dedicated_generator", "random_A_like"]


@contextmanager
def block_products(model, site, A, inputs, bn_mode="eval", cache_block_input=False):
    """Read-only matrix-free operators, also used by Phase 0 diagnostics."""
    if hasattr(site, 'resolve'):
        from methods.vna_sites import hook_products
        with hook_products(model, [site], {site: A}, inputs, bn_mode=bn_mode,
                           cache_block_input=cache_block_input) as p:
            yield p
        return
    before = _snapshot(model)
    modes = {module: module.training for module in model.modules()}
    try:
        _, block, names = _resolve_site(model, site)
        model.eval()
        if bn_mode == "train":
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.train()
        with torch.autocast(device_type=inputs.device.type, enabled=False), \
                _preserve_functional_state(model), _virtual_b_context(model, block, _as_a_dict(A)) as outgoing:
            ids = {id(p) for p in outgoing.parameters()}
            ext_names = tuple(n for n, p in model.named_parameters() if id(p) in ids)
            yield _Products(model, inputs, names, ext_names, bn_mode=bn_mode, cache_block_input=cache_block_input)
    finally:
        for module, training in modes.items():
            module.training = training
        _assert_unchanged(model, before)
