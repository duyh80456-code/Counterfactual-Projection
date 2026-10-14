"""Virtual-NTK Augmentation core products and diagnostics.

The implementation is deliberately controller-free: it builds the block-local
VNA correction for one selected Gromo site and returns tensors that can be
added to the matching parameter gradients by the caller.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import math
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, jvp, vjp

from probe.gromo_adapter import _assert_unchanged, _snapshot
from projection.cg import CGResult, conjugate_gradient
from projection.functional import _preserve_functional_state


def _module_path(model: nn.Module, target: nn.Module) -> str:
    matches = [name for name, module in model.named_modules() if module is target]
    if len(matches) != 1:
        raise RuntimeError("candidate block is not uniquely registered on model")
    return matches[0]


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


@dataclass(frozen=True)
class VNAConfig:
    lambda_vna: float = 1.0
    damping_rel: float = 1e-2
    cg_iters: int = 30
    cg_tol: float = 1e-2
    vna_every: int = 4
    vna_batch: int = 128
    r_max: int = 16
    tau_rel: float = 0.05
    rank_energy: float = 0.9
    dorfler_theta: float = 0.5
    max_sites: int = 3
    stat_samples: int = 1024
    n_boot: int = 4
    stat_ema: float = 0.5
    tau_rho: float = 0.3
    check_every: int = 1
    refresh_max_epochs: int = 10
    eta_drift: float = 0.5
    accept_steps: int = 50
    stop_patience: int = 3
    use_g3: bool = False
    bn_mode: str = "eval"
    hutchinson_probes: int = 8
    cache_block_input: bool = False
    seed: int = 20261013


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


def extract_tiny_A(model: nn.Module, candidate) -> tuple[dict[str, Tensor | None], int]:
    """Extract the incoming virtual-neuron map from a TINY candidate.

    In Gromo's ResNet BasicBlock this is the first layer's
    ``extended_output_layer``: it maps the block input into the r virtual hidden
    channels. The second layer's ``extended_input_layer`` is TINY's outgoing map
    and is intentionally ignored by VNA.
    """
    before = _snapshot(model)
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with candidate.virtual_direction(1.0):
            block = model.block(str(candidate.module_name))
            layer = block.first_layer.extended_output_layer
            if layer is None:
                raise RuntimeError("candidate has no first-layer virtual input map")
            A = {
                "weight": layer.weight.detach().clone(),
                "bias": None if layer.bias is None else layer.bias.detach().clone(),
            }
    finally:
        for module, training in modes.items():
            module.training = training
        _assert_unchanged(model, before)
    return A, _rank_of_a(A)


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

def _hutchinson_trace_and_diag(products: _Products, n: int, cfg: VNAConfig, device: torch.device) -> tuple[float, Tensor]:
    generator = dedicated_generator(cfg.seed, "vna-hutchinson", device=device)
    trace_terms = []
    diagonal = torch.zeros(n, dtype=torch.float64, device=device)
    for _ in range(cfg.hutchinson_probes):
        signs = torch.empty(n, dtype=torch.float64, device=device).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)
        image = products.K(signs.to(products.inputs)).to(dtype=torch.float64)
        trace_terms.append(float(torch.dot(signs, image)))
        diagonal.add_(signs * image)
    trace = sum(trace_terms) / max(1, len(trace_terms))
    diagonal = (diagonal / max(1, cfg.hutchinson_probes)).clamp_min(0)
    return trace, diagonal


def _phase0_cg(action, rhs, *, normalize_rhs=False, **kwargs):
    if not normalize_rhs:
        return conjugate_gradient(action, rhs, **kwargs)
    scale = float(rhs.norm())
    if not scale:
        return conjugate_gradient(action, rhs, **kwargs)
    if kwargs.get('x0') is not None:
        kwargs['x0'] = kwargs['x0'] / scale
    result = conjugate_gradient(action, rhs / scale, **kwargs)
    return replace(result, solution=result.solution * scale,
                   residual_norm=result.residual_norm * scale,
                   residual_history=tuple(v * scale for v in result.residual_history))


def _cg_record(result: CGResult, *, include_breakdown=False) -> dict[str, Any]:
    return {
        "iterations": result.iterations,
        "residual_norm": result.residual_norm,
        "relative_residual": result.relative_residual,
        "converged": result.converged,
        **({"breakdown": result.breakdown} if include_breakdown else {}),
        "residual_history": list(result.residual_history),
    }


def vna_block_correction(
    model: nn.Module,
    site,
    A: Tensor | Mapping[str, Tensor],
    batch: tuple[Tensor, Tensor],
    cfg: VNAConfig,
    warm_start: Mapping[str, Tensor] | None = None,
    *, damping_cache: Mapping[str, Any] | None = None,
    shared_logits: Tensor | None = None, shared_g: Tensor | None = None,
    apply_gain_guard: bool = True, _lite: bool = False, cg_breakdown_flags: bool = False,
    rhs_normalized_cg: bool = False, loss_spec=None, ctx=None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Return ``lambda_vna * J^T R K_E R K g`` for addition to gradients.

    Cache ``diag["damping_cache"]`` at refresh and pass it back on subsequent
    steps; ``diag["warm_start"]`` contains the previous z/q iterates.
    """
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize(next(model.parameters()).device)
    start = time.perf_counter()
    before = _snapshot(model)
    modes = {module: module.training for module in model.modules()}
    grads = {parameter: None if parameter.grad is None else parameter.grad.detach().clone()
             for parameter in model.parameters()}
    diag: dict[str, Any] = {"nonfinite": False, "skipped": False,
                            "apply_gain_guard": apply_gain_guard, "variant": "lite" if _lite else "full",
                            "jvp_calls": 0, "vjp_calls": 0,
                            "extension_jvp_calls": 0, "extension_vjp_calls": 0}
    products = None
    try:
        block_name, block, parameter_names = _resolve_site(model, site)
        inputs, targets = batch
        inputs = inputs.detach().to(dtype=next(model.parameters()).dtype)
        targets = targets.detach().to(inputs.device)
        A_dict = _as_a_dict(A)
        # Disable dropout in both BN modes; train mode applies only to BN.
        model.eval()
        if cfg.bn_mode == "train":
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.train()
        with block_products(model, site, A_dict, inputs, cfg.bn_mode,
                            cache_block_input=cfg.cache_block_input) as products:
            logits = products._call({}).reshape(len(inputs), -1) if shared_logits is None else shared_logits
            g = loss_logit_grad(logits, targets, loss_spec, ctx) if shared_g is None else shared_g
            n = g.numel()
            if damping_cache is None:
                trace, jacobi = _hutchinson_trace_and_diag(products, n, cfg, g.device)
                mu = cfg.damping_rel * max(trace / max(1, n), 0.0)
            else:
                mu = float(damping_cache["mu"])
                jacobi = damping_cache["jacobi"].to(g)
                trace = float(damping_cache.get("trace_K", float("nan")))
            if not math.isfinite(mu) or not torch.isfinite(g).all() or not torch.isfinite(jacobi).all():
                diag.update({"nonfinite": True})
                return {}, diag
            jt_g = products.JT(g.to(inputs)).to(dtype=torch.float64)
            w = products.Jv(jt_g.to(inputs)).to(dtype=torch.float64)
            inv_diag = (jacobi + mu).clamp_min(1e-12).reciprocal()

            def matvec(vector: Tensor) -> Tensor:
                return products.K(vector.to(inputs)).to(dtype=torch.float64) + mu * vector

            def precondition(vector: Tensor) -> Tensor:
                return inv_diag * vector

            if _lite:
                c = (trace / max(1, n)) ** 2
                if not math.isfinite(c) or c <= 0:
                    diag.update(nonfinite=True, skipped=True, skip_reason="invalid_trace_scale")
                    return {}, diag
                z = w / math.sqrt(c)
                s = products.KE(z.to(inputs)).to(dtype=torch.float64)
                q = s / math.sqrt(c)
                z_result = q_result = None
            else:
                z0 = None if warm_start is None else warm_start.get("z")
                q0 = None if warm_start is None else warm_start.get("q")
                z_result = _phase0_cg(
                    matvec, w, x0=None if z0 is None else z0.to(w),
                    max_iter=cfg.cg_iters, tolerance=cfg.cg_tol,
                    preconditioner=precondition, eps=1e-30, normalize_rhs=rhs_normalized_cg)
                z = z_result.solution
                s = products.KE(z.to(inputs)).to(dtype=torch.float64)
                q_result = _phase0_cg(
                    matvec, s, x0=None if q0 is None else q0.to(s),
                    max_iter=cfg.cg_iters, tolerance=cfg.cg_tol,
                    preconditioner=precondition, eps=1e-30, normalize_rhs=rhs_normalized_cg)
                q = q_result.solution
            d_flat = products.JT(q.to(inputs)).to(dtype=torch.float64)
            je_t_g = products.JET(g.to(inputs)).to(dtype=torch.float64)
            je_t_z = products.JET(z.to(inputs)).to(dtype=torch.float64)
            U = float(je_t_g.square().sum())
            V = float(je_t_z.square().sum())
            eta = float(V / U) if U > 0 else 0.0
            jt_g_norm_sq = float(jt_g.square().sum())
            rho = float(cfg.lambda_vna * V / jt_g_norm_sq) if jt_g_norm_sq > 0 else 0.0
            per_neuron = []
            offset = 0
            for name, ref in products.extension_refs:
                size = ref.numel()
                piece = je_t_z[offset:offset + size]
                offset += size
                if name.endswith(".weight") and ref.ndim >= 2:
                    per_neuron.extend(piece.reshape(ref.shape).movedim(1, 0).flatten(1).square().sum(dim=1).tolist())
            # An outgoing bias is shared across virtual channels. Attribute
            # its energy equally so the partition still sums exactly to V.
            if any(n.endswith('.bias') for n, _ in products.extension_refs):
                energy = float(je_t_z[-products.extension_refs[-1][1].numel():].square().sum())
                per_neuron = [value + energy / len(per_neuron) for value in per_neuron]
            finite = (
                torch.isfinite(d_flat).all()
                and torch.isfinite(z).all()
                and torch.isfinite(q).all()
                and math.isfinite(U)
                and math.isfinite(V)
            )
            if not finite:
                diag.update({"nonfinite": True, "wall_time": time.perf_counter() - start})
                return {}, diag
            grad_delta = {
                name: (cfg.lambda_vna * value).detach().to(dict(model.named_parameters())[name])
                for name, value in _unflatten(d_flat.to(dtype=products.parameter_refs[0][1].dtype), products.parameter_refs).items()
            }
            if not all(torch.isfinite(value).all() for value in grad_delta.values()):
                diag["nonfinite"] = True
                return {}, diag
            realized_gain = float(g.dot(products.Jv(_flatten(tuple(grad_delta[n] for n, _ in products.parameter_refs))).double()))
            diag["realized_gain"] = realized_gain
            nonpositive_gain = realized_gain <= 0
            if not math.isfinite(realized_gain):
                diag["nonfinite"] = True
                return {}, diag
            if inputs.is_cuda:
                torch.cuda.synchronize(inputs.device)
            diag.update({
                "realized_gain": realized_gain,
                "nonpositive_gain": nonpositive_gain,
                "skipped": apply_gain_guard and nonpositive_gain,
                "apply_gain_guard": apply_gain_guard,
                "variant": "lite" if _lite else "full",
                "skip_reason": "nonpositive_realized_gain" if apply_gain_guard and nonpositive_gain else None,
                "ce": float(F.cross_entropy(logits.double(), targets)),
                "bn_mode": cfg.bn_mode,
                "damping_cache": {"mu": mu, "jacobi": jacobi.detach(), "trace_K": trace},
                "site": block_name,
                "rank": _rank_of_a(A_dict),
                "mu": mu,
                "trace_K": trace,
                "U": U,
                "V": V,
                "eta": eta,
                "rho": rho,
                "per_neuron_V": per_neuron,
                "jt_g_norm_sq": jt_g_norm_sq,
                "cg_z": None if _lite else _cg_record(z_result, include_breakdown=cg_breakdown_flags),
                "cg_q": None if _lite else _cg_record(q_result, include_breakdown=cg_breakdown_flags),
                "warm_start": {"z": z.detach(), "q": q.detach()},
                "jvp_calls": products.jvp_calls,
                "vjp_calls": products.vjp_calls,
                "extension_jvp_calls": products.extension_jvp_calls,
                "extension_vjp_calls": products.extension_vjp_calls,
                "wall_time": time.perf_counter() - start,
            })
            if _lite:
                diag.update(c=c, V_lite=V, rho_lite=rho)
            return ({} if apply_gain_guard and nonpositive_gain else grad_delta), diag
    except (FloatingPointError, torch.linalg.LinAlgError) as error:
        diag.update({
            "nonfinite": True,
            "error": f"{type(error).__name__}: {error}",
            "wall_time": time.perf_counter() - start,
        })
        return {}, diag
    finally:
        if diag["nonfinite"]:
            diag.update(skipped=True, skip_reason=diag.get("skip_reason") or "nonfinite")
        if products is not None:
            for key in ("forward_calls", "jvp_calls", "vjp_calls", "extension_jvp_calls", "extension_vjp_calls"):
                diag[key] = getattr(products, key)
        if next(model.parameters()).is_cuda:
            torch.cuda.synchronize(next(model.parameters()).device)
        diag["wall_time"] = time.perf_counter() - start
        for module, training in modes.items():
            module.training = training
        for parameter, gradient in grads.items():
            parameter.grad = gradient
        _assert_unchanged(model, before)


def vna_lite_correction(
    model: nn.Module, site, A: Tensor | Mapping[str, Tensor],
    batch: tuple[Tensor, Tensor], cfg: VNAConfig, *,
    damping_cache: Mapping[str, Any] | None = None,
    shared_logits: Tensor | None = None, shared_g: Tensor | None = None,
    apply_gain_guard: bool = True, loss_spec=None, ctx=None,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Return λ Jᵀ K_E K g / c without CG, c=(tr(K)/n)².

    All products, including Jᵀg, use cfg.bn_mode. Stage 2 may reuse an SGD
    train-mode parameter gradient only with train BN and copied buffers;
    eval BN requires recomputing Jᵀg. shared_g is a logit gradient, not Jᵀg.
    A refresh cache supplies the same Hutchinson trace as full VNA.
    """
    return vna_block_correction(
        model, site, A, batch, cfg, damping_cache=damping_cache,
        shared_logits=shared_logits, shared_g=shared_g,
        apply_gain_guard=apply_gain_guard, _lite=True, loss_spec=loss_spec, ctx=ctx)


def add_vna_grad_(model: nn.Module, grad_delta: Mapping[str, Tensor], scale: float = 1.0) -> None:
    parameters = dict(model.named_parameters())
    for name, delta in grad_delta.items():
        parameter = parameters[name]
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        parameter.grad.add_(delta.to(parameter.grad), alpha=float(scale))



__all__ = ["VNAConfig", "dedicated_generator", "extract_tiny_A", "random_A_like",
           "vna_block_correction", "vna_lite_correction", "add_vna_grad_"]


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


def assert_disjoint_scopes(model, sites):
    seen = set()
    for site in sites:
        _, _, names = _resolve_site(model, site)
        if seen.intersection(names):
            raise ValueError("selected residual-path scopes overlap")
        seen.update(names)
