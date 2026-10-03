"""Counterfactual TINY transactions for Gromo's native VGG container."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VGGPairRef:
    stage: int
    link: int
    name: str
    module: object
    is_boundary: bool = False


class VGGConvPair:
    """A VGG conv link, optionally separated by its native MaxPool bridge."""

    def __init__(self, first_layer, second_layer, bridge=None):
        self.first_layer = first_layer
        self.second_layer = second_layer
        self.bridge = bridge
        self.is_boundary = bridge is not None

    @property
    def hidden_neurons(self) -> int:
        return int(self.second_layer.in_neurons)

    @property
    def eigenvalues_extension(self):
        return self.second_layer.eigenvalues_extension

    def missing_neurons(self) -> int:
        return int(self.second_layer.missing_neurons())

    def delete_update(self) -> None:
        self.second_layer.delete_update(include_previous=True)


class GromoVGG16(nn.Module):
    """CIFAR VGG16-BN exposing all twelve adjacent conv interfaces."""

    architecture_id = "cifar_gromo_vgg16_bn_full_random_v2_bridge_tiny"

    def __init__(self, num_classes=100, device="cpu", expansion_capacity=64):
        super().__init__()
        from gromo.containers.vgg import VGG

        widths = ((64, 64), (128, 128), (256, 256, 256),
                  (512, 512, 512), (512, 512, 512))
        cfg = []
        target_cfg = []
        total_convolutions = sum(map(len, widths))
        convolution_index = 0
        for stage in widths:
            cfg.extend(stage)
            cfg.append("M")
            for width in stage:
                target_cfg.append(
                    width + int(expansion_capacity)
                    if convolution_index < total_convolutions - 1 else width)
                convolution_index += 1
            target_cfg.append("M")
        self.core = VGG(
            cfg=cfg, target_cfg=target_cfg, in_features=3,
            normalization="batch", num_classes=num_classes,
            init_weights=True, dropout=0.0, number_of_fc_layers=1,
            fc_layer_width=512, input_spatial_shape=(32, 32), device=device)
        self._extended_forward = False
        refs = []
        for stage_index, stage in enumerate(self.core.stage_blocks):
            layers = list(stage.growing_modules)
            for link_index, (first, second) in enumerate(
                    zip(layers[:-1], layers[1:])):
                refs.append(VGGPairRef(
                    stage_index, link_index,
                    f"stages.{stage_index}.links.{link_index}",
                    VGGConvPair(first, second)))
            if stage_index < len(self.core.stage_blocks) - 1:
                source = layers[-1]
                consumer = self.core.stage_blocks[stage_index + 1].growing_modules[0]
                bridge_index = 2 * stage_index + 1
                bridge = self.core.features[bridge_index]
                if not isinstance(bridge, nn.MaxPool2d):
                    raise RuntimeError("expected native MaxPool at VGG stage boundary")
                consumer.previous_module = source
                source.next_module = consumer
                consumer._allow_growing = True
                consumer.target_in_neurons = int(source.target_in_neurons)
                refs.append(VGGPairRef(
                    stage_index, len(layers) - 1,
                    f"stages.{stage_index}.boundary_to_{stage_index + 1}",
                    VGGConvPair(source, consumer, bridge=bridge),
                    is_boundary=True))

        # The native container omits each stage's first conv from its growable
        # list. Register the four boundary consumers in architectural order so
        # each object-identity lookup remains unambiguous.
        convolution_targets = [ref.module.second_layer for ref in refs]
        classifier_targets = [layer for layer in self.core._growable_layers
                              if all(layer is not target
                                     for target in convolution_targets)]
        self.core._growable_layers = convolution_targets + classifier_targets
        target_ids = [id(layer) for layer in convolution_targets]
        if len(target_ids) != 12 or len(set(target_ids)) != 12:
            raise RuntimeError("VGG conv-link targets must be twelve unique modules")
        if any(sum(layer is target for layer in self.core._growable_layers) != 1
               for target in convolution_targets):
            raise RuntimeError("VGG conv-link target registration is not unique")
        for stage_index in range(1, len(self.core.stage_blocks)):
            self.core.stage_blocks[stage_index].growable_layers.insert(
                0, self.core.stage_blocks[stage_index].growing_modules[0])
        self.core.set_growing_layers(scheduling_method="all")
        if len(refs) != 12:
            raise RuntimeError(f"expected 12 VGG conv links, got {len(refs)}")
        self._pair_refs = refs

    def _apply(self, fn):
        result = super()._apply(fn)
        parameter = next(self.parameters(), None)
        if parameter is not None:
            for module in self.modules():
                if hasattr(module, "device"):
                    module.device = parameter.device
        return result

    def forward(self, inputs):
        return (self.core.extended_forward(inputs) if self._extended_forward
                else self.core(inputs))

    def growing_blocks(self):
        return list(self._pair_refs)

    def block(self, name):
        matches = [ref.module for ref in self._pair_refs if ref.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def projection_parameter_modules(self, name):
        pair = self.block(name)
        return (pair.first_layer.layer, pair.second_layer.layer,
                pair.first_layer.post_layer_function,
                pair.second_layer.post_layer_function)


class VggTinyAdapter:
    """Expose native VGG conv-link TINY solves as virtual candidates only."""

    def __init__(self, quantum_params, max_statistics_batches=0, **_kwargs):
        if quantum_params < 1:
            raise ValueError("quantum_params must be positive")
        self.quantum_params = int(quantum_params)
        self.max_statistics_batches = int(max_statistics_batches)
        self.scheduled_site = None
        self.scheduled_rank = None

    def schedule_site(self, module_name, rank):
        if rank < 1:
            raise ValueError("scheduled rank must be positive")
        self.scheduled_site = str(module_name)
        self.scheduled_rank = int(rank)

    @staticmethod
    def _delete_updates(model):
        for ref in model.growing_blocks():
            ref.module.delete_update()
        if hasattr(model.core, "currently_updated_layer_index"):
            model.core.currently_updated_layer_index = None
        if hasattr(model.core, "layer_to_grow_index"):
            model.core.layer_to_grow_index = -1

    @staticmethod
    def _slice_extension(pair, rank):
        outgoing = copy.deepcopy(pair.first_layer.extended_output_layer)
        incoming = copy.deepcopy(pair.second_layer.extended_input_layer)
        if outgoing is None or incoming is None:
            raise RuntimeError("Gromo VGG did not produce both TINY extensions")
        outgoing.weight = nn.Parameter(outgoing.weight[:rank].detach().clone())
        if outgoing.bias is not None:
            outgoing.bias = nn.Parameter(outgoing.bias[:rank].detach().clone())
        incoming.weight = nn.Parameter(incoming.weight[:, :rank].detach().clone())
        return outgoing, incoming

    @staticmethod
    @contextmanager
    def _virtual(model, pair, outgoing, incoming, gate):
        old = (pair.first_layer.extended_output_layer,
               pair.second_layer.extended_input_layer,
               pair.first_layer.output_extension_scaling,
               pair.second_layer.input_extension_scaling,
               pair.second_layer.optimal_delta_scaling,
               model._extended_forward)
        outgoing = copy.deepcopy(outgoing)
        incoming = copy.deepcopy(incoming)
        pair.first_layer.extended_output_layer = outgoing
        pair.second_layer.extended_input_layer = incoming
        pair.first_layer.output_extension_scaling = torch.ones(
            1, device=outgoing.weight.device, dtype=outgoing.weight.dtype)
        scale = torch.as_tensor(
            gate, device=incoming.weight.device,
            dtype=incoming.weight.dtype).reshape(1)
        pair.second_layer.input_extension_scaling = scale
        pair.second_layer.optimal_delta_scaling = torch.zeros_like(scale)
        model._extended_forward = True
        try:
            yield
        finally:
            (pair.first_layer.extended_output_layer,
             pair.second_layer.extended_input_layer,
             pair.first_layer.output_extension_scaling,
             pair.second_layer.input_extension_scaling,
             pair.second_layer.optimal_delta_scaling,
             model._extended_forward) = old

    def propose_all(self, model, statistics_loader, _budget):
        from dual_growth.growth.candidate import GrowthCandidate
        from gromo.utils.training_utils import compute_statistics

        if not isinstance(model, GromoVGG16):
            raise TypeError("VggTinyAdapter requires GromoVGG16")
        refs = model.growing_blocks()
        if self.scheduled_site is not None:
            refs = [ref for ref in refs if ref.name == self.scheduled_site]
        if len(refs) != 1:
            raise ValueError(
                f"expected one scheduled VGG site, found {len(refs)}")
        ref = refs[0]
        pair = ref.module
        rank = min(int(pair.missing_neurons()), int(self.scheduled_rank or 0))
        if rank < 1:
            return []
        if pair.is_boundary:
            return [self._propose_pool_bridge(
                model, ref, statistics_loader, rank)]
        indices = [index for index, layer in enumerate(
            model.core._growable_layers) if layer is pair.second_layer]
        if len(indices) != 1:
            raise RuntimeError("VGG TINY site is not uniquely schedulable")
        self._delete_updates(model)
        model.core.set_growing_layers(
            scheduling_method="sequential", index=indices[0])
        compute_statistics(
            model.core, statistics_loader,
            loss_function=nn.CrossEntropyLoss(reduction="sum"),
            batch_limit=(self.max_statistics_batches or None),
            device=next(model.parameters()).device)
        try:
            model.core.compute_optimal_updates(
                compute_delta=True, use_covariance=True, alpha_zero=False,
                use_projection=True, maximum_added_neurons=rank,
                numerical_threshold=1e-6, statistical_threshold=1e-3)
        finally:
            model.core.reset_computation()
        values = pair.eigenvalues_extension
        if values is None or values.numel() < 1:
            self._delete_updates(model)
            return []
        rank = min(rank, int(values.numel()))
        outgoing, incoming = self._slice_extension(pair, rank)
        first = pair.first_layer.layer
        second = pair.second_layer.layer
        parameter_cost = rank * (
            int(first.in_channels) * first.kernel_size[0] * first.kernel_size[1] +
            int(second.out_channels) * second.kernel_size[0] * second.kernel_size[1])
        payload = {
            "requested_rank": int(self.scheduled_rank),
            "effective_rank": rank,
            "tiny_eigenvalues": values[:rank].detach().cpu().tolist(),
            "history": {"module": ref.name, "rank": rank,
                        "architecture_id": model.architecture_id},
        }
        candidate = GrowthCandidate(
            "expressive", ref.name, parameter_cost, parameter_cost, 0.0,
            float(values[:rank].square().sum().item()), payload,
            _virtual=lambda gate: self._virtual(
                model, pair, outgoing, incoming, gate))
        self._delete_updates(model)
        return [candidate]

    def _propose_pool_bridge(self, model, ref, statistics_loader, rank):
        """Build a closed-form, operator-aware candidate across actual MaxPool.

        Gromo's native pair solver assumes adjacent layers share a spatial grid.
        For a pool bridge, gather source activations and destination pre-activation
        gradients from the ordinary O forward/backward, pass source channels
        through the actual pool, rank channels by gradient correlation, then fit
        the destination extension with a damped least-squares solve. No auxiliary
        branch is trained and no global optimizer/RNG state is consumed.
        """
        from dual_growth.growth.candidate import GrowthCandidate

        pair = ref.module
        source, target = pair.first_layer, pair.second_layer
        device, dtype = source.weight.device, source.weight.dtype
        source_conv, target_conv = source.layer, target.layer
        if pair.bridge is None:
            raise RuntimeError("pool-bridge candidate is missing its MaxPool")
        batches = []
        limit = self.max_statistics_batches or 2
        modes = {module: module.training for module in model.modules()}
        gradients = {parameter: (None if parameter.grad is None
                                 else parameter.grad.detach().clone())
                     for parameter in model.parameters()}
        source_activations = []
        source_outputs = []
        source_inputs = []
        target_gradients = []
        capture = {}

        def capture_source(_module, _inputs, output):
            capture["source"] = output

        def capture_source_input(_module, inputs):
            capture["source_input"] = inputs[0]

        def capture_target(_module, _inputs, output):
            output.retain_grad()
            capture["target"] = output

        handles = [source.register_forward_hook(capture_source),
                   source.register_forward_pre_hook(capture_source_input),
                   target_conv.register_forward_hook(capture_target)]
        try:
            model.eval()
            for batch in statistics_loader:
                inputs, labels = (value.to(device, non_blocking=True)
                                  for value in batch[:2])
                model.zero_grad(set_to_none=True)
                capture.clear()
                logits = model(inputs)
                # Match native Gromo TINY's summed CE convention. Summed
                # sufficient statistics are invariant to how the same samples
                # are partitioned into minibatches.
                loss = F.cross_entropy(logits.float(), labels, reduction="sum")
                loss.backward()
                if ("source" not in capture or "source_input" not in capture or
                        "target" not in capture):
                    raise RuntimeError("could not capture VGG bridge statistics")
                target_gradient = capture["target"].grad
                if target_gradient is None:
                    raise RuntimeError("destination pre-activation gradient is missing")
                # Invoke the exact MaxPool module used by core.extended_forward.
                # No resize or substitute pooling operator is used.
                pooled = pair.bridge(capture["source"].detach())
                if tuple(pooled.shape[-2:]) != tuple(target_gradient.shape[-2:]):
                    raise RuntimeError(
                        "MaxPool output and destination gradient spatial sizes differ")
                source_activations.append(pooled.detach())
                source_outputs.append(capture["source"].detach())
                source_inputs.append(capture["source_input"].detach())
                target_gradients.append(target_gradient.detach())
                batches.append((inputs.shape[0],))
                if len(batches) >= limit:
                    break
            if not batches:
                raise RuntimeError("VGG bridge-aware statistics loader is empty")
        finally:
            for handle in handles:
                handle.remove()
            model.zero_grad(set_to_none=True)
            for parameter, gradient in gradients.items():
                parameter.grad = gradient
            for module, training in modes.items():
                module.training = training

        # Use each existing post-BN/ReLU feature as a deterministic auxiliary
        # source direction. GrowingBatchNorm passes extension channels through
        # unchanged, so fold its affine transform into the copied conv filters.
        source_norm = next((module for module in source.post_layer_function.modules()
                           if hasattr(module, "running_mean") and
                           hasattr(module, "running_var") and
                           hasattr(module, "eps")), None)
        channel_scores = torch.zeros(source_conv.out_channels, device=device,
                                     dtype=torch.float64)
        target_kernel = target_conv.kernel_size
        target_stride = target_conv.stride
        target_padding = target_conv.padding
        target_dilation = target_conv.dilation
        for pooled, target_gradient in zip(source_activations, target_gradients):
            gradient = target_gradient.flatten(start_dim=-2).to(torch.float64)
            for channel in range(source_conv.out_channels):
                patches = F.unfold(
                    pooled[:, channel:channel + 1], kernel_size=target_kernel,
                    dilation=target_dilation, padding=target_padding,
                    stride=target_stride).to(torch.float64)
                cross = torch.einsum("nkl,nol->ko", patches, gradient)
                channel_scores[channel] += cross.square().sum()
        if not torch.isfinite(channel_scores).all() or not bool(channel_scores.max() > 0):
            raise RuntimeError("VGG bridge statistics produced no finite signal")
        channel_indices = torch.argsort(channel_scores, descending=True)[:rank]
        scale = torch.ones(source_conv.out_channels, device=device, dtype=dtype)
        offset = torch.zeros_like(scale)
        if source_norm is not None:
            if source_norm.weight is not None:
                scale *= source_norm.weight.detach()
            if source_norm.running_var is not None:
                scale /= torch.sqrt(source_norm.running_var.detach() + source_norm.eps)
            if source_norm.running_mean is not None:
                offset -= source_norm.running_mean.detach()
            if source_conv.bias is not None:
                offset += source_conv.bias.detach()
            if source_norm.bias is not None:
                offset = offset * scale + source_norm.bias.detach()
            else:
                offset *= scale
        else:
            if source_conv.bias is not None:
                offset.copy_(source_conv.bias.detach())

        # Conv2d's default reset_parameters draws random values even though we
        # immediately replace them with copied source filters. Keep this
        # temporary initialization from advancing the training RNG streams.
        rng_devices = ([device.index if device.index is not None
                        else torch.cuda.current_device()]
                       if device.type == "cuda" else [])
        with torch.random.fork_rng(devices=rng_devices):
            outgoing = nn.Conv2d(
                source_conv.in_channels, rank, source_conv.kernel_size,
                stride=source_conv.stride, padding=source_conv.padding,
                dilation=source_conv.dilation,
                bias=source_conv.bias is not None, device=device, dtype=dtype)
            with torch.no_grad():
                outgoing.weight.copy_(
                    source_conv.weight.detach()[channel_indices] *
                    scale[channel_indices, None, None, None])
                if outgoing.bias is not None:
                    outgoing.bias.copy_(offset[channel_indices])

        # The features used by the least-squares fit must be exactly the
        # features deployed by GrowingModule's extension path (BN identity,
        # stateless ReLU, then the actual MaxPool). Keep this as a runtime
        # diagnostic so a future change to Gromo's post-layer semantics cannot
        # silently make the fitted X differ from deployed X.
        deployed_features = []
        reference_features = []
        with torch.no_grad():
            for source_input, source_output in zip(source_inputs, source_outputs):
                extension_pre_activation = outgoing(source_input)
                extension_activation = F.relu(extension_pre_activation)
                deployed_features.append(pair.bridge(extension_activation))
                reference_features.append(
                    pair.bridge(source_output.index_select(
                        1, channel_indices.to(source_output.device))))
        feature_error_sq = sum(
            (actual.double() - expected.double()).square().sum()
            for actual, expected in zip(deployed_features, reference_features))
        feature_reference_sq = sum(
            expected.double().square().sum() for expected in reference_features)
        feature_relative_error = float(
            torch.sqrt(feature_error_sq / feature_reference_sq.clamp_min(1e-30)).item())
        if not torch.isfinite(torch.tensor(feature_relative_error)):
            raise RuntimeError("non-finite deployed-vs-fit bridge feature error")
        if feature_relative_error > 1e-4:
            raise RuntimeError(
                "deployed boundary extension does not match least-squares features: "
                f"relative_error={feature_relative_error:.3e}")

        gram = None
        cross = None
        gradient_norm_squared = torch.zeros((), device=device, dtype=torch.float64)
        for pooled, target_gradient in zip(source_activations, target_gradients):
            features = pooled.index_select(1, channel_indices.to(pooled.device))
            unfolded = F.unfold(
                features, kernel_size=target_kernel, dilation=target_dilation,
                padding=target_padding, stride=target_stride).to(torch.float64)
            gradient = target_gradient.flatten(start_dim=-2).to(torch.float64)
            gradient_norm_squared += gradient.square().sum()
            gram_update = torch.einsum("nkl,nml->km", unfolded, unfolded)
            cross_update = torch.einsum("nkl,nol->ko", unfolded, gradient)
            gram = gram_update if gram is None else gram + gram_update
            cross = cross_update if cross is None else cross + cross_update
        assert gram is not None and cross is not None
        damping = max(float(torch.trace(gram).item()) / max(gram.shape[0], 1), 1.0) * 1e-6
        system = gram + damping * torch.eye(gram.shape[0],
                                             device=device, dtype=torch.float64)
        incoming_weight = -torch.linalg.solve(system, cross).T
        fit_inner = (incoming_weight * cross.T).sum()
        fit_quadratic = torch.einsum(
            "ok,km,om->", incoming_weight, gram, incoming_weight)
        fit_residual_before = gradient_norm_squared
        fit_residual_after = (gradient_norm_squared + 2.0 * fit_inner +
                              fit_quadratic).clamp_min(0.0)
        with torch.random.fork_rng(devices=rng_devices):
            incoming = nn.Conv2d(
                rank, target_conv.out_channels, target_kernel,
                stride=target_stride, padding=target_padding,
                dilation=target_dilation, bias=False, device=device, dtype=dtype)
            with torch.no_grad():
                incoming.weight.copy_(
                    incoming_weight.to(dtype).reshape_as(incoming.weight))
        candidate_outgoing = outgoing
        candidate_incoming = incoming
        if not all(torch.isfinite(parameter).all() for parameter in
                   (*candidate_outgoing.parameters(), *candidate_incoming.parameters())):
            raise RuntimeError("non-finite VGG MaxPool-bridge candidate")
        parameter_cost = rank * (
            source_conv.in_channels * source_conv.kernel_size[0] * source_conv.kernel_size[1]
            + target_conv.out_channels * target_conv.kernel_size[0] * target_conv.kernel_size[1])
        payload = {
            "source": "vgg_pool_bridge_closed_form_autograd",
            "effective_rank": rank,
            "requested_rank": int(self.scheduled_rank),
            "bridge": "actual_maxpool_forward_with_destination_gradient",
            "bridge_statistics_finite": True,
            "solver": "damped_pool_aware_least_squares",
            "source_feature_basis": "copied_existing_post_activation_channels",
            "novel_source_feature_direction": False,
            "statistics_batches": len(batches),
            "statistics_samples": sum(batch[0] for batch in batches),
            "selected_source_channels": channel_indices.detach().cpu().tolist(),
            "channel_scores": channel_scores.detach().cpu().tolist(),
            "statistics_gram_trace": float(torch.trace(gram).item()),
            "statistics_cross_norm": float(cross.norm().item()),
            "statistics_damping": damping,
            "deployed_fit_feature_relative_error": feature_relative_error,
            "least_squares_residual_before": float(fit_residual_before.item()),
            "least_squares_residual_after": float(fit_residual_after.item()),
            "history": {"module": ref.name, "rank": rank,
                        "architecture_id": model.architecture_id,
                        "operator_aware": True},
        }
        return GrowthCandidate(
            "expressive", ref.name, parameter_cost, parameter_cost, 0.0,
            float(cross.square().sum().item()), payload,
            _virtual=lambda gate: self._virtual(
                model, pair, candidate_outgoing, candidate_incoming, gate))
