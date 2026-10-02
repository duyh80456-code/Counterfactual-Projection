"""Counterfactual TINY transactions for Gromo's native VGG container."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VGGPairRef:
    stage: int
    link: int
    name: str
    module: object


class VGGConvPair:
    """Two consecutive native Gromo convolutions sharing a hidden channel."""

    def __init__(self, first_layer, second_layer):
        self.first_layer = first_layer
        self.second_layer = second_layer

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
    """CIFAR VGG16-BN exposing its eight internal conv links to TINY."""

    architecture_id = "cifar_gromo_vgg16_bn_full_random_v1"

    def __init__(self, num_classes=100, device="cpu", expansion_capacity=64):
        super().__init__()
        from gromo.containers.vgg import VGG

        widths = ((64, 64), (128, 128), (256, 256, 256),
                  (512, 512, 512), (512, 512, 512))
        cfg = []
        target_cfg = []
        for stage in widths:
            cfg.extend(stage)
            cfg.append("M")
            target_cfg.extend(
                width + int(expansion_capacity)
                if index < len(stage) - 1 else width
                for index, width in enumerate(stage))
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
