from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from methods import EProjection
from baselines import ExpandedTrainProject, RealEOracle
from probe import (
    CandidateExpansionProbe, CounterfactualTinyProbe,
    TransactionalCandidateSource)
from projection import (
    FunctionalProjector, StructuralAuxiliarySpace,
    StructuralExpansionTransfer)


class ExpandableBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 3, 1)
        self.bn2 = nn.BatchNorm2d(3)
        self.downsample = nn.Conv2d(1, 3, 1)
        self.extension_in = None
        self.extension_out = None
        self.extension_gate = None

    def forward(self, inputs):
        output = (self.bn2(self.conv2(torch.relu(self.bn1(self.conv1(inputs))))) +
                  self.downsample(inputs))
        if self.extension_in is not None:
            output = output + self.extension_gate * self.extension_out(
                torch.relu(self.extension_in(inputs)))
        return output


class ExpandableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ExpandableBlock()

    def forward(self, inputs):
        return self.block(inputs).mean((2, 3))


class StructuralCandidate:
    module_name = "block"
    payload = {"tiny_eigenvalues": [2.0]}

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _layers():
        first = nn.Conv2d(1, 1, 3, padding=1)
        second = nn.Conv2d(1, 3, 1, bias=False)
        with torch.no_grad():
            first.weight.fill_(0.1)
            first.bias.fill_(0.2)
            second.weight.fill_(0.3)
        return first, second

    @contextmanager
    def virtual_direction(self, gate):
        assert self.model.block.extension_in is None
        first, second = self._layers()
        self.model.block.extension_in = first
        self.model.block.extension_out = second
        self.model.block.extension_gate = gate
        try:
            yield
        finally:
            self.model.block.extension_in = None
            self.model.block.extension_out = None
            self.model.block.extension_gate = None

    def commit(self):
        first, second = self._layers()
        self.model.block.extension_in = first
        self.model.block.extension_out = second
        self.model.block.extension_gate = 1.0
        return self.model.block


def test_candidate_probe_uses_transaction_without_factor_payload():
    torch.manual_seed(12)
    model = ExpandableModel().train()
    candidate = StructuralCandidate(model)
    batch = (torch.randn(4, 1, 6, 6), torch.tensor([0, 1, 2, 1]))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    signal = CandidateExpansionProbe()(
        model, candidate=candidate, batch=batch, gate=0.2)
    smaller_gate = CandidateExpansionProbe()(
        model, candidate=candidate, batch=batch, gate=0.1)
    assert signal.is_structural_expansion
    assert signal.source == "tiny_gromo_structural"
    assert signal.A_E is None and signal.B_E is None
    assert signal.delta_logits.norm() > 0
    assert signal.probe_gate == 0.2
    assert signal.observed_loss_gain is not None
    assert torch.allclose(signal.delta_logits, smaller_gate.delta_logits,
                          atol=1e-5, rtol=1e-5)
    assert model.training
    assert all(torch.equal(value, before[name])
               for name, value in model.state_dict().items())


def test_candidate_statistics_are_transactional():
    torch.manual_seed(121)
    model = ExpandableModel().train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    candidate = StructuralCandidate(model)

    class MutatingAdapter:
        def propose_all(self, target, _loader, _budget):
            target.eval()
            with torch.no_grad():
                target.block.conv1.weight.add_(10)
            return [candidate]

    result = TransactionalCandidateSource().propose(
        MutatingAdapter(), model, [], object())
    assert result == [candidate]
    assert model.training
    assert all(torch.equal(value, before[name])
               for name, value in model.state_dict().items())


def test_counterfactual_probe_overrides_full_width_ceiling_temporarily():
    class Second:
        in_neurons = 256
        target_in_neurons = 256

    class Block:
        second_layer = Second()

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.site = Block()

        def block(self, name):
            assert name == "layer3.1"
            return self.site

    model = FullModel()
    candidate = type("Candidate", (), {
        "payload": {"effective_rank": 4},
    })()

    class Adapter:
        def schedule_site(self, name, rank):
            assert (name, rank) == ("layer3.1", 4)

        def propose_all(self, target, _loader, _budget):
            assert target.site.second_layer.target_in_neurons == 260
            return [candidate]

    result = CounterfactualTinyProbe(4, "layer3.1").propose(
        Adapter(), model, [], object())
    assert result is candidate
    assert model.site.second_layer.target_in_neurons == 256
    assert candidate.payload["base_hidden_width"] == 256
    assert candidate.payload["counterfactual_target_width"] == 260


def test_counterfactual_probe_replaces_legacy_flops_with_runtime_spatial_flops():
    class Layer(nn.Module):
        def __init__(self, convolution, width=None):
            super().__init__()
            self.layer = convolution
            if width is not None:
                self.in_neurons = width
                self.target_in_neurons = width

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.first_layer = Layer(nn.Conv2d(3, 4, 3, padding=1))
            self.second_layer = Layer(nn.Conv2d(4, 5, 3, padding=1), 4)

        def forward(self, inputs):
            return self.second_layer.layer(torch.relu(
                self.first_layer.layer(inputs)))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.site = Block()

        def forward(self, inputs):
            return self.site(inputs).mean((2, 3))

        def block(self, _name):
            return self.site

    candidate = type("Candidate", (), {
        "payload": {"effective_rank": 2}, "extra_flops": 1.0})()

    class Adapter:
        def schedule_site(self, _name, _rank):
            pass

        def propose_all(self, _model, _loader, _budget):
            return [candidate]

    inputs = torch.randn(2, 3, 8, 8)
    model = Model()
    result = CounterfactualTinyProbe(2, "site").propose(
        Adapter(), model, [(inputs, torch.tensor([0, 1]))], object(),
        sample_inputs=inputs)
    expected = 2 * 2 * (8 * 8 * 3 * 3 * 3 + 8 * 8 * 5 * 3 * 3)
    assert result.extra_flops == expected
    assert result.payload["actual_extension_flops"] == expected


def test_structural_projection_and_auxiliary_transfer_are_concrete():
    torch.manual_seed(13)
    model = ExpandableModel().eval()
    candidate = StructuralCandidate(model)
    batch = (torch.randn(3, 1, 6, 6), torch.tensor([0, 1, 2]))
    projector = FunctionalProjector(damping=1e-3, max_iter=20)
    step = EProjection(projector=projector).discover_candidate(
        model, candidate, batch, gate=0.1)
    assert step.signal.is_structural_expansion
    assert all("downsample" not in name
               for name in step.projection.parameter_delta)
    assert {name.rsplit(".", 1)[0] for name in step.projection.parameter_delta} == {
        "block.conv1", "block.bn1", "block.conv2", "block.bn2"}
    conv_only = EProjection(projector=projector).discover_candidate(
        model, candidate, batch, gate=0.1, projection_scope="conv_only")
    assert {name.rsplit(".", 1)[0]
            for name in conv_only.projection.parameter_delta} == {
                "block.conv1", "block.conv2"}
    whole_block = EProjection(projector=projector).discover_candidate(
        model, candidate, batch, gate=0.1, projection_scope="whole_block")
    assert any("downsample" in name
               for name in whole_block.projection.parameter_delta)
    assert -1.0 <= step.projection.cosine_alignment <= 1.0
    transfer = StructuralExpansionTransfer(
        model, batch[0], step.signal, "block", projector)
    result = transfer.to_model(0.5)
    assert result.target_delta.shape == step.signal.delta_logits.shape
    output_gradient = torch.ones_like(step.signal.delta_logits)
    auxiliary = StructuralAuxiliarySpace(transfer, curvature=1.0)
    correction = auxiliary.correction(output_gradient)
    assert correction.parameter_delta


def test_structural_controls_train_temporarily_or_commit_real_e():
    torch.manual_seed(14)
    batch = (torch.randn(3, 1, 6, 6), torch.tensor([0, 1, 2]))
    model = ExpandableModel().eval()
    base_optimizer = torch.optim.SGD(
        model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    model.train()
    base_optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(batch[0]), batch[1]).backward()
    base_optimizer.step()
    model.eval()
    optimizer_state_before = {
        parameter: {key: (value.clone() if torch.is_tensor(value) else value)
                    for key, value in state.items()}
        for parameter, state in base_optimizer.state.items()}
    candidate = StructuralCandidate(model)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    heldout_batch = (torch.randn(4, 1, 6, 6), torch.tensor([2, 1, 0, 2]))
    untrained_heldout = CandidateExpansionProbe()(
        model, candidate=candidate, batch=heldout_batch, gate=1.0)
    control = ExpandedTrainProject(
        steps=2, learning_rate=0.05, momentum=0.9, weight_decay=5e-4,
        projector=FunctionalProjector(damping=1e-3, max_iter=10))
    train_mode_observations = []
    hook = model.block.bn1.register_forward_hook(
        lambda module, _inputs, _output: train_mode_observations.append(
            (module.training, torch.is_grad_enabled())))
    result = control.discover(
        model, candidate, batch, heldout_batch=heldout_batch,
        base_optimizer=base_optimizer)
    hook.remove()
    assert result.signal.source == "expanded_model_train_then_contract"
    assert len(result.expansion_train_losses) == 2
    assert result.heldout_delta_logits is not None
    assert result.heldout_loss_gain is not None
    assert result.temporary_base_parameter_update_norm > 0
    assert result.inherited_optimizer_states == len(base_optimizer.state)
    assert any(training and gradients
               for training, gradients in train_mode_observations)
    assert not torch.allclose(
        result.heldout_delta_logits, untrained_heldout.delta_logits)
    assert all(torch.equal(value, before[name])
               for name, value in model.state_dict().items())
    assert all(
        torch.equal(value, base_optimizer.state[parameter][key])
        if torch.is_tensor(value)
        else value == base_optimizer.state[parameter][key]
        for parameter, state in optimizer_state_before.items()
        for key, value in state.items())

    oracle_model = ExpandableModel()
    oracle = RealEOracle.commit_(
        oracle_model, StructuralCandidate(oracle_model))
    assert oracle.deploy_parameter_delta > 0


def test_oracle_commit_synchronizes_target_to_grown_current_width():
    class Second:
        in_neurons = 260
        target_in_neurons = 256

    class Committed:
        second_layer = Second()

    class Candidate:
        @staticmethod
        def commit():
            return Committed()

    model = nn.Linear(2, 2)
    result = RealEOracle.commit_(model, Candidate())
    assert result.committed_module.second_layer.target_in_neurons == 260
