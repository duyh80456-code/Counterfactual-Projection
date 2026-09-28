from contextlib import contextmanager

import torch
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
        self.conv2 = nn.Conv2d(2, 3, 1)
        self.extension_in = None
        self.extension_out = None
        self.extension_gate = None

    def forward(self, inputs):
        output = self.conv2(torch.relu(self.conv1(inputs)))
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
    candidate = StructuralCandidate(model)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    control = ExpandedTrainProject(
        steps=2, learning_rate=0.05,
        projector=FunctionalProjector(damping=1e-3, max_iter=10))
    result = control.discover(model, candidate, batch)
    assert result.signal.source == "expanded_train_then_contract"
    assert len(result.expansion_train_losses) == 2
    assert all(torch.equal(value, before[name])
               for name, value in model.state_dict().items())

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
