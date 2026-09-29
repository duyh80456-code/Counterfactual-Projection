from pathlib import Path
from types import SimpleNamespace
import sys
import types

import torch
from torch import nn

from baselines.bypass import (
    activations, add_extension_parameters_, contraction_norm,
    embed_relaxed_bypass, extension_parameters, project_relaxed_bypass_,
    remove_extension_parameters_, transition_from_opt2_)
from experiments.shared_protocol import (
    BOOTSTRAP_EPOCH, FORK_EPOCH, POST_FORK_EPOCHS, TOTAL_EPOCHS,
    load_shared_checkpoint, rebase_scheduler_from_theta150, restore_rng,
    save_shared_checkpoint)
from experiments.run_shared_comparison import (
    functional_loss_utility, select_projectability_aware_candidate,
    select_structural_candidate, structural_candidate_sites,
    supervised_functional_descent_direction)


class ToyResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.stages = nn.Sequential(
            nn.Conv2d(3, 4, 1), nn.ReLU(),
            nn.Conv2d(4, 4, 1), nn.ReLU())

    def forward(self, inputs):
        return self.core.stages(inputs)


def test_structural_candidate_sites_scan_all_or_preserve_fixed_site():
    model = SimpleNamespace(growing_blocks=lambda: [
        SimpleNamespace(name="stages.0.blocks.0"),
        SimpleNamespace(name="stages.1.blocks.0"),
    ])

    assert structural_candidate_sites(model, "auto") == [
        "stages.0.blocks.0", "stages.1.blocks.0"]
    assert structural_candidate_sites(
        model, "auto", " stages.1.blocks.0, stages.0.blocks.0 ") == [
            "stages.1.blocks.0", "stages.0.blocks.0"]
    assert structural_candidate_sites(
        model, "stages.2.blocks.0") == ["stages.2.blocks.0"]


def test_structural_selector_reuses_statistics_and_selects_max_score(
        monkeypatch):
    adapters = types.ModuleType("dual_growth.adapters")
    controllers = types.ModuleType("dual_growth.controller")
    adapters.TinyAdapter = lambda **_kwargs: object()
    controllers.GrowthBudget = lambda _budget: object()
    monkeypatch.setitem(sys.modules, "dual_growth", types.ModuleType("dual_growth"))
    monkeypatch.setitem(sys.modules, "dual_growth.adapters", adapters)
    monkeypatch.setitem(sys.modules, "dual_growth.controller", controllers)
    seen_statistics = []
    scores = {"site.a": 0.25, "site.b": 0.75}

    class FakeProbe:
        def __init__(self, rank, module_name):
            assert rank == 4
            self.module_name = module_name

        def propose(self, _adapter, _model, statistics, _budget,
                    sample_inputs=None):
            seen_statistics.append(statistics)
            assert sample_inputs is statistics[0][0]
            return SimpleNamespace(
                module_name=self.module_name,
                proposal_score=scores[self.module_name])

    monkeypatch.setattr(
        "experiments.run_shared_comparison.CounterfactualTinyProbe", FakeProbe)
    model = SimpleNamespace(growing_blocks=lambda: [
        SimpleNamespace(name="site.a"), SimpleNamespace(name="site.b")])
    statistics = [(torch.randn(2, 3), torch.tensor([0, 1]))]

    selected, diagnostics = select_structural_candidate(
        model, statistics, rank=4, site="auto", candidate_sites="",
        device=torch.device("cpu"))

    assert selected.module_name == "site.b"
    assert [id(value) for value in seen_statistics] == [
        id(statistics), id(statistics)]
    assert diagnostics["site_selection_mode"] == "tiny_score_argmax"
    assert diagnostics["selected_site"] == "site.b"
    assert diagnostics["site_scores"] == scores
    assert diagnostics["selected_site_score"] == scores["site.b"]
    assert diagnostics["site_selection_seconds"] >= 0


def test_projectability_selector_uses_top_k_projected_utility_and_when_gate(
        monkeypatch):
    candidates = [SimpleNamespace(module_name=name, proposal_score=score)
                  for name, score in (
                      ("site.a", 0.9), ("site.b", 0.8),
                      ("site.c", 0.7), ("site.d", 0.1))]
    monkeypatch.setattr(
        "experiments.run_shared_comparison.propose_structural_candidates",
        lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(
        "experiments.run_shared_comparison.candidate_projection_block",
        lambda _model, candidate: candidate.module_name)
    monkeypatch.setattr(
        "experiments.run_shared_comparison.candidate_projection_parameter_names",
        lambda *_args, **_kwargs: ("weight",))

    class ZeroLogits(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, inputs):
            return torch.zeros(inputs.shape[0], 2) * self.weight

    model = ZeroLogits()
    selection_batch = (torch.randn(2, 3), torch.tensor([0, 1]))
    descent = supervised_functional_descent_direction(model, selection_batch)
    expansion_directions = {
        "site.a": 2.0 * descent,
        "site.b": 1.0 * descent,
        "site.c": -1.0 * descent,
    }
    fitted_directions = {
        "site.a": 0.2 * descent,
        "site.b": 0.8 * descent,
        "site.c": -0.5 * descent,
    }

    class FakeExpansionProbe:
        def __call__(self, _model, *, candidate, batch, gate):
            assert batch is selection_batch and gate == 0.05
            return SimpleNamespace(
                delta_logits=expansion_directions[candidate.module_name])

    class FakeCheapProjector:
        def project(self, _model, _inputs, target, *, block,
                    parameter_names):
            assert parameter_names == ("weight",)
            fitted = fitted_directions[block]
            rho = float(fitted.norm() / target.norm())
            return SimpleNamespace(
                fitted_delta=fitted, fitted_norm_ratio=rho,
                relative_residual=0.1, cosine_alignment=1.0,
                cg=SimpleNamespace(iterations=25, converged=False))

    monkeypatch.setattr(
        "experiments.run_shared_comparison.CandidateExpansionProbe",
        FakeExpansionProbe)
    selected, diagnostics = select_projectability_aware_candidate(
        model, [(torch.randn(1, 3), torch.tensor([0]))], selection_batch,
        rank=4, site="auto", candidate_sites="", device=torch.device("cpu"),
        gate=0.05, top_k=3, cheap_projector=FakeCheapProjector(),
        min_utility=0.0, min_projectability=0.5)

    assert selected.module_name == "site.b"
    assert diagnostics["prescreen_top_k_sites"] == [
        "site.a", "site.b", "site.c"]
    assert "site.d" not in diagnostics["site_functional_evaluations"]
    assert diagnostics["selected_projected_utility"] == \
        functional_loss_utility(descent, fitted_directions["site.b"])
    assert diagnostics["selected_projectability_rho"] >= 0.5
    assert diagnostics["when_gate_passed"] is True
    assert diagnostics["when_gate_reason"] == "usable_candidate_found"

    _selected, rejected = select_projectability_aware_candidate(
        model, [(torch.randn(1, 3), torch.tensor([0]))], selection_batch,
        rank=4, site="auto", candidate_sites="", device=torch.device("cpu"),
        gate=0.05, top_k=3, cheap_projector=FakeCheapProjector(),
        min_utility=0.0, min_projectability=0.9)
    assert rejected["when_gate_passed"] is False
    assert rejected["when_gate_reason"] == "projectability_below_threshold"


def test_projection_only_target_is_negative_summed_ce_logit_gradient():
    model = nn.Linear(3, 2, bias=False)
    inputs = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0]])
    targets = torch.tensor([0, 1])
    with torch.no_grad():
        logits = model(inputs)
        expected = torch.nn.functional.one_hot(
            targets, num_classes=2).to(logits) - logits.softmax(dim=1)

    direction = supervised_functional_descent_direction(
        model, (inputs, targets))

    assert torch.allclose(direction, expected)
    assert torch.allclose(direction.sum(dim=1), torch.zeros(2), atol=1e-7)


def test_relaxed_bypass_embed_and_projection_are_function_preserving():
    torch.manual_seed(7)
    model = ToyResidualModel()
    inputs = torch.randn(2, 3, 5, 5)
    expected = model(inputs).detach()
    base_parameters = sum(parameter.numel() for parameter in model.parameters())

    paths = embed_relaxed_bypass(model)
    assert len(paths) == 2
    assert len(activations(model)) == 2
    assert torch.equal(model(inputs), expected)
    assert float(contraction_norm(model)) == 0.0

    extensions = extension_parameters(model)
    extension_ids = {id(parameter) for parameter in extensions}
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters()
         if id(parameter) not in extension_ids], lr=0.1)
    add_extension_parameters_(optimizer, extensions)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == 6
    remove_extension_parameters_(optimizer, extensions)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == 4

    assert project_relaxed_bypass_(model) == 2
    assert torch.equal(model(inputs), expected)
    assert sum(parameter.numel() for parameter in model.parameters()) == base_parameters


def test_shared_checkpoint_contains_exact_fork_state_and_hash(tmp_path):
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCHS)
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(17))
    protocol = {"fork_epoch": FORK_EPOCH, "post_fork_epochs": POST_FORK_EPOCHS}
    path = tmp_path / "shared_seed1_epoch300.pt"
    digest = save_shared_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=FORK_EPOCH, train_indices=[3, 5], validation_indices=[7],
        tuning_indices=[11], loader=loader, history=[{"epoch": FORK_EPOCH}],
        run_protocol=protocol)

    restored_model = nn.Linear(3, 2)
    restored_optimizer = torch.optim.SGD(
        restored_model.parameters(), lr=0.1, momentum=0.9)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=TOTAL_EPOCHS)
    checkpoint, actual = load_shared_checkpoint(
        path, digest, device="cpu", model=restored_model,
        optimizer=restored_optimizer, scheduler=restored_scheduler)
    assert actual == digest
    assert checkpoint["epoch"] == FORK_EPOCH
    assert checkpoint["train_indices"] == [3, 5]
    assert checkpoint["validation_indices"] == [7]
    assert checkpoint["tuning_indices"] == [11]
    assert "rng" in checkpoint and "train_loader_generator_state" in checkpoint
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[name])


def bypass_toy_with_optimizer():
    model = ToyResidualModel()
    embed_relaxed_bypass(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optimizer


def test_bypass_soft_cap_does_not_project_uncontracted_extension():
    model, optimizer = bypass_toy_with_optimizer()
    extensions = extension_parameters(model)
    for module in activations(model):
        module.d.data.fill_(0.1)

    transition = transition_from_opt2_(
        model, optimizer, epsilon=0.002, opt2_done=10, soft_cap=10)

    assert transition.phase == "opt2"
    assert not transition.criterion_met
    assert transition.soft_cap_exceeded
    assert transition.projected_count == 0
    assert len(activations(model)) == 2
    optimizer_ids = {id(parameter) for group in optimizer.param_groups
                     for parameter in group["params"]}
    assert all(id(parameter) in optimizer_ids for parameter in extensions)


def test_bypass_projects_contracted_extension_and_enters_train3():
    model, optimizer = bypass_toy_with_optimizer()
    extensions = extension_parameters(model)
    for module in activations(model):
        module.d.data.fill_(1e-5)

    transition = transition_from_opt2_(
        model, optimizer, epsilon=0.002, opt2_done=3, soft_cap=10)

    assert transition.phase == "train3"
    assert transition.criterion_met
    assert not transition.soft_cap_exceeded
    assert transition.projected_count == 2
    assert activations(model) == []
    optimizer_ids = {id(parameter) for group in optimizer.param_groups
                     for parameter in group["params"]}
    assert all(id(parameter) not in optimizer_ids for parameter in extensions)


def test_bypass_result_uses_explicit_completed_field():
    source = Path("baselines/run_bypass.py").read_text()
    assert '"bypass_completed": phase == "train3"' in source


def test_adaptive_result_preserves_per_epoch_site_selection_history():
    source = Path("experiments/run_shared_comparison.py").read_text()
    assert '"site_selection_history": [' in source
    for field in (
            "selected_site", "selected_site_score", "site_scores",
            "site_selection_seconds"):
        assert f'"{field}"' in source


def test_restore_rng_moves_mapped_cuda_states_back_to_cpu(monkeypatch):
    class MappedCudaState:
        def detach(self):
            return self

        def cpu(self):
            return torch.tensor([1, 2, 3], dtype=torch.uint8)

    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all",
        lambda values: restored.extend(values))
    state = {
        "python": __import__("random").getstate(),
        "torch": torch.get_rng_state(),
        "cuda": [MappedCudaState()],
    }

    restore_rng(state)

    assert len(restored) == 1
    assert restored[0].device.type == "cpu"
    assert restored[0].dtype == torch.uint8


def test_theta150_scheduler_continuation_preserves_lr_to_epoch350():
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0146)
    scheduler = rebase_scheduler_from_theta150(optimizer)
    assert scheduler.get_last_lr() == [0.0146]
    assert scheduler.T_max == TOTAL_EPOCHS - BOOTSTRAP_EPOCH
