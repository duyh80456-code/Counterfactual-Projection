from pathlib import Path
from types import SimpleNamespace

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


class ToyResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Module()
        self.core.stages = nn.Sequential(
            nn.Conv2d(3, 4, 1), nn.ReLU(),
            nn.Conv2d(4, 4, 1), nn.ReLU())

    def forward(self, inputs):
        return self.core.stages(inputs)


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
