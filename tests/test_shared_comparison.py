from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from baselines.bypass import (
    activations, add_extension_parameters_, contraction_norm,
    embed_relaxed_bypass, extension_parameters, project_relaxed_bypass_,
    remove_extension_parameters_)
from experiments.shared_protocol import (
    FORK_EPOCH, POST_FORK_EPOCHS, TOTAL_EPOCHS, load_shared_checkpoint,
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
    path = tmp_path / "shared_seed1_epoch20.pt"
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
