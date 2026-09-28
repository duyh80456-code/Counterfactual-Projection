from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import TensorDataset

from experiments.run_gromo_pilot import (
    intervention_batches, rebuild_sgd_after_growth)


def test_intervention_batches_are_fresh_disjoint_training_examples():
    dataset = TensorDataset(
        torch.arange(30, dtype=torch.float32).reshape(30, 1),
        torch.arange(30))
    train_indices = list(range(3, 27))
    args = SimpleNamespace(
        seed=4, statistics_samples=6, projection_samples=3,
        batch_size=2, workers=0)
    first_stats, first_projection, first_audit = intervention_batches(
        dataset, train_indices, args, 0, torch.device("cpu"))
    _, _, second_audit = intervention_batches(
        dataset, train_indices, args, 1, torch.device("cpu"))
    statistics_targets = torch.cat([targets for _, targets in first_stats])
    projection_targets = first_projection[1]
    assert set(statistics_targets.tolist()).issubset(train_indices)
    assert set(projection_targets.tolist()).issubset(train_indices)
    assert not set(statistics_targets.tolist()) & set(projection_targets.tolist())
    assert first_audit["statistics_projection_overlap"] == 0
    assert (first_audit["projection_indices_sha256"] !=
            second_audit["projection_indices_sha256"])


def test_oracle_growth_migrates_existing_sgd_momentum_prefix():
    class GrowingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 2)
            self.migrations = []

        def forward(self, inputs):
            return self.linear(inputs)

        def consume_parameter_migrations(self):
            migrations, self.migrations = self.migrations, []
            return migrations

    model = GrowingModel()
    args = SimpleNamespace(lr=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    old_weight = model.linear.weight
    old_momentum = optimizer.state[old_weight]["momentum_buffer"].clone()
    new_weight = nn.Parameter(torch.cat([
        old_weight.detach(), torch.zeros(1, old_weight.shape[1])]))
    model.linear.weight = new_weight
    model.migrations = [(old_weight, new_weight)]

    replacement, migrations = rebuild_sgd_after_growth(model, optimizer, args)
    migrated = replacement.state[new_weight]["momentum_buffer"]
    assert migrations == 1
    assert torch.equal(migrated[:old_weight.shape[0]], old_momentum)
    assert torch.count_nonzero(migrated[old_weight.shape[0]:]) == 0
    assert model.linear.bias in replacement.state
