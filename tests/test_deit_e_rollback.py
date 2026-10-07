"""Accuracy stall and lower-loss anchor updates must survive rollback/resume."""
import copy
from dataclasses import replace
import json
import random
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.deit_e_rollback import EAccuracyRollback, rollback_protocol
from experiments.deit_protocol import DeitRecipe, build_optimizer_scheduler, protocol, save_state
from experiments.shared_protocol import sha256_file, restore_rng


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


@pytest.fixture
def tiny_control():
    model = torch.nn.Linear(2, 2).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 / (epoch + 1))
    inputs = torch.tensor([[1., 2.], [-1., .5]], dtype=torch.float64)
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2,
                        generator=torch.Generator().manual_seed(3))
    def train():
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(inputs), labels).backward()
        optimizer.step(); scheduler.step()
    train()
    controller = EAccuracyRollback(model, optimizer, scheduler, loader,
                                  {"accuracy": .5, "loss": 2.}, 0, 10)
    return model, optimizer, scheduler, loader, controller, train


@pytest.mark.parametrize("accuracy,loss,improved,reason,counter", [
    (.6, 3., True, "higher_accuracy", 0),
    (.5, 1.9, True, "same_accuracy_lower_loss", 5),
    (.5, 2.1, False, "no_improvement", 5),
    (.4, 1., False, "no_improvement", 5)])
def test_anchor_order_and_accuracy_counter_update_once(tiny_control, accuracy, loss, improved, reason, counter):
    model, optimizer, scheduler, loader, controller, train = tiny_control
    initial = copy.deepcopy(controller.anchor)
    controller.accuracy_stall_counter = 4
    train()
    result = controller.observe(model, optimizer, scheduler, loader, {"accuracy": accuracy, "loss": loss}, 1, 1)
    assert result["controller_anchor_improved"] is improved
    assert result["controller_anchor_reason"] == reason
    assert result["accuracy_stall_counter"] == counter
    assert result["rollback_applied"] is False
    if improved:
        assert controller.anchor["epoch"] == 1
        assert_equal(controller.anchor["model"], model.state_dict())
        assert_equal(controller.anchor["optimizer"], optimizer.state_dict())
    else:
        assert_equal(controller.anchor, initial)


def test_rollback_uses_new_lower_loss_anchor_and_preserves_current_stream(tiny_control):
    model, optimizer, scheduler, loader, controller, train = tiny_control
    train()
    controller.observe(model, optimizer, scheduler, loader, {"accuracy": .6, "loss": 1.8}, 1, 1)
    train()
    tied = controller.observe(model, optimizer, scheduler, loader, {"accuracy": .6, "loss": 1.7}, 2, 2)
    assert tied["controller_anchor_improved"] and tied["accuracy_stall_counter"] == 1
    anchor = copy.deepcopy(controller.anchor)
    identities = dict(model.named_parameters())
    for epoch in range(3, 12):
        train()
        torch.rand(3); random.random()
        torch.rand(3, generator=loader.generator)
        rng, loader_rng = torch.get_rng_state().clone(), loader.generator.get_state().clone()
        python_rng = random.getstate()
        result = controller.observe(model, optimizer, scheduler, loader,
                                    {"accuracy": .55, "loss": 1.9}, epoch, epoch)
        assert result["rollback_applied"] is (epoch == 11)
    assert result["accuracy_stall_before_rollback"] == 10
    assert result["accuracy_stall_counter"] == 0 and result["rollback_anchor_epoch"] == 2
    assert_equal(model.state_dict(), anchor["model"])
    assert_equal(optimizer.state_dict(), anchor["optimizer"])
    assert_equal(scheduler.state_dict(), anchor["scheduler"])
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng
    assert torch.equal(loader.generator.get_state(), loader_rng)
    assert all(value is identities[name] for name, value in model.named_parameters())
    assert len(controller.rollback_events) == 1


def test_controller_resume_rejects_changed_patience(tiny_control):
    controller = tiny_control[4]
    with pytest.raises(ValueError):
        EAccuracyRollback.from_state(controller.state_dict(), 9, 0)
    assert rollback_protocol()["algorithm_patience"] == 10
    assert rollback_protocol()["retrigger"] is False


def test_lower_loss_tie_at_tenth_stall_rolls_back_to_that_new_anchor(tiny_control):
    model, optimizer, scheduler, loader, controller, train = tiny_control
    controller.accuracy_stall_counter = 9
    train()
    before = copy.deepcopy(model.state_dict())
    result = controller.observe(model, optimizer, scheduler, loader, {"accuracy": .5, "loss": 1.9}, 10, 10)
    assert result["controller_anchor_improved"] and result["rollback_applied"]
    assert result["controller_anchor_epoch"] == result["rollback_anchor_epoch"] == 10
    assert result["accuracy_stall_before_rollback"] == 10 and result["accuracy_stall_counter"] == 0
    assert_equal(model.state_dict(), before)


def test_strict_accuracy_improvement_at_tenth_epoch_prevents_rollback(tiny_control):
    model, optimizer, scheduler, loader, controller, train = tiny_control
    controller.accuracy_stall_counter = 9
    train()
    result = controller.observe(model, optimizer, scheduler, loader, {"accuracy": .6, "loss": 3.}, 10, 10)
    assert result["controller_anchor_improved"] and result["controller_accuracy_improved"]
    assert result["rollback_applied"] is False and result["accuracy_stall_counter"] == 0


def test_main_loop_rollback_and_mid_stall_resume_without_requery(deit_small, deit_batches, tmp_path, monkeypatch):
    import experiments.run_deit_fork as runner
    from models import DeiTTinyCifar
    dataset = TensorDataset(*deit_batches[0])
    recipe = replace(DeitRecipe(), workers=0, batch_size=8)
    optimizer, scheduler = build_optimizer_scheduler(deit_small, recipe)
    torch.nn.functional.cross_entropy(deit_small(deit_batches[0][0]), deit_batches[0][1]).backward()
    optimizer.step(); optimizer.zero_grad(); scheduler.step()
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(1))
    fork_path = tmp_path / "fork.pt"
    save_state(fork_path, model=deit_small, optimizer=optimizer, scheduler=scheduler, loader=loader,
        epoch=23, history=[{"epoch": 23, "validation_accuracy": .528, "validation_loss": 2.}],
        train_indices=list(range(8)), evaluation_indices=[0, 1], trigger_indices=[], source_tuning_indices=[],
        run_protocol=protocol(recipe, deit_small), kind="deit_plateau_fork",
        historical_best_accuracy=.528, historical_best_loss=2., historical_best_epoch=23)
    monkeypatch.setattr(runner, "checked_source", lambda path, _kinds:
                        (torch.load(path, weights_only=False), recipe))
    def context(_root, _recipe, device, source):
        model = DeiTTinyCifar(**deit_small.config).double().to(device)
        opt, sched = build_optimizer_scheduler(model, recipe)
        model.load_state_dict(source["model"]); opt.load_state_dict(source["optimizer"])
        sched.load_state_dict(source["scheduler"])
        gen = torch.Generator().set_state(source["train_loader_generator_state"])
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen)
        restore_rng(source["rng"])
        return model, opt, sched, train_loader, DataLoader(dataset, batch_size=8), dataset, list(range(8)), [0, 1], [], []
    monkeypatch.setattr(runner, "load_training_context", context)
    queries = []
    def jump(model, _optimizer, **_kwargs):
        queries.append(True)
        with torch.no_grad():
            model.blocks[0].mlp.fc1.bias.add_(.01)
        return {"correction_applied": True, "selected_site": "blocks.0.mlp"}
    monkeypatch.setattr(runner, "one_shot_intervention", jump)
    monkeypatch.setattr(runner, "materialize_probe_batches", lambda *_args: ({}, {}))
    post = [{"accuracy": .55, "loss": 1.8}, {"accuracy": .55, "loss": 1.7}]
    post += [{"accuracy": .54, "loss": 1.9}] * 9
    values = iter([{"accuracy": .528, "loss": 2.}, {"accuracy": .53, "loss": 1.9}] + post)
    monkeypatch.setattr(runner, "evaluate_without_rng", lambda *_args: next(values))
    midpoint = tmp_path / "mid_stall.pt"
    at_rollback = tmp_path / "after_rollback.pt"
    original_save = runner.save_state
    def capture(path, **kwargs):
        original_save(path, **kwargs)
        if kwargs["completed_epochs"] == 7:
            torch.save(torch.load(path, weights_only=False), midpoint)
        if kwargs["completed_epochs"] == 11:
            torch.save(torch.load(path, weights_only=False), at_rollback)
    monkeypatch.setattr(runner, "save_state", capture)
    output = tmp_path / "E"
    args = ["run_deit_fork", "--data-root", "unused", "--plateau-checkpoint", str(fork_path),
            "--plateau-checkpoint-hash", sha256_file(fork_path), "--method", "ours_e_driven_o",
            "--output", str(output), "--post-fork-epochs", "11", "--device", "cpu"]
    monkeypatch.setattr(sys, "argv", args)
    runner.main()
    uninterrupted = torch.load(output / "checkpoint_latest.pt", weights_only=False)
    result = json.loads((output / "result.json").read_text())
    assert len(queries) == 1 and len(result["interventions"]) == 1
    assert result["rollback_count"] == 1 and result["controller_anchor_epoch"] == 25
    assert result["report_best_epoch"] == 24 and result["report_best_loss"] == 1.8
    assert result["controller_anchor_loss"] == 1.7
    assert result["final_validation_accuracy"] == .55  # Actual final restored state.
    assert result["last_observed_validation_accuracy"] == .54
    assert result["final_model_state_epoch"] == 25
    assert result["history"][-1]["accuracy_stall_before_rollback"] == 10
    best = torch.load(output / "checkpoint_best.pt", weights_only=False)
    for key in ("model", "optimizer", "scheduler"):
        assert_equal(uninterrupted[key], best[key])
    assert torch.load(midpoint, weights_only=False)["accuracy_stall_counter"] == 6
    # Resume the embedded anchor with no separate checkpoint_best file.
    (output / "checkpoint_best.pt").unlink()
    values = iter(post[7:])
    monkeypatch.setattr(sys, "argv", args + ["--resume", str(midpoint)])
    runner.main()
    resumed = torch.load(output / "checkpoint_latest.pt", weights_only=False)
    assert len(queries) == 1
    for key in ("model", "optimizer", "scheduler", "rng", "train_loader_generator_state",
                "history", "e_controller"):
        assert_equal(resumed[key], uninterrupted[key])
    # A completed resume must not train or query E even at the rollback boundary.
    monkeypatch.setattr(runner, "train_epoch", lambda *_args: pytest.fail("completed arm trained again"))
    monkeypatch.setattr(sys, "argv", args + ["--resume", str(at_rollback)])
    runner.main()
    assert len(queries) == 1
    # Reject an old one-shot arm whose identity lacks the E rollback rule.
    old = copy.deepcopy(uninterrupted)
    del old["run_identity"]["e_controller"]
    old_path = tmp_path / "old.pt"; torch.save(old, old_path)
    monkeypatch.setattr(sys, "argv", args + ["--resume", str(old_path)])
    with pytest.raises(ValueError, match="mismatch"):
        runner.main()
