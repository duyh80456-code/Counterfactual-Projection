import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import experiments.run_plateau_fork as runner
from experiments.run_plateau_fork import (
    next_controller_stall_counter, validation_state_updates)


@pytest.mark.parametrize(
    ("accuracy", "loss", "expected_report", "expected_anchor", "reason",
     "expected_stall"),
    [
        (0.76, 1.5, True, True, "accuracy_increase", 0),
        (0.75, 1.39115, False, True, "same_accuracy_lower_loss", 0),
        (0.75, 1.6, False, False, None, 6),
        (0.74, 1.0, False, False, None, 6),
    ],
    ids=("higher-accuracy", "same-accuracy-lower-loss",
         "same-accuracy-higher-loss", "lower-accuracy"))
def test_report_anchor_and_stall_updates_cover_all_four_cases(
        accuracy, loss, expected_report, expected_anchor, reason,
        expected_stall):
    report_improved, anchor_improved, anchor_reason = validation_state_updates(
        accuracy=accuracy, loss=loss,
        report_best_accuracy=0.75,
        anchor_accuracy=0.75, anchor_loss=1.41308)

    assert report_improved is expected_report
    assert anchor_improved is expected_anchor
    assert anchor_reason == reason
    assert next_controller_stall_counter(5, anchor_improved) == expected_stall


def test_epoch_302_tie_resets_stall_without_changing_report_best():
    report_improved, anchor_improved, reason = validation_state_updates(
        accuracy=73.1333, loss=1.39115,
        report_best_accuracy=73.1333,
        anchor_accuracy=73.1333, anchor_loss=1.41308)

    assert (report_improved, anchor_improved, reason) == (
        False, True, "same_accuracy_lower_loss")
    assert next_controller_stall_counter(14, anchor_improved) == 0


def test_fifteen_epochs_without_anchor_improvement_reaches_retrigger():
    counter = 0
    for _ in range(15):
        counter = next_controller_stall_counter(counter, anchor_improved=False)

    assert counter == 15
    assert counter >= 15


def test_main_loop_rolls_back_to_epoch_302_anchor_after_fifteen_stalls(
        tmp_path, monkeypatch):
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([243.0]))

    class ToyScheduler:
        def __init__(self, state=None):
            self.last_epoch = 0
            if state is not None:
                self.load_state_dict(state)

        def step(self):
            self.last_epoch += 1

        def state_dict(self):
            return {"last_epoch": self.last_epoch}

        def load_state_dict(self, state):
            self.last_epoch = int(state["last_epoch"])

    args = argparse.Namespace(
        method="ours_e_driven_o", reference_root=str(tmp_path),
        data_root=str(tmp_path), plateau_checkpoint="",
        plateau_checkpoint_hash="", output=str(tmp_path / "arm"),
        post_fork_epochs=75, seed=1, architecture="resnet18", batch_size=1,
        workers=0, weight_decay=0.0, site="toy", o_only_site="",
        site_selection_mode="all_functional_gain", rank=1,
        probe_epsilon=0.05, statistics_samples=1, where_batches=1,
        where_samples=1, projection_samples=1, gate_samples=1,
        cg_iterations=1, cg_relative_tolerance=1e-2,
        cg_preconditioner_probes=1, damping=1e-3,
        line_search_scales="0.1", retrigger_patience=15, opt1_epochs=1,
        max_opt2_epochs=1, contraction_epsilon=0.002, gamma_slope=1e-6,
        gamma_increase_opt2_epoch=1, gamma_post_increase_multiplier=2.0)
    tracker = {"epoch": 243, "rng": 0, "loader": None,
               "scheduler": None, "rollback_observations": [],
               "expected_loader_state": None, "expected_rng": None}

    fork_model = ToyModel()
    fork_model.weight.data.fill_(243.0)
    fork_optimizer = torch.optim.SGD(
        fork_model.parameters(), lr=0.001, momentum=0.9)
    fork_optimizer.state[fork_model.weight]["momentum_buffer"] = torch.tensor(
        [243.0])
    loader_generator = torch.Generator().manual_seed(10)
    checkpoint = {
        "kind": "plateau_fork_checkpoint", "epoch": 243,
        "model": fork_model.state_dict(),
        "optimizer": fork_optimizer.state_dict(),
        "scheduler": {"last_epoch": 243},
        "rng": {"marker": 0},
        "train_loader_generator_state": loader_generator.get_state(),
        "train_indices": [0], "trigger_indices": [1],
        "evaluation_indices": [2], "source_tuning_indices": [3],
        "protocol": {"architecture": "Toy"},
    }
    checkpoint_path = tmp_path / "fork.pt"
    torch.save(checkpoint, checkpoint_path)
    args.plateau_checkpoint = str(checkpoint_path)
    args.plateau_checkpoint_hash = hashlib.sha256(
        checkpoint_path.read_bytes()).hexdigest()

    monkeypatch.setattr(runner, "arguments", lambda: args)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.torch.cuda, "reset_peak_memory_stats",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_allocated",
                        lambda *_args, **_kwargs: 0)
    original_load = torch.load
    monkeypatch.setattr(
        runner.torch, "load",
        lambda path, **kwargs: original_load(path, map_location="cpu",
                                             weights_only=False))
    monkeypatch.setattr(runner, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(runner, "architecture_label", lambda _arch: "Toy")
    monkeypatch.setattr(
        runner, "datasets_and_indices",
        lambda *_args: (object(), object(), [0], [1, 2], [3]))
    monkeypatch.setattr(runner, "build_cifar_gromo_resnet",
                        lambda *_args: ToyModel())

    def make_optimizer(model, *_args):
        return torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9), None

    monkeypatch.setattr(runner, "build_optimizer_scheduler", make_optimizer)

    def remember_scheduler(state):
        tracker["scheduler"] = ToyScheduler(state)
        return tracker["scheduler"]

    monkeypatch.setattr(
        runner, "scheduler_from_state",
        lambda _optimizer, state: remember_scheduler(state))

    def make_train_loader(*_args):
        generator = torch.Generator()
        generator.set_state(checkpoint["train_loader_generator_state"])
        tracker["loader"] = SimpleNamespace(generator=generator)
        return tracker["loader"]

    monkeypatch.setattr(runner, "make_train_loader", make_train_loader)
    monkeypatch.setattr(
        runner, "make_eval_loader",
        lambda _dataset, indices, *_args: SimpleNamespace(
            kind="trigger" if list(indices) == [1] else "evaluation"))
    monkeypatch.setattr(runner, "restore_rng", lambda state: tracker.update(
        rng=state["marker"]))
    monkeypatch.setattr(runner, "rng_state",
                        lambda: {"marker": tracker["rng"]})

    def train_epoch(model, loader, optimizer, _device, **_kwargs):
        tracker["epoch"] += 1
        epoch = tracker["epoch"]
        model.weight.data.fill_(float(epoch))
        optimizer.state[model.weight]["momentum_buffer"] = torch.tensor(
            [float(epoch)])
        torch.rand((), generator=loader.generator)
        tracker["rng"] += 1
        if epoch == 317:
            tracker["expected_loader_state"] = loader.generator.get_state().clone()
            tracker["expected_rng"] = tracker["rng"]
        return {"task_loss": 1.0, "accuracy": 0.5}

    monkeypatch.setattr(runner, "train_epoch", train_epoch)

    def evaluate(_model, _loader, _device):
        epoch = tracker["epoch"]
        accuracy = 0.731333
        if epoch == 243:
            loss = 1.6
        elif 250 <= epoch <= 300 and epoch % 10 == 0:
            loss = 1.4 - epoch / 10000
        elif epoch == 302:
            loss = 1.0
        else:
            loss = 2.0
        return {"accuracy": accuracy, "loss": loss}

    monkeypatch.setattr(runner, "evaluate", evaluate)

    def fake_intervention(model, optimizer, *_args):
        if tracker["epoch"] >= 317:
            momentum = optimizer.state[model.weight]["momentum_buffer"]
            tracker["rollback_observations"].append({
                "epoch": tracker["epoch"],
                "model_epoch": int(model.weight.item()),
                "optimizer_epoch": int(momentum.item()),
                "scheduler_epoch": tracker["scheduler"].last_epoch,
                "rng": tracker["rng"],
                "loader_state": tracker["loader"].generator.get_state().clone(),
            })
        return {"correction_applied": False}

    monkeypatch.setattr(runner, "run_intervention", fake_intervention)
    monkeypatch.setattr(
        runner, "atomic_json_save",
        lambda payload, path: path.write_text(json.dumps(payload)))

    runner.main()

    epoch_302 = next(row for row in json.loads(
        (Path(args.output) / "result.json").read_text())["history"]
        if row["epoch"] == 302)
    assert epoch_302["report_best_improved"] is False
    assert epoch_302["controller_anchor_improved"] is True
    assert epoch_302["controller_anchor_reason"] == "same_accuracy_lower_loss"
    assert epoch_302["controller_stall_counter"] == 0
    epoch_316 = next(row for row in json.loads(
        (Path(args.output) / "result.json").read_text())["history"]
        if row["epoch"] == 316)
    assert epoch_316["controller_stall_counter"] == 14

    rollback = next(iter(tracker["rollback_observations"]))
    assert rollback["epoch"] == 317
    assert rollback["model_epoch"] == 302
    assert rollback["optimizer_epoch"] == 302
    assert rollback["scheduler_epoch"] == 302
    assert rollback["rng"] == tracker["expected_rng"]
    assert torch.equal(rollback["loader_state"],
                       tracker["expected_loader_state"])

    best = original_load(Path(args.output) / "checkpoint_best.pt",
                         map_location="cpu", weights_only=False)
    assert best["controller_anchor_epoch"] == 302
    assert int(best["model"]["weight"].item()) == 302


def test_plateau_fork_has_recurrent_e_and_single_shot_o_control():
    source = Path("experiments/run_plateau_fork.py").read_text()
    assert 'METHODS = ("vanilla", "bypass", "ours_e_driven_o", "o_projection_only")' in source
    assert '"o_projection_only")' in source
    assert 'source.get("kind") != "plateau_fork_checkpoint"' in source
    assert "optimizer.load_state_dict(source[\"optimizer\"])" in source
    assert 'scheduler_from_state(optimizer, source["scheduler"])' in source
    assert "scheduler.sync_optimizer_groups()" in source
    assert "train_indices = list(source[\"train_indices\"])" in source
    assert '"mode": "recurrent_best_rollback"' in source
    assert 'choices=("all_functional_gain",)' in source
    assert 'parser.add_argument(\n        "--o-only-site"' in source
    assert "o_only_site = args.o_only_site or args.site" in source
    assert '"site_selection_mode": args.site_selection_mode' in source
    assert '"mode": "single_initial_intervention"' in source
    assert '"patience": args.retrigger_patience' in source
    assert "run_intervention(" in source
    assert "run_o_only_intervention(" in source
    assert "supervised_functional_descent_direction" in source
    assert '"raw_validation_stall"' in source
    assert 'args.method == "ours_e_driven_o" and' in source
    assert ('args.method in {"ours_e_driven_o", "o_projection_only"} and' not
            in source)
    assert "live_rng = rng_state()" in source
    assert "live_loader_state = train_loader.generator.get_state().clone()" in source
    assert 'best_state["model"]' in source
    assert 'optimizer.load_state_dict(best_state["optimizer"])' in source
    assert 'scheduler.load_state_dict(best_state["scheduler"])' in source
    assert 'restore_rng(live_rng)' in source
    assert 'train_loader.generator.set_state(live_loader_state.cpu())' in source
    assert "perform_intervention(" in source
    assert '"intervention_count"' in source
    assert '"rollback_count"' in source
    assert "pre_probe_rng = rng_state()" in source
    assert "restore_rng(pre_probe_rng)" in source
    assert "embed_relaxed_bypass" in source
    assert "transition_from_opt2_" in source
    assert '"time_spent_expanded_seconds"' in source
    assert '"peak_train_params"' in source
    assert '"epochs_to_best"' in source
    assert 'best_checkpoint = output / "checkpoint_best.pt"' in source
    assert '"plateau_fork_arm_best"' in source
    assert '"--post-fork-epochs", type=int, default=150' in source
    assert '"--retrigger-patience", type=int, default=10' in source
    assert '"--opt1-epochs", type=int, default=70' in source
    assert '"--max-opt2-epochs", type=int, default=30' in source
    assert '"--gamma-increase-opt2-epoch", type=int, default=15' in source
    assert '"--gamma-post-increase-multiplier", type=float, default=2.0' in source
    assert '"plateau_checkpoint_hash": fork_hash' in source
    assert '"train_indices": train_indices' in source
    assert '"trigger_indices": trigger_indices' in source
    assert '"evaluation_indices": evaluation_indices' in source
    assert '"theta_best_hash": fork_hash' in source
    assert '"opt1_epochs": opt1_done if args.method == "bypass" else None' in source
    assert 'validation_state_updates(' in source
    assert 'loss < anchor_loss' in source
    assert '"controller_anchor_rule": "accuracy, then lower loss on exact accuracy tie"' in source
    assert '"report_stall_counter": report_stall_counter' in source
    assert '"controller_stall_counter": controller_stall_counter' in source
    assert '"report_best_loss": report_best_loss' in source
    assert '"best_validation_loss": report_best_loss' in source
    assert 'controller_stall_counter = next_controller_stall_counter(' in source
    assert 'significant_improved = (' not in source
    assert 'row["controller_anchor_reason"] = controller_anchor_reason' in source
    assert 'row["exact_best_improved"]' not in source
    assert 'row["stall_counter"]' not in source
    assert 'scheduler.step(trigger["accuracy"])' not in source
    assert 'phase = "incomplete"' in source
    assert '"budget_exhausted_before_contraction"' in source
    assert '"compact_best_validation_accuracy"' in source
    assert "max(compact_rows," in source
    assert '"post_fork_epoch": 0' not in source
    assert 'accuracy > anchor_accuracy' in source
    assert "epochs_since_best" not in source
