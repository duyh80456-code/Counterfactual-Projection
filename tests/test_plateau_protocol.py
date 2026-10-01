from types import SimpleNamespace

import pytest
import torch

from experiments import run_plateau_comparison as runner
from experiments.run_vanilla_to_plateau import finalize_best_stall
from experiments.plateau_protocol import (
    BestCheckpointStallDetector, ConsecutiveWindowPlateauDetector,
    ConstantCheckpointScheduler, CosineFloorScheduler,
    SignificantPlateauScheduler, StandardMultiStepScheduler,
    scheduler_from_state, PlateauDetector, validation_stall_plan)


def test_plateau_requires_full_window_and_small_accuracy_and_loss_change():
    detector = PlateauDetector(
        window=3, accuracy_min_gain=0.001,
        loss_ema_min_drop=0.01, ema_alpha=1.0, minimum_epochs=2)
    assert detector.update(1, 0.70, 1.0)["plateau"] is False
    assert detector.update(2, 0.7004, 0.997)["plateau"] is False
    result = detector.update(3, 0.7005, 0.995)
    assert result["plateau"] is True


def test_real_improvement_prevents_plateau_and_reset_starts_new_segment():
    detector = PlateauDetector(
        window=3, accuracy_min_gain=0.001,
        loss_ema_min_drop=0.01, ema_alpha=1.0, minimum_epochs=2)
    detector.update(1, 0.70, 1.0)
    detector.update(2, 0.71, 0.98)
    assert detector.update(3, 0.72, 0.96)["plateau"] is False
    detector.reset()
    assert detector.update(4, 0.72, 0.96)["observations_since_probe"] == 1


def test_plateau_state_round_trip():
    detector = PlateauDetector(window=3, minimum_epochs=2)
    detector.update(10, 0.5, 1.5)
    restored = PlateauDetector(window=3, minimum_epochs=2)
    restored.load_state_dict(detector.state_dict())
    assert restored.records == detector.records


def test_convergence_detector_requires_two_complete_windows():
    detector = ConsecutiveWindowPlateauDetector(
        window=3, required_windows=2, accuracy_min_gain=0.01,
        loss_ema_min_drop=0.01, ema_alpha=1.0)
    outputs = []
    for epoch in range(1, 7):
        outputs.append(detector.update(epoch, 0.70, 1.0))
    assert outputs[2]["plateau"] is False
    assert outputs[2]["completed_window"]["qualifies"] is True
    assert outputs[5]["plateau"] is True
    assert outputs[5]["consecutive_plateau_windows"] == 2


def test_best_checkpoint_stall_tracks_best_and_waits_for_patience():
    detector = BestCheckpointStallDetector(patience=3, min_gain=0.01)
    assert detector.update(300, 0.70)["improved"] is True
    small = detector.update(301, 0.705)
    assert small["improved"] is True
    assert small["meaningful_improvement"] is False
    assert detector.update(302, 0.72)["meaningful_improvement"] is True
    assert detector.update(304, 0.719)["stalled"] is False
    result = detector.update(305, 0.71)
    assert result["stalled"] is True
    assert result["best_epoch"] == 302
    restored = BestCheckpointStallDetector(patience=3, min_gain=0.01)
    restored.load_state_dict(detector.state_dict())
    assert restored.state_dict() == detector.state_dict()


def test_stall_checkpoint_reuses_exactly_100_vanilla_epochs(tmp_path):
    detector = BestCheckpointStallDetector(patience=100, min_gain=0.001)
    history = []
    for epoch in range(300, 401):
        # This exact new best is below +0.1 pp, so it must be checkpointed
        # without restarting the significant-improvement stall clock.
        accuracy = 0.7705 if epoch == 350 else (0.77 if epoch == 300 else 0.76)
        detector.update(epoch, accuracy)
        history.append({
            "epoch": epoch, "validation_accuracy": accuracy,
            "validation_loss": 1.0 + (epoch - 300) * 0.001,
            "epoch_seconds": 2.0, "peak_gpu_memory": 123,
        })
    best_path = tmp_path / "checkpoint_best.pt"
    plateau_path = tmp_path / "plateau_checkpoint.pt"
    # theta_P is the last meaningful best (epoch 300); the smaller exact best
    # at epoch 350 is retained only as a diagnostic.
    torch.save({"epoch": 300, "kind": "vanilla_best_checkpoint"}, best_path)
    payload = finalize_best_stall(
        best_path, plateau_path, detector, history, {}, 42)
    control = payload["vanilla_control"]
    assert control["role"] == "matched_significant_best_to_stall_window"
    assert control["post_fork_epochs"] == 100
    assert control["fork_validation_accuracy"] == 0.77
    assert control["theta_P_validation_accuracy"] == 0.77
    assert control["meaningful_best_validation_accuracy"] == 0.77
    assert control["final_validation_accuracy"] == 0.76
    assert control["best_validation_accuracy"] == 0.7705
    assert control["best_validation_accuracy_delta"] == pytest.approx(0.0005)
    assert control["epochs_to_best"] == 50
    assert control["theta_P_validation_loss"] == 1.0
    assert control["best_validation_loss"] == 1.0
    assert control["exact_best_epoch_diagnostic"] == 350
    assert control["training_seconds"] == 200.0


def test_significant_threshold_is_inclusive_but_tiny_best_does_not_reset():
    detector = BestCheckpointStallDetector(patience=10, min_gain=0.001)
    detector.update(0, 0.7000)
    tiny = detector.update(1, 0.7005)
    assert tiny["improved"] is True
    assert tiny["meaningful_improvement"] is False
    threshold = detector.update(2, 0.7010)
    assert threshold["improved"] is True
    assert threshold["meaningful_improvement"] is True


def test_raw_validation_best_moves_the_exact_100_epoch_target():
    history = [
        {"epoch": epoch,
         "validation_accuracy": 0.71 if epoch == 140 else 0.70}
        for epoch in range(0, 240)]
    before = validation_stall_plan(history, patience=100)
    assert before["validation_best_epoch"] == 140
    assert before["target_epoch"] == 240
    assert before["minimum_additional_epochs_if_no_new_best"] == 1
    history.append({"epoch": 240, "validation_accuracy": 0.70})
    ready = validation_stall_plan(history, patience=100)
    assert ready["minimum_additional_epochs_if_no_new_best"] == 0
    assert ready["has_100_post_best_epochs"] is True


def test_post_recipe_validation_best_ignores_better_pre_recipe_epoch():
    history = [
        {"epoch": epoch,
         "validation_accuracy": (
             0.90 if epoch == 140 else 0.80 if epoch == 217 else 0.70)}
        for epoch in range(0, 318)]
    plan = validation_stall_plan(history, patience=100, min_epoch=200)
    assert plan["validation_best_epoch"] == 217
    assert plan["target_epoch"] == 317
    assert plan["has_100_post_best_epochs"] is True


def test_armed_exact_best_resets_patience_on_any_strict_validation_gain():
    detector = BestCheckpointStallDetector(
        patience=3, min_gain=0.0, require_arm=True,
        exact_best_patience=True)
    detector.update(199, 0.90)
    detector.update(200, 0.70)
    detector.arm_stall(200, 0.70)
    assert detector.update(201, 0.70001)["meaningful_improvement"] is True
    assert detector.update(202, 0.70001)["meaningful_improvement"] is False
    assert detector.update(203, 0.69)["stalled"] is False
    assert detector.update(204, 0.69)["stalled"] is True
    assert detector.best_epoch == 201


def test_exact_best_patience_does_not_reset_on_a_tie():
    detector = BestCheckpointStallDetector(
        patience=3, min_gain=0.0, exact_best_patience=True)
    assert detector.update(0, 0.70)["meaningful_improvement"] is True
    assert detector.update(1, 0.71)["meaningful_improvement"] is True
    assert detector.update(2, 0.71)["meaningful_improvement"] is False
    assert detector.update(4, 0.70)["stalled"] is True


def test_validation_best_fork_attaches_exactly_100_following_epochs(tmp_path):
    detector = BestCheckpointStallDetector(
        patience=100, min_gain=0.0, exact_best_patience=True)
    history = []
    for epoch in range(100, 241):
        accuracy = 0.80 if epoch == 140 else 0.79
        detector.update(epoch, accuracy)
        history.append({
            "epoch": epoch, "validation_accuracy": accuracy,
            "validation_loss": 1.0, "epoch_seconds": 1.0,
            "peak_gpu_memory": 1})
    best = tmp_path / "checkpoint_best.pt"
    plateau = tmp_path / "plateau_checkpoint.pt"
    torch.save({"epoch": 140}, best)
    payload = finalize_best_stall(
        best, plateau, detector, history,
        {"theta_P_scope": "global raw validation best"}, 42)
    control = payload["vanilla_control"]
    assert control["role"] == "matched_validation_best_to_100_epoch_window"
    assert control["stall_window_start_epoch"] == 141
    assert control["control_end_epoch"] == 240
    assert control["post_fork_epochs"] == 100


def test_final_stall_patience_does_not_count_before_lr_floor_arm():
    detector = BestCheckpointStallDetector(
        patience=3, min_gain=0.001, require_arm=True)
    detector.update(0, 0.70)
    detector.update(1, 0.80)
    assert detector.update(3, 0.70)["stalled"] is False
    detector.arm_stall(3, 0.70)
    assert detector.observations[-1]["epochs_without_improvement"] == 0
    assert detector.pre_arm_best_epoch == 1
    assert detector.pre_arm_best_metric == 0.80
    assert detector.best_epoch == 3
    assert detector.best_metric == 0.70
    assert detector.update(5, 0.70)["stalled"] is False
    result = detector.update(6, 0.70)
    assert result["stalled"] is True
    assert result["stall_armed_epoch"] == 3


def test_post_arm_exact_best_and_significant_reference_are_separate():
    detector = BestCheckpointStallDetector(
        patience=100, min_gain=0.001, require_arm=True)
    detector.update(160, 0.80)
    detector.update(200, 0.78)
    detector.arm_stall(200, 0.78)
    tiny = detector.update(217, 0.7805)
    assert tiny["improved"] is True
    assert tiny["meaningful_improvement"] is False
    assert detector.best_epoch == 217
    significant = detector.update(236, 0.7812)
    assert significant["improved"] is True
    assert significant["meaningful_improvement"] is True
    assert detector.best_epoch == 236
    assert detector.last_meaningful_improvement_epoch == 236


def test_strict_plateau_fork_uses_significant_best_not_tiny_exact_best(
        tmp_path):
    detector = BestCheckpointStallDetector(
        patience=3, min_gain=0.001, require_arm=True)
    detector.update(200, 0.70)
    detector.arm_stall(200, 0.70)
    history = [{
        "epoch": 200, "validation_accuracy": 0.75,
        "validation_loss": 1.0, "epoch_seconds": 0.0,
        "peak_gpu_memory": 1}]
    for epoch, trigger in ((201, 0.7005), (202, 0.7004), (203, 0.7003)):
        detector.update(epoch, trigger)
        history.append({
            "epoch": epoch, "validation_accuracy": 0.74,
            "validation_loss": 1.1, "epoch_seconds": 1.0,
            "peak_gpu_memory": 1})
    significant = tmp_path / "checkpoint_best.pt"
    plateau = tmp_path / "plateau_checkpoint.pt"
    torch.save({"epoch": 200}, significant)
    payload = finalize_best_stall(
        significant, plateau, detector, history, {}, 42)
    assert payload["epoch"] == 200
    assert payload["vanilla_control"]["post_fork_epochs"] == 3
    assert payload["vanilla_control"]["exact_best_epoch_diagnostic"] == 201


def test_constant_checkpoint_scheduler_preserves_lr_and_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.037, momentum=0.9)
    scheduler = ConstantCheckpointScheduler(optimizer)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.037
    restored = ConstantCheckpointScheduler(optimizer)
    restored.load_state_dict(scheduler.state_dict())
    assert restored.steps == 1


def test_cosine_floor_scheduler_is_single_trajectory_with_nonzero_floor():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = CosineFloorScheduler(
        optimizer, decay_epochs=300, eta_min=0.002)
    assert optimizer.param_groups[0]["lr"] == 0.1
    for _ in range(300):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.002
    for _ in range(100):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.002
    restored = scheduler_from_state(optimizer, scheduler.state_dict())
    assert restored.steps == 400
    assert restored.state_dict() == scheduler.state_dict()


def test_cosine_floor_scheduler_registers_bypass_group_without_restart():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([first], lr=0.1)
    scheduler = CosineFloorScheduler(
        optimizer, decay_epochs=10, eta_min=0.002)
    for _ in range(5):
        scheduler.step()
    inherited_lr = optimizer.param_groups[0]["lr"]
    optimizer.add_param_group({"params": [second], "lr": inherited_lr})
    scheduler.sync_optimizer_groups()
    assert len(scheduler.base_lrs) == 2
    assert optimizer.param_groups[1]["lr"] == inherited_lr
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == optimizer.param_groups[1]["lr"]


def test_event_driven_scheduler_has_no_epoch_horizon_and_reaches_floor():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = SignificantPlateauScheduler(
        optimizer, patience=2, factor=0.2, min_lr=0.002,
        threshold=0.001)
    scheduler.step(0.70)
    scheduler.step(0.70)
    scheduler.step(0.70)
    assert abs(optimizer.param_groups[0]["lr"] - 0.02) < 1e-12
    scheduler.step(0.70)
    scheduler.step(0.70)
    assert abs(optimizer.param_groups[0]["lr"] - 0.004) < 1e-12
    scheduler.step(0.70)
    scheduler.step(0.70)
    assert optimizer.param_groups[0]["lr"] == 0.002
    assert scheduler.at_floor() is True
    # A significant gain resets only the event counter, not the LR.
    scheduler.step(0.701)
    assert scheduler.bad_epochs == 0
    assert optimizer.param_groups[0]["lr"] == 0.002
    restored = scheduler_from_state(optimizer, scheduler.state_dict())
    assert restored.state_dict() == scheduler.state_dict()


def test_standard_backbone_recipe_is_metric_independent_and_resumable():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = StandardMultiStepScheduler(
        optimizer, milestones=(2, 3), gamma=0.1, recipe_epochs=4)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.1
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 0.01) < 1e-12
    scheduler.step()
    assert abs(optimizer.param_groups[0]["lr"] - 0.001) < 1e-12
    assert scheduler.recipe_complete() is False
    scheduler.step()
    assert scheduler.recipe_complete() is True
    restored = scheduler_from_state(optimizer, scheduler.state_dict())
    assert restored.state_dict() == scheduler.state_dict()


def test_loading_original_scheduler_preserves_horizon_lr_and_momentum():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.037, momentum=0.9)
    optimizer.state[parameter]["momentum_buffer"] = torch.tensor([2.0])
    original = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    for _ in range(10):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        original.step()
    optimizer_state = optimizer.state_dict()
    scheduler_state = original.state_dict()
    saved_momentum = optimizer.state[parameter]["momentum_buffer"].clone()
    restored_parameter = torch.nn.Parameter(torch.tensor([1.0]))
    restored_optimizer = torch.optim.SGD(
        [restored_parameter], lr=0.1, momentum=0.9)
    restored = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_optimizer, T_max=999)
    restored_optimizer.load_state_dict(optimizer_state)
    restored.load_state_dict(scheduler_state)
    assert restored.T_max == 60
    assert restored.last_epoch == 10
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert torch.equal(
        restored_optimizer.state[restored_parameter]["momentum_buffer"],
        saved_momentum)


def test_where_uses_mean_e_gain_not_tiny_score(monkeypatch):
    candidates = [
        SimpleNamespace(module_name="high_tiny", proposal_score=100.0),
        SimpleNamespace(module_name="high_gain", proposal_score=1.0),
    ]
    monkeypatch.setattr(
        runner, "propose_structural_candidates",
        lambda *_args, **_kwargs: candidates)

    gains = {
        "high_tiny": [0.01, 0.00, 0.01],
        "high_gain": [0.10, 0.08, 0.09],
    }

    class FakeProbe:
        def __call__(self, _model, *, candidate, batch, gate):
            index = int(batch[0].item())
            return SimpleNamespace(
                observed_loss_gain=gains[candidate.module_name][index])

    monkeypatch.setattr(runner, "CandidateExpansionProbe", FakeProbe)
    batches = [(torch.tensor([index]), torch.tensor([0]))
               for index in range(3)]
    selected, diagnostics = runner.select_by_expansion_gain(
        object(), [], batches,
        SimpleNamespace(rank=4, probe_epsilon=0.05), torch.device("cpu"))
    assert selected.module_name == "high_gain"
    assert diagnostics["site_evaluations"]["high_gain"]["rank"] == 1
    assert diagnostics["where_stability"] == 1.0


def test_plateau_runner_uses_shared_theta300_and_e_only_selects_where():
    source = open("experiments/run_plateau_comparison.py").read()
    assert 'source.get("kind") != "shared_fork_checkpoint"' in source
    assert 'int(source.get("epoch", -1)) != START_EPOCH' in source
    assert "optimizer.load_state_dict(source[\"optimizer\"])" in source
    assert "scheduler.load_state_dict(source[\"scheduler\"])" in source
    assert "theta300 checkpoint has no SGD optimizer state" in source
    assert 'group["lr"] =' not in source
    assert "single_fork_cosine_scheduler" not in source
    assert "scheduler.T_max" in source
    assert "scheduler.last_epoch" in source
    assert '"scheduler_restarted": False' in source
    assert '"scheduler_state_restored": True' in source
    assert "train_indices = list(source[\"train_indices\"])" in source
    assert "validation_indices[:args.trigger_samples]" in source
    assert "source_train[:" not in source
    assert "pre_probe_rng = rng_state()" in source
    assert "restore_rng(pre_probe_rng)" in source
    assert 'default=2000' in source
    assert 'default=15' in source
    assert 'default=1e-3' in source
    assert "checkpoint_pre_intervention_epoch" in source
    assert 'snapshot_kind="pre_intervention_plateau"' in source
    assert "mean_observed_structural_E_gain" in source
    assert "propose_structural_candidates" in source
    assert "projected_utility" not in source
    assert "0.0125,0.025,0.05" in source
    assert "best_gain > 0" in source
    assert "detector.reset()" in source
