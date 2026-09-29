from types import SimpleNamespace

import torch

from experiments import run_plateau_comparison as runner
from experiments.plateau_protocol import (
    BestCheckpointStallDetector, ConsecutiveWindowPlateauDetector,
    ConstantCheckpointScheduler,
    PlateauDetector)


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


def test_constant_checkpoint_scheduler_preserves_lr_and_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.037, momentum=0.9)
    scheduler = ConstantCheckpointScheduler(optimizer)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == 0.037
    restored = ConstantCheckpointScheduler(optimizer)
    restored.load_state_dict(scheduler.state_dict())
    assert restored.steps == 1


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
