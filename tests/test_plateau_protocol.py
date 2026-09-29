from types import SimpleNamespace

import torch

from experiments import run_plateau_comparison as runner
from experiments.plateau_protocol import PlateauDetector


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


def test_plateau_runner_uses_vanilla_theta360_and_e_only_selects_where():
    source = open("experiments/run_plateau_comparison.py").read()
    assert 'source_protocol.get("method") != "vanilla_continue"' in source
    assert 'int(source["history"][-1].get("epoch", -1)) != START_EPOCH' in source
    assert "optimizer.load_state_dict(source[\"optimizer\"])" in source
    assert "theta360 checkpoint has no SGD optimizer state" in source
    assert "checkpoint_pre_intervention_epoch" in source
    assert 'snapshot_kind="pre_intervention_plateau"' in source
    assert "mean_observed_structural_E_gain" in source
    assert "propose_structural_candidates" in source
    assert "projected_utility" not in source
    assert "0.0125,0.025,0.05" in source
    assert "best_gain > 0" in source
    assert "detector.reset()" in source
