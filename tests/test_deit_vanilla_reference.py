"""Vanilla export must reuse actual Phase 1 state without loading/training a model."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.deit_protocol import DeitRecipe, build_optimizer_scheduler, protocol, save_state
from experiments.shared_protocol import sha256_file
from experiments.deit_vanilla_reference import create_vanilla_reference, export_reused_vanilla_arm


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.fixture
def observed_window(deit_small, deit_batches, tmp_path, monkeypatch):
    import experiments.run_deit_fork as runner
    import experiments.deit_vanilla_reference as reuse
    recipe = replace(DeitRecipe(), workers=0, batch_size=8)
    optimizer, scheduler = build_optimizer_scheduler(deit_small, recipe)
    images, labels = deit_batches[0]
    torch.nn.functional.cross_entropy(deit_small(images), labels).backward()
    optimizer.step(); optimizer.zero_grad()
    loader = DataLoader(TensorDataset(images, labels), batch_size=8,
                        generator=torch.Generator().manual_seed(1))
    prefix = [{"epoch": 22, "validation_accuracy": .52, "validation_loss": 3.},
              {"epoch": 23, "validation_accuracy": .528, "validation_loss": 2.}]
    rows = [{"epoch": 24 + i, "validation_accuracy": .52633 if i < 2 else .512,
             "validation_loss": 2.9 if i == 0 else 2.8} for i in range(150)]
    fork_path = tmp_path / "plateau_checkpoint.pt"
    save_state(fork_path, model=deit_small, optimizer=optimizer, scheduler=scheduler, loader=loader,
        epoch=23, history=prefix, train_indices=list(range(8)), evaluation_indices=[8, 9],
        trigger_indices=[10], source_tuning_indices=[], run_protocol=protocol(recipe, deit_small),
        kind="deit_plateau_fork", historical_best_accuracy=.528, historical_best_loss=2.,
        historical_best_epoch=23, stall_detected_epoch=173, stall_history=rows)
    fork = torch.load(fork_path, weights_only=False)
    terminal = copy.deepcopy(fork)
    terminal.update(kind="deit_vanilla_latest", epoch=173, history=prefix + rows, plateau_detected=True)
    # Distinct late weights, nonzero Adam moments, scheduler, RNG and loader state.
    first = next(iter(terminal["model"]))
    terminal["model"][first].add_(.01)
    next(iter(terminal["optimizer"]["state"].values()))["exp_avg"].add_(.2)
    terminal["scheduler"]["last_epoch"] = 173
    terminal["rng"]["torch"] = torch.Generator().manual_seed(173).get_state()
    terminal["train_loader_generator_state"] = torch.Generator().manual_seed(174).get_state()
    terminal_path = tmp_path / "phase1_latest.pt"
    torch.save(terminal, terminal_path)
    def tiny_checked(path, kinds):
        state = torch.load(path, weights_only=False)
        assert state["kind"] in kinds
        return state, recipe
    monkeypatch.setattr(runner, "checked_source", tiny_checked)
    monkeypatch.setattr(reuse, "checked_source", tiny_checked)
    reference_path = tmp_path / "vanilla_reference.pt"
    reference = create_vanilla_reference(fork_path, terminal_path, reference_path)
    return fork_path, fork, terminal_path, terminal, reference_path, reference


def test_export_retains_actual_terminal_state_and_original_metrics(observed_window, tmp_path):
    fork_path, fork, terminal_path, terminal, _, reference = observed_window
    identity = {"fork_hash": sha256_file(fork_path), "post_fork_epochs": 150}
    output = tmp_path / "vanilla"
    result = export_reused_vanilla_arm(fork, reference, output, identity)
    state = torch.load(output / "checkpoint_latest.pt", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "rng", "train_loader_generator_state"):
        assert_state_equal(state[key], terminal[key])
    assert result["fork_epoch"] == 23 and state["epoch"] == 173
    assert result["source_phase1_checkpoint_hash"] == sha256_file(terminal_path)
    assert result["theta_best_hash"] == sha256_file(fork_path)
    assert result["history"][1:] == [{**row, "post_fork_epoch": i + 1}
                                    for i, row in enumerate(fork["stall_history"])]
    assert result["report_best_accuracy"] == .52633
    assert result["report_best_loss"] == 2.9  # Same-accuracy lower loss does not move strict best.
    assert result["report_best_epoch"] == 24
    assert result["delta_vs_historical_best"] == pytest.approx(-.00167)
    assert result["scientific_escape"] is False
    assert result["additional_training_epochs"] == 0
    assert len(result["validation_1_to_5_epochs_after"]) == 5


def test_vanilla_cli_never_loads_data_model_or_trains_even_on_resume(observed_window, tmp_path, monkeypatch):
    import experiments.run_deit_fork as runner
    fork_path, _, _, _, reference_path, _ = observed_window
    def forbidden(*_args, **_kwargs):
        pytest.fail("Vanilla reference export loaded data/model, evaluated or trained")
    for name in ("load_training_context", "train_epoch", "evaluate_without_rng",
                 "materialize_probe_batches", "one_shot_intervention", "seed_everything"):
        monkeypatch.setattr(runner, name, forbidden)
    output = tmp_path / "vanilla_cli"
    args = ["run_deit_fork", "--plateau-checkpoint", str(fork_path),
            "--plateau-checkpoint-hash", sha256_file(fork_path), "--method", "vanilla_continue",
            "--vanilla-reference", str(reference_path), "--output", str(output), "--device", "cuda:999"]
    monkeypatch.setattr(sys, "argv", args)
    runner.main()  # No data root or usable GPU is needed.
    first = json.loads((output / "result.json").read_text())
    monkeypatch.setattr(sys, "argv", args + ["--resume", str(output / "checkpoint_latest.pt")])
    runner.main()
    assert json.loads((output / "result.json").read_text()) == first
    assert first["trajectory_source"] == "phase1_plateau_window"


@pytest.mark.parametrize("problem", ["fork_hash", "horizon", "gap", "seed", "split"])
def test_reuse_rejects_unmatched_references(observed_window, tmp_path, problem):
    fork_path, fork, _, _, _, original = observed_window
    reference = copy.deepcopy(original)
    identity = {"fork_hash": sha256_file(fork_path), "post_fork_epochs": 150}
    if problem == "fork_hash":
        reference["theta_best_hash"] = "wrong"
    elif problem == "horizon":
        identity["post_fork_epochs"] = 149
    elif problem == "gap":
        reference["history"].pop(5)
    elif problem == "seed":
        reference["protocol"]["seed"] = 2
    else:
        reference["evaluation_indices"] = [99]
    with pytest.raises(ValueError):
        export_reused_vanilla_arm(fork, reference, tmp_path / "bad", identity)
    assert not (tmp_path / "bad" / "result.json").exists()


def test_reference_creation_rejects_terminal_from_another_trajectory(observed_window, tmp_path):
    fork_path, _, _, terminal, _, _ = observed_window
    wrong = copy.deepcopy(terminal)
    wrong["history"][0]["validation_loss"] = 100.
    wrong_path = tmp_path / "wrong_terminal.pt"
    torch.save(wrong, wrong_path)
    with pytest.raises(ValueError, match="does not match"):
        create_vanilla_reference(fork_path, wrong_path, tmp_path / "bad_reference.pt")


def test_notebook_exports_vanilla_and_launches_only_two_training_arms(tmp_path):
    from dataclasses import asdict
    from adapters.deit_cp_adapter import CPConfig
    notebook = json.loads(Path("notebooks/kaggle_deit_tiny_seed1_one_shot.ipynb").read_text())
    cell = next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and "# Export Vanilla on CPU" in "".join(cell["source"]))
    calls = []
    namespace = {"invoke": lambda module, args: calls.append((module, args)),
                 "OUTPUT": tmp_path, "FORK": tmp_path / "fork.pt", "FORK_HASH": "shared-fork",
                 "VANILLA_REFERENCE": tmp_path / "reference.pt", "POST_FORK_EPOCHS": 150, "ALGORITHM_PATIENCE": 10,
                 "CP": CPConfig(), "DATA_ROOT": "cifar", "states": [],
                 "flags": lambda _config: [], "asdict": asdict, "json": json}
    exec(compile(cell, "kaggle-arm-orchestration", "exec"), namespace)
    assert len(calls) == 3
    assert [args[args.index("--method") + 1] for _, args in calls] == [
        "vanilla_continue", "o_projection_only", "ours_e_driven_o"]
    assert "--data-root" not in calls[0][1]
    assert "--vanilla-reference" in calls[0][1]
    assert calls[0][1][calls[0][1].index("--device") + 1] == "cpu"
    for _, args in calls:
        assert args[args.index("--plateau-checkpoint-hash") + 1] == "shared-fork"
        assert args[args.index("--post-fork-epochs") + 1] == 150
    assert all("--data-root" in args for _, args in calls[1:])
    assert "--algorithm-patience" not in calls[1][1]
    assert calls[2][1][calls[2][1].index("--algorithm-patience") + 1] == 10
