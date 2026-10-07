"""Protocol/runner regression tests on a tiny model with actual AdamW steps."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from adapters.deit_cp_adapter import CPConfig, one_shot_intervention
from experiments.deit_protocol import (DeitRecipe, build_optimizer_scheduler,
    protocol, canonical_model_config, checked_source, save_state, evaluate_without_rng, historical_best)
from experiments.shared_protocol import restore_rng, rng_state, sha256_file


def test_checkpoint_recipe_flags_and_architecture_guard(tmp_path):
    recipe = DeitRecipe()
    payload = {"kind": "deit_plateau_fork", "protocol": protocol(recipe, canonical_model_config()),
               "epoch": 23, "historical_best_epoch": 23, "historical_best_accuracy": .528,
               "historical_best_loss": 2., "history": [
                   {"epoch": 23, "validation_accuracy": .528, "validation_loss": 2.}],
               "stall_history": [{"epoch": 25, "validation_accuracy": .512}]}
    path = tmp_path / "fork.pt"
    torch.save(payload, path)
    assert checked_source(path, {"deit_plateau_fork"})[1] == recipe
    for key, value in (("architecture", "CIFAR-ResNet18"), ("protocol_version", 0), ("protocol_version", 1), ("protocol_version", 2),
                       ("drop_path_rate", .1), ("pretrained", True)):
        wrong = copy.deepcopy(payload)
        wrong["protocol"][key] = value
        torch.save(wrong, path)
        with pytest.raises(ValueError):
            checked_source(path, {"deit_plateau_fork"})


def test_one_shot_reuses_where_raw_gain_and_no_aux_persistence(deit_small, deit_batches, deit_native_candidate):
    recipe = DeitRecipe(workers=0)
    optimizer, scheduler = build_optimizer_scheduler(deit_small, recipe)
    # Seed nonzero Adam moments to test the full apply path.
    torch.nn.functional.cross_entropy(deit_small(deit_batches[0][0]), deit_batches[0][1]).backward()
    optimizer.step()
    parameters_before = dict(deit_small.named_parameters())
    weights_before = {name: value.detach().clone() for name, value in parameters_before.items()}
    moments_before = {name: copy.deepcopy(optimizer.state[value])
                      for name, value in parameters_before.items()}
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    before_rng = torch.get_rng_state().clone()
    names = set(deit_small.state_dict())
    config = replace(CPConfig(), rank=2, projection_samples=8,
                     cg_iterations=8, cg_preconditioner_probes=0, o_only_site="blocks.1.mlp")
    result = one_shot_intervention(deit_small, optimizer, statistics=deit_batches[:2],
        where_batches=deit_batches[2:], projection_batch=deit_batches[2], gate_batch=deit_batches[3], config=config)
    assert result["selection_candidate_count"] == 2
    expected = max(result["site_evaluations"], key=lambda key: result["site_evaluations"][key]["mean_e_gain"])
    assert result["selected_site"] == expected
    assert result["site_evaluations"][expected]["gain_per_functional_delta_norm"] is not None
    assert result["where_selector"] == "mean_observed_structural_E_gain"
    changed = set(result["projection_changed_parameters"])
    assert result["correction_applied"] and changed  # Exercise an actual jump, not empty sets.
    assert result["momentum_states_reset"] == len(changed)
    assert isinstance(result["momentum_states_reset"], int)
    assert changed == set(result["adam_moments_reset_parameters"])
    assert all(name.startswith(result["selected_site"] + ".") for name in changed)
    assert changed.issubset(result["projection_parameter_names"])
    auxiliary_parameter_names = set(dict(deit_small.named_parameters())) - set(parameters_before)
    assert not auxiliary_parameter_names
    assert auxiliary_parameter_names.isdisjoint(changed)
    actual_changed = {name for name, value in deit_small.named_parameters()
                      if not torch.equal(value, weights_before[name])}
    assert changed == actual_changed
    for name, value in deit_small.named_parameters():
        assert value is parameters_before[name]
        state = optimizer.state[value]
        assert torch.equal(state["step"], moments_before[name]["step"])
        for key in ("exp_avg", "exp_avg_sq"):
            if name in changed:
                assert torch.count_nonzero(state[key]) == 0
            else:
                assert torch.equal(state[key], moments_before[name][key])
    assert scheduler.state_dict() == before_scheduler
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert set(deit_small.state_dict()) == names
    assert all(not block.mlp._forward_hooks for block in deit_small.blocks)
    assert result["evaluation_role"] == "gate_batch_used_for_scale_selection"
    assert result["actual_loss_improvement_role"] == result["evaluation_role"]
    assert not any("heldout" in key for key in result)
    for key in ("cg_attempts", "cg_converged", "cg_damping_used", "gate_cosine_alignment",
                "gate_relative_residual", "gate_fitted_norm_ratio", "selected_scale",
                "line_search_gains", "actual_loss_improvement", "site_functional_evaluations"):
        assert key in result


def _tiny_context(model_config, dataset, recipe, device, source):
    from models import DeiTTinyCifar
    model = DeiTTinyCifar(**model_config).double().to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, recipe)
    model.load_state_dict(source["model"])
    optimizer.load_state_dict(source["optimizer"])
    scheduler.load_state_dict(source["scheduler"])
    generator = torch.Generator().set_state(source["train_loader_generator_state"])
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=generator)
    eval_loader = DataLoader(dataset, batch_size=8)
    restore_rng(source["rng"])
    return model, optimizer, scheduler, loader, eval_loader, dataset, list(range(len(dataset))), [0, 1], [], []


@pytest.mark.parametrize("method", ["ours_e_driven_o", "o_projection_only", "vanilla_continue"])
def test_main_loop_resume_does_not_apply_one_shot_twice(deit_small, deit_batches, tmp_path, monkeypatch, method):
    import experiments.run_deit_fork as runner
    batch = deit_batches[0]
    dataset = TensorDataset(*batch)
    recipe = replace(DeitRecipe(), workers=0, batch_size=8)
    optimizer, scheduler = build_optimizer_scheduler(deit_small, recipe)
    generator = torch.Generator().manual_seed(recipe.seed)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=generator)
    source_path = tmp_path / "theta.pt"
    historical = evaluate_without_rng(deit_small, DataLoader(dataset, batch_size=8), torch.device("cpu"))
    save_state(source_path, model=deit_small, optimizer=optimizer, scheduler=scheduler,
        loader=loader, epoch=100, history=[{"epoch": 100, "validation_accuracy": historical["accuracy"],
                                         "validation_loss": historical["loss"]}],
        train_indices=list(range(8)), evaluation_indices=[0, 1],
        source_tuning_indices=[], trigger_indices=[], run_protocol=protocol(recipe, deit_small),
        kind="deit_plateau_fork", historical_best_accuracy=historical["accuracy"],
        historical_best_loss=historical["loss"], historical_best_epoch=100)
    monkeypatch.setattr(runner, "checked_source", lambda path, _kinds: (
        torch.load(path, weights_only=False), recipe))
    monkeypatch.setattr(runner, "load_training_context", lambda _root, _recipe, device, source:
        _tiny_context(deit_small.config, dataset, recipe, device, source))
    # The jump itself is separately covered with native TINY above. This test
    # exercises runner application count and bitwise AdamW/scheduler resume.
    calls = []
    def jump(model, _optimizer, **kwargs):
        calls.append(True)
        with torch.no_grad():
            model.blocks[0].mlp.fc1.bias.add_(.01)
        return {"selected_site": "blocks.0.mlp", "correction_applied": True}
    monkeypatch.setattr(runner, "one_shot_intervention", jump)
    monkeypatch.setattr(runner, "materialize_probe_batches", lambda *_args: ({}, {}))
    output = tmp_path / "arm"
    args = ["run_deit_fork", "--data-root", "unused", "--plateau-checkpoint", str(source_path),
        "--plateau-checkpoint-hash", sha256_file(source_path), "--method", method,
        "--output", str(output), "--horizon", "2", "--device", "cpu"]
    original_save = runner.save_state
    midpoint = tmp_path / "midpoint.pt"
    def capture(path, **kwargs):
        original_save(path, **kwargs)
        if kwargs["completed_epochs"] == 1:
            torch.save(torch.load(path, weights_only=False), midpoint)
    monkeypatch.setattr(runner, "save_state", capture)
    monkeypatch.setattr(sys, "argv", args)
    runner.main()
    uninterrupted = torch.load(output / "checkpoint_latest.pt", weights_only=False)
    expected = 0 if method == "vanilla_continue" else 1
    assert len(calls) == expected
    monkeypatch.setattr(sys, "argv", args + ["--resume", str(midpoint)])
    runner.main()
    resumed = torch.load(output / "checkpoint_latest.pt", weights_only=False)
    assert len(calls) == expected
    assert len(resumed["interventions"]) == expected
    assert resumed["scheduler"] == uninterrupted["scheduler"]
    assert resumed["history"] == uninterrupted["history"]
    assert all(torch.equal(value, uninterrupted["model"][key]) for key, value in resumed["model"].items())
    for key, state in resumed["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, uninterrupted["optimizer"]["state"][key][name])
    result = json.loads((output / "result.json").read_text())
    assert len(result["validation_1_to_5_epochs_after"]) == 2
    if expected:
        assert len(result["interventions"][0]["validation_1_to_5_epochs_after"]) == 2
    else:
        assert result["validation_immediately_after_projection"] == result["validation_before"]
        assert resumed["history"][0]["metric_timing"] == "before_SGD_no_projection"
    assert result["validation_before"] is not None
    assert result["validation_immediately_after_projection"] is not None
    assert result["historical_best_accuracy"] == historical["accuracy"]
    assert result["delta_vs_historical_best"] == result["report_best_accuracy"] - historical["accuracy"]
    assert result["scientific_escape"] == (result["report_best_accuracy"] > historical["accuracy"])
    assert result["report_best_accuracy"] == max(row["validation_accuracy"] for row in result["history"][1:])
    # Arbitrary report values must not change the jump, horizon or training trajectory.
    monkeypatch.setattr(runner, "evaluate_without_rng", lambda *_args:
                        {"accuracy": .999, "loss": 100.})
    alternate_args = list(args)
    alternate_output = tmp_path / "alternate_reports"
    alternate_args[alternate_args.index("--output") + 1] = str(alternate_output)
    monkeypatch.setattr(sys, "argv", alternate_args)
    runner.main()
    alternate = torch.load(alternate_output / "checkpoint_latest.pt", weights_only=False)
    assert len(calls) == 2 * expected
    assert alternate["completed_epochs"] == 2
    assert alternate["scheduler"] == uninterrupted["scheduler"]
    assert all(torch.equal(value, uninterrupted["model"][key])
               for key, value in alternate["model"].items())
    for key, state in alternate["optimizer"]["state"].items():
        for name, value in state.items():
            assert torch.equal(value, uninterrupted["optimizer"]["state"][key][name])


def test_notebook_cells_compile_and_no_recurrent_controller():
    notebook = json.loads(Path("notebooks/kaggle_deit_tiny_seed1_one_shot.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"deit-cell-{index}", "exec")
    text = Path("experiments/run_deit_fork.py").read_text()
    assert "controller_stall_counter" not in text
    assert "retrigger_patience" not in text


def test_three_arms_share_fork_and_compare_against_historical_best(deit_small, deit_batches, tmp_path, monkeypatch):
    import experiments.run_deit_fork as runner
    dataset = TensorDataset(*deit_batches[0])
    recipe = replace(DeitRecipe(), workers=0, batch_size=8)
    optimizer, scheduler = build_optimizer_scheduler(deit_small, recipe)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(1))
    path = tmp_path / "historical_best.pt"
    save_state(path, model=deit_small, optimizer=optimizer, scheduler=scheduler, loader=loader,
        epoch=23, history=[{"epoch": 23, "validation_accuracy": .528, "validation_loss": 2.}],
        train_indices=list(range(8)), evaluation_indices=[0, 1], source_tuning_indices=[], trigger_indices=[],
        run_protocol=protocol(recipe, deit_small), kind="deit_plateau_fork",
        historical_best_accuracy=.528, historical_best_loss=2., historical_best_epoch=23,
        stall_detected_epoch=25, stall_history=[{"epoch": 25, "validation_accuracy": .512}])
    fork = torch.load(path, weights_only=False)
    fork_hash = sha256_file(path)
    monkeypatch.setattr(runner, "checked_source", lambda p, _kinds: (torch.load(p, weights_only=False), recipe))
    starts = []
    def context(_root, _recipe, device, source):
        result = _tiny_context(deit_small.config, dataset, recipe, device, source)
        starts.append(copy.deepcopy(result[0].state_dict()))
        assert result[1].state_dict() == fork["optimizer"]
        assert result[2].state_dict() == fork["scheduler"]
        assert torch.equal(result[3].generator.get_state(), fork["train_loader_generator_state"])
        return result
    monkeypatch.setattr(runner, "load_training_context", context)
    monkeypatch.setattr(runner, "materialize_probe_batches", lambda *_args: ({}, {}))
    monkeypatch.setattr(runner, "one_shot_intervention", lambda *_args, **_kwargs:
                        {"correction_applied": False, "selected_site": "blocks.0.mlp"})
    for method, accuracy in (("vanilla_continue", .52633), ("o_projection_only", .524),
                             ("ours_e_driven_o", .51967)):
        # Baseline immediate metric is deliberately wrong for escape comparison.
        values = iter([{"accuracy": .512, "loss": 3.}] * (1 if method == "vanilla_continue" else 2)
                      + [{"accuracy": accuracy, "loss": 2.9}, {"accuracy": accuracy - .001, "loss": 2.8}])
        monkeypatch.setattr(runner, "evaluate_without_rng", lambda *_args: next(values))
        output = tmp_path / method
        monkeypatch.setattr(sys, "argv", ["run_deit_fork", "--data-root", "unused",
            "--plateau-checkpoint", str(path), "--plateau-checkpoint-hash", fork_hash,
            "--method", method, "--output", str(output), "--horizon", "2", "--device", "cpu"])
        runner.main()
        result = json.loads((output / "result.json").read_text())
        assert result["theta_best_hash"] == fork_hash and result["fork_epoch"] == 23
        assert result["historical_best_accuracy"] == .528
        assert result["report_best_accuracy"] == accuracy
        assert result["delta_vs_historical_best"] == pytest.approx(accuracy - .528)
        assert result["scientific_escape"] is False
    assert len(starts) == 3
    assert all(torch.equal(value, fork["model"][name]) for start in starts for name, value in start.items())


def test_reject_detection_epoch_fork_even_if_metrics_are_relabelled():
    fork = {"epoch": 25, "historical_best_epoch": 23, "historical_best_accuracy": .528,
            "historical_best_loss": 2., "history": [
                {"epoch": 23, "validation_accuracy": .528, "validation_loss": 2.},
                {"epoch": 25, "validation_accuracy": .512, "validation_loss": 3.}]}
    with pytest.raises(ValueError, match="historical validation-best"):
        historical_best(fork)


@pytest.mark.parametrize("offset", [0, 22])
def test_vanilla_plateau_exports_historical_best_not_detection_state(tmp_path, monkeypatch, offset):
    import experiments.deit_protocol as shared
    import experiments.train_deit_plateau as runner
    dataset = TensorDataset(torch.randn(6, 3, 32, 32), torch.arange(6))
    monkeypatch.setattr(shared, "datasets_and_indices", lambda *_args:
        (dataset, dataset, [0, 1], [2, 3, 4, 5], []))
    recipe = DeitRecipe(workers=0, batch_size=2, validation_samples=4, trigger_samples=2,
                        tuning_samples=0, min_plateau_epoch=1, patience=2,
                        max_epoch=offset + 3, schedule_epochs=40, warmup_epochs=1)
    output = tmp_path / "vanilla"
    output.mkdir()
    # Start at epoch22 for the reported regression: epoch23=.528, epoch25=.512.
    resume_args = []
    if offset:
        context = shared.load_training_context("unused", recipe, torch.device("cpu"))
        model, optimizer, scheduler, loader, *_ = context
        scheduler_state = scheduler.state_dict()
        scheduler_state["last_epoch"] = offset
        scheduler.load_state_dict(scheduler_state)
        common = dict(model=model, optimizer=optimizer, scheduler=scheduler, loader=loader,
            epoch=offset, history=[{"epoch": offset, "validation_accuracy": .52, "validation_loss": 5.}],
            train_indices=[0, 1], evaluation_indices=[4, 5], source_tuning_indices=[], trigger_indices=[2, 3],
            run_protocol=protocol(recipe, model), report_best_accuracy=.52, report_best_loss=5.,
            report_best_epoch=offset, historical_best_accuracy=.52, historical_best_loss=5.,
            historical_best_epoch=offset, epochs_since_best=0)
        save_state(output / "checkpoint_best.pt", kind="deit_vanilla_best", **common)
        save_state(output / "checkpoint_latest.pt", kind="deit_vanilla_latest", plateau_detected=False, **common)
        resume_args = ["--resume", str(output / "checkpoint_latest.pt")]
    measurements = [{"accuracy": .528, "loss": 4.}, {"accuracy": .528, "loss": 3.9},
                    {"accuracy": .512, "loss": 3.8}]
    if not offset:
        measurements.insert(0, {"accuracy": .52, "loss": 5.})
    values = iter(measurements)
    def scripted_evaluation(_model, loader, _device):
        assert list(loader.dataset.indices) == [4, 5]  # No trigger-loader call allowed.
        return next(values)
    monkeypatch.setattr(runner, "evaluate_without_rng", scripted_evaluation)
    arguments = ["train_deit_plateau", "--data-root", "unused", "--output", str(output), "--device", "cpu"]
    from dataclasses import asdict
    for key, value in asdict(recipe).items():
        arguments += ["--" + key.replace("_", "-"), str(value)]
    monkeypatch.setattr(sys, "argv", arguments + resume_args)
    runner.main()
    fork, _ = checked_source(output / "plateau_checkpoint.pt", {"deit_plateau_fork"})
    best = torch.load(output / "checkpoint_best.pt", weights_only=False)
    latest = torch.load(output / "checkpoint_latest.pt", weights_only=False)
    assert fork["epoch"] == offset + 1
    assert fork["stall_detected_epoch"] == offset + 3
    assert historical_best(fork) == (.528, 4., offset + 1)
    assert latest["epochs_since_best"] == 2
    assert not any("trigger_accuracy" in row for row in latest["history"])
    assert all(torch.equal(value, best["model"][key]) for key, value in fork["model"].items())
    assert fork["scheduler"]["last_epoch"] == offset + 1
    assert latest["scheduler"]["last_epoch"] == offset + 3
    assert any(not torch.equal(value, latest["model"][key]) for key, value in fork["model"].items())
    summary = json.loads((output / "result.json").read_text())
    assert summary["fork_epoch"] == offset + 1 and summary["historical_best_accuracy"] == .528
    # Completed resume must reload the same historical best without extra SGD.
    monkeypatch.setattr(sys, "argv", arguments + ["--resume", str(output / "checkpoint_latest.pt")])
    monkeypatch.setattr(runner, "train_epoch", lambda *_args: pytest.fail("completed plateau trained again"))
    runner.main()
    assert torch.load(output / "plateau_checkpoint.pt", weights_only=False)["epoch"] == offset + 1
