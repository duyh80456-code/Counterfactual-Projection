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
    protocol, canonical_model_config, checked_source, save_state, evaluate_without_rng)
from experiments.shared_protocol import restore_rng, rng_state, sha256_file


def test_checkpoint_recipe_flags_and_architecture_guard(tmp_path):
    recipe = DeitRecipe()
    payload = {"kind": "deit_plateau_fork", "protocol": protocol(recipe, canonical_model_config())}
    path = tmp_path / "fork.pt"
    torch.save(payload, path)
    assert checked_source(path, {"deit_plateau_fork"})[1] == recipe
    for key, value in (("architecture", "CIFAR-ResNet18"), ("protocol_version", 0),
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
    assert changed == set(result["momentum_states_reset"])
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
    for key in ("cg_attempts", "cg_converged", "cg_damping_used", "heldout_cosine_alignment",
                "heldout_relative_residual", "heldout_fitted_norm_ratio", "selected_scale",
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
    save_state(source_path, model=deit_small, optimizer=optimizer, scheduler=scheduler,
        loader=loader, epoch=100, history=[], train_indices=list(range(8)), evaluation_indices=[0, 1],
        source_tuning_indices=[], trigger_indices=[], run_protocol=protocol(recipe, deit_small),
        kind="deit_plateau_fork")
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


def test_notebook_cells_compile_and_no_recurrent_controller():
    notebook = json.loads(Path("notebooks/kaggle_deit_tiny_seed1_one_shot.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"deit-cell-{index}", "exec")
    text = Path("experiments/run_deit_fork.py").read_text()
    assert "controller_stall_counter" not in text
    assert "retrigger_patience" not in text


def test_vanilla_plateau_exports_strict_best_state_and_matching_loss(deit_small, tmp_path, monkeypatch):
    # Canonical backbone and real AdamW updates; validation values are scripted
    # to isolate plateau selection (tie with lower loss must retain report best).
    import experiments.deit_protocol as shared
    import experiments.train_deit_plateau as runner
    images = torch.randn(6, 3, 32, 32)
    labels = torch.arange(6)
    dataset = TensorDataset(images, labels)
    monkeypatch.setattr(shared, "datasets_and_indices", lambda *_args:
        (dataset, dataset, [0, 1], [2, 3, 4, 5], []))
    measurements = iter([{"accuracy": .1, "loss": 5.}, {"accuracy": .2, "loss": 4.},
                         {"accuracy": .2, "loss": 3.9}, {"accuracy": .1, "loss": 3.8}])
    monkeypatch.setattr(runner, "evaluate_without_rng", lambda *_args: next(measurements))
    output = tmp_path / "vanilla"
    arguments = ["train_deit_plateau", "--data-root", "unused", "--output", str(output),
        "--device", "cpu", "--workers", "0", "--batch-size", "2",
        "--validation-samples", "4", "--trigger-samples", "2", "--tuning-samples", "0",
        "--min-plateau-epoch", "1", "--patience", "2", "--max-epoch", "3",
        "--schedule-epochs", "8", "--warmup-epochs", "1"]
    monkeypatch.setattr(sys, "argv", arguments)
    runner.main()
    fork, recipe = checked_source(output / "plateau_checkpoint.pt", {"deit_plateau_fork"})
    best = torch.load(output / "checkpoint_best.pt", weights_only=False)
    latest = torch.load(output / "checkpoint_latest.pt", weights_only=False)
    assert fork["epoch"] == 1
    assert fork["stall_detected_epoch"] == 3
    assert fork["report_best_accuracy"] == .2
    assert fork["report_best_loss"] == 4.
    assert fork["history"][-1]["validation_loss"] == 4.  # paired report loss
    assert all(torch.equal(value, best["model"][key]) for key, value in fork["model"].items())
    assert fork["scheduler"]["last_epoch"] == 1
    assert latest["scheduler"]["last_epoch"] == 3
    assert any(not torch.equal(value, latest["model"][key]) for key, value in fork["model"].items())
    # A completed Vanilla resume must finalize immediately without extra SGD.
    monkeypatch.setattr(sys, "argv", arguments + ["--resume", str(output / "checkpoint_latest.pt")])
    monkeypatch.setattr(runner, "train_epoch", lambda *_args: pytest.fail("resumed completed plateau trained again"))
    runner.main()
    assert torch.load(output / "plateau_checkpoint.pt", weights_only=False)["epoch"] == 1
