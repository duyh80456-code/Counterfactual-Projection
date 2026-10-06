import json

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from scripts.analyze_projection_diagnostic import (
    load_interventions, spearman_bootstrap, summarize_sites, write_outputs)


def test_load_logs_keys_mapping_and_no_per_site_residual(tmp_path, capsys):
    folder = tmp_path / "vgg_seed1" / "ours_e_driven_o"
    folder.mkdir(parents=True)
    event = {"epoch": 302, "probe_index": 1, "selected_site": "boundary_to_4",
             "heldout_relative_residual": 0.7, "heldout_cosine_alignment": 0.8,
             "correction_applied": True, "actual_loss_improvement": 0.02,
             "site_evaluations": {"other": {"mean_e_gain": 0.2}}}
    result = {"method": "ours_e_driven_o", "protocol": {"architecture": "CIFAR-VGG16-BN"},
              "interventions": [event], "intervention": event, "history": []}
    (folder / "result.json").write_text(json.dumps(result))
    (folder / "history.jsonl").write_text(json.dumps({"e_driven_o_intervention": event}) + "\n")
    df = load_interventions(tmp_path, ["**/*.json", "**/*.jsonl"])
    assert len(df) == 1
    assert df.iloc[0].backbone == "VGG"
    assert df.iloc[0].seed == 1
    assert df.iloc[0].r_heldout == 0.7
    assert "heldout_relative_residual" in capsys.readouterr().out
    assert summarize_sites(df).iloc[0]["count"] == 1
    write_outputs(df, tmp_path / "analysis", bootstrap=10)
    assert (tmp_path / "analysis" / "interventions.csv").exists()
    assert (tmp_path / "analysis" / "heldout_distributions.png").exists()


def test_explicit_mapping_metadata_missing_and_non_e_branch(tmp_path):
    event = {"selected_site": "site", "correction_applied": False,
             "custom_residual": 0.4, "actual_loss_improvement": 0.0}
    (tmp_path / "history.json").write_text(json.dumps([event]))
    (tmp_path / "o_only.json").write_text(json.dumps({
        "method": "o_projection_only", "interventions": [event]}))
    df = load_interventions(
        tmp_path, ["*.json"], mapping={"r_heldout": ["custom_residual"]},
        metadata_overrides={"history.json": {"method": "ours_e_driven_o",
                                            "backbone": "R18", "seed": 0}})
    assert len(df) == 1
    assert df.iloc[0].r_heldout == 0.4
    assert pd.isna(df.iloc[0].cos_heldout)
    assert not df.iloc[0].applied


def test_spearman_bootstrap_ties_constants_and_small_n():
    df = pd.DataFrame({"r": [1, 2, 2, 4, 5], "gain": [5, 4, 4, 2, 1]})
    result = spearman_bootstrap(df, "r", "gain", bootstrap=100, seed=4)
    assert result["rho"] == pytest.approx(-1)
    assert result["ci_low"] == pytest.approx(-1)
    assert result["status"] == "small_n_descriptive_only"
    assert result == spearman_bootstrap(df, "r", "gain", bootstrap=100, seed=4)
    df["gain"] = 1
    assert spearman_bootstrap(df, "r", "gain")["rho"] is None
    assert spearman_bootstrap(df.iloc[:2], "r", "gain")["n"] == 2


def test_site_probe_real_projector_and_norm_matched_random_control():
    from contextlib import contextmanager
    from types import SimpleNamespace
    import torch
    from torch import nn
    from diagnostics.site_projection_probe import site_projection_probe
    from projection import FunctionalProjector

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(3, 2, bias=False)
            self.gate = 0.0

        def forward(self, inputs):
            return self.layer(inputs) + self.gate * inputs[:, :2]

        def projection_parameter_modules(self, _site):
            return {"residual_path": [self.layer]}

    model = Model()
    @contextmanager
    def direction(gate):
        model.gate = gate
        try:
            yield
        finally:
            model.gate = 0.0

    candidate = SimpleNamespace(module_name="layer", virtual_direction=direction,
                                payload={})
    generator = torch.Generator().manual_seed(4)
    fit = (torch.randn(64, 3, generator=generator), torch.zeros(64, dtype=torch.long))
    heldout = (torch.randn(256, 3, generator=generator), torch.zeros(256, dtype=torch.long))
    result = site_projection_probe(model, candidate, fit, heldout,
        projector=FunctionalProjector(damping=1e-7, max_iter=30, tolerance=1e-6))
    assert result["true_direction"]["r_E_heldout"] < 1e-3
    assert result["true_direction"]["cos_E_heldout"] > 0.99
    assert result["random_control"]["r_E_heldout"] > 0.5
    assert result["true_direction"]["fit_target_norm"] == pytest.approx(
        result["random_control"]["fit_target_norm"], rel=1e-5)
    assert model.gate == 0
    with pytest.raises(ValueError, match="256"):
        site_projection_probe(model, candidate, fit, fit)


def test_persistent_growth_trains_registered_extension_with_matched_stream(tmp_path, monkeypatch):
    import copy
    from contextlib import contextmanager
    from types import SimpleNamespace
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset
    import diagnostics.persistent_growth_probe as pg

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(3, 2)
            self.extension = None
            self.gate = 0.0

        def forward(self, inputs):
            value = self.base(inputs)
            return value if self.extension is None else value + self.gate * self.extension(inputs)

    generator = torch.Generator().manual_seed(2)
    dataset = TensorDataset(torch.randn(8, 3, generator=generator),
                            torch.tensor([0, 1] * 4))
    source = {"train_indices": list(range(8)), "evaluation_indices": list(range(8)),
              "rng": pg.rng_state(), "train_loader_generator_state": generator.get_state(),
              "protocol": {"seed": 1}}
    initial = Model().state_dict()
    contexts = []
    def context(*_args):
        model = Model()
        model.load_state_dict(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        result = SimpleNamespace(model=model, optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=100),
            source=source, train_set=dataset, eval_set=dataset,
            device=torch.device("cpu"), checkpoint_hash="test")
        contexts.append(result)
        return result
    monkeypatch.setattr(pg, "load_context", context)
    batch = tuple(t for t in dataset.tensors)
    monkeypatch.setattr(pg, "select_batches", lambda *_args, **_kwargs: (
        {"statistics": [batch], "where": batch}, {"where": list(range(8))}))
    weights_before_after = []
    def propose(model, *_args):
        @contextmanager
        def direction(scale):
            model.extension = nn.Linear(3, 2, bias=False)
            nn.init.constant_(model.extension.weight, 0.1)
            model.gate = scale
            before = model.extension.weight.detach().clone()
            try:
                yield
            finally:
                weights_before_after.append((before, model.extension.weight.detach().clone()))
                model.extension = None
        return [SimpleNamespace(virtual_direction=direction)]
    monkeypatch.setattr(pg, "propose", propose)
    result = pg.persistent_growth_gain("checkpoint", "site", 2,
        data_root="unused", reference_root="unused", architecture="resnet18",
        device="cpu", scales=(1.0,), output=tmp_path)
    assert result["persistent_growth"]["parameters"] > result["vanilla"]["parameters"]
    assert result["PG_gain"] == pytest.approx(
        result["persistent_growth"]["best_accuracy"] - result["vanilla"]["best_accuracy"])
    assert not torch.equal(*weights_before_after[-1])
    growth = torch.load(tmp_path / "growth_final.pt", weights_only=False)
    vanilla = torch.load(tmp_path / "vanilla_final.pt", weights_only=False)
    assert "extension.weight" in growth["model"]
    assert "extension.weight" not in vanilla["model"]
    assert torch.equal(growth["train_loader_generator_state"],
                       vanilla["train_loader_generator_state"])


def test_probe_data_splits_are_disjoint_and_heldout_has_256():
    from types import SimpleNamespace
    import torch
    from torch.utils.data import TensorDataset
    from diagnostics.protocol import select_batches
    context = SimpleNamespace(source={"train_indices": list(range(800))},
        eval_set=TensorDataset(torch.zeros(800, 3), torch.zeros(800, dtype=torch.long)),
        device=torch.device("cpu"))
    batches, indices = select_batches(context)
    flattened = [index for split in indices.values() for index in split]
    assert len(flattened) == len(set(flattened))
    assert len(batches["heldout"][0]) == 256
    assert len(batches["projection"][0]) == 64


def test_capacity_summary_joins_fork_and_keeps_seed_correlations_separate(tmp_path):
    from diagnostics.summarize_capacity import summarize_capacity
    projections, growths = [], []
    for seed in (0, 1):
        probe = {"checkpoint_hash": str(seed), "architecture": "resnet18", "seed": seed,
            "records": [{"site": f"s{i}", "true_direction": {
                "r_E_heldout": i / 3, "cos_E_heldout": 1 - i / 3},
                "random_control": {"r_E_heldout": 1.0}} for i in range(3)]}
        growth = {"records": [{"checkpoint_hash": str(seed), "architecture": "resnet18",
            "seed": seed, "site": f"s{i}", "PG_gain": i if seed == 0 else -i,
            "PG_loss_gain": 0.0, "horizon": 20} for i in range(3)]}
        p, g = tmp_path / f"p{seed}.json", tmp_path / f"g{seed}.json"
        p.write_text(json.dumps(probe))
        g.write_text(json.dumps(growth))
        projections.append(p)
        growths.append(g)
    merged, correlations = summarize_capacity(projections, growths, bootstrap=10)
    assert len(merged) == 6
    residual = correlations[correlations.x == "r_E_heldout"].set_index("seed")
    assert residual.loc[0, "rho"] == pytest.approx(1)
    assert residual.loc[1, "rho"] == pytest.approx(-1)


def test_kaggle_diagnostic_notebook_compiles_and_discovers_requested_runs(tmp_path, monkeypatch):
    from pathlib import Path
    import experiments.kaggle_checkpoint_discovery as discovery
    notebook = json.loads(Path(
        "notebooks/kaggle_projection_capacity_diagnostics_t4x2.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"diagnostics-cell-{index}", "exec")
    forks = []
    required = {"model", "optimizer", "scheduler", "rng",
                "train_loader_generator_state", "train_indices", "trigger_indices",
                "evaluation_indices", "source_tuning_indices", "epoch"}
    requested = {(architecture, seed) for architecture in ("resnet18", "resnet34")
                 for seed in (1, 2, 3)} | {("vgg16", 1)}
    labels = {"resnet18": "CIFAR-ResNet18", "resnet34": "CIFAR-ResNet34",
              "vgg16": "CIFAR-VGG16-BN"}
    for architecture, seed in sorted(requested | {("vgg16", 0)}):
        payload = dict.fromkeys(required)
        payload.update(epoch=243, protocol={"architecture": labels[architecture], "seed": seed})
        forks.append({"path": Path(f"{architecture}-{seed}.pt"), "payload": payload,
                      "sha256": f"{architecture}-{seed}"})
    monkeypatch.setattr(discovery, "discover_checkpoints",
                        lambda *_args, **_kwargs: (forks + [forks[0]], []))
    cell = next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and
                "# Discover original forks" in "".join(cell["source"]))
    import sys
    env = {"sys": sys, "json": json, "REPO": Path.cwd(),
           "RUN_ROOT": tmp_path, "RUN_FILTER": requested,
           "TOP_SITES": 3, "BOTTOM_SITES": 2, "HORIZON": 20}
    exec(cell, env)
    assert {(run["architecture"], run["seed"]) for run in env["RUNS"]} == requested
    assert len(env["RUNS"]) == 7


def test_log_cpu_provenance_boundary_horizons_and_run_separation(tmp_path):
    from scripts.analyze_projection_diagnostic import summarize_cosine_groups, run_correlations
    for name, sign in (("unlabelled_a", 1), ("unlabelled_b", -1)):
        folder = tmp_path / name
        folder.mkdir()
        interventions, history = [], []
        for index, epoch in enumerate((100, 120, 140, 160)):
            interventions.append({"epoch": epoch, "probe_index": index,
                "selected_site": "stages.3.boundary_to_4" if index < 2 else "stages.3.conv_0",
                "heldout_cosine_alignment": .2 * (index + 1),
                "heldout_relative_residual": .9 - .1 * index,
                "actual_cosine_alignment": .1 * (index + 1),
                "actual_relative_residual": .8,
                "actual_loss_improvement": sign * (index + 1),
                "cg_selected_attempt": 1,
                "cg_attempts": [{"functional_cosine_alignment": -1},
                                {"functional_cosine_alignment": .95,
                                 "functional_relative_residual": .05}],
                "correction_applied": True})
            history += [{"epoch": epoch + h, "validation_accuracy": 70 + sign * index + h / 100}
                        for h in (0, 1, 5, 15)]
        (folder / "result.json").write_text(json.dumps({
            "method": "ours_e_driven_o", "theta_best_hash": "same-fork",
            "interventions": interventions, "history": history}))
    df = load_interventions(tmp_path, ["**/result.json"], metadata_rules=[
        {"pattern": "unlabelled_*/result.json", "metadata": {"backbone": "VGG", "seed": 1}}])
    assert set(df.backbone) == {"VGG"}
    assert df.run_id.nunique() == 2  # same fork must not merge independent runs
    assert set(df.r_fit) == {.05}
    assert set(df.cos_fit_source) == {"selected_cg_attempt.functional_cosine_alignment"}
    assert df.is_boundary.sum() == 4
    row = df.iloc[0]
    assert row.acc_after_5 == pytest.approx(70.05)
    assert row.acc_gain_15 == pytest.approx(.15)
    stats = summarize_cosine_groups(df)
    boundary = stats[(stats.run_id == "unlabelled_a") & (stats.scope == "is_boundary")
                     & (stats.group == "True")].iloc[0]
    assert boundary.cos_heldout_q25 == pytest.approx(.25)
    assert boundary.cos_heldout_median == pytest.approx(.3)
    assert boundary.cos_heldout_q75 == pytest.approx(.35)
    correlations = run_correlations(df, bootstrap=10)
    selected = correlations[(correlations.x == "cos_heldout") &
                            (correlations.y == "realized_gain") &
                            (correlations.subset == "all")]
    assert dict(zip(selected.run_id, selected.rho)) == pytest.approx({"unlabelled_a": 1, "unlabelled_b": -1})
    endpoints = correlations[(correlations.x == "cos_heldout") &
                             (correlations.y == "acc_after_15") &
                             (correlations.subset == "all")]
    assert set(endpoints.n) == {4}
    write_outputs(df, tmp_path / "analysis", bootstrap=10)
    for filename in ("metric_sources.csv", "cosine_by_site_boundary.csv", "run_spearman.csv",
                     "backbone_quartiles.csv", "unknown_metadata.csv"):
        assert (tmp_path / "analysis" / filename).is_file()


def test_log_horizons_missing_conflicting_and_intervening_events(tmp_path):
    events = [{"epoch": epoch, "selected_site": "boundary_to_4",
               "correction_applied": True, "heldout_cosine_alignment": .8}
              for epoch in (100, 105)]
    history = [{"epoch": 100, "validation_accuracy": 70},
               {"epoch": 101, "validation_accuracy": 71},
               {"epoch": 101, "validation_accuracy": 72},
               {"epoch": 105, "validation_accuracy": 73},
               {"epoch": 115, "validation_accuracy": 74}]
    (tmp_path / "result.json").write_text(json.dumps({"method": "ours_e_driven_o",
        "interventions": events, "history": history}))
    df = load_interventions(tmp_path, ["*.json"])
    first = df.iloc[0]
    assert first.acc_1_status == "conflicting_history"
    assert pd.isna(first.acc_after_1)
    assert first.acc_5_status == "another_intervention_in_window"
    assert first.acc_15_status == "another_intervention_in_window"
    assert df.iloc[1].acc_1_status == "missing_epoch"
    assert set(df.backbone) == {"unknown"}
    assert df.is_boundary.isna().all()


def test_log_history_mapping_and_config_labels(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({
        "config": {"architecture": "resnet34", "seed": 3, "method": "ours_e_driven_o"},
        "interventions": [{"epoch": 200, "selected_site": "layer3", "correction_applied": False}],
        "history": [{"step": 200, "metrics": {"acc": .7}},
                    {"step": 201, "metrics": {"acc": .72}}]}))
    df = load_interventions(tmp_path, ["*.json"], history_mapping={
        "epoch": "step", "accuracy": "metrics.acc"})
    assert df.iloc[0].backbone == "R34"
    assert df.iloc[0].seed == 3
    assert df.iloc[0].acc_gain_1 == pytest.approx(.02)


def test_cpu_logs_follow_kaggle_symlinks_and_keep_glob_depth(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "result.json").write_text(json.dumps({
        "method": "ours_e_driven_o", "backbone": "unknown", "seed": 1,
        "interventions": [{"selected_site": "boundary_to_4", "correction_applied": False}]}))
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "mounted-vgg").symlink_to(actual, target_is_directory=True)
    df = load_interventions(inputs, ["**/result.json"], metadata_rules=[
        {"pattern": "mounted-vgg/**", "metadata": {"backbone": "VGG"}}])
    assert len(df) == 1
    assert df.iloc[0].backbone == "VGG"
    assert df.iloc[0].is_boundary
    assert load_interventions(inputs, ["*.json"]).empty
