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
