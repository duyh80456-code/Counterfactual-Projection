"""Stage 1 Gromo alignment, matrix-free math, invariance and CLI smoke tests."""
from dataclasses import replace
from pathlib import Path
import copy

import pytest
import torch
from torch.func import functional_call

from experiments.vna_support import tiny_model, tiny_candidate, isolated_rng
from methods.vna import (VNAConfig, dedicated_generator, extract_tiny_A, random_A_like,
    vna_block_correction, vna_lite_correction, block_products, _mean_ce_logit_grad, _virtual_b_context,
    _resolve_site, assert_disjoint_scopes)
from projection.functional import _preserve_functional_state


@pytest.fixture
def case():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    generator = dedicated_generator(42, "case")
    with isolated_rng(generator):
        model = tiny_model()
    batch = (torch.randn(2, 3, 8, 8, generator=generator), torch.tensor([0, 1]))
    site = model.growing_blocks()[0].name
    candidate = tiny_candidate(model, site, batch, 2)
    A, rank = extract_tiny_A(model, candidate)
    yield model, site, batch, candidate, A
    torch.set_num_threads(old)


def vector(values, refs):
    return torch.cat([values[name].reshape(-1) for name, _ in refs])


def state(model):
    return ({n: t.clone() for n, t in model.state_dict().items()},
            {n: None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()},
            [m.training for m in model.modules()])


def assert_state(model, before):
    after = state(model)
    assert after[2] == before[2]
    assert after[0].keys() == before[0].keys()
    for n, t in before[0].items():
        assert torch.equal(t, after[0][n]), n
    for n, t in before[1].items():
        assert (after[1][n] is None) if t is None else torch.equal(t, after[1][n])


def test_extension_finite_difference_adjoint_and_native_alignment(case):
    model, site, batch, candidate, A = case
    model.eval()
    with candidate.virtual_direction(1.):
        B = model.block(site).second_layer.extended_input_layer.weight.detach().clone()
        with torch.no_grad():
            native = model(batch[0]).clone()
    _, block, _ = _resolve_site(model, site)
    with _preserve_functional_state(model), _virtual_b_context(model, block, A):
        with torch.no_grad():
            block.second_layer.extended_input_layer.weight.copy_(B)
            assert torch.equal(model(batch[0]), native)
    with block_products(model, site, A, batch[0]) as p:
        generator = dedicated_generator(0, "fd")
        v = torch.randn(sum(r.numel() for _,r in p.extension_refs), generator=generator)
        v /= v.norm()
        image = p.JEv(v)
        # A double-precision finite-difference oracle avoids cancellation of
        # fp32 logits when native A has very small norm.
        oracle = copy.deepcopy(model).double()
        double_A = {n: None if t is None else t.double() for n,t in A.items()}
        with block_products(oracle, site, double_A, batch[0].double()) as op:
            epsilon = 1e-4
            refs = op.extension_refs
            chunks = v.double().split([r.numel() for _,r in refs])
            plus = op._call({n:r+epsilon*c.reshape_as(r) for (n,r),c in zip(refs,chunks)})
            minus = op._call({n:r-epsilon*c.reshape_as(r) for (n,r),c in zip(refs,chunks)})
            assert torch.allclose(image.double(), (plus-minus)/(2*epsilon), rtol=1e-3, atol=1e-9)
        u = torch.randn(image.shape, generator=generator)
        assert torch.allclose(u.dot(image), p.JET(u).dot(v), rtol=1e-5, atol=1e-7)


def test_use_a_only_and_random_norm(case):
    model, site, batch, candidate, A = case
    cfg = VNAConfig(cg_iters=70, cg_tol=1e-5)
    d, diag = vna_block_correction(model, site, A, batch, cfg)
    # A modified native B affects the expanded candidate but never extracted A.
    original = candidate.virtual_direction
    from contextlib import contextmanager
    @contextmanager
    def changed(gate):
        with original(gate):
            with torch.no_grad():
                model.block(site).second_layer.extended_input_layer.weight.fill_(123.)
            yield
    candidate.virtual_direction = changed
    altered, rank = extract_tiny_A(model, candidate)
    d2, _ = vna_block_correction(model, site, altered, batch, cfg)
    assert all(torch.equal(d[n], d2[n]) for n in d)
    assert rank == candidate.payload["effective_rank"]
    assert candidate.payload["counterfactual_target_width"] == model.block(site).hidden_neurons + rank
    control = random_A_like(A, generator=dedicated_generator(7, "A"))
    assert control["weight"].shape == A["weight"].shape
    assert torch.allclose(control["weight"].norm(), A["weight"].norm(), rtol=1e-6)
    assert not torch.equal(control["weight"], A["weight"])


@pytest.mark.parametrize("bn_mode", ["eval", "train"])
def test_invariance_rng_scope_warm_and_nonfinite(case, bn_mode):
    model, site, batch, candidate, A = case
    model.train()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = state(model)
    rng = torch.get_rng_state().clone()
    loader_generator = dedicated_generator(77, "training-loader")
    loader_state = loader_generator.get_state().clone()
    random_A_like(A, generator=dedicated_generator(2, "random"))
    cfg = VNAConfig(bn_mode=bn_mode, cg_iters=60, cg_tol=1e-5, lambda_vna=.7)
    d, diag = vna_block_correction(model, site, A, batch, cfg)
    assert not diag["nonfinite"]
    assert set(d) == set(_resolve_site(model, site)[2])
    assert all("downsample" not in n for n in d)
    assert_state(model, before)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(loader_state, loader_generator.get_state())
    _, warm = vna_block_correction(model, site, A, batch, cfg, diag["warm_start"], damping_cache=diag["damping_cache"])
    assert warm["cg_z"]["iterations"] <= diag["cg_z"]["iterations"]
    assert_state(model, before)
    bad = {"weight": torch.full_like(A["weight"], float("nan")), "bias": A["bias"]}
    d, bad_diag = vna_block_correction(model, site, bad, batch, cfg)
    assert not d and bad_diag["nonfinite"]
    assert_state(model, before)
    assert_disjoint_scopes(model, [site])
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_scopes(model, [site,site])


def test_dense_chain_full_rank_limit_and_exact_gain(case):
    model, site, batch, candidate, A = case
    with block_products(model, site, A, batch[0]) as p:
        logits = p._call({}).reshape(2, -1)
        g = _mean_ce_logit_grad(logits, batch[1])
        eye = torch.eye(g.numel())
        J = torch.stack([p.JT(u) for u in eye]).double()
        E = torch.stack([p.JET(u) for u in eye]).double()
        refs = p.parameter_refs
    K, KE = J @ J.T, E @ E.T
    assert torch.linalg.matrix_rank(K) == g.numel()
    cfg = VNAConfig(damping_rel=1e-2, cg_iters=200, cg_tol=1e-6, lambda_vna=.7)
    cache = {"mu": .01*float(K.trace()/len(K)), "jacobi": K.diag()}
    d, diag = vna_block_correction(model, site, A, batch, cfg, damping_cache=cache)
    R = torch.linalg.inv(K + cache["mu"] * torch.eye(len(K)))
    expected = cfg.lambda_vna * J.T @ R @ KE @ R @ K @ g
    flat = vector(d, refs).double()
    assert torch.allclose(flat, expected, rtol=2e-3, atol=2e-7)
    assert float(g @ J @ flat) == pytest.approx(cfg.lambda_vna*diag["V"], rel=2e-4, abs=1e-9)
    assert sum(diag["per_neuron_V"]) == pytest.approx(diag["V"], rel=1e-6)
    assert diag["U"] >= 0 and diag["eta"] >= 0
    # With damping tending to zero, the combined SGD+VNA function update.
    cfg = replace(cfg, lambda_vna=1., damping_rel=0.)
    d, diag = vna_block_correction(model, site, A, batch, cfg, damping_cache={"mu":0., "jacobi":K.diag()})
    delta = -(J.T @ g + vector(d, refs).double())
    target = -(K + KE) @ g
    assert float((J @ delta - target).norm()/target.norm()) < 1e-2


def test_descent_fifty_random_batches(case):
    model, site, _, _, A = case
    generator = dedicated_generator(11, "50-batches")
    cfg = VNAConfig(cg_iters=100, cg_tol=1e-5)
    for _ in range(50):
        batch = (torch.randn(2,3,8,8,generator=generator), torch.randint(3,(2,),generator=generator))
        d, diag = vna_block_correction(model, site, A, batch, cfg)
        assert not diag["nonfinite"]
        with block_products(model, site, A, batch[0]) as p:
            g = _mean_ce_logit_grad(p._call({}).reshape(2,-1),batch[1])
            gain = float(g @ p.Jv(vector(d,p.parameter_refs)).double())
        assert gain >= -1e-9
        assert gain == pytest.approx(cfg.lambda_vna * diag["V"], rel=3e-3, abs=1e-8)


def test_gromo_statistics_sample_normalization_and_transaction(case):
    model, site, batch, _, A = case
    model.train()
    before = state(model)
    c1 = tiny_candidate(model,site,batch,1)
    repeated = (batch[0].repeat(4,1,1,1),batch[1].repeat(4))
    c2 = tiny_candidate(model,site,repeated,1)
    assert c2.payload["tiny_eigenvalues"][0] == pytest.approx(c1.payload["tiny_eigenvalues"][0], rel=.03)
    assert_state(model,before)


def test_no_forbidden_paths():
    import re
    root = Path(__file__).resolve().parents[1]
    for name in ["methods/vna.py", "experiments/train_recipe_checkpoints.py", "experiments/vna_support.py", "diagnostics/signal_curve.py"]:
        source = (root/name).read_text()
        assert not re.search(r"\b(?:apply_|reset_projected_momentum|plateau_checkpoint|rollback|controller_anchor)\b",source)


def test_recipe_resume_and_signal_smoke(tmp_path):
    from experiments.train_recipe_checkpoints import parser, run
    from diagnostics.signal_curve import parser as signal_parser, run as signal_run
    common = ["--fake-data","--limit-batches","1","--seed","9"]
    straight = run(parser().parse_args(common+["--output",str(tmp_path/"straight"),"--max-epochs","2"]))
    first = run(parser().parse_args(common+["--output",str(tmp_path/"resumed"),"--max-epochs","1"]))
    for flag in ("gpu-data", "channels-last", "cudnn-benchmark"):
        with pytest.raises(ValueError, match=flag.replace("-", "_") + ": saved=False, requested=True"):
            run(parser().parse_args(common+["--output", str(tmp_path/"resumed"),
                "--max-epochs", "2", "--resume", str(first), "--" + flag]))
    resumed = run(parser().parse_args(common+["--output",str(tmp_path/"resumed"),"--max-epochs","2","--resume",str(first)]))
    a,b = [torch.load(p,weights_only=False) for p in [straight,resumed]]
    assert all(torch.equal(a["model"][n],b["model"][n]) for n in a["model"])
    assert torch.equal(a["rng"]["torch"],b["rng"]["torch"])
    assert torch.equal(a["train_loader_generator_state"],b["train_loader_generator_state"])
    assert a["scheduler"] == b["scheduler"]
    for pid, values in a["optimizer"]["state"].items():
        assert torch.equal(values["momentum_buffer"],b["optimizer"]["state"][pid]["momentum_buffer"])
    rows = signal_run(signal_parser().parse_args(["--fake-data","--smoke","--channels-last","--cudnn-benchmark",
        "--diagnostic-stride","1","--checkpoints",str(resumed),"--output",str(tmp_path/"signal")]))
    assert len(rows) == 5 and all(not r["nonfinite"] for r in rows)
    assert {r["requested_rank"] for r in rows if r["row_kind"] == "sweep"} == {3,4,16,64}
    assert all(r["sgd_batch_size"] == 64 for r in rows)
    assert all(len(r["lambda_shuffled_values"]) == 3 for r in rows)
    assert all(r["stat_samples"] == r["shuffled_samples"] == 16 for r in rows)
    assert all(r["lite_realized_gain"] >= 0 and r["random_apply_gain_guard"] is False for r in rows)
    assert all(r["significant_vs_shuffled"] == (r["lambda_full"] > max(r["lambda_shuffled_values"])) for r in rows)
    assert all(r["significant_split_half"] == (r["lambda_split_half"] > max(r["lambda_shuffled_values"])) for r in rows)
    assert all(r["rank"] == r["random_rank"] <= r["requested_rank"] for r in rows)
    assert rows[0]["per_neuron_V"] and rows[0]["lambda_split_half"] >= 0
    assert "lr_boost_derivative" in rows[0]
    assert (tmp_path/"signal"/"signal_curve.csv").exists()
    assert (tmp_path/"signal"/"summary.json").exists()


@pytest.mark.parametrize("bn_mode", ["eval", "train"])
def test_cached_pullbacks_match_old_path(case, bn_mode):
    from torch.func import vjp
    model, site, batch, _, A = case
    with block_products(model, site, A, batch[0], bn_mode) as p:
        u = torch.arange(6, dtype=torch.float32)
        for refs, method in ((p.parameter_refs, p.JT), (p.extension_refs, p.JET)):
            _, old = vjp(p._function_for(refs), *(r for _, r in refs))
            expected = torch.cat([r.detach().reshape(-1) for r in old(u)])
            assert torch.equal(method(u), expected)
            forwards = p.forward_calls
            assert torch.equal(method(u), expected)
            assert p.forward_calls == forwards  # application, no new forward
        assert p.vjp_calls == 2 and p.extension_vjp_calls == 2


def test_nonpositive_realized_gain_is_skipped(case, monkeypatch):
    import methods.vna as core
    model, site, batch, _, A = case
    original = core.conjugate_gradient
    calls = 0
    def negative_q(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        return replace(result, solution=-result.solution) if calls == 2 else result
    monkeypatch.setattr(core, "conjugate_gradient", negative_q)
    correction, diag = vna_block_correction(model, site, A, batch, VNAConfig(cg_iters=100, cg_tol=1e-5))
    assert not correction
    assert diag["realized_gain"] < 0 and diag["nonpositive_gain"] and diag["skipped"]
    assert diag["skip_reason"] == "nonpositive_realized_gain"


def test_exact_cached_suffix_and_outer_autocast(case):
    model, site, batch, _, A = case
    outputs = []
    for cached in (False, True):
        with block_products(model, site, A, batch[0], cache_block_input=cached) as p:
            logits = p._call({})
            u = torch.arange(logits.numel(), dtype=torch.float32)
            jt, jet = p.JT(u), p.JET(u)
            outputs.append((logits, jt, jet, p.Jv(jt), p.JEv(jet)))
    assert all(torch.equal(a, b) for a, b in zip(*outputs))
    cfg = VNAConfig(cg_iters=100, cg_tol=1e-5, cache_block_input=True)
    expected, expected_diag = vna_block_correction(model, site, A, batch, cfg)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, diag = vna_block_correction(model, site, A, batch, cfg)
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[k], actual[k]) and actual[k].dtype == torch.float32 for k in expected)
    assert expected_diag["realized_gain"] == diag["realized_gain"]


def test_device_augmentation_paired_streams_and_resume():
    from types import SimpleNamespace
    import numpy as np
    from experiments.vna_support import DeviceCIFAR, DeviceTrainLoader, augmented_batch
    dataset = SimpleNamespace(data=np.arange(12*32*32*3, dtype=np.uint8).reshape(12,32,32,3), targets=list(range(12)))
    resident = DeviceCIFAR(dataset, "cpu")
    a = DeviceTrainLoader(resident, list(range(12)), 4, 91)
    b = DeviceTrainLoader(resident, list(range(12)), 4, 91)
    rng = torch.get_rng_state().clone()
    for first, second in zip(a, b):
        assert all(torch.equal(x, y) for x, y in zip(first, second))
    saved = a.generator.get_state()
    resumed = DeviceTrainLoader(resident, list(range(12)), 4, 91, saved)
    for first, second in zip(a, resumed):
        assert all(torch.equal(x, y) for x, y in zip(first, second))
    batch1 = augmented_batch(resident, list(range(12)), 4, dedicated_generator(9,"statistics"), "cpu")
    batch2 = augmented_batch(resident, list(range(12)), 4, dedicated_generator(9,"statistics"), "cpu")
    assert all(torch.equal(x, y) for x, y in zip(batch1, batch2))
    assert torch.equal(rng, torch.get_rng_state())


def test_watch_markers_shards_merge_resume_and_archive(tmp_path, monkeypatch):
    from experiments.train_recipe_checkpoints import parser as train_parser, run as train_run
    from diagnostics.signal_curve import parser, run, merge
    import diagnostics.signal_curve as signal
    from experiments.vna_kaggle import archive_outputs, restore_outputs
    import json
    recipe = tmp_path / "original" / "recipe"
    latest = train_run(train_parser().parse_args(["--fake-data", "--limit-batches", "1", "--max-epochs", "1", "--output", str(recipe)]))
    payload = torch.load(latest, weights_only=False)
    for epoch in (10,20):
        periodic = {k: payload[k] for k in ("kind", "epoch", "model", "protocol")}
        periodic["epoch"] = epoch
        torch.save(periodic, recipe/f"epoch_{epoch:03d}.pt")
    # Expensive diagnostics already covered by integration smoke; use transactions
    # with controlled rows to test scheduling and stop/restart without duplication.
    calls = []
    def compute(args):
        epoch = torch.load(args.checkpoints[0], weights_only=False)["epoch"]
        calls.append(epoch)
        if epoch == 20:
            stop.touch()
        return [{"epoch":epoch, "site":"test", "nonfinite":True}]
    monkeypatch.setattr(signal, "compute", compute)
    output = tmp_path/"original"/"signal"
    stop = tmp_path/"stop"
    base = ["--fake-data","--smoke","--diagnostic-stride","10","--checkpoint-dir",str(recipe),"--output",str(output)]
    run(parser().parse_args(base+["--watch","--stop-file",str(stop),"--no-merge"]))
    assert calls == [10,20]
    run(parser().parse_args(base+["--watch","--stop-file",str(stop),"--no-merge"]))
    assert calls == [10,20]
    (output/"completed"/"epoch_010.json").unlink()
    (output/"completed"/"epoch_020.json").unlink()
    for shard in (0,1):
        run(parser().parse_args(base+["--checkpoints", str(recipe/"epoch_010.pt"), str(recipe/"epoch_020.pt"),
                                    "--shards","2","--shard",str(shard),"--no-merge"]))
    assert calls == [10,20,20,10]
    assert len(merge(parser().parse_args(base))) == 2
    for flag in ("gpu-data", "channels-last", "cudnn-benchmark", "cache-block-input"):
        changed = parser().parse_args(base+["--" + flag])
        for operation in (run, merge):
            with pytest.raises(ValueError, match=flag.replace("-", "_") + ": saved=False, requested=True"):
                operation(changed)
    archive = archive_outputs(tmp_path/"original")
    attached = tmp_path/"input"
    attached.mkdir()
    import shutil
    shutil.copy2(archive, attached/archive.name)
    restored = tmp_path/"restored"
    restore_outputs(attached, restored, payload["protocol"])
    assert (restored/"recipe"/"latest.pt").exists()
    assert len(list((restored/"recipe").glob("epoch_*.pt"))) == 2
    assert len(list((restored/"signal"/"completed").glob("*.json"))) == 2
    assert not (restored/"_resume_cache").exists()
    assert not (restored/"recipe"/"epoch_010.pt").is_symlink()
    for flag in ("gpu_data", "amp", "channels_last", "cudnn_benchmark"):
        changed = {**payload["protocol"], flag: True}
        for destination in (restored, tmp_path / ("mismatch_" + flag)):
            with pytest.raises(ValueError, match=flag + ": saved=False, requested=True"):
                restore_outputs(attached, destination, changed)
            assert not (destination/"_resume_cache").exists()


def test_notebook_branch_setup_and_syntax(tmp_path, monkeypatch):
    from scripts.build_kaggle_vna_signal_curve_notebook import build
    branch = "vna-phase0"
    notebook = build(branch)
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook", "exec")
    assert f'REPO_BRANCH = "{branch}"' in source
    assert 'private_clone(MAIN_URL, REPO, REPO_BRANCH)' in source
    assert "REPO_COMMIT" not in source and "code_commit.json" in source
    assert '"tests/test_vna.py"' in source and '"CUDA_VISIBLE_DEVICES": ""' in source
    assert '{"train", "test", "meta"}' in source
    assert '"--watch"' in source and '"--shards", "2"' in source
    import experiments.vna_kaggle as kaggle
    from experiments.train_recipe_checkpoints import parser as train_parser
    from diagnostics.signal_curve import parser as signal_parser
    protocols = []
    monkeypatch.setattr(kaggle, "restore_outputs", lambda input_root, output, protocol: protocols.append(protocol))
    import sys
    namespace = {"Path": Path, "RUN_ROOT": tmp_path, "DATA_ROOT": tmp_path/"data", "sys": sys}
    exec("".join(notebook["cells"][1]["source"]), namespace)
    exec("".join(notebook["cells"][4]["source"]), namespace)
    training = train_parser().parse_args(namespace["training"][4:])
    diagnostic = signal_parser().parse_args(namespace["diagnostic"][4:])
    for name in ("gpu_data", "channels_last", "cudnn_benchmark"):
        assert getattr(training, name) is True
        assert getattr(diagnostic, name) is True
        assert protocols[0][name] is True
    assert training.amp is False and protocols[0]["amp"] is False
    assert diagnostic.cache_block_input is True
    with pytest.raises(ValueError): build("--invalid")


def test_cached_suffix_after_earlier_blocks_and_shared_logits():
    from gromo.containers.resnet import init_full_resnet_structure
    from experiments.vna_support import tiny_model
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with isolated_rng(dedicated_generator(2, "deep-model")):
            model = tiny_model()
            model.core = init_full_resnet_structure(input_shape=(3,8,8), out_features=3,
                number_of_blocks_per_stage=1, inplanes=3, nb_stages=2, small_inputs=True,
                use_preactivation=False, device="cpu")
        site = model.growing_blocks()[-1].name
        batch = (torch.randn(2,3,8,8, generator=dedicated_generator(1,"inputs")), torch.tensor([0,1]))
        A = extract_tiny_A(model, tiny_candidate(model, site, batch, 2))[0]
        results = []
        for cached in (False, True):
            with block_products(model, site, A, batch[0], cache_block_input=cached) as p:
                logits = p._call({})
                g = _mean_ce_logit_grad(logits.reshape(2,-1), batch[1])
                jt, jet = p.JT(g), p.JET(g)
                results.append((logits, jt, jet, p.Jv(jt), p.JEv(jet)))
        assert all(torch.equal(x,y) for x,y in zip(*results))
        cfg = VNAConfig(cg_iters=100, cg_tol=1e-5, cache_block_input=True)
        d, diag = vna_block_correction(model,site,A,batch,cfg)
        d2, diag2 = vna_block_correction(model,site,A,batch,cfg,shared_logits=logits.reshape(2,-1),shared_g=g)
        assert all(torch.equal(d[k],d2[k]) for k in d)
        assert diag["V"] == diag2["V"] and diag["realized_gain"] == diag2["realized_gain"]
        assert diag["forward_calls"] == diag2["forward_calls"] + 1
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("bn_mode", ["eval", "train"])
def test_lite_dense_identity_finite_difference_invariance(case, bn_mode, monkeypatch):
    import methods.vna as core
    model, site, batch, _, A = case
    model.train()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = state(model)
    rng = torch.get_rng_state().clone()
    cfg = VNAConfig(bn_mode=bn_mode, lambda_vna=.7)
    with block_products(model, site, A, batch[0], bn_mode) as products:
        g = _mean_ce_logit_grad(products._call({}).reshape(2, -1), batch[1])
        eye = torch.eye(g.numel())
        J = torch.stack([products.JT(u) for u in eye]).double()
        E = torch.stack([products.JET(u) for u in eye]).double()
        refs = products.parameter_refs
    K, KE = J @ J.T, E @ E.T
    trace = float(K.trace())
    cache = {"trace_K": trace, "mu": .01*trace/len(K), "jacobi": K.diag()}
    monkeypatch.setattr(core, "conjugate_gradient", lambda *a, **k: pytest.fail("lite called CG"))
    correction, diag = vna_lite_correction(model, site, A, batch, cfg, damping_cache=cache)
    c = (trace / len(K))**2
    R = torch.eye(len(K), dtype=torch.float64) / c**.5
    expected = cfg.lambda_vna * J.T @ R @ KE @ R @ K @ g
    flat = vector(correction, refs).double()
    assert torch.allclose(flat, expected, rtol=2e-5, atol=1e-9)
    gain = float(g @ J @ flat)
    predicted = float(cfg.lambda_vna * (K@g) @ KE @ (K@g) / c)
    assert gain >= 0 and gain == pytest.approx(predicted, rel=2e-5, abs=1e-10)
    assert diag["realized_gain"] == pytest.approx(predicted, rel=2e-5, abs=1e-10)
    assert diag["V_lite"] == diag["V"] and diag["rho_lite"] == diag["rho"]
    assert sum(diag["per_neuron_V"]) == pytest.approx(diag["V"], rel=1e-6)
    assert diag["cg_z"] is None and diag["cg_q"] is None
    assert_state(model, before)
    assert torch.equal(rng, torch.get_rng_state())
    # Double forward oracle checks the actual returned parameter direction.
    oracle = copy.deepcopy(model).double()
    double_A = {n: None if v is None else v.double() for n,v in A.items()}
    with block_products(oracle, site, double_A, batch[0].double(), bn_mode) as products:
        epsilon = 1e-3
        delta = {n: correction[n].double() for n,_ in products.parameter_refs}
        plus = products._call({n: r + epsilon*delta[n] for n,r in products.parameter_refs})
        minus = products._call({n: r - epsilon*delta[n] for n,r in products.parameter_refs})
        image = products.Jv(vector(delta, products.parameter_refs)).double()
        assert torch.allclose(image, (plus-minus)/(2*epsilon), rtol=1e-3, atol=1e-9)
    assert_state(model, before)
    bad = {"weight": torch.full_like(A["weight"], float("nan")), "bias": A["bias"]}
    empty, bad_diag = vna_lite_correction(model, site, bad, batch, cfg)
    assert not empty and bad_diag["nonfinite"]
    assert_state(model, before)


def test_negative_random_control_is_retained_and_empty_excluded(case, monkeypatch):
    import methods.vna as core
    from diagnostics.signal_curve import heldout, criterion_results, rank_summary
    model, site, batch, _, A = case
    cfg = VNAConfig(cg_iters=100, cg_tol=1e-5)
    original = core.conjugate_gradient
    calls = 0
    def negative_q(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        return replace(result, solution=-result.solution) if calls % 2 == 0 else result
    monkeypatch.setattr(core, "conjugate_gradient", negative_q)
    random_A = random_A_like(A, generator=dedicated_generator(1, "forced-negative-random"))
    random, diag = vna_block_correction(model, site, random_A, batch, cfg, apply_gain_guard=False)
    assert random and diag["realized_gain"] < 0 and diag["nonpositive_gain"] and not diag["skipped"]
    measured = heldout(model, site, random_A, batch, {"random": {n:-v for n,v in random.items()}, "vna": {}}, cfg)
    assert measured["random_derivative"] > 0
    assert measured["vna_empty"] and measured["vna_derivative"] is None
    row = {"row_kind":"sweep", "requested_rank":4, "rank":2, "random_rank":2, "site":site, "epoch":20,
        "V": 10., "random_V":1., "nonfinite":False, "random_nonfinite":False,
        "vna_valid":True, "vna_empty":False, "vna_derivative":-1.,
        "random_valid":False, "random_empty":True, "random_derivative":0.,
        "lr_boost_valid":True, "lr_boost_empty":False, "lr_boost_derivative":-.5}
    assert criterion_results(row,"full")["derivative"] == (False,False)
    assert criterion_results(row,"full")["V"] == (False,False)
    table = rank_summary([row])[0]
    assert table["excluded_full_derivative"] == 1 and table["full_joint_rows"] == 0
    row.update(random_valid=True, random_empty=False, random_derivative=float("nan"))
    assert criterion_results(row,"full")["derivative"] == (False,False)
    row.update(random_derivative=2.)
    assert criterion_results(row,"full")["derivative"] == (True,False)


def test_stride20_watch_drain_shards_and_old_identity(tmp_path, monkeypatch):
    import json
    import diagnostics.signal_curve as signal
    from experiments.train_recipe_checkpoints import parser as train_parser, run as train_run
    checkpoint = train_run(train_parser().parse_args(["--fake-data", "--limit-batches", "1", "--max-epochs", "1", "--output", str(tmp_path/"recipe")]))
    payload = torch.load(checkpoint, weights_only=False)
    for epoch in (10,20,30,40):
        torch.save({**payload, "epoch":epoch}, tmp_path/"recipe"/f"epoch_{epoch:03d}.pt")
    (tmp_path/"recipe"/"training_done.json").unlink(missing_ok=True)
    calls = []
    stop = tmp_path/"stop"
    def compute(args):
        epoch = torch.load(args.checkpoints[0], weights_only=False)["epoch"]
        calls.append(epoch)
        if epoch == 40: stop.touch()
        return [{"epoch":epoch,"site":"test","nonfinite":True}]
    monkeypatch.setattr(signal,"compute",compute)
    base = ["--fake-data","--smoke","--checkpoint-dir",str(tmp_path/"recipe"),"--output",str(tmp_path/"signal")]
    args = signal.parser().parse_args(base)
    assert args.diagnostic_stride == 20
    signal.run(signal.parser().parse_args(base+["--watch","--stop-file",str(stop),"--no-merge"]))
    assert calls == [20,40]
    for marker in (tmp_path/"signal"/"completed").glob("*.json"): marker.unlink()
    for shard in (0,1):
        signal.run(signal.parser().parse_args(base+["--shards","2","--shard",str(shard),"--no-merge"]))
    assert calls == [20,40,40,20]  # stride-normalized sharding uses both GPUs
    assert len(signal.merge(args)) == 2
    marker = tmp_path/"signal"/"completed"/"epoch_020.json"
    old = json.loads(marker.read_text())
    old["identity"].pop("schema_version")
    marker.write_text(json.dumps(old))
    for operation in (signal.run, signal.merge):
        with pytest.raises(ValueError, match="schema_version"):
            operation(args)


def test_lite_hutchinson_cache_consistency_and_product_counts(case):
    model, site, batch, _, A = case
    cfg = VNAConfig(cg_iters=100, cg_tol=1e-5)
    _, full = vna_block_correction(model, site, A, batch, cfg)
    cold, lite = vna_lite_correction(model, site, A, batch, cfg)
    cached, warm = vna_lite_correction(model, site, A, batch, cfg, damping_cache=full["damping_cache"])
    assert all(torch.equal(cold[n], cached[n]) for n in cold)
    assert lite["trace_K"] == full["trace_K"]
    assert lite["c"] == (full["trace_K"] / 6)**2
    assert warm["jvp_calls"] == 2 and warm["vjp_calls"] == 2
    assert warm["extension_jvp_calls"] == 1 and warm["extension_vjp_calls"] == 3
    assert lite["jvp_calls"] == warm["jvp_calls"] + cfg.hutchinson_probes


def test_rank_selection_uses_separate_batches(case, tmp_path, monkeypatch):
    import diagnostics.signal_curve as signal
    model, _, _, _, _ = case
    checkpoint = tmp_path/"epoch_020.pt"
    torch.save({"kind":"vna_recipe", "epoch":20, "model":model.state_dict()}, checkpoint)
    batches = {}
    original_batch = signal.augmented_batch
    original_candidate = signal.tiny_candidate
    original_correction = signal.vna_block_correction
    statistics_calls, correction_calls = [], []
    def batch(*args, **kwargs):
        result = original_batch(*args, **kwargs)
        batches[id(result[0])] = result[0].clone()
        return result
    def candidate(model, site, batch, rank, *args):
        statistics_calls.append((id(batch[0]), rank))
        return original_candidate(model, site, batch, rank, *args)
    def correction(model, site, A, batch, cfg, *args, **kwargs):
        correction_calls.append((id(batch[0]), kwargs.get("apply_gain_guard", True)))
        return original_correction(model, site, A, batch, cfg, *args, **kwargs)
    monkeypatch.setattr(signal,"augmented_batch",batch)
    monkeypatch.setattr(signal,"tiny_candidate",candidate)
    monkeypatch.setattr(signal,"vna_block_correction",correction)
    rows = signal.compute(signal.parser().parse_args([
        "--fake-data","--smoke","--checkpoints",str(checkpoint),"--output",str(tmp_path/"signal")]))
    assert len(rows) == 5
    selection_stats, measurement_stats = statistics_calls[0][0], statistics_calls[1][0]
    selection_products, measurement_products = correction_calls[0][0], correction_calls[1][0]
    assert len({selection_stats, measurement_stats, selection_products, measurement_products}) == 4
    assert not torch.equal(batches[selection_stats], batches[measurement_stats])
    assert not torch.equal(batches[selection_products], batches[measurement_products])
    assert correction_calls[0][1] is False
    assert all(row["selected_top3"] for row in rows)


def test_per_rank_criteria_decay_and_epoch_boundaries():
    from diagnostics.signal_curve import rank_summary
    base = {"row_kind":"sweep", "site":"block", "rank":4, "random_rank":4,
        "nonfinite":False, "lite_nonfinite":False, "random_nonfinite":False, "random_lite_nonfinite":False,
        "U":10., "V":10., "rho":1., "lite_U":10., "lite_V":10., "lite_rho":1.,
        "random_V":1., "random_lite_V":1.,
        "cold_sgd_ratio":20., "warm_sgd_ratio":10., "lite_sgd_ratio":2.}
    for label in ("vna","lite","random","random_lite","lr_boost","lite_lr_boost"):
        base[label+"_valid"] = True
        base[label+"_empty"] = False
        base[label+"_derivative"] = -4. if label in ("vna","lite") else -1.
    rows = []
    for rank in (4,16):
        for epoch in (20,140,150,180,200):
            row = {**base,"epoch":epoch,"requested_rank":rank}
            if epoch >=180:
                for key in ("U","V","rho","lite_U","lite_V","lite_rho"): row[key] *= .05
            if rank == 16: row.update(lite_valid=False, lite_empty=True, lite_derivative=None)
            rows.append(row)
    table = rank_summary(rows)
    assert table[0]["rows_before_150"] == 2
    assert table[0]["full_joint_rows"] == table[0]["lite_joint_rows"] == 2
    assert table[0]["full_go"] and table[0]["lite_go"] and table[0]["complete"]
    assert table[0]["full_decay_eligible"] == 4 and table[0]["excluded_full_decay"] == 0
    assert table[1]["excluded_lite_V"] == table[1]["excluded_lite_derivative"] == table[1]["excluded_lite_lr_boost"] == 2
    assert table[1]["excluded_lite_decay"] == 4 and not table[1]["lite_go"]
    assert table[1]["full_go"] and table[1]["lite_decay_by_site"]["block"]["V"] is None


def test_branch_tip_session_commit_records(tmp_path):
    import json
    from types import SimpleNamespace
    from scripts.build_kaggle_vna_signal_curve_notebook import build as phase0
    from scripts.build_kaggle_vna_training_notebooks import build as training
    for notebook in [phase0(), training(seed=0), training(seed=1)]:
        bootstrap = ''.join(notebook['cells'][2]['source'])
        assert 'private_clone(MAIN_URL, REPO, REPO_BRANCH)' in bootstrap
        source = bootstrap[bootstrap.index('from datetime import datetime, timezone'):]
        scope = dict(subprocess=SimpleNamespace(check_output=lambda *a,**k:'a'*40+'\n'),
                     REPO=tmp_path, OUTPUT=tmp_path, REPO_BRANCH='vna-phase0', json=json)
        exec(compile(source, 'bootstrap-provenance', 'exec'), scope)
        scope['subprocess'] = SimpleNamespace(check_output=lambda *a,**k:'b'*40+'\n')
        exec(compile(source, 'bootstrap-provenance', 'exec'), scope)
        records = json.loads((tmp_path/'code_commit.json').read_text())
        assert [r['commit'] for r in records[-2:]] == ['a'*40,'b'*40]
        assert all(r['timestamp'] and r['branch']=='vna-phase0' for r in records)
        assert scope['CODE_COMMIT'] == 'b'*40


def test_phase0_completion_commit_separate_from_config(tmp_path, monkeypatch):
    import json
    import diagnostics.signal_curve as signal
    from experiments.train_recipe_checkpoints import parser as train_parser, run as train_run
    cp = train_run(train_parser().parse_args(['--fake-data','--limit-batches','1',
        '--max-epochs','1','--output',str(tmp_path/'recipe')]))
    payload = torch.load(cp, weights_only=False)
    torch.save({**payload,'epoch':20}, tmp_path/'recipe/epoch_020.pt')
    calls = []
    monkeypatch.setattr(signal, 'compute', lambda a: calls.append(a.checkpoints) or [])
    monkeypatch.setattr(signal, 'code_commit', lambda: 'a'*40)
    args = signal.parser().parse_args(['--fake-data','--smoke','--no-merge',
        '--checkpoint-dir',str(tmp_path/'recipe'),'--output',str(tmp_path/'signal')])
    signal.run(args)
    marker = tmp_path/'signal/completed/epoch_020.json'
    assert json.loads(marker.read_text())['code_commit'] == 'a'*40
    monkeypatch.setattr(signal, 'code_commit', lambda: 'b'*40)
    signal.run(args)
    assert len(calls) == 1
    torch.save({**payload,'epoch':40}, tmp_path/'recipe/epoch_040.pt')
    signal.run(args)
    assert json.loads((tmp_path/'signal/completed/epoch_040.json').read_text())['code_commit'] == 'b'*40
    args.cache_block_input = not args.cache_block_input
    with pytest.raises(ValueError, match='protocol mismatch'): signal.run(args)
