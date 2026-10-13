"""Stage 1 Gromo alignment, matrix-free math, invariance and CLI smoke tests."""
from dataclasses import replace
from pathlib import Path
import copy

import pytest
import torch
from torch.func import functional_call

from experiments.vna_support import tiny_model, tiny_candidate, isolated_rng
from methods.vna import (VNAConfig, dedicated_generator, extract_tiny_A, random_A_like,
    vna_block_correction, block_products, _mean_ce_logit_grad, _virtual_b_context,
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
    resumed = run(parser().parse_args(common+["--output",str(tmp_path/"resumed"),"--max-epochs","2","--resume",str(first)]))
    a,b = [torch.load(p,weights_only=False) for p in [straight,resumed]]
    assert all(torch.equal(a["model"][n],b["model"][n]) for n in a["model"])
    assert torch.equal(a["rng"]["torch"],b["rng"]["torch"])
    assert torch.equal(a["train_loader_generator_state"],b["train_loader_generator_state"])
    assert a["scheduler"] == b["scheduler"]
    for pid, values in a["optimizer"]["state"].items():
        assert torch.equal(values["momentum_buffer"],b["optimizer"]["state"][pid]["momentum_buffer"])
    rows = signal_run(signal_parser().parse_args(["--fake-data","--smoke","--checkpoints",str(resumed),"--output",str(tmp_path/"signal")]))
    assert len(rows) == 1 and not rows[0]["nonfinite"]
    assert rows[0]["per_neuron_V"] and rows[0]["lambda_split_half"] >= 0
    assert "lr_boost_derivative" in rows[0]
    assert (tmp_path/"signal"/"signal_curve.csv").exists()
    assert (tmp_path/"signal"/"summary.json").exists()
