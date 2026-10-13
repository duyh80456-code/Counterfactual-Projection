"""Phase 0: read-only VNA signals, noise controls, transfer and cost audit."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import time
import statistics

import torch
import torch.nn.functional as F
from experiments.shared_protocol import atomic_json_save
from experiments.vna_support import (setup_data, build_model, augmented_batch,
                                      tiny_candidate, isolated_rng)
from methods.vna import (VNAConfig, extract_tiny_A, random_A_like, dedicated_generator,
                         vna_block_correction, block_products, _mean_ce_logit_grad,
                         assert_disjoint_scopes)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def direction_vector(direction, products):
    return torch.cat([direction[name].reshape(-1) for name, _ in products.parameter_refs])


def heldout(model, site, A, batch, directions, cfg):
    inputs, targets = batch
    result = {}
    with block_products(model, site, A, inputs, cfg.bn_mode) as products:
        logits = products._call({}).reshape(len(targets), -1)
        g = _mean_ce_logit_grad(logits, targets)
        desired = -(products.K(g) + products.KE(g)).double()
        for label, direction in directions.items():
            image = products.Jv(direction_vector(direction, products)).double()
            denom = image.norm() * desired.norm()
            result[label + "_cosine"] = float(image.dot(desired) / denom) if denom > 0 else 0.
            result[label + "_derivative"] = float(g.dot(image))
        result["heldout_ce"] = float(F.cross_entropy(logits.double(), targets))
    return result


def outside_top_space(model, site, A, batch, cfg, q=20):
    with block_products(model, site, A, batch[0], cfg.bn_mode) as products:
        logits = products._call({}).reshape(len(batch[1]), -1)
        g = _mean_ce_logit_grad(logits, batch[1])
        n, width = g.numel(), min(g.numel(), q + 8)
        omega = torch.randn(n, width, device=g.device, dtype=torch.float64,
                            generator=dedicated_generator(cfg.seed, "k-range", g.device))
        def images(matrix):
            return torch.stack([products.K(v).double() for v in matrix.T], dim=1)
        basis = torch.linalg.qr(images(omega)).Q
        for _ in range(2):
            basis = torch.linalg.qr(images(basis)).Q
        projected = basis.T @ images(basis)
        values, vectors = torch.linalg.eigh((projected + projected.T) / 2)
        # Only positive eigenmodes count; QR can manufacture null-space columns.
        keep = values > max(float(values.max()) * 1e-8, 1e-12)
        space = basis @ vectors[:, keep][:, -q:]
        return max(0., min(1., float(1 - (space.T @ g).square().sum() / g.square().sum())))


def cross_validated_lambda(first, second):
    """Half-1 generalized left singular vector, half-2 squared Rayleigh score.

    Return its square root in the native singular-value units. N includes the
    native existing-weight projection independently estimated on each half.
    """
    S, N = first["S"], first["N"]
    values, vectors = torch.linalg.eigh((S + S.T) / 2)
    keep = values > max(float(values.max()) * 1e-10, 1e-12)
    inverse_root = (vectors[:, keep] * values[keep].rsqrt()) @ vectors[:, keep].T
    u, _, _ = torch.linalg.svd(inverse_root @ N, full_matrices=False)
    a = inverse_root @ u[:, 0]
    numerator = (a @ second["N"]).square().sum()
    denominator = a @ second["S"] @ a
    return float((numerator / denominator.clamp_min(1e-30)).sqrt())


def sgd_seconds(model, batch):
    state = {n: t.clone() for n, t in model.state_dict().items()}
    modes = {m: m.training for m in model.modules()}
    grads = {p: p.grad for p in model.parameters()}
    device = batch[0].device
    try:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)
        sync(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(batch[0]), batch[1]).backward()
        optimizer.step()
        sync(device)
        return time.perf_counter() - start
    finally:
        model.load_state_dict(state)
        for m, mode in modes.items():
            m.training = mode
        for p, grad in grads.items():
            p.grad = grad


def serial_diag(diag):
    return {k: v for k, v in diag.items() if k not in {"warm_start", "damping_cache"}}


def run(args):
    device = torch.device(args.device)
    if args.fake_data:
        torch.set_num_threads(1)
    cfg = VNAConfig(seed=args.seed, bn_mode=args.vna_bn)
    if args.smoke:
        cfg = replace(cfg, stat_samples=16, vna_batch=2, r_max=2, cg_iters=10)
    # Model initialization is private too, so callers may embed the diagnostic.
    with isolated_rng(dedicated_generator(args.seed, "model")):
        model = build_model(device, args.fake_data)
    train, evaluation, indices, trigger, _, _ = setup_data(args.data_root, args.fake_data, args.seed)
    sites = [ref.name for ref in model.growing_blocks()]
    assert_disjoint_scopes(model, sites)
    checkpoints = [Path(p) for p in args.checkpoints]
    if not checkpoints:
        checkpoints = sorted(Path(args.checkpoint_dir).glob("epoch_*.pt"))
        latest = Path(args.checkpoint_dir) / "latest.pt"
        if latest.exists():
            checkpoints.append(latest)
    if not checkpoints:
        raise ValueError("no recipe checkpoints found")
    rows = []
    # Fixed augmented train and trigger batches pair sites/checkpoints exactly.
    stat = augmented_batch(train, indices, cfg.stat_samples, dedicated_generator(args.seed, "statistics"), device)
    vna_batch = augmented_batch(train, indices, cfg.vna_batch, dedicated_generator(args.seed, "vna"), device)
    check = augmented_batch(evaluation, trigger, 4 if args.smoke else 512,
                            dedicated_generator(args.seed, "heldout"), device)
    sgd_batch = augmented_batch(train, indices, 4 if args.smoke else 64,
                                dedicated_generator(args.seed, "cost-sgd"), device)
    seen_epochs = set()
    for path in checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get("kind") != "vna_recipe":
            raise ValueError("expected a vna_recipe checkpoint")
        epoch = checkpoint["epoch"]
        if epoch in seen_epochs:
            continue
        seen_epochs.add(epoch)
        model.load_state_dict(checkpoint["model"])
        # Discard first-call setup; compare with the median time of a single
        # SGD step, recording all three trials rather than concealing noise.
        sgd_seconds(model, sgd_batch)
        sgd_trials = [sgd_seconds(model, sgd_batch) for _ in range(3)]
        base_cost = statistics.median(sgd_trials)
        for site in sites:
            refresh_start = time.perf_counter()
            candidate = tiny_candidate(model, site, stat, cfg.r_max, args.seed)
            spectrum = candidate.payload["tiny_eigenvalues"][:16]
            A, rank = extract_tiny_A(model, candidate)
            random_A = random_A_like(A, generator=dedicated_generator(args.seed, f"random-A:{site}", device))
            lambdas = []
            for xb, yb in zip(stat[0].chunk(4), stat[1].chunk(4)):
                sub = tiny_candidate(model, site, (xb, yb), 1, args.seed)
                lambdas.append(sub.payload["tiny_eigenvalues"][0])
            halves = []
            for xb, yb in zip(stat[0].chunk(2), stat[1].chunk(2)):
                capture = {}
                tiny_candidate(model, site, (xb, yb), 1, args.seed, capture)
                halves.append(capture)
            cv = cross_validated_lambda(*halves)
            order = torch.randperm(len(stat[1]), device=device,
                                   generator=dedicated_generator(args.seed, "shuffled-labels", device))
            null = tiny_candidate(model, site, (stat[0], stat[1][order]), 1, args.seed).payload["tiny_eigenvalues"][0]
            std = float(torch.tensor(lambdas, dtype=torch.float64).std(unbiased=False))
            mean = sum(lambdas) / 4
            noise = mean + 2 * std
            sync(device)
            refresh_seconds = time.perf_counter() - refresh_start
            correction, cold = vna_block_correction(model, site, A, vna_batch, cfg)
            if cold["nonfinite"]:
                rows.append({"epoch": epoch, "site": site, "nonfinite": True})
                continue
            _, warm = vna_block_correction(model, site, A, vna_batch, cfg,
                                           warm_start=cold["warm_start"], damping_cache=cold["damping_cache"])
            random_correction, random_diag = vna_block_correction(model, site, random_A, vna_batch, cfg,
                                                                  damping_cache=cold["damping_cache"])
            with block_products(model, site, A, vna_batch[0], cfg.bn_mode) as products:
                logits = products._call({}).reshape(len(vna_batch[1]), -1)
                plain = products.JT(_mean_ce_logit_grad(logits, vna_batch[1]))
                sizes = [r.numel() for _, r in products.parameter_refs]
                plain_dict = {name: -piece.reshape_as(ref) for (name, ref), piece in
                              zip(products.parameter_refs, plain.split(sizes))}
                correction_norm = torch.cat([v.reshape(-1) for v in correction.values()]).norm()
                boost = float(correction_norm / plain.norm().clamp_min(1e-30))
            directions = {"vna": {n: -v for n, v in correction.items()}, "plain": plain_dict,
                          "random": {n: -v for n, v in random_correction.items()},
                          "lr_boost": {n: v * boost for n, v in plain_dict.items()}}
            row = {"checkpoint": str(path), "epoch": epoch, "site": site, "rank": rank,
                   "lambda_spectrum": spectrum, "lambda_full": spectrum[0],
                   "lambda_sub_mean": mean, "lambda_sub_std": std, "lambda_sub_values": lambdas,
                   "lambda_split_half": cv, "lambda_shuffled": null, "noise_floor": noise,
                   "d_mu": sum(v / (v + noise) for v in spectrum),
                   **{k:v for k,v in serial_diag(cold).items() if k != "site"}, "block_path": cold["site"], "random_V": random_diag.get("V"),
                   "random_nonfinite": random_diag["nonfinite"],
                   **heldout(model, site, A, check, {k:v for k,v in directions.items() if k != "random"}, cfg),
                   **heldout(model, site, random_A, check, {"random": directions["random"]}, cfg),
                   "g_outside_top_q": outside_top_space(model, site, A, vna_batch, cfg),
                   "refresh_seconds": refresh_seconds, "sgd_seconds": base_cost, "sgd_step_timings": sgd_trials,
                   "cold_seconds": cold["wall_time"], "warm_seconds": warm["wall_time"],
                   "warm_jvp_calls": warm["jvp_calls"], "warm_vjp_calls": warm["vjp_calls"],
                   "warm_extension_jvp_calls": warm["extension_jvp_calls"],
                   "warm_extension_vjp_calls": warm["extension_vjp_calls"],
                   "warm_cg_z": warm["cg_z"], "warm_cg_q": warm["cg_q"]}
            for kind in ("cold", "warm"):
                ratio = row[kind + "_seconds"] / base_cost
                row[kind + "_sgd_ratio"] = ratio
                row[kind + "_overhead_1_site"] = ratio / cfg.vna_every
                row[kind + "_overhead_3_sites"] = 3 * ratio / cfg.vna_every
            rows.append(row)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    temporary = output / "signal_curve.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    temporary.replace(output / "signal_curve.csv")
    valid = [r for r in rows if not r["nonfinite"]]
    summary = {"config": asdict(cfg), "rows": len(rows), "nonfinite_rows": len(rows) - len(valid),
               "rho_above_0_3": sum(r["rho"] > .3 for r in valid),
               "rho_above_0_3_fraction": sum(r["rho"] > .3 for r in valid) / max(1, len(valid)),
               "rho_by_site": {site: [r["rho"] for r in valid if r["site"] == site] for site in sites},
               "fake_data": args.fake_data, "smoke": args.smoke,
               "phase0_acceptance": "not assessed: smoke/partial checkpoints cannot establish training trends",
               "grow_epochs": "TODO: paired persistent-growth experiment deferred; no growth installed"}
    summary["tiny_V_at_least_3x_random_before_150"] = sum(
        r["epoch"] < 150 and r["random_V"] is not None and r["V"] >= 3*r["random_V"] for r in valid)
    summary["heldout_negative_at_least_3x_random_before_150"] = sum(
        r["epoch"] < 150 and r["vna_derivative"] < 0 and
        abs(r["vna_derivative"]) >= 3*abs(r["random_derivative"]) for r in valid)
    summary["signals_by_epoch"] = [
        {"epoch": r["epoch"], "site": r["site"], "U": r["U"], "V": r["V"], "rho": r["rho"]}
        for r in valid]
    atomic_json_save(summary, output / "summary.json")
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", nargs="*", default=[])
    p.add_argument("--checkpoint-dir", default="outputs/vna_recipe")
    p.add_argument("--output", default="outputs/vna_signal")
    p.add_argument("--data-root", default="data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vna-bn", choices=["eval", "train"], default="eval")
    p.add_argument("--fake-data", action="store_true")
    p.add_argument("--smoke", action="store_true", help="small batches/solves for CPU end-to-end verification")
    p.add_argument("--grow-epochs", type=int, default=0, help="TODO: paired real-growth reference is deferred")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.grow_epochs:
        raise NotImplementedError("--grow-epochs is deferred; see STAGE1_REPORT.md")
    run(args)
