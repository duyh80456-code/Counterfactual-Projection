"""Phase 0: read-only VNA signals, noise controls, transfer and cost audit."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import time
import statistics
import os
import hashlib
import shutil

import torch
import torch.nn.functional as F
from experiments.shared_protocol import atomic_json_save, validate_resume_protocol
from experiments.vna_support import (setup_data, build_model, augmented_batch,
                                      tiny_candidate, isolated_rng, DeviceCIFAR)
from methods.vna import (VNAConfig, extract_tiny_A, random_A_like, dedicated_generator,
                         vna_block_correction, block_products, _mean_ce_logit_grad,
                         assert_disjoint_scopes)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def direction_vector(direction, products):
    return torch.cat([direction.get(name, torch.zeros_like(ref)).reshape(-1) for name, ref in products.parameter_refs])


def heldout(model, site, A, batch, directions, cfg, shared=None):
    inputs, targets = batch
    result = {}
    with block_products(model, site, A, inputs, cfg.bn_mode, cache_block_input=cfg.cache_block_input) as products:
        logits, g = shared if shared is not None else (products._call({}).reshape(len(targets), -1), None)
        if g is None:
            g = _mean_ce_logit_grad(logits, targets)
        desired = -(products.K(g) + products.KE(g)).double()
        for label, direction in directions.items():
            image = products.Jv(direction_vector(direction, products)).double()
            denom = image.norm() * desired.norm()
            result[label + "_cosine"] = float(image.dot(desired) / denom) if denom > 0 else 0.
            result[label + "_derivative"] = float(g.dot(image))
        result["heldout_ce"] = float(F.cross_entropy(logits.double(), targets))
    return result


def outside_top_space(model, site, A, batch, cfg, q=20, shared=None):
    with block_products(model, site, A, batch[0], cfg.bn_mode, cache_block_input=cfg.cache_block_input) as products:
        if shared is None:
            logits = products._call({}).reshape(len(batch[1]), -1)
            g = _mean_ce_logit_grad(logits, batch[1])
        else:
            _, g = shared
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


def compute(args):
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    if args.fake_data:
        torch.set_num_threads(1)
    cfg = VNAConfig(seed=args.seed, bn_mode=args.vna_bn, cache_block_input=args.cache_block_input)
    if args.smoke:
        cfg = replace(cfg, stat_samples=16, vna_batch=2, r_max=2, cg_iters=10)
    # Model initialization is private too, so callers may embed the diagnostic.
    with isolated_rng(dedicated_generator(args.seed, "model")):
        model = build_model(device, args.fake_data)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    train, evaluation, indices, trigger, _, _ = setup_data(args.data_root, args.fake_data, args.seed)
    if args.gpu_data and not args.fake_data:
        train = DeviceCIFAR(train, device)
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
    if args.channels_last:
        stat, vna_batch, check, sgd_batch = [
            (inputs.contiguous(memory_format=torch.channels_last), labels)
            for inputs, labels in (stat, vna_batch, check, sgd_batch)]
    seen_epochs = set()
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
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
        # At B=0 logits/g are site-independent. Cache once per checkpoint.
        def shared(batch):
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                model.eval()
                if cfg.bn_mode == "train":
                    from torch.func import functional_call
                    for module in model.modules():
                        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                            module.train()
                    logits = functional_call(model, {n: b.clone() for n, b in model.named_buffers()}, (batch[0],))
                    model.eval()
                else:
                    logits = model(batch[0])
            return logits.detach(), _mean_ce_logit_grad(logits, batch[1])
        vna_logits, vna_g = shared(vna_batch)
        heldout_shared = shared(check)
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
            correction, cold = vna_block_correction(model, site, A, vna_batch, cfg, shared_logits=vna_logits, shared_g=vna_g)
            if cold["nonfinite"]:
                rows.append({"epoch": epoch, "site": site, "nonfinite": True})
                continue
            _, warm = vna_block_correction(model, site, A, vna_batch, cfg,
                                           warm_start=cold["warm_start"], damping_cache=cold["damping_cache"], shared_logits=vna_logits, shared_g=vna_g)
            random_correction, random_diag = vna_block_correction(model, site, random_A, vna_batch, cfg,
                                                                  damping_cache=cold["damping_cache"], shared_logits=vna_logits, shared_g=vna_g)
            with block_products(model, site, A, vna_batch[0], cfg.bn_mode, cache_block_input=cfg.cache_block_input) as products:
                plain = products.JT(vna_g)
                sizes = [r.numel() for _, r in products.parameter_refs]
                plain_dict = {name: -piece.reshape_as(ref) for (name, ref), piece in
                              zip(products.parameter_refs, plain.split(sizes))}
                correction_norm = direction_vector(correction, products).norm()
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
                   **heldout(model, site, A, check, {k:v for k,v in directions.items() if k != "random"}, cfg, shared=heldout_shared),
                   **heldout(model, site, random_A, check, {"random": directions["random"]}, cfg, shared=heldout_shared),
                   "g_outside_top_q": outside_top_space(model, site, A, vna_batch, cfg, shared=(vna_logits, vna_g)),
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
    return rows


def write_outputs(rows, args):
    cfg = VNAConfig(seed=args.seed, bn_mode=args.vna_bn, cache_block_input=args.cache_block_input)
    if args.smoke:
        cfg = replace(cfg, stat_samples=16, vna_batch=2, r_max=2, cg_iters=10)
    sites = sorted({r["site"] for r in rows})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    temporary = output / "signal_curve.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    temporary.replace(output / "signal_curve.csv")
    valid = [r for r in rows if not r["nonfinite"] and "rho" in r]
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
    early = [r for r in valid if r["epoch"] < 150]
    joint = [r for r in early if r["random_V"] is not None and r["V"] >= 3*r["random_V"]
             and r["vna_derivative"] < 0
             and abs(r["vna_derivative"]) >= 3*abs(r["random_derivative"])]
    # SPEC gives no numerical definition of "toward 0". Use a declared,
    # untuned operational rule: late (>=180) mean below 10% of early (<150).
    trends = {}
    for site in sites:
        a = [r for r in early if r["site"] == site]
        b = [r for r in valid if r["site"] == site and r["epoch"] >= 180]
        trends[site] = {key: (statistics.mean(r[key] for r in b) <= .1*statistics.mean(r[key] for r in a))
                        if a and b else None for key in ("U", "V", "rho")}
    complete = bool(valid) and max(r["epoch"] for r in valid) >= 200 and all(
        value is not None for trend in trends.values() for value in trend.values())
    decay = complete and all(value for trend in trends.values() for value in trend.values())
    summary.update(phase0_acceptance=("not assessed: synthetic/partial run" if args.fake_data or args.smoke or not complete
                    else "go" if joint and decay else "no-go"),
                   simultaneous_early_criteria_rows=len(joint), signal_decay_by_site=trends,
                   decay_rule="mean epochs >=180 <= 0.1 * mean epochs <150, per site and U/V/rho",
                   skipped_nonpositive_gain=sum(r.get("nonpositive_gain", False) for r in rows))
    atomic_json_save(summary, output / "summary.json")
    return rows


def checkpoint_identity(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("kind") != "vna_recipe":
        raise ValueError("expected a vna_recipe checkpoint")
    digest = hashlib.sha256()
    for name, value in sorted(checkpoint["model"].items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return checkpoint["epoch"], digest.hexdigest()


def diagnostic_identity(args):
    identity = {k: getattr(args, k) for k in (
        "seed", "vna_bn", "fake_data", "smoke", "gpu_data", "cache_block_input",
        "channels_last", "cudnn_benchmark")}
    identity["amp"] = False  # VNA products and solves always use fp32.
    # Include algorithm source so markers from changed math are never reused.
    identity["source"] = hashlib.sha256(Path(__file__).read_bytes() +
        (Path(__file__).resolve().parents[1]/"methods/vna.py").read_bytes()).hexdigest()
    return identity


def run(args):
    """Atomic per-checkpoint transactions; CSV is a rebuildable materialization."""
    from copy import copy
    output = Path(args.output)
    done = output / "completed"
    done.mkdir(parents=True, exist_ok=True)
    identity = diagnostic_identity(args)
    while True:
        paths = [Path(p) for p in args.checkpoints] or sorted(Path(args.checkpoint_dir).glob("epoch_*.pt"))
        latest = Path(args.checkpoint_dir)/"latest.pt"
        # Watch only retained 10-epoch checkpoints: latest is atomically replaced
        # each epoch and is reserved for final/partial non-watch diagnostics.
        if not args.watch and not args.checkpoints and latest.exists():
            paths.append(latest)
        for path in paths:
            if args.watch and ((args.stop_file and Path(args.stop_file).exists()) or
                               (Path(args.checkpoint_dir)/"training_done.json").exists()):
                break
            epoch, digest = checkpoint_identity(path)
            if (epoch // 10) % args.shards != args.shard:
                continue
            marker = done / f"epoch_{epoch:03d}.json"
            if marker.exists():
                previous = json.loads(marker.read_text())
                validate_resume_protocol(previous["identity"], identity)
                if previous["model_sha256"] != digest:
                    raise ValueError(f"incompatible completed diagnostic: {marker}")
                continue
            claim = done / f"epoch_{epoch:03d}.lock"
            try:
                claim.mkdir()
            except FileExistsError:
                owner = claim / "pid"
                stale = not owner.exists() and time.time() - claim.stat().st_mtime > 60
                if owner.exists():
                    try:
                        os.kill(int(owner.read_text()), 0)
                    except ProcessLookupError:
                        stale = True
                if not stale:
                    continue
                shutil.rmtree(claim)
                try:
                    claim.mkdir()
                except FileExistsError:
                    continue
            (claim/"pid").write_text(str(os.getpid()))
            try:
                single = copy(args)
                single.checkpoints = [str(path)]
                rows = compute(single)
                atomic_json_save({"identity": identity, "model_sha256": digest, "epoch": epoch, "rows": rows}, marker)
                print(f"Completed diagnostic epoch {epoch}", flush=True)
            finally:
                shutil.rmtree(claim)
        if not args.watch:
            break
        if args.stop_file and Path(args.stop_file).exists():
            break
        if (Path(args.checkpoint_dir)/"training_done.json").exists():
            break
        time.sleep(args.poll_seconds)
    if not args.no_merge:
        return merge(args)
    return []


def merge(args):
    rows = []
    identities = set()
    for path in sorted((Path(args.output)/"completed").glob("epoch_*.json")):
        payload = json.loads(path.read_text())
        validate_resume_protocol(payload["identity"], diagnostic_identity(args))
        identities.add(json.dumps(payload["identity"], sort_keys=True))
        rows.extend(payload["rows"])
    if len(identities) > 1:
        raise ValueError("cannot merge incompatible diagnostics")
    return write_outputs(rows, args)


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
    p.add_argument("--watch", action="store_true")
    p.add_argument("--stop-file")
    p.add_argument("--poll-seconds", type=float, default=10.)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--no-merge", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--gpu-data", action="store_true")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--cudnn-benchmark", action="store_true")
    p.add_argument("--cache-block-input", action="store_true", help="exact eval-BN native Gromo suffix; train BN uses full path")
    p.add_argument("--grow-epochs", type=int, default=0, help="TODO: paired real-growth reference is deferred")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.grow_epochs:
        raise NotImplementedError("--grow-epochs is deferred; see STAGE1_REPORT.md")
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        raise ValueError("invalid shard")
    merge(args) if args.merge_only else run(args)
