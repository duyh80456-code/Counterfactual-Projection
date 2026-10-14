"""Phase 0: read-only VNA signals, noise controls, transfer and cost audit."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import csv
import json
from pathlib import Path
import time
import statistics
import math
import os
import hashlib
import shutil

import torch
import torch.nn.functional as F
from experiments.shared_protocol import atomic_json_save, validate_resume_protocol
from experiments.vna_provenance import code_commit
from experiments.vna_support import (setup_data, build_model, augmented_batch,
                                      tiny_candidate, isolated_rng, DeviceCIFAR)
from methods.vna import (VNAConfig, extract_tiny_A, random_A_like, dedicated_generator,
                         vna_block_correction, vna_lite_correction, block_products, _mean_ce_logit_grad,
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
            valid = bool(direction) and all(torch.isfinite(v).all() for v in direction.values())
            if not valid:
                result.update({label + "_empty": not bool(direction), label + "_valid": False,
                               label + "_cosine": None, label + "_derivative": None})
                continue
            image = products.Jv(direction_vector(direction, products)).double()
            result[label + "_empty"] = False
            result[label + "_valid"] = bool(torch.isfinite(image).all())
            denom = image.norm() * desired.norm()
            result[label + "_cosine"] = float(image.dot(desired) / denom) if denom > 0 else None
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
    selection_stat = augmented_batch(train, indices, cfg.stat_samples,
        dedicated_generator(args.seed, "rank-selection-statistics"), device)
    selection = augmented_batch(train, indices, cfg.vna_batch,
        dedicated_generator(args.seed, "rank-selection"), device)
    # Fixed augmented train and trigger batches pair sites/checkpoints exactly.
    stat = augmented_batch(train, indices, cfg.stat_samples, dedicated_generator(args.seed, "statistics"), device)
    vna_batch = augmented_batch(train, indices, cfg.vna_batch, dedicated_generator(args.seed, "vna"), device)
    check = augmented_batch(evaluation, trigger, 4 if args.smoke else 512,
                            dedicated_generator(args.seed, "heldout"), device)
    sgd_batch = augmented_batch(train, indices, 64,
                                dedicated_generator(args.seed, "cost-sgd"), device)
    if args.channels_last:
        stat, vna_batch, check, sgd_batch, selection, selection_stat = [
            (inputs.contiguous(memory_format=torch.channels_last), labels)
            for inputs, labels in (stat, vna_batch, check, sgd_batch, selection, selection_stat)]
    seen_epochs = set()
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("kind") != "vna_recipe":
            raise ValueError("expected a vna_recipe checkpoint")
        epoch = checkpoint["epoch"]
        if epoch % args.diagnostic_stride or epoch in seen_epochs:
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
        # Selection has its own augmented data stream, never measurement V.
        selection_logits, selection_g = shared(selection)
        selection_scores = {}
        selection_start = time.perf_counter()
        for site in sites:
            selected = tiny_candidate(model, site, selection_stat, cfg.r_max, args.seed)
            selected_A, _ = extract_tiny_A(model, selected)
            _, selected_diag = vna_block_correction(
                model, site, selected_A, selection, cfg,
                shared_logits=selection_logits, shared_g=selection_g, apply_gain_guard=False)
            score = selected_diag.get("V", float("nan"))
            if not selected_diag["nonfinite"] and math.isfinite(score):
                selection_scores[site] = score
        sync(device)
        selection_seconds = time.perf_counter() - selection_start
        top_sites = sorted(selection_scores, key=lambda site: (-selection_scores[site], site))[:3]
        for site in sites:
            refresh_start = time.perf_counter()
            default_candidate = tiny_candidate(model, site, stat, cfg.r_max, args.seed)
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
            nulls = []
            for shuffle in range(3):
                order = torch.randperm(len(stat[1]), device=device,
                    generator=dedicated_generator(args.seed, f"shuffled-labels:{shuffle}", device))
                nulls.append(tiny_candidate(model, site, (stat[0], stat[1][order]), 1,
                                           args.seed).payload["tiny_eigenvalues"][0])
            null_mean, null_max = statistics.mean(nulls), max(nulls)
            std = statistics.pstdev(lambdas)
            mean = statistics.mean(lambdas)
            noise = mean + 2 * std
            sync(device)
            refresh_seconds = time.perf_counter() - refresh_start
            width = int(model.block(site).hidden_neurons)
            requests = [("default", cfg.r_max)]
            if site in top_sites:
                requests += [("sweep", r) for r in dict.fromkeys((4, 16, 64, width))]
            for row_kind, requested in requests:
                rank_start = time.perf_counter()
                candidate = default_candidate if row_kind == "default" else tiny_candidate(
                    model, site, stat, requested, args.seed)
                spectrum = candidate.payload["tiny_eigenvalues"]
                A, rank = extract_tiny_A(model, candidate)
                sync(device)
                rank_refresh_seconds = time.perf_counter() - rank_start
                random_A = random_A_like(A, generator=dedicated_generator(
                    args.seed, f"random-A:{site}:{requested}", device))
                common = dict(shared_logits=vna_logits, shared_g=vna_g)
                correction, cold = vna_block_correction(model, site, A, vna_batch, cfg, **common)
                cache = cold.get("damping_cache")
                warm = {}
                if cache is not None:
                    _, warm = vna_block_correction(model, site, A, vna_batch, cfg,
                        warm_start=cold["warm_start"], damping_cache=cache, **common)
                random_correction, random_diag = vna_block_correction(
                    model, site, random_A, vna_batch, cfg, damping_cache=cache,
                    apply_gain_guard=False, **common)
                lite_correction, lite = vna_lite_correction(
                    model, site, A, vna_batch, cfg, damping_cache=cache, **common)
                random_lite_correction, random_lite = vna_lite_correction(
                    model, site, random_A, vna_batch, cfg, damping_cache=cache,
                    apply_gain_guard=False, **common)
                with block_products(model, site, A, vna_batch[0], cfg.bn_mode,
                                    cache_block_input=cfg.cache_block_input) as products:
                    plain = products.JT(vna_g)
                    sizes = [r.numel() for _, r in products.parameter_refs]
                    plain_dict = {name: -piece.reshape_as(ref) for (name, ref), piece in
                                  zip(products.parameter_refs, plain.split(sizes))}
                    boost = float(direction_vector(correction, products).norm() / plain.norm().clamp_min(1e-30))
                    lite_boost = float(direction_vector(lite_correction, products).norm() / plain.norm().clamp_min(1e-30))
                    lr_gain = float(vna_g.dot(products.Jv(plain * boost).double())) if correction else None
                    lite_lr_gain = float(vna_g.dot(products.Jv(plain * lite_boost).double())) if lite_correction else None
                directions = {
                    "vna": {n: -v for n, v in correction.items()}, "plain": plain_dict,
                    "lite": {n: -v for n, v in lite_correction.items()},
                    "lr_boost": {n: v * boost for n, v in plain_dict.items()} if correction else {},
                    "lite_lr_boost": {n: v * lite_boost for n, v in plain_dict.items()} if lite_correction else {}}
                row = {"checkpoint": str(path), "epoch": epoch, "site": site,
                       "row_kind": row_kind, "requested_rank": requested, "effective_rank": rank,
                       "site_width": width, "rank": rank, "random_rank": rank,
                       "lr_boost_realized_gain": lr_gain, "lite_lr_boost_realized_gain": lite_lr_gain,
                       "lr_boost_scale": boost, "lite_lr_boost_scale": lite_boost,
                       "selection_batch_purpose": "rank-selection", "selection_V": selection_scores.get(site),
                       "selected_top3": site in top_sites, "selection_seconds": selection_seconds,
                       "lambda_spectrum": spectrum, "lambda_full": spectrum[0],
                       "lambda_spectrum_count": len(spectrum), "lambda_spectrum_min": min(spectrum),
                       "lambda_spectrum_sum": sum(spectrum), "stat_samples": len(stat[1]),
                       "lambda_sub_mean": mean, "lambda_sub_std": std, "lambda_sub_values": lambdas,
                       "lambda_split_half": cv, "lambda_shuffled": null_mean,
                       "lambda_shuffled_values": nulls, "lambda_shuffled_mean": null_mean,
                       "lambda_shuffled_max": null_max, "shuffled_samples": len(stat[1]),
                       "significant_vs_shuffled": spectrum[0] > null_max,
                       "significant_split_half": cv > null_max,
                       "noise_floor": noise, "d_mu": sum(v / (v + null_max) for v in spectrum),
                       **{k:v for k,v in serial_diag(cold).items() if k != "site"},
                       "block_path": cold.get("site"),
                       **{"random_"+k: v for k,v in serial_diag(random_diag).items()},
                       **{"lite_"+k: v for k,v in serial_diag(lite).items()},
                       **{"random_lite_"+k: v for k,v in serial_diag(random_lite).items()},
                       **heldout(model, site, A, check, directions, cfg, shared=heldout_shared),
                       **heldout(model, site, random_A, check, {
                           "random": {n:-v for n,v in random_correction.items()},
                           "random_lite": {n:-v for n,v in random_lite_correction.items()}}, cfg, shared=heldout_shared),
                       "refresh_seconds": refresh_seconds, "rank_refresh_seconds": rank_refresh_seconds,
                       "sgd_batch_size": len(sgd_batch[1]), "sgd_seconds": base_cost, "sgd_step_timings": sgd_trials,
                       "cold_seconds": cold["wall_time"], "warm_seconds": warm.get("wall_time"),
                       "lite_seconds": lite["wall_time"],
                       **{"warm_"+k:v for k,v in serial_diag(warm).items()}}
                if row_kind == "default":
                    row["g_outside_top_q"] = outside_top_space(
                        model, site, A, vna_batch, cfg, shared=(vna_logits, vna_g))
                for kind in ("cold", "warm", "lite"):
                    seconds = row[kind + "_seconds"]
                    ratio = None if seconds is None else seconds / base_cost
                    row[kind + "_sgd_ratio"] = ratio
                    row[kind + "_overhead_1_site"] = None if ratio is None else ratio / cfg.vna_every
                    row[kind + "_overhead_3_sites"] = None if ratio is None else 3 * ratio / cfg.vna_every
                rows.append(row)
    return rows


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def criterion_results(row, variant):
    """Every criterion explicitly requires its finite, nonempty directions."""
    own, random = ("vna", "random") if variant == "full" else ("lite", "random_lite")
    prefix = "" if variant == "full" else "lite_"
    lr = "lr_boost" if variant == "full" else "lite_lr_boost"
    def direction_ok(label):
        return row.get(label + "_valid", False) and not row.get(label + "_empty", True) and finite(row.get(label+"_derivative"))
    def diag_ok(p):
        return not row.get(p+"nonfinite", True) and finite(row.get(p+"V"))
    both = (direction_ok(own) and direction_ok(random) and diag_ok(prefix) and diag_ok(random+"_")
            and row.get("random_rank") == row.get("rank") and row.get("rank", 0) > 0)
    d, rd = row.get(own+"_derivative"), row.get(random+"_derivative")
    v, rv = row.get(prefix+"V"), row.get(random+"_V")
    return {
        "V": (both, bool(both and v >= 3*rv)),
        "derivative": (both, bool(both and d < 0 and abs(d) >= 3*abs(rd))),
        "lr_boost": (direction_ok(own) and diag_ok(prefix) and direction_ok(lr),
            bool(direction_ok(own) and diag_ok(prefix) and direction_ok(lr) and d < 0 and d < row[lr+"_derivative"]))}


def rank_summary(rows):
    result = []
    sweep = [r for r in rows if r.get("row_kind") == "sweep"]
    for rank in sorted({r["requested_rank"] for r in sweep}):
        group = [r for r in sweep if r["requested_rank"] == rank]
        early = [r for r in group if r["epoch"] < 150]
        item = {"requested_rank": rank, "rows_before_150": len(early),
                "effective_ranks": sorted({r["rank"] for r in group})}
        for variant in ("full", "lite"):
            outcomes = [criterion_results(r, variant) for r in early]
            for criterion in ("V", "derivative", "lr_boost"):
                eligible = sum(o[criterion][0] for o in outcomes)
                item[variant+"_"+criterion+"_eligible"] = eligible
                item[variant+"_"+criterion+"_passed"] = sum(o[criterion][1] for o in outcomes)
                item["excluded_"+variant+"_"+criterion] = len(early) - eligible
            item[variant+"_joint_rows"] = sum(all(v[1] for v in o.values()) for o in outcomes)
            prefix = "" if variant == "full" else "lite_"
            trends = {}
            decay_eligible = decay_excluded = 0
            for site in sorted({r["site"] for r in group}):
                def usable(r):
                    label = "vna" if variant == "full" else "lite"
                    return (r.get(label+"_valid", False) and not r.get(label+"_empty", True)
                            and not r.get(prefix+"nonfinite", True) and finite(r.get(label+"_derivative"))
                            and all(finite(r.get(prefix+k)) for k in ("U", "V", "rho")))
                a = [r for r in early if r["site"] == site and usable(r)]
                b = [r for r in group if r["site"] == site and r["epoch"] >=180 and usable(r)]
                relevant = [r for r in group if r["site"] == site and (r["epoch"] <150 or r["epoch"] >=180)]
                decay_eligible += sum(usable(r) for r in relevant)
                decay_excluded += sum(not usable(r) for r in relevant)
                trends[site] = {k: statistics.mean(r[prefix+k] for r in b) <= .1*statistics.mean(r[prefix+k] for r in a)
                                if a and b else None for k in ("U", "V", "rho")}
            item[variant+"_decay_eligible"] = decay_eligible
            item["excluded_"+variant+"_decay"] = decay_excluded
            item[variant+"_decay_by_site"] = trends
            item[variant+"_decay"] = bool(trends) and all(v is True for t in trends.values() for v in t.values())
            item[variant+"_go"] = bool(item[variant+"_joint_rows"] and item[variant+"_decay"])
        item["complete"] = max(r["epoch"] for r in group) >=200 and all(
            v is not None for variant in ("full", "lite") for t in item[variant+"_decay_by_site"].values() for v in t.values())
        for key in ("cold_sgd_ratio", "warm_sgd_ratio", "lite_sgd_ratio"):
            costs = [r[key] for r in group if finite(r.get(key))]
            item[key] = statistics.mean(costs) if costs else None
        result.append(item)
    return result


def write_outputs(rows, args):
    cfg = VNAConfig(seed=args.seed, bn_mode=args.vna_bn, cache_block_input=args.cache_block_input)
    if args.smoke:
        cfg = replace(cfg, stat_samples=16, vna_batch=2, r_max=2, cg_iters=10)
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
    valid = [r for r in rows if not r["nonfinite"] and finite(r.get("rho"))]
    summary = {"config": asdict(cfg), "diagnostic_stride": args.diagnostic_stride,
               "rows": len(rows), "nonfinite_rows": sum(r["nonfinite"] for r in rows),
               "fake_data": args.fake_data, "smoke": args.smoke,
               "rho_above_0_3_fraction": sum(r["rho"] > .3 for r in valid) / max(1, len(valid)),
               "skipped_nonpositive_gain": sum(r.get("skipped", False) for r in rows),
               "significance_rule": "full and split-half lambda > max of 3 full-stat-sample shuffled nulls",
               "significance_by_checkpoint_site": [{k:r[k] for k in (
                   "epoch", "site", "lambda_full", "lambda_split_half", "lambda_shuffled_mean",
                   "lambda_shuffled_max", "significant_vs_shuffled", "significant_split_half")}
                   for r in rows if r.get("row_kind") == "default"],
               "grow_epochs": "TODO: paired persistent-growth experiment deferred; no growth installed"}
    table = rank_summary(rows)
    summary["per_rank"] = table
    summary["decay_rule"] = "mean epochs >=180 <= 0.1 * mean epochs <150, per requested rank/site and U/V/rho"
    complete = bool(table) and all(t["complete"] for t in table)
    summary["phase0_acceptance"] = ("not assessed: synthetic/partial run" if args.fake_data or args.smoke or not complete
        else "go" if any(t["full_go"] or t["lite_go"] for t in table) else "no-go")
    print("r | full V/derivative/LR (pass/eligible; excluded), decay | lite V/derivative/LR, decay | cold/warm/lite SGD cost", flush=True)
    for t in table:
        def cells(variant):
            outcomes = " ".join(f"{t[variant+'_'+c+'_passed']}/{t[variant+'_'+c+'_eligible']} (x{t['excluded_'+variant+'_'+c]})"
                            for c in ("V", "derivative", "lr_boost"))
            decay = t[variant+"_decay"] if all(v is not None for trend in t[variant+"_decay_by_site"].values() for v in trend.values()) else "n/a"
            return outcomes + f", decay={decay}"
        print(f"{t['requested_rank']} | {cells('full')} | {cells('lite')} | "
              f"{t['cold_sgd_ratio']} / {t['warm_sgd_ratio']} / {t['lite_sgd_ratio']}", flush=True)
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
        "channels_last", "cudnn_benchmark", "diagnostic_stride")}
    identity["schema_version"] = "stage1c-v1"
    identity["amp"] = False  # VNA products and solves always use fp32.
    return identity


def validate_diagnostic_identity(previous, current):
    # Legacy source fingerprints are code provenance, not experiment config.
    validate_resume_protocol({k:v for k,v in previous.items() if k != 'source'}, current)


def run(args):
    """Atomic per-checkpoint transactions; CSV is a rebuildable materialization."""
    from copy import copy
    output = Path(args.output)
    done = output / "completed"
    done.mkdir(parents=True, exist_ok=True)
    if args.diagnostic_stride < 1:
        raise ValueError("diagnostic_stride must be positive")
    identity = diagnostic_identity(args)
    # Reject every old marker before filtering epochs or starting new work.
    for marker in done.glob("epoch_*.json"):
        validate_diagnostic_identity(json.loads(marker.read_text())["identity"], identity)
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
            if epoch % args.diagnostic_stride:
                continue
            if (epoch // args.diagnostic_stride) % args.shards != args.shard:
                continue
            marker = done / f"epoch_{epoch:03d}.json"
            if marker.exists():
                previous = json.loads(marker.read_text())
                validate_diagnostic_identity(previous["identity"], identity)
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
                atomic_json_save({"identity": identity, "model_sha256": digest, "epoch": epoch, "rows": rows, "code_commit": code_commit()}, marker)
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
    if args.diagnostic_stride < 1:
        raise ValueError("diagnostic_stride must be positive")
    rows = []
    identities = set()
    for path in sorted((Path(args.output)/"completed").glob("epoch_*.json")):
        payload = json.loads(path.read_text())
        validate_diagnostic_identity(payload["identity"], diagnostic_identity(args))
        identities.add(json.dumps({k:v for k,v in payload["identity"].items() if k != "source"}, sort_keys=True))
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
    p.add_argument("--diagnostic-stride", type=int, default=20)
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
