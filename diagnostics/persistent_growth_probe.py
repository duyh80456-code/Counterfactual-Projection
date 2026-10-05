"""Matched short-horizon persistent structural extension versus Vanilla.

The candidate's Gromo extension remains registered for the entire growth arm.
Scale is optimized on WHERE by a grid and bounded numerical refinement in
virtual_direction's output scaling convention; this does not claim an
unconstrained native Gromo scalar optimum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.shared_protocol import (evaluate, make_train_loader,
    make_eval_loader, restore_rng, rng_state, train_epoch)
from diagnostics.protocol import load_context, select_batches, propose


def optimize_extension_scale(model, candidate, batch, scales, refine_steps=20):
    scales = sorted(set(float(scale) for scale in scales))
    if not scales or any(scale < 0 or scale > 1 for scale in scales):
        raise ValueError("extension scales must lie in [0, 1]")
    modes = {module: module.training for module in model.modules()}
    losses = {}
    try:
        model.eval()
        with torch.no_grad():
            def loss_at(scale):
                with candidate.virtual_direction(scale):
                    losses[scale] = float(F.cross_entropy(model(batch[0]), batch[1]))
                return losses[scale]
            for scale in scales:
                loss_at(scale)
            best_index = min(range(len(scales)), key=lambda i: losses[scales[i]])
            low = scales[max(0, best_index - 1)]
            high = scales[min(len(scales) - 1, best_index + 1)]
            # Refine the best grid basin in the candidate's Gromo output
            # scaling convention. This is a bounded numerical CE optimum.
            ratio = (5 ** 0.5 - 1) / 2
            for _ in range(refine_steps if high > low else 0):
                left = high - ratio * (high - low)
                right = low + ratio * (high - low)
                if loss_at(left) < loss_at(right):
                    high = right
                else:
                    low = left
    finally:
        for module, mode in modes.items():
            module.training = mode
    scale = min(losses, key=losses.get)
    return scale, losses


def persistent_growth_gain(checkpoint, site, horizon, *, data_root,
                           reference_root, architecture, device="cuda:0",
                           rank=4, seed=0, batch_size=64, workers=0,
                           scales=(0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0),
                           output=None):
    if horizon < 1:
        raise ValueError("horizon must be positive")
    context = load_context(checkpoint, data_root, reference_root, architecture, device)
    batches, indices = select_batches(context, seed=seed)
    candidates = propose(context.model, batches["statistics"], rank, site)
    if len(candidates) != 1:
        raise ValueError("persistent growth requires exactly one site candidate")
    candidate = candidates[0]
    scale, scale_losses = optimize_extension_scale(
        context.model, candidate, batches["where"], scales)
    source = context.source
    train_ids = source["train_indices"]
    evaluation_ids = source["evaluation_indices"]
    base_params = sum(p.numel() for p in context.model.parameters())

    def train_arm(arm_context, *, grown=False):
        # Proposal/scale selection must not affect either arm's training stream.
        restore_rng(source["rng"])
        train_loader = make_train_loader(arm_context.train_set, train_ids,
            batch_size, workers, source["train_loader_generator_state"], seed)
        val_loader = make_eval_loader(arm_context.eval_set, evaluation_ids,
                                     2 * batch_size, workers)
        inherited = {id(p) for group in arm_context.optimizer.param_groups
                     for p in group["params"]}
        added = [p for p in arm_context.model.parameters() if id(p) not in inherited]
        if grown:
            # Register extension weights with inherited SGD hyperparameters;
            # existing weights keep their fork momentum and scheduler state.
            if not added:
                raise RuntimeError("candidate context registered no extension parameters")
            arm_context.optimizer.param_groups[0]["params"].extend(added)
        start = evaluate(arm_context.model, val_loader, arm_context.device)
        best = {**start, "epoch": 0}
        history = []
        for epoch in range(1, horizon + 1):
            train = train_epoch(arm_context.model, train_loader,
                                arm_context.optimizer, arm_context.device)
            validation = evaluate(arm_context.model, val_loader, arm_context.device)
            arm_context.scheduler.step()
            history.append({"epoch": epoch, **validation,
                            "train_loss": train["task_loss"]})
            if validation["accuracy"] > best["accuracy"]:
                best = {**validation, "epoch": epoch}
        params = sum(p.numel() for p in arm_context.model.parameters())
        if grown and params <= base_params:
            raise RuntimeError("growth arm did not preserve additional capacity")
        if output:
            destination = Path(output)
            destination.mkdir(parents=True, exist_ok=True)
            torch.save({"model": arm_context.model.state_dict(),
                        "optimizer": arm_context.optimizer.state_dict(),
                        "scheduler": arm_context.scheduler.state_dict(),
                        "rng": rng_state(),
                        "train_loader_generator_state": train_loader.generator.get_state(),
                        "site": site, "scale": scale if grown else None,
                        "extension_active": grown, "horizon": horizon,
                        "checkpoint_hash": context.checkpoint_hash,
                        "reconstruction": "repropose candidate at source theta_P, enter virtual_direction(scale), then load state"},
                       destination / ("growth_final.pt" if grown else "vanilla_final.pt"))
        return {"best_accuracy": best["accuracy"],
                "loss_at_best_accuracy": best["loss"], "best_epoch": best["epoch"],
                "min_loss": min([start["loss"]] + [row["loss"] for row in history]),
                "final_accuracy": history[-1]["accuracy"],
                "final_loss": history[-1]["loss"], "parameters": params,
                "history": history}

    vanilla_context = load_context(checkpoint, data_root, reference_root, architecture, device)
    vanilla = train_arm(vanilla_context)
    # The temporary-probe context is kept open throughout H training epochs:
    # no collapse/projection to original O capacity occurs during this arm.
    with candidate.virtual_direction(scale):
        growth = train_arm(context, grown=True)
    result = {"site": site, "horizon": horizon, "architecture": architecture,
              "seed": source.get("protocol", {}).get("seed"),
              "diagnostic_seed": seed,
              "checkpoint_hash": context.checkpoint_hash,
              "candidate_rank": rank, "extension_scale": scale,
              "scale_selection": "WHERE bounded CE line search: grid then golden-section refinement",
              "scale_losses": scale_losses, "split_indices": indices,
              "growth_representation": "registered extension active throughout horizon",
              "vanilla": vanilla, "persistent_growth": growth,
              "PG_gain": growth["best_accuracy"] - vanilla["best_accuracy"],
              "PG_loss_gain": vanilla["min_loss"] - growth["min_loss"]}
    if output:
        (Path(output) / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    from probe import CandidateExpansionProbe
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--architecture", choices=("resnet18", "resnet34", "vgg16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sites", nargs="+", help="otherwise top-3 and bottom-2 WHERE sites")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--bottom", type=int, default=2)
    parser.add_argument("--resume-completed", action="store_true",
                        help="reuse completed per-site results with matching fork/config")
    args = parser.parse_args()
    if args.top < 0 or args.bottom < 0:
        parser.error("site counts must be nonnegative")
    context = load_context(args.checkpoint, args.data_root, args.reference_root,
                           args.architecture, args.device)
    checkpoint_hash = context.checkpoint_hash
    batches, _ = select_batches(context, seed=args.seed)
    ranked = []
    if not args.sites:
        for candidate in propose(context.model, batches["statistics"], args.rank):
            signal = CandidateExpansionProbe()(context.model, candidate=candidate,
                                               batch=batches["where"], gate=0.05)
            ranked.append({"site": candidate.module_name, "where_gain": signal.observed_loss_gain})
        ranked.sort(key=lambda row: row["where_gain"], reverse=True)
        selection = ranked[:args.top] + (ranked[-args.bottom:] if args.bottom else [])
        sites = list(dict.fromkeys(row["site"] for row in selection))
    else:
        sites = args.sites
    # Free the selection model before allocating two matched training arms.
    del context
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    records = []
    for site in sites:
        prior = args.output / site / "result.json"
        if args.resume_completed and prior.is_file():
            saved = json.loads(prior.read_text())
            expected = {"checkpoint_hash": checkpoint_hash,
                        "site": site, "horizon": args.horizon,
                        "candidate_rank": args.rank,
                        "diagnostic_seed": args.seed,
                        "architecture": args.architecture}
            if all(saved.get(key) == value for key, value in expected.items()):
                records.append(saved)
                print(f"Reusing completed growth comparison: {site}", flush=True)
                continue
        records.append(persistent_growth_gain(args.checkpoint, site, args.horizon,
            data_root=args.data_root, reference_root=args.reference_root,
            architecture=args.architecture, device=args.device, rank=args.rank,
            seed=args.seed, output=args.output / site))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({
        "where_ranking": ranked, "records": records}, indent=2))


if __name__ == "__main__":
    main()
