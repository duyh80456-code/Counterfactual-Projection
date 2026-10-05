"""Independent held-out site projection diagnostics and matched random controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from methods.e_projection import (
    candidate_projection_block, candidate_projection_parameter_names)
from probe import CandidateExpansionProbe
from projection import FunctionalProjector


def site_projection_probe(model, candidate, projection_batch, heldout_batch, *,
                          random_control=True, gate=0.05, projector=None,
                          random_seed=0, min_heldout_samples=256):
    if heldout_batch[0].shape[0] < min_heldout_samples:
        raise ValueError(f"heldout batch must contain >= {min_heldout_samples} samples")
    projector = projector or FunctionalProjector()
    probe = CandidateExpansionProbe()
    fit_target = probe(model, candidate=candidate, batch=projection_batch,
                       gate=gate).delta_logits
    heldout_target = probe(model, candidate=candidate, batch=heldout_batch,
                           gate=gate).delta_logits
    block = candidate_projection_block(model, candidate)
    names = candidate_projection_parameter_names(model, candidate)

    def measure(fit, heldout):
        result = projector.project(model, projection_batch[0], fit,
                                   block=block, parameter_names=names)
        evaluation = projector.evaluate_direction(
            model, heldout_batch[0], heldout, result.parameter_delta)
        return {"r_fit": float(result.relative_residual),
                "cos_fit": float(result.cosine_alignment),
                "r_E_heldout": float(evaluation.relative_residual),
                "cos_E_heldout": float(evaluation.cosine_alignment),
                "fit_target_norm": float(fit.norm()),
                "heldout_target_norm": float(heldout.norm()),
                "damping_used": float(result.damping_used),
                "cg_converged": bool(result.cg.converged)}

    output = {"site": str(candidate.module_name), "gate": gate,
              "projection_samples": len(projection_batch[0]),
              "heldout_samples": len(heldout_batch[0]),
              "true_direction": measure(fit_target, heldout_target)}
    if random_control:
        generator = torch.Generator(device=fit_target.device).manual_seed(random_seed)
        def random_like(target):
            direction = torch.randn(target.shape, device=target.device,
                                    dtype=target.dtype, generator=generator)
            return direction * (target.norm() / direction.norm().clamp_min(1e-12))
        output["random_control"] = measure(random_like(fit_target), random_like(heldout_target))
        output["random_control_definition"] = (
            "independent Gaussian logit directions on fit/heldout, each matched to true target norm")
        output["random_seed"] = random_seed
    return output


def main():
    from diagnostics.protocol import load_context, select_batches, propose
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--architecture", choices=("resnet18", "resnet34", "vgg16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    context = load_context(args.checkpoint, args.data_root, args.reference_root,
                           args.architecture, args.device)
    batches, indices = select_batches(context, seed=args.seed)
    candidates = propose(context.model, batches["statistics"], args.rank)
    records = [site_projection_probe(context.model, candidate, batches["projection"],
                                    batches["heldout"], random_seed=args.seed)
               for candidate in candidates]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"architecture": args.architecture,
        "seed": context.source.get("protocol", {}).get("seed"),
        "checkpoint_hash": context.checkpoint_hash, "split_indices": indices,
        "records": records}, indent=2))


if __name__ == "__main__":
    main()
