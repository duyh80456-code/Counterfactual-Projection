"""Join independent site diagnostics with PG labels within each seed."""
import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.analyze_projection_diagnostic import spearman_bootstrap


def summarize_capacity(projection_files, growth_files, *, bootstrap=2000):
    probes, labels = [], []
    for path in projection_files:
        payload = json.loads(Path(path).read_text())
        for record in payload["records"]:
            probes.append({"checkpoint_hash": payload["checkpoint_hash"],
                "architecture": payload["architecture"], "seed": payload["seed"],
                "site": record["site"], **record["true_direction"],
                "r_random_heldout": record.get("random_control", {}).get("r_E_heldout")})
    for path in growth_files:
        payload = json.loads(Path(path).read_text())
        for record in payload.get("records", [payload]):
            labels.append({key: record[key] for key in (
                "checkpoint_hash", "architecture", "seed", "site", "PG_gain", "PG_loss_gain", "horizon")})
    keys = ["checkpoint_hash", "architecture", "seed", "site"]
    merged = pd.DataFrame(probes).merge(pd.DataFrame(labels), on=keys, validate="one_to_one")
    merged["r_true_minus_random"] = merged.r_E_heldout - merged.r_random_heldout
    correlations = []
    for (architecture, seed, checkpoint_hash, horizon), group in merged.groupby(
            ["architecture", "seed", "checkpoint_hash", "horizon"], dropna=False):
        for x in ("r_E_heldout", "cos_E_heldout"):
            correlations.append({"architecture": architecture, "seed": seed,
                "checkpoint_hash": checkpoint_hash, "horizon": horizon, "x": x,
                **spearman_bootstrap(group, x, "PG_gain", bootstrap=bootstrap)})
    return merged, pd.DataFrame(correlations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", nargs="+", required=True, type=Path)
    parser.add_argument("--growth", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merged, correlations = summarize_capacity(args.projection, args.growth)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output / "capacity_labels.csv", index=False)
    correlations.to_csv(args.output / "within_seed_spearman.csv", index=False)
    # Report the seed-level estimates individually; no pooled site correlation.
    aggregation = correlations.groupby(["architecture", "horizon", "x"]).agg(
        seed_runs=("rho", "count"), median_rho=("rho", "median"),
        min_rho=("rho", "min"), max_rho=("rho", "max"))
    aggregation.to_csv(args.output / "across_seed_summary.csv")


if __name__ == "__main__":
    main()
