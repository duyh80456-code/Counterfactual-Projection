"""CPU analysis of selected-site E-to-O intervention logs.

No per-site residuals or persistent-growth labels are inferred from WHERE scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MAPPING = {
    "selected_site": ["selected_site"],
    "r_heldout": ["heldout_relative_residual"],
    "cos_heldout": ["heldout_cosine_alignment"],
    "applied": ["correction_applied"],
    "realized_gain": ["actual_loss_improvement"],
    "actual_cosine": ["actual_cosine_alignment", "actual_heldout_cosine_alignment"],
    "selected_scale": ["selected_scale"],
    "epoch": ["epoch"],
    "probe_index": ["probe_index"],
}
E_METHODS = {"ours_e_driven_o", "e_driven_o", "e_projection"}


def _documents(path):
    text = path.read_text()
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        documents = []
        for line in text.splitlines():
            start = line.find("{")
            if start >= 0:
                try:
                    documents.append(json.loads(line[start:]))
                except json.JSONDecodeError:
                    pass
        if not documents:
            raise ValueError(f"No JSON records in {path}")
        return documents


def _events(document, inherited=None):
    metadata = dict(inherited or {})
    if isinstance(document, list):
        for item in document:
            yield from _events(item, metadata)
        return
    if not isinstance(document, dict):
        return
    protocol = document.get("protocol") or {}
    for key in ("architecture", "backbone", "seed", "method", "theta_best_hash",
                "plateau_checkpoint_hash", "run_id"):
        value = document.get(key, protocol.get(key))
        if value is not None:
            metadata[key] = value
    if isinstance(document.get("interventions"), list):
        for event in document["interventions"]:
            yield event, metadata
        return  # latest intervention/history are duplicate views of this list
    if isinstance(document.get("intervention"), dict):
        yield document["intervention"], metadata
    if isinstance(document.get("e_driven_o_intervention"), dict):
        metadata["method"] = "ours_e_driven_o"
        yield document["e_driven_o_intervention"], metadata
    elif "selected_site" in document and (
            "heldout_relative_residual" in document or
            "correction_applied" in document):
        yield document, metadata
    for key in ("history", "ours_e_driven_o"):
        if key in document:
            child_meta = dict(metadata)
            if key == "ours_e_driven_o":
                child_meta["method"] = key
            yield from _events(document[key], child_meta)


def _backbone(value):
    value = str(value).lower()
    for canonical, aliases in {
        "R18": ("resnet18", "resnet-18", "r18"),
        "R34": ("resnet34", "resnet-34", "r34"),
        "VGG": ("vgg",),
    }.items():
        if any(alias in value for alias in aliases):
            return canonical
    return "unknown"


def _mapped(event, keys):
    for key in keys if isinstance(keys, list) else [keys]:
        value = event
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if value is not None:
            return value
    return None


def load_interventions(root, patterns, *, mapping=None, metadata_overrides=None):
    """One row per selected-site intervention; print observed keys first.

    metadata_overrides maps paths relative to root to explicit run metadata.
    Missing or ambiguous metadata remains unknown; non-E branches are excluded.
    """
    root = Path(root)
    mapping = {**DEFAULT_MAPPING, **(mapping or {})}
    overrides = metadata_overrides or {}
    paths = sorted({p for pattern in patterns for p in root.glob(pattern)
                    if p.is_file()})
    records, inventory = [], set()
    for path in paths:
        for document in _documents(path):
            for event, metadata in _events(document):
                if not isinstance(event, dict):
                    continue
                inventory.update(event)
                records.append((path, event, metadata))
    print("Available intervention keys:", json.dumps(sorted(inventory)))
    print("Configured mapping:", json.dumps(mapping, sort_keys=True))
    rows, seen = [], set()
    for path, event, metadata in records:
        relative = str(path.relative_to(root))
        metadata = {**metadata, **overrides.get(relative, {})}
        method = metadata.get("method")
        # Standalone records require a structural-E selector or explicit metadata.
        structural = event.get("uses_structural_E") is True or (
            "site_evaluations" in event or "selected_e_gain" in event)
        if method not in E_METHODS and not (method is None and structural):
            continue
        row = {name: _mapped(event, keys) for name, keys in mapping.items()}
        backbone = _backbone(metadata.get("backbone", metadata.get(
            "architecture", relative)))
        seed = metadata.get("seed")
        if seed is None:
            match = re.search(r"seed[_-]?(\d+)", relative, re.I)
            seed = int(match.group(1)) if match else None
        run = metadata.get("run_id") or metadata.get("theta_best_hash") or (
            metadata.get("plateau_checkpoint_hash")) or str(path.parent.relative_to(root))
        # Snapshot/result and JSONL views of the same event are counted once.
        identity = (str(run), backbone, str(seed), row.get("probe_index"),
                    row.get("epoch"), row.get("selected_site"))
        if row.get("probe_index") is None and row.get("epoch") is None:
            identity += (hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest(),)
        if identity in seen:
            continue
        seen.add(identity)
        row.update(backbone=backbone, seed=seed, run_id=str(run),
                   source_file=relative, method=method or "structural_E_inferred",
                   diagnostic_scope="selected_site_only",
                   heldout_role="logged_batch_independence_unverified")
        rows.append(row)
    columns = list(mapping) + ["backbone", "seed", "run_id", "source_file",
                               "method", "diagnostic_scope", "heldout_role"]
    df = pd.DataFrame(rows, columns=columns)
    for column in ("r_heldout", "cos_heldout", "realized_gain", "actual_cosine",
                   "selected_scale", "epoch", "probe_index"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[~np.isfinite(df[column]), column] = np.nan
    df["applied"] = df["applied"].map(
        lambda v: v if isinstance(v, bool) else None).astype("boolean")
    df.attrs["key_inventory"] = sorted(inventory)
    df.attrs["mapping"] = mapping
    return df


def summarize_sites(df):
    counts = df.groupby(["backbone", "selected_site"], dropna=False).size().rename(
        "count").reset_index()
    counts["fraction"] = counts["count"] / counts.groupby("backbone")["count"].transform("sum")
    return counts


def spearman_bootstrap(frame, x, y, *, bootstrap=2000, seed=0):
    pairs = frame[[x, y]].dropna()
    n = len(pairs)
    result = {"n": n, "rho": None, "ci_low": None, "ci_high": None,
              "bootstrap_valid": 0, "status": "insufficient_or_constant"}
    if n < 3 or pairs[x].nunique() < 2 or pairs[y].nunique() < 2:
        return result
    def rho(values):
        ranks = pd.DataFrame(values).rank().to_numpy()
        if np.any(np.std(ranks, axis=0) == 0):
            return np.nan
        return float(np.corrcoef(ranks.T)[0, 1])
    values = pairs.to_numpy()
    result["rho"] = rho(values)
    rng = np.random.default_rng(seed)
    estimates = [rho(values[rng.integers(n, size=n)]) for _ in range(bootstrap)]
    estimates = np.asarray(estimates)
    estimates = estimates[np.isfinite(estimates)]
    result["bootstrap_valid"] = len(estimates)
    if len(estimates):
        result["ci_low"], result["ci_high"] = map(float, np.quantile(estimates, [0.025, 0.975]))
    result["status"] = "small_n_descriptive_only" if n < 20 else "descriptive_only"
    return result


def write_outputs(df, output, *, bootstrap=2000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "interventions.csv", index=False)
    sites = summarize_sites(df)
    sites.to_csv(output / "selected_sites.csv", index=False)
    summary = df.groupby("backbone", dropna=False).agg(
        interventions=("backbone", "size"), applied_known=("applied", "count"),
        applied_rate=("applied", "mean"), r_median=("r_heldout", "median"),
        cos_median=("cos_heldout", "median"))
    summary.to_csv(output / "backbone_summary.csv")
    correlations = []
    groups = [("backbone", name, group) for name, group in df.groupby("backbone")]
    groups += [("backbone_seed", f"{backbone}/seed={seed_id}", group)
               for (backbone, seed_id), group in df.dropna(subset=["seed"]).groupby(["backbone", "seed"])]
    for scope, name, group in groups:
        for subset in ("all", "applied_only"):
            selected = group if subset == "all" else group[group.applied.fillna(False)]
            for x in ("r_heldout", "cos_heldout"):
                correlations.append({"scope": scope, "group": name, "subset": subset,
                                     "x": x, "y": "realized_gain",
                                     **spearman_bootstrap(selected, x, "realized_gain",
                                                          bootstrap=bootstrap, seed=seed)})
    pd.DataFrame(correlations).to_csv(output / "spearman.csv", index=False)
    backbones = sorted(df.backbone.unique())
    if backbones:
        fig, axes = plt.subplots(len(backbones), 1, figsize=(10, 4 * len(backbones)), squeeze=False)
        for ax, backbone in zip(axes.flat, backbones):
            selected = sites[sites.backbone == backbone].sort_values("count")
            ax.barh(selected.selected_site.fillna("unknown"), selected["count"])
            ax.set(title=f"{backbone}: selected sites", xlabel="Interventions")
        fig.tight_layout()
        fig.savefig(output / "selected_sites.png", dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(len(backbones), 2, figsize=(10, 3 * len(backbones)), squeeze=False)
        for row, backbone in enumerate(backbones):
            group = df[df.backbone == backbone]
            for col, metric in enumerate(("r_heldout", "cos_heldout")):
                axes[row, col].hist(group[metric].dropna(), bins=15)
                axes[row, col].set(title=f"{backbone}: {metric}", ylabel="Interventions")
        fig.tight_layout()
        fig.savefig(output / "heldout_distributions.png", dpi=160)
        plt.close(fig)
    report = {
        "key_inventory": df.attrs.get("key_inventory", []),
        "mapping": df.attrs.get("mapping", {}), "rows": len(df),
        "notes": ["Residuals describe selected sites only, not all WHERE sites.",
                  "realized_gain is immediate logged gate-batch loss reduction, not PG_gain or accuracy gain.",
                  "Logged heldout batch may also select scale; independence is not established.",
                  "Bootstrap resamples interventions; repeated trials within a run may be dependent.",
                  "Correlations are descriptive; no capacity-need conclusion, especially for small n.",
                  "Unknown seed/backbone metadata should be supplied through metadata_overrides."]}
    (output / "analysis_metadata.json").write_text(json.dumps(report, indent=2))
    print(summary.to_string())
    print(f"Saved {len(df)} interventions to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--patterns", nargs="+", default=["**/result.json"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON with mapping and metadata_overrides")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    config = json.loads(args.config.read_text()) if args.config else {}
    df = load_interventions(args.root, args.patterns, mapping=config.get("mapping"),
                            metadata_overrides=config.get("metadata_overrides"))
    if df.empty:
        parser.error("No E-to-O interventions found; inspect patterns, keys, and metadata.")
    write_outputs(df, args.output, bootstrap=args.bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
