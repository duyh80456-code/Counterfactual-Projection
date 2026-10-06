"""CPU analysis of selected-site E-to-O intervention logs.

No per-site residuals or persistent-growth labels are inferred from WHERE scores.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MAPPING = {
    "selected_site": ["selected_site"],
    "r_heldout": ["heldout_relative_residual"],
    "cos_heldout": ["heldout_cosine_alignment"],
    "r_fit": ["selected_cg_attempt.functional_relative_residual", "relative_residual"],
    "cos_fit": ["selected_cg_attempt.functional_cosine_alignment", "cosine_alignment"],
    "r_actual": ["actual_relative_residual", "actual_heldout_relative_residual"],
    "applied": ["correction_applied"],
    "realized_gain": ["actual_loss_improvement"],
    "actual_cosine": ["actual_cosine_alignment", "actual_heldout_cosine_alignment"],
    "selected_scale": ["selected_scale"],
    "epoch": ["epoch"],
    "probe_index": ["probe_index"],
}
DEFAULT_HISTORY_MAPPING = {"epoch": ["epoch"],
                           "accuracy": ["validation_accuracy"]}
METRICS = ("r_fit", "cos_fit", "r_heldout", "cos_heldout", "r_actual", "actual_cosine")
E_METHODS = {"ours_e_driven_o", "e_driven_o", "e_projection"}


def _metadata(document, inherited=None):
    metadata = dict(inherited or {})
    configuration = {**(document.get("config") or {}),
                     **(document.get("protocol") or {}), **document}
    for key in ("architecture", "backbone", "seed", "method", "theta_best_hash",
                "plateau_checkpoint_hash", "run_id"):
        if configuration.get(key) is not None:
            metadata[key] = configuration[key]
    return metadata


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
    metadata = _metadata(document, metadata)
    if isinstance(document.get("interventions"), list):
        for event in document["interventions"]:
            yield event, metadata
        return  # latest intervention/history are duplicate views of this list
    if isinstance(document.get("intervention"), dict):
        yield document["intervention"], metadata
    if isinstance(document.get("e_driven_o_intervention"), dict):
        metadata["method"] = "ours_e_driven_o"
        event = dict(document["e_driven_o_intervention"])
        if "epoch" not in event and "epoch" in document:
            event["epoch"] = document["epoch"]
        yield event, metadata
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


def _source_key(event, keys):
    return next((key for key in keys if _mapped(event, [key]) is not None), None)


def _matches_pattern(parts, pattern):
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return _matches_pattern(parts, pattern[1:]) or bool(
            parts and _matches_pattern(parts[1:], pattern))
    return bool(parts and fnmatch.fnmatch(parts[0], pattern[0]) and
                _matches_pattern(parts[1:], pattern[1:]))


def _history_rows(document, inherited=None):
    if isinstance(document, list):
        for item in document:
            yield from _history_rows(item, inherited)
    elif isinstance(document, dict):
        metadata = _metadata(document, inherited)
        yield document, metadata
        protocol = document.get("protocol") or {}
        if document.get("fork_validation_accuracy") is not None:
            yield {"epoch": protocol.get("fork_epoch"),
                   "validation_accuracy": document["fork_validation_accuracy"]}, metadata
        for key in ("history", "ours_e_driven_o"):
            if key in document:
                child = dict(metadata)
                if key == "ours_e_driven_o":
                    child["method"] = key
                yield from _history_rows(document[key], child)


def _run_metadata(root, path, metadata, overrides, rules):
    relative = path.relative_to(root).as_posix()
    inferred = {}
    for rule in rules:
        if fnmatch.fnmatch(relative, rule["pattern"]):
            inferred.update(rule["metadata"])
    # Run JSON wins over directory rules; exact overrides are authoritative.
    known_metadata = {key: value for key, value in metadata.items()
                      if value is not None and str(value).lower() != "unknown"}
    metadata = {**inferred, **known_metadata, **overrides.get(relative, {})}
    backbone = _backbone(metadata.get("backbone", metadata.get("architecture")))
    if backbone == "unknown":
        backbone = _backbone(relative)
    seed = metadata.get("seed")
    if seed is None:
        match = re.search(r"seed[_-]?(\d+)", relative, re.I)
        seed = int(match.group(1)) if match else None
    # A fork hash identifies a starting state, not a distinct training run.
    run = metadata.get("run_id") or str(path.parent.relative_to(root))
    return metadata, backbone, seed, str(run)


def _seed_key(seed):
    if seed is None or pd.isna(seed):
        return "unknown"
    try:
        return str(int(seed))
    except (ValueError, TypeError):
        return str(seed)


def _attach_horizons(df, timelines, horizons):
    for column in ("acc_at_intervention",):
        df[column] = np.nan
    df["acc_baseline_status"] = "missing"
    for horizon in horizons:
        df[f"acc_after_{horizon}"] = np.nan
        df[f"acc_gain_{horizon}"] = np.nan
        df[f"acc_{horizon}_status"] = "missing_epoch"
    for _, group in df.groupby(["run_id", "backbone", "seed"], dropna=False):
        epochs = sorted(group.epoch.dropna().unique())
        for index, row in group.iterrows():
            if pd.isna(row.epoch) or not float(row.epoch).is_integer():
                continue
            epoch = int(row.epoch)
            history = timelines.get((row.run_id, row.backbone, _seed_key(row.seed)), {})

            def accuracy_at(target):
                values = history.get(target, set())
                if len(values) == 1:
                    return next(iter(values)), "observed"
                return np.nan, "conflicting_history" if values else "missing_epoch"

            baseline, status = accuracy_at(epoch)
            df.loc[index, "acc_at_intervention"] = baseline
            df.loc[index, "acc_baseline_status"] = status
            next_epoch = next((value for value in epochs if value > epoch), None)
            for horizon in horizons:
                value, status = accuracy_at(epoch + horizon)
                # Keep the observed endpoint, but exclude confounded windows
                # from the per-intervention horizon correlations below.
                if next_epoch is not None and next_epoch <= epoch + horizon:
                    status = "another_intervention_in_window"
                df.loc[index, f"acc_after_{horizon}"] = value
                df.loc[index, f"acc_gain_{horizon}"] = value - baseline
                df.loc[index, f"acc_{horizon}_status"] = status


def load_interventions(root, patterns, *, mapping=None, metadata_overrides=None,
                       metadata_rules=None, history_mapping=None, horizons=(1, 5, 15)):
    """One row per selected-site intervention; print observed keys first.

    metadata_overrides maps paths relative to root to explicit run metadata.
    Missing or ambiguous metadata remains unknown; non-E branches are excluded.
    """
    root = Path(root)
    mapping = {**DEFAULT_MAPPING, **(mapping or {})}
    mapping = {name: keys if isinstance(keys, list) else [keys]
               for name, keys in mapping.items()}
    overrides = metadata_overrides or {}
    rules = metadata_rules or []
    history_mapping = {**DEFAULT_HISTORY_MAPPING, **(history_mapping or {})}
    paths = set()
    # Attached Kaggle outputs can be directory symlinks; Path.glob does not
    # descend into those. Match glob segments after walking mounted directories.
    visited = set()
    for directory, subdirectories, filenames in os.walk(root, followlinks=True):
        resolved = Path(directory).resolve()
        if resolved in visited:
            subdirectories[:] = []
            continue
        visited.add(resolved)
        for filename in filenames:
            path = Path(directory) / filename
            parts = path.relative_to(root).parts
            if any(_matches_pattern(parts, tuple(pattern.split("/"))) for pattern in patterns):
                paths.add(path)
    paths = sorted(paths)
    records, inventory, history_inventory, timelines = [], set(), set(), {}
    for path in paths:
        for document in _documents(path):
            for record, metadata in _history_rows(document):
                history_inventory.update(record)
                metadata, backbone, seed, run = _run_metadata(
                    root, path, metadata, overrides, rules)
                if metadata.get("method") not in E_METHODS and metadata.get("method") is not None:
                    continue
                epoch = _mapped(record, history_mapping["epoch"])
                accuracy = _mapped(record, history_mapping["accuracy"])
                try:
                    epoch, accuracy = float(epoch), float(accuracy)
                    if not epoch.is_integer() or not np.isfinite(accuracy):
                        continue
                except (TypeError, ValueError):
                    continue
                timelines.setdefault((run, backbone, _seed_key(seed)), {}).setdefault(
                    int(epoch), set()).add(accuracy)
            for event, metadata in _events(document):
                if not isinstance(event, dict):
                    continue
                inventory.update(event)
                records.append((path, event, metadata))
    print("Available intervention keys:", json.dumps(sorted(inventory)))
    print("Configured mapping:", json.dumps(mapping, sort_keys=True))
    print("Available history keys:", json.dumps(sorted(history_inventory)))
    print("History mapping:", json.dumps(history_mapping, sort_keys=True))
    rows, seen = [], set()
    for path, event, metadata in records:
        relative = str(path.relative_to(root))
        metadata, backbone, seed, run = _run_metadata(root, path, metadata, overrides, rules)
        method = metadata.get("method")
        # Standalone records require a structural-E selector or explicit metadata.
        structural = event.get("uses_structural_E") is True or (
            "site_evaluations" in event or "selected_e_gain" in event)
        if method not in E_METHODS and not (method is None and structural):
            continue
        event = dict(event)
        attempts, selected = event.get("cg_attempts", []), event.get("cg_selected_attempt")
        if isinstance(selected, int) and 0 <= selected < len(attempts):
            event["selected_cg_attempt"] = attempts[selected]
        row = {name: _mapped(event, keys) for name, keys in mapping.items()}
        row.update({name + "_source": _source_key(event, mapping[name]) for name in METRICS})
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
        row["is_boundary"] = (bool(re.search(r"(?:^|\.)boundary_to_\d+$", str(row["selected_site"])))
                              if backbone == "VGG" and row["selected_site"] is not None else None)
        rows.append(row)
    columns = list(mapping) + ["backbone", "seed", "run_id", "source_file",
                               "method", "diagnostic_scope", "heldout_role", "is_boundary"]
    columns += [metric + "_source" for metric in METRICS]
    df = pd.DataFrame(rows, columns=columns)
    for column in (*METRICS, "realized_gain",
                   "selected_scale", "epoch", "probe_index"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[~np.isfinite(df[column]), column] = np.nan
    df["applied"] = df["applied"].map(
        lambda v: v if isinstance(v, bool) else None).astype("boolean")
    df["is_boundary"] = df["is_boundary"].astype("boolean")
    _attach_horizons(df, timelines, horizons)
    df.attrs["key_inventory"] = sorted(inventory)
    df.attrs["mapping"] = mapping
    df.attrs["history_mapping"] = history_mapping
    df.attrs["history_key_inventory"] = sorted(history_inventory)
    df.attrs["horizons"] = list(horizons)
    print("Resolved metric sources:", json.dumps({metric: df[metric + "_source"].value_counts().to_dict()
                                                  for metric in METRICS}, sort_keys=True))
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


def _quartiles(frame, metrics):
    output = {}
    for metric in metrics:
        values = frame[metric].dropna()
        output[metric + "_n"] = len(values)
        for label, quantile in (("q25", .25), ("median", .5), ("q75", .75)):
            output[metric + "_" + label] = values.quantile(quantile) if len(values) else np.nan
    return output


def summarize_cosine_groups(df):
    """Quartiles for each run/site, plus VGG boundary/non-boundary groups."""
    rows = []
    metrics = ("cos_fit", "cos_heldout", "actual_cosine")
    keys = ["backbone", "seed", "run_id"]
    for identity, run in df.groupby(keys, dropna=False):
        base = dict(zip(keys, identity))
        groups = [("selected_site", site, group)
                  for site, group in run.groupby("selected_site", dropna=False)]
        if base["backbone"] == "VGG":
            groups += [("is_boundary", "unknown" if pd.isna(boundary) else str(bool(boundary)), group)
                       for boundary, group in run.groupby("is_boundary", dropna=False)]
        for scope, name, group in groups:
            rows.append({**base, "scope": scope, "group": name,
                         "interventions": len(group), **_quartiles(group, metrics)})
    return pd.DataFrame(rows)


def run_correlations(df, *, bootstrap=2000, seed=0):
    rows = []
    keys = ["backbone", "seed", "run_id"]
    horizons = df.attrs.get("horizons", [1, 5, 15])
    targets = [("realized_gain", None)] + [(f"{prefix}_{horizon}", horizon)
               for horizon in horizons for prefix in ("acc_after", "acc_gain")]
    for identity, run in df.groupby(keys, dropna=False):
        for subset in ("all", "applied_only"):
            group = run if subset == "all" else run[run.applied.fillna(False)]
            for y, horizon in targets:
                selected = group if horizon is None else group[
                    group[f"acc_{horizon}_status"] == "observed"]
                for x in ("cos_fit", "cos_heldout", "actual_cosine"):
                    pairs = selected[[x, y]].dropna()
                    rows.append({**dict(zip(keys, identity)), "subset": subset,
                                 "x": x, "y": y, "horizon": horizon,
                                 "window_rows": len(group),
                                 "excluded_windows": len(group) - len(selected),
                                 **spearman_bootstrap(selected, x, y, bootstrap=bootstrap, seed=seed),
                                 **_quartiles(pairs, (x, y))})
    return pd.DataFrame(rows)


def write_outputs(df, output, *, bootstrap=2000, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "interventions.csv", index=False)
    sites = summarize_sites(df)
    sites.to_csv(output / "selected_sites.csv", index=False)
    summarize_cosine_groups(df).to_csv(output / "cosine_by_site_boundary.csv", index=False)
    run_correlations(df, bootstrap=bootstrap, seed=seed).to_csv(
        output / "run_spearman.csv", index=False)
    provenance = [{"metric": metric, "role": role,
                   "source_key": source, "count": count}
                  for metric, role in (("r_fit", "fit"), ("cos_fit", "fit"),
                                       ("r_heldout", "heldout"), ("cos_heldout", "heldout"),
                                       ("r_actual", "actual"), ("actual_cosine", "actual"))
                  for source, count in df[metric + "_source"].fillna("missing").value_counts().items()]
    pd.DataFrame(provenance).to_csv(output / "metric_sources.csv", index=False)
    df.loc[(df.backbone == "unknown") | df.seed.isna(),
           ["source_file", "run_id", "backbone", "seed"]].drop_duplicates().to_csv(
        output / "unknown_metadata.csv", index=False)
    summary = df.groupby("backbone", dropna=False).agg(
        interventions=("backbone", "size"), applied_known=("applied", "count"),
        applied_rate=("applied", "mean"), r_median=("r_heldout", "median"),
        cos_median=("cos_heldout", "median"))
    summary.to_csv(output / "backbone_summary.csv")
    quartiles = [{"backbone": backbone, **_quartiles(group, METRICS)}
                 for backbone, group in df.groupby("backbone", dropna=False)]
    pd.DataFrame(quartiles).to_csv(output / "backbone_quartiles.csv", index=False)
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
        "history_mapping": df.attrs.get("history_mapping", {}),
        "history_key_inventory": df.attrs.get("history_key_inventory", []),
        "horizons": df.attrs.get("horizons", []),
        "notes": ["Residuals describe selected sites only, not all WHERE sites.",
                  "realized_gain is immediate logged gate-batch loss reduction, not PG_gain or accuracy gain.",
                  "Logged heldout batch may also select scale; independence is not established.",
                  "Bootstrap resamples interventions; repeated trials within a run may be dependent.",
                  "Correlations are descriptive; no capacity-need conclusion, especially for small n.",
                  "run_spearman uses each run separately; pooled spearman.csv is legacy descriptive output.",
                  "Accuracy endpoints use exact logged epochs, not best-over-window or interpolation.",
                  "Accuracy gain subtracts the logged accuracy at the intervention epoch; its timing may be pre/post update.",
                  "Windows containing another logged intervention are excluded from horizon correlations.",
                  "Unknown seed/backbone metadata should be supplied through metadata_overrides."]}
    (output / "analysis_metadata.json").write_text(json.dumps(report, indent=2))
    print(summary.to_string())
    print(f"Saved {len(df)} interventions to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--patterns", nargs="+", default=["**/result.json"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON with mapping, history_mapping, metadata_rules and metadata_overrides")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 15])
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    if any(horizon < 1 for horizon in args.horizons):
        parser.error("--horizons must be positive")
    config = json.loads(args.config.read_text()) if args.config else {}
    df = load_interventions(args.root, args.patterns, mapping=config.get("mapping"),
                            metadata_overrides=config.get("metadata_overrides"),
                            metadata_rules=config.get("metadata_rules"),
                            history_mapping=config.get("history_mapping"),
                            horizons=args.horizons)
    if df.empty:
        parser.error("No E-to-O interventions found; inspect patterns, keys, and metadata.")
    write_outputs(df, args.output, bootstrap=args.bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
