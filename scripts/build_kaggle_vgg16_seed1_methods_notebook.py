"""Build the VGG16 seed-1 method-only notebook from an existing theta_P."""

import copy
import json
from pathlib import Path


SOURCE = Path("notebooks/kaggle_vgg16_seed1_end_to_end_t4x2.ipynb")
DESTINATION = Path(
    "notebooks/kaggle_vgg16_seed1_methods_from_plateau_t4x2.ipynb")


def source(cell):
    return "".join(cell.get("source", []))


def markdown(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


template = json.loads(SOURCE.read_text())
bootstrap = copy.deepcopy(template["cells"][1])
bootstrap_text = source(bootstrap).replace(
    '/kaggle/working/vgg16_seed1_stall150_v1',
    '/kaggle/working/vgg16_seed1_methods_from_plateau_v1')
bootstrap["source"] = bootstrap_text.splitlines(keepends=True)
bootstrap["execution_count"] = None
bootstrap["outputs"] = []

discovery = r'''import importlib.util

def load_checkout_module(name, relative_path):
    path = REPO / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import repository helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

discover_checkpoints = load_checkout_module(
    "_counterfactual_kaggle_checkpoint_discovery",
    "experiments/kaggle_checkpoint_discovery.py").discover_checkpoints

SEED = 1
schedule_id = "cifar-vgg16-bn-sgd-multistep-200-v1-post200-val-best"
candidates, rejected = discover_checkpoints(
    "/kaggle/input", RUN_ROOT / "plateau_discovery_cache",
    kind="plateau_fork_checkpoint")
compatible = []
required = {
    "model", "optimizer", "scheduler", "rng",
    "train_loader_generator_state", "train_indices", "trigger_indices",
    "evaluation_indices", "source_tuning_indices", "history", "epoch",
}
for item in candidates:
    payload = item["payload"]
    protocol = payload.get("protocol", {})
    if (required.issubset(payload)
            and protocol.get("seed") == SEED
            and protocol.get("dataset") == "CIFAR-100"
            and protocol.get("architecture") == "CIFAR-VGG16-BN"
            and protocol.get("schedule_id") == schedule_id
            and payload.get("vanilla_baseline_complete") is True
            and payload.get("vanilla_control", {}).get(
                "post_fork_epochs") == 150):
        compatible.append(item)

if not compatible:
    inventory = [{
        "kind": item["payload"].get("kind"),
        "epoch": item["payload"].get("epoch"),
        "seed": item["payload"].get("protocol", {}).get("seed"),
        "architecture": item["payload"].get("protocol", {}).get(
            "architecture"),
        "schedule_id": item["payload"].get("protocol", {}).get(
            "schedule_id"),
    } for item in candidates]
    raise FileNotFoundError(
        "Attach a completed VGG16 seed-1 plateau_checkpoint.pt. "
        f"Visible plateau inventory: {inventory}; rejected: {rejected}")

by_hash = {}
for item in compatible:
    by_hash.setdefault(item["sha256"], item)
if len(by_hash) != 1:
    raise RuntimeError(
        "Multiple distinct compatible VGG16 seed-1 theta_P checkpoints are "
        f"attached: {sorted(by_hash)}. Attach exactly one run.")

selected = next(iter(by_hash.values()))
PLATEAU_CHECKPOINT = Path(selected["path"])
PLATEAU_HASH = selected["sha256"]
PLATEAU_PAYLOAD = selected["payload"]
PLATEAU_EPOCH = int(PLATEAU_PAYLOAD["epoch"])
VANILLA_CONTROL = dict(PLATEAU_PAYLOAD["vanilla_control"])
if hashlib.sha256(PLATEAU_CHECKPOINT.read_bytes()).hexdigest() != PLATEAU_HASH:
    raise RuntimeError("theta_P changed after checkpoint discovery")
print("Loaded VGG16 seed-1 theta_P:", {
    "epoch": PLATEAU_EPOCH,
    "sha256": PLATEAU_HASH,
    "path": str(PLATEAU_CHECKPOINT),
    "vanilla_best": VANILLA_CONTROL.get("best_validation_accuracy"),
    "vanilla_final": VANILLA_CONTROL.get("final_validation_accuracy"),
})
'''

cells = [
    markdown("""# VGG16-BN seed 1 — methods from existing theta_P

Attach CIFAR-100 and the prior output containing the completed VGG16 seed-1
`plateau_checkpoint.pt`. This notebook does **not** train Vanilla. It verifies
the full checkpoint state and SHA-256, then starts both methods from the same
byte-identical theta_P:

- GPU0: recurrent E-driven O, patience 25, 150 SGD epochs;
- GPU1: one O-only projection at theta_P, then 150 uninterrupted SGD epochs.

Old recurrent O-only checkpoints are rejected. The official CIFAR-100 test set
is never constructed.
"""),
    bootstrap,
    copy.deepcopy(template["cells"][2]),
    code(discovery),
    markdown("""## Run methods directly from the attached theta_P

No Phase-1 command is executed below.
"""),
    copy.deepcopy(template["cells"][8]),
    copy.deepcopy(template["cells"][9]),
    code("""archive = shutil.make_archive(str(RUN_ROOT), "gztar", root_dir=RUN_ROOT)
print("Method-only archive:", archive)
print("Loaded fork checkpoint:", PLATEAU_CHECKPOINT)
"""),
]
for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook = {
    "cells": cells,
    "metadata": copy.deepcopy(template.get("metadata", {})),
    "nbformat": 4,
    "nbformat_minor": 5,
}
DESTINATION.write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(DESTINATION)
