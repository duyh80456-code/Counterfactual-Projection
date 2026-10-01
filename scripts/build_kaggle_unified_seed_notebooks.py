"""Build three from-scratch, protocol-identical Kaggle T4x2 notebooks."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def text(cell):
    return "".join(cell["source"])


phase2 = json.loads(Path(
    "notebooks/kaggle_plateau_fork_t4x2.ipynb").read_text())


def build(seed):
    run_root = f"/kaggle/working/unified_seed{seed}_end_to_end_v4"
    bootstrap = text(phase2["cells"][1]).replace(
        'OUTPUT = Path("/kaggle/working/plateau_fork_three_methods_100ep_v5")',
        f'RUN_ROOT = Path("{run_root}")\nOUTPUT = RUN_ROOT / "phase2"')
    phase2_commands = text(phase2["cells"][4]).replace(
        '"--seed", "1"', f'"--seed", "{seed}"')
    phase2_summary = text(phase2["cells"][5])

    cells = [
        markdown(f"""# Unified from-scratch stall experiment — seed {seed}

This notebook is one complete, self-contained replication. It needs only
CIFAR-100, Kaggle **T4 x2**, and the `github_token` secret. It does not load a
theta150/theta300 checkpoint and never rebases or restarts the learning-rate
schedule.

Phase 1 trains a randomly initialized CIFAR-ResNet18 with its metric-independent
base recipe: SGD for 200 recipe epochs, LR 0.1 with MultiStep drops at epochs
100 and 150 (`gamma=0.1`), momentum 0.9, and weight decay 5e-4. Every raw
validation best is fully checkpointed from the start. Once the base recipe is
complete, a stall is confirmed when exactly 100 epochs have elapsed since the
last raw validation best. Those 100 epochs are the matched Vanilla control and
Theta_P is that raw validation-best checkpoint.

Phase 2 forks that run's raw validation-best theta_P, including optimizer momentum,
scheduler position, RNG, loader stream, and data indices. GPU0 runs recurrent
E-driven O only. GPU1 runs scaled Bypass 70/30 and then launches
a fresh recurrent O-only process. Each method consumes 100 SGD epochs and writes per-epoch history,
latest/best checkpoints, diagnostics, timing, memory, and final/best accuracy.
Only `seed={seed}` differs from the other two generated notebooks.
"""),
        code(bootstrap),
        phase2["cells"][2],
        code(f"""if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.plateau_protocol import BestCheckpointStallDetector

SEED = {seed}
PHASE1_OUTPUT = RUN_ROOT / "vanilla_stall"
PHASE1_OUTPUT.mkdir(parents=True, exist_ok=True)
schedule_id = "cifar-resnet18-sgd-multistep-200-v4-validation-best"

def copy_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)

checkpoint_kinds = {{
    "unified_vanilla_progress", "vanilla_validation_best_checkpoint",
    "vanilla_significant_best_checkpoint", "vanilla_exact_best_diagnostic",
    "plateau_fork_checkpoint"}}
discovered, rejected_checkpoints = discover_checkpoints(
    "/kaggle/input", PHASE1_OUTPUT, kind=checkpoint_kinds)
progresses = [item for item in discovered
              if item["payload"].get("kind") == "unified_vanilla_progress"]
bests = [item for item in discovered
         if item["payload"].get("kind") ==
         "vanilla_validation_best_checkpoint"]
compatible = [item for item in progresses
              if item["payload"].get("protocol", {{}}).get("seed") == SEED
              and item["payload"].get("protocol", {{}}).get(
                  "schedule_id") == schedule_id]
PHASE1_MAX_EPOCH = 800
if not compatible:
    # A v3 run used the same training recipe but selected theta_P by trigger.
    # It is safe to migrate only when one attached full checkpoint lands
    # exactly on the raw validation-best epoch recorded in its history.
    legacy = [item for item in progresses
              if item["payload"].get("protocol", {{}}).get("seed") == SEED
              and item["payload"].get("protocol", {{}}).get(
                  "schedule_id") ==
                  "cifar-resnet18-sgd-multistep-200-v3-significant-fork"]
    if legacy:
        old_progress = max(
            legacy, key=lambda item: int(item["payload"]["epoch"]))
        old_payload = old_progress["payload"]
        old_history = old_payload["history"]
        old_best_row = max(
            old_history, key=lambda row: row["validation_accuracy"])
        old_best_epoch = int(old_best_row["epoch"])
        exact_snapshots = [item for item in discovered
                           if item["payload"].get("protocol", {{}}).get(
                               "seed") == SEED
                           and int(item["payload"].get("epoch", -1)) ==
                               old_best_epoch]
        if exact_snapshots:
            old_protocol = dict(old_payload["protocol"])
            old_protocol.pop("exact_best_min_gain", None)
            old_protocol.pop("significant_min_gain", None)
            old_protocol.update({{
                "schedule_id": schedule_id,
                "selection_metric":
                    "validation accuracy (3,000-sample split)",
                "evaluation_role":
                    "model selection and reporting; official test unused",
                "stall_gate": (
                    "base recipe complete and 100 epochs after raw "
                    "validation best"),
                "theta_P_scope": "global raw validation best",
                "validation_best_min_gain": 0.0,
            }})
            detector = BestCheckpointStallDetector(
                patience=100, min_gain=0.0, require_arm=False,
                exact_best_patience=True)
            for row in old_history:
                detector.update(
                    int(row["epoch"]), float(row["validation_accuracy"]))
            migrated_progress = {{
                **old_payload, "protocol": old_protocol,
                "best_stall_detector": detector.state_dict()}}
            migrated_best = {{
                **exact_snapshots[0]["payload"],
                "kind": "vanilla_validation_best_checkpoint",
                "protocol": old_protocol}}
            torch.save(
                migrated_progress, PHASE1_OUTPUT / "checkpoint_latest.pt")
            torch.save(migrated_best, PHASE1_OUTPUT / "checkpoint_best.pt")
            torch.save(
                migrated_best, PHASE1_OUTPUT / "checkpoint_exact_best.pt")
            compatible = [{{
                "path": PHASE1_OUTPUT / "checkpoint_latest.pt",
                "payload": migrated_progress,
                "source": f"migrated:{{old_progress['source']}}"}}]
            bests = [{{
                "path": PHASE1_OUTPUT / "checkpoint_best.pt",
                "payload": migrated_best,
                "source": "migrated-validation-best"}}]
            print("Migrated compatible v3 trajectory at exact validation best",
                  old_best_epoch)
        else:
            print("Cannot migrate attached v3 run: validation-best epoch",
                  old_best_epoch, "has no full checkpoint snapshot")
if compatible:
    progress = max(compatible, key=lambda item: int(item["payload"]["epoch"]))
    payload = progress["payload"]
    history = payload["history"]
    best_row = max(history, key=lambda row: row["validation_accuracy"])
    best_epoch = int(best_row["epoch"])
    matching_bests = [item for item in bests
                      if item["payload"].get("protocol") == payload["protocol"]
                      and int(item["payload"].get("epoch", -1)) == best_epoch]
    if not matching_bests:
        if int(payload["epoch"]) != best_epoch:
            raise RuntimeError(
                f"Found resumable epoch {{payload['epoch']}} but no matching "
                f"validation-best checkpoint at epoch {{best_epoch}}")
        selected_best = progress
        print("Repairing best checkpoint from atomic latest checkpoint")
    else:
        selected_best = matching_bests[0]
    copy_checkpoint(progress["path"], PHASE1_OUTPUT / "checkpoint_latest.pt")
    copy_checkpoint(
        selected_best["path"], PHASE1_OUTPUT / "checkpoint_best.pt")
    copy_checkpoint(
        selected_best["path"], PHASE1_OUTPUT / "checkpoint_exact_best.pt")
    remaining = max(0, best_epoch + 100 - int(payload["epoch"]))
    PHASE1_MAX_EPOCH = max(
        PHASE1_MAX_EPOCH, int(payload["epoch"]) + remaining)
    print("Resuming Phase 1:", {{
        "checkpoint_epoch": int(payload["epoch"]),
        "validation_best_epoch": best_epoch,
        "minimum_additional_epochs_if_no_new_best": remaining,
        "source": progress["source"],
    }})
else:
    print("No compatible Phase-1 checkpoint in /kaggle/input; starting epoch 0")
"""),
        code(f"""SEED = {seed}
PHASE1_OUTPUT = RUN_ROOT / "vanilla_stall"
PHASE1_OUTPUT.mkdir(parents=True, exist_ok=True)

phase1_command = [
    sys.executable, "-m", "experiments.run_unified_vanilla_to_stall",
    "--reference-root", str(REFERENCE), "--data-root", str(DATA_ROOT),
    "--output", str(PHASE1_OUTPUT), "--seed", str(SEED),
    "--max-epoch", str(PHASE1_MAX_EPOCH), "--batch-size", "64",
    "--validation-samples", "5000", "--trigger-samples", "2000",
    "--tuning-samples", "128", "--lr", "0.1", "--recipe-epochs", "200",
    "--lr-milestones", "100,150", "--lr-gamma", "0.1",
    "--weight-decay", "0.0005", "--stall-patience", "100"]

phase1_log = (PHASE1_OUTPUT / "notebook_stream.log").open("a", buffering=1)
phase1_env = os.environ.copy()
phase1_env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
                  PYTHONPATH=RUNTIME_PYTHONPATH)
phase1_process = subprocess.Popen(
    phase1_command, cwd=REPO, env=phase1_env, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in phase1_process.stdout:
    phase1_log.write(line); phase1_log.flush()
    print(f"[seed{{SEED}} vanilla] {{line}}", end="", flush=True)
phase1_code = phase1_process.wait()
phase1_log.close()
if phase1_code:
    raise RuntimeError(f"unified Vanilla exited with code {{phase1_code}}")
"""),
        code("""phase1_result = json.loads(
    (PHASE1_OUTPUT / "result.json").read_text())
if not phase1_result["plateau_found"]:
    archive = shutil.make_archive(
        str(RUN_ROOT), "gztar", root_dir=RUN_ROOT)
    raise RuntimeError(
        "No confirmed 100-epoch stall by max_epoch. All resumable Phase-1 "
        f"checkpoints were retained in {archive}")

PLATEAU_CHECKPOINT = Path(phase1_result["plateau_checkpoint"])
PLATEAU_PAYLOAD = torch.load(
    PLATEAU_CHECKPOINT, map_location="cpu", weights_only=False)
if PLATEAU_PAYLOAD.get("kind") != "plateau_fork_checkpoint":
    raise RuntimeError("Phase 1 did not produce a plateau fork checkpoint")
PLATEAU_HASH = hashlib.sha256(
    PLATEAU_CHECKPOINT.read_bytes()).hexdigest()
PLATEAU_EPOCH = int(PLATEAU_PAYLOAD["epoch"])
VANILLA_CONTROL = dict(PLATEAU_PAYLOAD["vanilla_control"])
if VANILLA_CONTROL["post_fork_epochs"] != 100:
    raise RuntimeError("Vanilla control is not exactly 100 observed epochs")
if VANILLA_CONTROL["role"] != "matched_validation_best_to_100_epoch_window":
    raise RuntimeError("Phase-1 Vanilla control has an invalid role")
if int(PLATEAU_PAYLOAD["protocol"]["seed"]) != SEED:
    raise RuntimeError("theta_best seed mismatch")
print("theta_best:", PLATEAU_EPOCH, PLATEAU_HASH)
print(json.dumps(VANILLA_CONTROL, indent=2, sort_keys=True))
"""),
        markdown("""## Phase 2 — three method forks plus matched Vanilla history

The Phase-1 window is exactly 100 epochs after raw validation-best theta_P and is
the Vanilla control. Three method jobs load byte-identical theta_P. GPU0 runs
E-driven O only; GPU1 runs Bypass then O-only.
"""),
        code("""import threading

OUTPUT = RUN_ROOT / "phase2"
OUTPUT.mkdir(parents=True, exist_ok=True)
"""),
        code(phase2_commands),
        code(phase2_summary),
        code("""full_archive = shutil.make_archive(
    str(RUN_ROOT), "gztar", root_dir=RUN_ROOT)
print("Complete seed archive:", full_archive)
print("Phase-1 latest:", PHASE1_OUTPUT / "checkpoint_latest.pt")
print("Phase-1 best:", PHASE1_OUTPUT / "checkpoint_best.pt")
print("Fork checkpoint:", PLATEAU_CHECKPOINT)
"""),
    ]
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "gpu", "dataSources": []}},
        "nbformat": 4, "nbformat_minor": 5}


for seed in (0, 1, 2):
    destination = Path(
        f"notebooks/kaggle_unified_seed{seed}_end_to_end_t4x2.ipynb")
    destination.write_text(
        json.dumps(build(seed), indent=1, ensure_ascii=False) + "\n")
    print(destination)
