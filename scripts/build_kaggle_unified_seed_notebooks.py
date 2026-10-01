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
    run_root = f"/kaggle/working/unified_seed{seed}_end_to_end_v5"
    bootstrap = text(phase2["cells"][1]).replace(
        'OUTPUT = Path("/kaggle/working/plateau_fork_three_methods_150ep_v6")',
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
validation best from epoch 200 onward is fully checkpointed. A stall is
confirmed when exactly 100 epochs have elapsed since the latest such best.
Vanilla then continues the same trajectory for 50 more epochs, giving every arm
the same 150-epoch post-fork horizon. Theta_P is the raw validation-best
checkpoint at or after epoch 200.

Phase 2 forks that run's raw validation-best checkpoint theta_P, including optimizer momentum,
scheduler position, RNG, loader stream, and data indices. GPU0 runs recurrent
E-driven O only. GPU1 runs scaled Bypass 70/30 and then launches
a fresh recurrent O-only process. Each method consumes 150 SGD epochs and writes per-epoch history,
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
schedule_id = "cifar-resnet18-sgd-multistep-200-v5-post200-val-best"

def copy_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)

def compatible_training_lineage(candidate, reference):
    # Compare state-producing invariants, not incidental report metadata.
    required_state = {{
        "model", "optimizer", "scheduler", "rng",
        "train_loader_generator_state", "train_indices", "trigger_indices",
        "evaluation_indices", "source_tuning_indices", "history", "epoch"}}
    if not required_state.issubset(candidate) or not required_state.issubset(reference):
        return False
    protocol_fields = (
        "seed", "dataset", "architecture", "input_size", "learning_rate",
        "weight_decay", "batch_size", "schedule_id", "optimizer",
        "model_state_lineage", "train_indices_sha256",
        "validation_indices_sha256", "tuning_indices_sha256")
    left, right = candidate.get("protocol", {{}}), reference.get("protocol", {{}})
    if any(left.get(key) != right.get(key) for key in protocol_fields):
        return False
    for key in ("train_indices", "trigger_indices", "evaluation_indices",
                "source_tuning_indices"):
        if list(candidate[key]) != list(reference[key]):
            return False
    scheduler_fields = ("kind", "milestones", "gamma", "recipe_epochs")
    if any(candidate["scheduler"].get(key) != reference["scheduler"].get(key)
           for key in scheduler_fields):
        return False
    optimizer_fields = ("momentum", "dampening", "weight_decay", "nesterov")
    candidate_groups = candidate["optimizer"].get("param_groups", [])
    reference_groups = reference["optimizer"].get("param_groups", [])
    if len(candidate_groups) != len(reference_groups):
        return False
    if any(any(a.get(key) != b.get(key) for key in optimizer_fields)
           for a, b in zip(candidate_groups, reference_groups)):
        return False
    candidate_shapes = {{key: tuple(value.shape)
                        for key, value in candidate["model"].items()}}
    reference_shapes = {{key: tuple(value.shape)
                        for key, value in reference["model"].items()}}
    return candidate_shapes == reference_shapes

def is_post200_raw_best_snapshot(item):
    payload = item["payload"]
    if not isinstance(payload, dict) or "history" not in payload:
        return False
    epoch = int(payload.get("epoch", -1))
    rows = [row for row in payload["history"]
            if 200 <= int(row["epoch"]) <= epoch]
    if not rows:
        return False
    observed = max(rows, key=lambda row: float(row["validation_accuracy"]))
    return int(observed["epoch"]) == epoch

checkpoint_kinds = {{
    "unified_vanilla_progress", "vanilla_validation_best_checkpoint",
    "vanilla_significant_best_checkpoint", "vanilla_exact_best_diagnostic",
    "plateau_fork_checkpoint"}}
input_discovered, input_rejected = discover_checkpoints(
    "/kaggle/input", RUN_ROOT / "input_discovery_cache",
    kind=checkpoint_kinds)
local_discovered, local_rejected = discover_checkpoints(
    PHASE1_OUTPUT, RUN_ROOT / "local_discovery_cache",
    kind=checkpoint_kinds)
discovered_by_identity = {{}}
for item in [*input_discovered, *local_discovered]:
    identity = (
        item["sha256"], item["payload"].get("kind"),
        int(item["payload"].get("epoch", -1)))
    discovered_by_identity.setdefault(identity, item)
discovered = list(discovered_by_identity.values())
rejected_checkpoints = [*input_rejected, *local_rejected]
print("Phase-1 checkpoint inventory:", [{{
    "kind": item["payload"].get("kind"),
    "epoch": int(item["payload"].get("epoch", -1)),
    "schedule_id": item["payload"].get("protocol", {{}}).get("schedule_id"),
    "seed": item["payload"].get("protocol", {{}}).get("seed"),
    "source": item["source"],
}} for item in discovered])
progresses = [item for item in discovered
              if item["payload"].get("kind") == "unified_vanilla_progress"]
bests = [item for item in discovered if is_post200_raw_best_snapshot(item)]
compatible = [item for item in progresses
              if item["payload"].get("protocol", {{}}).get("seed") == SEED
              and item["payload"].get("protocol", {{}}).get(
                  "schedule_id") == schedule_id
              and item["payload"].get("protocol", {{}}).get(
                  "dataset") == "CIFAR-100"
              and item["payload"].get("protocol", {{}}).get(
                  "architecture") == "CIFAR-ResNet18"
              and item["payload"].get("protocol", {{}}).get(
                  "batch_size") == 64
              and item["payload"].get("protocol", {{}}).get(
                  "optimizer") ==
                  "SGD(momentum=0.9, weight_decay=0.0005)"
              and item["payload"].get("protocol", {{}}).get(
                  "model_state_lineage") == f"random_init_seed_{{SEED}}"]
PHASE1_MAX_EPOCH = 800
if not compatible:
    # Older unified runs used the same base recipe. Choose the closest full
    # checkpoint at or before the observed post-200 validation best and replay
    # forward; never synthesize weights from history alone.
    legacy = [item for item in progresses
              if item["payload"].get("protocol", {{}}).get("seed") == SEED
              and item["payload"].get("protocol", {{}}).get(
                  "schedule_id") in {{
                  "cifar-resnet18-sgd-multistep-200-v3-significant-fork",
                  "cifar-resnet18-sgd-multistep-200-v4-validation-best"}}]
    if legacy:
        old_progress = max(
            legacy, key=lambda item: int(item["payload"]["epoch"]))
        old_payload = old_progress["payload"]
        old_history = old_payload["history"]
        post200 = [row for row in old_history if int(row["epoch"]) >= 200]
        if not post200:
            raise RuntimeError("Legacy progress does not reach epoch 200")
        old_best_row = max(
            post200, key=lambda row: row["validation_accuracy"])
        old_best_epoch = int(old_best_row["epoch"])
        replay_sources = [item for item in discovered
                          if item["payload"].get("protocol", {{}}).get(
                              "seed") == SEED
                          and item["payload"].get("protocol", {{}}).get(
                              "schedule_id") ==
                              old_payload["protocol"].get("schedule_id")
                          and int(item["payload"].get("epoch", -1)) <=
                              old_best_epoch]
        if replay_sources:
            replay_source = max(
                replay_sources,
                key=lambda item: int(item["payload"]["epoch"]))
            replay_payload = replay_source["payload"]
            replay_epoch = int(replay_payload["epoch"])
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
                    "100 epochs after raw validation best at epoch >=200"),
                "theta_P_scope": (
                    "raw validation best at or after recipe epoch 200"),
                "validation_best_min_gain": 0.0,
                "post_fork_epochs": 150,
                "optimizer": "SGD(momentum=0.9, weight_decay=0.0005)",
                "model_state_lineage": f"random_init_seed_{{SEED}}",
                "batch_size": 64,
            }})
            detector = BestCheckpointStallDetector(
                patience=100, min_gain=0.0, require_arm=True,
                exact_best_patience=True)
            replay_history = list(replay_payload["history"])
            for row in replay_history:
                row_epoch = int(row["epoch"])
                metric = float(row["validation_accuracy"])
                detector.update(row_epoch, metric)
                if row_epoch == 200:
                    detector.arm_stall(row_epoch, metric)
            missing_best_epoch = (
                int(detector.best_epoch)
                if int(detector.best_epoch) != replay_epoch else None)
            migrated_progress = {{
                **replay_payload, "protocol": old_protocol,
                "best_stall_detector": detector.state_dict(),
                "replay_missing_best_epoch": missing_best_epoch}}
            migrated_best = {{
                **replay_payload,
                "kind": "vanilla_validation_best_checkpoint",
                "protocol": old_protocol,
                "replay_missing_best_epoch": missing_best_epoch,
                "placeholder_for_missing_best": missing_best_epoch}}
            torch.save(
                migrated_progress, PHASE1_OUTPUT / "checkpoint_latest.pt")
            torch.save(migrated_best, PHASE1_OUTPUT / "checkpoint_best.pt")
            torch.save(
                migrated_best, PHASE1_OUTPUT / "checkpoint_exact_best.pt")
            compatible = [{{
                "path": PHASE1_OUTPUT / "checkpoint_latest.pt",
                "payload": migrated_progress,
                "source": f"replay:{{replay_source['source']}}"}}]
            bests = [{{
                "path": PHASE1_OUTPUT / "checkpoint_best.pt",
                "payload": migrated_best,
                "source": "migrated-validation-best"}}]
            print("Replaying legacy trajectory:", {{
                "observed_validation_best_epoch": old_best_epoch,
                "replay_checkpoint_epoch": replay_epoch,
                "epochs_to_observed_best": old_best_epoch - replay_epoch,
            }})
        else:
            print("Cannot replay attached legacy run: no compatible full "
                  "checkpoint at or before validation-best epoch",
                  old_best_epoch)
if compatible:
    progress = max(compatible, key=lambda item: int(item["payload"]["epoch"]))
    payload = progress["payload"]
    history = payload["history"]
    post200 = [row for row in history if int(row["epoch"]) >= 200]
    if post200:
        best_row = max(post200, key=lambda row: row["validation_accuracy"])
        best_epoch = int(best_row["epoch"])
    else:
        best_epoch = int(payload["best_stall_detector"]["best_epoch"])
    pending_missing_best = payload.get("replay_missing_best_epoch")
    matching_bests = [item for item in bests
                      if compatible_training_lineage(item["payload"], payload)
                      and int(item["payload"].get("epoch", -1)) == best_epoch]
    if pending_missing_best is not None:
        selected_best = progress
        print("Replaying with historical raw-best metric but missing weights:", {{
            "missing_best_epoch": int(pending_missing_best),
            "replay_epoch": int(payload["epoch"]),
            "rule": "must observe a new strict raw validation best before fork",
        }})
    elif not matching_bests:
        earlier_bests = [item for item in bests
                         if compatible_training_lineage(
                             item["payload"], payload)
                         and int(item["payload"].get("epoch", -1)) <=
                         best_epoch]
        if earlier_bests:
            # The observed best weights are unavailable, so replay from the
            # closest complete raw-best checkpoint at or before that epoch.
            # Its model/optimizer/scheduler/RNG/loader state remain intact.
            selected_best = max(
                earlier_bests,
                key=lambda item: int(item["payload"]["epoch"]))
            progress = selected_best
            payload = progress["payload"]
            history = payload["history"]
            print("Exact observed validation-best checkpoint is absent; "
                  "replaying from closest compatible raw best:", {{
                      "required_epoch": best_epoch,
                      "replay_epoch": int(payload["epoch"]),
                      "source": progress["source"],
                  }})
        elif int(payload["epoch"]) == best_epoch:
            selected_best = progress
            print("Repairing best checkpoint from atomic latest checkpoint")
        else:
            raise RuntimeError(
                f"Found resumable epoch {{payload['epoch']}} but no compatible "
                f"full checkpoint at or before validation-best epoch "
                f"{{best_epoch}}")
    else:
        selected_best = matching_bests[0]
    copy_checkpoint(progress["path"], PHASE1_OUTPUT / "checkpoint_latest.pt")
    copy_checkpoint(
        selected_best["path"], PHASE1_OUTPUT / "checkpoint_best.pt")
    copy_checkpoint(
        selected_best["path"], PHASE1_OUTPUT / "checkpoint_exact_best.pt")
    resumed_history = payload["history"]
    resumed_post200 = [row for row in resumed_history
                       if int(row["epoch"]) >= 200]
    resumed_best_epoch = (
        int(max(resumed_post200,
                key=lambda row: row["validation_accuracy"])["epoch"])
        if resumed_post200 else 200)
    target_epoch = resumed_best_epoch + 150
    remaining = max(0, target_epoch - int(payload["epoch"]))
    PHASE1_MAX_EPOCH = max(
        PHASE1_MAX_EPOCH, int(payload["epoch"]) + remaining)
    print("Resuming Phase 1:", {{
        "checkpoint_epoch": int(payload["epoch"]),
        "validation_best_epoch": resumed_best_epoch,
        "minimum_additional_epochs_if_no_new_best": remaining,
        "source": progress["source"],
    }})
else:
    print("No compatible Phase-1 checkpoint in /kaggle/input or the current "
          "working run; starting epoch 0")
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
    "--weight-decay", "0.0005", "--stall-patience", "100",
    "--post-fork-epochs", "150"]

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
if VANILLA_CONTROL["post_fork_epochs"] != 150:
    raise RuntimeError("Vanilla baseline is not exactly 150 post-fork epochs")
if VANILLA_CONTROL["role"] != "matched_validation_best_to_100_epoch_window":
    raise RuntimeError("Phase-1 Vanilla control has an invalid role")
if int(PLATEAU_PAYLOAD["protocol"]["seed"]) != SEED:
    raise RuntimeError("theta_best seed mismatch")
print("theta_best:", PLATEAU_EPOCH, PLATEAU_HASH)
print(json.dumps(VANILLA_CONTROL, indent=2, sort_keys=True))
"""),
        markdown("""## Phase 2 — three method forks plus matched Vanilla history

The first 100 Phase-1 epochs after raw validation-best theta_P confirm plateau;
the same trajectory continues for 50 more epochs as the 150-epoch Vanilla
baseline. Three method jobs load byte-identical theta_P. GPU0 runs
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
