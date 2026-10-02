"""Build the seed-1 CIFAR-ResNet34 Vanilla/O-only/E-to-O Kaggle run."""

import json
from pathlib import Path


TEMPLATE = Path("notebooks/kaggle_unified_seed1_end_to_end_t4x2.ipynb")
DESTINATION = Path("notebooks/kaggle_resnet34_seed1_end_to_end_t4x2.ipynb")


def source(cell):
    return "".join(cell["source"])


def replace(cell, old, new):
    text = source(cell)
    if old not in text:
        raise RuntimeError(f"notebook template is missing {old!r}")
    cell["source"] = text.replace(old, new).splitlines(keepends=True)


def set_source(cell, text):
    cell["source"] = text.splitlines(keepends=True)


notebook = json.loads(TEMPLATE.read_text())
cells = notebook["cells"]

set_source(cells[0], """# CIFAR-ResNet34 stall experiment — seed 1

This is a from-scratch, self-contained comparison using CIFAR-100 and Kaggle
T4 x2. It needs only the CIFAR-100 input and the `github_token` secret.

Phase 1 trains a randomly initialized full-width Gromo CIFAR-ResNet34 with the
fixed base recipe: SGD, LR 0.1, milestones 100/150, gamma 0.1, momentum 0.9,
weight decay 5e-4, and 200 recipe epochs. Starting at epoch 200, every strict
raw validation best is saved. Theta_P is accepted only after **150 consecutive
epochs without another raw validation best**. Those exact 150 epochs are the
Vanilla comparison trajectory; Vanilla is not restarted.

Phase 2 loads the byte-identical theta_P into two fresh processes:

- GPU0: recurrent E-driven O, 150 SGD epochs;
- GPU1: recurrent O-only, 150 SGD epochs.

Both projected arms use raw validation-best rollback with patience 10. There is
no Bypass arm in this experiment. E-driven O scans all 16 ResNet34 BasicBlocks,
lets structural E-gain select WHERE, and projects only the winner. The official
CIFAR-100 test set is never constructed.
""")

replace(
    cells[1],
    'RUN_ROOT = Path("/kaggle/working/unified_seed1_end_to_end_v5")',
    'RUN_ROOT = Path("/kaggle/working/resnet34_seed1_stall150_v1")')

# Phase-1 input discovery accepts only the new ResNet34 state lineage. Legacy
# ResNet18 checkpoints are deliberately not migrated across architectures.
replace(
    cells[3],
    'schedule_id = "cifar-resnet18-sgd-multistep-200-v5-post200-val-best"',
    'schedule_id = "cifar-resnet34-sgd-multistep-200-v1-post200-val-best"')
replace(cells[3], '"architecture") == "CIFAR-ResNet18"',
        '"architecture") == "CIFAR-ResNet34"')
replace(cells[3], 'f"random_init_seed_{SEED}"',
        'f"random_init_resnet34_seed_{SEED}"')
replace(cells[3], '    if legacy:', '    legacy = []\n    if legacy:')

replace(
    cells[4],
    '"--output", str(PHASE1_OUTPUT), "--seed", str(SEED),',
    '"--output", str(PHASE1_OUTPUT), "--seed", str(SEED),\n'
    '    "--architecture", "resnet34",')
replace(cells[4], '"--weight-decay", "0.0005", "--stall-patience", "100",',
        '"--weight-decay", "0.0005", "--stall-patience", "150",')

replace(cells[5], 'No confirmed 100-epoch stall by max_epoch.',
        'No confirmed 150-epoch stall by max_epoch.')
replace(cells[5], '"matched_validation_best_to_100_epoch_window"',
        '"matched_validation_best_to_150_epoch_window"')

set_source(cells[6], """## Phase 2 — two projected arms plus matched Vanilla

The Phase-1 trajectory already contains exactly 150 Vanilla epochs after
theta_P. E-driven O and O-only now load the same theta_P hash and each consume
the same 150-SGD-epoch budget. They run concurrently on the two T4 GPUs.
""")

set_source(cells[8], r'''def base_args(output):
    return [
        "--reference-root", str(REFERENCE), "--data-root", str(DATA_ROOT),
        "--plateau-checkpoint", str(PLATEAU_CHECKPOINT),
        "--plateau-checkpoint-hash", PLATEAU_HASH,
        "--output", str(output), "--post-fork-epochs", "150",
        "--seed", "1", "--architecture", "resnet34",
        "--batch-size", "64", "--rank", "4",
        "--probe-epsilon", "0.05", "--where-batches", "3",
        "--retrigger-patience", "10",
        "--line-search-scales", "0.0125,0.025,0.05"]

method_names = ("ours_e_driven_o", "o_projection_only")
commands = {
    name: [sys.executable, "-m", "experiments.run_plateau_fork",
           "--method", name] + base_args(OUTPUT / name)
    for name in method_names}

# Resume only arm checkpoints produced from this exact theta_P and protocol.
arm_checkpoints, rejected_arm_checkpoints = discover_checkpoints(
    "/kaggle/input", OUTPUT,
    kind={"plateau_fork_arm_progress", "plateau_fork_arm_best"})
arm_progress = [item for item in arm_checkpoints
                if item["payload"]["kind"] == "plateau_fork_arm_progress"]
arm_bests = [item for item in arm_checkpoints
             if item["payload"]["kind"] == "plateau_fork_arm_best"]
for name in method_names:
    matches = [item for item in [*arm_progress, *arm_bests]
               if item["payload"].get("theta_best_hash") == PLATEAU_HASH
               and item["payload"].get("protocol", {}).get("method") == name
               and item["payload"].get("protocol", {}).get("architecture") ==
                   "CIFAR-ResNet34"]
    if not matches:
        continue
    progress = max(
        matches, key=lambda item: int(item["payload"]["post_fork_epoch"]))
    best_matches = [item for item in arm_bests
                    if item["payload"].get("theta_best_hash") == PLATEAU_HASH
                    and item["payload"].get("protocol") ==
                    progress["payload"].get("protocol")]
    if not best_matches:
        raise RuntimeError(
            f"Found resumable {name} progress without its best checkpoint")
    best = max(
        best_matches,
        key=lambda item: int(item["payload"]["post_fork_epoch"]))
    arm_output = OUTPUT / name
    arm_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(progress["path"], arm_output / "checkpoint_latest.pt")
    shutil.copy2(best["path"], arm_output / "checkpoint_best.pt")
    print(f"Resuming {name} at post-fork epoch "
          f"{progress['payload']['post_fork_epoch']}")

def stream(name, process, log):
    for line in process.stdout:
        log.write(line); log.flush()
        print(f"[{name}] {line}", end="", flush=True)

def launch(gpu, name):
    out = OUTPUT / name; out.mkdir(parents=True, exist_ok=True)
    log = (out / "run.log").open("a", buffering=1)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               PYTHONPATH=RUNTIME_PYTHONPATH)
    process = subprocess.Popen(
        commands[name], cwd=REPO, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    thread = threading.Thread(
        target=stream, args=(name, process, log), daemon=True)
    thread.start()
    print(f"GPU{gpu}: started {name}, pid={process.pid}")
    return name, process, thread, log

def finish(job):
    name, process, thread, log = job
    code = process.wait()
    thread.join(); log.close()
    print(name, "exit=", code)
    return name, code

print("GPU0: E-driven O; GPU1: O-only")
jobs = [launch(0, "ours_e_driven_o"),
        launch(1, "o_projection_only")]
statuses = [finish(job) for job in jobs]
failures = [status for status in statuses if status[1]]
if failures:
    raise RuntimeError(f"failed arms: {failures}")
''')

set_source(cells[9], r'''results = {
    name: json.loads((OUTPUT / name / "result.json").read_text())
    for name in ("ours_e_driven_o", "o_projection_only")}
for name, result in results.items():
    if result["plateau_checkpoint_hash"] != PLATEAU_HASH:
        raise RuntimeError(f"{name} used another theta_P")
    if result["theta_best_hash"] != PLATEAU_HASH:
        raise RuntimeError(f"{name} used another theta_best")
    if result["post_fork_epochs"] != 150:
        raise RuntimeError(f"{name} did not complete 150 epochs")
    if result.get("protocol", {}).get("architecture") != "CIFAR-ResNet34":
        raise RuntimeError(f"{name} did not run CIFAR-ResNet34")
    if not Path(result["best_checkpoint"]).is_file():
        raise RuntimeError(f"{name} did not save checkpoint_best.pt")

VANILLA_CONTROL["plateau_checkpoint_hash"] = PLATEAU_HASH
VANILLA_CONTROL["theta_best_hash"] = PLATEAU_HASH
VANILLA_CONTROL["best_checkpoint"] = str(PLATEAU_CHECKPOINT)
results["vanilla"] = VANILLA_CONTROL
metric_fields = (
    "fork_validation_accuracy", "fork_validation_loss",
    "theta_P_validation_accuracy", "theta_P_validation_loss",
    "final_validation_accuracy", "best_validation_accuracy",
    "final_validation_loss", "best_validation_loss",
    "validation_accuracy_delta", "best_validation_accuracy_delta",
    "epochs_to_best", "training_seconds", "peak_gpu_memory",
    "peak_train_params", "deploy_params", "best_checkpoint",
    "intervention_count", "correction_application_count",
    "correction_application_rate", "intervention_seconds",
    "rollback_count", "retrigger_patience", "interventions")
summary = {
    "architecture": "CIFAR-ResNet34", "seed": SEED,
    "plateau_epoch": PLATEAU_EPOCH,
    "stall_patience": 150,
    "plateau_checkpoint_hash": PLATEAU_HASH,
    "theta_best_hash": PLATEAU_HASH,
    "post_fork_epochs": 150, "official_test_used": False,
    "phase1_vanilla_control": VANILLA_CONTROL,
    "results": {
        name: {key: result.get(key) for key in metric_fields}
        for name, result in results.items()},
}
(OUTPUT / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
print("\nmethod                  fork_acc   best_acc  final_acc  epoch_best")
def metric(value):
    return "n/a" if value is None else f"{value:.4f}"

for name in ("vanilla", "ours_e_driven_o", "o_projection_only"):
    row = results[name]
    print(f"{name:23s} {metric(row['fork_validation_accuracy']):>8s} "
          f"{metric(row['best_validation_accuracy']):>9s} "
          f"{metric(row['final_validation_accuracy']):>10s} "
          f"{str(row['epochs_to_best']):>10s}")
archive = shutil.make_archive(str(OUTPUT), "gztar", root_dir=OUTPUT)
print("Archive:", archive)
''')

for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(DESTINATION)
