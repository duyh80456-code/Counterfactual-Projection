"""Build the three-method theta_P plateau-fork Kaggle notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Phase 2 — Three methods from selected theta_P

Attach the Phase-1 output containing `plateau_checkpoint.pt`. It is the last
selected best state saved before the no-new-best patience expired. This notebook forks
its model, optimizer, scheduler, RNG, loader state, and data split into three
newly trained 150-epoch method arms. Vanilla uses the same 150-epoch horizon:
the first 100 epochs confirm plateau and the final 50 continue the trajectory.

- scaled matched-horizon Bypass (70 opt1 + up to 30 opt2; never force-project);
- Ours: an initial all-eight-site structural E scan, E-gain WHERE selection,
  winner-only O projection, then recurrent 20-epoch best-checkpoint trials.
- O-only: supervised functional projection at the fixed residual-path site,
  with the same recurrent rollback/retrigger schedule for a fair ablation.

GPU0 is reserved for E-driven O only. GPU1 runs Bypass and then fresh O-only.
Every method process reloads the same theta_best checkpoint. The
official test set is never constructed. Bypass accuracy remains diagnostic if
contraction does not complete within the matched budget.
"""),
    code("""import hashlib, json, os, shutil, subprocess, sys, threading, time, zipfile
from pathlib import Path
from kaggle_secrets import UserSecretsClient

MAIN_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
REPO = Path("/kaggle/working/counterfactual-projection")
REFERENCE_URL = "https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git"
REFERENCE = Path("/kaggle/working/One-Shot-TAS-CCIL")
GROMO_URL = "https://github.com/growingnet/gromo.git"
GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"
GROMO = Path("/kaggle/working/gromo")
OUTPUT = Path("/kaggle/working/plateau_fork_three_methods_150ep_v6")

def private_clone(url, destination, branch):
    token = UserSecretsClient().get_secret("github_token").strip()
    if not token: raise RuntimeError("Kaggle Secret github_token is unavailable")
    askpass = Path("/kaggle/working/.counterfactual_git_askpass.py")
    askpass.write_text("#!/usr/bin/env python3\\nimport os,sys\\np=sys.argv[1] if len(sys.argv)>1 else ''\\nprint('x-access-token' if 'Username' in p else os.environ['GITHUB_TOKEN_RUNTIME'])\\n")
    askpass.chmod(0o700)
    env = os.environ.copy()
    env.update(GITHUB_TOKEN_RUNTIME=token, GIT_ASKPASS=str(askpass),
               GIT_TERMINAL_PROMPT="0")
    try:
        subprocess.run(["git", "clone", "--branch", branch, "--single-branch",
                        url, str(destination)], env=env, check=True)
    finally:
        askpass.unlink(missing_ok=True)

for checkout in (REPO, REFERENCE, GROMO):
    if checkout.exists(): shutil.rmtree(checkout)
OUTPUT.mkdir(parents=True, exist_ok=True)
private_clone(MAIN_URL, REPO, "main")
private_clone(REFERENCE_URL, REFERENCE, "ccil-residual-capacity")
subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                GROMO_URL, str(GROMO)], check=True)
subprocess.run(["git", "-C", str(GROMO), "fetch", "--depth", "1",
                "origin", GROMO_COMMIT], check=True)
subprocess.run(["git", "-C", str(GROMO), "checkout", "--detach",
                GROMO_COMMIT], check=True)
print("main:", subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())
"""),
    code("""subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(GROMO)], check=True)
GROMO_SRC = GROMO / "src"
RUNTIME_PYTHONPATH = os.pathsep.join(filter(None, (
    str(REPO), str(GROMO_SRC), str(REFERENCE), os.environ.get("PYTHONPATH", ""))))
test_env = os.environ.copy(); test_env.update(PYTHONPATH=RUNTIME_PYTHONPATH)
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO,
               env=test_env, check=True)
import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError(f"Select T4 x2; found {torch.cuda.device_count()} GPU(s)")
cifar_dirs = sorted({path.parent.resolve()
                     for path in Path("/kaggle/input").rglob("cifar-100-python")})
if not cifar_dirs: raise FileNotFoundError("Attach CIFAR-100")
DATA_ROOT = cifar_dirs[0]
"""),
    code("""if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.kaggle_checkpoint_discovery import discover_checkpoints

forks, rejected = discover_checkpoints(
    "/kaggle/input", OUTPUT, kind="plateau_fork_checkpoint")
print("Checkpoint candidates rejected:", rejected)
if not forks:
    raise FileNotFoundError(
        "No payload with kind=plateau_fork_checkpoint was found anywhere "
        "under /kaggle/input")
if len({item["sha256"] for item in forks}) != 1:
    raise RuntimeError("Multiple different plateau checkpoints attached")
selected = forks[0]
PLATEAU_CHECKPOINT = selected["path"]
PLATEAU_HASH = selected["sha256"]
PLATEAU_PAYLOAD = selected["payload"]
PLATEAU_EPOCH = int(PLATEAU_PAYLOAD["epoch"])
VANILLA_CONTROL = dict(PLATEAU_PAYLOAD["vanilla_control"])
if VANILLA_CONTROL["post_fork_epochs"] != 150:
    raise RuntimeError(
        "Phase 1 must contain the complete 150-epoch Vanilla horizon")
if VANILLA_CONTROL["role"] not in {
        "matched_validation_best_to_100_epoch_window",
        "matched_significant_best_to_stall_window"}:
    raise RuntimeError("Phase 1 Vanilla control has an invalid role")
print("theta_P:", PLATEAU_EPOCH, PLATEAU_CHECKPOINT, PLATEAU_HASH)
print("theta_P source:", selected["source"])
"""),
    code("""def base_args(output):
    return [
        "--reference-root", str(REFERENCE), "--data-root", str(DATA_ROOT),
        "--plateau-checkpoint", str(PLATEAU_CHECKPOINT),
        "--plateau-checkpoint-hash", PLATEAU_HASH,
        "--output", str(output), "--post-fork-epochs", "150",
        "--seed", "1", "--batch-size", "64", "--rank", "4",
        "--opt1-epochs", "70", "--max-opt2-epochs", "30",
        "--probe-epsilon", "0.05", "--where-batches", "3",
        "--retrigger-patience", "20",
        "--gamma-increase-opt2-epoch", "15",
        "--gamma-post-increase-multiplier", "2.0",
        "--line-search-scales", "0.0125,0.025,0.05"]

commands = {
    name: [sys.executable, "-m", "experiments.run_plateau_fork",
           "--method", name] + base_args(OUTPUT / name)
    for name in ("bypass", "ours_e_driven_o", "o_projection_only")}

# A newly started Kaggle session can resume arm-level progress from an attached
# prior output. Only checkpoints with the same theta_P hash and full protocol
# are paired; unrelated seeds/runs are ignored.
arm_checkpoints, rejected_arm_checkpoints = discover_checkpoints(
    "/kaggle/input", OUTPUT,
    kind={"plateau_fork_arm_progress", "plateau_fork_arm_best"})
arm_progress = [item for item in arm_checkpoints
                if item["payload"]["kind"] == "plateau_fork_arm_progress"]
arm_bests = [item for item in arm_checkpoints
             if item["payload"]["kind"] == "plateau_fork_arm_best"]
for name in ("bypass", "ours_e_driven_o", "o_projection_only"):
    matches = [item for item in [*arm_progress, *arm_bests]
               if item["payload"].get("theta_best_hash") == PLATEAU_HASH
               and item["payload"].get("protocol", {}).get("method") == name]
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
    thread = threading.Thread(target=stream, args=(name, process, log), daemon=True)
    thread.start()
    print(f"GPU{gpu}: started {name}, pid={process.pid}")
    return name, process, thread, log

def finish(job):
    name, process, thread, log = job
    code = process.wait()
    thread.join(); log.close()
    print(name, "exit=", code)
    return name, code

print("GPU0: E-driven O only")
print("GPU1: Bypass -> fresh O-only")
ours_job = launch(0, "ours_e_driven_o")
bypass_job = launch(1, "bypass")
bypass_status = finish(bypass_job)
if bypass_status[1]: raise RuntimeError(f"failed arm: {bypass_status}")
o_only_job = launch(1, "o_projection_only")
statuses = [finish(ours_job), finish(o_only_job)]
failures = [status for status in statuses if status[1]]
if failures: raise RuntimeError(f"failed arms: {failures}")
"""),
    code("""results = {name: json.loads((OUTPUT / name / "result.json").read_text())
           for name in ("bypass", "ours_e_driven_o", "o_projection_only")}
for name, result in results.items():
    if result["plateau_checkpoint_hash"] != PLATEAU_HASH:
        raise RuntimeError(f"{name} used another theta_P")
    if result["theta_best_hash"] != PLATEAU_HASH:
        raise RuntimeError(f"{name} used another theta_best")
    if (name != "bypass" or result["bypass_completed"]):
        if result["post_fork_epochs"] != 150:
            raise RuntimeError(f"{name} did not complete 150 epochs")
    elif not result["budget_exhausted_before_contraction"]:
        raise RuntimeError("Incomplete Bypass has no contraction-budget marker")
    if not Path(result["best_checkpoint"]).is_file():
        raise RuntimeError(f"{name} did not save checkpoint_best.pt")
VANILLA_CONTROL["plateau_checkpoint_hash"] = PLATEAU_HASH
VANILLA_CONTROL["theta_best_hash"] = PLATEAU_HASH
VANILLA_CONTROL["best_checkpoint"] = str(PLATEAU_CHECKPOINT)
results["vanilla"] = VANILLA_CONTROL
summary = {
    "plateau_epoch": PLATEAU_EPOCH,
    "plateau_checkpoint_hash": PLATEAU_HASH,
    "theta_best_hash": PLATEAU_HASH,
    "post_fork_epochs": 150, "official_test_used": False,
    "phase1_vanilla_control": VANILLA_CONTROL,
    "results": {name: {key: result.get(key) for key in (
        "fork_validation_accuracy", "fork_validation_loss",
        "theta_P_validation_accuracy", "theta_P_validation_loss",
        "meaningful_best_validation_accuracy",
        "fork_trigger_accuracy", "fork_trigger_loss",
        "exact_best_validation_accuracy",
        "final_validation_accuracy", "best_validation_accuracy",
        "final_validation_accuracy_space", "best_validation_accuracy_space",
        "final_validation_loss", "best_validation_loss",
        "validation_accuracy_delta", "best_validation_accuracy_delta",
        "epochs_to_best", "training_seconds", "peak_gpu_memory",
        "peak_train_params", "deploy_params", "time_spent_expanded_seconds",
        "bypass_completed", "contraction_at_projection",
        "bypass_comparison_eligible",
        "budget_exhausted_before_contraction",
        "unused_post_fork_epoch_budget",
        "expanded_best_validation_accuracy",
        "expanded_final_validation_accuracy",
        "compact_best_validation_accuracy",
        "compact_final_validation_accuracy",
        "opt1_epochs", "opt2_epochs", "train3_epochs",
        "gamma_increase_opt2_epoch", "gamma_post_increase_multiplier",
        "projection_loss_jump", "best_checkpoint", "intervention_count",
        "correction_application_count", "correction_application_rate",
        "intervention_seconds", "rollback_count", "retrigger_patience",
        "interventions")}
        for name, result in results.items()},
    "bypass_comparison_status": (
        "completed" if results["bypass"]["bypass_completed"] else
        "budget_exhausted_before_contraction_accuracy_is_diagnostic"),
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
print("\\nmethod                  fork_acc   best_acc  final_acc  epoch_best")
def metric(value):
    return "n/a" if value is None else f"{value:.4f}"

for name in ("vanilla", "ours_e_driven_o", "bypass", "o_projection_only"):
    row = results[name]
    print(f"{name:23s} {metric(row['fork_validation_accuracy']):>8s} "
          f"{metric(row['best_validation_accuracy']):>9s} "
          f"{metric(row['final_validation_accuracy']):>10s} "
          f"{str(row['epochs_to_best']):>10s}")
archive = shutil.make_archive(str(OUTPUT), "gztar", root_dir=OUTPUT)
print("Archive:", archive)
"""),
]

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "dataSources": []}},
    "nbformat": 4, "nbformat_minor": 5}
destination = Path("notebooks/kaggle_plateau_fork_t4x2.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
