"""Build the exact-original-schedule plateau comparison notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Exact-original plateau-triggered E→O from theta300

This T4x2 run forks the exact shared checkpoint at epoch 300 into two equal
trajectories that stop at the horizon stored in the original scheduler:

- GPU 0: Vanilla continuation;
- GPU 1: plateau-triggered structural E→O.

Both arms restore the exact model, optimizer, scheduler, RNG, split, and loader
state. No LR, `initial_lr`, `T_max`, or scheduler state is changed. The original
training indices are unchanged. The original 5,000-example held-out validation
pool is split into 2,000 trigger and 3,000 evaluation examples. The official
test set is never constructed.

E→O is not run every epoch. A separate 2,000-example trigger set declares a
plateau after 15 observations with <0.05 percentage-point accuracy gain and
negligible loss-EMA decrease. At a plateau, TINY proposes all eight blocks;
WHERE is selected only by mean observed structural E loss gain over three
separate batches. O projects only the winner. A correction is applied only if
one of {0.0125, 0.025, 0.05} lowers loss on a separate gate batch.
"""),
    code("""import hashlib, json, os, shutil, subprocess, sys, threading, zipfile
from pathlib import Path
from kaggle_secrets import UserSecretsClient

MAIN_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
REPO = Path("/kaggle/working/counterfactual-projection")
REFERENCE_URL = "https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git"
REFERENCE = Path("/kaggle/working/One-Shot-TAS-CCIL")
GROMO_URL = "https://github.com/growingnet/gromo.git"
GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"
GROMO = Path("/kaggle/working/gromo")
OUTPUT = Path("/kaggle/working/exact_plateau_eo_theta300_v3")

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
        token = None

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
test_env = os.environ.copy()
test_env.update(PYTHONPATH=RUNTIME_PYTHONPATH)
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO,
               env=test_env, check=True)

import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError(f"Select Kaggle T4 x2; found {torch.cuda.device_count()} GPU(s)")
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
    text=True))
cifar_dirs = sorted({path.parent.resolve()
                     for path in Path("/kaggle/input").rglob("cifar-100-python")})
if not cifar_dirs:
    raise FileNotFoundError("Attach CIFAR-100 containing cifar-100-python")
DATA_ROOT = cifar_dirs[0]
"""),
    code("""def materialize(path, index):
    if path.is_file(): return path
    pickles = list(path.rglob("data.pkl"))
    if len(pickles) != 1: return None
    root = pickles[0].parent
    target = OUTPUT / "repacked_input" / f"theta300_{index}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for record in sorted(root.rglob("*")):
            if record.is_file():
                relative = record.relative_to(root).as_posix()
                archive.writestr(f"shared_seed1_epoch300/{relative}", record.read_bytes())
    return target

candidates = []
for index, raw in enumerate(Path("/kaggle/input").rglob("shared_seed1_epoch300.pt")):
    path = materialize(raw, index)
    if path is None: continue
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print("Ignoring unreadable checkpoint", raw, repr(error))
        continue
    history = payload.get("history", [])
    if (payload.get("kind") == "shared_fork_checkpoint" and
            int(payload.get("epoch", -1)) == 300 and history and
            int(history[-1].get("epoch", -1)) == 300):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        candidates.append((path, digest.hexdigest(), raw))
if not candidates:
    raise FileNotFoundError(
        "Attach the complete shared_seed1_epoch300.pt fork checkpoint")
hashes = {item[1] for item in candidates}
if len(hashes) != 1:
    raise RuntimeError("Multiple different valid theta300 checkpoints attached")
THETA300, THETA300_HASH, RAW_THETA300 = candidates[0]
THETA300_PAYLOAD = torch.load(THETA300, map_location="cpu", weights_only=False)
SCHEDULER_STATE = THETA300_PAYLOAD["scheduler"]
EXPECTED_CONTINUATION_EPOCHS = (
    int(SCHEDULER_STATE["T_max"]) - int(SCHEDULER_STATE["last_epoch"]))
if EXPECTED_CONTINUATION_EPOCHS <= 0:
    raise RuntimeError("theta300 scheduler has already reached its horizon")
EXPECTED_FINAL_EPOCH = 300 + EXPECTED_CONTINUATION_EPOCHS
print("shared theta300 input:", RAW_THETA300)
print("materialized checkpoint:", THETA300)
print("theta300 SHA-256:", THETA300_HASH)
print("original scheduler continuation:", EXPECTED_CONTINUATION_EPOCHS,
      "epochs; final epoch:", EXPECTED_FINAL_EPOCH)
"""),
    code("""SEED = 1
def base_args(output):
    return [
        "--reference-root", str(REFERENCE), "--data-root", str(DATA_ROOT),
        "--fork-checkpoint", str(THETA300),
        "--fork-checkpoint-hash", THETA300_HASH,
        "--output", str(output), "--seed", str(SEED),
        "--batch-size", "64", "--validation-samples", "5000",
        "--trigger-samples", "2000",
        "--weight-decay", "0.0005"]

vanilla = [sys.executable, "-m", "experiments.run_plateau_comparison",
           "--method", "vanilla_continue"] + base_args(OUTPUT / "vanilla_continue")
plateau = [sys.executable, "-m", "experiments.run_plateau_comparison",
           "--method", "plateau_e_driven_o"] + base_args(OUTPUT / "plateau_e_driven_o") + [
    "--plateau-window", "15", "--plateau-accuracy-min-gain", "0.0005",
    "--plateau-loss-ema-min-drop", "0.001", "--minimum-sgd-epochs", "15",
    "--rank", "4", "--probe-epsilon", "0.05", "--where-batches", "3",
    "--line-search-scales", "0.0125,0.025,0.05"]

def run_wave(assignments):
    running = []
    def stream_output(name, process, log):
        for line in process.stdout:
            log.write(line); log.flush()
            print(f"[{name}] {line}", end="", flush=True)
        process.stdout.close()
    for gpu, name, command in assignments:
        out = OUTPUT / name
        out.mkdir(parents=True, exist_ok=True)
        log = (out / "run.log").open("a", buffering=1)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   PYTHONPATH=RUNTIME_PYTHONPATH)
        process = subprocess.Popen(
            command, cwd=REPO, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        thread = threading.Thread(
            target=stream_output, args=(name, process, log), daemon=True)
        thread.start()
        running.append((name, process, thread, log))
        print(f"GPU{gpu}: started {name}, pid={process.pid}")
    failures = []
    for name, process, thread, log in running:
        code = process.wait(); thread.join(); log.close()
        print(f"{name}: exit={code}")
        if code: failures.append((name, code))
    if failures:
        raise RuntimeError(f"failed plateau comparison arms: {failures}")

print("T4x2: vanilla continuation + plateau-triggered E→O")
run_wave([(0, "vanilla_continue", vanilla),
          (1, "plateau_e_driven_o", plateau)])
"""),
    code("""results = {}
for name in ("vanilla_continue", "plateau_e_driven_o"):
    path = OUTPUT / name / "result.json"
    result = json.loads(path.read_text())
    if result["source_checkpoint_hash"] != THETA300_HASH:
        raise RuntimeError(f"{name} used a different theta300")
    if (result["final_epoch"] != EXPECTED_FINAL_EPOCH or
            result["continuation_epochs"] != EXPECTED_CONTINUATION_EPOCHS):
        raise RuntimeError(f"{name} did not finish the original schedule")
    if (result["scheduler_state_restored"] is not True or
            result["scheduler_restarted"] is not False):
        raise RuntimeError(f"{name} did not preserve the original scheduler")
    results[name] = result

summary = {
    "source_epoch": 300, "final_epoch": EXPECTED_FINAL_EPOCH,
    "continuation_epochs": EXPECTED_CONTINUATION_EPOCHS,
    "source_checkpoint_hash": THETA300_HASH,
    "official_test_used": False,
    "results": {
        name: {key: value for key, value in result.items() if key in {
            "final_validation_accuracy", "best_validation_accuracy",
            "final_validation_loss", "best_validation_loss",
            "validation_accuracy_delta", "intervention_count",
            "correction_application_count", "correction_application_rate",
            "training_seconds", "peak_gpu_memory", "deploy_params",
            "optimizer_state_preserved", "scheduler_state_restored",
            "scheduler_restarted"}}
        for name, result in results.items()
    },
    "plateau_sequence": [
        {key: event.get(key) for key in (
            "epoch", "selected_site", "selected_e_gain",
            "top1_top2_e_gain_gap", "where_stability",
            "correction_applied", "selected_scale", "loss_before",
            "loss_after", "actual_loss_improvement")}
        for event in results["plateau_e_driven_o"]["interventions"]
    ],
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
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
destination = Path("notebooks/kaggle_plateau_eo_t4x2.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
