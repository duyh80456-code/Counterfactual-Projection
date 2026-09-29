"""Build the three-arm theta_P plateau-fork Kaggle notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Phase 2 — Four-arm fork from converged theta_P

Attach the Phase-1 output containing `plateau_checkpoint.pt`. This notebook
forks its exact model, optimizer, constant-LR scheduler, RNG, loader state, and
data split into four 60-epoch arms:

- Vanilla continuation;
- relaxed matched-budget Bypass (40 opt1 + up to 20 opt2; never force-project);
- Ours: one immediate all-eight-site structural E scan, E-gain WHERE selection,
  winner-only O projection, independent gate line search, then ordinary SGD.
- O-only: one supervised functional projection at the fixed residual-path site,
  the same independent gate/line search, then ordinary SGD.

The two T4 GPUs consume a dynamic job queue. Ours and Bypass start first; as soon
as either GPU becomes free it immediately receives O-only, then Vanilla. The
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
OUTPUT = Path("/kaggle/working/plateau_fork_four_arm_60ep_v2")

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
    code("""def materialize(path, index):
    if path.is_file(): return path
    pickles = list(path.rglob("data.pkl"))
    if len(pickles) != 1: return None
    root = pickles[0].parent
    target = OUTPUT / "repacked_input" / f"plateau_checkpoint_{index}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for record in sorted(root.rglob("*")):
            if record.is_file():
                relative = record.relative_to(root).as_posix()
                archive.writestr(f"plateau_checkpoint/{relative}", record.read_bytes())
    return target

forks = []
for index, raw in enumerate(Path("/kaggle/input").rglob("plateau_checkpoint.pt")):
    path = materialize(raw, index)
    if path is None: continue
    try: payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception: continue
    if payload.get("kind") == "plateau_fork_checkpoint":
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        forks.append((path, digest, int(payload["epoch"])))
if not forks: raise FileNotFoundError("Attach Phase-1 plateau_checkpoint.pt")
if len({digest for _, digest, _ in forks}) != 1:
    raise RuntimeError("Multiple different plateau checkpoints attached")
PLATEAU_CHECKPOINT, PLATEAU_HASH, PLATEAU_EPOCH = forks[0]
print("theta_P:", PLATEAU_EPOCH, PLATEAU_CHECKPOINT, PLATEAU_HASH)
"""),
    code("""def base_args(output):
    return [
        "--reference-root", str(REFERENCE), "--data-root", str(DATA_ROOT),
        "--plateau-checkpoint", str(PLATEAU_CHECKPOINT),
        "--plateau-checkpoint-hash", PLATEAU_HASH,
        "--output", str(output), "--post-fork-epochs", "60",
        "--seed", "1", "--batch-size", "64", "--rank", "4",
        "--probe-epsilon", "0.05", "--where-batches", "3",
        "--line-search-scales", "0.0125,0.025,0.05"]

commands = {
    name: [sys.executable, "-m", "experiments.run_plateau_fork",
           "--method", name] + base_args(OUTPUT / name)
    for name in ("vanilla", "bypass", "ours_e_driven_o", "o_projection_only")}

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

pending = ["ours_e_driven_o", "bypass", "o_projection_only", "vanilla"]
running = {gpu: launch(gpu, pending.pop(0)) for gpu in (0, 1)}
failures = []
while running:
    for gpu, (name, process, thread, log) in list(running.items()):
        code = process.poll()
        if code is None: continue
        thread.join(); log.close(); del running[gpu]
        print(name, "exit=", code)
        if code: failures.append((name, code))
        if pending:
            running[gpu] = launch(gpu, pending.pop(0))
    if running:
        time.sleep(1)
if failures:
    raise RuntimeError(f"failed arms: {failures}")
"""),
    code("""results = {name: json.loads((OUTPUT / name / "result.json").read_text())
           for name in ("vanilla", "bypass", "ours_e_driven_o", "o_projection_only")}
for name, result in results.items():
    if result["plateau_checkpoint_hash"] != PLATEAU_HASH:
        raise RuntimeError(f"{name} used another theta_P")
    if result["post_fork_epochs"] != 60:
        raise RuntimeError(f"{name} did not complete 60 epochs")
summary = {
    "plateau_epoch": PLATEAU_EPOCH,
    "plateau_checkpoint_hash": PLATEAU_HASH,
    "post_fork_epochs": 60, "official_test_used": False,
    "results": {name: {key: result.get(key) for key in (
        "final_validation_accuracy", "best_validation_accuracy",
        "final_validation_loss", "best_validation_loss",
        "validation_accuracy_delta", "best_validation_accuracy_delta",
        "epochs_to_best", "training_seconds", "peak_gpu_memory",
        "peak_train_params", "deploy_params", "time_spent_expanded_seconds",
        "bypass_completed", "contraction_at_projection",
        "projection_loss_jump", "intervention")}
        for name, result in results.items()},
    "bypass_comparison_status": (
        "completed" if results["bypass"]["bypass_completed"] else
        "budget_exhausted_before_contraction_accuracy_is_diagnostic"),
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
destination = Path("notebooks/kaggle_plateau_fork_t4x2.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
