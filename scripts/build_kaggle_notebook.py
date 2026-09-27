"""Generate the committed Kaggle T4x2 notebook from readable cell sources."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Counterfactual Projection — CIFAR-100 T4x2 pilot

This notebook clones the private research repository with the Kaggle Secret
`github_token`, verifies the implementation, and schedules independent
method/seed arms across both T4 GPUs. The official CIFAR-100 test partition is
not evaluated; 5,000 examples from the training partition are held out using a
fixed split seed.
"""),
    code("""import json, os, queue, shutil, subprocess, sys, threading
from pathlib import Path
from kaggle_secrets import UserSecretsClient

REPO_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
BRANCH = "main"
REPO = Path("/kaggle/working/counterfactual-projection")
OUTPUT = Path("/kaggle/working/counterfactual_projection_t4x2")

if REPO.exists():
    shutil.rmtree(REPO)
token = UserSecretsClient().get_secret("github_token").strip()
if not token:
    raise RuntimeError("Kaggle Secret github_token is empty or unavailable")
askpass = Path("/kaggle/working/.counterfactual_git_askpass.py")
askpass.write_text(
    "#!/usr/bin/env python3\\n"
    "import os, sys\\n"
    "prompt = sys.argv[1] if len(sys.argv) > 1 else ''\\n"
    "print('x-access-token' if 'Username' in prompt else os.environ['GITHUB_TOKEN_RUNTIME'])\\n")
askpass.chmod(0o700)
clone_env = os.environ.copy()
clone_env.update(GITHUB_TOKEN_RUNTIME=token, GIT_ASKPASS=str(askpass),
                 GIT_TERMINAL_PROMPT="0")
try:
    subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch",
                    REPO_URL, str(REPO)], env=clone_env, check=True)
finally:
    askpass.unlink(missing_ok=True)
    clone_env.pop("GITHUB_TOKEN_RUNTIME", None)
    token = None

commit = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
print("Repository revision:", commit)
OUTPUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)],
               check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO, check=True)
"""),
    code("""import torch

gpu_count = torch.cuda.device_count()
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,name,memory.total",
     "--format=csv,noheader"], text=True))
if gpu_count != 2:
    raise RuntimeError(f"Select the Kaggle T4 x2 accelerator; found {gpu_count} GPU(s)")

input_root = Path("/kaggle/input")
cifar_dirs = sorted({p.parent.resolve() for p in input_root.rglob("cifar-100-python")})
if not cifar_dirs:
    raise FileNotFoundError(
        "Attach a Kaggle CIFAR-100 dataset containing cifar-100-python")
DATA_ROOT = cifar_dirs[0]
print("CIFAR-100 root:", DATA_ROOT)
"""),
    code("""# Quick gate: 4 methods x 2 seeds x 3 epochs. Increase to 3-5 seeds and
# 20+ epochs only after projection_ratio/residual and runtime look sensible.
METHODS = ["vanilla", "random_projection", "e_repopt", "e_projection"]
SEEDS = [0, 1]
EPOCHS = 3
BATCH_SIZE = 64
IMAGE_SIZE = 128
TRAIN_SAMPLES = 12000       # Set 0 for all non-validation training examples.
VALIDATION_SAMPLES = 5000
BLOCK = "layer3.1.conv2"
RANK = 4
CG_ITERATIONS = 12

jobs = [(method, seed) for seed in SEEDS for method in METHODS]
print(f"Scheduled {len(jobs)} arms in {len(jobs) / 2:.0f} two-GPU waves")
"""),
    code("""# Dynamic two-worker queue: each GPU immediately picks up the next arm.
# This is faster for independent ablations than synchronizing both T4s with DDP.
job_queue = queue.Queue()
for job in jobs:
    job_queue.put(job)
failures = []
lock = threading.Lock()

def run_worker(gpu):
    while True:
        try:
            method, seed = job_queue.get_nowait()
        except queue.Empty:
            return
        label = f"{method}_seed{seed}"
        arm_dir = OUTPUT / label
        result = arm_dir / "result.json"
        if result.is_file():
            print(f"[{label}] completed; skipping", flush=True)
            job_queue.task_done()
            continue
        command = [
            sys.executable, "-m", "experiments.run_cifar_pilot",
            "--method", method, "--seed", str(seed),
            "--epochs", str(EPOCHS), "--batch-size", str(BATCH_SIZE),
            "--image-size", str(IMAGE_SIZE),
            "--train-samples", str(TRAIN_SAMPLES),
            "--validation-samples", str(VALIDATION_SAMPLES),
            "--block", BLOCK, "--rank", str(RANK),
            "--cg-iterations", str(CG_ITERATIONS),
            "--data-root", str(DATA_ROOT), "--output", str(arm_dir),
        ]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        arm_dir.mkdir(parents=True, exist_ok=True)
        with (arm_dir / "run.log").open("a") as log:
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush()
                print(f"[GPU{gpu}:{label}] {line}", end="", flush=True)
            code_value = process.wait()
        if code_value:
            with lock:
                failures.append((label, code_value))
        job_queue.task_done()

workers = [threading.Thread(target=run_worker, args=(gpu,), daemon=True)
           for gpu in range(2)]
for worker in workers: worker.start()
for worker in workers: worker.join()
if failures:
    raise RuntimeError(f"Failed arms: {failures}")
print("All arms completed")
"""),
    code("""import statistics

rows = []
for method, seed in jobs:
    path = OUTPUT / f"{method}_seed{seed}" / "result.json"
    result = json.loads(path.read_text())
    if result["deploy_parameter_delta"] != 0:
        raise RuntimeError(f"Deploy-size invariant failed: {path}")
    rows.append(result)

summary = {"repo_commit": commit, "test_evaluated": False, "methods": {}}
for method in METHODS:
    values = [row["validation_accuracy"] for row in rows
              if row["method"] == method]
    residuals = [epoch["projection"]["relative_residual"]
                 for row in rows if row["method"] == method
                 for epoch in row["history"]
                 if epoch["projection"] and
                    "relative_residual" in epoch["projection"]]
    summary["methods"][method] = {
        "validation_accuracy_mean": statistics.mean(values),
        "validation_accuracy_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "mean_projection_residual": statistics.mean(residuals) if residuals else None,
        "seeds": len(values),
    }
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))

archive = shutil.make_archive(str(OUTPUT), "gztar", root_dir=OUTPUT)
print("Download or save as Kaggle Dataset:", archive)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "gpu", "dataSources": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = Path("notebooks/kaggle_counterfactual_projection_t4x2.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
