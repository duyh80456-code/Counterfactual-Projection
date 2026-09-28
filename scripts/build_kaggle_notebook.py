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

This notebook clones the private method repository plus the audited
One-Shot-TAS-CCIL/Gromo references, then schedules independent method/seed arms
across both T4 GPUs. The model is a full-width ImageNet-pretrained ResNet-18;
the primary E signal is a counterfactual temporary hidden-width extension from
TINY/Gromo beyond that full width, not a low-rank factorization of an existing
kernel. Statistics and projection fitting use fresh disjoint batches sampled
from the common training pool; checking uses a fixed held-out batch. The
CIFAR-100 test partition is not evaluated.
"""),
    code("""import json, os, queue, shutil, subprocess, sys, threading
from pathlib import Path
from kaggle_secrets import UserSecretsClient

REPO_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
BRANCH = "main"
REPO = Path("/kaggle/working/counterfactual-projection")
REFERENCE_URL = "https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git"
REFERENCE_BRANCH = "ccil-residual-capacity"
REFERENCE = Path("/kaggle/working/One-Shot-TAS-CCIL")
GROMO_URL = "https://github.com/growingnet/gromo.git"
GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"
GROMO = Path("/kaggle/working/gromo")
OUTPUT = Path("/kaggle/working/counterfactual_projection_t4x2_fair_v2")

for checkout in (REPO, REFERENCE, GROMO):
    if checkout.exists(): shutil.rmtree(checkout)
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
    subprocess.run(["git", "clone", "--branch", REFERENCE_BRANCH,
                    "--single-branch", REFERENCE_URL, str(REFERENCE)],
                   env=clone_env, check=True)
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
subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                GROMO_URL, str(GROMO)], check=True)
subprocess.run(["git", "-C", str(GROMO), "fetch", "--depth", "1", "origin",
                GROMO_COMMIT], check=True)
subprocess.run(["git", "-C", str(GROMO), "checkout", "--detach", GROMO_COMMIT],
               check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(GROMO)],
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
# Populate the shared weight cache and verify torchvision/Gromo parity before
# launching two independent processes.
sys.path.insert(0, str(REFERENCE))
from probe import build_pretrained_gromo_resnet18
smoke_model = build_pretrained_gromo_resnet18(100, device="cuda:0")
assert [int(ref.module.hidden_neurons) for ref in smoke_model.growing_blocks()] == [
    64, 64, 128, 128, 256, 256, 512, 512]
del smoke_model
torch.cuda.empty_cache()

input_root = Path("/kaggle/input")
cifar_dirs = sorted({p.parent.resolve() for p in input_root.rglob("cifar-100-python")})
if not cifar_dirs:
    raise FileNotFoundError(
        "Attach a Kaggle CIFAR-100 dataset containing cifar-100-python")
DATA_ROOT = cifar_dirs[0]
print("CIFAR-100 root:", DATA_ROOT)
"""),
    code("""# Structural-E gate. Increase seeds/epochs only after the measured
# residual, cosine alignment and loss gains look sensible.
METHODS = ["vanilla", "random_projection", "tiny_projection",
           "expand_train_project", "real_e_oracle"]
SEEDS = [0, 1]
EPOCHS = 3
WARMUP_EPOCHS = 3
BATCH_SIZE = 64
TRAIN_SAMPLES = 12000       # Set 0 for all non-validation training examples.
VALIDATION_SAMPLES = 5000
SITE = "stages.2.blocks.0"
RANK = 4
CG_ITERATIONS = 12
IMAGE_SIZE = 128
STATISTICS_SAMPLES = 256
PROJECTION_SAMPLES = 32
CHECK_SAMPLES = 128
EPSILONS = [0.01, 0.05, 0.1]

jobs = []
for seed in SEEDS:
    for method in METHODS:
        gates = EPSILONS if method == "tiny_projection" else [0.05]
        jobs.extend((method, seed, epsilon) for epsilon in gates)
print(f"Scheduled {len(jobs)} arms in {len(jobs) / 2:.0f} two-GPU waves")
"""),
    code("""# Create exactly one warm-up checkpoint per seed. All method arms load
# both model weights and SGD momentum from this shared artifact.
WARMUP_ROOT = OUTPUT / "warmup"
WARMUP_ROOT.mkdir(parents=True, exist_ok=True)
warmup_processes = []
WARMUP_CHECKPOINTS = {}
for gpu, seed in enumerate(SEEDS):
    checkpoint = WARMUP_ROOT / f"seed{seed}.pt"
    WARMUP_CHECKPOINTS[seed] = checkpoint
    warmup_output = WARMUP_ROOT / f"seed{seed}_manifest"
    command = [
        sys.executable, "-m", "experiments.run_gromo_pilot",
        "--method", "vanilla", "--seed", str(seed),
        "--prepare-warmup", "--warmup-epochs", str(WARMUP_EPOCHS),
        "--warmup-checkpoint", str(checkpoint),
        "--batch-size", str(BATCH_SIZE),
        "--reference-root", str(REFERENCE),
        "--train-samples", str(TRAIN_SAMPLES),
        "--validation-samples", str(VALIDATION_SAMPLES),
        "--statistics-samples", str(STATISTICS_SAMPLES),
        "--projection-samples", str(PROJECTION_SAMPLES),
        "--check-samples", str(CHECK_SAMPLES),
        "--image-size", str(IMAGE_SIZE),
        "--data-root", str(DATA_ROOT), "--output", str(warmup_output),
    ]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="2")
    warmup_output.mkdir(parents=True, exist_ok=True)
    log = (warmup_output / "run.log").open("a")
    process = subprocess.Popen(
        command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    warmup_processes.append((seed, process, log))

for seed, process, log in warmup_processes:
    return_code = process.wait()
    log.close()
    if return_code:
        raise RuntimeError(f"Warm-up failed for seed {seed}: exit {return_code}")
print("Shared warm-up checkpoints:", WARMUP_CHECKPOINTS)
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
            method, seed, epsilon = job_queue.get_nowait()
        except queue.Empty:
            return
        epsilon_label = str(epsilon).replace(".", "p")
        label = f"{method}_eps{epsilon_label}_seed{seed}"
        arm_dir = OUTPUT / label
        result = arm_dir / "result.json"
        if result.is_file():
            print(f"[{label}] completed; skipping", flush=True)
            job_queue.task_done()
            continue
        command = [
            sys.executable, "-m", "experiments.run_gromo_pilot",
            "--method", method, "--seed", str(seed),
            "--epochs", str(EPOCHS), "--batch-size", str(BATCH_SIZE),
            "--warmup-epochs", str(WARMUP_EPOCHS),
            "--warmup-checkpoint", str(WARMUP_CHECKPOINTS[seed]),
            "--reference-root", str(REFERENCE),
            "--train-samples", str(TRAIN_SAMPLES),
            "--validation-samples", str(VALIDATION_SAMPLES),
            "--statistics-samples", str(STATISTICS_SAMPLES),
            "--projection-samples", str(PROJECTION_SAMPLES),
            "--check-samples", str(CHECK_SAMPLES),
            "--image-size", str(IMAGE_SIZE),
            "--site", SITE, "--rank", str(RANK),
            "--probe-epsilon", str(epsilon),
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
for method, seed, epsilon in jobs:
    epsilon_label = str(epsilon).replace(".", "p")
    path = OUTPUT / f"{method}_eps{epsilon_label}_seed{seed}" / "result.json"
    result = json.loads(path.read_text())
    expected_growth = method == "real_e_oracle"
    if (result["deploy_parameter_delta"] > 0) != expected_growth:
        raise RuntimeError(f"Deploy-size invariant failed: {path}")
    rows.append(result)

for seed in SEEDS:
    hashes = {row["initial_model_sha256"] for row in rows if row["seed"] == seed}
    if len(hashes) != 1:
        raise RuntimeError(f"Methods do not share initialization for seed {seed}: {hashes}")
    for epoch in range(1, EPOCHS + 1):
        audits = {(row["history"][epoch - 1]["diagnostics"]["statistics_indices_sha256"],
                   row["history"][epoch - 1]["diagnostics"]["projection_indices_sha256"])
                  for row in rows if row["seed"] == seed and row["method"] != "vanilla"}
        if len(audits) != 1:
            raise RuntimeError(
                f"Probe batches differ across methods for seed {seed}, epoch {epoch}")
    epoch_audits = [next(row for row in rows
                         if row["seed"] == seed and row["method"] == "tiny_projection")
                    ["history"][epoch]["diagnostics"]["projection_indices_sha256"]
                    for epoch in range(EPOCHS)]
    if len(set(epoch_audits)) != EPOCHS:
        raise RuntimeError(f"Projection batches were reused for seed {seed}")

summary = {"repo_commit": commit, "test_evaluated": False, "arms": {}}
groups = [(method, epsilon)
          for method in METHODS
          for epsilon in (EPSILONS if method == "tiny_projection" else [0.05])]
for method, epsilon in groups:
    key = f"{method}@epsilon={epsilon}"
    selected = [row for row in rows
                if row["method"] == method and
                   row["config"]["probe_epsilon"] == epsilon]
    values = [row["validation_accuracy"] for row in selected]
    residuals = [epoch["diagnostics"]["relative_residual"]
                 for row in selected
                 for epoch in row["history"]
                 if epoch["diagnostics"] and
                    "relative_residual" in epoch["diagnostics"]]
    cosines = [epoch["diagnostics"]["cosine_alignment"]
               for row in selected
               for epoch in row["history"]
               if epoch["diagnostics"] and
                  "cosine_alignment" in epoch["diagnostics"]]
    structural_gains = [epoch["diagnostics"].get(
                            "structural_loss_gain",
                            epoch["diagnostics"].get("local_structural_loss_gain"))
                        for row in selected
                        for epoch in row["history"]
                        if epoch["diagnostics"] and
                           ("structural_loss_gain" in epoch["diagnostics"] or
                            "local_structural_loss_gain" in epoch["diagnostics"])]
    check_structural_gains = [epoch["diagnostics"]["check_structural_loss_gain"]
                              for row in selected
                              for epoch in row["history"]
                              if epoch["diagnostics"] and
                                 epoch["diagnostics"].get("check_structural_loss_gain") is not None]
    projected_gains = [epoch["diagnostics"].get(
                           "check_projected_loss_gain",
                           epoch["diagnostics"].get("check_local_projected_loss_gain"))
                       for row in selected
                       for epoch in row["history"]
                       if epoch["diagnostics"] and
                          (epoch["diagnostics"].get("check_projected_loss_gain") is not None or
                           epoch["diagnostics"].get("check_local_projected_loss_gain") is not None)]
    e_times = [epoch["diagnostics"]["e_statistics_solve_seconds"]
               for row in selected for epoch in row["history"]
               if epoch["diagnostics"]]
    projection_times = [epoch["diagnostics"]["projection_seconds"]
                        for row in selected for epoch in row["history"]
                        if epoch["diagnostics"]]
    peak_memory = [epoch["diagnostics"]["gpu_peak_allocated_bytes"]
                   for row in selected for epoch in row["history"]
                   if epoch["diagnostics"]]
    peak_reserved = [epoch["diagnostics"]["gpu_peak_reserved_bytes"]
                     for row in selected for epoch in row["history"]
                     if epoch["diagnostics"]]
    jvp_calls = [epoch["diagnostics"]["jvp_calls"]
                 for row in selected for epoch in row["history"]
                 if epoch["diagnostics"]]
    vjp_calls = [epoch["diagnostics"]["vjp_calls"]
                 for row in selected for epoch in row["history"]
                 if epoch["diagnostics"]]
    recovery = [epoch["diagnostics"]["check_local_recovery_fraction"]
                for row in selected for epoch in row["history"]
                if epoch["diagnostics"] and
                   epoch["diagnostics"].get("check_local_recovery_fraction") is not None]
    summary["arms"][key] = {
        "method": method,
        "probe_epsilon": epsilon,
        "validation_accuracy_mean": statistics.mean(values),
        "validation_accuracy_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "mean_projection_residual": statistics.mean(residuals) if residuals else None,
        "mean_cosine_alignment": statistics.mean(cosines) if cosines else None,
        "mean_projection_structural_loss_gain": statistics.mean(structural_gains) if structural_gains else None,
        "mean_held_out_structural_loss_gain": statistics.mean(check_structural_gains) if check_structural_gains else None,
        "mean_held_out_projected_loss_gain": statistics.mean(projected_gains) if projected_gains else None,
        "mean_e_statistics_solve_seconds": statistics.mean(e_times) if e_times else None,
        "mean_projection_seconds": statistics.mean(projection_times) if projection_times else None,
        "max_gpu_peak_allocated_bytes": max(peak_memory) if peak_memory else None,
        "max_gpu_peak_reserved_bytes": max(peak_reserved) if peak_reserved else None,
        "mean_jvp_calls": statistics.mean(jvp_calls) if jvp_calls else None,
        "mean_vjp_calls": statistics.mean(vjp_calls) if vjp_calls else None,
        "mean_held_out_local_recovery_fraction": statistics.mean(recovery) if recovery else None,
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
