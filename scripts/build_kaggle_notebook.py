"""Build the restart-safe shared-theta300 Kaggle T4x2 notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Shared-checkpoint CIFAR-100 comparison

The full theta_150 checkpoint is resumed with its optimizer and RNG state, then
continued under Vanilla for another 150 epochs. The exact
model, optimizer, scheduler, data split, loader generator, and RNG state at
`theta_300` are hashed and forked into three 50-epoch arms:

- `ours_e_driven_o` (GPU 0) and relaxed Bypass (GPU 1), concurrently;
- `vanilla_continue` (GPU 0) in wave 2.

The total budget is 350 epochs for every arm. The official CIFAR-100 test set is
never constructed. Every process saves a resumable checkpoint each epoch.
Bypass treats opt2 epoch 10 as a soft cap: it never force-projects a nonzero D,
continues opt2 within the remaining budget, and is rejected by aggregation if
the contraction criterion is still unmet at epoch 350.
"""),
    code("""import json, os, shutil, subprocess, sys, threading
from pathlib import Path
from kaggle_secrets import UserSecretsClient

MAIN_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
REPO = Path("/kaggle/working/counterfactual-projection")
REFERENCE_URL = "https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git"
REFERENCE = Path("/kaggle/working/One-Shot-TAS-CCIL")
GROMO_URL = "https://github.com/growingnet/gromo.git"
GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"
GROMO = Path("/kaggle/working/gromo")
OUTPUT = Path("/kaggle/working/counterfactual_shared_theta300_350ep_v3")

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
        env.pop("GITHUB_TOKEN_RUNTIME", None)
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
if subprocess.check_output(["git", "-C", str(GROMO), "rev-parse", "HEAD"],
                           text=True).strip() != GROMO_COMMIT:
    raise RuntimeError("Gromo revision mismatch")
print("main:", subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())
print("reference:", subprocess.check_output(
    ["git", "-C", str(REFERENCE), "rev-parse", "HEAD"], text=True).strip())
print("gromo:", GROMO_COMMIT)
"""),
    code("""subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(GROMO)], check=True)
GROMO_SRC = GROMO / "src"
RUNTIME_PYTHONPATH = os.pathsep.join(filter(None, (
    str(REPO), str(GROMO_SRC), str(REFERENCE), os.environ.get("PYTHONPATH", ""))))
test_env = os.environ.copy()
test_env.update(PYTHONPATH=RUNTIME_PYTHONPATH, REQUIRE_GROMO_INTEGRATION="1")
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO,
               env=test_env, check=True)

import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError(f"Select Kaggle T4 x2; found {torch.cuda.device_count()} GPU(s)")
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
    text=True))
cifar_dirs = sorted({p.parent.resolve()
                     for p in Path("/kaggle/input").rglob("cifar-100-python")})
if not cifar_dirs:
    raise FileNotFoundError(
        "Attach a Kaggle CIFAR-100 dataset containing cifar-100-python")
DATA_ROOT = cifar_dirs[0]
print("CIFAR-100 root:", DATA_ROOT)
"""),
    code("""SEED = 1
BOOTSTRAP_EPOCH = 150
FORK_EPOCH = 300
TOTAL_EPOCHS = 350
POST_FORK_EPOCHS = 50
BATCH_SIZE = 64
VALIDATION_SAMPLES = 5000
TUNING_SAMPLES = 128
LR = 0.1
WEIGHT_DECAY = 5e-4
SHARED_CHECKPOINT = OUTPUT / "warmup" / "shared_seed1_epoch300.pt"

# Restore a previous Kaggle output archive/dataset before deciding what to run.
for prior_manifest in Path("/kaggle/input").rglob("shared_seed1_epoch300.json"):
    prior_root = prior_manifest.parent.parent
    if (prior_root / "warmup" / "shared_seed1_epoch300.pt").is_file():
        for child in prior_root.iterdir():
            destination = OUTPUT / child.name
            if destination.exists(): continue
            if child.is_dir(): shutil.copytree(child, destination)
            else: shutil.copy2(child, destination)
        print("Restored prior run from", prior_root)
        break

def base_args(output):
    return ["--reference-root", str(REFERENCE), "--data-root", str(DATA_ROOT),
        "--output", str(output), "--shared-checkpoint", str(SHARED_CHECKPOINT),
        "--seed", str(SEED), "--batch-size", str(BATCH_SIZE),
        "--validation-samples", str(VALIDATION_SAMPLES),
        "--tuning-samples", str(TUNING_SAMPLES), "--lr", str(LR),
        "--weight-decay", str(WEIGHT_DECAY)]

theta150_manifest = None
local_theta150 = Path("/kaggle/working/counterfactual_shared_theta150_fresh_200ep_v2/warmup/shared_seed1_epoch150.json")
if local_theta150.is_file():
    theta150_manifest = local_theta150
else:
    theta150_manifest = next(
        Path("/kaggle/input").rglob("shared_seed1_epoch150.json"), None)
if theta150_manifest is None and not SHARED_CHECKPOINT.is_file():
    raise FileNotFoundError(
        "Attach the fresh shared_seed1_epoch150.pt/json output before continuing")
theta150_args = []
if theta150_manifest is not None and not SHARED_CHECKPOINT.is_file():
    theta150 = json.loads(theta150_manifest.read_text())
    theta150_args = ["--bootstrap-checkpoint", str(theta150_manifest.with_suffix(".pt")),
        "--bootstrap-checkpoint-hash", theta150["sha256"]]
    print("Continuing from theta_150:", theta150_manifest.with_suffix(".pt"))

env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
           PYTHONPATH=RUNTIME_PYTHONPATH)
prepare = [sys.executable, "-m", "experiments.run_shared_comparison",
           "--method", "prepare_shared"] + base_args(OUTPUT / "warmup") + theta150_args
subprocess.run(prepare, cwd=REPO, env=env, check=True)
manifest = json.loads(SHARED_CHECKPOINT.with_suffix(".json").read_text())
SHARED_HASH = manifest["sha256"]
if manifest["epoch"] != FORK_EPOCH:
    raise RuntimeError(f"shared checkpoint is at epoch {manifest['epoch']}")
print("theta_300 SHA-256:", SHARED_HASH)
"""),
    code("""def run_wave(assignments):
    running = []

    def stream_output(name, process, log):
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(f"[{name}] {line}", end="", flush=True)
        process.stdout.close()

    for gpu, name, command in assignments:
        arm_dir = OUTPUT / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        log = (arm_dir / "run.log").open("a")
        process_env = os.environ.copy()
        process_env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                           OMP_NUM_THREADS="2", PYTHONPATH=RUNTIME_PYTHONPATH)
        process = subprocess.Popen(
            command, cwd=REPO, env=process_env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        reader = threading.Thread(
            target=stream_output, args=(name, process, log), daemon=True)
        reader.start()
        running.append((name, process, reader, log))
        print(f"GPU{gpu}: started {name}, pid={process.pid}")
    failures = []
    for name, process, reader, log in running:
        return_code = process.wait()
        reader.join()
        log.close()
        print(f"{name}: exit={return_code}")
        if return_code: failures.append((name, return_code))
    if failures:
        tails = {name: (OUTPUT / name / "run.log").read_text(
            errors="replace").splitlines()[-100:] for name, _ in failures}
        raise RuntimeError(json.dumps(
            {"failures": failures, "log_tails": tails}, indent=2))

ours = [sys.executable, "-m", "experiments.run_shared_comparison",
    "--method", "ours_e_driven_o"] + base_args(OUTPUT / "ours_e_driven_o") + [
    "--shared-checkpoint-hash", SHARED_HASH, "--site", "stages.2.blocks.0",
    "--rank", "4", "--probe-epsilon", "0.05", "--cg-iterations", "200",
    "--cg-relative-tolerance", "1e-2", "--cg-preconditioner-probes", "8"]
bypass = [sys.executable, "-m", "baselines.run_bypass"] + base_args(
    OUTPUT / "bypass") + ["--shared-checkpoint-hash", SHARED_HASH,
    "--opt1-epochs", "20", "--max-opt2-epochs", "10",
    "--contraction-epsilon", "0.002", "--gamma-slope", "3e-6"]

print("Wave 1/2: Ours + Bypass")
run_wave([(0, "ours_e_driven_o", ours), (1, "bypass", bypass)])
"""),
    code("""vanilla = [sys.executable, "-m", "experiments.run_shared_comparison",
    "--method", "vanilla_continue"] + base_args(OUTPUT / "vanilla_continue") + [
    "--shared-checkpoint-hash", SHARED_HASH]
print("Wave 2/2: Vanilla continuation")
run_wave([(0, "vanilla_continue", vanilla)])
"""),
    code("""required = {"method", "shared_checkpoint_hash", "fork_epoch",
    "post_fork_epochs", "final_validation_accuracy", "best_validation_accuracy",
    "final_validation_loss", "training_seconds", "peak_gpu_memory", "deploy_params"}
results = []
for name in ("ours_e_driven_o", "bypass", "vanilla_continue"):
    path = OUTPUT / name / "result.json"
    result = json.loads(path.read_text())
    missing = sorted(required - result.keys())
    if missing: raise RuntimeError(f"{name} missing result fields: {missing}")
    if result["shared_checkpoint_hash"] != SHARED_HASH:
        raise RuntimeError(f"{name} did not fork from theta_300")
    if result["fork_epoch"] != FORK_EPOCH or result["post_fork_epochs"] != POST_FORK_EPOCHS:
        raise RuntimeError(f"{name} did not complete the 300+50 protocol")
    if name == "bypass" and (
            result.get("contraction_criterion_met") is not True or
            result.get("bypass_completed") is not True):
        raise RuntimeError(
            "Bypass did not both reach contraction epsilon and return to "
            "train3; result.json is preserved but is not a valid comparator")
    if not (OUTPUT / name / "checkpoint_latest.pt").is_file():
        raise RuntimeError(f"{name} has no resumable checkpoint")
    results.append(result)

summary = {"dataset": "CIFAR-100", "architecture": "CIFAR-ResNet18",
    "input_size": 32, "seed": SEED, "fork_epoch": FORK_EPOCH,
    "post_fork_epochs": POST_FORK_EPOCHS, "total_epochs": TOTAL_EPOCHS,
    "shared_checkpoint_hash": SHARED_HASH, "official_test_used": False,
    "results": [{key: row.get(key) for key in sorted(required | {
        "correction_application_rate", "actual_cosine_alignment",
        "actual_relative_residual", "opt1_epochs", "opt2_epochs",
        "train3_epochs", "contraction_norm", "projection_loss_jump",
        "contraction_criterion_met", "bypass_completed",
        "opt2_soft_cap_exceeded"})}
        for row in results]}
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
destination = Path("notebooks/kaggle_counterfactual_projection_t4x2.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
