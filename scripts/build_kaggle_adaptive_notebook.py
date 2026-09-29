"""Build the warmup-input-only adaptive E-driven O Kaggle notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Adaptive-site E-driven O from theta_300

This notebook runs only the main adaptive E-driven O arm. It does not train the
warm-up, Vanilla, Bypass, projection-only, or fixed-site ablation.

Required Kaggle inputs:

1. CIFAR-100 containing `cifar-100-python`;
2. the complete `shared_seed1_epoch300.pt` and matching JSON manifest.

For each of 50 epochs, one shared statistics set is used to scan all eight
growing blocks with rank-4 TINY. Raw `proposal_score` only pre-screens top-3.
A separate 16-sample selection batch and 25-CG cheap projection rank those
candidates by transferable loss utility. Full functional projection runs only
when the winner has positive utility and projectability rho at least 0.05. A
full resumable checkpoint is written after every epoch.
"""),
    code("""import hashlib, json, os, shutil, subprocess, sys
from collections import Counter
from pathlib import Path
from kaggle_secrets import UserSecretsClient

MAIN_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
REPO = Path("/kaggle/working/counterfactual-projection")
REFERENCE_URL = "https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git"
REFERENCE = Path("/kaggle/working/One-Shot-TAS-CCIL")
GROMO_URL = "https://github.com/growingnet/gromo.git"
GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"
GROMO = Path("/kaggle/working/gromo")
OUTPUT = Path("/kaggle/working/e_driven_o_when_where_how_theta300_350_v2")

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
if torch.cuda.device_count() < 1:
    raise RuntimeError("Select a Kaggle GPU accelerator")
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
    text=True))

cifar_dirs = sorted({path.parent.resolve()
                     for path in Path("/kaggle/input").rglob("cifar-100-python")})
if not cifar_dirs:
    raise FileNotFoundError(
        "Attach a Kaggle CIFAR-100 dataset containing cifar-100-python")
DATA_ROOT = cifar_dirs[0]

warmups = []
for manifest_path in Path("/kaggle/input").rglob("shared_seed1_epoch300.json"):
    checkpoint_path = manifest_path.with_suffix(".pt")
    if not checkpoint_path.is_file(): continue
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("epoch", -1)) != 300: continue
    warmups.append((manifest_path, checkpoint_path, manifest))
if not warmups:
    raise FileNotFoundError(
        "Attach shared_seed1_epoch300.pt and shared_seed1_epoch300.json")
hashes = {item[2]["sha256"] for item in warmups}
if len(hashes) != 1:
    raise RuntimeError("Multiple different theta_300 checkpoints are attached")
WARMUP_MANIFEST, SHARED_CHECKPOINT, manifest = warmups[0]
SHARED_HASH = manifest["sha256"]

digest = hashlib.sha256()
with SHARED_CHECKPOINT.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != SHARED_HASH:
    raise RuntimeError("theta_300 checkpoint SHA-256 does not match its manifest")
print("CIFAR-100 root:", DATA_ROOT)
print("theta_300:", SHARED_CHECKPOINT)
print("theta_300 SHA-256:", SHARED_HASH)
"""),
    code("""SEED = 1
ARM_OUTPUT = OUTPUT / "ours_e_driven_o"
command = [
    sys.executable, "-m", "experiments.run_shared_comparison",
    "--method", "ours_e_driven_o",
    "--reference-root", str(REFERENCE),
    "--data-root", str(DATA_ROOT),
    "--output", str(ARM_OUTPUT),
    "--shared-checkpoint", str(SHARED_CHECKPOINT),
    "--shared-checkpoint-hash", SHARED_HASH,
    "--seed", str(SEED),
    "--batch-size", "64",
    "--validation-samples", "5000",
    "--tuning-samples", "128",
    "--lr", "0.1",
    "--weight-decay", "5e-4",
    "--site", "auto",
    "--candidate-sites", "",
    "--site-selection-mode", "projectability_utility",
    "--selection-top-k", "3",
    "--selection-samples", "16",
    "--selection-cg-iterations", "25",
    "--selection-cg-relative-tolerance", "5e-2",
    "--selection-preconditioner-probes", "2",
    "--selection-min-utility", "0.0",
    "--selection-min-projectability", "0.05",
    "--rank", "4",
    "--probe-epsilon", "0.05",
    "--cg-iterations", "200",
    "--cg-relative-tolerance", "1e-2",
    "--cg-preconditioner-probes", "8",
]
run_env = os.environ.copy()
run_env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="2", PYTHONPATH=RUNTIME_PYTHONPATH)
ARM_OUTPUT.mkdir(parents=True, exist_ok=True)
with (ARM_OUTPUT / "run.log").open("a") as log:
    process = subprocess.Popen(
        command, cwd=REPO, env=run_env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1)
    for line in process.stdout:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)
    process.stdout.close()
    return_code = process.wait()
if return_code:
    tail = (ARM_OUTPUT / "run.log").read_text(errors="replace").splitlines()[-100:]
    raise RuntimeError(json.dumps({"exit": return_code, "log_tail": tail}, indent=2))
"""),
    code("""result_path = ARM_OUTPUT / "result.json"
checkpoint_path = ARM_OUTPUT / "checkpoint_latest.pt"
if not result_path.is_file() or not checkpoint_path.is_file():
    raise RuntimeError("adaptive run did not produce result and checkpoint files")
result = json.loads(result_path.read_text())
if result["shared_checkpoint_hash"] != SHARED_HASH:
    raise RuntimeError("adaptive arm did not fork from the attached theta_300")
if result["fork_epoch"] != 300 or result["post_fork_epochs"] != 50:
    raise RuntimeError("adaptive arm did not complete epochs 301-350")
if result.get("site_selection_mode") != "tiny_topk_projectability_utility":
    raise RuntimeError("WHEN-WHERE-HOW selector was not active")

selection = result["site_selection_history"]
counts = Counter(row["selected_site"] for row in selection)
dominant_site, dominant_count = counts.most_common(1)[0]
summary = {
    "method": "ours_e_driven_o",
    "site_selection_mode": result["site_selection_mode"],
    "shared_checkpoint_hash": SHARED_HASH,
    "fork_epoch": result["fork_epoch"],
    "post_fork_epochs": result["post_fork_epochs"],
    "final_validation_accuracy": result["final_validation_accuracy"],
    "best_validation_accuracy": result["best_validation_accuracy"],
    "final_validation_loss": result["final_validation_loss"],
    "correction_application_rate": result["correction_application_rate"],
    "when_gate_pass_rate": result["when_gate_pass_rate"],
    "full_projection_attempt_rate": result["full_projection_attempt_rate"],
    "site_counts": dict(counts),
    "dominant_site": dominant_site,
    "dominant_site_fraction": dominant_count / len(selection),
    "site_selection_history": selection,
    "checkpoint": str(checkpoint_path),
}
(OUTPUT / "adaptive_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
if dominant_count == len(selection):
    print("WARNING: one site won every epoch; inspect layer/width score scaling before interpreting it.")
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
destination = Path("notebooks/kaggle_adaptive_e_driven_o.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
