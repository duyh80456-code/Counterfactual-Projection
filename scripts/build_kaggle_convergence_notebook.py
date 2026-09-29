"""Build the resumable Vanilla theta300-to-plateau Kaggle notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Phase 1 — Vanilla theta300 until best-checkpoint stall

This notebook trains only Vanilla from the shared theta300 checkpoint. It keeps
the original training indices, optimizer momentum, RNG, and loader state. The
LR stored at theta300 is held constant as an explicitly labelled matched
extended-convergence protocol, preventing a zero-LR scheduler horizon from
creating a false plateau.

The original held-out 5,000 examples are split into 2,000 trigger/selection and
3,000 report-only evaluation examples. Every exact trigger improvement is saved
immediately as `checkpoint_best.pt`, even if tiny. The independent stall clock
resets only after a trigger improvement of at least 0.1 percentage point. After
100 epochs without such a significant improvement, the run
forks from that saved best state—not from the later,
possibly degraded state. Epoch 500 is only a review horizon. If no stall is
found, attach this notebook's output, increase `MAX_EPOCH`, and rerun; both
latest progress and the exact best state are restored.
"""),
    code("""import hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
from kaggle_secrets import UserSecretsClient

MAIN_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
REPO = Path("/kaggle/working/counterfactual-projection")
REFERENCE_URL = "https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git"
REFERENCE = Path("/kaggle/working/One-Shot-TAS-CCIL")
GROMO_URL = "https://github.com/growingnet/gromo.git"
GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"
GROMO = Path("/kaggle/working/gromo")
OUTPUT = Path("/kaggle/working/vanilla_best_stall_theta300_v2")
MAX_EPOCH = 500  # raise this on a resumed run if plateau_found is false

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
if torch.cuda.device_count() < 1:
    raise RuntimeError("Select a Kaggle GPU accelerator")
cifar_dirs = sorted({path.parent.resolve()
                     for path in Path("/kaggle/input").rglob("cifar-100-python")})
if not cifar_dirs: raise FileNotFoundError("Attach CIFAR-100")
DATA_ROOT = cifar_dirs[0]
"""),
    code("""def materialize(path, index, prefix):
    if path.is_file(): return path
    pickles = list(path.rglob("data.pkl"))
    if len(pickles) != 1: return None
    root = pickles[0].parent
    target = OUTPUT / "repacked_input" / f"{prefix}_{index}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for record in sorted(root.rglob("*")):
            if record.is_file():
                relative = record.relative_to(root).as_posix()
                archive.writestr(f"{prefix}/{relative}", record.read_bytes())
    return target

forks = []
for index, raw in enumerate(Path("/kaggle/input").rglob("shared_seed1_epoch300.pt")):
    path = materialize(raw, index, "shared_seed1_epoch300")
    if path is None: continue
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") == "shared_fork_checkpoint" and payload.get("epoch") == 300:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        forks.append((path, digest))
if not forks: raise FileNotFoundError("Attach shared_seed1_epoch300.pt")
if len({digest for _, digest in forks}) != 1:
    raise RuntimeError("Multiple different theta300 checkpoints attached")
THETA300, THETA300_HASH = forks[0]

# Optional continuation from an earlier compatible Phase-1 output.
resume_payload = None
for index, raw in enumerate(Path("/kaggle/input").rglob("checkpoint_latest.pt")):
    path = materialize(raw, index, "checkpoint_latest")
    if path is None: continue
    try: payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception: continue
    if (payload.get("kind") == "vanilla_convergence_progress" and
            payload.get("protocol", {}).get("source_checkpoint_hash") == THETA300_HASH and
            payload.get("protocol", {}).get("schedule_id") == "constant-theta300-lr-best-stall-v1"):
        shutil.copy2(path, OUTPUT / "checkpoint_latest.pt")
        resume_payload = payload
        print("Restored Phase-1 progress at epoch", payload["epoch"])
        break
if resume_payload is not None:
    matching_best = []
    for index, raw in enumerate(Path("/kaggle/input").rglob("checkpoint_best.pt")):
        path = materialize(raw, index, "checkpoint_best")
        if path is None: continue
        try: payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception: continue
        if (payload.get("kind") == "vanilla_best_checkpoint" and
                payload.get("protocol") == resume_payload.get("protocol") and
                int(payload.get("epoch", -1)) ==
                int(resume_payload["best_stall_detector"]["best_epoch"])):
            matching_best.append(path)
    if len(matching_best) != 1:
        raise RuntimeError("Resume requires exactly one matching checkpoint_best.pt")
    shutil.copy2(matching_best[0], OUTPUT / "checkpoint_best.pt")
    print("Restored best checkpoint at epoch",
          resume_payload["best_stall_detector"]["best_epoch"])
print("theta300:", THETA300, THETA300_HASH)
"""),
    code("""command = [
    sys.executable, "-m", "experiments.run_vanilla_to_plateau",
    "--reference-root", str(REFERENCE), "--data-root", str(DATA_ROOT),
    "--fork-checkpoint", str(THETA300),
    "--fork-checkpoint-hash", THETA300_HASH,
    "--output", str(OUTPUT), "--max-epoch", str(MAX_EPOCH),
    "--seed", "1", "--batch-size", "64", "--validation-samples", "5000",
    "--trigger-samples", "2000",
    "--no-new-best-patience", "100",
    "--best-min-gain", "0.001"]
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
           PYTHONPATH=RUNTIME_PYTHONPATH)
process = subprocess.Popen(command, cwd=REPO, env=env,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
with (OUTPUT / "run.log").open("a", buffering=1) as log:
    for line in process.stdout:
        log.write(line); print(line, end="", flush=True)
if process.wait(): raise RuntimeError("Vanilla convergence search failed")
"""),
    code("""result = json.loads((OUTPUT / "result.json").read_text())
print(json.dumps({key: result[key] for key in (
    "status", "plateau_found", "plateau_epoch", "best_epoch",
    "stall_detected_epoch", "epochs_without_improvement", "review_epoch_reached",
    "final_validation_accuracy", "best_validation_accuracy",
    "final_validation_loss", "plateau_checkpoint")}, indent=2))
if not result["plateau_found"]:
    print("NO CONVERGENCE CLAIM: attach this output, increase MAX_EPOCH, and continue.")
else:
    print("Phase 2 fork is ready:", result["plateau_checkpoint"])
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
destination = Path("notebooks/kaggle_vanilla_to_plateau.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
