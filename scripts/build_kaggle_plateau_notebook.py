"""Build the theta360-to-500 plateau-triggered comparison notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Plateau-triggered E→O from vanilla theta360

This T4x2 run forks the exact completed `vanilla_continue` checkpoint at epoch
360 into two equal 140-epoch continuations ending at epoch 500:

- GPU 0: Vanilla continuation;
- GPU 1: plateau-triggered structural E→O.

Both arms preserve model weights, SGD momentum, RNG, split, and loader state.
Because the previous cosine segment ended at zero LR, both arms use the same
explicit cosine restart at LR 0.01 for 140 epochs. The official test set is
never constructed.

E→O is not run every epoch. A separate 128-example trigger set declares a
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
OUTPUT = Path("/kaggle/working/plateau_eo_theta360_500_v1")

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
    target = OUTPUT / "repacked_input" / f"theta360_{index}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for record in sorted(root.rglob("*")):
            if record.is_file():
                relative = record.relative_to(root).as_posix()
                archive.writestr(f"checkpoint_latest/{relative}", record.read_bytes())
    return target

candidates = []
for index, raw in enumerate(Path("/kaggle/input").rglob("checkpoint_latest.pt")):
    path = materialize(raw, index)
    if path is None: continue
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print("Ignoring unreadable checkpoint", raw, repr(error))
        continue
    protocol = payload.get("protocol", {})
    history = payload.get("history", [])
    if (protocol.get("method") == "vanilla_continue" and
            int(payload.get("post_epoch", -1)) == 60 and history and
            int(history[-1].get("epoch", -1)) == 360):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        candidates.append((path, digest.hexdigest(), raw))
if not candidates:
    raise FileNotFoundError(
        "Attach vanilla_continue/checkpoint_latest.pt completed at epoch360")
hashes = {item[1] for item in candidates}
if len(hashes) != 1:
    raise RuntimeError("Multiple different valid vanilla theta360 checkpoints attached")
THETA360, THETA360_HASH, RAW_THETA360 = candidates[0]
print("vanilla theta360 input:", RAW_THETA360)
print("materialized checkpoint:", THETA360)
print("theta360 SHA-256:", THETA360_HASH)
"""),
    code("""SEED = 1
def base_args(output):
    return [
        "--reference-root", str(REFERENCE), "--data-root", str(DATA_ROOT),
        "--resume-checkpoint", str(THETA360),
        "--resume-checkpoint-hash", THETA360_HASH,
        "--output", str(output), "--seed", str(SEED),
        "--batch-size", "64", "--validation-samples", "5000",
        "--tuning-samples", "128", "--continuation-lr", "0.01",
        "--weight-decay", "0.0005"]

vanilla = [sys.executable, "-m", "experiments.run_plateau_comparison",
           "--method", "vanilla_continue"] + base_args(OUTPUT / "vanilla_continue")
plateau = [sys.executable, "-m", "experiments.run_plateau_comparison",
           "--method", "plateau_e_driven_o"] + base_args(OUTPUT / "plateau_e_driven_o") + [
    "--plateau-window", "15", "--plateau-accuracy-min-gain", "0.0005",
    "--plateau-loss-ema-min-drop", "0.001", "--minimum-sgd-epochs", "10",
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
    if result["source_checkpoint_hash"] != THETA360_HASH:
        raise RuntimeError(f"{name} used a different theta360")
    if result["final_epoch"] != 500 or result["continuation_epochs"] != 140:
        raise RuntimeError(f"{name} did not finish epoch500")
    results[name] = result

summary = {
    "source_epoch": 360, "final_epoch": 500,
    "continuation_epochs": 140,
    "source_checkpoint_hash": THETA360_HASH,
    "official_test_used": False,
    "results": {
        name: {key: value for key, value in result.items() if key in {
            "final_validation_accuracy", "best_validation_accuracy",
            "final_validation_loss", "best_validation_loss",
            "validation_accuracy_delta", "intervention_count",
            "correction_application_count", "correction_application_rate",
            "training_seconds", "peak_gpu_memory", "deploy_params",
            "optimizer_state_preserved", "scheduler_restarted"}}
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

