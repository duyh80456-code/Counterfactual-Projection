"""Build the restart-safe four-arm Kaggle T4x2 experiment notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Counterfactual Projection — four-arm CIFAR-100, 80 epochs

This notebook runs one fixed seed in two waves on two T4 GPUs:

- wave 1: `ours_e_driven_o` and official RepAn;
- wave 2: official ExpandNets and official RepOptimizer.

Every arm checkpoints after every epoch. Re-running resumes from
`checkpoint_latest.pt`; increasing `TARGET_EPOCHS` continues the same phase.
No arm reads the official CIFAR-100 test set. ExpandNets' official CIFAR model
is intrinsically 32x32, so its thin input adapter downsamples the common 128px
batch and records that protocol deviation.
"""),
    code("""import json, os, shutil, subprocess, sys
from pathlib import Path
from kaggle_secrets import UserSecretsClient

MAIN_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
REPO = Path("/kaggle/working/counterfactual-projection")
REFERENCE_URL = "https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git"
REFERENCE = Path("/kaggle/working/One-Shot-TAS-CCIL")
GROMO_URL = "https://github.com/growingnet/gromo.git"
GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"
GROMO = Path("/kaggle/working/gromo")
THIRD_PARTY = Path("/kaggle/working/third_party")
OUTPUT = Path("/kaggle/working/counterfactual_projection_4arm_80ep_v1")

OFFICIAL = {
    "repan": {"url": "https://github.com/xfey/RepAn.git",
        "commit": "7cb05e93cdcd8cd83f18da1b5514bad75bc68e16",
        "path": THIRD_PARTY / "RepAn",
        "license": "NOT SPECIFIED (no license file at pinned commit)"},
    "expandnets": {"url": "https://github.com/GUOShuxuan/expandnets.git",
        "commit": "065d4d3aebfeb442c02227d1d5c16ee11a518945",
        "path": THIRD_PARTY / "ExpandNets",
        "license": "BSD-3-Clause terms (LICENSE heading says MIT License)"},
    "repoptimizer": {"url": "https://github.com/DingXiaoH/RepOptimizers.git",
        "commit": "2e45ff5388e9d7aabf112d7e2973df8183e6c6d9",
        "path": THIRD_PARTY / "RepOptimizers", "license": "MIT"},
}

def pinned_clone(url, destination, commit, env=None):
    if destination.exists(): shutil.rmtree(destination)
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                    url, str(destination)], env=env, check=True)
    subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1",
                    "origin", commit], env=env, check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", commit],
                   env=env, check=True)
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError(f"revision mismatch for {destination}: {actual}")

for checkout in (REPO, REFERENCE, GROMO):
    if checkout.exists(): shutil.rmtree(checkout)
THIRD_PARTY.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)  # deliberately never deleted

token = UserSecretsClient().get_secret("github_token").strip()
if not token: raise RuntimeError("Kaggle Secret github_token is unavailable")
askpass = Path("/kaggle/working/.counterfactual_git_askpass.py")
askpass.write_text("#!/usr/bin/env python3\\nimport os,sys\\np=sys.argv[1] if len(sys.argv)>1 else ''\\nprint('x-access-token' if 'Username' in p else os.environ['GITHUB_TOKEN_RUNTIME'])\\n")
askpass.chmod(0o700)
private_env = os.environ.copy()
private_env.update(GITHUB_TOKEN_RUNTIME=token, GIT_ASKPASS=str(askpass),
                   GIT_TERMINAL_PROMPT="0")
try:
    subprocess.run(["git", "clone", "--branch", "main", "--single-branch",
                    MAIN_URL, str(REPO)], env=private_env, check=True)
    subprocess.run(["git", "clone", "--branch", "ccil-residual-capacity",
                    "--single-branch", REFERENCE_URL, str(REFERENCE)],
                   env=private_env, check=True)
finally:
    askpass.unlink(missing_ok=True)
    private_env.pop("GITHUB_TOKEN_RUNTIME", None)
    token = None

pinned_clone(GROMO_URL, GROMO, GROMO_COMMIT)
for metadata in OFFICIAL.values():
    pinned_clone(metadata["url"], metadata["path"], metadata["commit"])
manifest = {name: {"method": name, "repo_url": item["url"],
    "git_commit_head": item["commit"], "license": item["license"]}
    for name, item in OFFICIAL.items()}
(OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
print(json.dumps(manifest, indent=2, sort_keys=True))
"""),
    code("""subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(GROMO)], check=True)
GROMO_SRC = GROMO / "src"
RUNTIME_PYTHONPATH = os.pathsep.join(filter(None, (str(REPO), str(GROMO_SRC), str(REFERENCE), os.environ.get("PYTHONPATH", ""))))
test_env = os.environ.copy()
test_env.update(PYTHONPATH=RUNTIME_PYTHONPATH, REQUIRE_GROMO_INTEGRATION="1")
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO, env=test_env, check=True)

import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError(f"Select Kaggle T4 x2; found {torch.cuda.device_count()} GPU(s)")
print(subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], text=True))
input_root = Path("/kaggle/input")
cifar_dirs = sorted({p.parent.resolve() for p in input_root.rglob("cifar-100-python")})
if not cifar_dirs:
    raise FileNotFoundError("Attach a Kaggle CIFAR-100 dataset containing cifar-100-python")
DATA_ROOT = cifar_dirs[0]
print("CIFAR-100 root:", DATA_ROOT)
"""),
    code("""SEED = 1
TARGET_EPOCHS = 80  # increase later; checkpoints resume instead of restarting
WARMUP_EPOCHS = 3
BATCH_SIZE = 64
TRAIN_SAMPLES = 12000
VALIDATION_SAMPLES = 5000
TUNING_SAMPLES = 128
IMAGE_SIZE = 128
LR = 0.01

# To continue in a fresh Kaggle session, attach the previous output archive as
# a Dataset. The notebook auto-detects and restores its four arm directories.
for summary_path in Path("/kaggle/input").rglob("summary.json"):
    prior_root = summary_path.parent
    if all((prior_root / name / "checkpoint_latest.pt").is_file()
           for name in ("ours_e_driven_o", "repan", "expandnets", "repoptimizer")):
        for child in prior_root.iterdir():
            destination = OUTPUT / child.name
            if destination.exists():
                continue
            if child.is_dir(): shutil.copytree(child, destination)
            else: shutil.copy2(child, destination)
        print("Restored prior phase from", prior_root)
        break

legacy_warmup = Path("/kaggle/working/counterfactual_projection_t4x2_fair_v15/warmup/seed1.pt")
WARMUP_CHECKPOINT = OUTPUT / "warmup" / "seed1.pt"
if not WARMUP_CHECKPOINT.is_file() and legacy_warmup.is_file():
    WARMUP_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_warmup, WARMUP_CHECKPOINT)
if not WARMUP_CHECKPOINT.is_file():
    warmup_output = OUTPUT / "warmup" / "seed1_manifest"
    warmup_output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "experiments.run_gromo_pilot",
        "--method", "vanilla", "--seed", str(SEED), "--prepare-warmup",
        "--warmup-epochs", str(WARMUP_EPOCHS), "--warmup-checkpoint", str(WARMUP_CHECKPOINT),
        "--batch-size", str(BATCH_SIZE), "--reference-root", str(REFERENCE),
        "--train-samples", str(TRAIN_SAMPLES), "--validation-samples", str(VALIDATION_SAMPLES),
        "--tuning-samples", str(TUNING_SAMPLES), "--image-size", str(IMAGE_SIZE),
        "--data-root", str(DATA_ROOT), "--output", str(warmup_output)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1", PYTHONPATH=RUNTIME_PYTHONPATH)
    subprocess.run(command, cwd=REPO, env=env, check=True)
print("Ours warm-up:", WARMUP_CHECKPOINT)
"""),
    code("""def common_official_args(name):
    source = OFFICIAL[name]
    return ["--official-root", str(source["path"]), "--data-root", str(DATA_ROOT),
        "--output", str(OUTPUT / name), "--seed", str(SEED), "--epochs", str(TARGET_EPOCHS),
        "--batch-size", str(BATCH_SIZE), "--train-samples", str(TRAIN_SAMPLES),
        "--validation-samples", str(VALIDATION_SAMPLES), "--tuning-samples", str(TUNING_SAMPLES),
        "--image-size", str(IMAGE_SIZE), "--lr", str(LR), "--source-url", source["url"],
        "--source-commit", source["commit"], "--license", source["license"]]

ours = [sys.executable, "-m", "experiments.run_gromo_pilot", "--method", "ours_e_driven_o",
    "--seed", str(SEED), "--epochs", str(TARGET_EPOCHS), "--warmup-epochs", str(WARMUP_EPOCHS),
    "--warmup-checkpoint", str(WARMUP_CHECKPOINT), "--batch-size", str(BATCH_SIZE),
    "--reference-root", str(REFERENCE), "--train-samples", str(TRAIN_SAMPLES),
    "--validation-samples", str(VALIDATION_SAMPLES), "--tuning-samples", str(TUNING_SAMPLES),
    "--image-size", str(IMAGE_SIZE), "--data-root", str(DATA_ROOT),
    "--output", str(OUTPUT / "ours_e_driven_o"), "--site", "stages.2.blocks.0",
    "--rank", "4", "--probe-epsilon", "0.05", "--cg-iterations", "200",
    "--cg-relative-tolerance", "1e-2", "--cg-preconditioner-probes", "8"]
repan = [sys.executable, "-m", "baselines.run_repan"] + common_official_args("repan")
expandnets = [sys.executable, "-m", "baselines.run_expandnets"] + common_official_args("expandnets")
repoptimizer = [sys.executable, "-m", "baselines.run_repoptimizer"] + common_official_args("repoptimizer")

def run_wave(assignments):
    running = []
    for gpu, name, command in assignments:
        arm_dir = OUTPUT / name
        arm_dir.mkdir(parents=True, exist_ok=True)
        log = (arm_dir / "run.log").open("a")
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", PYTHONPATH=RUNTIME_PYTHONPATH)
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        running.append((name, process, log))
        print(f"GPU{gpu}: started {name}, pid={process.pid}")
    failures = []
    for name, process, log in running:
        return_code = process.wait(); log.close()
        print(f"{name}: exit={return_code}")
        if return_code: failures.append((name, return_code))
    if failures:
        tails = {name: (OUTPUT / name / "run.log").read_text(errors="replace").splitlines()[-80:]
                 for name, _ in failures}
        raise RuntimeError(json.dumps({"failures": failures, "log_tails": tails}, indent=2))

print("Wave 1/2")
run_wave([(0, "ours_e_driven_o", ours), (1, "repan", repan)])
"""),
    code("""print("Wave 2/2")
run_wave([(0, "expandnets", expandnets), (1, "repoptimizer", repoptimizer)])
"""),
    code("""required = {"method", "seed", "epoch", "train_accuracy", "validation_accuracy",
    "validation_loss", "peak_train_params", "deploy_params", "training_seconds",
    "peak_gpu_memory", "source_repo", "source_commit"}
results = []
for name in ("ours_e_driven_o", "repan", "expandnets", "repoptimizer"):
    result = json.loads((OUTPUT / name / "result.json").read_text())
    missing = sorted(required - result.keys())
    if missing: raise RuntimeError(f"{name} missing result fields: {missing}")
    if result["epoch"] != TARGET_EPOCHS: raise RuntimeError(f"{name} stopped at epoch {result['epoch']}")
    if not (OUTPUT / name / "checkpoint_latest.pt").is_file():
        raise RuntimeError(f"{name} has no resumable checkpoint")
    results.append(result)
summary = {"target_epochs": TARGET_EPOCHS, "seed": SEED, "official_test_used": False,
    "results": [{key: row.get(key) for key in sorted(required)} for row in results],
    "continuation": "increase TARGET_EPOCHS and rerun; completed epochs are skipped"}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
archive = shutil.make_archive(str(OUTPUT), "gztar", root_dir=OUTPUT)
print("Archive:", archive)
"""),
]

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "dataSources": []}},
    "nbformat": 4, "nbformat_minor": 5}
destination = Path("notebooks/kaggle_counterfactual_projection_t4x2.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
