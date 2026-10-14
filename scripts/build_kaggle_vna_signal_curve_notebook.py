"""Build the pinned, resumable Phase-0 T4x2 notebook from existing setup cells."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


# Self-contained setup cells retained for all VNA notebook builders.
_BOOTSTRAP = (
    'import hashlib, json, os, shutil, subprocess, sys, threading, time, zipfile\n'
    'from pathlib import Path\n'
    'from kaggle_secrets import UserSecretsClient\n'
    '\n'
    'MAIN_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"\n'
    'REPO = Path("/kaggle/working/counterfactual-projection")\n'
    'REFERENCE_URL = "https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git"\n'
    'REFERENCE = Path("/kaggle/working/One-Shot-TAS-CCIL")\n'
    'GROMO_URL = "https://github.com/growingnet/gromo.git"\n'
    'GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"\n'
    'GROMO = Path("/kaggle/working/gromo")\n'
    'RUN_ROOT = Path("/kaggle/working/vna_phase0_seed0")\n'
    'OUTPUT = RUN_ROOT\n'
    '\n'
    'def private_clone(url, destination, branch):\n'
    '    token = UserSecretsClient().get_secret("github_token").strip()\n'
    '    if not token: raise RuntimeError("Kaggle Secret github_token is unavailable")\n'
    '    askpass = Path("/kaggle/working/.counterfactual_git_askpass.py")\n'
    '    askpass.write_text("#!/usr/bin/env python3\\nimport os,sys\\np=sys.argv[1] if len(sys.argv)>1 else \'\'\\nprint(\'x-access-token\' if \'Username\' in p else os.environ[\'GITHUB_TOKEN_RUNTIME\'])\\n")\n'
    '    askpass.chmod(0o700)\n'
    '    env = os.environ.copy()\n'
    '    env.update(GITHUB_TOKEN_RUNTIME=token, GIT_ASKPASS=str(askpass),\n'
    '               GIT_TERMINAL_PROMPT="0")\n'
    '    try:\n'
    '        subprocess.run(["git", "clone", "--branch", branch, "--single-branch",\n'
    '                        url, str(destination)], env=env, check=True)\n'
    '    finally:\n'
    '        askpass.unlink(missing_ok=True)\n'
    '\n'
    'for checkout in (REPO, REFERENCE, GROMO):\n'
    '    if checkout.exists(): shutil.rmtree(checkout)\n'
    'OUTPUT.mkdir(parents=True, exist_ok=True)\n'
    'private_clone(MAIN_URL, REPO, "main")\n'
    'private_clone(REFERENCE_URL, REFERENCE, "ccil-residual-capacity")\n'
    'subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",\n'
    '                GROMO_URL, str(GROMO)], check=True)\n'
    'subprocess.run(["git", "-C", str(GROMO), "fetch", "--depth", "1",\n'
    '                "origin", GROMO_COMMIT], check=True)\n'
    'subprocess.run(["git", "-C", str(GROMO), "checkout", "--detach",\n'
    '                GROMO_COMMIT], check=True)\n'
    'print("main:", subprocess.check_output(\n'
    '    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())\n'
)

_SETUP = (
    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)\n'
    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(GROMO)], check=True)\n'
    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-upstream.txt")], check=True)\n'
    'subprocess.run(["bash", "scripts/fetch_third_party.sh"], cwd=REPO, check=True)\n'
    'GROMO_SRC = GROMO / "src"\n'
    'RUNTIME_PYTHONPATH = os.pathsep.join(filter(None, (\n'
    '    str(REPO), str(GROMO_SRC), str(REFERENCE), os.environ.get("PYTHONPATH", ""))))\n'
    'test_env = os.environ.copy(); test_env.update(PYTHONPATH=RUNTIME_PYTHONPATH, REQUIRE_UPSTREAM="1", PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")\n'
    'subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO,\n'
    '               env=test_env, check=True)\n'
    'import torch\n'
    'if torch.cuda.device_count() != 2:\n'
    '    raise RuntimeError(f"Select T4 x2; found {torch.cuda.device_count()} GPU(s)")\n'
    'required_cifar_files = {"train", "test", "meta"}\n'
    'cifar_dirs = []\n'
    'for directory, _, filenames in os.walk("/kaggle/input", followlinks=True):\n'
    '    path = Path(directory)\n'
    '    if (path.name == "cifar-100-python" and\n'
    '            required_cifar_files.issubset(filenames)):\n'
    '        cifar_dirs.append(path.parent)\n'
    'cifar_dirs = sorted(set(cifar_dirs), key=str)\n'
    'if not cifar_dirs:\n'
    '    visible = []\n'
    '    for directory, dirnames, filenames in os.walk(\n'
    '            "/kaggle/input", followlinks=True):\n'
    '        visible.append({\n'
    '            "directory": directory,\n'
    '            "directories": sorted(dirnames)[:10],\n'
    '            "files": sorted(filenames)[:10],\n'
    '        })\n'
    '        if len(visible) == 20:\n'
    '            break\n'
    '    raise FileNotFoundError(\n'
    '        "Attach CIFAR-100 containing cifar-100-python/{train,test,meta}. "\n'
    '        f"Visible input inventory: {visible}")\n'
    'DATA_ROOT = cifar_dirs[0]\n'
    'print("CIFAR-100 root:", DATA_ROOT)\n'
)

def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(commit=None):
    if commit is None:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be a full immutable commit hash")
    bootstrap = _BOOTSTRAP
    # Reuse the established token/askpass mechanism for both clone and fetch.
    original_clone = '\n'.join([
        '        subprocess.run(["git", "clone", "--branch", branch, "--single-branch",',
        '                        url, str(destination)], env=env, check=True)'])
    pinned_clone = '\n'.join([
        '        if branch == REPO_COMMIT:',
        '            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",',
        '                            url, str(destination)], env=env, check=True)',
        '            subprocess.run(["git", "-C", str(destination), "fetch", "origin", REPO_COMMIT], env=env, check=True)',
        '            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", REPO_COMMIT], check=True)',
        '        else:',
        '            subprocess.run(["git", "clone", "--branch", branch, "--single-branch",',
        '                            url, str(destination)], env=env, check=True)'])
    if original_clone not in bootstrap:
        raise RuntimeError("Existing clone setup changed; inspect before regenerating")
    bootstrap = bootstrap.replace(original_clone, pinned_clone).replace(
        'private_clone(MAIN_URL, REPO, "main")', 'private_clone(MAIN_URL, REPO, REPO_COMMIT)')
    bootstrap += '\nif subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() != REPO_COMMIT:\n    raise RuntimeError("Pinned repository revision mismatch")\n'
    bootstrap += '\n' + '\n'.join([
        'from datetime import datetime, timezone',
        'CODE_COMMIT = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()',
        'print("Resolved code commit:", CODE_COMMIT, "pin:", REPO_COMMIT)',
        'commit_path = OUTPUT / "code_commit.json"',
        'sessions = json.loads(commit_path.read_text()) if commit_path.exists() else []',
        'sessions.append({"timestamp": datetime.now(timezone.utc).isoformat(), "pin": REPO_COMMIT, "commit": CODE_COMMIT})',
        'temporary = commit_path.with_suffix(".json.tmp")',
        'temporary.write_text(json.dumps(sessions, indent=2) + "\\n")',
        'temporary.replace(commit_path)',
    ]) + '\n'
    setup = _SETUP
    setup = setup.replace('"-e", str(REPO)', '"-e", str(REPO) + "[test,vision,diagnostics]"')
    setup = setup.replace('"pytest", "-q"', '"pytest", "-q", "tests/test_vna.py", "tests/test_vna_stage_b_upstream.py", "tests/test_vna_stage_b_losses.py"')
    setup = setup.replace('env=test_env, check=True)', 'env={**test_env, "CUDA_VISIBLE_DEVICES": ""}, check=True)')
    setup += '''
if not (REPO / "experiments/vna_kaggle.py").exists():
    raise RuntimeError("Required module experiments/vna_kaggle.py is missing from the pinned commit")
for module in ("experiments/vna_provenance.py", "diagnostics/signal_curve.py", "methods/vna.py"):
    if not (REPO / module).exists(): raise RuntimeError(f"Required module {module} is missing from the pinned commit")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "matplotlib"], check=True)
sys.path[:0] = [str(REPO), str(GROMO_SRC), str(REFERENCE)]
print("Official TEST verified:", DATA_ROOT / "cifar-100-python/test")
'''
    cells = [cell('markdown', '''# VNA Phase 0 — seed 0, pinned repository, Kaggle T4×2
Enable **Internet**, **T4×2**, and secret **github_token**. Attach CIFAR-100 with
`cifar-100-python/{train,test,meta}`. Optionally attach previous notebook outputs
(directory, tar.gz or zip). Run all cells; compatible Stage 1C checkpoint diagnostics are reused.
Diagnostics run every 20 epochs; training retains checkpoints every 10.
Old diagnostic completion markers are rejected; use a fresh signal directory.
Training on GPU0 overlaps the checkpoint watcher on GPU1; after training both
GPUs process remaining checkpoints. Outputs and hourly atomic archives survive
normal interruptions. A hard session kill can lose only the current epoch and
unfinished checkpoint diagnostic; download/attach the saved output directory too.
Code is pinned to an immutable commit; every session records its resolved commit in code_commit.json.
Regenerate with `python scripts/build_kaggle_vna_signal_curve_notebook.py --commit FULL_HASH`
to change the pin; the default is current HEAD. The pin must include Phase 0.
'''), cell('code', f'''REPO_COMMIT = "{commit}"  # immutable, builder --commit; default current HEAD
GPU_DATA = True  # supervisor: Phase 0 has one SGD trajectory; prioritize T4 speed
CHANNELS_LAST = True
CUDNN_BENCHMARK = True
AMP = False  # stays off: no measured >=1.5x CUDA speedup
DIAGNOSTIC_STRIDE = 20
CACHE_BLOCK_INPUT = True  # exact native eval-BN suffix, covered by parity tests
'''), cell('code', bootstrap), cell('code', setup), cell('code', '''from experiments.vna_kaggle import restore_outputs, archive_outputs
from experiments.train_recipe_checkpoints import parser as recipe_parser

RECIPE = RUN_ROOT / "recipe"
SIGNAL = RUN_ROOT / "signal"
training_flags = []
for enabled, flag in ((GPU_DATA, "--gpu-data"), (CHANNELS_LAST, "--channels-last"),
                      (CUDNN_BENCHMARK, "--cudnn-benchmark"), (AMP, "--amp")):
    if enabled: training_flags.append(flag)
protocol = {"seed": 0, "fake_data": False, "batch_size": 64, "workers": 2,
            "limit_batches": 0, "gpu_data": GPU_DATA, "amp": AMP,
            "channels_last": CHANNELS_LAST, "cudnn_benchmark": CUDNN_BENCHMARK,
            "epochs": 200, "architecture": "resnet18"}
restore_outputs(Path("/kaggle/input"), RUN_ROOT, protocol)
# Remove stale completion sentinel when extending an incomplete restored run.
(RECIPE / "training_done.json").unlink(missing_ok=True)
stop_file = RUN_ROOT / "stop_watch"
stop_file.unlink(missing_ok=True)
common = ["--data-root", str(DATA_ROOT), "--seed", "0"]
training = [sys.executable, "-u", "-m", "experiments.train_recipe_checkpoints",
            *common, "--device", "cuda:0", "--output", str(RECIPE), "--workers", "2", *training_flags]
if (RECIPE / "latest.pt").exists():
    training += ["--resume", str(RECIPE / "latest.pt")]
diagnostic = [sys.executable, "-u", "-m", "diagnostics.signal_curve", *common,
              "--checkpoint-dir", str(RECIPE), "--output", str(SIGNAL),
              "--diagnostic-stride", str(DIAGNOSTIC_STRIDE)]
for enabled, flag in ((GPU_DATA, "--gpu-data"), (CHANNELS_LAST, "--channels-last"),
                      (CUDNN_BENCHMARK, "--cudnn-benchmark")):
    if enabled: diagnostic.append(flag)
if CACHE_BLOCK_INPUT: diagnostic += ["--cache-block-input"]
print("Training flags:", training_flags, "AMP decision: off unless independently measured >=1.5x")
'''), cell('code', '''# Concurrent subprocesses share only atomic checkpoints/completion files.
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
processes, logs = [], []
archive_errors = []
archive_stop = threading.Event()
def periodic_archive():
    while not archive_stop.wait(3600):
        try: archive_outputs(RUN_ROOT)
        except Exception as error:
            archive_errors.append(str(error)); print("Archive error:", error, flush=True)
def launch(command, name):
    log = (RUN_ROOT / f"{name}.log").open("a", buffering=1)
    logs.append(log)
    process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=log, stderr=subprocess.STDOUT)
    processes.append(process)
    return process
archiver = threading.Thread(target=periodic_archive, daemon=True)
archiver.start()
try:
    watcher = launch(diagnostic + ["--device", "cuda:1", "--watch", "--stop-file", str(stop_file), "--no-merge"], "watch_gpu1")
    trainer = launch(training, "training_gpu0")
    while trainer.poll() is None:
        time.sleep(30)
        if watcher.poll() not in (None, 0):
            raise RuntimeError("Diagnostic watcher failed; see watch_gpu1.log")
        history = RECIPE / "history.json"
        if history.exists():
            print("Training:", json.loads(history.read_text())[-1], flush=True)
    if trainer.returncode:
        raise RuntimeError("Training failed; see training_gpu0.log")
    stop_file.touch()
    # Allow the current atomic checkpoint transaction to finish before splitting.
    while watcher.poll() is None:
        time.sleep(30)
        print("Finishing current watcher checkpoint...", flush=True)
    if watcher.returncode: raise RuntimeError("Watcher failed")
    workers = [launch(diagnostic + ["--device", f"cuda:{gpu}", "--shards", "2", "--shard", str(gpu), "--no-merge"], f"drain_gpu{gpu}")
               for gpu in (0, 1)]
    while any(p.poll() is None for p in workers):
        time.sleep(30)
        print("Completed checkpoints:", len(list((SIGNAL / "completed").glob("*.json"))), flush=True)
    if any(p.returncode for p in workers): raise RuntimeError("Diagnostic drain failed; see logs")
    subprocess.run(diagnostic + ["--merge-only", "--device", "cpu"], cwd=REPO, env=ENV, check=True)
finally:
    archive_stop.set()
    for process in processes:
        if process.poll() is None: process.terminate()
    for process in processes:
        if process.poll() is None: process.wait()
    archiver.join()
    for log in logs: log.close()
    print("Resumable archive:", archive_outputs(RUN_ROOT))
if archive_errors: print("Earlier archive failures:", archive_errors)
'''), cell('code', '''import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
frame = pd.read_csv(SIGNAL / "signal_curve.csv")
summary = json.loads((SIGNAL / "summary.json").read_text())
frame["V_TINY/V_random"] = frame["V"] / frame["random_V"]
columns = ["epoch", "site", "row_kind", "requested_rank", "effective_rank", "U", "V", "eta", "rho",
           "lite_U", "lite_V", "lite_eta", "lite_rho", "V_TINY/V_random", "vna_derivative",
           "lite_derivative", "random_derivative", "random_lite_derivative", "lr_boost_derivative",
           "lite_lr_boost_derivative", "realized_gain", "random_realized_gain", "lite_realized_gain",
           "random_lite_realized_gain", "skipped", "random_skipped", "lite_skipped", "random_lite_skipped",
           "cold_sgd_ratio", "warm_sgd_ratio", "lite_sgd_ratio", "lambda_full", "lambda_split_half",
           "lambda_shuffled_mean", "lambda_shuffled_max", "significant_vs_shuffled", "significant_split_half"]
display(frame.reindex(columns=columns))
display(pd.DataFrame(summary["significance_by_checkpoint_site"]))
rank_table = pd.DataFrame(summary["per_rank"])
display(rank_table)
metrics = ["U", "V", "rho", "lite_V", "lite_rho", "vna_derivative", "lite_derivative", "V_TINY/V_random"]
fig, axes = plt.subplots(4, 2, figsize=(16, 18))
for axis, metric in zip(axes.flat, metrics):
    for (site, requested), values in frame[frame["row_kind"] == "sweep"].groupby(["site", "requested_rank"]):
        axis.plot(values["epoch"], values[metric], marker=".", label=f"{site} r={requested}")
    axis.set(title=metric, xlabel="epoch"); axis.grid(alpha=.3)
axes.flat[0].legend(fontsize=7)
plt.tight_layout(); fig.savefig(SIGNAL / "signal_curve.png"); plt.show()
print("Per-rank Phase-0 decision:", summary["phase0_acceptance"])
print("Decay rule:", summary["decay_rule"])
print("Excluded counts appear separately for each full/lite criterion in the rank table.")
print("rho > 0.3 frequency:", summary["rho_above_0_3_fraction"])
print("Skipped non-positive realized gains:", summary["skipped_nonpositive_gain"])
print("Final archive:", archive_outputs(RUN_ROOT))
''')]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "dataSources": []}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', default=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    parser.add_argument('--output', default=str(ROOT/'notebooks/kaggle_vna_signal_curve.ipynb'))
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(build(args.commit), indent=1, ensure_ascii=False) + '\n')
    print(args.output)
