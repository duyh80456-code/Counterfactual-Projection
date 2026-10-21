"""Shared pinned setup and resumable runtime for v4 and B5 notebooks."""
import re


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


BOOTSTRAP = r'''import hashlib, json, os, shutil, subprocess, sys, threading, time, zipfile
os.environ.update(TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
from pathlib import Path
from kaggle_secrets import UserSecretsClient

MAIN_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
REPO = Path("/kaggle/working/counterfactual-projection")
RUN_ROOT = Path("/kaggle/working/vna4_main")
OUTPUT = RUN_ROOT

def private_clone(url, destination, branch):
    token = UserSecretsClient().get_secret("github_token").strip()
    if not token: raise RuntimeError("Kaggle Secret github_token is unavailable")
    askpass = Path("/kaggle/working/.counterfactual_git_askpass.py")
    askpass.write_text("#!/usr/bin/env python3\nimport os,sys\np=sys.argv[1] if len(sys.argv)>1 else ''\nprint('x-access-token' if 'Username' in p else os.environ['GITHUB_TOKEN_RUNTIME'])\n")
    askpass.chmod(0o700)
    env = os.environ.copy()
    env.update(GITHUB_TOKEN_RUNTIME=token, GIT_ASKPASS=str(askpass),
               GIT_TERMINAL_PROMPT="0")
    try:
        if len(branch) == 40 and all(c in "0123456789abcdef" for c in branch):
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                            url, str(destination)], env=env, check=True)
            subprocess.run(["git", "-C", str(destination), "fetch", "origin", branch], env=env, check=True)
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", branch], env=env, check=True)
        else:
            subprocess.run(["git", "clone", "--branch", branch, "--single-branch",
                            url, str(destination)], env=env, check=True)
    finally:
        askpass.unlink(missing_ok=True)

for checkout in (REPO,):
    if checkout.exists(): shutil.rmtree(checkout)
OUTPUT.mkdir(parents=True, exist_ok=True)
private_clone(MAIN_URL, REPO, REPO_COMMIT)
print("main:", subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

if subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip() != REPO_COMMIT:
    raise RuntimeError("Pinned repository revision mismatch")

from datetime import datetime, timezone
CODE_COMMIT = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
print("Resolved code commit:", CODE_COMMIT, "pin:", REPO_COMMIT)
commit_path = OUTPUT / "code_commit.json"
sessions = json.loads(commit_path.read_text()) if commit_path.exists() else []
sessions.append({"timestamp": datetime.now(timezone.utc).isoformat(), "pin": REPO_COMMIT, "commit": CODE_COMMIT})
temporary = commit_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(sessions, indent=2) + "\n")
temporary.replace(commit_path)
'''

SETUP = r'''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO) + "[test,vision,diagnostics]"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-upstream.txt")], check=True)
subprocess.run(["bash", "scripts/fetch_third_party.sh"], cwd=REPO, check=True)
RUNTIME_PYTHONPATH = os.pathsep.join(filter(None, (
    str(REPO), os.environ.get("PYTHONPATH", ""))))
test_env = os.environ.copy(); test_env.update(PYTHONUNBUFFERED="1", TQDM_DISABLE="1", PYTHONPATH=RUNTIME_PYTHONPATH, REQUIRE_UPSTREAM="1", PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
import torch
if torch.cuda.device_count() != 2:
    raise RuntimeError(f"Select T4 x2; found {torch.cuda.device_count()} GPU(s)")
required_cifar_files = {"train", "test", "meta"}
cifar_dirs = []
for directory, _, filenames in os.walk("/kaggle/input", followlinks=True):
    path = Path(directory)
    if (path.name == "cifar-100-python" and
            required_cifar_files.issubset(filenames)):
        cifar_dirs.append(path.parent)
cifar_dirs = sorted(set(cifar_dirs), key=str)
if not cifar_dirs:
    visible = []
    for directory, dirnames, filenames in os.walk(
            "/kaggle/input", followlinks=True):
        visible.append({
            "directory": directory,
            "directories": sorted(dirnames)[:10],
            "files": sorted(filenames)[:10],
        })
        if len(visible) == 20:
            break
    raise FileNotFoundError(
        "Attach CIFAR-100 containing cifar-100-python/{train,test,meta}. "
        f"Visible input inventory: {visible}")
DATA_ROOT = cifar_dirs[0]
print("CIFAR-100 root:", DATA_ROOT)

if not (REPO / "experiments/vna_kaggle.py").exists():
    raise RuntimeError("Required module experiments/vna_kaggle.py is missing from the pinned commit")
for module in ("experiments/vna_provenance.py", "methods/vna.py"):
    if not (REPO / module).exists(): raise RuntimeError(f"Required module {module} is missing from the pinned commit")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "matplotlib"], check=True)
sys.path[:0] = [str(REPO)]
print("Official TEST verified:", DATA_ROOT / "cifar-100-python/test")

'''

RUNTIME = r'''from concurrent.futures import ThreadPoolExecutor
from experiments.vna_kaggle import restore_stage_archives
from experiments.vna_kaggle import archive_outputs as _archive_outputs
STAGE_B = RUN_ROOT / "outputs"
STAGE_B.mkdir(parents=True, exist_ok=True)
restore_stage_archives(Path('/kaggle/input'), STAGE_B, 'outputs')
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
       "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python", "VNA_DEADLINE": str(VNA_DEADLINE)}
session_paused = threading.Event()
lane_state = {gpu: {"job": "idle", "last": ""} for gpu in range(2)}
def pause_due():
    if time.time() >= VNA_DEADLINE - 600: session_paused.set()
    return session_paused.is_set()
def launch(cmd, gpu, name, log_path):
    from experiments.vna_resume import launch_logged
    return launch_logged(cmd, cwd=REPO, env=gpu_env(gpu), log_path=log_path,
                         lane=gpu, job=name, lanes=lane_state,
                         paused=session_paused, deadline=VNA_DEADLINE)
def heartbeat():
    from experiments.vna_resume import print_heartbeat
    while not archive_stop.wait(600):
        print_heartbeat(RUN_ROOT / "outputs", SESSION_START, VNA_DEADLINE, lane_state)
archive_lock = threading.Lock(); archive_stop = threading.Event(); archive_errors = []
def archive_outputs():
    with archive_lock:
        return _archive_outputs(RUN_ROOT, budget=DISK_CAP_BYTES,
                                folders=("outputs",), exclude_parts=("data",))
def periodic_archive():
    while not archive_stop.wait(3600):
        try: archive_outputs()
        except Exception as error:
            archive_errors.append(str(error)); print("Archive error:", error, flush=True)
def gpu_env(gpu): return {**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}
archive_errors = []
lane_timings = {}
archiver = threading.Thread(target=periodic_archive, daemon=True); archiver.start()
heartbeat_thread = threading.Thread(target=heartbeat, daemon=True); heartbeat_thread.start()
'''


def notebook(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be a full immutable commit hash")
    return dict(cells=[cell('markdown', ''), cell('code', ''), cell('code', BOOTSTRAP),
                       cell('code', SETUP), cell('code', RUNTIME), cell('code', '')],
                metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}, 'kaggle': {'accelerator': 'gpu', 'dataSources': []}}, nbformat=4, nbformat_minor=5)
