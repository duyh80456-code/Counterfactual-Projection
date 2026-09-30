"""Build three from-scratch, protocol-identical Kaggle T4x2 notebooks."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def text(cell):
    return "".join(cell["source"])


phase2 = json.loads(Path(
    "notebooks/kaggle_plateau_fork_t4x2.ipynb").read_text())


def build(seed):
    run_root = f"/kaggle/working/unified_seed{seed}_end_to_end_v1"
    bootstrap = text(phase2["cells"][1]).replace(
        'OUTPUT = Path("/kaggle/working/plateau_fork_three_jobs_100ep_v3")',
        f'RUN_ROOT = Path("{run_root}")\nOUTPUT = RUN_ROOT / "phase2"')
    phase2_commands = text(phase2["cells"][4]).replace(
        '"--seed", "1"', f'"--seed", "{seed}"')
    phase2_summary = text(phase2["cells"][5])

    cells = [
        markdown(f"""# Unified from-scratch stall experiment — seed {seed}

This notebook is one complete, self-contained replication. It needs only
CIFAR-100, Kaggle **T4 x2**, and the `github_token` secret. It does not load a
theta150/theta300 checkpoint and never rebases or restarts the learning-rate
schedule.

Phase 1 trains a randomly initialized CIFAR-ResNet18 with its metric-independent
base recipe: SGD for 200 recipe epochs, LR 0.1 with MultiStep drops at epochs
100 and 150 (`gamma=0.1`), momentum 0.9, and weight decay 5e-4. Every exact
trigger best is fully checkpointed. Only after the base recipe completes is the
method-independent stall detector armed. Another 100 consecutive epochs
without a significant +0.1 pp trigger gain confirm stall; those 100 observed
epochs are the Vanilla control.

Phase 2 forks that run's exact theta_best, including optimizer momentum,
scheduler position, RNG, loader stream, and data indices. GPU0 runs recurrent
E-driven O. GPU1 runs scaled Bypass 70/30 and then launches a fresh recurrent
O-only process. Each method consumes 100 SGD epochs and writes per-epoch history,
latest/best checkpoints, diagnostics, timing, memory, and final/best accuracy.
Only `seed={seed}` differs from the other two generated notebooks.
"""),
        code(bootstrap),
        phase2["cells"][2],
        code(f"""SEED = {seed}
PHASE1_OUTPUT = RUN_ROOT / "vanilla_stall"
PHASE1_OUTPUT.mkdir(parents=True, exist_ok=True)

phase1_command = [
    sys.executable, "-m", "experiments.run_unified_vanilla_to_stall",
    "--reference-root", str(REFERENCE), "--data-root", str(DATA_ROOT),
    "--output", str(PHASE1_OUTPUT), "--seed", str(SEED),
    "--max-epoch", "800", "--batch-size", "64",
    "--validation-samples", "5000", "--trigger-samples", "2000",
    "--tuning-samples", "128", "--lr", "0.1", "--recipe-epochs", "200",
    "--lr-milestones", "100,150", "--lr-gamma", "0.1",
    "--weight-decay", "0.0005",
    "--stall-patience", "100", "--best-min-gain", "0.0",
    "--significant-min-gain", "0.001"]

phase1_log = (PHASE1_OUTPUT / "notebook_stream.log").open("a", buffering=1)
phase1_env = os.environ.copy()
phase1_env.update(CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
                  PYTHONPATH=RUNTIME_PYTHONPATH)
phase1_process = subprocess.Popen(
    phase1_command, cwd=REPO, env=phase1_env, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in phase1_process.stdout:
    phase1_log.write(line); phase1_log.flush()
    print(f"[seed{{SEED}} vanilla] {{line}}", end="", flush=True)
phase1_code = phase1_process.wait()
phase1_log.close()
if phase1_code:
    raise RuntimeError(f"unified Vanilla exited with code {{phase1_code}}")
"""),
        code("""phase1_result = json.loads(
    (PHASE1_OUTPUT / "result.json").read_text())
if not phase1_result["plateau_found"]:
    archive = shutil.make_archive(
        str(RUN_ROOT), "gztar", root_dir=RUN_ROOT)
    raise RuntimeError(
        "No confirmed 100-epoch stall by max_epoch. All resumable Phase-1 "
        f"checkpoints were retained in {archive}")

PLATEAU_CHECKPOINT = Path(phase1_result["plateau_checkpoint"])
PLATEAU_PAYLOAD = torch.load(
    PLATEAU_CHECKPOINT, map_location="cpu", weights_only=False)
if PLATEAU_PAYLOAD.get("kind") != "plateau_fork_checkpoint":
    raise RuntimeError("Phase 1 did not produce a plateau fork checkpoint")
PLATEAU_HASH = hashlib.sha256(
    PLATEAU_CHECKPOINT.read_bytes()).hexdigest()
PLATEAU_EPOCH = int(PLATEAU_PAYLOAD["epoch"])
VANILLA_CONTROL = dict(PLATEAU_PAYLOAD["vanilla_control"])
if VANILLA_CONTROL["post_fork_epochs"] != 100:
    raise RuntimeError("Vanilla control is not exactly 100 observed epochs")
if int(PLATEAU_PAYLOAD["protocol"]["seed"]) != SEED:
    raise RuntimeError("theta_best seed mismatch")
print("theta_best:", PLATEAU_EPOCH, PLATEAU_HASH)
print(json.dumps(VANILLA_CONTROL, indent=2, sort_keys=True))
"""),
        markdown("""## Phase 2 — identical four-arm comparison

The Vanilla control is the already observed stall trajectory. Three new jobs
start from the byte-identical theta_best checkpoint. GPU0 is dedicated to the
heavier E-driven O job; GPU1 runs Bypass and then a fresh O-only process.
"""),
        code("""import threading

OUTPUT = RUN_ROOT / "phase2"
OUTPUT.mkdir(parents=True, exist_ok=True)
"""),
        code(phase2_commands),
        code(phase2_summary),
        code("""full_archive = shutil.make_archive(
    str(RUN_ROOT), "gztar", root_dir=RUN_ROOT)
print("Complete seed archive:", full_archive)
print("Phase-1 latest:", PHASE1_OUTPUT / "checkpoint_latest.pt")
print("Phase-1 best:", PHASE1_OUTPUT / "checkpoint_best.pt")
print("Fork checkpoint:", PLATEAU_CHECKPOINT)
"""),
    ]
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "gpu", "dataSources": []}},
        "nbformat": 4, "nbformat_minor": 5}


for seed in (1, 2, 3):
    destination = Path(
        f"notebooks/kaggle_unified_seed{seed}_end_to_end_t4x2.ipynb")
    destination.write_text(
        json.dumps(build(seed), indent=1, ensure_ascii=False) + "\n")
    print(destination)
