"""Build pinned B5-only Notebook 1 (no Stage A or B6)."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.build_kaggle_vna_signal_curve_notebook import build as base_build, cell


def build(commit):
    base = base_build(commit)
    constants = '''from pathlib import Path
REPO_COMMIT = "__PIN__"
GPU_HOURS_BUDGET = 8
ALLOW_OVER_BUDGET = False
IMAGENET32_DIR = None  # directory containing per-class <class_id>.npy
MNIST_DIR = None       # directory containing mnist_ or torchvision MNIST/raw
MU_REL_DEFAULT = 1e-2 # set from Stage A summary.json if available
DISK_CAP_BYTES = 5_000_000_000
'''.replace('__PIN__', commit)
    bootstrap = ''.join(base['cells'][2]['source']).replace('/kaggle/working/vna_phase0_seed0', '/kaggle/working/vna_stage_b5')
    setup = ''.join(base['cells'][3]['source']).split('required_cifar_files =')[0]
    setup = setup.replace('"-q", "tests/test_vna.py", "tests/test_vna_stage_b_upstream.py", "tests/test_vna_stage_b_losses.py"', '"-q"')
    setup = setup.replace('print("Official TEST verified:", DATA_ROOT / "cifar-100-python/test")\n', '')
    setup += '''\n# Prefer attached input data. ImageNet absence is an explicit skip.
if IMAGENET32_DIR is None:
    candidates = sorted({p.parent for p in Path("/kaggle/input").rglob("0.npy") if (p.parent / "1.npy").exists()})
    IMAGENET32_DIR = candidates[0] if candidates else None
if MNIST_DIR is None:
    pickles = sorted(Path("/kaggle/input").rglob("mnist_"))
    raw = sorted(Path("/kaggle/input").rglob("train-images-idx3-ubyte"))
    MNIST_DIR = pickles[0].parent if pickles else (raw[0].parents[2] if raw else None)
print("MNIST:", MNIST_DIR, "ImageNet-32:", IMAGENET32_DIR)
'''
    cells = [cell('markdown', '''# VNA Stage B5 — measure only, T4×2
Set the immutable `--commit` pin after publishing this implementation. This
notebook includes neither Stage A nor B6. Install pinned upstream dependencies,
run the full required suite, and smoke two tasks of each registered arm before
budget approval. GPU0 runs ImageNet BP then CBP; GPU1 runs regression BP then
all five registered CBP replacement rates, then MNIST BP and CBP. A synthetic
smoke is explicitly flagged invalid; it never establishes the baseline gate.
Hourly archives retain task-boundary resume state and RNG. Attach the prior
archive to resume. MNIST uses the original upstream pickle builder; attached
raw data is preferred, otherwise its loader downloads MNIST. Missing ImageNet
records `skipped: data unavailable` and continues. No resets or VNA corrections
are applied to the training models.
'''), cell('code', constants), cell('code', bootstrap), cell('code', setup),
    cell('code', '''from experiments.vna_kaggle import archive_outputs as _archive_outputs
from experiments.stage_b5 import restore_archives
from experiments.stage_b5 import RATES, finalize
import threading
STAGE_B = RUN_ROOT / "outputs" / "stageB"
STAGE_B.mkdir(parents=True, exist_ok=True)
restore_archives(Path("/kaggle/input"), STAGE_B)
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
archive_lock = threading.Lock()
def archive_outputs():
    with archive_lock:
        return _archive_outputs(RUN_ROOT, budget=DISK_CAP_BYTES, folders=("outputs/stageB",), exclude_parts=("data",))
archive_stop = threading.Event()
archive_errors = []
def periodic_archive():
    while not archive_stop.wait(3600):
        try: archive_outputs()
        except Exception as error:
            archive_errors.append(str(error)); print("Archive error:", error, flush=True)
archiver = threading.Thread(target=periodic_archive, daemon=True); archiver.start()
def command(benchmark, arm, gpu, out, *, smoke=False, rate=None):
    cmd = [sys.executable, "-u", "-m", "experiments.stage_b5", benchmark, "--arm", arm,
           "--device", f"cuda:{gpu}", "--out", str(out), "--mu-rel-default", str(MU_REL_DEFAULT)]
    if smoke: cmd += ["--smoke"]
    else: cmd += ["--resume"]
    if rate is not None: cmd += ["--replacement-rate", str(rate)]
    data = IMAGENET32_DIR if benchmark == "imagenet" else MNIST_DIR if benchmark == "mnist" else None
    if data: cmd += ["--data-dir", str(data)]
    return cmd
CONFIGS = [("regression", "bp", 1, None)] + [("regression", "cbp", 1, r) for r in RATES]
CONFIGS += [("mnist", "bp", 1, None), ("mnist", "cbp", 1, None)]
if IMAGENET32_DIR is not None:
    CONFIGS += [("imagenet", "bp", 0, None), ("imagenet", "cbp", 0, None)]
else:
    subprocess.run(command("imagenet", "bp", 0, STAGE_B), cwd=REPO, env=ENV, check=True)
'''), cell('code', '''# Smoke native architectures and real data; measure the full diagnostics cost.
# MNIST's tiny rank-log fixture is used only in smoke and CPU tests.
SMOKE = RUN_ROOT / "smoke" / "outputs" / "stageB"
try:
    smoke_seconds = 0.
    projected = 0.
    for benchmark, arm, gpu, rate in CONFIGS:
        start = time.monotonic()
        subprocess.run(command(benchmark, arm, gpu, SMOKE, smoke=True, rate=rate), cwd=REPO, env=ENV, check=True)
        elapsed = time.monotonic()-start; smoke_seconds += elapsed
        arm_name = arm if rate is None else f"cbp_rr_{rate:g}"
        summary = json.loads((SMOKE / "b5" / benchmark / arm_name / "summary.json").read_text())
        full_steps = {"regression": 1000000, "mnist": 12000000, "imagenet": 600000}[benchmark]
        # Training and diagnostics have different scaling: project training
        # per step, then the registered 16/18 measurement points separately.
        # The maximum smoke point includes a full sliding window (task 2 mid).
        train_seconds = summary["training_seconds"] * full_steps / summary["steps_run"]
        point_seconds = max((p["seconds"] for p in summary["measurement_point_seconds"]), default=0.)
        point_count = 16 if benchmark == "regression" else 18
        hours = (train_seconds + point_count*point_seconds) / 3600
        projected += hours
        print(benchmark, arm_name, f"{hours:.3f} projected GPU-hours", flush=True)
    total = projected + smoke_seconds/3600
    print(f"Total including smoke: {total:.3f} GPU-hours; budget: {GPU_HOURS_BUDGET}")
    (STAGE_B / "cost_projection.json").write_text(json.dumps({"projected_gpu_hours": projected,
        "smoke_gpu_hours": smoke_seconds/3600, "total_gpu_hours": total}, indent=2))
    archive_outputs()
    if total > GPU_HOURS_BUDGET and not ALLOW_OVER_BUDGET:
        raise RuntimeError("Projected B5 cost exceeds GPU_HOURS_BUDGET; review before setting ALLOW_OVER_BUDGET")
except Exception:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs(); raise
'''), cell('code', '''from concurrent.futures import ThreadPoolExecutor
def worker(gpu):
    for benchmark, arm, configured_gpu, rate in CONFIGS:
        if configured_gpu != gpu: continue
        arm_name = arm if rate is None else f"cbp_rr_{rate:g}"
        with (RUN_ROOT / f"{benchmark}_{arm_name}_gpu{gpu}.log").open("a", buffering=1) as log:
            subprocess.run(command(benchmark, arm, gpu, STAGE_B, rate=rate), cwd=REPO, env=ENV,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Finished", benchmark, arm_name, "GPU", gpu, flush=True)
        archive_outputs()
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in (0, 1)]
        for future in futures: future.result()
finally:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs()
if archive_errors: raise RuntimeError(archive_errors)
'''), cell('code', '''import pandas as pd
import matplotlib.pyplot as plt
for benchmark in ("regression", "mnist", "imagenet"):
    if (STAGE_B / "b5" / benchmark).exists(): finalize(STAGE_B, benchmark, mu_rel_default=MU_REL_DEFAULT)
print("BASELINE GATE")
print((STAGE_B / "b5_baseline_gate.json").read_text())
gates = json.loads((STAGE_B / "gate.json").read_text())
for benchmark, gate in gates.items():
    if not gate.get("measurements_valid", False):
        print("*** INVALID MEASUREMENTS:", benchmark, gate.get("baseline", gate), "***")
display(pd.DataFrame(gates).T)
for path in (STAGE_B / "b5").glob("*/bp/measurements.csv"):
    rows = pd.read_csv(path)
    benchmark = path.parents[1].name
    for (rank, mu), group in rows.groupby(["rank", "mu_rel"]):
        group.groupby("task").rho_joint.mean().plot(marker="o", label=f"r={rank}, mu={mu}")
    plt.xlabel("task"); plt.ylabel("rho_joint"); plt.title(benchmark); plt.legend()
    plt.savefig(path.parent / "rho_vs_task.png", bbox_inches="tight"); plt.show()
    display(rows[["task", "step", "site", "rank"] + [c for c in rows if c.endswith("_m") or c.endswith("_ratio")]])
archive_outputs()
''')]
    return {'cells': cells, 'metadata': base['metadata'], 'nbformat': 4, 'nbformat_minor': 5}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    p.add_argument('--output', default=str(ROOT/'notebooks/kaggle_vna_stage_b_nb1.ipynb'))
    args = p.parse_args(); path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(args.commit), indent=1)+'\n')


if __name__ == '__main__': main()
