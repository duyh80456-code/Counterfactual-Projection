"""Build Notebook 1: MNIST GPU lanes plus CPU regression."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_kaggle_vna_signal_curve_notebook import build as base_build
from scripts.build_kaggle_vna_signal_curve_notebook import cell
from scripts.build_kaggle_vna_stage_b_rest import SESSION_RUNTIME


def build(commit):
    base = base_build(commit)
    constants = f'''from pathlib import Path
import time
SESSION_HOURS = 11.5
SESSION_START = time.time()
VNA_DEADLINE = SESSION_START + SESSION_HOURS * 3600
REPO_COMMIT = "{commit}"
GPU_HOURS_BUDGET = 8
ALLOW_OVER_BUDGET = False
MNIST_DIR = None
DISK_CAP_BYTES = 5_000_000_000
'''
    bootstrap = ''.join(base['cells'][2]['source']).replace('vna_phase0_seed0', 'vna_stage_b_nb1')
    setup = ''.join(base['cells'][3]['source'])
    setup = setup.replace('test_env.update(', 'test_env.update(TQDM_DISABLE="1", ')
    start = setup.index('required_cifar_files =')
    end = setup.index('print("CIFAR-100 root:", DATA_ROOT)') + len('print("CIFAR-100 root:", DATA_ROOT)')
    setup = setup[:start] + setup[end:]
    setup = setup.replace('print("Official TEST verified:", DATA_ROOT / "cifar-100-python/test")', '')
    import re
    setup = re.sub(r'"pytest", "-q"[^\n]*?\], cwd=REPO,', '"pytest", "-q"], cwd=REPO,', setup)
    return {'cells': [cell('markdown', '''# VNA Notebook 1 — B5 MNIST and regression
Enable Internet, T4×2 and `github_token`. All three lanes start immediately:
GPU0 runs measured MNIST BP; GPU1 runs MNIST CBP (rr 1e-4); CPU reruns all six measured regression arms with the round-2 schedule (38 points,
22 in the gate window). MNIST BP and CBP use 36 points (20 in the gate window). No Stage A outputs, Phase-0
checkpoints or CIFAR data are required. Stage A remains a standalone notebook.
BP measures both μ=1e-3 and μ=1e-2; Gate G uses μ=1e-2.
ImageNet-32 is dropped. MNIST uses upstream load_mnist.py with attached data as
fallback. The 11.5-hour deadline pauses with a final archive; attach its output to resume.
Task and mid-task checkpoints are archived every 15 minutes. Every loader uses two
workers; CPU torch threads are limited to two. Attach prior Stage-B archives to resume.
Regenerate with the published commit containing this implementation before Kaggle.
'''), cell('code', constants), cell('code', bootstrap), cell('code', setup),
        cell('code', RUNTIME), cell('code', RUN), cell('code', REPORT)],
        'metadata': base['metadata'], 'nbformat': 4, 'nbformat_minor': 5}


RUNTIME = '''from concurrent.futures import ThreadPoolExecutor
from experiments.vna_kaggle import archive_with_status as _archive_outputs
from experiments.stage_b5 import restore_archives as restore_b, RATES, finalize, merge_json
STAGE_B = RUN_ROOT / "outputs/stageB/round2"
STAGE_B.mkdir(parents=True, exist_ok=True)
restore_b(Path("/kaggle/input"), STAGE_B.parent)
merge_json(STAGE_B / "gate.json", "imagenet", "dropped (supervisor 2026-10-15)")
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
       "VNA_DEADLINE": str(VNA_DEADLINE)}
CPU_ENV = {k: v for k, v in ENV.items() if k != "CUDA_VISIBLE_DEVICES"}
archive_lock = threading.Lock(); archive_stop = threading.Event(); archive_errors = []
lane_timings = {}; projections = {}
MU_REL_DEFAULT = 1e-2
MU_SOURCE = "B5 fixed candidates: 1e-3 and 1e-2; gate 1e-2"
if MNIST_DIR is None:
    pickles = sorted(Path("/kaggle/input").rglob("mnist_"))
    raw = sorted(Path("/kaggle/input").rglob("train-images-idx3-ubyte"))
    MNIST_DIR = pickles[0].parent if pickles else (raw[0].parents[2] if raw else None)
def archive_outputs():
    with archive_lock:
        return _archive_outputs(RUN_ROOT, budget=DISK_CAP_BYTES,
                                folders=("outputs/stageB",), exclude_parts=("data",))
def periodic_archive():
    while not archive_stop.wait(900):
        try: archive_outputs()
        except Exception as error: archive_errors.append(str(error)); print("Archive error:", error, flush=True)
archiver = threading.Thread(target=periodic_archive, daemon=True); archiver.start()
def command(benchmark, arm, out, mu, source, *, smoke=False, rate=None, attached=False):
    device = "cpu" if benchmark == "regression" else "cuda:0"
    cmd = [sys.executable, "-u", "-m", "experiments.stage_b5", benchmark, "--arm", arm,
           "--device", device, "--out", str(out), "--mu-rel-default", str(mu),
           "--mu-rel-default-source", source]
    cmd += ["--smoke"] if smoke else ["--resume"]
    if rate is not None: cmd += ["--replacement-rate", str(rate)]
    if attached and MNIST_DIR: cmd += ["--data-dir", str(MNIST_DIR)]
    return cmd
def run_b5(benchmark, arm, lane, mu, source, *, smoke=False, rate=None):
    out = RUN_ROOT / "smoke/outputs/stageB/round2" if smoke else STAGE_B
    name = arm if rate is None else f"cbp_rr_{rate:g}"
    if not launch(command(benchmark, arm, out, mu, source, smoke=smoke, rate=rate,
                          attached=bool(MNIST_DIR)), lane, f"{benchmark}_{name}",
                  RUN_ROOT / f"{benchmark}_{name}_{lane}.log"):
        return None
    return json.loads((out / "b5" / benchmark / name / "summary.json").read_text())
def b5_job(benchmark, arm, lane, *, rate=None):
    mu, source = (MU_REL_DEFAULT, MU_SOURCE) if arm == "bp" else (MU_REL_DEFAULT, MU_SOURCE)
    if mu is None or pause_due(): return
    name = arm if rate is None else f"cbp_rr_{rate:g}"
    previous = STAGE_B / "b5" / benchmark / name / "summary.json"
    if previous.exists() and json.loads(previous.read_text()).get("status") != "PAUSED":
        run_b5(benchmark, arm, lane, mu, source, rate=rate)
        print("Reusing completed", benchmark, name, flush=True)
        return
    start = time.monotonic()
    summary = run_b5(benchmark, arm, lane, mu, source, smoke=True, rate=rate)
    if summary is None: return
    steps = 1000000 if benchmark == "regression" else 12000000
    point_seconds = (sum(p["seconds"] for p in summary["measurement_point_seconds"]) /
                     len(summary["measurement_point_seconds"])) if summary["measurement_point_seconds"] else 0
    import csv
    solve_path = RUN_ROOT / "smoke/outputs/stageB/round2/b5" / benchmark / name / "solves.csv"
    solves = list(csv.DictReader(solve_path.open())) if solve_path.exists() else []
    print(benchmark, name, "CG solves", len(solves), "mean seconds/solve",
          sum(float(r["seconds"]) for r in solves)/max(1, len(solves)), flush=True)
    import statistics
    measurement_path = RUN_ROOT / "smoke/outputs/stageB/round2/b5" / benchmark / name / "measurements.csv"
    measurement_rows = list(csv.DictReader(measurement_path.open())) if measurement_path.exists() else []
    consistency = [float(r["V_consistency"]) for r in measurement_rows if r.get("V_consistency")]
    print(benchmark, name, "V_consistency min/median/max",
          (min(consistency), statistics.median(consistency), max(consistency)) if consistency else (None, None, None), flush=True)
    startup = summary["startup_seconds"]
    steady = summary["steady_step_seconds"]
    points = 38 if benchmark == "regression" else 36
    hours = (startup + steady * steps + points * point_seconds) / 3600
    print(benchmark, arm, rate, f"startup {startup:.3f} s; {steady*1000:.3f} ms/step; "
          f"full steps {steps}; measurement {point_seconds:.3f} s/point; "
          f"projected {hours:.3f} hours; per-session GPU-hours {min(hours, SESSION_HOURS):.3f}; "
          f"estimated sessions {__import__('math').ceil(hours/SESSION_HOURS)}", flush=True)
    projections.setdefault(lane, []).append(hours + (time.monotonic()-start)/3600)
    projected = sum(projections[lane])
    print(lane, f"{projected:.3f} projected {'CPU' if lane == 'CPU' else 'GPU'}-hours including smoke", flush=True)
    gpu_total = sum(sum(values) for key, values in projections.items() if key != "CPU")
    if lane != "CPU" and gpu_total > GPU_HOURS_BUDGET and not ALLOW_OVER_BUDGET:
        raise RuntimeError("Projected total GPU-hours exceed budget")
    run_b5(benchmark, arm, lane, mu, source, rate=rate)
    archive_outputs()
'''

RUNTIME += SESSION_RUNTIME + '''
lane_state = {lane: {"job": "idle", "last": ""} for lane in ("CPU", "GPU0", "GPU1")}
def gpu_env(lane):
    return CPU_ENV if lane == "CPU" else {**ENV, "CUDA_VISIBLE_DEVICES": lane[-1]}
heartbeat_thread = threading.Thread(target=heartbeat, daemon=True); heartbeat_thread.start()
'''

RUN = '''def gpu0():
    start = time.monotonic()
    try: b5_job("mnist", "bp", "GPU0")
    finally: lane_timings["GPU0"] = time.monotonic()-start

def gpu1():
    start = time.monotonic()
    try: b5_job("mnist", "cbp", "GPU1", rate=1e-4)
    finally: lane_timings["GPU1"] = time.monotonic()-start

def cpu():
    start = time.monotonic()
    try:
        for rate in RATES: b5_job("regression", "cbp", "CPU", rate=rate)
        b5_job("regression", "bp", "CPU")
    finally: lane_timings["CPU"] = time.monotonic()-start

try:
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(gpu0), pool.submit(gpu1), pool.submit(cpu)]
        for future in futures: future.result()
finally:
    archive_stop.set(); archiver.join(timeout=10); heartbeat_thread.join(timeout=2)
    costs = {lane: sum(values) for lane, values in projections.items()}
    (STAGE_B / "cost_projection.json").write_text(json.dumps({"per_lane_hours": costs}, indent=2))
    (STAGE_B / "lane_timings.json").write_text(json.dumps(lane_timings, indent=2))
    archive_outputs()
if session_paused.is_set(): print("PAUSED — rerun with this version's output attached", flush=True)
if archive_errors: raise RuntimeError(archive_errors)
'''

REPORT = '''import pandas as pd
import matplotlib.pyplot as plt
for benchmark in (() if session_paused.is_set() else ("regression", "mnist")):
    finalize(STAGE_B, benchmark, mu_rel_default=MU_REL_DEFAULT)
gate_path = STAGE_B / "gate.json"
gates = json.loads(gate_path.read_text()) if gate_path.exists() else {}
gates["imagenet"] = "dropped (supervisor 2026-10-15)"
gate_path.write_text(json.dumps(gates, indent=2))
if (STAGE_B / "b5_baseline_gate.json").exists(): print("B5 BASELINE GATE", (STAGE_B / "b5_baseline_gate.json").read_text())
print("GATE G (B5 rows)", gates)
for path in (STAGE_B / "b5").glob("*/bp/measurements.csv"):
    rows = pd.read_csv(path); display(rows)
    for (rank, mu), group in rows.groupby(["rank", "mu_rel"]):
        group.groupby("task").rho_joint.mean().plot(marker="o", label=f"r={rank}, mu={mu}")
    plt.xlabel("task"); plt.ylabel("rho_joint"); plt.title(path.parents[1].name); plt.legend()
    plt.savefig(path.parent / "rho_vs_task.png", bbox_inches="tight"); plt.show()
print("Per-lane seconds:", lane_timings)
print("Per-lane projections:", {lane: sum(values) for lane, values in projections.items()})
archive_outputs()
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    p.add_argument('--output', default=str(ROOT/'notebooks/kaggle_vna_stage_b_nb1.ipynb'))
    args = p.parse_args()
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(args.commit), indent=1)+'\n')


if __name__ == '__main__':
    main()
