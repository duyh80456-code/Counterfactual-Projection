"""Build Notebook 1: Stage A and MNIST GPU lanes plus CPU regression."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_kaggle_vna_stage_a_notebook import build as stage_a_build
from scripts.build_kaggle_vna_signal_curve_notebook import cell
from scripts.build_kaggle_vna_stage_b_rest import SESSION_RUNTIME


def build(commit):
    base = stage_a_build(commit)
    constants = ''.join(base['cells'][1]['source'])
    constants = constants.replace('GPU_HOURS_BUDGET = 3', 'GPU_HOURS_BUDGET = 8')
    constants += 'MNIST_DIR = None\nDISK_CAP_BYTES = 5_000_000_000\n'
    bootstrap = ''.join(base['cells'][2]['source']).replace('vna_stage_a', 'vna_stage_b_nb1')
    setup = ''.join(base['cells'][3]['source'])
    return {'cells': [cell('markdown', '''# VNA Notebook 1 — Stage A → B5
Run this notebook first with Internet, T4×2, CIFAR-100 and the extracted Phase-0
checkpoint dataset. GPU0 runs the cost smoke → a5 → null → selection → joint → mu → solver →
summary, then measured MNIST BP. GPU1 immediately runs MNIST CBP (rr 1e-4).
A separate CPU subprocess runs the five regression CBP arms immediately, then
measured BP when Stage A's summary exists. BP uses the newly produced default μ
and measures both 1e-3 and 1e-2; Gate G uses 1e-2. A failed Stage A permits BP with a recorded 1e-2 fallback.
ImageNet-32 is dropped. MNIST uses upstream load_mnist.py with attached data as
fallback. The 11.5-hour deadline pauses with a final archive; attach its output to resume.
Task and mid-task checkpoints are archived every 15 minutes. Every loader uses two workers; CPU torch threads are limited to two.
Attach prior archives to resume with the existing Stage A identity rules.
Keep the standalone Stage A notebook as a fallback. Attach this notebook's
Stage A summary to nb2/nb3 and run those in separate sessions.
Regenerate with the published commit containing this implementation before Kaggle.
'''), cell('code', constants), cell('code', bootstrap), cell('code', setup),
        cell('code', RUNTIME), cell('code', RUN), cell('code', REPORT)],
        'metadata': base['metadata'], 'nbformat': 4, 'nbformat_minor': 5}


RUNTIME = '''from concurrent.futures import ThreadPoolExecutor
from experiments.vna_kaggle import archive_with_status as _archive_outputs
from experiments.stage_b5 import restore_archives as restore_b, RATES, finalize, merge_json
from diagnostics.stage_a import restore_archives as restore_a
STAGE_A = RUN_ROOT / "outputs/stageA"
STAGE_B = RUN_ROOT / "outputs/stageB"
STAGE_A.mkdir(parents=True, exist_ok=True); STAGE_B.mkdir(parents=True, exist_ok=True)
restore_a(Path("/kaggle/input"), STAGE_A); restore_b(Path("/kaggle/input"), STAGE_B)
merge_json(STAGE_B / "gate.json", "imagenet", "dropped (supervisor 2026-10-15)")
from experiments.phase0_discovery import discover_phase0
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
       "VNA_DEADLINE": str(VNA_DEADLINE)}
CPU_ENV = {k: v for k, v in ENV.items() if k != "CUDA_VISIBLE_DEVICES"}
archive_lock = threading.Lock(); archive_stop = threading.Event(); archive_errors = []
stage_a_done = threading.Event(); stage_a_errors = []; lane_timings = {}; projections = {}
MU_REL_DEFAULT = 1e-2
MU_SOURCE = "fallback: stage A failed"
if MNIST_DIR is None:
    pickles = sorted(Path("/kaggle/input").rglob("mnist_"))
    raw = sorted(Path("/kaggle/input").rglob("train-images-idx3-ubyte"))
    MNIST_DIR = pickles[0].parent if pickles else (raw[0].parents[2] if raw else None)
def archive_outputs():
    with archive_lock:
        return _archive_outputs(RUN_ROOT, budget=DISK_CAP_BYTES,
                                folders=("outputs/stageA", "outputs/stageB"), exclude_parts=("data",))
def periodic_archive():
    while not archive_stop.wait(900):
        try: archive_outputs()
        except Exception as error: archive_errors.append(str(error)); print("Archive error:", error, flush=True)
archiver = threading.Thread(target=periodic_archive, daemon=True); archiver.start()
def stage_command(measurement, *, cost=False):
    common = ["--checkpoint-dir", str(RECIPE), "--data-root", str(DATA_ROOT),
              "--output", str(STAGE_A), "--seed", "0"]
    cmd = [sys.executable, "-u", "-m", "diagnostics.stage_a", measurement, *common, "--device", "cuda:0"]
    if cost: cmd += ["--checkpoints", str(RECIPE / "epoch_090.pt"), str(RECIPE / "epoch_100.pt")]
    return cmd
def wait_mu():
    path = STAGE_A / "summary.json"
    # Poll the file, but do not reuse a restored summary before Stage A validates it.
    while not stage_a_done.is_set() or not path.exists():
        if pause_due(): return None, "PAUSED"
        if stage_a_done.wait(2) and stage_a_errors:
            print("*** WARNING: Stage A failed; BP μ=1e-2 ***", stage_a_errors, flush=True)
            return 1e-2, "fallback: stage A failed"
    if stage_a_errors: return 1e-2, "fallback: stage A failed"
    return json.loads(path.read_text())["mu_rel_default"], str(path)
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
    out = RUN_ROOT / "smoke/outputs/stageB" if smoke else STAGE_B
    name = arm if rate is None else f"cbp_rr_{rate:g}"
    if not launch(command(benchmark, arm, out, mu, source, smoke=smoke, rate=rate,
                          attached=bool(MNIST_DIR)), lane, f"{benchmark}_{name}",
                  RUN_ROOT / f"{benchmark}_{name}_{lane}.log"):
        return None
    return json.loads((out / "b5" / benchmark / name / "summary.json").read_text())
def b5_job(benchmark, arm, lane, *, rate=None):
    mu, source = wait_mu() if arm == "bp" else (1e-2, "cbp: unmeasured; no mu required")
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
    solve_path = RUN_ROOT / "smoke/outputs/stageB/b5" / benchmark / name / "solves.csv"
    solves = list(csv.DictReader(solve_path.open())) if solve_path.exists() else []
    print(benchmark, name, "CG solves", len(solves), "mean seconds/solve",
          sum(float(r["seconds"]) for r in solves)/max(1, len(solves)), flush=True)
    startup = summary["startup_seconds"]
    steady = summary["steady_step_seconds"]
    points = (16 if benchmark == "regression" else 18) if arm == "bp" else 0
    hours = (startup + steady * steps + points * point_seconds) / 3600
    print(benchmark, arm, rate, f"startup {startup:.3f} s; {steady*1000:.3f} ms/step; "
          f"full steps {steps}; measurement {point_seconds:.3f} s/point; "
          f"projected {hours:.3f} hours", flush=True)
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
    global MU_REL_DEFAULT, MU_SOURCE
    start = time.monotonic()
    try:
        try:
            global RECIPE
            RECIPE = discover_phase0()
            for measurement in ("cost", "a5", "null", "selection", "joint", "mu", "solver", "summary"):
                if not launch(stage_command(measurement, cost=measurement == "cost"),
                              "GPU0", "stageA_"+measurement,
                              RUN_ROOT / f"stage_a_{measurement}_gpu0.log"): return
                if measurement == "cost":
                    cost = json.loads((STAGE_A / "cost_projection.json").read_text())
                    hours = cost["total_projected_gpu_hours"] + cost["smoke_gpu_hours"]
                    projections.setdefault("GPU0", []).append(hours)
                    print("GPU0 Stage A:", hours, "projected GPU-hours including smoke", flush=True)
                    if hours > GPU_HOURS_BUDGET and not ALLOW_OVER_BUDGET:
                        raise RuntimeError("Stage A projected GPU-hours exceed budget")
                archive_outputs()
        except Exception as error:
            stage_a_errors.append(str(error)); print("Stage A failed:", error, flush=True)
        finally: stage_a_done.set()
        MU_REL_DEFAULT, MU_SOURCE = wait_mu()
        b5_job("mnist", "bp", "GPU0")
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
summary = STAGE_A / "summary.json"
print("STAGE A SUMMARY", summary.read_text() if not stage_a_errors and summary.exists() else stage_a_errors)
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
