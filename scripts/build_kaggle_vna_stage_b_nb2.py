"""Build pinned Notebook 2: GPU0 B3, GPU1 reserved for B2 then B4."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.build_kaggle_vna_signal_curve_notebook import build as base_build, cell


def build(commit):
    base = base_build(commit)
    bootstrap = ''.join(base['cells'][2]['source']).replace('vna_phase0_seed0', 'vna_stage_b_nb2')
    setup = ''.join(base['cells'][3]['source'])
    tests = '"tests/test_vna.py", "tests/test_vna_stage_b_upstream.py", "tests/test_vna_stage_b_losses.py"'
    if tests not in setup: raise RuntimeError('Inspect changed base setup test command')
    setup = setup.replace(tests, '"tests/test_vna_stage_b3.py"')
    cells = [cell('markdown', '''# VNA Stage B3 — iCaRL, measure only (Notebook 2)
Enable Internet, T4×2 and secret `github_token`; attach CIFAR-100 and optional
prior Stage-B archives. The immutable pin must include this implementation.
GPU0 runs PyCIL's own pipeline with one GPU; GPU1 is reserved for B2 then B4.
Only the three tiny B3 CPU tests run. Smoke uses two tasks × two epochs on real
data and includes the full measurement cost. The budget stops an over-budget
run. Hourly archives preserve logs; task resume is unavailable, so a crash
restarts B3 training. Completed runs are reused. No correction is applied.
Reproduction compares mean NME top-1 across ten tasks with 64.42 ±1.5pp;
last-task reference ≈49.5 is recorded separately. Smoke/fake data cannot pass.
'''), cell('code', f'''REPO_COMMIT = "{commit}"
GPU_HOURS_BUDGET = 8
ALLOW_OVER_BUDGET = False
MU_REL_DEFAULT = 1e-2
DISK_CAP_BYTES = 5_000_000_000
'''), cell('code', bootstrap), cell('code', setup), cell('code', '''from experiments.stage_b5 import restore_archives
from experiments.vna_kaggle import archive_outputs as _archive_outputs
STAGE_B = RUN_ROOT / "outputs" / "stageB"
STAGE_B.mkdir(parents=True, exist_ok=True)
restore_archives(Path("/kaggle/input"), STAGE_B)
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
       "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}
archive_lock = threading.Lock()
archive_stop = threading.Event()
archive_errors = []
def archive_outputs():
    with archive_lock:
        return _archive_outputs(RUN_ROOT, budget=DISK_CAP_BYTES,
                                folders=("outputs/stageB",), exclude_parts=("data",))
def periodic_archive():
    while not archive_stop.wait(3600):
        try: archive_outputs()
        except Exception as error:
            archive_errors.append(str(error)); print("Archive error:", error, flush=True)
def command(out, *, smoke=False):
    cmd = [sys.executable, "-u", "-m", "experiments.stage_b3", "--out", str(out),
           "--device", "cuda:0", "--workers", "4", "--data-dir", str(DATA_ROOT),
           "--mu-rel-default", str(MU_REL_DEFAULT)]
    return cmd + (["--smoke"] if smoke else [])
archiver = threading.Thread(target=periodic_archive, daemon=True); archiver.start()
'''), cell('code', '''# Real-data smoke: 2 tasks × 2 epochs, full ranks/sites/128-sample diagnostics.
SMOKE = RUN_ROOT / "smoke" / "outputs" / "stageB" / "b3"
COMPLETED = (STAGE_B / "b3" / "summary.json").exists()
try:
    if not COMPLETED:
        started = time.monotonic()
        subprocess.run(command(SMOKE, smoke=True), cwd=REPO, env=ENV, check=True)
        smoke_seconds = time.monotonic()-started
        summary = json.loads((SMOKE / "summary.json").read_text())
        # 200 initial + 9*170 incremental epochs; 4 points per task.
        training = summary["training_seconds"] * 1730 / summary["epochs_run"]
        point = max(p["seconds"] for p in summary["measurement_point_seconds"])
        # Allow for 100 final outputs versus 20 at the end of smoke.
        diagnostic_output_factor = 100/20
        projected = (training + 40*point*diagnostic_output_factor)/3600
        total = projected + smoke_seconds/3600
        projection = {"projected_gpu_hours": projected, "smoke_gpu_hours": smoke_seconds/3600,
                      "total_gpu_hours": total, "full_epochs": 1730, "full_measurement_points": 40,
                      "diagnostic_output_factor": diagnostic_output_factor}
        (STAGE_B / "b3_cost_projection.json").write_text(json.dumps(projection, indent=2))
        print(projection)
        archive_outputs()
        if total > GPU_HOURS_BUDGET and not ALLOW_OVER_BUDGET:
            raise RuntimeError("Projected B3 GPU-hours exceed budget; review before ALLOW_OVER_BUDGET")
    else:
        print("Reusing completed B3 run")
except Exception:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs(); raise
'''), cell('code', '''# GPU1 PLACEHOLDER — B2 then B4 will be added later.
# No GPU1 training job is launched by this version.
'''), cell('code', '''try:
    if not COMPLETED:
        with (RUN_ROOT / "b3_gpu0.log").open("a", buffering=1) as log:
            subprocess.run(command(STAGE_B / "b3"), cwd=REPO, env=ENV, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
finally:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs()
if archive_errors: raise RuntimeError(archive_errors)
'''), cell('code', '''import pandas as pd
import matplotlib.pyplot as plt
B3 = STAGE_B / "b3"
print("REPRODUCTION GATE", (B3 / "reproduction_gate.json").read_text())
print("GATE G", (B3 / "gate.json").read_text())
gate = json.loads((B3 / "gate.json").read_text())
if not gate["measurements_valid"]:
    print("*** INVALID MEASUREMENTS: reproduction did not pass ***")
rows = pd.read_csv(B3 / "measurements.csv")
for (rank, mu), group in rows.groupby(["rank", "mu_rel"]):
    points = group.groupby(["task", "epoch"]).rho_joint.mean()
    x = [task + epoch/(200 if task == 1 else 170) for task, epoch in points.index]
    plt.plot(x, points.values, marker="o", label=f"r={rank}, mu={mu}")
plt.xlabel("task + epoch / task epochs"); plt.ylabel("rho_joint")
plt.title("B3 rho vs task/epoch"); plt.legend()
plt.savefig(B3 / "rho_vs_task_epoch.png", bbox_inches="tight"); plt.show()
display(rows[["task", "epoch", "site", "rank", "rho_joint", "rho_exemplar", "rho_new"]])
archive_outputs()
''')]
    return {'cells': cells, 'metadata': base['metadata'], 'nbformat': 4, 'nbformat_minor': 5}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    p.add_argument('--output', default=str(ROOT/'notebooks/kaggle_vna_stage_b_nb2.ipynb'))
    args = p.parse_args(); path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(args.commit), indent=1)+'\n')


if __name__ == '__main__': main()
