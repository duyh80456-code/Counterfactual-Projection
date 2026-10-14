"""Build the pinned two-queue CIFAR pilot notebook, without full gates or archives."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.build_kaggle_vna_stage_b_rest import build as build_rest
from scripts.build_kaggle_vna_signal_curve_notebook import cell


LANES = [
    [{'name': 'b3', 'module': 'experiments.stage_b3', 'args': []},
     {'name': 'b1_resnet8x4', 'module': 'experiments.stage_b12', 'args': ['b1', '--student', 'resnet8x4']},
     {'name': 'b1_wrn_16_2', 'module': 'experiments.stage_b12', 'args': ['b1', '--student', 'wrn_16_2']}],
    [{'name': 'b2_resnet8x4', 'module': 'experiments.stage_b12', 'args': ['b2', '--student', 'resnet8x4']},
     {'name': 'b2_wrn_16_2', 'module': 'experiments.stage_b12', 'args': ['b2', '--student', 'wrn_16_2']},
     {'name': 'b4', 'module': 'experiments.stage_b4', 'args': ['train']}],
]


def build(commit):
    base = build_rest(commit, 2)
    bootstrap = ''.join(base['cells'][2]['source']).replace('vna_stage_b_nb2', 'vna_cifar_pilot')
    setup = ''.join(base['cells'][3]['source'])
    # Retain pinned dependencies/upstreams and dataset discovery; pilot runs are the checks.
    return {'cells': [cell('markdown', '''# CIFAR pilot — Kaggle T4×2
Enable Internet, T4×2 and the `github_token` secret; attach real CIFAR-100.
Two queues run six bounded real-data pilots, aiming for 15–20 minutes total.
Each uses two epochs, ≤20 training batches/epoch, ≤5 validation/test batches,
and one real measurement (rank 4, two null repeats, CG 200).
B3 runs two tasks so the final measurement exercises iCaRL KD.
B2 fetches and hash-verifies its teacher and reports a bounded teacher gate.
B4 trains then measures the epoch-2 checkpoint in the same process.
All gates are PILOT. Each job has its own log and failures leave both queues running.
No archives or full training runs. Reduced diagnostics can underestimate full costs.
Regenerate with the eventual published commit containing this implementation before Kaggle.
'''), cell('code', f'REPO_COMMIT = "{commit}"\nPILOT_BATCHES = 20\nMU_REL_DEFAULT = 1e-2\nLANES = {LANES!r}\n'),
        cell('code', bootstrap), cell('code', setup), cell('code', RUN), cell('code', REPORT)],
        'metadata': base['metadata'], 'nbformat': 4, 'nbformat_minor': 5}


RUN = '''from concurrent.futures import ThreadPoolExecutor
PILOT = RUN_ROOT / "outputs" / "pilot"
PILOT.mkdir(parents=True, exist_ok=True)
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
       "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}
def pilot_lane(gpu, jobs):
    results = []
    for job in jobs:
        out = PILOT / job["name"]
        # Fresh outputs ensure a failed rerun cannot reuse a stale summary.
        if out.exists(): shutil.rmtree(out)
        out.mkdir(parents=True)
        log_path = PILOT / (job["name"] + ".log")
        cmd = [sys.executable, "-u", "-m", job["module"], *job["args"],
               "--pilot", "--pilot-batches", str(PILOT_BATCHES), "--out", str(out),
               "--device", "cuda:0", "--workers", "2", "--data-dir", str(DATA_ROOT),
               "--mu-rel-default", str(MU_REL_DEFAULT), "--mu-source", MU_SOURCE]
        try:
            with log_path.open("w", buffering=1) as log:
                completed = subprocess.run(cmd, cwd=REPO, env={**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
            ok = completed.returncode == 0 and (out / "summary.json").exists()
            error = "" if ok else f"exit {completed.returncode}"
        except Exception as exc:
            ok, error = False, str(exc)
            with log_path.open("a") as log: log.write(error + "\\n")
        results.append({"setting": job["name"], "ok": ok, "error": error, "log": str(log_path)})
        print(job["name"], "OK" if ok else "FAILED", flush=True)
    return results
with ThreadPoolExecutor(max_workers=2) as pool:
    RESULTS = [result for lane in pool.map(lambda item: pilot_lane(*item), enumerate(LANES)) for result in lane]
(PILOT / "jobs.json").write_text(json.dumps(RESULTS, indent=2))
'''

REPORT = '''import pandas as pd
rows = []
for result in RESULTS:
    name = result["setting"]
    path = PILOT / name / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    status = "OK" if result["ok"] else "FAILED: " + result["error"] + "\\n" + "\\n".join(Path(result["log"]).read_text().splitlines()[-20:])
    rows.append({"setting": name, "status": status,
                 "epoch seconds": summary.get("epoch_seconds"),
                 "measurement seconds": summary.get("seconds_per_measurement_point"),
                 "peak memory (bytes)": summary.get("peak_gpu_memory_bytes"),
                 "projected full GPU-hours": summary.get("projected_full_gpu_hours"),
                 "gate keys present": summary.get("gate_keys_present", False)})
with pd.option_context("display.max_colwidth", None):
    display(pd.DataFrame(rows))
for notebook, lanes in {"nb2": (("b3",), ("b2_resnet8x4", "b4")),
                        "nb3": (("b1_resnet8x4", "b1_wrn_16_2"), ("b2_wrn_16_2",))}.items():
    for gpu, names in enumerate(lanes):
        selected = [r for r in rows if r["setting"] in names]
        total = sum(r["projected full GPU-hours"] or 0 for r in selected)
        if notebook == "nb2":
            b4_path = PILOT / "b4/summary.json"
            if b4_path.exists():
                b4 = json.loads(b4_path.read_text())
                measure_hours = 12 * b4.get("seconds_per_measurement_point", 0) / 3600
                total += measure_hours if gpu == 0 else -measure_hours
        complete = all(r["status"] == "OK" for r in selected)
        print(f"{notebook} GPU{gpu}: {total:.3f} projected GPU-hours" +
              ("" if complete else " (partial; failed jobs omitted)"))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    p.add_argument('--output', default=str(ROOT/'notebooks/kaggle_vna_cifar_pilot.ipynb'))
    args = p.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(args.commit), indent=1)+'\n')


if __name__ == '__main__': main()
