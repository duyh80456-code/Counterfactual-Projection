"""Shared notebook runtime for the preregistered performance pilots."""
from scripts.build_kaggle_vna_stage_b_rest import build as stage_build, RUNTIME
from scripts.build_kaggle_vna_signal_curve_notebook import cell


def notebook(commit, *, b5=False, quick=False):
    base = stage_build(commit, 2)
    cells = base['cells']
    config = f'''import time
SESSION_HOURS = 11.5
SESSION_START = time.time()
VNA_DEADLINE = SESSION_START + SESSION_HOURS*3600
REPO_COMMIT = "{commit}"
ALLOW_OVER_BUDGET = True
DISK_CAP_BYTES = 5_000_000_000
'''
    setup = ''.join(cells[3]['source'])
    setup = setup[:setup.index('\nMU_REL_DEFAULT =')]
    if b5:
        start = setup.index('import torch\n')
        end = setup.index('if not (REPO / \"experiments/vna_kaggle.py\").exists():')
        setup = setup[:start]+setup[end:]
        setup = setup.replace('print(\"Official TEST verified:\", DATA_ROOT / \"cifar-100-python/test\")', '')
    setup += '\nMU_REL_DEFAULT = 1e-2\nOFFICIAL_MU = 1e-2\nMU_SOURCE = \"preregistered-pilot\"\n'
    bootstrap = ''.join(cells[2]['source']).replace('vna_stage_b_nb2', 'vna_pilot_b5' if b5 else ('vna_b3_quick' if quick else 'vna_pilot_b3'))
    title = '# GIR/VNA B5 regression pilot\nCPU only, at most four processes. Three paired seeds; all upstream CBP rates.' if b5 else '# GIR/VNA B3 full performance pilot\nGPU0: icarl_vna → icarl_gir_vna. GPU1: icarl_ngd → icarl_gir → icarl_gir_randdir.'
    if quick: title = '# B3 iCaRL quick measurement pilot\nGPU0 only: icarl, with --measure-profile quick.'
    # Inherited environments cover bootstrap, installation, tests and all workers.
    bootstrap = bootstrap.replace('import hashlib, json, os, shutil, subprocess, sys, threading, time, zipfile\n',
        'import hashlib, json, os, shutil, subprocess, sys, threading, time, zipfile\nos.environ.update(TQDM_DISABLE="1", PYTHONUNBUFFERED="1")\n')
    setup = setup.replace('test_env.update(', 'test_env.update(PYTHONUNBUFFERED="1", ')
    title += '\nAttach previous stageB archives to resume. Smoke precedes full training. The current HEAD pin must be republished by the supervisor after committing. No real-data results are embedded.'
    runtime = RUNTIME[:RUNTIME.index('def prepare_teacher(')]
    runtime = runtime[:runtime.index('def finished(')] + runtime[runtime.index('def pause_due():'):]
    runtime = runtime[:runtime.index('def command(')] + runtime[runtime.index('def gpu_env('):]
    runtime += '''archive_errors = []
lane_timings = {}
archiver = threading.Thread(target=periodic_archive, daemon=True); archiver.start()
heartbeat_thread = threading.Thread(target=heartbeat, daemon=True); heartbeat_thread.start()
'''
    return dict(cells=[cell('markdown', title), cell('code', config), cell('code', bootstrap),
                       cell('code', setup), cell('code', runtime), cell('code', B5 if b5 else (QUICK if quick else B3))],
                metadata=base['metadata'], nbformat=4, nbformat_minor=5)


B5 = '''from experiments.pilot_b5 import ARMS, comparison
from experiments.stage_b5 import RATES
ROOT = STAGE_B / "pilot_b5"
ENV.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
jobs = [(arm, seed, rate) for seed in (0,1,2) for arm in ARMS
        for rate in (RATES if arm.startswith("cbp") else (None,))]
def cpu_job(job, smoke=False):
    arm, seed, rate = job
    name = f"{arm}_seed{seed}_rr{rate}"
    out = ROOT / "_smoke" if smoke else ROOT
    cmd = [sys.executable, "-u", "-m", "experiments.pilot_b5", "--arm", arm,
           "--seed", str(seed), "--out", str(out), "--resume"]
    if rate is not None: cmd += ["--replacement-rate", str(rate)]
    if smoke: cmd += ["--smoke"]
    if pause_due(): return
    # CPU workers use existing deadline and log forwarding infrastructure.
    from experiments.vna_resume import launch_logged
    return launch_logged(cmd, cwd=REPO, env={**ENV, "CUDA_VISIBLE_DEVICES":""},
        log_path=RUN_ROOT / f"{name}_{'smoke' if smoke else 'full'}.log",
        lane=name, job=name, lanes=lane_state, paused=session_paused, deadline=VNA_DEADLINE)
try:
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda j: cpu_job(j, True), [j for j in jobs if j[1]==0]))
    smoke_rows = [json.loads(p.read_text()) for p in (ROOT/"_smoke").glob("*/seed*/summary.json")]
    estimates = [{"arm":r["arm"], "replacement_rate":r["replacement_rate"],
                  "seconds_per_seed":r["estimated_full_seconds"]} for r in smoke_rows]
    seconds = sum(r["seconds_per_seed"]*3 for r in estimates)
    print("CPU estimates per arm × seed:", estimates)
    print("Four-core ideal hours:", seconds/4/3600, "total CPU-hours:", seconds/3600)
    (ROOT/"cost_projection.json").write_text(json.dumps(estimates, indent=2))
    archive_outputs()
    if not session_paused.is_set():
        with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(cpu_job, jobs))
finally:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs()
import pandas as pd
display(pd.DataFrame(comparison(ROOT)))
if session_paused.is_set(): print("PAUSED: attach this output and rerun to resume")
if archive_errors: raise RuntimeError(archive_errors)
'''

B3 = '''from experiments.pilot_b3 import ARMS
ROOT = STAGE_B / "pilot_b3"
LANES = [["icarl_vna", "icarl_gir_vna"], ["icarl_ngd", "icarl_gir", "icarl_gir_randdir"]]
SCHEDULE = ROOT / "_smoke/icarl_vna/schedule.json"
def pilot_command(arm, out, smoke=False):
    cmd = [sys.executable, "-u", "-m", "experiments.pilot_b3", "--arm", arm,
           "--out", str(out), "--device", "cuda:0", "--workers", "2", "--data-dir", str(DATA_ROOT)]
    if smoke:
        cmd += ["--smoke"]
        if "vna" in arm or "ngd" in arm: cmd += ["--calibrate"]
    elif "vna" in arm or "ngd" in arm: cmd += ["--schedule", str(SCHEDULE)]
    return cmd

def smoke_lane(gpu, arms):
    results = []
    for arm in arms:
        if pause_due(): break
        out = ROOT / "_smoke" / arm
        if not launch(pilot_command(arm, out, True), gpu, arm+"_smoke", RUN_ROOT/f"{arm}_smoke.log"): break
        summary = json.loads((out/"summary.json").read_text())
        train = summary["training_seconds"]*1730/max(1, summary["epochs_run"])
        quick = max((p["seconds"]*6400/(p["batch_size"]//2*10*p["task"]) for p in summary["measurement_point_seconds"]), default=0)*3 if arm=="icarl" else 0
        results.append(dict(arm=arm, training_seconds=train, quick_seconds=quick,
                            calibration=summary["calibration"], batches=max(summary["full_batches"]),
                            gir_seconds=5*sum(e["seconds"] for e in summary["intervention_events"] if e["kind"]=="gir_task")))
    return results
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        costs = list(pool.map(lambda item: smoke_lane(*item), enumerate(LANES)))
    if not session_paused.is_set():
        schedule = json.loads(SCHEDULE.read_text())
        print("K selected before full:", schedule)
        per_lane = []
        for lane in costs:
            total = 0
            for row in lane:
                correction = schedule["projected_vna_seconds"]*1730*row["batches"]/schedule["K"] if ("vna" in row["arm"] or "ngd" in row["arm"]) else 0
                total += row["training_seconds"]+row["quick_seconds"]+correction+row["gir_seconds"]
            per_lane.append(total/3600)
        estimate = dict(per_lane_gpu_hours=per_lane, total_gpu_hours=sum(per_lane),
                        sessions=max(per_lane)/SESSION_HOURS, schedule=schedule, smoke=costs)
        (ROOT/"cost_projection.json").write_text(json.dumps(estimate, indent=2))
        print("GPU-hour estimate:", estimate)
        archive_outputs()
        if estimate["sessions"] >= 3:
            raise RuntimeError("Smoke projects at least three Kaggle sessions; full run withheld by preregistered rule")
        def train_lane(gpu, arms):
            for arm in arms:
                if pause_due(): break
                out = ROOT / arm
                done = out/"summary.json"
                if done.exists():
                    summary = json.loads(done.read_text())
                    if summary.get("status")=="COMPLETED":
                        if summary["identity"]["code_commit"] != REPO_COMMIT: raise ValueError("completed commit mismatch")
                        continue
                if not launch(pilot_command(arm, out), gpu, arm, RUN_ROOT/f"{arm}_full.log"): break
                archive_outputs()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda item: train_lane(*item), enumerate(LANES)))
finally:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs()
import pandas as pd
rows = []
paths = [ROOT/arm/"summary.json" for arm in ARMS if arm != "icarl"]
for path in sorted(Path("/kaggle/input").rglob("summary.json")):
    if "_smoke" in path.parts: continue
    try: baseline = json.loads(path.read_text())
    except (OSError, ValueError): continue
    if (baseline.get("arm") == "icarl" and baseline.get("measure_profile") == "quick"
            and baseline.get("status") == "COMPLETED" and baseline.get("tasks_run") == 10):
        paths.insert(0, path)
        print("Attached quick baseline:", path)
        break
for path in paths:
    if not path.exists(): continue
    s = json.loads(path.read_text())
    if s.get("status")!="COMPLETED": continue
    if s["arm"]=="icarl": print("Baseline reproduction gate (64.42 ± 1.5 NME):", s["reproduction"])
    row = dict(arm=s["arm"])
    for label in ("cnn", "nme"):
        last = s["tasks"][-1]
        row.update({label+"_average":last[label+"_average_accuracy"], label+"_last":last[label+"_top1"], label+"_forgetting":last[label+"_forgetting"]})
    rows.append(row)
control = next((r for r in rows if r["arm"]=="icarl"), None)
if control:
    for row in rows:
        for k,v in control.items():
            if k!="arm": row["delta_"+k] = row[k]-v
        # Iterate a copy because delta fields are added to the control row too.
if not control: print("No attached completed iCaRL quick summary found; showing arms alone.")
display(pd.DataFrame(rows))
if session_paused.is_set(): print("PAUSED: attach this output and rerun to resume")
if archive_errors: raise RuntimeError(archive_errors)
'''
# Stable iteration while the baseline row gains its delta fields.
B3 = B3.replace('for k,v in control.items():', 'for k,v in list(control.items()):').replace('if k!="arm": row["delta_"+k]', 'if k!="arm" and not k.startswith("delta_"): row["delta_"+k]')


QUICK = B3[:B3.index('try:\n    with ThreadPoolExecutor')]
QUICK = QUICK.replace('LANES = [["icarl_vna", "icarl_gir_vna"], ["icarl_ngd", "icarl_gir", "icarl_gir_randdir"]]', 'LANES = [["icarl"]]')
QUICK = QUICK.replace('"pilot_b3"', '"pilot_b3_quick"')
QUICK = QUICK.replace('"--arm", arm,', '"--arm", arm, "--measure-profile", "quick",')
QUICK += '''try:
    costs = smoke_lane(0, ["icarl"])
    print("Quick smoke cost projection:", costs)
    archive_outputs()
    if not session_paused.is_set():
        launch(pilot_command("icarl", ROOT / "icarl"), 0, "icarl", RUN_ROOT / "icarl_full.log")
finally:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs()
import pandas as pd
out = ROOT / "icarl"
summary = out / "summary.json"
print("Baseline reproduction gate (64.42 ± 1.5 NME):")
if summary.exists(): print(json.loads(summary.read_text())["reproduction"])
else: print("Pending completion")
measurements = out / "measurements.csv"
print("Quick measurement rows summary:")
if measurements.exists() and measurements.stat().st_size:
    rows = pd.read_csv(measurements)
    print("rows:", len(rows))
    keys = [k for k in ("task", "epoch", "site", "status") if k in rows.columns]
    if keys: display(rows.groupby(keys, dropna=False).size().reset_index(name="rows"))
else: print("No measurement rows yet")
gate = out / "gate_quick.json"
print("gate_quick.json:", gate.read_text() if gate.exists() else "Pending completion")
if session_paused.is_set(): print("PAUSED: attach this output and rerun to resume")
if archive_errors: raise RuntimeError(archive_errors)
'''
