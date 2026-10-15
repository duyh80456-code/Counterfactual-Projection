"""Shared pinned setup, two GPU queues, budget, archives and report for NB2/3."""
from scripts.build_kaggle_vna_signal_curve_notebook import build as base_build, cell


def build(commit, number):
    base = base_build(commit)
    bootstrap = ''.join(base['cells'][2]['source']).replace('vna_phase0_seed0', f'vna_stage_b_nb{number}')
    setup = ''.join(base['cells'][3]['source'])
    setup = setup.replace('test_env.update(', 'test_env.update(TQDM_DISABLE="1", ')
    # Base setup names a subset; every Stage B lane requires the full suite.
    import re
    setup = re.sub(r'"pytest", "-q"[^\n]*?\], cwd=REPO,', '"pytest", "-q"], cwd=REPO,', setup) + MU_SETUP
    if number == 3:
        return {'cells': [cell('markdown', '# VNA Stage B — Notebook 3 (deferred)\nB1 and B2 run after supervisor review of B3. No jobs are launched.'),
                          cell('code', f'REPO_COMMIT = "{commit}"\nprint("B1/B2 deferred until after B3 review")')],
                'metadata': base['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
    lanes = [[{'name': 'b3', 'module': 'experiments.stage_b3', 'args': [], 'epochs': 1730,
               'points': 13, 'output_factor': 5}], []]
    return {'cells': [cell('markdown', f'''# VNA Stage B — Notebook {number}
T4×2, Internet and `github_token`; attach CIFAR-100 and prior Stage-B archives.
Both μ candidates (1e-3 and 1e-2) are measured independently of nb1.
Gate G always reads μ=1e-2. Attach the Stage A summary for its provenance.
Only B3 runs (13 measurement points); B4 is deferred. Every loader uses two workers.
Real-data two-epoch
smokes include all measurement sites/ranks/nulls and feed the total GPU-hour
budget. Hourly archives retain full checkpoints and logs. B3 resumes tasks and epoch prefixes.
The 11.5-hour session deadline pauses cleanly; attach its output to resume. The pin must include these files: since implementation
is uncommitted, regenerate with the eventual published commit before Kaggle.
'''), cell('code', f'''import time
SESSION_HOURS = 11.5
SESSION_START = time.time()
VNA_DEADLINE = SESSION_START + SESSION_HOURS * 3600
REPO_COMMIT = "{commit}"
GPU_HOURS_BUDGET = 8
ALLOW_OVER_BUDGET = False
MU_REL_DEFAULT = 1e-3
DISK_CAP_BYTES = 5_000_000_000
LANES = {repr(lanes)}
'''), cell('code', bootstrap), cell('code', setup), cell('code', RUNTIME),
        cell('code', SMOKE), cell('code', TRAIN), cell('code', REPORT)],
        'metadata': base['metadata'], 'nbformat': 4, 'nbformat_minor': 5}


RUNTIME = '''from concurrent.futures import ThreadPoolExecutor
from experiments.stage_b5 import restore_archives
from experiments.vna_kaggle import archive_outputs as _archive_outputs
STAGE_B = RUN_ROOT / "outputs" / "stageB"
STAGE_B.mkdir(parents=True, exist_ok=True)
restore_archives(Path("/kaggle/input"), STAGE_B)
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
       "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python", "VNA_DEADLINE": str(VNA_DEADLINE)}
session_paused = threading.Event()
lane_state = {gpu: {"job": "idle", "last": ""} for gpu in range(2)}
def finished(out, job):
    filename = "training_summary.json" if job.get("train_only") else "summary.json"
    path = out / filename
    if not path.exists(): return False
    summary = json.loads(path.read_text())
    if summary.get("status") != "COMPLETED": return False
    identity = summary.get("identity", summary)
    expected_setting = "B4" if job["name"] == "b4" else job["name"].split("_")[0].upper()
    if identity.get("setting") != expected_setting or identity.get("workers") != 2:
        raise ValueError(f"completed job protocol mismatch: {out}")
    if "--student" in job["args"] and identity.get("student") != job["args"][job["args"].index("--student")+1]:
        raise ValueError(f"completed job student identity mismatch: {out}")
    if job.get("teacher"):
        from scripts.fetch_teachers import TEACHERS
        if identity.get("teacher_hash") != TEACHERS[job["teacher"]][1]:
            raise ValueError(f"completed job teacher identity mismatch: {out}")
    if identity.get("code_commit") != REPO_COMMIT:
        raise ValueError(f"completed job code identity mismatch: {out}")
    candidates = identity.get("mu_candidates") or [1e-3, 1e-2]
    if candidates != sorted({MU_REL_DEFAULT, .01}):
        raise ValueError(f"completed job μ identity mismatch: {out}")
    return True
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
                                folders=("outputs/stageB",), exclude_parts=("data",))
def periodic_archive():
    while not archive_stop.wait(3600):
        try: archive_outputs()
        except Exception as error:
            archive_errors.append(str(error)); print("Archive error:", error, flush=True)
def command(job, out, *, smoke=False, measure=False):
    args = ["measure"] if measure else job["args"]
    cmd = [sys.executable, "-u", "-m", job["module"], *args, "--out", str(out),
           "--device", "cuda:0", "--workers", "2", "--solver", "auto", "--data-dir", str(DATA_ROOT),
           "--mu-rel-default", str(MU_REL_DEFAULT), "--mu-source", MU_SOURCE]
    if OFFICIAL_MU is not None:
        cmd += ["--official-mu", str(OFFICIAL_MU)]
    if job["name"] == "b4" and (out / "recipe/latest.pt").exists() and job["args"] == ["train"] and not measure:
        cmd += ["--resume", str(out / "recipe/latest.pt")]
    return cmd + (["--smoke"] if smoke else [])
def gpu_env(gpu): return {**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)}
def prepare_teacher(job, gpu):
    name = job.get("teacher")
    if not name: return
    gate_path = STAGE_B / "teacher_gate.json"
    gates = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    if gates.get(name, {}).get("status") == "PASS":
        # Re-fetch/re-verify bytes if teachers are absent after archive restore.
        from scripts.fetch_teachers import fetch
        fetch(name)
        return
    subprocess.run([sys.executable, "-u", "-m", "scripts.fetch_teachers", "--teacher", name,
                    "--data-dir", str(DATA_ROOT), "--device", "cuda:0", "--workers", "2", "--gate-out", str(gate_path)],
                   cwd=REPO, env=gpu_env(gpu), check=True)
b4_train_done = threading.Event(); b4_train_errors = []
lane_timings = {}
archiver = threading.Thread(target=periodic_archive, daemon=True); archiver.start()
heartbeat_thread = threading.Thread(target=heartbeat, daemon=True); heartbeat_thread.start()
'''

SMOKE = '''def smoke_lane(gpu, jobs):
    costs = []
    for job in jobs:
        if pause_due(): break
        if job.get("wait_for_train"): continue
        if finished(STAGE_B / job["name"], job): continue
        started = time.monotonic()
        prepare_teacher(job, gpu)
        smoke_out = STAGE_B / "_smoke" / job["name"]
        if not launch(command(job, smoke_out, smoke=True), gpu, job["name"]+"_smoke",
                      RUN_ROOT / f'{job["name"]}_smoke_gpu{gpu}.log'): break
        summary = json.loads((smoke_out / "summary.json").read_text())
        import csv
        solves = list(csv.DictReader((smoke_out / "solves.csv").open()))
        print(job["name"], "CG solves", len(solves), "mean seconds/solve",
              sum(float(r["seconds"]) for r in solves)/max(1, len(solves)), flush=True)
        import statistics
        measurement_path = smoke_out / "measurements.csv"
        measurement_rows = list(csv.DictReader(measurement_path.open())) if measurement_path.exists() else []
        for row in measurement_rows:
            print(job["name"], "task", row.get("task"), "n_out", row.get("n_out"),
                  "M_build_seconds", row.get("M_build_seconds"), "factor_seconds", row.get("factor_seconds"),
                  "peak RSS", row.get("peak_rss_bytes"),
                  "tiny_V peak CUDA", row.get("tiny_V_peak_cuda_bytes"), flush=True)
        consistency = [float(r["V_consistency"]) for r in measurement_rows if r.get("V_consistency")]
        print(job["name"], "V_consistency min/median/max",
              (min(consistency), statistics.median(consistency), max(consistency)) if consistency else (None, None, None), flush=True)
        training = summary["training_seconds"] * job["epochs"] / summary["epochs_run"]
        point = max(p["seconds"] for p in summary["measurement_point_seconds"])
        full = (training + job["points"]*point*job.get("output_factor", 1))/3600
        cost = {"setting": job["name"], "projected_gpu_hours": full,
                "smoke_gpu_hours": (time.monotonic()-started)/3600}
        costs.append(cost); print(cost, flush=True)
    return costs
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        projections = list(pool.map(lambda item: smoke_lane(*item), enumerate(LANES)))
    per_lane = {f"GPU{gpu}": sum(c["projected_gpu_hours"] + c["smoke_gpu_hours"] for c in costs)
                for gpu, costs in enumerate(projections)}
    # B4 smoke train includes measurement; charge its measurement cost to GPU0.
    b4_smoke = STAGE_B / "_smoke/b4/summary.json"
    if b4_smoke.exists() and any(j.get("wait_for_train") for j in LANES[0]):
        b4 = json.loads(b4_smoke.read_text())
        per_lane["GPU0"] += 12 * max(p["seconds"] for p in b4["measurement_point_seconds"]) / 3600
    for lane, hours in per_lane.items(): print(lane, f"{hours:.3f} projected GPU-hours", flush=True)
    projections = [cost for lane in projections for cost in lane]
    total = sum(per_lane.values())
    (STAGE_B / "cost_projection.json").write_text(json.dumps({"total_gpu_hours": total, "per_lane_gpu_hours": per_lane, "settings": projections}, indent=2))
    archive_outputs()
    if not session_paused.is_set() and total > GPU_HOURS_BUDGET and not ALLOW_OVER_BUDGET:
        raise RuntimeError("Projected total GPU-hours exceed budget; review before ALLOW_OVER_BUDGET")
except Exception:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs(); raise
'''

TRAIN = '''def train_lane(gpu, jobs):
    started = time.monotonic()
    try:
        for job in jobs:
            if pause_due(): break
            out = STAGE_B / job["name"]
            if job.get("wait_for_train"):
                while not (out / "training_summary.json").exists():
                    if pause_due(): return
                    if b4_train_done.wait(2):
                        if b4_train_errors or not (out / "training_summary.json").exists():
                            raise RuntimeError("B4 final checkpoint unavailable: " + str(b4_train_errors))
            if finished(out, job):
                if job.get("train_only"): b4_train_done.set()
                print("Reusing completed", job["name"], flush=True); continue
            prepare_teacher(job, gpu)
            try:
                if not launch(command(job, out), gpu, job["name"],
                              RUN_ROOT / f'{job["name"]}_gpu{gpu}.log'): break
            except Exception as error:
                if job.get("train_only"): b4_train_errors.append(str(error))
                raise
            finally:
                if job.get("train_only"): b4_train_done.set()
            archive_outputs()
    except Exception as error:
        if gpu == 1:
            b4_train_errors.append(str(error)); b4_train_done.set()
        raise
    finally:
        lane_timings[f"GPU{gpu}"] = time.monotonic() - started
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: train_lane(*item), enumerate(LANES)))
finally:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs()
if session_paused.is_set(): print("PAUSED — rerun with this version's output attached", flush=True)
if archive_errors: raise RuntimeError(archive_errors)
'''

REPORT = '''import pandas as pd
import matplotlib.pyplot as plt
print("Per-lane seconds:", lane_timings)
teacher = STAGE_B / "teacher_gate.json"
print("TEACHER GATE", teacher.read_text() if teacher.exists() else "no teachers in this archive")
gates = {}; reproduction = []; gate_rows = []
for folder in sorted(STAGE_B.iterdir()):
    if not folder.is_dir() or not (folder / "gate.json").exists(): continue
    gate = json.loads((folder / "gate.json").read_text()); gates[folder.name] = gate
    for mu, result in gate["by_mu"].items():
        gate_rows.append({"setting": folder.name, "mu": float(mu),
                          "status": result["status"], "official": float(mu) == OFFICIAL_MU})
    for filename in ("reproduction.json", "reproduction_gate.json"):
        path = folder / filename
        if path.exists(): reproduction.append({"setting_name": folder.name, **json.loads(path.read_text())})
    path = folder / "measurements.csv"
    if not path.exists(): continue
    rows = pd.read_csv(path)
    if "rho_joint" not in rows: continue
    plt.figure()
    for (rank, mu), group in rows.groupby(["rank", "mu_rel"]):
        keys = ["task", "epoch"] if folder.name == "b3" else ["epoch"]
        points = group.groupby(keys).rho_joint.mean()
        x = [task + epoch/(200 if task == 1 else 170) for task, epoch in points.index] if folder.name == "b3" else points.index
        plt.plot(x, points.values, marker="o", label=f"r={rank}, mu={mu}")
    plt.xlabel("task + epoch/task epochs" if folder.name == "b3" else "epoch")
    plt.ylabel("rho_joint"); plt.title(folder.name); plt.legend()
    plt.savefig(folder / "rho_vs_epoch.png", bbox_inches="tight"); plt.show()
merged_path = STAGE_B / "gate.json"
merged = json.loads(merged_path.read_text()) if merged_path.exists() else {}
merged.update(gates); merged_path.write_text(json.dumps(merged, indent=2))
if OFFICIAL_MU is None: print("official μ pending standalone Stage A notebook")
gate_table = pd.DataFrame(gate_rows, columns=["setting", "mu", "status", "official"])
display(gate_table.style.apply(lambda row: ["background-color: #fff2cc" if row["official"] else ""] * len(row), axis=1))
display(pd.DataFrame(reproduction))
archive_outputs()
'''



MU_SETUP = '''
MU_REL_DEFAULT = 1e-3
OFFICIAL_MU = 1e-2
MU_SOURCE = "both-candidates"
inputs = Path("/kaggle/input")
candidates = sorted(inputs.glob("**/stageA/summary.json"))
candidates += [p for p in sorted(inputs.rglob("summary.json")) if p not in candidates]
for path in candidates:
    try:
        summary = json.loads(path.read_text())
    except (ValueError, OSError): continue
    if "mu_rel_default" in summary and (path.parent.name == "stageA" or "mu_rule" in summary):
        break
print("measurement μ candidates:", (1e-3, 1e-2), "official μ:", OFFICIAL_MU)
'''


# NB1 and the standalone Stage A notebook use the same deadline/launch/heartbeat.
SESSION_RUNTIME = RUNTIME[RUNTIME.index('session_paused ='):RUNTIME.index('def finished(')]
SESSION_RUNTIME += RUNTIME[RUNTIME.index('def pause_due():'):RUNTIME.index('archive_lock =')]
