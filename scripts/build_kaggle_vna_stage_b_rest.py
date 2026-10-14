"""Shared pinned setup, two GPU queues, budget, archives and report for NB2/3."""
import json
from scripts.build_kaggle_vna_signal_curve_notebook import build as base_build, cell


def build(commit, number):
    base = base_build(commit)
    bootstrap = ''.join(base['cells'][2]['source']).replace('vna_phase0_seed0', f'vna_stage_b_nb{number}')
    setup = ''.join(base['cells'][3]['source'])
    old = '"tests/test_vna.py", "tests/test_vna_stage_b_upstream.py", "tests/test_vna_stage_b_losses.py"'
    tests = ['tests/test_vna.py', 'tests/test_vna_training.py', 'tests/test_vna_stage_b12.py',
             'tests/test_vna_stage_b4.py', 'tests/test_vna_stage_b_losses.py']
    if number == 2: tests.append('tests/test_vna_stage_b3.py')
    if old not in setup: raise RuntimeError('Inspect changed base test cell')
    setup = setup.replace(old, ', '.join(json.dumps(s) for s in tests))
    if number == 2:
        lanes = [[{'name': 'b3', 'module': 'experiments.stage_b3', 'args': [], 'epochs': 1730,
                   'points': 40, 'output_factor': 5}],
                 [{'name': 'b2_resnet8x4', 'module': 'experiments.stage_b12',
                   'args': ['b2', '--student', 'resnet8x4'], 'teacher': 'resnet32x4', 'epochs': 240, 'points': 15},
                  {'name': 'b4', 'module': 'experiments.stage_b4', 'args': ['train'],
                   'measure_args': ['measure'], 'epochs': 200, 'points': 12}]]
    else:
        lanes = [[{'name': 'b1_resnet8x4', 'module': 'experiments.stage_b12',
                   'args': ['b1', '--student', 'resnet8x4'], 'epochs': 240, 'points': 15},
                  {'name': 'b1_wrn_16_2', 'module': 'experiments.stage_b12',
                   'args': ['b1', '--student', 'wrn_16_2'], 'epochs': 240, 'points': 15}],
                 [{'name': 'b2_wrn_16_2', 'module': 'experiments.stage_b12',
                   'args': ['b2', '--student', 'wrn_16_2'], 'teacher': 'wrn_40_2', 'epochs': 240, 'points': 15}]]
    return {'cells': [cell('markdown', f'''# VNA Stage B — Notebook {number}
T4×2, Internet and `github_token`; attach CIFAR-100 and prior Stage-B archives.
Two GPU queues run the registered settings; no correction is applied.
Teachers are fetched, hash-checked and gated before B2. Real-data two-epoch
smokes include all measurement sites/ranks/nulls and feed the total GPU-hour
budget. Hourly archives retain logs and B4 checkpoints. B1/B2/B3 restart after
an interrupted training run; completed summaries are reused. B4 resumes its
latest recipe checkpoint. The pin must include these files: since implementation
is uncommitted, regenerate with the eventual published commit before Kaggle.
'''), cell('code', f'''REPO_COMMIT = "{commit}"
GPU_HOURS_BUDGET = 8
ALLOW_OVER_BUDGET = False
MU_REL_DEFAULT = 1e-2
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
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
       "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}
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
    args = job["measure_args"] if measure else job["args"]
    cmd = [sys.executable, "-u", "-m", job["module"], *args, "--out", str(out),
           "--device", "cuda:0", "--workers", "4", "--data-dir", str(DATA_ROOT),
           "--mu-rel-default", str(MU_REL_DEFAULT)]
    if not smoke and job["name"] == "b4" and (out / "recipe/latest.pt").exists() and not measure:
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
                    "--data-dir", str(DATA_ROOT), "--device", "cuda:0", "--gate-out", str(gate_path)],
                   cwd=REPO, env=gpu_env(gpu), check=True)
archiver = threading.Thread(target=periodic_archive, daemon=True); archiver.start()
'''

SMOKE = '''def smoke_lane(gpu, jobs):
    costs = []
    for job in jobs:
        if (STAGE_B / job["name"] / "summary.json").exists(): continue
        started = time.monotonic()
        prepare_teacher(job, gpu)
        smoke_out = RUN_ROOT / "smoke" / "outputs" / "stageB" / job["name"]
        subprocess.run(command(job, smoke_out, smoke=True), cwd=REPO, env=gpu_env(gpu), check=True)
        summary = json.loads((smoke_out / "summary.json").read_text())
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
    projections = [cost for lane in projections for cost in lane]
    total = sum(c["projected_gpu_hours"] + c["smoke_gpu_hours"] for c in projections)
    (STAGE_B / "cost_projection.json").write_text(json.dumps({"total_gpu_hours": total, "settings": projections}, indent=2))
    archive_outputs()
    if total > GPU_HOURS_BUDGET and not ALLOW_OVER_BUDGET:
        raise RuntimeError("Projected total GPU-hours exceed budget; review before ALLOW_OVER_BUDGET")
except Exception:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs(); raise
'''

TRAIN = '''def train_lane(gpu, jobs):
    for job in jobs:
        out = STAGE_B / job["name"]
        if (out / "summary.json").exists():
            print("Reusing completed", job["name"], flush=True); continue
        prepare_teacher(job, gpu)
        with (RUN_ROOT / f'{job["name"]}_gpu{gpu}.log').open("a", buffering=1) as log:
            subprocess.run(command(job, out), cwd=REPO, env=gpu_env(gpu), check=True,
                           stdout=log, stderr=subprocess.STDOUT)
            if "measure_args" in job:
                subprocess.run(command(job, out, measure=True), cwd=REPO, env=gpu_env(gpu), check=True,
                               stdout=log, stderr=subprocess.STDOUT)
        archive_outputs()
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: train_lane(*item), enumerate(LANES)))
finally:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs()
if archive_errors: raise RuntimeError(archive_errors)
'''

REPORT = '''import pandas as pd
import matplotlib.pyplot as plt
teacher = STAGE_B / "teacher_gate.json"
print("TEACHER GATE", teacher.read_text() if teacher.exists() else "no teachers in this archive")
gates = {}; reproduction = []
for folder in sorted(STAGE_B.iterdir()):
    if not folder.is_dir() or not (folder / "gate.json").exists(): continue
    gate = json.loads((folder / "gate.json").read_text()); gates[folder.name] = gate
    print("GATE G", folder.name, gate)
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
display(pd.DataFrame(reproduction))
archive_outputs()
'''
