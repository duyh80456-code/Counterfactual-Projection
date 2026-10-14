"""Build the immutable-pinned, dual-GPU Stage-A checkpoint notebook."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_kaggle_vna_signal_curve_notebook import build as phase0_build, cell


def build(commit=None):
    base = phase0_build(commit)
    constants = ''.join(base['cells'][1]['source']).split('GPU_DATA =')[0]
    constants += '''GPU_HOURS_BUDGET = 3
ALLOW_OVER_BUDGET = False
PHASE0_DIR = Path("/kaggle/input/vna-phase0-seed0/vna_phase0_seed0")  # extracted tar.gz; recipe/epoch_*.pt
'''
    # Path is imported explicitly so the configuration cell runs standalone.
    constants = 'from pathlib import Path\n' + constants
    bootstrap = ''.join(base['cells'][2]['source']).replace('/kaggle/working/vna_phase0_seed0', '/kaggle/working/vna_stage_a')
    setup = ''.join(base['cells'][3]['source']).replace('"tests/test_vna.py"', '*sorted(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_vna*.py"))')
    setup += '''\nfor module in ("diagnostics/stage_a.py", "methods/vna_joint.py", "diagnostics/vna_gate.py"):
    if not (REPO / module).exists(): raise RuntimeError(f"Missing Stage-A module: {module}")
'''
    cells = [cell('markdown', '''# VNA Stage A — Phase-0 checkpoints, T4×2
Attach the extracted `vna_phase0_seed0.tar.gz` and CIFAR-100. Set PHASE0_DIR to
its root (containing recipe/epoch_*.pt). Enable Internet, T4×2, github_token.
Only diagnostics run. All VNA tests run first on CPU. Immutable REPO_COMMIT must
include Stage A; regenerate with --commit after making the code available remotely.
GPU0 runs A5 first, then the two-checkpoint cost smoke, then A1 and A3.
The smoke prints projected GPU-hours and rejects totals above 3 unless explicitly allowed. GPU1 runs A4 then A2 then A6. Completion markers resume
per measurement/epoch with schema/config/checkpoint-hash validation. Hourly
archives use the existing notebook disk cap. Attach prior Stage-A outputs to resume.
'''), cell('code', constants), cell('code', bootstrap), cell('code', setup),
        cell('code', '''from experiments.vna_kaggle import archive_outputs as _archive_outputs
from diagnostics.stage_a import restore_archives
import shutil
def archive_outputs(output):
    return _archive_outputs(output, folders=("outputs/stageA",))
STAGE_A = RUN_ROOT / "outputs" / "stageA"
STAGE_A.mkdir(parents=True, exist_ok=True)
RECIPE = PHASE0_DIR / "recipe"
if not list(RECIPE.glob("epoch_*.pt")):
    raise RuntimeError(f"No extracted Phase-0 checkpoints in {RECIPE}")
# Restore Stage-A directory and tar/zip outputs from prior sessions.
restore_archives(Path("/kaggle/input"), STAGE_A)
common = ["--checkpoint-dir", str(RECIPE), "--data-root", str(DATA_ROOT),
          "--output", str(STAGE_A), "--seed", "0"]
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
archive_stop = threading.Event()
archive_errors = []
def periodic_archive():
    while not archive_stop.wait(3600):
        try: archive_outputs(RUN_ROOT)
        except Exception as error:
            archive_errors.append(str(error)); print("Archive error:", error, flush=True)
archiver = threading.Thread(target=periodic_archive, daemon=True)
archiver.start()
'''), cell('code', '''try:
    # A5 is the first real-checkpoint diagnostic, on GPU0.
    subprocess.run([sys.executable, "-u", "-m", "diagnostics.stage_a", "a5", *common,
                    "--device", "cuda:0"], cwd=REPO, env=ENV, check=True)
    # Two distinct real checkpoints, full tensor shapes and production CG settings.
    cost_paths = [RECIPE / "epoch_090.pt", RECIPE / "epoch_100.pt"]
    if not all(p.exists() for p in cost_paths):
        raise RuntimeError("Cost smoke requires recipe/epoch_090.pt and epoch_100.pt")
    subprocess.run([sys.executable, "-u", "-m", "diagnostics.stage_a", "cost", *common,
                    "--checkpoints", *map(str, cost_paths), "--device", "cuda:0"],
                   cwd=REPO, env=ENV, check=True)
    cost = json.loads((STAGE_A / "cost_projection.json").read_text())
    for name, record in cost["cost_projection"].items():
        print(f"{name}: {record['projected_gpu_hours']:.3f} projected GPU-hours")
    projected_total = cost["total_projected_gpu_hours"] + cost["smoke_gpu_hours"]
    print(f"Total (including smoke): {projected_total:.3f} GPU-hours; budget: {GPU_HOURS_BUDGET}")
    archive_outputs(RUN_ROOT)
    if projected_total > GPU_HOURS_BUDGET and not ALLOW_OVER_BUDGET:
        raise RuntimeError("Stage A exceeds GPU_HOURS_BUDGET; review cost_projection.json or explicitly set ALLOW_OVER_BUDGET")
except Exception:
    # Stop/archive even when preflight or its budget guard rejects the run.
    if 'archive_stop' in globals():
        archive_stop.set(); archiver.join(timeout=10)
    archive_outputs(RUN_ROOT)
    raise
'''), cell('code', '''from concurrent.futures import ThreadPoolExecutor
def worker(gpu, measurements):
    for measurement in measurements:
        command = [sys.executable, "-u", "-m", "diagnostics.stage_a", measurement,
                   *common, "--device", f"cuda:{gpu}"]
        with (RUN_ROOT / f"{measurement}_gpu{gpu}.log").open("a", buffering=1) as log:
            subprocess.run(command, cwd=REPO, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Finished {measurement} on GPU{gpu}", flush=True)
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [pool.submit(worker, 0, ["null", "selection"]),
                   pool.submit(worker, 1, ["joint", "mu", "solver"])]
        for worker_result in workers: worker_result.result()
finally:
    archive_stop.set(); archiver.join(timeout=10)
    archive_outputs(RUN_ROOT)
if archive_errors: raise RuntimeError(archive_errors)
'''), cell('code', '''subprocess.run([sys.executable, "-m", "diagnostics.stage_a", "summary", *common,
                "--device", "cuda:0"], cwd=REPO, env=ENV, check=True)
import pandas as pd
import matplotlib.pyplot as plt
print((STAGE_A / "summary.json").read_text())
for filename in ("a5.csv", "null.csv", "mu_sweep.csv", "selection.csv", "joint.csv", "solver.csv"):
    print(filename)
    display(pd.read_csv(STAGE_A / filename))
mu = pd.read_csv(STAGE_A / "mu_sweep.csv")
fig, ax = plt.subplots()
for (epoch, site), group in mu.groupby(["epoch", "site"]):
    ax.plot(group.mu_rel, group.rho, marker="o", label=f"{epoch} {site}")
ax.set_xscale("log"); ax.set(xlabel="mu_rel", ylabel="rho")
fig.savefig(STAGE_A / "rho_mu.png", bbox_inches="tight"); plt.show()
selection = pd.read_csv(STAGE_A / "selection.csv")
selection.groupby(["rank", "kind"]).V.mean().unstack().plot.bar(ylabel="V")
plt.tight_layout(); plt.savefig(STAGE_A / "selection_V.png"); plt.show()
null = pd.read_csv(STAGE_A / "null.csv")
null.boxplot(column="p_value", by=["null_kind", "stat"])
plt.axhline(1/9, color="red"); plt.tight_layout()
plt.savefig(STAGE_A / "null_p_values.png"); plt.show()
archive_outputs(RUN_ROOT)
print("Archive:", RUN_ROOT.with_suffix(".tar.gz"), "includes outputs/stageA/")
''')]
    return {'cells': cells, 'metadata': base['metadata'], 'nbformat': 4, 'nbformat_minor': 5}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit')
    p.add_argument('--output', default=str(ROOT / 'notebooks/kaggle_vna_stage_a.ipynb'))
    args = p.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(args.commit), indent=1) + '\n')


if __name__ == '__main__':
    main()
