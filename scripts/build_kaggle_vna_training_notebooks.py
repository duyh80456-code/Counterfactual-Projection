"""Generate seed 0/1 Stage 2a Kaggle notebooks with the Phase-0 bootstrap."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_kaggle_vna_signal_curve_notebook import build as phase0_build, cell, ROOT


def build(commit,seed):
    base=phase0_build(commit)
    bootstrap=''.join(base['cells'][2]['source']).replace('vna_phase0_seed0',f'vna_training_seed{seed}')
    setup=''.join(base['cells'][3]['source'])
    setup=setup.replace('experiments/vna_kaggle.py','experiments/run_vna_training.py').replace("supervisor's committed Stage 1C hash","committed Stage 2a hash")
    return {**base,'cells':[
        cell('markdown',f'''# VNA Stage 2a — seed {seed}, T4×2
Enable Internet, T4×2 and `github_token`. Attach official CIFAR-100 train/test/meta.
Attach Phase-0 checkpoints at epochs 40, 100, 160 to enable validation by default.
Attach prior Stage 2a directory/zip/tar.gz outputs to resume. Only latest per arm
is retained; hourly atomic archives enforce a conservative 5 GB peak budget.
Each arm runs in a fresh process. Final official test metrics use the final model.
**This pin must include Stage 2a.** After the supervisor commits these changes,
regenerate with `python scripts/build_kaggle_vna_training_notebooks.py --commit FULL_HASH`.
The builder never commits. T4 runtime must be measured by Cell A.
'''),
        cell('code',f'''REPO_COMMIT = "{commit}"
SEED = {seed}
SOLVER_OVERRIDE = None  # exact / kfac / lite wins over validation
VALIDATE_SOLVERS = None  # None = ON when Phase-0 checkpoint attached
GPU_DATA = True
CHANNELS_LAST = True
CUDNN_BENCHMARK = True
AMP = False
'''),cell('code',bootstrap),cell('code',setup),
        cell('code','''# Cell A: discover model-only recipe checkpoints, including archive members.
from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.vna_training_kaggle import choose_solver, restore_training, archive_training
from experiments.run_vna_training import parser as training_parser, training_protocol
import shutil
validation_cache = RUN_ROOT / "_phase0_cache"
attached, rejected = discover_checkpoints(Path("/kaggle/input"), validation_cache,
    kind="vna_recipe", payload_transform=lambda p: {"epoch": p["epoch"]})
checkpoints = {epoch: next((p["path"] for p in attached if p["payload"]["epoch"] == epoch), None)
               for epoch in (40, 100, 160)}
enabled = bool(attached) if VALIDATE_SOLVERS is None else VALIDATE_SOLVERS
validation_rows = []
try:
    if enabled:
        missing = [e for e, path in checkpoints.items() if path is None]
        if missing: raise RuntimeError(f"Validation needs Phase-0 epochs {missing}; attach them or explicitly disable Cell A")
        for epoch, path in checkpoints.items():
            destination = RUN_ROOT / "validation" / f"epoch_{epoch:03d}"
            command = [sys.executable, "-u", "-m", "diagnostics.vna_solver_validation",
                "--mode", "recipe", "--checkpoint", str(path), "--data-root", str(DATA_ROOT),
                "--device", "cuda:0", "--seed", str(SEED), "--output", str(destination)]
            subprocess.run(command, cwd=REPO, env=test_env, check=True)
            validation_rows += json.loads((destination / "summary.json").read_text())["rows"]
    SOLVER = choose_solver(validation_rows, SOLVER_OVERRIDE)
    import pandas as pd
    from IPython.display import display
    if validation_rows:
        display(pd.DataFrame(validation_rows)[["epoch", "site", "relative_mu", "variant", "cosine",
            "cos_W1", "cos_W2", "cos_BN", "realized_gain", "sgd_units", "cg_converged"]])
    print("SOLVER decision:", SOLVER, "override:", SOLVER_OVERRIDE,
          "rule: median KFAC cosine >= 0.7; otherwise exact median cost <= 30; otherwise lite")
    if not enabled: print("Validation disabled/unavailable: conservative exact default unless overridden")
finally:
    shutil.rmtree(validation_cache, ignore_errors=True)
'''),
        cell('code','''# Cell B: priority queue over two GPUs, one fresh process per arm.
import threading
from collections import deque
arms = [("sgd", "sgd", "tiny_V"), ("vna_tiny_V", "vna", "tiny_V"),
        ("ngd", "ngd", "tiny_V"), ("vna_random", "vna_random", "tiny_V"),
        ("vna_tiny_U", "vna", "tiny_U"), ("sgd_cm", "sgd_cm", "tiny_V")]
flags = [flag for enabled, flag in [(GPU_DATA, "--gpu-data"), (CHANNELS_LAST, "--channels-last"),
         (CUDNN_BENCHMARK, "--cudnn-benchmark"), (AMP, "--amp")] if enabled]
commands, protocols = {}, {}
for name, arm, selection in arms:
    arguments = ["--arm", arm, "--solver", SOLVER, "--selection", selection, "--seed", str(SEED),
        "--data-root", str(DATA_ROOT), "--workers", "2", "--cache-block-input", *flags]
    protocols[name] = training_protocol(training_parser().parse_args(arguments))
    commands[name] = [sys.executable, "-u", "-m", "experiments.run_vna_training", *arguments,
                      "--output", str(RUN_ROOT / name), "--budget-root", str(RUN_ROOT)]
restore_training(Path("/kaggle/input"), RUN_ROOT, protocols)
pending = deque(name for name, _, _ in arms if not (RUN_ROOT / name / "final.json").exists())
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
running, handles = {}, []
archive_stop = threading.Event(); archive_errors = []
def hourly_archive():
    while not archive_stop.wait(3600):
        try: archive_training(RUN_ROOT)
        except Exception as error: archive_errors.append(str(error)); print("Archive error:", error, flush=True)
archiver = threading.Thread(target=hourly_archive, daemon=True); archiver.start()
try:
    while pending or running:
        for gpu in (0, 1):
            if gpu not in running and pending:
                name = pending.popleft(); command = commands[name] + ["--device", f"cuda:{gpu}"]
                latest = RUN_ROOT / name / "latest.pt"
                if latest.exists(): command += ["--resume", str(latest)]
                log = (RUN_ROOT / f"{name}.log").open("a", buffering=1); handles.append(log)
                process = subprocess.Popen(command, cwd=REPO, env=ENV, stdout=log, stderr=subprocess.STDOUT)
                running[gpu] = (name, process); print("Launched", name, "GPU", gpu, flush=True)
        for gpu, (name, process) in list(running.items()):
            status = process.poll()
            if status is not None:
                if status: raise RuntimeError(f"{name} failed; see {name}.log")
                print("Completed", name, flush=True); del running[gpu]
        if running: time.sleep(30)
finally:
    archive_stop.set()
    for _, process in running.values():
        if process.poll() is None: process.terminate()
    for _, process in running.values(): process.wait()
    archiver.join()
    for log in handles: log.close()
    print("Resumable archive:", archive_training(RUN_ROOT))
if archive_errors: raise RuntimeError(f"Hourly archive failures: {archive_errors}")
'''),
        cell('code','''# Final metrics and curves, including all computation in each arm's cost.
import matplotlib.pyplot as plt
results = []
for name, _, _ in arms:
    record = json.loads((RUN_ROOT / name / "final.json").read_text())
    results.append({"arm": name, "test_acc": record["test"]["accuracy"], "test_loss": record["test"]["loss"],
        "params": record["params"], "GPU-h": record["gpu_hours"], "wall-h": record["wall_hours"],
        "VNA_seconds": record["vna_seconds"], "peak_GB": record["peak_memory_bytes"] / 1e9})
frame = pd.DataFrame(results); baseline = float(frame.loc[frame.arm == "sgd", "test_acc"].iloc[0])
frame["delta_vs_sgd_pp"] = 100 * (frame.test_acc - baseline)
assert frame.params.nunique() == 1
frame["cost_vs_sgd"] = frame["GPU-h"] / float(frame.loc[frame.arm == "sgd", "GPU-h"].iloc[0])
display(frame); frame.to_csv(RUN_ROOT / "results.csv", index=False)
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for name, _, _ in arms:
    history = json.loads((RUN_ROOT / name / "history.json").read_text())
    epochs = [r["epoch"] for r in history]
    for ax, split, metric in [(axes[0], "train", "loss"), (axes[1], "eval", "accuracy")]:
        ax.plot(epochs, [r[split][metric] for r in history], label=name)
    axes[2].plot(epochs, [r["mean_rho"] for r in history], label=name)
for ax, title in zip(axes, ["Train loss", "Eval accuracy", "rho"]):
    ax.set(title=title, xlabel="epoch"); ax.grid(alpha=.3)
axes[0].legend(fontsize=7); plt.tight_layout(); fig.savefig(RUN_ROOT / "curves.png"); plt.show()
print("Final archive:", archive_training(RUN_ROOT))
''')]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit',default=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    p.add_argument('--output-dir',default=str(ROOT/'notebooks'))
    args=p.parse_args()
    for seed in (0,1):
        path=Path(args.output_dir)/f'kaggle_vna_training_seed{seed}.ipynb'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(build(args.commit,seed),indent=1,ensure_ascii=False)+'\n'); print(path)
