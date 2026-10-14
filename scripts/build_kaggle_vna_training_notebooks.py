"""Generate three two-arm Stage 2a Kaggle notebooks per seed."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_kaggle_vna_signal_curve_notebook import build as phase0_build, cell, ROOT


PART_ARMS = {1: [("sgd", "sgd", "tiny_V"), ("vna_tiny_V", "vna", "tiny_V")],
             2: [("ngd", "ngd", "tiny_V"), ("vna_random", "vna_random", "tiny_V")],
             3: [("vna_tiny_U", "vna", "tiny_U"), ("sgd_cm", "sgd_cm", "tiny_V")]}


def build(commit=None,seed=0,part=1):
    if seed not in (0, 1) or part not in PART_ARMS: raise ValueError("invalid seed/part")
    if commit is None:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    base=phase0_build(commit)
    bootstrap=''.join(base['cells'][2]['source']).replace('vna_phase0_seed0',f'vna_seed{seed}_part{part}')
    setup=''.join(base['cells'][3]['source'])
    setup=setup.replace('"tests/test_vna.py"', '"tests/test_vna.py", "tests/test_vna_stage2.py", "tests/test_vna_training.py", "tests/test_vna_split3.py"')
    setup=setup.replace('experiments/vna_kaggle.py','experiments/run_vna_training.py')
    setup += '\nfor module in ("experiments/vna_training_support.py", "experiments/vna_training_config.py", "experiments/vna_training_kaggle.py", "diagnostics/vna_solver_validation.py", "experiments/vna_compare.py"):\n    if not (REPO / module).exists(): raise RuntimeError(f"Required module {module} is missing from the pinned commit")\n'
    return {**base,'cells':[
        cell('markdown',f'''# VNA Stage 2a — seed {seed}, part {part}, T4×2
Enable Internet, T4×2 and `github_token`. Attach official CIFAR-100 train/test/meta.
Attach Phase-0 checkpoints at epochs 40, 100, 160 to enable validation by default.
Attach prior Stage 2a directory/zip/tar.gz outputs to resume. Only latest per arm
is retained; hourly atomic archives enforce a conservative 5 GB peak budget.
Each arm runs in a fresh process. Final official test metrics use the final model.
Code is pinned to an immutable commit; each session records its resolved commit.
Regenerate with `python scripts/build_kaggle_vna_training_notebooks.py --commit FULL_HASH`
to change the pin; the default is current HEAD. The pin must include Stage 2a.
GPU_HOURS_BUDGET=15 is per notebook. Parts 1–2 raise if over budget unless
ALLOW_OVER_BUDGET=True. Only part3 drops sgd_cm, then vna_tiny_U.
Each arm has a fixed GPU (first GPU0, second GPU1).
Corrections default to spread: λ·d at each of the next VNA_EVERY steps,
including the computation step. Smoke thresholds are report-only:
p95 applied-delta/gradient > 3 → IMPULSE_RISK; median < 0.05 → LIKELY_NULL.
Cell A uses minimum batch agreement across sites/epochs: cos16 >= 0.5 →
(16,10), else cos64 >= 0.5 → (64,20), else (128,20), before solver choice.
The batch/every override constants take precedence.
T4 runtime must be measured by Cell A.
'''),
        cell('code',f'''REPO_COMMIT = "{commit}"
SEED = {seed}
PART = {part}
SOLVER_OVERRIDE = None  # exact / kfac / lite wins over validation
VALIDATE_SOLVERS = None  # None = ON when Phase-0 checkpoint attached
GPU_DATA = True
CHANNELS_LAST = True
CUDNN_BENCHMARK = True
AMP = False
VNA_BATCH_OVERRIDE = None
VNA_EVERY_OVERRIDE = None
LAMBDA_VNA = 1.0
SMOKE_EPOCHS = 2
GPU_HOURS_BUDGET = 15
ALLOW_OVER_BUDGET = False
'''),cell('code',bootstrap),cell('code',setup),
        cell('code','''# Cell A: discover model-only recipe checkpoints, including archive members.
from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.vna_training_kaggle import choose_solver, choose_batch, apply_budget_policy, smoke_diagnostics, restore_training, archive_training
from experiments.run_vna_training import parser as training_parser, training_protocol
import shutil
# Parser defaults are the shared source for all remaining VNA settings.
from experiments.vna_training_config import add_vna_arguments
import argparse
shared_parser = add_vna_arguments(argparse.ArgumentParser())
shared_defaults = vars(shared_parser.parse_args([]))
shared_defaults.update(lambda_vna=LAMBDA_VNA,
                       cache_block_input=True)
shared_arguments = []
for key, value in shared_defaults.items():
    flag = "--" + key.replace("_", "-")
    if isinstance(value, bool):
        if value: shared_arguments.append(flag)
    else: shared_arguments += [flag, str(value)]
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
                "--device", "cuda:0", "--seed", str(SEED), "--output", str(destination), *shared_arguments]
            subprocess.run(command, cwd=REPO, env=test_env, check=True)
            validation_rows += json.loads((destination / "summary.json").read_text())["rows"]
    VNA_BATCH, VNA_EVERY = choose_batch(validation_rows, VNA_BATCH_OVERRIDE, VNA_EVERY_OVERRIDE,
                                       damping_rel=shared_defaults["damping_rel"])
    shared_defaults.update(vna_batch=VNA_BATCH, vna_every=VNA_EVERY)
    # Re-measure solver cost/fidelity at the pre-registered chosen batch.
    validation_rows = []
    shared_arguments = []
    for key, value in shared_defaults.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value: shared_arguments.append(flag)
        else: shared_arguments += [flag, str(value)]
    if enabled:
        for epoch, path in checkpoints.items():
            destination = RUN_ROOT / "validation_chosen" / f"epoch_{epoch:03d}"
            command = [sys.executable, "-u", "-m", "diagnostics.vna_solver_validation",
                "--mode", "recipe", "--checkpoint", str(path), "--data-root", str(DATA_ROOT),
                "--device", "cuda:0", "--seed", str(SEED), "--output", str(destination), *shared_arguments]
            subprocess.run(command, cwd=REPO, env=test_env, check=True)
            validation_rows += json.loads((destination / "summary.json").read_text())["rows"]
    SOLVER = choose_solver(validation_rows, SOLVER_OVERRIDE, damping_rel=shared_defaults["damping_rel"], vna_every=VNA_EVERY)
    import pandas as pd
    from IPython.display import display
    if validation_rows:
        display(pd.DataFrame(validation_rows)[["epoch", "site", "relative_mu", "variant", "cosine",
            "cos_W1", "cos_W2", "cos_BN", "realized_gain", "sgd_units", "cg_converged"]])
    print("SOLVER decision:", SOLVER, "override:", SOLVER_OVERRIDE,
          "rule: minimum KFAC cosine >= 0.9 with positive gains; otherwise affordable solver by fidelity")
    print("Training VNA config:", shared_defaults)
finally:
    shutil.rmtree(validation_cache, ignore_errors=True)
'''),
        cell('code', 'PART_ARMS = ' + repr(PART_ARMS) + '\n' + '''# Cell B: fixed arm pair, one fresh process per GPU.
import threading
from collections import deque
arms = PART_ARMS[PART]
gpu_by_arm = {name: gpu for gpu, (name, _, _) in enumerate(arms)}
flags = [flag for enabled, flag in [(GPU_DATA, "--gpu-data"), (CHANNELS_LAST, "--channels-last"),
         (CUDNN_BENCHMARK, "--cudnn-benchmark"), (AMP, "--amp")] if enabled]
from experiments.vna_compare import save_run_config
run_config = dict(shared_defaults, seed=SEED, solver=SOLVER, code_commit=REPO_COMMIT,
                  gpu_data=GPU_DATA, channels_last=CHANNELS_LAST,
                  cudnn_benchmark=CUDNN_BENCHMARK, amp=AMP)
config_hash = save_run_config(RUN_ROOT / "run_config.json", run_config)
print("config_hash:", config_hash)
commands, protocols = {}, {}
for name, arm, selection in arms:
    arguments = ["--arm", arm, "--solver", SOLVER, "--selection", selection, "--seed", str(SEED),
        "--data-root", str(DATA_ROOT), "--workers", "2", *shared_arguments,
        "--selection", selection, *flags]
    protocols[name] = training_protocol(training_parser().parse_args(arguments))
    commands[name] = [sys.executable, "-u", "-m", "experiments.run_vna_training", *arguments,
                      "--output", str(RUN_ROOT / name), "--budget-root", str(RUN_ROOT),
                      "--run-config", str(RUN_ROOT / "run_config.json")]
# Smoke runs use isolated outputs and activate corrections in epoch 1 so that
# epoch 2 measures the active arm, including scheduled factor refresh.
SMOKE_ROOT = RUN_ROOT.parent / f"{RUN_ROOT.name}_smoke"
shutil.rmtree(SMOKE_ROOT, ignore_errors=True)
SMOKE_ROOT.mkdir(parents=True)
ENV = {**test_env, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
projections = {}
for name, _, _ in arms:
    smoke_command = commands[name][:-2] + ["--output", str(SMOKE_ROOT / name),
        "--budget-root", str(SMOKE_ROOT), "--max-epochs", str(SMOKE_EPOCHS),
        "--vna-start-epoch", "1", "--device", f"cuda:{gpu_by_arm[name]}"]
    subprocess.run(smoke_command, cwd=REPO, env=ENV, check=True)
    smoke_history = json.loads((SMOKE_ROOT / name / "history.json").read_text())
    seconds = next(row["epoch_seconds"] for row in smoke_history if row["epoch"] == 2)
    projections[name] = seconds * (260 if name == "sgd_cm" else 200) / 3600
    print(name, "epoch-2 seconds:", seconds, "projected GPU-hours:", projections[name])
    if name in ("vna_tiny_V", "ngd", "vna_random", "vna_tiny_U"):
        print(name, "smoke diagnostics:", smoke_diagnostics(SMOKE_ROOT / name))
arms, projected_total = apply_budget_policy(arms, projections, GPU_HOURS_BUDGET, ALLOW_OVER_BUDGET, part=PART)
shutil.rmtree(SMOKE_ROOT, ignore_errors=True)
restore_training(Path("/kaggle/input"), RUN_ROOT, {name: protocols[name] for name, _, _ in arms}, config_hash=config_hash)
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
            name = next((n for n in pending if gpu_by_arm[n] == gpu), None)
            if gpu not in running and name is not None:
                pending.remove(name); command = commands[name] + ["--device", f"cuda:{gpu}"]
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
        cell('code','''# Compare finished arms from every part and attached archives.
from experiments.vna_compare import compare
frame, missing, fig = compare(RUN_ROOT.parent, SEED, input_root=Path("/kaggle/input"), output=RUN_ROOT)
display(frame)
print("Missing arms:", missing)
plt = __import__("matplotlib.pyplot", fromlist=["show"])
plt.show()
print("Final archive:", archive_training(RUN_ROOT))
''')]}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit',default=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    p.add_argument('--output-dir',default=str(ROOT/'notebooks'))
    args=p.parse_args()
    for seed in (0,1):
        for part in (1,2,3):
            path=Path(args.output_dir)/f'kaggle_vna_seed{seed}_part{part}.ipynb'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(build(args.commit,seed,part),indent=1,ensure_ascii=False)+'\n'); print(path)
