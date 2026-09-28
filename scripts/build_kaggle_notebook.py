"""Generate the committed Kaggle T4x2 notebook from readable cell sources."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


cells = [
    markdown("""# Counterfactual Projection — CIFAR-100 T4x2 pilot

This focused notebook runs only `tiny_projection` for 3 post-warm-up epochs on
the two existing seeds. It reuses the shared fair-v15 warm-up model/optimizer
checkpoints and does not rerun baselines or touch the official CIFAR-100 test.
The fixed configuration is TINY/Gromo at `stages.2.blocks.0`, rank 4, epsilon
0.05, with residual-path functional projection. Its sole question is whether
an E-driven correction that is actually applied improves validation accuracy.
"""),
    code("""import json, os, queue, shutil, subprocess, sys, threading
from pathlib import Path
from kaggle_secrets import UserSecretsClient

REPO_URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
BRANCH = "main"
REPO = Path("/kaggle/working/counterfactual-projection")
REFERENCE_URL = "https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git"
REFERENCE_BRANCH = "ccil-residual-capacity"
REFERENCE = Path("/kaggle/working/One-Shot-TAS-CCIL")
GROMO_URL = "https://github.com/growingnet/gromo.git"
GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"
GROMO = Path("/kaggle/working/gromo")
OUTPUT = Path("/kaggle/working/counterfactual_projection_t4x2_fair_v15")

for checkout in (REPO, REFERENCE, GROMO):
    if checkout.exists(): shutil.rmtree(checkout)
token = UserSecretsClient().get_secret("github_token").strip()
if not token:
    raise RuntimeError("Kaggle Secret github_token is empty or unavailable")
askpass = Path("/kaggle/working/.counterfactual_git_askpass.py")
askpass.write_text(
    "#!/usr/bin/env python3\\n"
    "import os, sys\\n"
    "prompt = sys.argv[1] if len(sys.argv) > 1 else ''\\n"
    "print('x-access-token' if 'Username' in prompt else os.environ['GITHUB_TOKEN_RUNTIME'])\\n")
askpass.chmod(0o700)
clone_env = os.environ.copy()
clone_env.update(GITHUB_TOKEN_RUNTIME=token, GIT_ASKPASS=str(askpass),
                 GIT_TERMINAL_PROMPT="0")
try:
    subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch",
                    REPO_URL, str(REPO)], env=clone_env, check=True)
    subprocess.run(["git", "clone", "--branch", REFERENCE_BRANCH,
                    "--single-branch", REFERENCE_URL, str(REFERENCE)],
                   env=clone_env, check=True)
finally:
    askpass.unlink(missing_ok=True)
    clone_env.pop("GITHUB_TOKEN_RUNTIME", None)
    token = None

commit = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
print("Repository revision:", commit)
OUTPUT.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)],
               check=True)
subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                GROMO_URL, str(GROMO)], check=True)
subprocess.run(["git", "-C", str(GROMO), "fetch", "--depth", "1", "origin",
                GROMO_COMMIT], check=True)
subprocess.run(["git", "-C", str(GROMO), "checkout", "--detach", GROMO_COMMIT],
               check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(GROMO)],
               check=True)
GROMO_SRC = GROMO / "src"
RUNTIME_PYTHONPATH = os.pathsep.join(filter(None, (
    str(REPO), str(GROMO_SRC), str(REFERENCE), os.environ.get("PYTHONPATH", ""))))
test_env = os.environ.copy()
test_env.update(
    PYTHONPATH=RUNTIME_PYTHONPATH,
    REQUIRE_GROMO_INTEGRATION="1")
subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO,
               env=test_env, check=True)
"""),
    code("""import torch

gpu_count = torch.cuda.device_count()
print(subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,name,memory.total",
     "--format=csv,noheader"], text=True))
if gpu_count != 2:
    raise RuntimeError(f"Select the Kaggle T4 x2 accelerator; found {gpu_count} GPU(s)")
# Populate the shared weight cache and verify torchvision/Gromo parity before
# launching two independent processes.
# Editable installs happened after this notebook kernel started, so their new
# .pth files are only discovered by child interpreters. Pin all three import
# roots explicitly. Purging gromo also makes rerunning this cell safe after a
# prior failed import cached an unrelated top-level package of the same name.
for root in reversed((REPO, GROMO_SRC, REFERENCE)):
    root_text = str(root)
    while root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
for module_name in tuple(sys.modules):
    if any(module_name == package or module_name.startswith(package + ".")
           for package in ("probe", "gromo", "dual_growth")):
        del sys.modules[module_name]

import dual_growth, gromo, probe

def assert_import_root(module, expected_root):
    origin = Path(module.__file__).resolve()
    expected = expected_root.resolve()
    if not origin.is_relative_to(expected):
        raise ImportError(
            f"{module.__name__} resolved to {origin}, expected under {expected}")
    print(f"{module.__name__}: {origin}")

assert_import_root(probe, REPO)
assert_import_root(gromo, GROMO_SRC)
assert_import_root(dual_growth, REFERENCE)
from gromo.containers.resnet import init_full_resnet_structure  # noqa: F401
from dual_growth.adapters import GromoResNet18, TinyAdapter  # noqa: F401
from probe import build_pretrained_gromo_resnet18
smoke_model = build_pretrained_gromo_resnet18(100, device="cuda:0")
assert [int(ref.module.hidden_neurons) for ref in smoke_model.growing_blocks()] == [
    64, 64, 128, 128, 256, 256, 512, 512]
del smoke_model
torch.cuda.empty_cache()

input_root = Path("/kaggle/input")
cifar_dirs = sorted({p.parent.resolve() for p in input_root.rglob("cifar-100-python")})
if not cifar_dirs:
    raise FileNotFoundError(
        "Attach a Kaggle CIFAR-100 dataset containing cifar-100-python")
DATA_ROOT = cifar_dirs[0]
print("CIFAR-100 root:", DATA_ROOT)
"""),
    code("""# Focused E-driven O run: no epsilon sweep and no baseline reruns.
METHODS = ["tiny_projection"]
CG_METHODS = {"tiny_projection"}
PROBE_METHODS = {"tiny_projection"}
APPLY_METHODS = {"tiny_projection"}
E_MATCHED_METHODS = {"tiny_projection"}
SEEDS = [0, 1]
EPOCHS = 3
WARMUP_EPOCHS = 3
BATCH_SIZE = 64
TRAIN_SAMPLES = 12000       # Set 0 for all non-validation training examples.
VALIDATION_SAMPLES = 5000
SITE = "stages.2.blocks.0"
CANDIDATE_SITES = ""  # Used only when SITE="auto"; empty means all blocks.
RANK = 4
CG_ITERATIONS = 200
CG_RELATIVE_TOLERANCE = 1e-2
CG_PRECONDITIONER_PROBES = 8
SOLVER_REVISION = "e-driven-o-best-functional-fit-v1"
APPLICATION_MAX_HELDOUT_RESIDUAL = 1.0
APPLICATION_MIN_HELDOUT_COSINE = 0.0
IMAGE_SIZE = 128
STATISTICS_SAMPLES = 256
PROJECTION_SAMPLES = 32
TUNING_SAMPLES = 128
EPSILONS = [0.05]

jobs = []
for seed in SEEDS:
    for method in METHODS:
        jobs.append((method, seed, 0.05))
print(f"Scheduled only {len(jobs)} focused tiny_projection arms")
"""),
    code("""# Create exactly one warm-up checkpoint per seed. All method arms load
# both model weights and SGD momentum from this shared artifact.
WARMUP_ROOT = OUTPUT / "warmup"
WARMUP_ROOT.mkdir(parents=True, exist_ok=True)
warmup_processes = []
WARMUP_CHECKPOINTS = {}
for gpu, seed in enumerate(SEEDS):
    checkpoint = WARMUP_ROOT / f"seed{seed}.pt"
    WARMUP_CHECKPOINTS[seed] = checkpoint
    warmup_output = WARMUP_ROOT / f"seed{seed}_manifest"
    command = [
        sys.executable, "-m", "experiments.run_gromo_pilot",
        "--method", "vanilla", "--seed", str(seed),
        "--prepare-warmup", "--warmup-epochs", str(WARMUP_EPOCHS),
        "--warmup-checkpoint", str(checkpoint),
        "--batch-size", str(BATCH_SIZE),
        "--reference-root", str(REFERENCE),
        "--train-samples", str(TRAIN_SAMPLES),
        "--validation-samples", str(VALIDATION_SAMPLES),
        "--statistics-samples", str(STATISTICS_SAMPLES),
        "--projection-samples", str(PROJECTION_SAMPLES),
        "--tuning-samples", str(TUNING_SAMPLES),
        "--image-size", str(IMAGE_SIZE),
        "--data-root", str(DATA_ROOT), "--output", str(warmup_output),
    ]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="2", PYTHONPATH=RUNTIME_PYTHONPATH)
    warmup_output.mkdir(parents=True, exist_ok=True)
    log = (warmup_output / "run.log").open("a")
    process = subprocess.Popen(
        command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    warmup_processes.append((seed, process, log))

for seed, process, log in warmup_processes:
    return_code = process.wait()
    log.close()
    if return_code:
        raise RuntimeError(f"Warm-up failed for seed {seed}: exit {return_code}")
print("Shared warm-up checkpoints:", WARMUP_CHECKPOINTS)
"""),
    code("""# Dynamic two-worker queue: each GPU immediately picks up the next arm.
# This is faster for independent ablations than synchronizing both T4s with DDP.
# Reuse the expensive shared warm-up checkpoints from fair_v15, but archive
# completed CG-arm results produced by the old 1e-5 unpreconditioned solver.
def solver_config_is_current(result):
    config = result.get("config", {})
    return (
        config.get("solver_revision") == SOLVER_REVISION and
        config.get("cg_relative_tolerance") == CG_RELATIVE_TOLERANCE and
        config.get("cg_preconditioner_probes") == CG_PRECONDITIONER_PROBES and
        config.get("application_max_heldout_residual") ==
            APPLICATION_MAX_HELDOUT_RESIDUAL and
        config.get("application_min_heldout_cosine") ==
            APPLICATION_MIN_HELDOUT_COSINE)

for method, seed, epsilon in jobs:
    if method not in CG_METHODS:
        continue
    epsilon_label = str(epsilon).replace(".", "p")
    arm_dir = OUTPUT / f"{method}_eps{epsilon_label}_seed{seed}"
    result_path = arm_dir / "result.json"
    if not result_path.is_file():
        continue
    previous = json.loads(result_path.read_text())
    if not solver_config_is_current(previous):
        archived = arm_dir / "result.pre_e_driven_o.json"
        if archived.exists():
            archived.unlink()
        result_path.rename(archived)
        print(f"[{arm_dir.name}] archived stale solver result; reusing warm-up")

job_queue = queue.Queue()
for job in jobs:
    job_queue.put(job)
failures = []
lock = threading.Lock()

def run_worker(gpu):
    while True:
        try:
            method, seed, epsilon = job_queue.get_nowait()
        except queue.Empty:
            return
        epsilon_label = str(epsilon).replace(".", "p")
        label = f"{method}_eps{epsilon_label}_seed{seed}"
        arm_dir = OUTPUT / label
        result = arm_dir / "result.json"
        if result.is_file():
            print(f"[{label}] completed; skipping", flush=True)
            job_queue.task_done()
            continue
        command = [
            sys.executable, "-m", "experiments.run_gromo_pilot",
            "--method", method, "--seed", str(seed),
            "--epochs", str(EPOCHS), "--batch-size", str(BATCH_SIZE),
            "--warmup-epochs", str(WARMUP_EPOCHS),
            "--warmup-checkpoint", str(WARMUP_CHECKPOINTS[seed]),
            "--reference-root", str(REFERENCE),
            "--train-samples", str(TRAIN_SAMPLES),
            "--validation-samples", str(VALIDATION_SAMPLES),
            "--statistics-samples", str(STATISTICS_SAMPLES),
            "--projection-samples", str(PROJECTION_SAMPLES),
            "--tuning-samples", str(TUNING_SAMPLES),
            "--image-size", str(IMAGE_SIZE),
            "--site", SITE, "--rank", str(RANK),
            "--candidate-sites", CANDIDATE_SITES,
            "--probe-epsilon", str(epsilon),
            "--cg-iterations", str(CG_ITERATIONS),
            "--cg-relative-tolerance", str(CG_RELATIVE_TOLERANCE),
            "--cg-preconditioner-probes", str(CG_PRECONDITIONER_PROBES),
            "--solver-revision", SOLVER_REVISION,
            "--application-max-heldout-residual",
            str(APPLICATION_MAX_HELDOUT_RESIDUAL),
            "--application-min-heldout-cosine",
            str(APPLICATION_MIN_HELDOUT_COSINE),
            "--data-root", str(DATA_ROOT), "--output", str(arm_dir),
        ]
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
                   PYTHONPATH=RUNTIME_PYTHONPATH)
        arm_dir.mkdir(parents=True, exist_ok=True)
        with (arm_dir / "run.log").open("a") as log:
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush()
                print(f"[GPU{gpu}:{label}] {line}", end="", flush=True)
            code_value = process.wait()
        if code_value:
            with lock:
                failures.append((label, code_value))
        job_queue.task_done()

workers = [threading.Thread(target=run_worker, args=(gpu,), daemon=True)
           for gpu in range(2)]
for worker in workers: worker.start()
for worker in workers: worker.join()
if failures:
    raise RuntimeError(f"Failed arms: {failures}")
print("All arms completed")
"""),
    code("""import statistics

rows = []
for method, seed, epsilon in jobs:
    epsilon_label = str(epsilon).replace(".", "p")
    path = OUTPUT / f"{method}_eps{epsilon_label}_seed{seed}" / "result.json"
    result = json.loads(path.read_text())
    expected_growth = method == "real_e_growth"
    if (result["deploy_parameter_delta"] > 0) != expected_growth:
        raise RuntimeError(f"Deploy-size invariant failed: {path}")
    if result["official_test_accuracy"] is not None:
        raise RuntimeError(f"development arm touched official test: {path}")
    if method in APPLY_METHODS and result["correction_application_rate"] != 1.0:
        raise RuntimeError(
            f"incomplete correction application rate for {path}: "
            f"{result['corrections_applied']}/{result['correction_attempts']}")
    required = {
        "correction_applied", "parameter_delta_norm",
        "actual_cosine_alignment", "actual_relative_residual",
        "loss_before", "loss_after", "validation_accuracy"}
    for epoch in result["history"]:
        missing = required - set(epoch["diagnostics"])
        if missing:
            raise RuntimeError(
                f"missing E-driven O diagnostics {sorted(missing)} in {path}")
        if epoch["diagnostics"]["correction_applied"] is not True:
            raise RuntimeError(
                f"E-driven correction was not applied in {path}, "
                f"epoch {epoch['epoch']}")
    rows.append(result)

for seed in SEEDS:
    hashes = {row["initial_model_sha256"] for row in rows if row["seed"] == seed}
    if len(hashes) != 1:
        raise RuntimeError(f"Methods do not share initialization for seed {seed}: {hashes}")
    for epoch in range(1, EPOCHS + 1):
        audits = {(row["history"][epoch - 1]["diagnostics"]["statistics_indices_sha256"],
                   row["history"][epoch - 1]["diagnostics"]["projection_indices_sha256"])
                  for row in rows if row["seed"] == seed and
                     row["method"] in PROBE_METHODS}
        if len(audits) != 1:
            raise RuntimeError(
                f"Probe batches differ across methods for seed {seed}, epoch {epoch}")
    epoch_audits = [next(row for row in rows
                         if row["seed"] == seed and row["method"] == "tiny_projection")
                    ["history"][epoch]["diagnostics"]["projection_indices_sha256"]
                    for epoch in range(EPOCHS)]
    if len(set(epoch_audits)) != EPOCHS:
        raise RuntimeError(f"Projection batches were reused for seed {seed}")

# Epsilon is fixed a priori for this focused mechanism run. Do not launch the
# baseline/final-test matrix; the only question here is whether applied E-driven
# O improves validation performance over the existing table.
validation_by_epsilon = {
    epsilon: statistics.mean(
        row["validation_accuracy"] for row in rows
        if row["method"] == "tiny_projection" and
           row["config"]["probe_epsilon"] == epsilon)
    for epsilon in EPSILONS}
SELECTED_EPSILON = max(validation_by_epsilon, key=validation_by_epsilon.get)
FINAL_CONFIGS = []
final_queue = queue.Queue()
for seed in SEEDS:
    for method, epsilon in FINAL_CONFIGS:
        final_queue.put((method, seed, epsilon))
final_failures = []
final_lock = threading.Lock()

def run_final_test_worker(gpu):
    while True:
        try:
            method, seed, epsilon = final_queue.get_nowait()
        except queue.Empty:
            return
        epsilon_label = str(epsilon).replace(".", "p")
        label = f"final_test_{method}_eps{epsilon_label}_seed{seed}"
        arm_dir = OUTPUT / label
        result_path = arm_dir / "result.json"
        if result_path.is_file() and method in CG_METHODS:
            previous = json.loads(result_path.read_text())
            if not solver_config_is_current(previous):
                archived = arm_dir / "result.pre_e_driven_o.json"
                if archived.exists():
                    archived.unlink()
                result_path.rename(archived)
        if not result_path.is_file():
            command = [
                sys.executable, "-m", "experiments.run_gromo_pilot",
                "--method", method, "--seed", str(seed),
                "--epochs", str(EPOCHS), "--batch-size", str(BATCH_SIZE),
                "--warmup-epochs", str(WARMUP_EPOCHS),
                "--warmup-checkpoint", str(WARMUP_CHECKPOINTS[seed]),
                "--reference-root", str(REFERENCE),
                "--train-samples", str(TRAIN_SAMPLES),
                "--validation-samples", str(VALIDATION_SAMPLES),
                "--statistics-samples", str(STATISTICS_SAMPLES),
                "--projection-samples", str(PROJECTION_SAMPLES),
                "--tuning-samples", str(TUNING_SAMPLES),
                "--image-size", str(IMAGE_SIZE), "--site", SITE,
                "--candidate-sites", CANDIDATE_SITES,
                "--rank", str(RANK), "--probe-epsilon", str(epsilon),
                "--cg-iterations", str(CG_ITERATIONS),
                "--cg-relative-tolerance", str(CG_RELATIVE_TOLERANCE),
                "--cg-preconditioner-probes", str(CG_PRECONDITIONER_PROBES),
                "--solver-revision", SOLVER_REVISION,
                "--evaluate-official-test", "--data-root", str(DATA_ROOT),
                "--output", str(arm_dir),
            ]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="2", PYTHONPATH=RUNTIME_PYTHONPATH)
            arm_dir.mkdir(parents=True, exist_ok=True)
            with (arm_dir / "run.log").open("a") as log:
                return_code = subprocess.run(
                    command, cwd=REPO, env=env, stdout=log,
                    stderr=subprocess.STDOUT).returncode
            if return_code:
                with final_lock:
                    final_failures.append((label, return_code))
        final_queue.task_done()

final_workers = [threading.Thread(
    target=run_final_test_worker, args=(gpu,), daemon=True) for gpu in range(2)]
for worker in final_workers: worker.start()
for worker in final_workers: worker.join()
if final_failures:
    raise RuntimeError(f"Final official-test runs failed: {final_failures}")
final_test_rows = []
for seed in SEEDS:
    for method, epsilon in FINAL_CONFIGS:
        epsilon_label = str(epsilon).replace(".", "p")
        path = OUTPUT / f"final_test_{method}_eps{epsilon_label}_seed{seed}" / "result.json"
        final_result = json.loads(path.read_text())
        if final_result["official_test_accuracy"] is None:
            raise RuntimeError(f"final arm did not evaluate official test: {path}")
        final_test_rows.append(final_result)

for test_row in final_test_rows:
    tuning_matches = [
        row for row in rows
        if row["method"] == test_row["method"] and
           row["seed"] == test_row["seed"] and
           row["config"]["probe_epsilon"] == test_row["config"]["probe_epsilon"]]
    expected_hash = next(
        row["initial_model_sha256"] for row in rows
        if row["seed"] == test_row["seed"])
    if test_row["initial_model_sha256"] != expected_hash:
        raise RuntimeError("final test rerun did not load the tuning checkpoint")
    if (tuning_matches and abs(
            test_row["validation_accuracy"] -
            tuning_matches[0]["validation_accuracy"]) > 1e-3):
        raise RuntimeError("final test rerun materially diverged from tuning")
    if test_row["method"] in CG_METHODS:
        failed_epochs = [epoch["epoch"] for epoch in test_row["history"]
                         if epoch["diagnostics"].get("cg_converged") is not True]
        if failed_epochs:
            raise RuntimeError(
                f"final CG did not converge for {test_row['method']} "
                f"at epochs {failed_epochs}")
    if (test_row["method"] in APPLY_METHODS and
            test_row["correction_application_rate"] != 1.0):
        raise RuntimeError(
            f"final correction application rate is incomplete for "
            f"{test_row['method']}")

historical_summary = OUTPUT / "summary.pre_e_driven_o.json"
current_summary = OUTPUT / "summary.json"
if current_summary.is_file():
    previous_summary = json.loads(current_summary.read_text())
    if not previous_summary.get("focused_e_driven_o_only"):
        shutil.copy2(current_summary, historical_summary)

summary = {"repo_commit": commit, "official_test_evaluated_final_only": False,
           "focused_e_driven_o_only": True,
           "historical_comparison_summary": (
               str(historical_summary) if historical_summary.is_file() else None),
           "arms": {}}
groups = sorted(
    {(row["method"], row["config"]["probe_epsilon"]) for row in rows} |
    set(FINAL_CONFIGS))
for method, epsilon in groups:
    key = f"{method}@epsilon={epsilon}"
    selected = [row for row in rows
                if row["method"] == method and
                   row["config"]["probe_epsilon"] == epsilon]
    final_selected = [row for row in final_test_rows
                      if row["method"] == method and
                         row["config"]["probe_epsilon"] == epsilon]
    metric_rows = selected or final_selected
    values = [row["validation_accuracy"] for row in metric_rows]
    official_values = [
        row["official_test_accuracy"] for row in final_test_rows
        if row["method"] == method and
           row["config"]["probe_epsilon"] == epsilon]
    fit_residuals = [epoch["diagnostics"]["relative_residual"]
                 for row in metric_rows
                 for epoch in row["history"]
                 if epoch["diagnostics"] and
                    "relative_residual" in epoch["diagnostics"]]
    fit_cosines = [epoch["diagnostics"]["cosine_alignment"]
               for row in metric_rows
               for epoch in row["history"]
               if epoch["diagnostics"] and
                  "cosine_alignment" in epoch["diagnostics"]]
    heldout_residuals = [epoch["diagnostics"]["heldout_relative_residual"]
                         for row in metric_rows for epoch in row["history"]
                         if epoch["diagnostics"] and
                            "heldout_relative_residual" in epoch["diagnostics"]]
    heldout_cosines = [epoch["diagnostics"]["heldout_cosine_alignment"]
                       for row in metric_rows for epoch in row["history"]
                       if epoch["diagnostics"] and
                          "heldout_cosine_alignment" in epoch["diagnostics"]]
    actual_heldout_residuals = [
        epoch["diagnostics"]["actual_heldout_relative_residual"]
        for row in metric_rows for epoch in row["history"]
        if epoch["diagnostics"] and
           "actual_heldout_relative_residual" in epoch["diagnostics"]]
    actual_heldout_cosines = [
        epoch["diagnostics"]["actual_heldout_cosine_alignment"]
        for row in metric_rows for epoch in row["history"]
        if epoch["diagnostics"] and
           "actual_heldout_cosine_alignment" in epoch["diagnostics"]]
    structural_gains = [epoch["diagnostics"].get(
                            "structural_loss_gain",
                            epoch["diagnostics"].get("local_structural_loss_gain"))
                        for row in metric_rows
                        for epoch in row["history"]
                        if epoch["diagnostics"] and
                           ("structural_loss_gain" in epoch["diagnostics"] or
                            "local_structural_loss_gain" in epoch["diagnostics"])]
    tuning_structural_gains = [epoch["diagnostics"]["tuning_structural_loss_gain"]
                              for row in metric_rows
                              for epoch in row["history"]
                              if epoch["diagnostics"] and
                                 epoch["diagnostics"].get("tuning_structural_loss_gain") is not None]
    projected_gains = [epoch["diagnostics"].get(
                           "tuning_projected_loss_gain",
                           epoch["diagnostics"].get("tuning_local_projected_loss_gain"))
                       for row in metric_rows
                       for epoch in row["history"]
                       if epoch["diagnostics"] and
                          (epoch["diagnostics"].get("tuning_projected_loss_gain") is not None or
                           epoch["diagnostics"].get("tuning_local_projected_loss_gain") is not None)]
    e_times = [epoch["diagnostics"]["e_statistics_solve_seconds"]
               for row in metric_rows for epoch in row["history"]
               if epoch["diagnostics"] and
                  "e_statistics_solve_seconds" in epoch["diagnostics"]]
    projection_times = [epoch["diagnostics"]["projection_seconds"]
                        for row in metric_rows for epoch in row["history"]
                        if epoch["diagnostics"] and
                           "projection_seconds" in epoch["diagnostics"]]
    peak_memory = [epoch["diagnostics"]["gpu_peak_allocated_bytes"]
                   for row in metric_rows for epoch in row["history"]
                   if epoch["diagnostics"] and
                      "gpu_peak_allocated_bytes" in epoch["diagnostics"]]
    peak_reserved = [epoch["diagnostics"]["gpu_peak_reserved_bytes"]
                     for row in metric_rows for epoch in row["history"]
                     if epoch["diagnostics"] and
                        "gpu_peak_reserved_bytes" in epoch["diagnostics"]]
    jvp_calls = [epoch["diagnostics"]["jvp_calls"]
                 for row in metric_rows for epoch in row["history"]
                 if epoch["diagnostics"] and
                    "jvp_calls" in epoch["diagnostics"]]
    vjp_calls = [epoch["diagnostics"]["vjp_calls"]
                 for row in metric_rows for epoch in row["history"]
                 if epoch["diagnostics"] and
                    "vjp_calls" in epoch["diagnostics"]]
    recovery = [epoch["diagnostics"]["tuning_local_recovery_fraction"]
                for row in metric_rows for epoch in row["history"]
                if epoch["diagnostics"] and
                   epoch["diagnostics"].get("tuning_local_recovery_fraction") is not None]
    cg_rows = [epoch["diagnostics"]
               for row in metric_rows for epoch in row["history"]
               if epoch["diagnostics"] and
                  epoch["diagnostics"].get("cg_converged") is not None]
    summary["arms"][key] = {
        "method": method,
        "probe_epsilon": epsilon,
        "validation_accuracy_mean": statistics.mean(values),
        "validation_accuracy_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "official_test_accuracy_mean": (
            statistics.mean(official_values) if official_values else None),
        "mean_heldout_projection_residual": statistics.mean(heldout_residuals) if heldout_residuals else None,
        "mean_heldout_cosine_alignment": statistics.mean(heldout_cosines) if heldout_cosines else None,
        "mean_actual_heldout_relative_residual": statistics.mean(actual_heldout_residuals) if actual_heldout_residuals else None,
        "mean_actual_heldout_cosine_alignment": statistics.mean(actual_heldout_cosines) if actual_heldout_cosines else None,
        "mean_fit_projection_residual": statistics.mean(fit_residuals) if fit_residuals else None,
        "mean_fit_cosine_alignment": statistics.mean(fit_cosines) if fit_cosines else None,
        "mean_projection_structural_loss_gain": statistics.mean(structural_gains) if structural_gains else None,
        "mean_tuning_structural_loss_gain": statistics.mean(tuning_structural_gains) if tuning_structural_gains else None,
        "mean_tuning_projected_loss_gain": statistics.mean(projected_gains) if projected_gains else None,
        "mean_e_statistics_solve_seconds": statistics.mean(e_times) if e_times else None,
        "mean_projection_seconds": statistics.mean(projection_times) if projection_times else None,
        "max_gpu_peak_allocated_bytes": max(peak_memory) if peak_memory else None,
        "max_gpu_peak_reserved_bytes": max(peak_reserved) if peak_reserved else None,
        "mean_jvp_calls": statistics.mean(jvp_calls) if jvp_calls else None,
        "mean_vjp_calls": statistics.mean(vjp_calls) if vjp_calls else None,
        "mean_tuning_local_recovery_fraction": statistics.mean(recovery) if recovery else None,
        "cg_all_converged": (all(row["cg_converged"] for row in cg_rows)
                             if cg_rows else None),
        "mean_cg_final_residual_norm": (statistics.mean(
            row["cg_residual_norm"] for row in cg_rows) if cg_rows else None),
        "mean_cg_relative_residual": (statistics.mean(
            row["cg_relative_residual"] for row in cg_rows) if cg_rows else None),
        "mean_cg_damping_used": (statistics.mean(
            row["cg_damping_used"] for row in cg_rows) if cg_rows else None),
        "max_cg_damping_used": (max(
            row["cg_damping_used"] for row in cg_rows) if cg_rows else None),
        "mean_cg_attempt_count": (statistics.mean(
            row["cg_attempt_count"] for row in cg_rows) if cg_rows else None),
        "mean_cg_target_scale": (statistics.mean(
            row["cg_target_scale"] for row in cg_rows) if cg_rows else None),
        "mean_cg_residual_norm_at_12": (statistics.mean(
            row["cg_residual_norm_at_12"] for row in cg_rows
            if row["cg_residual_norm_at_12"] is not None)
            if any(row["cg_residual_norm_at_12"] is not None
                   for row in cg_rows) else None),
        "mean_cg_residual_norm_at_25": (statistics.mean(
            row["cg_residual_norm_at_25"] for row in cg_rows
            if row["cg_residual_norm_at_25"] is not None)
            if any(row["cg_residual_norm_at_25"] is not None
                   for row in cg_rows) else None),
        "mean_cg_residual_norm_at_50": (statistics.mean(
            row["cg_residual_norm_at_50"] for row in cg_rows
            if row["cg_residual_norm_at_50"] is not None)
            if any(row["cg_residual_norm_at_50"] is not None
                   for row in cg_rows) else None),
        "mean_cg_residual_norm_at_100": (statistics.mean(
            row["cg_residual_norm_at_100"] for row in cg_rows
            if row["cg_residual_norm_at_100"] is not None)
            if any(row["cg_residual_norm_at_100"] is not None
                   for row in cg_rows) else None),
        "mean_cg_residual_norm_at_200": (statistics.mean(
            row["cg_residual_norm_at_200"] for row in cg_rows
            if row["cg_residual_norm_at_200"] is not None)
            if any(row["cg_residual_norm_at_200"] is not None
                   for row in cg_rows) else None),
        "cg_solver_spaces": sorted({
            row["cg_solver_space"] for row in cg_rows
            if row["cg_solver_space"] is not None}),
        "cg_solver_dtypes": sorted({
            row["cg_solver_dtype"] for row in cg_rows
            if row["cg_solver_dtype"] is not None}),
        "cg_preconditioners": sorted({
            row["cg_preconditioner"] for row in cg_rows
            if row["cg_preconditioner"] is not None}),
        "cg_preconditioner_probes": sorted({
            row["cg_preconditioner_probes"] for row in cg_rows
            if row["cg_preconditioner_probes"] is not None}),
        "mean_cg_system_dimension": (statistics.mean(
            row["cg_system_dimension"] for row in cg_rows)
            if cg_rows else None),
        "mean_correction_application_rate": (statistics.mean(
            row["correction_application_rate"] for row in metric_rows
            if row["correction_application_rate"] is not None)
            if any(row["correction_application_rate"] is not None
                   for row in metric_rows) else None),
        "seeds": len(values),
    }
selected_tiny_key = f"tiny_projection@epsilon={SELECTED_EPSILON}"
summary["epsilon_selection"] = {
    "criterion": "fixed a priori for focused E-driven O run",
    "selected_arm": selected_tiny_key,
    "selected_epsilon": summary["arms"][selected_tiny_key]["probe_epsilon"],
    "official_test_accuracy_mean_after_selection":
        summary["arms"][selected_tiny_key]["official_test_accuracy_mean"],
}
current_summary.write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))

archive = shutil.make_archive(str(OUTPUT), "gztar", root_dir=OUTPUT)
print("Download or save as Kaggle Dataset:", archive)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "gpu", "dataSources": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destination = Path("notebooks/kaggle_counterfactual_projection_t4x2.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
