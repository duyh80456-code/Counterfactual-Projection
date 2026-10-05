"""Build one Kaggle notebook for attached R18/R34/VGG diagnostic forks."""
import copy
import json
from pathlib import Path


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def markdown(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


template = json.loads(Path(
    "notebooks/kaggle_vgg16_seed1_methods_from_plateau_t4x2.ipynb").read_text())
bootstrap = copy.deepcopy(template["cells"][1])
bootstrap["source"] = "".join(bootstrap["source"]).replace(
    "/kaggle/working/vgg16_seed1_methods_from_plateau_v3",
    "/kaggle/working/projection_capacity_diagnostics_v1").splitlines(keepends=True)
setup = "".join(template["cells"][2]["source"])
setup = setup.replace('"-e", str(REPO)', '"-e", f"{REPO}[diagnostics]"')
setup = setup.replace('"-m", "pytest", "-q"',
    '"-m", "pytest", "-q", "tests/test_projection_diagnostic.py"')
setup = setup.replace('if torch.cuda.device_count() != 2:',
                       'if torch.cuda.device_count() < 1:')
setup = setup.replace('Select T4 x2;', 'Select a CUDA GPU (T4 x2 recommended);')

cells = [markdown('''# Projection and capacity diagnostics — attached checkpoints

Attach CIFAR-100 and previous notebook outputs containing ORIGINAL
`plateau_checkpoint.pt` forks. Attach E-to-O `result.json` outputs for phase A.
This notebook automatically discovers architecture and seed from checkpoint
metadata: ResNet18, ResNet34 and VGG16. No architecture is guessed from filenames.

Set Internet on, enable GPU, and enable the `github_token` Kaggle Secret.
Run All performs CPU log analysis, independent per-site projection probes,
short persistent-growth/Vanilla comparisons, and within-seed correlation reports.
Results and a downloadable archive are saved under `/kaggle/working`.

Default: 64 projection samples, 256 independent held-out samples, rank 4,
top-3 plus bottom-2 WHERE sites, H=20. These are diagnostic experiments,
not a rerun of the original E-to-O training arms. Growth uses an active registered
extension transaction throughout H epochs and bounded output-scale optimization;
it is not yet validated as native materialized Gromo widening.

Five sites cost 2 × 5 × H training epochs PER fork (200 at H=20).
With several seeds/backbones, this can require multiple Kaggle sessions.
Reattach this notebook's prior output to skip completed matching runs/sites.
Interrupted sites restart their short H-epoch comparison from theta_P.
'''), bootstrap, code(setup), code('''# Configuration: all supported attached forks run automatically.
# Optional subset example: {("vgg16", 1), ("resnet18", 1)}
RUN_FILTER = {("vgg16", 1)} | {(architecture, seed)
    for architecture in ("resnet18", "resnet34") for seed in (1, 2, 3)}
HORIZON = 20
RANK = 4
TOP_SITES = 3
BOTTOM_SITES = 2
DIAGNOSTIC_SEED = 0  # same deterministic split recipe for every fork
DEVICE = "cuda:0"   # runs sequentially; GPU1 is not required
REPO_COMMIT = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
ENV = os.environ.copy()
ENV["PYTHONPATH"] = RUNTIME_PYTHONPATH

def invoke(module, arguments):
    subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                   cwd=REPO, env=ENV, check=True)
'''), code('''# Discover original forks and show exactly which attached runs will execute.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.kaggle_checkpoint_discovery import discover_checkpoints

candidates, rejected = discover_checkpoints(
    "/kaggle/input", RUN_ROOT / "discovery", kind="plateau_fork_checkpoint")
architectures = {
    "CIFAR-ResNet18": "resnet18", "CIFAR-ResNet34": "resnet34",
    "CIFAR-VGG16-BN": "vgg16",
}
required = {"model", "optimizer", "scheduler", "rng",
            "train_loader_generator_state", "train_indices", "trigger_indices",
            "evaluation_indices", "source_tuning_indices", "epoch"}
RUNS = []
seen = set()
for item in candidates:
    payload = item["payload"]
    protocol = payload.get("protocol", {})
    architecture = architectures.get(protocol.get("architecture"))
    seed = protocol.get("seed")
    if not architecture or seed is None or not required.issubset(payload):
        print("Ignoring unsupported/incomplete fork:", item["path"], protocol)
        continue
    seed = int(seed)
    if RUN_FILTER is not None and (architecture, seed) not in RUN_FILTER:
        continue
    if item["sha256"] in seen:
        continue
    seen.add(item["sha256"])
    RUNS.append({"architecture": architecture, "seed": seed,
                 "checkpoint": item["path"], "sha256": item["sha256"],
                 "fork_epoch": int(payload["epoch"])})
RUNS.sort(key=lambda run: (run["architecture"], run["seed"], run["sha256"]))
print("Rejected checkpoint candidates:", rejected)
print("Discovered diagnostic runs:", json.dumps(RUNS, indent=2))
if not RUNS:
    raise FileNotFoundError("Attach original plateau_checkpoint.pt forks with architecture/seed metadata")
print("Estimated SGD epochs:", len(RUNS) * 2 * (TOP_SITES + BOTTOM_SITES) * HORIZON)
# Release loaded model states before GPU probing.
del candidates
if "payload" in globals(): del payload
if "item" in globals(): del item
'''), code('''# A: copy ordinary result.json files from mounted outputs (including symlinks).
# Archives must expose their JSON files; checkpoint discovery also handles .pt archives.
import pandas as pd
from scripts.analyze_projection_diagnostic import load_interventions, write_outputs

LOG_ROOT = RUN_ROOT / "attached_logs"
LOG_ROOT.mkdir(parents=True, exist_ok=True)
for directory, _, filenames in os.walk("/kaggle/input", followlinks=True):
    if "result.json" not in filenames:
        continue
    source = Path(directory) / "result.json"
    target = LOG_ROOT / source.relative_to("/kaggle/input")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
metadata_overrides = {}
fork_metadata = {run["sha256"]: run for run in RUNS}
for path in LOG_ROOT.rglob("result.json"):
    try:
        payload = json.loads(path.read_text())
    except (ValueError, OSError):
        continue
    if not isinstance(payload, dict):
        continue
    identity = payload.get("theta_best_hash", payload.get("plateau_checkpoint_hash"))
    if identity in fork_metadata:
        run = fork_metadata[identity]
        metadata_overrides[str(path.relative_to(LOG_ROOT))] = {
            "architecture": run["architecture"], "seed": run["seed"]}
df = load_interventions(LOG_ROOT, ["**/result.json"],
                        metadata_overrides=metadata_overrides)
if len(df):
    write_outputs(df, RUN_ROOT / "phase_a")
else:
    print("No E-to-O JSON logs attached. Skipping phase A; phase B can run from forks.")
'''), code('''# B: independent site probes + matched short persistent-growth labels.
# Restore completed results from attached prior diagnostic notebook output.
for run in RUNS:
    name = f"{run['architecture']}_seed{run['seed']}_{run['sha256'][:12]}"
    destination = RUN_ROOT / "phase_b" / name
    destination.mkdir(parents=True, exist_ok=True)
    signature = {"checkpoint_hash": run["sha256"], "horizon": HORIZON,
                 "rank": RANK, "top": TOP_SITES, "bottom": BOTTOM_SITES,
                 "diagnostic_seed": DIAGNOSTIC_SEED, "repo_commit": REPO_COMMIT}
    # Only copy results with the exact fork/config/code signature.
    for directory, _, filenames in os.walk("/kaggle/input", followlinks=True):
        if "manifest.json" not in filenames:
            continue
        prior = Path(directory)
        try:
            matching = json.loads((prior / "manifest.json").read_text()) == signature
        except (ValueError, OSError):
            matching = False
        if matching:
            shutil.copytree(prior, destination, dirs_exist_ok=True)
            break
    (destination / "manifest.json").write_text(json.dumps(signature, indent=2))
    common = [run["checkpoint"], "--data-root", DATA_ROOT,
              "--reference-root", REFERENCE, "--architecture", run["architecture"],
              "--seed", DIAGNOSTIC_SEED, "--rank", RANK, "--device", DEVICE]
    if not (destination / "site_probes.json").is_file():
        invoke("diagnostics.site_projection_probe", [*common,
               "--output", destination / "site_probes.json"])
    if not (destination / "growth" / "summary.json").is_file():
        invoke("diagnostics.persistent_growth_probe", [*common,
               "--horizon", HORIZON, "--top", TOP_SITES, "--bottom", BOTTOM_SITES,
               "--resume-completed", "--output", destination / "growth"])
    invoke("diagnostics.summarize_capacity", [
        "--projection", destination / "site_probes.json",
        "--growth", destination / "growth" / "summary.json",
        "--output", destination / "capacity"])
    print("Completed:", name)
'''), code('''# Aggregate within-seed estimates across all attached forks.
probe_files = sorted((RUN_ROOT / "phase_b").glob("*/site_probes.json"))
growth_files = sorted((RUN_ROOT / "phase_b").glob("*/growth/summary.json"))
invoke("diagnostics.summarize_capacity", [
    "--projection", *probe_files, "--growth", *growth_files,
    "--output", RUN_ROOT / "capacity_all_seeds"])
for filename in ("within_seed_spearman.csv", "across_seed_summary.csv"):
    print(filename)
    display(pd.read_csv(RUN_ROOT / "capacity_all_seeds" / filename))
archive = shutil.make_archive(str(RUN_ROOT), "gztar", root_dir=RUN_ROOT)
print("Download archive:", archive)
print("Output folder:", RUN_ROOT)
''')]
notebook = {"cells": cells, "metadata": copy.deepcopy(template["metadata"]),
            "nbformat": 4, "nbformat_minor": 5}
path = Path("notebooks/kaggle_projection_capacity_diagnostics_t4x2.ipynb")
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(path)
