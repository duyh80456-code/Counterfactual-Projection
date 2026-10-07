"""Build the first DeiT-Tiny-CIFAR one-shot Kaggle experiment."""
import copy
import json
from pathlib import Path


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


# Reuse the token-safe repository clone/update helper from the CPU notebook.
template = json.loads(Path("notebooks/kaggle_projection_log_analysis_cpu.ipynb").read_text())
bootstrap = "".join(template["cells"][1]["source"])
bootstrap = bootstrap[:bootstrap.index('subprocess.run([sys.executable, "-m", "pip"')]
bootstrap = bootstrap.replace("projection-log-analysis-repo", "deit-one-shot-repo")
bootstrap += '''GROMO = Path("/kaggle/working/deit-gromo")
GROMO_COMMIT = "8d19107b61a9459a9021065a329b699adcb0f25b"
if not GROMO.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                    "https://github.com/growingnet/gromo.git", str(GROMO)], check=True)
subprocess.run(["git", "-C", str(GROMO), "fetch", "--depth", "1", "origin", GROMO_COMMIT], check=True)
subprocess.run(["git", "-C", str(GROMO), "checkout", "--detach", GROMO_COMMIT], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[vision,test]", "-e", str(GROMO)], check=True)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
ENV = os.environ.copy()
ENV["PYTHONPATH"] = os.pathsep.join((str(REPO), str(GROMO / "src")))
ENV["REQUIRE_DEIT_INTEGRATION"] = "1"
subprocess.run([sys.executable, "-m", "pytest", "-q",
    "tests/test_deit_function_preserving.py", "tests/test_deit_sites.py",
    "tests/test_deit_projection.py", "tests/test_deit_no_aux_persistence.py",
    "tests/test_deit_protocol.py", "tests/test_deit_vanilla_reference.py", "tests/test_deit_e_rollback.py"], cwd=REPO, env=ENV, check=True)
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Enable a CUDA GPU; this notebook trains DeiT")
print("Repository commit:", subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())
'''
cells = [markdown('''# DeiT-Tiny-CIFAR — first one-shot E-to-O experiment

Attach CIFAR-100, enable Internet, a CUDA GPU (T4 x2 is supported), and the
`github_token` Secret. Run All trains random-init Vanilla to strict-best stall,
exports theta_P, and runs matched Vanilla, fixed-site O-only and E-to-O for K
ordinary AdamW epochs. This notebook runs sequentially on GPU0.

Geometry: image32, patch4, 64 patches + CLS, embed192, depth12, heads3,
MLP hidden768; no distillation, pretrained weights, dropout or DropPath.
TINY uses rank8 requested at all 12 MLP sites, with effective rank logged.
WHERE selects raw mean observed E gain. HOW only changes fc1/fc2 at that site.
Vanilla reuses the observed Phase 1 plateau window; only O-only and E-to-O
train new 150-epoch arms. E-to-O rolls back after 10 epochs without strict
accuracy improvement to its accuracy/loss best; it never queries E again. O-only
has no rollback. No persistent extension is enabled.

AdamW recipe is a declared initial recipe, not the original ImageNet DeiT
recipe: LR=5e-4, WD=.05, 5 warmup epochs, fixed global cosine to epoch400,
minimum LR ratio .01. STALL_PATIENCE=150, no minimum-epoch gate, max epoch300.
Validation uses the same CIFAR split recipe as CNN runs (2000 reserved and unused,
3000 validation for historical-best fork and plateau selection). POST_FORK_EPOCHS=150.
Protocol v5 rejects older forks. Change these in the next config cell if needed;
resume requires the identical recipe and intervention configuration.

Optional: reattach this notebook's prior **expanded output files** to reuse a
fork or continue epoch checkpoints. To reuse Vanilla, attach vanilla_reference.pt
or the full Phase 1 checkpoint_latest.pt with the fork. A fork alone is insufficient.
Ambiguous distinct forks are rejected.
Probe tests use a small model and the native Gromo solver; passing them does
not establish full-width CUDA performance or accuracy.
'''), code(bootstrap), code('''from dataclasses import asdict
from experiments.deit_protocol import DeitRecipe
from adapters.deit_cp_adapter import CPConfig

SEED = 1
STALL_PATIENCE = 150
POST_FORK_EPOCHS = 150
ALGORITHM_PATIENCE = 10  # E-only rollback; ties with lower loss update anchor, not accuracy stall
RECIPE = DeitRecipe(seed=SEED, stall_patience=STALL_PATIENCE)
CP = CPConfig()  # rank8, epsilon .05, projection64, CG200, scales through .2
if POST_FORK_EPOCHS != STALL_PATIENCE:
    raise ValueError("Vanilla reuse requires the same Phase 1 stall and Phase 2 horizon")
OUTPUT = Path(f"/kaggle/working/deit_tiny_seed{SEED}_one_shot_v5")
OUTPUT.mkdir(parents=True, exist_ok=True)
# Override DATA_ROOT manually if more than one CIFAR dataset is attached.
roots = set()
for directory, _, filenames in os.walk("/kaggle/input", followlinks=True):
    if Path(directory).name == "cifar-100-python" and "train" in filenames:
        roots.add(Path(directory).parent.resolve())
if len(roots) != 1:
    raise FileNotFoundError(f"Attach exactly one CIFAR-100 input (or set DATA_ROOT manually): {roots}")
DATA_ROOT = next(iter(roots))
print("CIFAR root:", DATA_ROOT)
print("Vanilla recipe:", asdict(RECIPE))
print("One-shot configuration:", asdict(CP))

def invoke(module, arguments):
    subprocess.run([sys.executable, "-m", module, *map(str, arguments)],
                   cwd=REPO, env=ENV, check=True)

def flags(values):
    result = []
    for key, value in values.items():
        result += ["--" + key.replace("_", "-"),
                   ",".join(map(str, value)) if isinstance(value, tuple) else str(value)]
    return result
'''), code('''# Discover only this backbone's states. Cache is excluded from results archive.
from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.deit_protocol import protocol, canonical_model_config
from experiments.shared_protocol import sha256_file

states, rejected = discover_checkpoints("/kaggle/input",
    OUTPUT.parent / (OUTPUT.name + "_cache"), kind={"deit_plateau_fork",
        "deit_vanilla_latest", "deit_vanilla_best", "deit_fork_arm_latest", "deit_vanilla_reference"})
EXPECTED = protocol(RECIPE, canonical_model_config())
states = [item for item in states if item["payload"].get("protocol") == EXPECTED]
print("Discovery rejections:", rejected)

# Never guess among different selected theta_P forks.
forks = [item for item in states if item["payload"]["kind"] == "deit_plateau_fork"]
if len({item["sha256"] for item in forks}) > 1:
    raise RuntimeError("Multiple distinct matching DeiT forks attached; attach one run")
PHASE1 = OUTPUT / "vanilla_to_plateau"
PHASE1.mkdir(parents=True, exist_ok=True)
FORK = PHASE1 / "plateau_checkpoint.pt"
if not FORK.exists() and forks:
    shutil.copy2(forks[0]["path"], FORK)
if not FORK.exists():
    latest = [item for item in states if item["payload"]["kind"] == "deit_vanilla_latest"]
    if latest:
        greatest = max(item["payload"]["epoch"] for item in latest)
        latest = [item for item in latest if item["payload"]["epoch"] == greatest]
        if len(latest) != 1:
            raise RuntimeError("Ambiguous Vanilla resume states")
        selected = latest[0]
        bests = [item for item in states if item["payload"]["kind"] == "deit_vanilla_best" and
                 item["payload"]["epoch"] == selected["payload"]["historical_best_epoch"]]
        if len(bests) != 1:
            raise RuntimeError("Reattach both Vanilla latest and its matching best checkpoint")
        shutil.copy2(selected["path"], PHASE1 / "checkpoint_latest.pt")
        shutil.copy2(bests[0]["path"], PHASE1 / "checkpoint_best.pt")
    command = ["--data-root", DATA_ROOT, "--output", PHASE1, *flags(asdict(RECIPE))]
    if (PHASE1 / "checkpoint_latest.pt").exists():
        command += ["--resume", PHASE1 / "checkpoint_latest.pt"]
    invoke("experiments.train_deit_plateau", command)
FORK_HASH = sha256_file(FORK)
FORK_STATE = torch.load(FORK, map_location="cpu", weights_only=False)
if FORK_STATE["protocol"] != EXPECTED:
    raise RuntimeError("Local fork has a different recipe; select a new OUTPUT directory")
print("Selected theta_P:", FORK, FORK_HASH)
# Recover the existing Vanilla window, including its actual terminal state.
from experiments.deit_vanilla_reference import create_vanilla_reference
VANILLA_REFERENCE = PHASE1 / "vanilla_reference.pt"
if POST_FORK_EPOCHS != RECIPE.stall_patience:
    raise ValueError("Reusing Vanilla requires post-fork epochs equal to stall patience")
if not VANILLA_REFERENCE.exists():
    references = [item for item in states if item["payload"]["kind"] == "deit_vanilla_reference"
                  and item["payload"].get("theta_best_hash") == FORK_HASH
                  and item["payload"].get("completed_epochs") == POST_FORK_EPOCHS]
    if len({item["sha256"] for item in references}) > 1:
        raise RuntimeError("Ambiguous matching Vanilla references")
    if references:
        shutil.copy2(references[0]["path"], VANILLA_REFERENCE)
    else:
        terminals = [item for item in states if item["payload"]["kind"] == "deit_vanilla_latest"]
        local_terminal = PHASE1 / "checkpoint_latest.pt"
        if local_terminal.exists():
            terminals.append({"path": local_terminal, "sha256": sha256_file(local_terminal),
                              "payload": torch.load(local_terminal, map_location="cpu", weights_only=False)})
        terminals = [item for item in terminals if item["payload"].get("protocol") == EXPECTED
            and item["payload"].get("plateau_detected") is True
            and item["payload"]["epoch"] == FORK_STATE["epoch"] + POST_FORK_EPOCHS
            and item["payload"].get("historical_best_epoch") == FORK_STATE["epoch"]
            and item["payload"]["history"][:len(FORK_STATE["history"])] == FORK_STATE["history"]
            and item["payload"]["history"][len(FORK_STATE["history"]):] == FORK_STATE["stall_history"]]
        if len({item["sha256"] for item in terminals}) != 1:
            raise FileNotFoundError("Attach vanilla_reference.pt or one matching full Phase 1 latest checkpoint; Vanilla is not retrained")
        create_vanilla_reference(FORK, terminals[0]["path"], VANILLA_REFERENCE)
print("Observed Vanilla reference:", VANILLA_REFERENCE)
'''), code('''# Export Vanilla on CPU, then train only the two intervention arms.
from experiments.deit_e_rollback import rollback_protocol
invoke("experiments.run_deit_fork", ["--plateau-checkpoint", FORK,
    "--plateau-checkpoint-hash", FORK_HASH, "--method", "vanilla_continue",
    "--vanilla-reference", VANILLA_REFERENCE, "--output", OUTPUT / "vanilla_continue",
    "--post-fork-epochs", POST_FORK_EPOCHS, "--device", "cpu", *flags(asdict(CP))])
for method in ("o_projection_only", "ours_e_driven_o"):
    destination = OUTPUT / method
    destination.mkdir(parents=True, exist_ok=True)
    identity = {"fork_hash": FORK_HASH, "method": method, "post_fork_epochs": POST_FORK_EPOCHS,
                "cp_config": json.loads(json.dumps(asdict(CP)))}
    if method == "ours_e_driven_o":
        identity["e_controller"] = rollback_protocol(ALGORITHM_PATIENCE)
    resumable = [item for item in states if item["payload"]["kind"] == "deit_fork_arm_latest"
                 and item["payload"].get("run_identity") == identity]
    if resumable and not (destination / "checkpoint_latest.pt").exists():
        greatest = max(item["payload"]["completed_epochs"] for item in resumable)
        selected = [item for item in resumable if item["payload"]["completed_epochs"] == greatest]
        if len(selected) != 1:
            raise RuntimeError(f"Ambiguous resume checkpoints for {method}")
        shutil.copy2(selected[0]["path"], destination / "checkpoint_latest.pt")
    command = ["--data-root", DATA_ROOT, "--plateau-checkpoint", FORK,
               "--plateau-checkpoint-hash", FORK_HASH, "--method", method,
               "--output", destination, "--post-fork-epochs", POST_FORK_EPOCHS, *flags(asdict(CP))]
    if method == "ours_e_driven_o":
        command += ["--algorithm-patience", ALGORITHM_PATIENCE]
    if (destination / "checkpoint_latest.pt").exists():
        command += ["--resume", destination / "checkpoint_latest.pt"]
    invoke("experiments.run_deit_fork", command)
'''), code('''results = [json.loads((OUTPUT / method / "result.json").read_text())
           for method in ("vanilla_continue", "o_projection_only", "ours_e_driven_o")]
assert all(result["theta_best_hash"] == FORK_HASH for result in results)
assert len({result["fork_epoch"] for result in results}) == 1
assert len({result["historical_best_accuracy"] for result in results}) == 1
for result in results:
    print(json.dumps({"method": result["method"],
        "fork_epoch": result["fork_epoch"],
        "post_fork_epochs": result["post_fork_epochs"],
        "historical_best_accuracy": result["historical_best_accuracy"],
        "delta_vs_historical_best": result["delta_vs_historical_best"],
        "before": result["validation_before"],
        "immediately_after": result["validation_immediately_after_projection"],
        "report_best_accuracy": result["report_best_accuracy"],
        "report_best_loss": result["report_best_loss"],
        "report_best_epoch": result["report_best_epoch"],
        "final_accuracy": result["final_validation_accuracy"],
        "scientific_escape": result["scientific_escape"],
        "trajectory_source": result.get("trajectory_source", "new_intervention_arm"),
        "rollback_count": result.get("rollback_count", 0),
        "controller_anchor_epoch": result.get("controller_anchor_epoch"),
        "additional_training_epochs": result.get("additional_training_epochs", result["post_fork_epochs"])}, indent=2))
archive = shutil.make_archive(str(OUTPUT), "gztar", root_dir=OUTPUT)
print("Download:", archive)
''')]
metadata = copy.deepcopy(template["metadata"])
metadata["kaggle"] = {"isGpuEnabled": True, "isInternetEnabled": True}
notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
path = Path("notebooks/kaggle_deit_tiny_seed1_one_shot.ipynb")
path.write_text(json.dumps(notebook, indent=1) + "\n")
print(path)
