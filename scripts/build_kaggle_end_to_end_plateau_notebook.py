"""Build the legacy theta300 -> meaningful-stall -> method Kaggle notebook."""

import json
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}


def source(cell):
    return "".join(cell["source"])


phase1 = json.loads(Path("notebooks/kaggle_vanilla_to_plateau.ipynb").read_text())
phase2 = json.loads(Path("notebooks/kaggle_plateau_fork_t4x2.ipynb").read_text())
legacy_phase2_commands = source(phase2["cells"][4]).replace(
    '"--post-fork-epochs", "150"', '"--post-fork-epochs", "100"')
legacy_phase2_summary = source(phase2["cells"][5]).replace(
    'result["post_fork_epochs"] != 150',
    'result["post_fork_epochs"] != 100').replace(
        "did not complete 150 epochs", "did not complete 100 epochs").replace(
            '"post_fork_epochs": 150', '"post_fork_epochs": 100')

cells = [
    markdown("""# End-to-end meaningful-stall experiment — theta300 to method fork

This is the single-file runner. Attach only CIFAR-100 and the dataset containing
`shared_seed1_epoch300.pt`, select **T4 x2**, and provide the Kaggle Secret
`github_token`.

The notebook keeps the theta300 LR unchanged, saves every exact trigger best,
and waits for 100 epochs without a significant +0.1 pp trigger improvement. It
then forks the saved
best state as `plateau_checkpoint.pt`, never the later degraded state,
into E-driven O, O-only, and matched-horizon Bypass. The already-observed
100-epoch Vanilla stall trajectory is reused as the control rather than trained
twice. There is no need to save and reattach a Phase-1 dataset between stages.

Epoch 500 remains a review boundary. If no stall is found, Phase 2 is not run;
both `checkpoint_latest.pt` and `checkpoint_best.pt` are retained. Attach that
notebook output, increase `MAX_EPOCH`, and rerun to resume Phase 1 exactly.
"""),
    # Clone/install/check GPUs and discover CIFAR exactly as in Phase 1.
    phase1["cells"][1],
    phase1["cells"][2],
    code("""if torch.cuda.device_count() != 2:
    raise RuntimeError(
        f"Select the Kaggle T4 x2 accelerator; found {torch.cuda.device_count()} GPU(s)")
"""),
    # Find theta300 and optionally resume Phase-1 progress.
    phase1["cells"][3],
    markdown("""## Phase 1 — Vanilla until no-new-best stall

Only GPU0 is needed during convergence search. The second T4 becomes active as
soon as Phase 2 starts.
"""),
    phase1["cells"][4],
    code("""phase1_result = json.loads((OUTPUT / "result.json").read_text())
print(json.dumps({key: phase1_result[key] for key in (
    "status", "plateau_found", "plateau_epoch", "best_epoch",
    "stall_detected_epoch", "epochs_without_improvement", "review_epoch_reached",
    "final_validation_accuracy", "best_validation_accuracy",
    "final_validation_loss", "plateau_checkpoint")}, indent=2))

PHASE1_OUTPUT = OUTPUT
if not phase1_result["plateau_found"]:
    archive = shutil.make_archive(str(PHASE1_OUTPUT), "gztar", root_dir=PHASE1_OUTPUT)
    print("NO CONVERGENCE CLAIM. Resume this notebook with the current output.")
    print("Phase-1 archive:", archive)
    raise RuntimeError(
        "Review horizon reached without a no-new-best stall; Phase 2 was intentionally skipped")

PLATEAU_CHECKPOINT = Path(phase1_result["plateau_checkpoint"])
if not PLATEAU_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Phase 1 reported a missing checkpoint: {PLATEAU_CHECKPOINT}")
plateau_payload = torch.load(
    PLATEAU_CHECKPOINT, map_location="cpu", weights_only=False)
if plateau_payload.get("kind") != "plateau_fork_checkpoint":
    raise RuntimeError("Phase 1 did not produce a plateau fork checkpoint")
PLATEAU_EPOCH = int(plateau_payload["epoch"])
PLATEAU_HASH = hashlib.sha256(PLATEAU_CHECKPOINT.read_bytes()).hexdigest()
PLATEAU_PAYLOAD = plateau_payload
VANILLA_CONTROL = dict(PLATEAU_PAYLOAD["vanilla_control"])
if VANILLA_CONTROL["post_fork_epochs"] != 100:
    raise RuntimeError("Phase 1 must contain exactly 100 matched Vanilla epochs")
print("theta_P ready:", PLATEAU_EPOCH, PLATEAU_CHECKPOINT, PLATEAU_HASH)
"""),
    markdown("""## Phase 2 — Fixed T4 x2 schedule

GPU0 runs E-driven O only. GPU1 runs Bypass then fresh O-only. Every method is
required to report
the exact hash produced by Phase 1.
"""),
    code("""import threading

OUTPUT = Path("/kaggle/working/plateau_end_to_end_three_methods_100ep_v4")
OUTPUT.mkdir(parents=True, exist_ok=True)
"""),
    # Reuse the already-tested Phase-2 command/queue and aggregation cells.
    code(legacy_phase2_commands),
    code(legacy_phase2_summary),
]

# Avoid carrying outputs/execution counters from source notebooks.
for cell in cells:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "kaggle": {"accelerator": "gpu", "dataSources": []}},
    "nbformat": 4, "nbformat_minor": 5}
destination = Path("notebooks/kaggle_plateau_end_to_end_t4x2.ipynb")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(destination)
