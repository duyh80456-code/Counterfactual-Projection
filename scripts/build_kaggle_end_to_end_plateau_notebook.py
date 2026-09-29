"""Build the one-file theta300 -> plateau -> four-arm Kaggle notebook."""

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

cells = [
    markdown("""# End-to-end plateau experiment — theta300 to four-arm fork

This is the single-file runner. Attach only CIFAR-100 and the dataset containing
`shared_seed1_epoch300.pt`, select **T4 x2**, and provide the Kaggle Secret
`github_token`.

The notebook first continues Vanilla from theta300 until the robust plateau
criterion is met. It then immediately forks the resulting `plateau_checkpoint.pt`
into Vanilla, E-driven O, O-only, and matched-horizon Bypass. There is no need to
save and reattach a Phase-1 dataset between the two stages.

Epoch 500 remains a review boundary. If no plateau is found, Phase 2 is not run;
the complete `checkpoint_latest.pt` is retained. Attach that notebook output,
increase `MAX_EPOCH`, and rerun this same notebook to resume Phase 1.
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
    markdown("""## Phase 1 — Vanilla until robust plateau

Only GPU0 is needed during convergence search. The second T4 becomes active as
soon as Phase 2 starts.
"""),
    phase1["cells"][4],
    code("""phase1_result = json.loads((OUTPUT / "result.json").read_text())
print(json.dumps({key: phase1_result[key] for key in (
    "status", "plateau_found", "plateau_epoch", "review_epoch_reached",
    "final_validation_accuracy", "best_validation_accuracy",
    "final_validation_loss", "plateau_checkpoint")}, indent=2))

PHASE1_OUTPUT = OUTPUT
if not phase1_result["plateau_found"]:
    archive = shutil.make_archive(str(PHASE1_OUTPUT), "gztar", root_dir=PHASE1_OUTPUT)
    print("NO CONVERGENCE CLAIM. Resume this notebook with the current output.")
    print("Phase-1 archive:", archive)
    raise RuntimeError(
        "Review horizon reached without robust plateau; Phase 2 was intentionally skipped")

PLATEAU_CHECKPOINT = Path(phase1_result["plateau_checkpoint"])
if not PLATEAU_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Phase 1 reported a missing checkpoint: {PLATEAU_CHECKPOINT}")
plateau_payload = torch.load(
    PLATEAU_CHECKPOINT, map_location="cpu", weights_only=False)
if plateau_payload.get("kind") != "plateau_fork_checkpoint":
    raise RuntimeError("Phase 1 did not produce a plateau fork checkpoint")
PLATEAU_EPOCH = int(plateau_payload["epoch"])
PLATEAU_HASH = hashlib.sha256(PLATEAU_CHECKPOINT.read_bytes()).hexdigest()
print("theta_P ready:", PLATEAU_EPOCH, PLATEAU_CHECKPOINT, PLATEAU_HASH)
"""),
    markdown("""## Phase 2 — Dynamic four-arm queue on T4 x2

Ours and Bypass start first. The first free GPU receives O-only, followed by
Vanilla. Every arm is required to report the exact hash produced by Phase 1.
"""),
    code("""OUTPUT = Path("/kaggle/working/plateau_end_to_end_four_arm_60ep_v1")
OUTPUT.mkdir(parents=True, exist_ok=True)
"""),
    # Reuse the already-tested Phase-2 command/queue and aggregation cells.
    phase2["cells"][4],
    phase2["cells"][5],
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
