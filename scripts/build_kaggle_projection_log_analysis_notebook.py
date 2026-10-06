"""Build the CPU-only Kaggle notebook for attached E-to-O JSON/history logs."""
import json
from pathlib import Path


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [cell("markdown", """# Projection log analysis — CPU only

Set Accelerator to **None**, Internet on, and enable the `github_token` Secret.
Attach previous R18/R34/VGG E-to-O notebook outputs exposing `result.json`,
`history.json` or `history.jsonl`. One attached run is enough; all visible runs
are analyzed. No CIFAR data, model checkpoint, Gromo or GPU is required.

This notebook reads JSON files directly from mounted inputs and does not extract
checkpoint archives. If your logs exist only inside an archive, expose the JSON
files as a Kaggle input first. Phase B (site probing/persistent growth) is not run.

Run All prints metric keys/sources and writes per-run Spearman/bootstrap results
and Q25/median/Q75 cosine tables by selected site and VGG boundary group.
Edit the metadata rules in the configuration cell for runs reported as unknown.
Missing history epochs remain missing. Accuracy windows containing another
logged intervention are excluded from horizon correlations.
"""), cell("code", r'''import json, os, shutil, subprocess, sys
from pathlib import Path
from kaggle_secrets import UserSecretsClient

REPO = Path("/kaggle/working/projection-log-analysis-repo")
OUTPUT = Path("/kaggle/working/projection_log_analysis")
URL = "https://github.com/duyh80456-code/Counterfactual-Projection.git"
token = UserSecretsClient().get_secret("github_token").strip()
if not token:
    raise RuntimeError("Enable the github_token Kaggle Secret")
askpass = Path("/kaggle/working/.projection_log_git_askpass.py")
askpass.write_text("#!/usr/bin/env python3\nimport os,sys\np=sys.argv[1] if len(sys.argv)>1 else ''\nprint('x-access-token' if 'Username' in p else os.environ['GITHUB_TOKEN_RUNTIME'])\n")
askpass.chmod(0o700)
env = os.environ.copy()
env.update(GITHUB_TOKEN_RUNTIME=token, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT="0")
try:
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "main", URL, str(REPO)], env=env, check=True)
    else:
        subprocess.run(["git", "-C", str(REPO), "fetch", "--depth", "1", "origin", "main"], env=env, check=True)
        subprocess.run(["git", "-C", str(REPO), "merge", "--ff-only", "origin/main"], env=env, check=True)
finally:
    askpass.unlink(missing_ok=True)
    env.pop("GITHUB_TOKEN_RUNTIME", None)
    del token
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy", "pandas", "matplotlib"], check=True)
print("Repository commit:", subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())
'''), cell("code", '''# Paths in these rules are relative to /kaggle/input.
# JSON metadata is used first. Rules fill missing/unknown labels.
# Replace example paths below with paths printed in unknown_metadata.csv.
CONFIG = {
    "mapping": {},             # defaults printed by the analyzer; override actual keys here
    "history_mapping": {"epoch": ["epoch"], "accuracy": ["validation_accuracy"]},
    "metadata_rules": [
        # {"pattern": "notebooks/yourname/vgg-run/**", "metadata": {"backbone": "VGG", "seed": 1}},
        # {"pattern": "notebooks/yourname/r18-run/**", "metadata": {"backbone": "R18", "seed": 1}},
        # {"pattern": "notebooks/yourname/r34-run/**", "metadata": {"backbone": "R34", "seed": 1}},
    ],
    "metadata_overrides": {},  # exact relative file path -> metadata, overrides JSON too
}
PATTERNS = ["**/result.json", "**/history.json", "**/history.jsonl"]
BOOTSTRAP = 2000
HORIZONS = [1, 5, 15]
OUTPUT.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = OUTPUT / "analysis_config.json"
CONFIG_PATH.write_text(json.dumps(CONFIG, indent=2))
'''), cell("code", '''# CPU analysis only: no checkpoint discovery, model loading or training.
subprocess.run([
    sys.executable, str(REPO / "scripts/analyze_projection_diagnostic.py"),
    "/kaggle/input", "--patterns", *PATTERNS,
    "--config", str(CONFIG_PATH), "--output", str(OUTPUT),
    "--bootstrap", str(BOOTSTRAP), "--horizons", *map(str, HORIZONS),
], check=True)
'''), cell("code", '''import pandas as pd
from IPython.display import display, Image
for filename in ("metric_sources.csv", "unknown_metadata.csv",
                 "cosine_by_site_boundary.csv", "run_spearman.csv"):
    print(filename)
    display(pd.read_csv(OUTPUT / filename))
for filename in ("selected_sites.png", "heldout_distributions.png"):
    if (OUTPUT / filename).is_file():
        display(Image(filename=str(OUTPUT / filename)))
archive = shutil.make_archive(str(OUTPUT), "gztar", root_dir=OUTPUT)
print("Output:", OUTPUT)
print("Download:", archive)
print("For unknown labels: edit CONFIG, then rerun from its cell onward.")
''')]
notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "kaggle": {"isGpuEnabled": False, "isInternetEnabled": True}},
    "nbformat": 4, "nbformat_minor": 5}
path = Path("notebooks/kaggle_projection_log_analysis_cpu.ipynb")
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(path)
