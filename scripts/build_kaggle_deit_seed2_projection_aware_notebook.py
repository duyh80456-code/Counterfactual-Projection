"""Seed2 D0 + two new top3 arms; existing controls are never rerun."""
import copy
import json
from pathlib import Path


def build():
    base = json.loads(Path('notebooks/kaggle_deit_tiny_seed2_rollback_comparison.ipynb').read_text())
    notebook = copy.deepcopy(base)
    notebook['cells'][0]['source'] = '''# DeiT-Tiny seed2 — projection-aware exploration: D0 / A3 / A4

**Input:** CIFAR-100 plus the original seed2 historical-best fork `.pt` or an
output archive containing it: **epoch375, validation accuracy56.20%, protocol v5**.
No Vanilla plateau training or old arm training runs here. Old `result.json` files
are optional for comparison; they must have the same fork hash, protocol, CP config
and horizon. Attach full original output archives for reliable hash preservation.
Enable Internet, T4x2 and Kaggle Secret `github_token`.

1. D0 scans all12 MLP sites, projects E and the negative summed-CE logit gradient
   into each same block, previews the gate grid, saves CSV/summary, and commits nothing.
2. A3 `e2o_top3_best_gate`: raw TINY rank → evaluate all top3 projections on
   the same untouched anchor/gate batch → commit at most one positive finite gate correction.
3. A4 `e2o_top3_recurrent`: same first search, plus one fresh top3 search after
   each rollback while training budget remains. No intervention after the final epoch.

A3 and A4 run independently from identical full fork state on two GPUs,150
post-fork training epochs each. Patience10, original rank4/scales(.0125,.025,.05),
WHERE3x32, projection32, gate32, adaptive dual-CG200, batch64, inherited AdamW/LR.
No cross-block projection; later training optimizes the full original network.
Gate is a selection batch, not unbiased held-out evidence. Validation never
selects site, damping, scale or a candidate; it only observes committed states,
updates report/anchor and triggers rollback.

Controller semantics are inherited: report best is strict accuracy with paired
loss; anchor is higher accuracy OR exact accuracy tie with lower loss, and stall
resets on anchor improvement. Rollback restores model/optimizer/scheduler,
preserves current RNG/loader stream. Recurrent searches draw fresh probe seeds
from current CPU RNG; the first search uses the original index0 partition.
A separate fixed original projection batch measures direction cosine across
searches and is never used to select candidates. No claim of a different direction
follows merely from selecting the same block or using different RNG.

Full latest checkpoints include committed events, prior direction and pending
search state. Resume skips complete arms and completes a pending rollback search
without repeating an already committed intervention. Outputs: D0 CSV+summary;
per arm result.json, interventions.jsonl, epoch_history.csv, best_checkpoint.pt,
run_metadata.json, checkpoint_latest.pt. Best observed after training and stored
anchor are reported separately. A3/A4 spend more proposal/CG compute than top1;
comparisons do not establish an equal-compute advantage.
'''.splitlines(keepends=True)
    bootstrap = ''.join(notebook['cells'][1]['source'])
    bootstrap = bootstrap.replace('"tests/test_deit_resume.py"', '"tests/test_deit_resume.py", "tests/test_deit_projection_aware.py"')
    notebook['cells'][1]['source'] = bootstrap.splitlines(keepends=True)
    notebook['cells'][2]['source'] = '''import os
OUTPUT = Path('/kaggle/working/deit_seed2_projection_aware_v1')
OUTPUT.mkdir(parents=True, exist_ok=True)
GPU_DEVICES = 'auto'
roots = set()
for directory, _, files in os.walk('/kaggle/input', followlinks=True):
    if Path(directory).name == 'cifar-100-python' and {'train', 'meta'}.issubset(files):
        roots.add(Path(directory).parent.resolve())
if len(roots) != 1:
    raise FileNotFoundError(f'Attach exactly one CIFAR-100 dataset: {roots}')
DATA_ROOT = next(iter(roots))
'''.splitlines(keepends=True)
    notebook['cells'][3]['source'] = '''# CPU-only discovery; requires a completed fork, never trains Vanilla.
from experiments.deit_projection_resume import prepare_projection_resume
PLAN = prepare_projection_resume(['/kaggle/input'], OUTPUT)
'''.splitlines(keepends=True)
    notebook['cells'][4]['source'] = '''command = [sys.executable, '-m', 'experiments.run_deit_projection_aware_suite',
    '--data-root', str(DATA_ROOT), '--output', str(OUTPUT),
    '--resume-root', '/kaggle/input', '--gpu-devices', GPU_DEVICES]
try:
    subprocess.run(command, cwd=REPO, env=ENV, check=True)
finally:
    archive = shutil.make_archive(str(OUTPUT), 'gztar', root_dir=OUTPUT)
    print('Resumable output archive:', archive)
'''.splitlines(keepends=True)
    notebook['cells'][5]['source'] = '''for name in ('all12_transfer_diagnostic/result.json', 'comparison.json', 'arm_status.json'):
    path = OUTPUT / name
    if path.exists():
        result = json.loads(path.read_text())
        print(name, json.dumps(result.get('summary', result), indent=2))
'''.splitlines(keepends=True)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []; cell['execution_count'] = None
    path = Path('notebooks/kaggle_deit_tiny_seed2_projection_aware.ipynb')
    path.write_text(json.dumps(notebook, indent=1) + '\n')
    return path


if __name__ == '__main__':
    print(build())
