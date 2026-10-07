"""Build the self-contained DeiT algorithm ablation notebook."""
import json
from pathlib import Path


def code(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': source.splitlines(keepends=True)}


def markdown(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)}


template = json.loads(Path('notebooks/kaggle_deit_tiny_seed1_one_shot.ipynb').read_text())
bootstrap = ''.join(template['cells'][1]['source']).replace('deit-one-shot-repo', 'deit-all-arms-repo')
bootstrap = bootstrap.replace('"tests/test_deit_e_rollback.py"', '"tests/test_deit_e_rollback.py", "tests/test_deit_ablation.py", "tests/test_deit_ablation_integration.py", "tests/test_deit_logging.py", "tests/test_deit_parallel.py", "tests/test_deit_resume.py", "tests/test_kaggle_checkpoint_discovery.py"')
cells = [markdown('''# DeiT-Tiny CIFAR-100 — algorithm ablations A0–A6

Input: attach CIFAR-100 (`cifar-100-python/train`, `meta`), enable Internet,
a CUDA GPU and Kaggle Secret `github_token`. Run All needs no pretrained model.
Optional: attach the previous output .tar.gz or original .pt files from this exact protocol v5 experiment.
Discovery runs on CPU; completed arms are skipped, partial arms restore full training/controller state.
JSON/history alone cannot resume. See resume_plan.json for decisions.
Old v4 / one-shot forks are incompatible; they are not silently reused.

Res18-matched experimental budgets: batch64, rank4, projection32, scales
{.0125,.025,.05}, WHERE3x32, stats256, gate32, CG200, patience10.
DeiT-specific: AdamW LR5e-4/WD.05, no decay on bias/norm/CLS/position,
warmup5 + cosine300 to LR5e-6. This is an initial CIFAR recipe, not a claim
of optimal DeiT training. Plateau tracking starts at epoch0, with no minimum
epoch or requirement to finish the cosine schedule. Every strict validation
accuracy best resets the stall. After150 epochs without exceeding the latest
best, reload that historical-best checkpoint and reuse those same150 observed
Vanilla epochs as A0. Each method starts from the identical best and trains150
epochs at the inherited scheduler position; cosine continues without rebasing.

A0 observed Vanilla; A1 fixed last-MLP O-only; A2 raw E WHERE; A3 normalized WHERE;
A4a Gaussian original-MLP parameters matched per tensor with A2's scale;
A4b Gaussian logits matched in norm with its own projector/gate line search;
A5 persistent width growth; A6 B-only Gauss–Newton, at most5 accepted steps.
Requested rank4 adds1540 parameters at dim192; log actual rank/count.
A5 gate grid is {0,.025,.05,.1,.2}; A6 projection grid is {.1,.25,.5,1}.
Those deliberately different grids are recorded, so A6 vs A2 is not a pure
solver-only comparison. Opt-E fit/selection batches are independent and appended
without changing the original probe partition. Neither gate nor opt_val is
an unbiased held-out diagnostic.

A2/A3/A4/A5/successful-A6 share anchor rollback after10 non-improving epochs:
higher accuracy OR equal accuracy/lower loss moves anchor and resets counter.
Restore model/AdamW/scheduler, retain current RNG/data stream. No retrigger in
this one-intervention ablation. A0/A1 have no rollback, matching their control
roles; failed A6 trains as Vanilla with no rollback. Thus comparisons with A0/A1
include the controller effect. E directions only touch original selected MLP;
A5 is explicitly the exception that keeps extra width and migrates Adam state.
Validation selects plateau/rollback and reports scientific escape. Official test
is unused. Phase1 runs on GPU0. Phase2 runs independent arms concurrently, at most one
arm on each visible GPU (up to2); free GPUs take the next queued arm. A0 exports
on CPU before the GPU jobs. Opt-E runs last after all other arms finish; a failed
arm is logged and later arms still run. Tagged JSON console rows report train/val,
LR, report best, anchor, stalls, interventions, rollback, epoch time and peak
GPU bytes; each output folder also saves console.jsonl. No full-width GPU accuracy/runtime
claim follows from the tiny-model tests.
'''), code(bootstrap), code('''from dataclasses import asdict
from experiments.deit_protocol import DeitRecipe, protocol, canonical_model_config
from adapters.deit_ablation import res18_cp_config
from experiments.shared_protocol import sha256_file

SEED = 1
SCHEDULE_EPOCHS = 300
STALL_PATIENCE = 150
POST_FORK_EPOCHS = 150
ALGORITHM_PATIENCE = 10
MAX_EPOCH = 800
BATCH_SIZE = 64
GPU_DEVICES = 'auto'  # T4x2: two workers; one GPU: one worker.
LEARNING_RATE = 5e-4  # Explicit per-arm LR; no implicit ImageNet batch scaling.
RANK = 4
PROJECTION_SAMPLES = 32
SCALES = '.0125,.025,.05'
OUTPUT = Path(f'/kaggle/working/deit_tiny_seed{SEED}_all_arms_v5')
OUTPUT.mkdir(parents=True, exist_ok=True)
RECIPE = DeitRecipe(seed=SEED, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
    schedule_epochs=SCHEDULE_EPOCHS, stall_start_epoch=0,
    stall_patience=STALL_PATIENCE, reference_epochs=POST_FORK_EPOCHS, max_epoch=MAX_EPOCH)
RECIPE.validate()
EXPECTED = protocol(RECIPE, canonical_model_config())
CP = res18_cp_config()
from dataclasses import replace
CP = replace(CP, rank=RANK, projection_samples=PROJECTION_SAMPLES,
             scales=tuple(map(float, SCALES.split(','))))
roots = set()
for directory, _, files in os.walk('/kaggle/input', followlinks=True):
    if Path(directory).name == 'cifar-100-python' and {'train', 'meta'}.issubset(files):
        roots.add(Path(directory).parent.resolve())
if len(roots) != 1:
    raise FileNotFoundError(f'Attach exactly one CIFAR-100 dataset: {roots}')
DATA_ROOT = next(iter(roots))
print('Recipe:', asdict(RECIPE))
print('Projection:', asdict(CP))
'''), code('''# Import full outputs (original .pt, .tar.gz/.zip, or mounted notebook output).
import shutil
from experiments.deit_resume import prepare_resume
from experiments.run_deit_all_arms import ALL_ARMS
RESUME_PLAN = prepare_resume(['/kaggle/input'], OUTPUT, RECIPE, CP,
    horizon=POST_FORK_EPOCHS, patience=ALGORITHM_PATIENCE, inner_steps=5, arms=ALL_ARMS)
PHASE1 = OUTPUT / 'vanilla_stall'
FORK = PHASE1 / 'plateau_checkpoint.pt'
VANILLA_REFERENCE = PHASE1 / 'vanilla_reference.pt'
'''), code('''command = [sys.executable, '-m', 'experiments.run_deit_all_arms', '--data-root', str(DATA_ROOT),
    '--output', str(OUTPUT), '--seed', str(SEED), '--schedule-epochs', str(SCHEDULE_EPOCHS),
    '--max-epoch', str(MAX_EPOCH), '--batch-size', str(BATCH_SIZE), '--learning-rate', str(LEARNING_RATE),
    '--stall-patience', str(STALL_PATIENCE), '--post-fork-epochs', str(POST_FORK_EPOCHS),
    '--algorithm-patience', str(ALGORITHM_PATIENCE), '--rank', str(RANK),
    '--projection-samples', str(PROJECTION_SAMPLES), '--scales', SCALES, '--opt-inner-steps', '5',
    '--gpu-devices', GPU_DEVICES]
if FORK.exists():
    command += ['--plateau-checkpoint', str(FORK), '--vanilla-reference', str(VANILLA_REFERENCE)]
try:
    subprocess.run(command, cwd=REPO, env=ENV, check=True)
finally:
    archive = shutil.make_archive(str(OUTPUT), 'gztar', root_dir=OUTPUT)
    print('Resumable output archive:', archive)
'''), code('''summary_path = OUTPUT / 'summary.json'
if summary_path.exists():
    print(summary_path.read_text())
print('Status:', (OUTPUT / 'arm_status.json').read_text())
''')]
notebook = {'cells': cells, 'metadata': template['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
Path('notebooks/kaggle_deit_tiny_seed1_all_arms.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
