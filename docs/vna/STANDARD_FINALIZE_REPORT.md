# Finalize H1–H4 notebooks — 2026-10-28

Implementation source pin: SOURCE_PIN_TO_FILL.
Work base: c820c09 on codex/gir-live-task1, in the original repository folder.
No mathematical-core redesign, new worktree, ZIP overlay or GPU benchmark launch.
The reference Archive(3).zip was not present in the supplied attachment/Downloads;
the latest checked-out implementation and pinned upstream sources were used.

## Delivered protocol

- H1: original current CIFAR100 B0 Inc5/20-task iCaRL vs Replay-Aware VNA.
- H2: official LPR vs Geometry-VNA on actual Avalanche0.4 online Split-CIFAR100.
  README's omega0=4,beta=2,every_iter30 replaces generic1/1/None defaults. Backbone,
  class order, init seed0, batch10×3 repeats, SGD0.1 without momentum/WD/scheduler,
  memory2000,5% validation and evaluation are shared.
- H3-small: configs/standard_cl/h3_imagenet_small.json:100×50, batch100,
  cadence4000/window128, maximum16 exact candidate swaps plus identity. Both arms
  use identical horizon, original ConvNet initialization/optimizer/head reset and
  first100 original seed0 class pairs. Whole-model weights are not reset between
  tasks. Full5000×250 config is preserved and not queued.
- H4: ER vs Domain-VNA on all20 rotations, fixed ten-label classifier.

Every pair has matching native configuration and seed. Algorithm-specific
updates/corrections differ intentionally; this is not matched wall-time or a
kernel-only ablation. No test-set tuning/acceptance or extra ablation is added.
Each notebook is independent: baselineGPU0, fullGPU1, immutable source pin.

## Bounded H3 search

Generate existing tiny_V proposals. Preliminary ranking is
-gᵀδθ_reset-μ||δθ_reset||²/2; μ is the same damping used in the full identity
objective. Stable candidate-name tie-breaks, no random shortlist. Evaluate at
most16 resets with the existing full GN-recovery surrogate and componentwise
loss acceptance. Identity is always included; only the evaluated winner State
is committed. Snapshot/optimizer/RNG rollback and failed-solver audit remain.
Log all ranking/pruning scores, shortlist/exact counts, exact trial rejection
reasons, solver failures, ranking/solver/overall time and actual swaps.
Guarantee is only within the evaluated shortlist plus identity, not exhaustive
or global. Affine sites fc1→fc2/fc2→fc3 are unchanged from the prior implementation.

The original class_order file has14000 entries per run, not a single permutation.
A too-strict check introduced during this change failed against that source and
was corrected before delivery. This is not evidence that historical runs used
the wrong order. The shared helper preserves the actual native prefix/repetition convention. H3-small hashes/validates only200 used
files. CLI horizon overrides enter protocol/checkpoint identity. This is a
reduced pilot, not reproduction of published full-horizon CBP accuracy.

## H2 dependency and API verification

Real local interpreter: /tmp/vna_h2_py310/bin/python, Python3.10.18, macOS CPU.
Created a clean environment and installed the pinned stack; no Protocol shim.
Exact versions and dependency audit: H2_ENVIRONMENT_VERIFICATION.json.

- Torch2.2.2, torchvision0.17.2, Avalanche0.4.0, typing-extensions4.4.0.
- torchmetrics0.11.4,pytorchcv0.0.67,wandb0.15.12,protobuf4.25.8.
- scipy1.15.3,pandas2.2.3,matplotlib3.10.8,scikit-learn1.6.1,grpcio1.62.3.
- Hydra1.3.2, jsonlines4.0.0,kornia0.7.0 are installed. Ray/asynchronous evaluation
  are not used by this focused adapter; no full Hydra/Ray experiment is claimed.

The only approved dependency exception is Torch's typing>=4.8 metadata versus
original Avalanche's required typing4.4. Protocol constructors, optimizer,
torch.func projection, callbacks and resume were executed with4.4 on Python3.10.
Newer grpcio and test-only exceptiongroup produced additional typing conflicts;
grpcio is pinned1.62.3 and the verification test environment uses
pytest8.3.5/exceptiongroup1.2.2. Other dependency conflicts fail the startup gate.
Setuptools private vendored dist-info is excluded from installed-package audit.

Verified actual API construction of Naive/ClassBalancedBuffer and a real
ReplayDataLoader minibatch. Avalanche0.4 concatenates incoming first, replay
second. Baseline retains that exact untagged loader. Geometry-VNA receives
explicit per-sample origin fields (before task labels), independently of order.
Tests permute real samples and check gradient decomposition from one shared
physical forward. Full loader never puts origin wrappers into baseline storage.
Official LPR Git/source hash are checked; its before/after-backward callbacks
execute and change gradients. Original README hyperparameters are recorded.
Geometry weight changes the actual physical projection on the CPU fixture.
Online checkpoint/resume, including correction interruption during head growth,
is checked bitwise. Native LPR endpoint matches the upstream-style online loop.

Kaggle installs the same stack through isolated Python3.10, fetches pinned LPR,
prints versions, audits dependencies and executes these import/API/loader/math
gates before either training lane. CUDA visibility and an actual CUDA operation,
plus the native backbone forward, must pass in that interpreter. A local CPU
PASS does not authorize claiming a Kaggle/CUDA PASS.

## Inputs, lightweight preflight and artifacts

CIFAR100/MNIST attachments are optional with Internet enabled: official
 torchvision downloads/integrity checks are used when absent. Attach original
ImageNet32 files from H3_SMALL_REQUIRED_FILES.json:200 files, shape700×3×32×32,
600 train/100 test per class. No unverified large ImageNet download or fake
fallback is attempted. No old checkpoint, unrelated Output or source ZIP needed.

Preflight validates labels/task order/structure and finite native forwards. H2
also runs clearly labelled CPU fixture gates. No full benchmark is trained by
preflight. A failure writes FAILED_PREFLIGHT and blocks both lanes. Resume uses
strict source/config/data/method identity; optional MNIST gzip copies do not
change canonical raw-file identity. Native deadline/exit75, heartbeat, archives
and continuation after declared divergence remain. H3-small has a separate tag.

Exports: protocols, tasks, learning curves, summaries, interventions, checkpoints,
comparison.csv, curves.csv, accuracy_curves.png, wall/GPU time, peak VRAM and
solver time. H2 adds incoming pre-update online accuracy/replay/feature drift;
H3 adds learning speed,last50 error,candidate counts and actual swaps. Collection
rejects incomplete COMPLETED records; paused/failed performance stays unclaimed
and partial curves keep their status labels. Archive includes preflight evidence.

## Verification and limitations

Focused Python3.10 run:28 passed,12 warnings,47.34s; log
STANDARD_FINALIZE_CPU_TESTS.log. It covers the three focused notebook/protocol
files plus specific H2 shared-penalty/zero-penalty checks, H3 generated-candidate,
transaction/deadline and stored-winner checks, and H4 two-domain resume checks.
A strengthened LPR pause after its preconditioner becomes nonempty and H2
mid-correction resume passed2/2 (19.19s), logged in STANDARD_FINALIZE_H2_RESUME.log.
The normal repository Python3.12 CPU checks passed9 tests,1 deselected (25.22s),
logged in STANDARD_FINALIZE_CPU_LOCAL.log; real Avalanche checks use Python3.10.
All generated cells compile and all paired commands/configurations are checked.
The only warnings in the actual Python3.10 suite are disabled Avalanche loggers. No full
pytest, real dataset training, GPU pilot, live Kaggle installation or archive
transport was run. Dataset download behavior is tested with synthetic loader
fixtures, not by downloading CIFAR/MNIST here. The source ZIP reference was not
available; no ZIP overlay is required by the generated notebooks.

H3-small has60000 native updates/arm, not15million. Exact T4 runtime and required
session count remain unmeasured. Any run may need multiple11-hour sessions;
resume preserves the registered horizon rather than cutting it. These notebooks
are prepared with startup gates; a successful Kaggle GPU run is not claimed.

Regenerate all notebooks:

    .venv/bin/python -m scripts.build_kaggle_standard_compare --commit SOURCE_PIN_TO_FILL
