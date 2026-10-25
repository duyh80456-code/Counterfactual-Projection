# P1 seed 2 with the amended policy

Current delivery uses repository code pinned to the published code commit,
without a source ZIP input. `SOURCE_DIGEST` is SHA-256 of `git:<pinned-commit>`;
bootstrap verifies HEAD against the pin before recording `git_commit_sha256`
provenance. Attach CIFAR-100 and the original baseline Output. ZIP instructions
below describe the earlier delivery and are superseded by this Git workflow.

Use `notebooks/vna4_M1s2.ipynb` with its matching digest-verified source ZIP.
This supersedes the earlier notebook pinned to 84f833d without an overlay.

- Train only `icarl_full` seed 2: noBN/SkipInit ResNet-32, Inc5, all 20 tasks,
  superclass order, memory 2000, LR .05, first/later epochs 100/50, no warmup,
  no clipping, final-epoch model, Gram/power10.
- Policy: `schur_t32_sharpness_trust_v3`, with the same scientific engine
  implementation as the amended NB1 source.
- GPU0 performs seed-0 preflight and then full seed 2. GPU1 has no training job.
- Output: `outputs/p1_inc5_full_v3_s2/icarl_full__s2/`.

## Inputs and baseline

Attach CIFAR-100, the matching new source ZIP/dataset, and the old P1/M1 Output
containing the seed-2 baseline outcome. No old seed-0 authority is required.
Attach this notebook's own previous Output when resuming a session.

Restore only `icarl__nobn__superclass__s2` or `icarl__s2` into an isolated baseline
directory. Validate source provenance, upstream/config, actual dataset hashes,
class order, epochs/milestones, no warmup/clipping and persisted protocol fields.
Completed references require 20 tasks with finite CNN/NME. The registered failed
reference instead requires `FAILED_DIVERGED`, one completed task, and recorded
task 2 / epoch 1 / step 8 with a nonempty tensor identifier. Metrics after
divergence are not required. Store the original outcome and coordinates in
`baseline_reuse__s2.json`; do not use this failed reference to estimate complete
training time or imply that all 20 baseline tasks finished. The inspected original M1 commit
84f833db7b152264003c26b236c5c628c64b8773 with no overlay is accepted only for
this baseline; previously registered P1 overlay sources remain supported.
Preserve baseline bytes/source and record hashes in `baseline_reuse__s2.json`.
Do not train a baseline or fabricate its missing historical measurements.

## New common authority

The locked authority is archived at
`outputs/p1_inc5_full_v3_authority/resnet32_nobn_inc5_seed0.json`, together with
its resumable seed-0 calibration under `_preflight_seed0/`.
Verify policy, source digest/commit, noBN LR scale, superclass order, dataset
fingerprint, init_cls=increment=5 and calibration_seed=0 before reuse. This is
the common authority for future amended-policy Inc5 seeds using the same source
bundle. Do not read, convert or fall back to the old `gram_power10.json`.

## Measurements and runtime

Tracker runs on every task (r=0..10, each five updates to 200, then epochs).
GGN power10 uses the native mixed new/exemplar window and CE+KD at
r=0,1,2,5,10,20,50,100,200, on a copy with bitwise live-state verification.
Native `run/interventions.json` and root `intervention_log.json` retain VNA/GIR
certificates, loss checks, sharpness decisions, rejection and swap trials.
Native epoch checkpoints and exit-75/deadline handling remain unchanged.

Class-dependent timing estimates use the new calibration, exclude re-training
the reused baseline and warn above 11 hours without shortening tasks. A legacy
baseline's timing can omit Tracker/GGN overhead, so the estimate is not an upper
bound. No T4 run or commit was performed for this notebook delivery.
