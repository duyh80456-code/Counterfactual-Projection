# Counterfactual Projection

Look into a larger model without becoming one.

This repository implements the block-local pilot for:

1. asking TINY/Gromo for a temporary rank-r over-expansion beyond a full-width
   ResNet-18;
2. measuring `delta_logits = (f_E(epsilon) - f) / epsilon` without installing
   or training the expansion;
3. solving `(J J^T + mu I) u = delta_logits`, `delta_theta = J^T u` with
   matrix-free JVP/VJP products and conjugate gradient;
4. applying the correction to the original block with unchanged deploy size.

The primary expansion source is the real TINY/Gromo transaction
`candidate.virtual_direction(gate)`. The expanded hidden channels exist only
inside that context. The old convolution-gradient SVD is retained under the
explicit name `GradientLowRankControlProbe`; it is not evidence for the main
hypothesis.

## Quick start

```python
from methods import EProjection

method = EProjection()
step = method.discover_candidate(
    model, counterfactual_tiny_candidate, projection_batch, gate=0.05)
print(step.projection.relative_residual)
print(step.projection.cosine_alignment)
step.projection.apply_(model, scale=0.05)
```

The core invariant is checked on every probe: every parameter and buffer before
the probe must be bitwise identical afterward. Existing gradients and
train/eval modes are also restored.

`StructuralAuxiliarySpace` is currently only a one-dimensional scalar-gate
diagnostic with user-supplied curvature. It is not presented as the full
rank-r auxiliary-space method. Likewise, `ERepOpt` remains a gradient-SVD
control because structural TINY candidates do not yet expose the rank-r
coordinates needed by that optimizer.

## Test

```bash
python3 -m pytest
```

## Kaggle T4 x2 unified three-seed experiment

The current comparison uses three separately generated, protocol-identical
notebooks:

- [`notebooks/kaggle_unified_seed1_end_to_end_t4x2.ipynb`](notebooks/kaggle_unified_seed1_end_to_end_t4x2.ipynb)
- [`notebooks/kaggle_unified_seed2_end_to_end_t4x2.ipynb`](notebooks/kaggle_unified_seed2_end_to_end_t4x2.ipynb)
- [`notebooks/kaggle_unified_seed3_end_to_end_t4x2.ipynb`](notebooks/kaggle_unified_seed3_end_to_end_t4x2.ipynb)

Each starts a randomly initialized CIFAR-ResNet18 at epoch 0. No theta150 or
theta300 input is accepted. Base training uses a metric-independent CIFAR
recipe: SGD for 200 recipe epochs, LR 0.1 with MultiStep drops at epochs 100 and
150 (`gamma=0.1`), momentum 0.9, and weight decay 5e-4. Trigger accuracy never
controls LR. Every exact trigger-set best is a complete resumable checkpoint.
The final stall clock is armed only after the base recipe completes; only a
+0.1 pp gain resets its 100-epoch patience. Once stalled, the already observed
100 post-recipe epochs are the Vanilla control
and the exact best checkpoint is forked into recurrent E-driven O, scaled
Bypass 70/30, and recurrent O-only, each with a 100-SGD-epoch budget.

All configuration, splits, schedules, GPU placement, checkpoint fields, and
metrics are identical across notebooks; only the seed and output directory
differ. GPU0 runs E-driven O. GPU1 runs Bypass and then starts O-only in a fresh
process from the same checkpoint hash. Per-epoch histories plus latest, best,
fork, and intervention checkpoints are retained for later plots and resume.

## Legacy shared-checkpoint runs

The earlier theta300 workflow remains available for reproducing pilot runs:

For the simplest Kaggle workflow, use the single-file
[`notebooks/kaggle_plateau_end_to_end_t4x2.ipynb`](notebooks/kaggle_plateau_end_to_end_t4x2.ipynb).
It accepts theta300 plus CIFAR-100, saves every exact Vanilla trigger-set best,
and waits for 100 epochs without a significant +0.1 pp trigger improvement
before launching the method jobs
from that exact best checkpoint. The already-observed 100-epoch Vanilla stall
trajectory is reused as the control rather than trained twice. If epoch 500 is
reached without a stall, attach its output and rerun with a larger `MAX_EPOCH`;
both the latest and best Phase-1 states resume.

The same workflow is also available as two explicit notebooks:

1. [`notebooks/kaggle_vanilla_to_plateau.ipynb`](notebooks/kaggle_vanilla_to_plateau.ipynb)
   trains only Vanilla from shared theta300 while preserving the checkpoint LR.
   The held-out pool is split into 2,000 trigger/selection and 3,000 report-only
   evaluation samples. Every exact trigger improvement is saved as a complete
   `checkpoint_best.pt`, while only a +0.1 pp trigger improvement resets the
   stall clock. After 100 epochs without such a significant improvement, the
   run creates `plateau_checkpoint.pt` from that exact best state rather than
   from the later potentially degraded model. Epoch 500 remains a review point;
   both latest progress and the best checkpoint are required when resuming.
2. [`notebooks/kaggle_plateau_fork_t4x2.ipynb`](notebooks/kaggle_plateau_fork_t4x2.ipynb)
   launches supervised O-only, scaled matched-horizon Bypass, and E→O, each for
   a 100-SGD-epoch budget. E→O and O-only run recurrent 10-epoch trials: if no
   new best appears, trainable state rolls back to the arm's best checkpoint,
   the stochastic stream remains advanced, and a fresh intervention starts.
   Vanilla is reused from
   Phase 1. All methods inherit the same model, optimizer, scheduler, momentum,
   RNG, training order, and held-out split. Every trained method saves latest
   and exact-trigger-best checkpoints and reports fork/best/final evaluation
   accuracy, loss, deltas, epoch to best, wall time, peak memory/parameters,
   and expanded time. GPU0 runs E→O to completion. GPU1 runs Bypass with a
   70-epoch opt1 and at most 30-epoch opt2, then starts O-only as a fresh
   process from the same hashed theta_best checkpoint. The scaled Bypass
   penalty switches from ×1 to ×10 at opt2 epoch 15; this is explicitly a
   matched-horizon scaling, not an exact reproduction of the native schedule.
   An uncontracted Bypass run
   is retained diagnostically but not presented as a completed comparator.

For the exact-original-schedule plateau experiment from shared theta-300,
use [`notebooks/kaggle_plateau_eo_t4x2.ipynb`](notebooks/kaggle_plateau_eo_t4x2.ipynb).
It restores the exact model, optimizer, scheduler, SGD momentum, RNG, data
indices, and loader state for both arms. No LR, `initial_lr`, `T_max`, or
scheduler state is changed. The run stops at the horizon encoded in the source
scheduler rather than stepping beyond it. The original training indices remain
unchanged; the pre-existing 5,000-example held-out validation pool is split
into 2,000 trigger and 3,000 evaluation examples. E selects WHERE from observed
structural expansion loss gain across all eight blocks; projection runs only
for that winner and is applied only when a separate gate batch accepts a
line-search scale. The official test set is not used. A theta300→500 schedule
would be a separate matched extended-training experiment, not this exact
original-training baseline.

To run only the current main algorithm from an existing theta-300 warm-up,
use [`notebooks/kaggle_adaptive_e_driven_o.ipynb`](notebooks/kaggle_adaptive_e_driven_o.ipynb).
It requires only CIFAR-100 plus the matching `shared_seed1_epoch300.pt/json`,
does not run any comparison arm or warm-up, and writes a complete resumable
adaptive-arm checkpoint after every epoch.

Use
[`notebooks/kaggle_counterfactual_projection_t4x2.ipynb`](notebooks/kaggle_counterfactual_projection_t4x2.ipynb).
Enable **T4 x2**, attach CIFAR-100 with a `cifar-100-python` directory, and add
the private-repository token as the Kaggle Secret `github_token`. The notebook
also requires the completed `shared_seed1_epoch300.pt/json` pair as input; it
does not accept theta-150 or silently rebuild the warm-up.

The comparison uses one randomly initialized full-width CIFAR-ResNet18 at
32x32. Vanilla training produces the sole fork checkpoint
`warmup/shared_seed1_epoch300.pt`, containing the model, SGD, cosine scheduler,
epoch, exact train/validation/tuning indices, loader-generator state, and all
RNG states. Its SHA-256 is checked by every arm.

The shared epoch-300 checkpoint retains its full optimizer, RNG, split, and
loader state. At the fork, every arm preserves the checkpoint LR and starts the
same 60-epoch cosine segment that reaches zero at epoch 360. This is explicitly
a post-fork/rebased LR schedule and is not equivalent to training from
initialization with one `CosineAnnealingLR(T_max=360)`.

All four trajectories therefore have the same 360-epoch budget:

```text
theta_300
  +-- vanilla_continue: 60 epochs
  +-- ours_e_driven_o: 60 epochs (WHEN-WHERE-HOW adaptive selection)
  +-- bypass: opt1 40, opt2 <= 20, train3 after early contraction
  +-- o_projection_only: 60 epochs (projection-only supervised control)
```

Wave 1 runs Ours on GPU 0 and Bypass on GPU 1. Wave 2 runs Vanilla on GPU 0 and
O/projection-only on GPU 1. Projection-only uses no virtual expansion: its
functional target is the negative summed-cross-entropy logit gradient
`one_hot(y) - softmax(f(x))`, fitted through the same residual-path projector
and application gate as Ours. It is a supervised projector control, not merely
“Ours minus E”; a win would show that the supervised functional target is
strong, not by itself prove that structural E is useless. Each arm atomically
saves `checkpoint_latest.pt` after
every epoch, including model, optimizer, scheduler, complete history, exact
split indices, RNG, phase, and loader-generator state. Burn-in likewise writes
a full rolling `shared_seed1_progress.pt`; a new Kaggle session can resume by
attaching the previous output as an input dataset. The official CIFAR-100 test
set is never constructed.

At each Ours intervention, `--site auto` evaluates rank-4 TINY proposals at all
eight growing residual blocks using exactly the same statistics batches. Raw
`proposal_score` is logged but does not filter the main search. On a separate
16-sample selection batch, every candidate produces a structural functional
direction and a cheap 25-CG residual-path projection. The main selector maximizes the
projected loss utility `mean((one_hot(y) - p) * J delta_theta)` rather than the
raw TINY score. Full projection runs only when the best finite projected
utility is positive; otherwise that epoch performs normal SGD only.
Projectability rho, residual, and cosine remain solver diagnostics rather than
scientific thresholds. Per-epoch diagnostics preserve all raw scores, expansion
utilities, projectabilities, projected utilities, the WHEN decision, and the
selected site. `--site-selection-mode tiny_score_argmax` retains the old TINY
argmax ablation; `--site-selection-mode fast_topk_projectability` retains the
top-k plus rho-threshold runtime approximation; and a concrete
`--site stages.2.blocks.0` retains fixed-site E-to-O. Full O uses epsilon 0.05,
residual-path scope, 200 CG iterations, and the existing held-out application
gate.

Bypass follows the relaxed residual-network form of Algorithm 1: every ReLU in
the residual stages is embedded as `ReLU(x) + D x` with `D=0`; opt1 trains task
loss in the extended space; opt2 adds `gamma(t) * sum(||D||)` and projects
activations whose contraction norm reaches epsilon; the final projection drops
the remaining D coordinates and train3 continues in the original ResNet. The
arm is a matched-60-epoch-budget relaxed Bypass control, not the full long-opt1
ResNet reproduction from the paper. It uses 40 opt1 epochs and at most 20 opt2
epochs. If contraction succeeds early, projection occurs and train3 consumes
the remainder. If contraction has not succeeded at epoch 360, the expanded
checkpoint is preserved with `bypass_completed=false`; it is never forcibly
projected. Such a run is marked `accuracy_comparison_eligible=false` and is
reported as budget-exhausted diagnostic output, not as a completed Bypass
accuracy comparator. The shared
SGD/cosine optimizer is pilot scaling, not the paper's original long-run Adam
hyperparameter schedule.

The comparison reports both final and best validation accuracy. Primary deltas
are anchored to the validation metrics measured directly at the shared fork:
`validation_accuracy_delta = final_accuracy - accuracy(theta_300)` and
`validation_loss_delta = loss(theta_300) - final_loss`. Best-accuracy and
best-loss deltas are reported separately so transient improvements are not
hidden by the final epoch.

## Reference implementation

The local, ignored checkout at `third_party/One-Shot-TAS-CCIL` points to
`duyh80456-code/One-Shot-TAS-CCIL`, branch `ccil-residual-capacity`. Ideas
reused at the interface level are transactional virtual directions, explicit
RepOpt gradient handlers, and invariant-focused tests. No Gromo source is
copied into this project.
