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

## Kaggle T4 x2 shared-checkpoint run

To run only the current main algorithm from an existing theta-300 warm-up,
use [`notebooks/kaggle_adaptive_e_driven_o.ipynb`](notebooks/kaggle_adaptive_e_driven_o.ipynb).
It requires only CIFAR-100 plus the matching `shared_seed1_epoch300.pt/json`,
does not run any comparison arm or warm-up, and writes a complete resumable
adaptive-arm checkpoint after every epoch.

Use
[`notebooks/kaggle_counterfactual_projection_t4x2.ipynb`](notebooks/kaggle_counterfactual_projection_t4x2.ipynb).
Enable **T4 x2**, attach CIFAR-100 with a `cifar-100-python` directory, and add
the private-repository token as the Kaggle Secret `github_token`.

The comparison uses one randomly initialized full-width CIFAR-ResNet18 at
32x32. Vanilla training produces the sole fork checkpoint
`warmup/shared_seed1_epoch300.pt`, containing the model, SGD, cosine scheduler,
epoch, exact train/validation/tuning indices, loader-generator state, and all
RNG states. Its SHA-256 is checked by every arm.

The fresh epoch-150 checkpoint is resumed with its full optimizer, RNG, split,
and loader state. Vanilla then continues for another 150 epochs. The LR at
epoch 150 is preserved and a new cosine segment anneals it to zero at epoch
350. This is explicitly a two-stage/rebased LR schedule and is not equivalent
to training from initialization with one `CosineAnnealingLR(T_max=350)`.

All four trajectories therefore have the same 350-epoch budget:

```text
theta_300
  +-- vanilla_continue: 50 epochs
  +-- ours_e_driven_o: 50 epochs (WHEN-WHERE-HOW adaptive selection)
  +-- bypass: opt1 20, opt2 until contraction, train3 for the remainder
  +-- o_projection_only: 50 epochs (projection-only supervised control)
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
`proposal_score` only pre-screens the top three. On a separate 16-sample
selection batch, each top candidate produces a structural functional direction
and a cheap 25-CG residual-path projection. The main selector maximizes the
projected loss utility `mean((one_hot(y) - p) * J delta_theta)` rather than the
raw TINY score. Full projection runs only when the winner has positive utility
and fitted-norm projectability at least 0.05; otherwise that epoch performs
normal SGD only. Per-epoch diagnostics preserve all raw scores, top-k expansion
utilities, projectabilities, projected utilities, the WHEN decision, and the
selected site. `--site-selection-mode tiny_score_argmax` retains the old TINY
argmax ablation, while a concrete `--site stages.2.blocks.0` retains fixed-site
E-to-O. Full O uses epsilon 0.05, residual-path scope, 200 CG iterations, and
the existing held-out application gate.

Bypass follows the relaxed residual-network form of Algorithm 1: every ReLU in
the residual stages is embedded as `ReLU(x) + D x` with `D=0`; opt1 trains task
loss in the extended space; opt2 adds `gamma(t) * sum(||D||)` and projects
activations whose contraction norm reaches epsilon; the final projection drops
the remaining D coordinates and train3 continues in the original ResNet. The
arm is a matched-50-epoch-budget relaxed Bypass control, not the full long-opt1
ResNet reproduction from the paper. The
Epoch 10 of opt2 is a soft warning boundary, never a forced projection. If the
criterion is still false, opt2 continues within the remaining 50-epoch budget.
If contraction still has not succeeded at epoch 350, the expanded checkpoint is
preserved but aggregation rejects Bypass as an invalid comparator. The shared
SGD/cosine optimizer is pilot scaling, not the paper's original long-run Adam
hyperparameter schedule.

## Reference implementation

The local, ignored checkout at `third_party/One-Shot-TAS-CCIL` points to
`duyh80456-code/One-Shot-TAS-CCIL`, branch `ccil-residual-capacity`. Ideas
reused at the interface level are transactional virtual directions, explicit
RepOpt gradient handlers, and invariant-focused tests. No Gromo source is
copied into this project.
