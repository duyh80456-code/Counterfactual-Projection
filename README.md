# Counterfactual Projection

Look into a larger model without becoming one.

This repository implements the first block-local pilot for:

1. starting from a full-width ImageNet-pretrained ResNet-18 and asking
   TINY/Gromo for a temporary rank-r over-expansion beyond its target width;
2. measuring the finite-difference direction
   `delta_logits = (f_E(epsilon) - f) / epsilon` without installing or
   training the expansion;
3. solving the equivalent dual ridge system
   `(J J^T + mu I) u = delta_logits`, `delta_theta = J^T u`, with matrix-free
   JVP/VJP products and conjugate gradient;
4. applying the correction to the original block, with unchanged deploy size.

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
print(step.projection.fitted_norm_ratio)
print(step.projection.relative_residual)
print(step.projection.cosine_alignment)
step.projection.apply_(model, scale=0.05)
```

The core invariant is checked on every probe: every parameter and buffer before
the probe must be bitwise identical afterward. Existing `.grad` tensors and
train/eval modes are also restored.

`StructuralAuxiliarySpace` is currently only a one-dimensional scalar-gate
diagnostic with user-supplied curvature. It is not presented as the full
rank-r auxiliary-space method. Likewise, `ERepOpt` remains a gradient-SVD
control: structural TINY candidates do not yet expose the rank-r coordinates
needed by that optimizer.

## Test

```bash
python3 -m pytest
```

## Kaggle T4 x2 pilot

Use
[`notebooks/kaggle_counterfactual_projection_t4x2.ipynb`](notebooks/kaggle_counterfactual_projection_t4x2.ipynb).
Before running it:

1. enable the **T4 x2** accelerator;
2. attach a CIFAR-100 dataset containing `cifar-100-python`;
3. create a Kaggle Secret named `github_token` with read access to this repo.

The notebook never embeds the token in the clone URL or prints it. It creates a
short-lived `GIT_ASKPASS` helper and deletes it immediately after cloning.

The structural gate schedules `vanilla`, `vanilla_matched_compute` (runs and
discards the exact TINY + projection computation), `vanilla_extra_sgd` (uses
the same extra labeled samples for ordinary SGD), `vanilla_momentum_reset`,
global-norm-matched `random_projection`, tensor-norm-preserving
`sign_randomized_projection`, `tiny_projection`,
`tiny_projection_conv_only` and `tiny_projection_whole_block` (ablations),
`expand_train_project` (jointly trains the temporary `theta + theta_E` space;
still a RepAn/Bypass-like control rather than an official reproduction), and
`real_e_growth` for two seeds. The main arm sweeps epsilon
over `0.01`, `0.05`, and `0.1`. TINY statistics, functional
projection, and held-out checks use distinct batches. Statistics and projection
batches are freshly sampled at every intervention from the same training pool
used by every arm; only the fixed check batch is held out. Two seed-specific
warm-up checkpoints include both model and SGD state, and every arm starts from
the exact same checkpoint hash for its seed. A dynamic queue gives each GPU one
independent arm at a time. Only the committed-E growth control may increase
deploy parameters.
Outputs are
restart-safe at the completed-arm level and are aggregated into `summary.json`
plus a downloadable `.tar.gz` archive.
The PCG revision keeps the existing `fair_v15` output root: shared warm-up
model/optimizer checkpoints are reused, while result files carrying the older
solver configuration are archived as `result.pre_dual_pcg.json` and only those
post-warm-up arms are rerun.
The notebook pins the main checkout, `gromo/src`, and the One-Shot-TAS checkout
in both the live kernel's `sys.path` and every child process's `PYTHONPATH`.
It asserts the resolved source path of `probe`, `gromo`, and `dual_growth`
before constructing a model; editable installs performed after kernel startup
are otherwise visible only to newly launched Python processes.

The input pipeline uses ImageNet normalization and resized CIFAR-100 images so
the pretrained backbone sees its expected input distribution. The deploy model
starts at full `64/128/256/512` width (`missing_neurons() == 0`); a
`CounterfactualTinyProbe` temporarily raises only the selected block's target
from `current_width` to `current_width + rank` while TINY constructs E, then
restores the configured target.

The fixed tuning batch and the larger validation split may be observed during
development. Epsilon is selected only by validation accuracy. No development
arm constructs the official CIFAR-100 test set. After every choice is frozen,
all selected method configurations are rerun and evaluated on test once.

Each intervention logs TINY statistics/solve time, projection time, JVP/VJP
counts, CG convergence and residual norms (including iterations
12/25/50/100/200),
peak allocated GPU memory, and any SGD momentum states reset after a direct
projected parameter jump. CG runs for at most 200 iterations and an unconverged
correction is never applied. If the requested damping does not converge, the
solver retries transparently through at most `10000x` damping; every attempt,
residual, and effective damping is logged, and only a converged final
attempt may be applied. The dual output-space system is algebraically
equivalent to the parameter-space normal equation for positive damping, but
avoids the poorly scaled `J^T delta_logits` right-hand side and a CG vector with
millions of block parameters. Before CG, the structural target is normalized
to unit norm and the solved parameter direction is scaled back afterward. This
leaves the ridge solution unchanged while preventing nearly
function-preserving E signals from falling below float32 numerical scale. The
network JVP/VJP remains in the model's native dtype, while the small dual CG
vectors, dot products, and recurrence use float64 to prevent loss of Krylov
conjugacy on the real ResNet operator. The pilot uses matrix-free
preconditioned CG with an eight-probe Hutchinson estimate of
`diag(J J^T)`, and accepts convergence by the explicit relative linear-system
residual `||b - A u|| / ||b|| <= 1e-2`; both the achieved ratio and the
preconditioner configuration are logged. This tolerance concerns the inner
ridge solve only—held-out tangent and realized nonlinear functional residuals
remain the scientific fit metrics. Gromo's optional forward caches are
disabled transactionally during JVP/VJP and restored afterward. Registered
parameters and buffers are likewise restored by object identity, preventing
functorch tensor wrappers from leaking into a later real-growth commit. The
real-E growth control intervenes at
the same epoch frequency as the main method and logs structural and projected
local gains before every irreversible commit. Real-E growth commits synchronize the
new current width back to the target width before the next intervention.

The primary projection scope contains the two residual-path convolutions and
their BatchNorm affine parameters. Shortcut/downsample parameters are excluded.
Conv-only and whole-block projection are reported separately as ablations. The
main fit metric is evaluated by applying the fitted parameter tangent on the
unseen tuning batch and comparing it with that batch's independently measured
structural delta; fit-batch residuals are secondary diagnostics.
After the parameter jump, the runner also measures the realized nonlinear
direction `(f(theta + scale * delta_theta) - f(theta)) / scale` on that held-out
batch and reports its residual and cosine against the same structural target.
The momentum-reset control targets that same residual conv+BN parameter set.
`expand_train_project` measures its held-out target from the trained temporary
expansion itself, before the expansion transaction is removed. It rolls the
temporary base-parameter update back before projecting that functional delta
at the original fixed-size state. Its temporary optimization runs in train mode
and uses SGD with the main run's learning rate, momentum 0.9, weight decay
`5e-4`, and copied base-parameter momentum state. Its two-step/32-sample budget
is logged and remains a diagnostic control, not an official reproduction. All controls
whose behavior depends on the finite-difference E gate use the epsilon selected
for the main arm in the final frozen comparison; `expand_train_project` remains
the explicitly defined gate-1 trained-expansion control.

Each run reports `corrections_applied / correction_attempts` and its application
rate; the Kaggle gate rejects any projection arm below 100%.

The committed notebook is an intentionally small mechanism pilot (12k training
examples and 3 warm-up + 3 intervention epochs). It can establish whether the
projection behaves coherently, but must not support performance-superiority
claims. Such claims require a separately frozen full-CIFAR-100, long-schedule
experiment after this gate passes.

The Kaggle test gate includes a real CUDA integration test of the complete
full-ResNet → TINY over-expansion → delta-f_E → functional-projection path; it
cannot silently skip that test. Candidate FLOPs are recomputed from runtime
feature-map sizes and convolution kernels rather than using TINY's legacy
32x32 CIFAR estimate.

The current pilot deliberately fixes the predeclared `stages.2.blocks.0` site.
For the subsequent generalized experiment, `--site auto` probes either every
growing block or the comma-separated `--candidate-sites` list on the training
statistics batch and selects `argmax_l proposal_score(l)` without consulting
tuning, validation, or official-test data.

## Reference implementation

The local, ignored checkout at `third_party/One-Shot-TAS-CCIL` points to
`duyh80456-code/One-Shot-TAS-CCIL`, branch `ccil-residual-capacity`, commit
`ec3ebc4ca7e468835935848016233204e0025b4a`. Ideas reused at the interface
level are transactional virtual directions, explicit RepOpt gradient handlers,
and invariant-focused tests. No Gromo source is copied into this project.
