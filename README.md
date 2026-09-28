# Counterfactual Projection

Look into a larger model without becoming one.

This repository implements the first block-local pilot for:

1. starting from a full-width ImageNet-pretrained ResNet-18 and asking
   TINY/Gromo for a temporary rank-r over-expansion beyond its target width;
2. measuring the finite-difference direction
   `delta_logits = (f_E(epsilon) - f) / epsilon` without installing or
   training the expansion;
3. solving `(J^T J + mu I) delta_theta = J^T delta_logits` with matrix-free
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

The structural gate schedules `vanilla`, norm-matched `random_projection`,
`tiny_projection`, `expand_train_project` (explicitly a RepAn/Bypass-like
control, not a reproduction), and `real_e_oracle` for two seeds. The main arm
sweeps epsilon over `0.01`, `0.05`, and `0.1`. TINY statistics, functional
projection, and held-out checks use distinct batches. Statistics and projection
batches are freshly sampled at every intervention from the same training pool
used by every arm; only the fixed check batch is held out. Two seed-specific
warm-up checkpoints include both model and SGD state, and every arm starts from
the exact same checkpoint hash for its seed. A dynamic queue gives each GPU one
independent arm at a time. Only the oracle may increase deploy parameters.
Outputs are
restart-safe at the completed-arm level and are aggregated into `summary.json`
plus a downloadable `.tar.gz` archive.

The input pipeline uses ImageNet normalization and resized CIFAR-100 images so
the pretrained backbone sees its expected input distribution. The deploy model
starts at full `64/128/256/512` width (`missing_neurons() == 0`); a
`CounterfactualTinyProbe` temporarily raises only the selected block's target
from `current_width` to `current_width + rank` while TINY constructs E, then
restores the configured target.

Each intervention logs TINY statistics/solve time, projection time, JVP/VJP
counts, CG iterations, peak allocated GPU memory, and any SGD momentum states
reset after a direct projected parameter jump. The real-E oracle intervenes at
the same epoch frequency as the main method and logs structural and projected
local gains before every irreversible commit.

## Reference implementation

The local, ignored checkout at `third_party/One-Shot-TAS-CCIL` points to
`duyh80456-code/One-Shot-TAS-CCIL`, branch `ccil-residual-capacity`, commit
`ec3ebc4ca7e468835935848016233204e0025b4a`. Ideas reused at the interface
level are transactional virtual directions, explicit RepOpt gradient handlers,
and invariant-focused tests. No Gromo source is copied into this project.
