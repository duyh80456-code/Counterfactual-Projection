# Counterfactual Projection

Look into a larger model without becoming one.

This repository implements the first block-local pilot for:

1. asking TINY/Gromo for a temporary rank-r hidden-width expansion;
2. measuring its functional change `delta_logits` without installing or
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
step = method.discover_candidate(model, tiny_candidate, (images, labels))
print(step.projection.fitted_norm_ratio)
print(step.projection.relative_residual)
print(step.projection.cosine_alignment)
step.projection.apply_(model)
```

The core invariant is checked on every probe: every parameter and buffer before
the probe must be bitwise identical afterward. Existing `.grad` tensors and
train/eval modes are also restored.

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
`tiny_projection`, `expand_train_project` (RepAn/Bypass-like), and
`real_e_oracle` for two seeds. A dynamic queue gives each GPU one independent
arm at a time. Only the oracle may increase deploy parameters. Outputs are
restart-safe at the completed-arm level and are aggregated into `summary.json`
plus a downloadable `.tar.gz` archive.

This first structural gate uses the audited Gromo CIFAR ResNet because its
hidden-width transaction is exact. The separate torchvision control runner
uses ImageNet normalization for pretrained weights, but its gradient-SVD arms
are controls rather than the primary method.

## Reference implementation

The local, ignored checkout at `third_party/One-Shot-TAS-CCIL` points to
`duyh80456-code/One-Shot-TAS-CCIL`, branch `ccil-residual-capacity`, commit
`ec3ebc4ca7e468835935848016233204e0025b4a`. Ideas reused at the interface
level are transactional virtual directions, explicit RepOpt gradient handlers,
and invariant-focused tests. No Gromo source is copied into this project.
