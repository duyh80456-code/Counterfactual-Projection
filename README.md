# Counterfactual Projection

Look into a larger model without becoming one.

This repository implements the first block-local pilot for:

1. creating a temporary rank-r expansion direction at a `Conv2d` block;
2. measuring its functional change `delta_logits` without installing or
   training the expansion;
3. solving `(J^T J + mu I) delta_theta = J^T delta_logits` with matrix-free
   JVP/VJP products and conjugate gradient;
4. applying the correction to the original block, with unchanged deploy size.

The initial expansion source is a one-backward-pass truncated-SVD direction of
the negative convolution weight gradient. `probe/gromo_adapter.py` is the
boundary for plugging in a TINY/Gromo candidate from the reference repository.

## Quick start

```python
from methods import EProjection

method = EProjection()
step = method.discover(model, (images, labels), block="layer3.1.conv2", rank=4)
print(step.projection.projection_ratio)
print(step.projection.relative_residual)
step.projection.apply_(model)
```

The core invariant is checked on every probe: every parameter and buffer before
the probe must be bitwise identical afterward. Existing `.grad` tensors and
train/eval modes are also restored.

## Test

```bash
python3 -m pytest
```

## Reference implementation

The local, ignored checkout at `third_party/One-Shot-TAS-CCIL` points to
`duyh80456-code/One-Shot-TAS-CCIL`, branch `ccil-residual-capacity`, commit
`ec3ebc4ca7e468835935848016233204e0025b4a`. Ideas reused at the interface
level are transactional virtual directions, explicit RepOpt gradient handlers,
and invariant-focused tests. No Gromo source is copied into this project.

