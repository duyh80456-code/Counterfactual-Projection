# Research references

- `One-Shot-TAS-CCIL`: https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git
- inspected branch: `ccil-residual-capacity`
- inspected commit: `ec3ebc4ca7e468835935848016233204e0025b4a`

The checkout is intentionally ignored. Clone it into
`third_party/One-Shot-TAS-CCIL` when the optional adapter or audit trail is
needed.

## Official comparison sources

The four-arm Kaggle notebook clones comparison implementations at fixed
revisions instead of vendoring or rewriting them:

- RepAn: `https://github.com/xfey/RepAn.git` at
  `7cb05e93cdcd8cd83f18da1b5514bad75bc68e16`. That revision has no license
  file, so the run manifest records its license as unspecified.
- ExpandNets: `https://github.com/GUOShuxuan/expandnets.git` at
  `065d4d3aebfeb442c02227d1d5c16ee11a518945` (BSD-3-Clause terms in a file
  whose heading says "MIT License"; the manifest preserves this ambiguity).
- RepOptimizers: `https://github.com/DingXiaoH/RepOptimizers.git` at
  `2e45ff5388e9d7aabf112d7e2973df8183e6c6d9` (MIT).

Only data/budget/checkpoint/metric wrappers live in this repository. External
operators are imported from those checkouts at runtime.

## Stage 1 VNA / TINY alignment

Gromo uses the official `https://github.com/growingnet/gromo` revision
`8d19107b61a9459a9021065a329b699adcb0f25b` (BSD-3-Clause, installed
`gromo-0.0.1.dist-info/licenses/LICENCE.md`). The installed VCS metadata verifies
this pin; existing notebooks use the same detached checkout and install its
package and declared dependencies.

For the non-preactivation CIFAR BasicBlock, A is
`first_layer.extended_output_layer`, B is
`second_layer.extended_input_layer`; the virtual activation is ReLU with no BN:
`GrowingBatchNorm.extended_forward` passes extension channels through unchanged
(`gromo/modules/growing_normalisation.py:270-296`). B enters before the original
second BN. The native candidate forward and extracted-A forward are tested for
exact agreement; VNA treats B as a variable at zero, ignoring native B and delta.
Counterfactual proposals override target width to current width plus requested
rank (`probe/counterfactual_tiny.py:63-84`).

**Spec discrepancy:** sum-reduction CE does not imply batch-size-scaled TINY
values. Native S/M statistics sum spatial positions but divide by sample count
(`gromo/utils/tensor_statistic.py:122-140`,
`gromo/modules/conv2d_growing_module.py:998-1061,2064-2083`). The payload contains
retained **singular values**, not squares; native threshold/rank limits and the
adapter's effective-rank slice truncate the spectrum
(`gromo/utils/tools.py:304-339`, `tiny_adapter.py:489-522`). Stage 1 leaves these
in native units and uses mean CE independently for VNA. Statistics force eval
mode (`gromo/utils/training_utils.py:375-387`); standard tracked BN does not update
running buffers in this pinned path. The transactional wrapper still restores
all state buffers, gradients and modes (`probe/gromo_adapter.py:89-107`).

See `docs/vna/STAGE1_REPORT.md` for full file:line evidence, empirical scaling,
noise/cost measurements, limitations and CPU tests. Paths under `gromo/` above
are relative to the pinned installed package, and `tiny_adapter.py` is under
`third_party/One-Shot-TAS-CCIL/dual_growth/adapters/`.

## Stage 2-core K-FAC library decision and TINY cotangent alignment

Audited official sources on 2026-10-14 (no dependency or environment changes):

- [curvlinops](https://github.com/f-dangel/curvlinops/tree/34783a5f923639305e508605dd6263838b20e3a8),
  commit `34783a5f923639305e508605dd6263838b20e3a8`, MIT. Its declared
  torch floor is 2.0, so this is **not a proven torch-2.2 incompatibility**.
  The documented supported KFAC/EKFAC parameter modules are Linear and Conv2d,
  and the documented operator lifecycle caches factors for a fixed model:
  [official EKFAC documentation](https://curvlinops.readthedocs.io/en/stable/_modules/curvlinops/ekfac.html).
  Gromo's residual scope also includes growing BN affine parameters; the task
  requires their true diagonal GGN, explicit EMA and factor/eigenbasis state
  refresh/load. Those are not a drop-in supported full-scope operator here.
  Source was downloaded only to `/tmp/stage2-curvlinops-audit` for inspection.
  BackPACK, einconv, linear_operator and curvlinops are absent in the pinned
  Phase-0 environment. No end-to-end library compatibility claim is made.
- [ASDL](https://github.com/kazukiosawa/asdl/tree/b7afcaabd6d9c4501f1466f8d4610d238bfc2e87),
  commit `b7afcaabd6d9c4501f1466f8d4610d238bfc2e87`, MIT. Its official README
  advertises compatibility with torch 1.13; no validated torch-2.2/Gromo
  integration was found. It is not installed or used.

Decision: use the task's explicitly allowed minimal-factor fallback in
`methods/vna_kfac.py`, with no copied library source. This avoids introducing
unvalidated hooks/dependencies into the live Phase-0 environment and supplies
BN diagonal blocks and explicit refresh/serialization in one tested operator.
This is a conservative integration decision, not evidence that either library
can never support Gromo with additional adapters. Reconsider for Stage 2a.

Factors are from **exact softmax GGN**, summed over per-sample class-Hessian
square-root columns, independently of observed labels. A uses unfolded patch
covariances; G uses output-gradient covariances. Optional EKFAC correction uses
the exact diagonal GGN in the Kronecker eigenbasis; Tikhonov damping adds mu
once to eigenvalues. Cross curvature is never factored. CPU tests compare
uncorrected eigenbasis inversion with dense `G ⊗ A`, check the corrected
spectrum against an independent dense GGN, and check symmetry/descent and
serialization. Refresh cost is intentionally larger than MC true-Fisher.

`methods/vna_select.py` reuses pinned Gromo statistics and optimal-update solve
at the existing pin above. A pseudo-loss `sum(f*z)` has logit cotangent `z`.
Native `N_z=-M_prev` (plus nuisance residualization only when requested),
`S=tensor_s_growth`; native SVD of `S^(-1/2)N_z` implements the thresholded
pencil `(N_z N_z^T, S)`. For CE parity use `z=p-y`, the sum-CE cotangent; the
mean-CE `g` differs by batch size and therefore scales raw A by its square-root.
The parity test uses that known scaling and a direct sum-CE test. No upstream
solver or sign convention is replaced. Default existing-weight projection is
off. Native statistics sample-average all batches; the solve runs on a deep
copy so scheduler/statistics/grad state of the caller is untouched.
