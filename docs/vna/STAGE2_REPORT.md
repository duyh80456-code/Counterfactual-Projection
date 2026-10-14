# Stage 2-core — GGN solvers, K-FAC, and projection-aware selection

Date: 2026-10-14. Branch `vna-phase0`; no commit, push, clone, or Git metadata
write. The pre-existing `.DS_Store` edit and untracked task/design inputs are
retained. `methods/vna.py`, `diagnostics/signal_curve.py`, its resume identity,
Phase-0 notebook and notebook builder are unchanged. No training runner or
notebook was added. Environment: existing `.venv`, torch 2.2.2, CPU, pinned
Gromo `8d19107b61a9459a9021065a329b699adcb0f25b`.

## Files and interfaces

- `methods/vna_ggn.py`: `GGNProducts`, `GGNConfig`, `vna_exact_ggn`,
  `vna_lite_ggn`, `ngd_control`, `norm_match`, `filtered_cotangent`,
  `project_bn_tangent`, and `bn_weight_norms`.
- `methods/vna_kfac.py`: `KFACBlockPreconditioner` and `vna_kfac`.
- `methods/vna_select.py`: `tiny_V`, `greedy_V`, neuron normalization,
  sign scoring, exact extension trace, and trace-matched random-A control.
- `diagnostics/vna_solver_validation.py`: standalone CPU diagnostic with
  synthetic or supplied tiny-model checkpoint, CSV and JSON outputs.
- `tests/test_vna_stage2.py`: eight tests including parameterization, dense
  oracles, functional finite differences, invariance, selection, and CLI output.
- `third_party/REFERENCES.md`: official library revisions/licenses, fallback
  decision, and pinned TINY pseudo-loss alignment evidence.
- `docs/vna/stage2_validation/`: final solver/selection CSVs, summary JSON,
  and the small round-tripped synthetic checkpoint used for the diagnostic.

Public corrections return `(parameter_direction, diagnostics)` and do not add
to gradients or apply an optimizer step. Directions exclude `lambda_vna`; a
later caller chooses its coefficient. All products use eval BN, native network
precision with autocast disabled, and float64 curvature actions/CG reductions.
The existing `_Products` provides exact residual-path JVP/VJP and virtual B=0
products. No upstream implementation or Phase-0 helper is edited. Nonfinite
returned directions are skipped with explicit diagnostics. The exact reference
logs both CG residuals/convergence rather than silently treating a truncated
solve as exact. Damping/trace can be passed in to reuse refresh estimates.

## Implemented formulas and normalization

Write `b` for the number of samples and `n_f=b*C` for flattened logit dimension.
To resolve the overloaded `n` in DESIGN_V2 versus SPEC, use **mean CE**:

```
g_i = (p_i - onehot(y_i))/b
Hbar_i = (diag(p_i) - p_i p_i^T)/b
F = J^T Hbar J
C_OE = J^T Hbar J_E, C_EO = C_OE^T
h = J^T g
```

There is one sample normalization per curvature operator, inside `Hbar`; none
is added after JVP/VJP or in the factors. `Hbar` is applied by per-sample
probability multiplication in float64, without constructing even a C-by-C
matrix in the matrix-free solver. It is true GGN, not observed-label Fisher.
Mean gradient normalization is independently required by mean CE.

The exact correction is `M C_OE C_EO M h`, `M=(F+mu I)^-1`, with two
parameter-space CG solves, float64 vectors, configurable iteration cap and
true-residual convergence diagnostics. Default cap/tolerance are 200/1e-6;
validation uses 500/1e-5 for fp32 products. Hutchinson uses eight parameter-space
Rademacher probes with a dedicated generator, and
`mu=damping_rel*tr(F)/num_block_parameters` (positive floor 1e-12).

Dense tests independently verify

```
d = J^T (Hbar K+mu I)^-1 Hbar K_E Hbar K (Hbar K+mu I)^-1 g
K=J J^T, K_E=J_E J_E^T.
```

The nonsymmetric function-space expression is an **oracle only**; production
never runs CG on `Hbar K`. NGD uses the same inverse twice:
`d_ctrl=M F M h`. Norm matching is applied after the same BN projection as
VNA and returns its scale. Lite performs no CG:
`d_lite=J^T Hbar K_E Hbar K g/c`,
`c=(tr(Hbar K)/n_f)^2=(tr(F)/n_f)^2`, as specified by the logit-dimension
trace normalization. This scalar normalization is separate from mean curvature.

## K-FAC and EK-FAC

The task-permitted minimal implementation is used, not a new external dependency.
Official curvlinops and ASDL revisions and MIT licenses are recorded in
`third_party/REFERENCES.md`. Curvlinops's conv/linear-only curvature blocks and
fixed-model cache do not directly provide the full Gromo BN-diagonal/EMA/state
lifecycle here; ASDL advertises torch-1.13 compatibility, with no verified
Gromo/torch-2.2 integration. This is a conservative fallback decision, **not** a
claim that their conv hooks cannot work with an additional adapter. No library
was installed into the live Phase-0 environment. Sources:
[official curvlinops EKFAC documentation](https://curvlinops.readthedocs.io/en/stable/_modules/curvlinops/ekfac.html),
[official ASDL repository](https://github.com/kazukiosawa/asdl).

For each ungrouped residual conv, A is unfolded-patch covariance averaged over
samples and positions (append a homogeneous coordinate if biased). G sums
spatial output-gradient outer products, averaged over samples. This is the
spatial-expand K-FAC convention; it deliberately approximates spatial cross
terms. Factors come from exact per-sample softmax Hessian square-root columns,
not labels. Eval-mode BN affine parameters use their exact diagonal mean GGN.

At explicit `refresh`, factors are EMA updated (default old weight 0.5), then
eigendecomposed. Default EKFAC replaces products of factor eigenvalues by the
exact diagonal mean GGN in the refreshed Kronecker eigenbasis. A second streamed
backward sweep computes that diagonal without retaining all sample/class
parameter gradients. With C classes and b samples, refresh uses one forward,
b*C backward actions for G/BN, and b*C additional backward actions for the
corrected spectrum. Factor memory is bounded by matrices, activations and one
parameter-gradient set; exact class sweeps remain expensive. Previous corrected
diagonals are rotated into the refreshed basis using squared overlaps before
EMA. Setting `eigenvalue_correction=False` retains the plain Kronecker spectrum.

`apply` only computes the cached symmetric inverse in that eigenbasis:
`Q_G [(Q_G^T D Q_A)/(lambda+mu)] Q_A^T`, with **one** Tikhonov damping, not
separately damped factors. BN uses `d/(diag_F+mu)`. `state_dict/load_state_dict`
round-trip factors, eigenvectors, corrected eigenvalues, EMA and refresh count.
`vna_kfac` uses this M on both sides of **exact** C_EO/C_OE. Tests prove the
uncorrected inverse equals dense `(G ⊗ A+mu I)^-1`, the corrected spectrum equals
an independently constructed dense GGN diagonal, and the inverse is symmetric
positive definite. Random-batch raw correction gains are nonnegative.

## BN tangent projection and selection cotangent

Every residual conv with a BN in its post-layer function is projected by
`d_W <- d_W - (<d_W,W>/||W||^2) W`, on by default, configurable with
`GGNConfig(project_bn=False)`. BN affine parameters and biases are untouched.
`bn_weight_norms` supplies layer norms for later logging. This stage has no epoch
loop. Tests check tangency and norm-matched NGD. **PSD descent is established
before projection**; applying a Euclidean tangent projector only on the output
of the preconditioner does not preserve that general proof. Both raw and
projected realized gains are reported; all measured projected gains are positive.

The precise selection cotangent is

```
z = Hbar J M J^T g
V(A) = ||J_E(A)^T z||^2 = h^T M C_OE C_EO M h
```

For exact M, `z=(Hbar K+mu I)^-1 Hbar K g`. For K-FAC, the same
`Hbar J M_kfac J^T g` definition applies (pass the preconditioner to
`filtered_cotangent`). This is the residual filtered in the GGN metric and
includes Hbar: simply filtering `g` in Euclidean logit space would optimize a
different objective. V describes the unprojected PSD correction; projected
realized gain is measured separately.

`tiny_V` runs pinned Gromo statistics with `sum(f*z)`, whose logit derivative is
exactly z. Native `N_z=-M_prev` sets the TINY sign convention; existing-weight
residualization is off by default, with an explicit ablation flag. Streaming
statistics sample-average N_z and S across batches **before** the one native
solve. Native SVD of `S^(-1/2)N_z` solves the generalized eigenproblem
`(N_z N_z^T,S)` with eigenvalue threshold 1e-6; returned squared singular values
are checked against an independent whitened eigensolve. This is TINY's
linearized-ReLU proposal; outgoing B and existing-weight delta are discarded.
The entire solve runs on a deep copy, preserving caller scheduler/statistics,
params, buffers, gradients, modes and global RNG.

Parity with tiny_U is tested for `z=p-y` (upstream sum CE). For the requested
`z=g=(p-y)/b`, raw A has the same directions but scale `1/sqrt(b)` because
native A includes sqrt(singular value); the test verifies this scaling. After
required neuron normalization, the directions match up to sign/order/roundoff.
No upstream sign or loss convention was silently replaced. Each proposed neuron
is normalized to the mean existing-neuron norm, both signs are scored through
the true native ReLU J_E, and the better sign is retained for tiny_V.

`greedy_V` builds normalized random and tiny_V directions plus both signs,
removes identical input maps, constructs **true-ReLU** Jacobian features, and
uses group OMP (all outgoing coordinates for a neuron form a group). Its Gram
is the Euclidean logit-feature Gram; scores use the GGN-filtered z. Greedy gain
is `c_S^T(G_SS+nu I)^-1 c_S`, with nu=1e-8, stopping at theta=0.6 of whole-pool
gain or r_max. Tests verify unique picks, monotone gain and threshold attainment.
The diagnostic uses theta=1 with r_max=2 to compare arms at equal rank; it is
not an assertion that rank 2 captures the whole pool's gain. Sign alternatives
remain in the greedy pool because conditioning can change their marginal gain.

Random-A matches rank and **exact tr(K_E)** on the measurement batch, with a
single positive scaling (also applied to bias, if any). ReLU homogeneity makes
trace scale quadratically. Final trace matching takes precedence over exact
per-neuron norm equality; those constraints generally cannot both hold for a
random draw. This tradeoff is recorded rather than hiding the rescaling.

## CPU validation

Reproduce without training or CIFAR data:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m diagnostics.vna_solver_validation --output docs/vna/stage2_validation
```

The diagnostic round-trips a checkpoint of native one-block Gromo ResNet, width
3, three classes, 8x8 inputs; correction batch 3, independent statistics batch
12 streamed as two batches of 6. A is norm-normalized rank 2. A supplied tiny
checkpoint can replace synthetic initialization with `--checkpoint PATH`.
The four mu values use the same eight-probe trace estimate. Exact/KFAC/lite/NGD
reuse that trace so per-correction timings exclude trace refresh; Euclidean VNA
uses its own metric-relative damping and cold trace, as the old comparison arm.
SGD units use the median of three restored batch-64 steps after setup. Counts
include CG verification and diagnostic products; forward calls and pullback
applications are reported separately because cached VJP applications need not
perform a new forward. CPU single-run timings are not GPU predictions.

Synthetic checkpoint seed 20261014; Hutchinson tr(F) = 0.261198764. All eight exact-reference CG solves converged. K-FAC agreement is weak (whole-vector cosine 0.008–0.231); lite is also weak (0.076–0.266). The older Euclidean direction has cosine 0.317–0.944. These are descriptive CPU results, not training acceptance.

| mu / (tr(F)/p) | Method | Whole cosine | W1 | W2 | BN | Projected gain | SGD units | Product passes |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1e-04 | exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0005044 | 28.50 | 49 |
| 1e-04 | kfac | 0.2313 | 0.1590 | 0.3264 | 0.0322 | 0.0651 | 5.55 | 7 |
| 1e-04 | lite | 0.0759 | 0.0044 | -0.0113 | 0.2668 | 0.002984 | 3.98 | 6 |
| 1e-04 | ngd | 0.8799 | 0.9515 | 0.8969 | 0.6365 | 0.0007134 | 26.24 | 47 |
| 1e-04 | euclidean | 0.9429 | 0.9801 | 0.9764 | 0.7761 | 0.000495 | 55.20 | 102 |
| 1e-04 | cross_dropped | 0.2127 | 0.0000 | 0.3029 | 0.0379 | 0.02379 | 4.18 | 5 |
| 1e-03 | exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0004962 | 26.62 | 49 |
| 1e-03 | kfac | 0.2022 | 0.1029 | 0.3275 | 0.0207 | 0.03621 | 4.68 | 7 |
| 1e-03 | lite | 0.0770 | 0.0048 | -0.0106 | 0.2692 | 0.002984 | 3.85 | 6 |
| 1e-03 | ngd | 0.8770 | 0.9504 | 0.8945 | 0.6257 | 0.0007011 | 26.12 | 47 |
| 1e-03 | euclidean | 0.9440 | 0.9795 | 0.9758 | 0.7737 | 0.0003554 | 62.80 | 118 |
| 1e-03 | cross_dropped | 0.2154 | 0.0000 | 0.3098 | 0.0280 | 0.01581 | 4.01 | 5 |
| 1e-02 | exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0004275 | 27.35 | 49 |
| 1e-02 | kfac | 0.0646 | -0.0642 | 0.2119 | -0.0041 | 0.01091 | 5.01 | 7 |
| 1e-02 | lite | 0.0886 | 0.0085 | -0.0033 | 0.2934 | 0.002984 | 4.26 | 6 |
| 1e-02 | ngd | 0.8468 | 0.9388 | 0.8700 | 0.5109 | 0.0005948 | 26.28 | 49 |
| 1e-02 | euclidean | 0.7094 | 0.7547 | 0.7766 | 0.5516 | 0.0001761 | 52.13 | 96 |
| 1e-02 | cross_dropped | 0.1289 | 0.0000 | 0.1987 | -0.0024 | 0.007089 | 3.92 | 5 |
| 1e-01 | exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0002141 | 25.63 | 47 |
| 1e-01 | kfac | 0.0084 | -0.1277 | 0.0722 | 0.1179 | 0.006718 | 5.66 | 7 |
| 1e-01 | lite | 0.2662 | 0.0595 | 0.1566 | 0.5608 | 0.002984 | 4.52 | 6 |
| 1e-01 | ngd | 0.5219 | 0.8030 | 0.6619 | -0.5055 | 0.0001852 | 28.14 | 45 |
| 1e-01 | euclidean | 0.3169 | 0.0298 | 0.3297 | 0.5159 | 0.00017 | 43.71 | 84 |
| 1e-01 | cross_dropped | 0.0741 | 0.0000 | 0.0730 | 0.1194 | 0.005002 | 3.79 | 5 |

Baseline SGD: **0.004228 s**, batch 64, plain SGD lr=.01, no momentum/weight decay. K-FAC/EKFAC refresh: **0.012543 s / 2.97 SGD units**, one forward and 18 backward actions (b=3,C=3, two sweeps). Correction-only times reuse factors and trace. Exact/NGD trace refresh would add eight F actions (8 JVP + 8 VJP); exact includes two CG solves. Costs measure complete correction setup/restoration and diagnostics, not a bare matrix multiply.

Equal-rank, true-ReLU V (all effective ranks 2):

| Relative mu | tiny_U | tiny_V | greedy_V | Trace-matched random |
|---:|---:|---:|---:|---:|
| 1e-04 | 0.000479223 | 0.00264895 | 0.00264895 | 0.000907946 |
| 1e-03 | 0.000471653 | 0.00270207 | 0.00270207 | 0.000923535 |
| 1e-02 | 0.000408158 | 0.00320807 | 0.00320807 | 0.00107192 |
| 1e-01 | 0.000210725 | 0.00685276 | 0.00685276 | 0.002509 |

In this pool, greedy selects the same two feature directions as tiny_V. Tiny_V/random V is approximately 2.73–2.99 and tiny_V/tiny_U approximately 5.53–32.52. The small batch and synthetic checkpoint cannot establish selection quality on trained CIFAR models. Full raw gains, seconds, group norms, product counters and selection values are in [summary.json](stage2_validation/summary.json), [solvers.csv](stage2_validation/solvers.csv), and [selection.csv](stage2_validation/selection.csv).

The cross-dropped ablation zeros only W1 rows of C on **both** sides under the
same K-FAC inverse; it does not edit F or change the inverse. W1's direction is
identically zero and its reported cosine is 0 by convention (the CSV also
records norm 0). The W1 cosine therefore collapses at every mu, whereas full
K-FAC retains a nonzero W1 direction. Negative full W1 cosines at some damping
values are measured approximation errors, not a missing coupling. Neither
approximation is accepted as a good default on the basis of this toy.

## Test results and final checks

- Final focused: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -o addopts='' -q tests/test_vna*.py`
  → **32 passed, 36 warnings in 28.29 s** (24 existing Stage-1 cases and eight
  Stage-2 cases). Log: `/tmp/stage2-focused-delivery.log`.
- Final full suite: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -o addopts='' -q --basetemp=/tmp/vna-stage2-full-final`
  → **263 passed, 1 failed, 5 skipped, 36 warnings in 133.61 s**.
  Log: `/tmp/stage2-full-final.log`. The failure is the previously documented
  `tests/test_plateau_fork.py::test_plateau_fork_has_recurrent_e_and_single_shot_o_control`
  at line 252, asserting an obsolete `METHODS` source string. Its runner/test
  remain untouched; no Stage-2 failure or disk-reserve failure occurred.
- Earlier full run: **262 passed, 1 same failure, 5 skipped in 148.10 s**, before
  adding the CLI-artifact test. Log: `/tmp/stage2-full.log`.
- Final isolated validation CLI finished successfully, writing 24 solver rows
  and 16 selection rows. All reference solves converge, every cross-dropped
  W1 norm is zero, all compared selection ranks equal 2. Log:
  `/tmp/stage2-validation-delivery.log`. The CLI integration test independently
  checks these invariants and output files. Final standalone timings above were
  measured after test processes finished.
- `py_compile` on all five Python files and `git diff --check` pass. Protected
  Phase-0 source/notebook/builder diffs are empty. HEAD remains
  `56dfe37c7c837b6f98ebe513991615deca1eb351`. No commit was made.

Dense double-precision tests cover exact/lite/NGD formulas, nonsymmetric
function-space equivalence, converged CG, descent and the exact filtered z.
Independent finite differences cover H-weighted ordinary and virtual tangents;
adjoint tests cover both cross products. Lite is tested with the solver patched
to fail if called. K-FAC tests cover dense Kronecker inversion, independent
EKFAC diagonal, EMA refresh, symmetry/PSD, repeated random-batch descent, and
state round trip. Read-only tests include nonempty pre-existing gradients,
train/eval modes, parameters, buffers and global torch RNG. TINY tests prove
native sum-CE parity and the explicit mean-CE square-root scaling, generalized
eigenvalue parity and multi-batch averaging. Greedy tests verify no duplicate
picks, monotone objective, the default 0.6 threshold, neuron norms, and exact
random-control trace matching. No acceptance thresholds were tuned.


## Open questions recorded without asking

1. DESIGN_V2 overloads n between sample count and logit dimension; this report
   fixes mean-CE Hbar and uses n_f for lite's trace scalar. Confirm that intended
   scalar convention before Stage 2a, without changing the pinned Phase-0 path.
2. Low K-FAC/lite cosines on this synthetic checkpoint need broader trained
   checkpoint validation. Exact-GGN refresh is costly; should a later MC
   true-Fisher refresh replace it after a variance/cost study? No damping,
   thresholds or factor convention were tuned to improve agreement here.
3. BN output-only tangent projection can destroy the PSD descent guarantee.
   Should Stage 2a sandwich the entire preconditioner by the tangent projector,
   or retain output projection with a realized-gain check? This stage follows
   the requested output projection and exposes both gains.
4. Random control exact trace matching and equal per-neuron norm are generally
   incompatible. The trace-specific control uses final trace rescaling; decide
   whether a separate norm-only arm is also needed later.
5. Greedy's logit Gram is Euclidean, with metric filtering carried by z. Would
   an H-weighted Gram be a useful separately named ablation? It is not silently
   substituted for the requested OMP objective here.
6. Future library integration, GPU costs, full-scope large-model memory,
   train-BN versus eval-BN discrepancy and statistical generalization remain
   unmeasured. No Stage 2a training or Phase-0 acceptance claim is made.
