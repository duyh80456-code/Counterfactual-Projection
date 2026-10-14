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

## Stage 2a

Implemented the solver-agnostic training runner, checkpoint validation mode, and
seed-0/seed-1 Kaggle training notebooks. No commit was made. The protected Phase-0
signal diagnostic, resume identity, notebook, builder, and `methods/vna.py` remain
unchanged. Existing `.DS_Store` changes and the supplied task file are retained.
This environment has no CUDA or local CIFAR data; no real-checkpoint GPU audit,
200-epoch CIFAR training result, or measured T4 runtime is claimed.

### Files and behavior

- `experiments/run_vna_training.py`: recipe SGD from random initialization;
  arms `sgd`, `sgd_cm`, `vna`, `vna_random`, and `ngd`. Recipe splits, optimizer,
  throughput flags and MultiStep milestones match the recipe runner. `sgd_cm`
  defaults to 260 epochs and milestones 130/195; its default stop cap permits
  all 260 epochs. `--max-epochs` permits resumable partial execution without
  changing run identity. Final official CIFAR test evaluation is saved in
  `latest.pt` and `final.json`; finished resumes never evaluate it again.
- `experiments/vna_training_support.py`: periodic always-active corrections,
  Dörfler site marking on V (minimal descending prefix, capped by `max_sites`),
  explicit comma-separated sites, rank/selection configuration, cached damping,
  K-FAC EMA factors and exact-CG warm starts. Refresh occurs at epochs 1, 6,
  11, ... by default. Statistics use several independently augmented training
  batches; correction batches have their own stream. Every solver uses eval BN.
- `experiments/vna_training_kaggle.py`: compatible full-state restore from
  directories, ZIP/TAR archives and Kaggle-expanded torch archives; hourly
  immutable atomic archives, a conservative 5 GB peak budget, and the solver
  decision rule. Incompatible resumes are excluded; stale unfinished completion
  files are removed. Checkpoint writes also check the shared output budget,
  preserving the previous latest if the budget would be exceeded.
- `diagnostics/vna_solver_validation.py`: retains the unchanged callable toy
  `run()` interface and default CLI mode. `--mode recipe` loads model-only or
  full recipe checkpoints, uses augmented CIFAR training data, selects the top
  three tiny-V sites, and performs the four-relative-mu sweep. It reports whole,
  W1/W2/BN cosines against exact, projected gains, synchronized correction cost
  in batch-64 recipe-SGD units, refresh costs, and every exact-CG residual.
  Nonconverged exact references are saved for inspection and fail validation;
  they cannot silently drive the notebook decision.
- `scripts/build_kaggle_vna_training_notebooks.py` and
  `notebooks/kaggle_vna_training_seed{0,1}.ipynb`: reuse Phase-0 bootstrap cells
  without editing the Phase-0 builder. They retain the immutable repository
  pin, `github_token`, Gromo pin and official CIFAR train/test/meta discovery.
  Cell A defaults on with attached recipe checkpoints and requires epochs
  40/100/160. Across their selected-site/mu rows, median K-FAC whole cosine
  >=0.7 chooses K-FAC; otherwise median exact correction cost <=30 SGD units
  chooses exact; otherwise lite. `SOLVER_OVERRIDE` wins. With no checkpoint
  attached the explicit printed fallback is exact. Cell B queues fresh
  processes over two GPUs in the prescribed order and skips completed arms.
  The final cell writes test/delta/parameter/cost columns and training curves.
- `tests/test_vna_training.py`: bitwise CPU resume including optimizer, model,
  factors, warm starts, controller, all dedicated generators, NumPy state and
  correction records; zero-lambda identity for all solvers; final-test reuse;
  unchanged parameter count; NGD norm matching; other selection/random arms;
  recipe-validation artifacts and convergence; immutable pin/notebook syntax;
  archive restore; conservative disk enforcement; forbidden-path checks.
- `docs/vna/stage2a_validation/{summary.json,solvers.csv}`: CPU integration
  evidence from a saved tiny recipe-compatible checkpoint, **not CIFAR or T4**.
  All eight exact solves converge; 20 rows cover four damping values and five
  methods on its single available site. The test independently repeats this
  checkpoint-mode path. The toy diagnostic still has its original 24 rows.

Corrections are added to gradients after backward and AMP unscale, before the
SGD step; momentum and weight decay are untouched. Zero lambda still performs
refresh and diagnostic work but makes no gradient write. Realized gain is
measured against mean-CE gradient on the same dedicated correction batch after
BN tangent projection. Nonfinite or nonpositive gains skip that site and log a
reason. No activation or acceptance gate is enforced.

For NGD, first compute the full VNA correction at the **same weights, cached A,
site, damping, solver, and batch**. Apply the same BN tangent projection to
both directions, then multiply NGD by `||d_VNA_projected||/||d_NGD_projected||`.
Only the reference norm determines this multiplier; the reference direction is
not added. Both norms and both projected gains are logged. Exact and K-FAC use
`M F M h`; lite uses `M=I/(tr(F)/n_f)`, hence `F h/(tr(F)/n_f)^2`. This scalar
choice reproduces the existing lite VNA formula. Selection consistently uses
`z=Hbar J M h`; tiny-V/greedy retain native TINY and true-ReLU scoring.
Random A matches the reference rank and exact extension trace on the refresh
measurement batch, then uses that cached A until the next refresh.

The saved latest includes model, SGD momentum, scheduler, scaler, splits,
protocol, global Python/torch/CUDA/NumPy state, training-loader generator,
private statistics/correction/selection/Hutchinson/random-A/construction streams,
A/ranks/damping/trace, K-FAC factors, CG warm starts and the always-active
controller. Hutchinson's private generator is seeded from its checkpointed
stream each refresh; K-FAC currently uses deterministic true-GGN class sweeps,
so no MC-label stream is consumed. Refresh invalidates CG starts. They are warm
between corrections; nonconvergence logs the residual and uses the iterate.

G1 is logged using the native TINY U-spectrum: lambda_1 against mean+2 population
standard deviations of four disjoint sub-batch lambda_1 values. G2 uses
`rho=lambda_vna*V/||h||^2 > .3`. G3 logs loss decrease per second against half
its first available post-refresh rate. G3 is wall-time diagnostic metadata,
outside numerical controller state. None changes the training schedule or
selects a checkpoint. JSONL contains epoch metrics, LR, component timings,
applied/skipped counts, mean rho/V/projected gain, sites/ranks, every weight
norm and GPU peak allocated memory; correction JSONL adds per-site gates,
CG residuals, skip reasons and the matched norms. GPU-hours include measured
epoch compute/evaluation and final-test seconds; filesystem save/archive time
is excluded. Component refresh total includes selection/factors/trace/gate
statistics; these subcomponents are not double counted in total cost.

### Run instructions

```bash
# Normal recipe; same flags across paired arms.
.venv/bin/python -m experiments.run_vna_training --arm sgd --device cuda:0 \
  --data-root /path/to/cifar --output outputs/vna_seed0/sgd --seed 0
.venv/bin/python -m experiments.run_vna_training --arm vna --solver exact \
  --selection tiny_V --device cuda:0 --data-root /path/to/cifar \
  --output outputs/vna_seed0/vna_tiny_V --seed 0
# Resume with exactly the same numerical/configuration flags plus:
# --resume outputs/vna_seed0/vna_tiny_V/latest.pt

.venv/bin/python -m diagnostics.vna_solver_validation --mode recipe \
  --checkpoint /path/to/epoch_040.pt --data-root /path/to/cifar \
  --device cuda:0 --output outputs/validation/epoch_040
# Repeat at 100/160. Raise --cg-iters if the saved exact residuals do not converge.

# Supervisor must first commit the reviewed changes; this task does not commit.
.venv/bin/python -m scripts.build_kaggle_vna_training_notebooks --commit FULL_40_HEX_HASH
```

The delivered notebooks pin the current read-only HEAD
`2aeb36daff90604a9f550abbf9c64f9cc3957b00`, which precedes these uncommitted
Stage-2a files. Their setup deliberately requires the Stage-2a runner; regenerate
with the supervisor's eventual Stage-2a commit before a remote Kaggle run.

### T4 runtime and disk planning (unmeasured)

There are 44,872 training examples, 702 batch-64 steps per epoch, approximately
140,400 SGD steps and 14,040 correction events in 200 epochs. At three sites,
there are up to 42,120 per-site corrections. Reuse the earlier **planning
assumption**, 10–40 seconds per SGD epoch (0.6–2.2 GPU-hours total), rather than
treating CPU toy timings as T4 measurements. For a solver's measured per-site
cost `c` in recipe-SGD units and selected count `s`, estimate
`T_arm = T_sgd*(1+s*c/10) + R_solver`, where `R_solver` is all 40 refreshes,
including site proposals, Hutchinson, gate statistics and factors. Actual site
counts and refresh costs are logged. Training uses batch 16 for corrections;
Cell A's default batch 8 gives an initial screening cost, so revise c from
training logs rather than treating it as an exact extrapolation.

The following are illustrative capacity ranges for **each arm**, conditional
on three selected sites. The assumed correction costs are exact 10–30,
K-FAC 3–10 and lite 2–8 SGD units; they are not GPU observations.

| Arm | exact GPU-h | kfac GPU-h | lite GPU-h |
|---|---:|---:|---:|
| sgd | 0.6–2.2 | 0.6–2.2 | 0.6–2.2 |
| sgd_cm | 0.8–2.9 | 0.8–2.9 | 0.8–2.9 |
| vna tiny_V | 2.4–22 + R_exact | 1.1–8.8 + R_kfac | 1.0–7.5 + R_lite |
| vna tiny_U | 2.4–22 + R_exact,U | 1.1–8.8 + R_kfac,U | 1.0–7.5 + R_lite,U |
| vna_random | 2.4–22 + R_exact,random | 1.1–8.8 + R_kfac,random | 1.0–7.5 + R_lite,random |
| ngd | 3.3–32 + R_exact | 1.4–12 + R_kfac | 1.1–10 + R_lite |

NGD costs more because the reference correction is also computed; this example
assumes approximately 1.5 times correction cost with the first inverse reused.
Random refresh includes exact extension-trace matching. K-FAC refresh uses
`2*b*C` backward sweeps per proposed site (b=16, C=100 → 3,200), plus expensive
factor eigensolves. If total refresh is hypothetically 1–10 minutes per
all-site refresh, add 0.7–6.7 GPU-hours per active arm; actual K-FAC may exceed
that. These ranges support session/resume planning, not performance claims.
Two GPUs reduce queue elapsed time, with the longest individual arm still a
lower bound. No <=30% overhead or accuracy improvement is asserted.

The full-width model-only checkpoint is about 45 MB and model+momentum about
90 MB. Six plain latest checkpoints start near 0.54 GB; no intervention or
periodic model checkpoints are retained. Exact adds two warm float64 vectors
per site (NGD adds a third), A and small metadata. K-FAC can dominate: a
512-input 3x3 conv has a 4608x4608 input factor and eigenbasis, about 340 MB in
float64 together, before its other factors. Several late sites across four
active arms can exceed the 5 GB output/archive budget. The runner and archiver
**fail before the budgeted write**, keeping the last complete state, rather
than dropping factors or silently changing sites/precision. Archive accounting
bounds raw+old+new archives at three times raw bytes; hard-linked snapshots
avoid duplicate checkpoint data. A completed restored arm keeps its full latest
and final metrics, so it is skipped safely.

### Open questions recorded without asking

1. What are real epochs-40/100/160 solver cosines, exact residuals, factor refresh
   costs and training-batch correction costs on T4? Cell A is ready, but this
   workspace has neither real checkpoints/CIFAR nor a GPU.
2. If K-FAC validates but its full factors exceed the 5 GB bound, should a later
   task add lossless factor packing, reduce the configured site cap, or change
   factor precision with a new numerical validation? This stage preserves the
   requested full state and halts at the cap; it does not silently alter the
   selected sites, solver, or precision.
3. Is the logged native-U spectrum/noise G1 statistic also the desired diagnostic
   for tiny-V/greedy, or should a future report compare V-specific bootstrapped
   significance? G1 does not gate Stage 2a. G3 uses a stated half-rate diagnostic
   because its coefficient was not numerically fixed in the task.
4. Confirm the documented lite scalar inverse convention for NGD/selection and
   retain the existing Stage-2 mean-CE normalization. No formula is tuned toward
   solver cosine thresholds.
5. Exact validation fails explicitly if 1,000 iterations at tolerance 1e-5 do
   not converge. Real checkpoint conditioning may require a larger cap or a
   separately validated higher-precision product path.
6. GPU nondeterminism from cudnn benchmarking is an explicit shared throughput
   setting. CPU trajectories are bitwise tested; GPU bitwise identity/resume
   is not claimed. The official test is cached on normal completed resume;
   a hard kill between finishing its evaluation and atomic latest publication
   can repeat that evaluation on recovery.

### Stage 2a verification results

- Focused delivery: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m
  pytest -o addopts='' -q tests/test_vna*.py` → **50 passed, 38 warnings in
  41.87 s**. Log: `/tmp/stage2a-focused-delivery.log`. This comprises the 32
  previous VNA cases and 18 new Stage-2a cases. The separate training-only run
  passed all **18 cases in 17.47 s**, including refresh on both epochs and
  bitwise correction-record/state comparison on resume.
- Final full suite: `PYTHONPATH=/tmp:$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  .venv/bin/python -m pytest -p vna_stage2a_tmp_cleanup -o addopts='' -q
  --basetemp=/tmp/vna-stage2a-full-clean` → **281 passed, 1 failed, 5 skipped,
  38 warnings in 124.96 s**. Log: `/tmp/stage2a-full-clean.log`. The one failure
  is the same pre-existing
  `tests/test_plateau_fork.py::test_plateau_fork_has_recurrent_e_and_single_shot_o_control`,
  which asserts an obsolete `METHODS` source string; its runner/test are untouched.
  The temporary `/tmp/vna_stage2a_tmp_cleanup.py` plugin removes only passed
  function-scoped `tmp_path` directories after fixture finalization. It changes
  no assertions, active fixture data, guards, collection, or skip decisions.
- Standard full-suite attempts were also completed: **259 passed, 2 failed,
  5 skipped, 32 errors in 135.74 s**, then **267 passed, 3 failed, 5 skipped,
  14 errors in 107.22 s**. Logs: `/tmp/stage2a-full.log` and
  `/tmp/stage2a-full-final.log`. Beyond the same source assertion, failures/errors
  are low-disk scratch/checkpoint-write errors in existing DeiT tests. Only
  this task's completed scratch directories were cleaned. The final cleanup
  run eliminates these environmental failures without bypassing disk guards.
- Standalone checkpoint-mode CPU audit succeeds with all exact references
  converged. Multi-batch, two-epoch `vna_random`/lite smoke also succeeds with
  refresh every epoch (`/tmp/stage2a-multi-smoke.log`). Real CIFAR/T4 evidence
  remains unavailable, as stated above.
- Direct `python scripts/build_kaggle_vna_training_notebooks.py --output-dir
  /tmp/stage2a-notebooks` succeeds. Delivered notebooks equal builder output;
  all Python cells parse, and all six Python files compile. `git diff --check`
  passes. Protected Phase-0 diffs are empty; HEAD remains
  `2aeb36daff90604a9f550abbf9c64f9cc3957b00`. No commit was made.

## Stage 2a-fix

Date: 2026-10-14. All twelve requested changes are implemented. No commit or
Git metadata write was made. Protected Phase-0 files have empty diffs; the
pre-existing `.DS_Store` edit and untracked task input are retained.

### Solver audit and configuration

`experiments/vna_training_config.py` is the single source for VNA arguments.
Training and recipe validation share rank (16), correction batch (16), relative
damping (0.01), CG (200 iterations, 1e-5 tolerance), selection (tiny_V), site cap
(3), statistics, projection, and cache settings. Validation no longer forces
rank 4, batch 8, CG 1000, three sites with theta=1, or cached inputs. Notebook
Cell A and training commands receive the same shared argument list; the tiny_U
control explicitly changes selection. SGD cost uses the training batch size.
The fixed small dense-oracle toy remains a separate diagnostic fixture.

Recipe validation includes the requested damping in its sweep, saves every CG
convergence flag, and requires convergence only for exact rows at training mu.
The summary exposes `exact_converged_at_train_mu`; the older all-sweep
`exact_converged` remains descriptive. A regression deliberately marks an exact
nontraining-mu solve unconverged and verifies that validation completes.
At selected sites and training mu, exact directions additionally report
`cos_batch_128` and `cos_disjoint_batches`. The two small batches contain
explicitly disjoint dataset indices; all three directions use the same A and
mu. These measurements appear in JSON and training-mu CSV rows.

`choose_solver` filters to training mu. K-FAC passes fidelity only when its
minimum whole-vector cosine over checkpoints/sites is at least 0.9 and every
projected realized gain is positive. Otherwise affordability uses the maximum
measured per-correction SGD cost times the selected site count, divided by
VNA_EVERY; affordable candidates are preferred exact, then kfac, then lite.
No affordable solver raises with the cost table and requires SOLVER_OVERRIDE.
No checkpoint evidence without an override also raises. Disjoint cosine below
0.3 prints a warning recommending VNA_BATCH=64 and larger VNA_EVERY.

### Training state, strength, and logs

The buffer regression found upstream TINY lazily registering
`second_layer.bordering_weight` during refresh. Stage-2a wrappers now restore
all original buffer registrations and tensor values, and every module mode,
including on exceptions. Twelve cases exercise all three solvers, both input
cache settings, and both fully train-mode and mixed child modes across site/A
refresh, corrections, and factor/trace refresh. Running means, variances, and
num_batches_tracked are included. Phase-0 helpers were not edited.

`--lambda-mode time_avg` is default and applies lambda_vna*vna_every to VNA and
the norm-matched NGD control; `per_step` retains the literal coefficient.
Corrections log the coefficient, raw and projected realized gains for the
chosen direction and VNA reference, and the applied correction norm divided
by the training minibatch block-gradient norm. Epoch JSONL includes its mean
and maximum and mean raw/projected gains, without assuming gain equality.

No training refresh or correction occurs before `--vna-start-epoch` (5).
Sites/A first refresh there and then every `--refresh-epochs` (5). Curvature
factors and mu/trace refresh every `--factor-refresh-epochs` (1) while active,
without changing A/sites; warm starts are invalidated. Their time is included
in final VNA cost. Train loss and correct counts accumulate on device and are
transferred together once per epoch.

Compact correction records append to corrections.jsonl, with CG iteration,
convergence, and relative-residual summaries only. Epoch aggregates remain in
epochs.jsonl. latest.pt stores correction_count and numerical resume state,
not correction records. Resume protocol format is now 2, so older Stage-2a
checkpoints with different strength/schedule/logging semantics are rejected.
Uninterrupted and resumed trajectories and correction JSONL records match.

### Notebook delivery

Seed-0 and seed-1 notebooks are regenerated with default HEAD pin
`3c2e2a0ad430a1d1af47d01680fafd742a58510e`; the supervisor must re-pin after
committing the fixes. CPU bootstrap tests include all three VNA test files.
VNA_BATCH, VNA_EVERY, and LAMBDA_VNA are constants shared by validation/training.

Cell B first runs all six arms in a separate smoke directory, with no resume,
for SMOKE_EPOCHS=2. Smoke activates VNA in epoch 1 so epoch 2 measures the active
arm rather than warmup SGD. The full run retains start epoch 5. It prints
second-epoch seconds and projected GPU-hours (200 epochs, 260 for sgd_cm),
then stops before full restore/launch if the sum exceeds GPU_HOURS_BUDGET=30,
unless ALLOW_OVER_BUDGET=True. Queue order is sgd, vna_tiny_V, ngd, vna_random,
vna_tiny_U, sgd_cm. Executed notebook-cell tests check isolated outputs, all
six smoke arms, epoch-2 projections, budget rejection and explicit override.

### Verification and open questions

- Focused final: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest
  -o addopts='' -q --basetemp=/tmp/vna-stage2afix-focused-final tests/test_vna*.py`
  → **67 passed, 39 warnings in 45.40 s**.
  Log: `/tmp/stage2afix-focused-delivery.log`.
- Full final: `PYTHONPATH=/tmp:$PWD OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  .venv/bin/python -m pytest -p vna_stage2afix_tmp_cleanup -o addopts='' -q
  --basetemp=/tmp/vna-stage2afix-full`
  → **292 passed, 2 failed, 5 skipped, 10 errors, 39 warnings in 129.33 s**.
  Log: `/tmp/stage2afix-full-final.log`. One failure is the previously documented
  obsolete METHODS-string assertion in test_plateau_fork. The other failure
  and ten setup/teardown errors are disk exhaustion in existing DeiT
  checkpoint/export tests (PytorchStreamWriter write failure and Errno 28).
  Every VNA case passes in the full run. The temporary cleanup plugin deletes
  only successful function-scoped test scratch after fixture finalization;
  it does not change assertions, tests, collection, disk guards, or fixtures.
- Notebook files exactly equal builder output; every code cell parses. All
  seven changed/new Python files compile; git diff --check passes. Protected
  Phase-0 diffs are empty. Machine-readable counts are in
  `stage2afix_test_results.json`.

Open questions, recorded without asking:
1. Real CIFAR checkpoint agreement, batch sensitivity, solver affordability,
   factor memory and smoke GPU-hour projections still require T4 evidence.
   This CPU workspace has no attached Phase-0 checkpoints or CUDA GPU.
2. If shared CG=200 does not converge at training mu on real checkpoints, the
   supervisor must increase CG settings for both validation and training, or
   explicitly choose an override after reviewing saved diagnostics.
3. Smoke extrapolation uses epoch 2, with active factor refresh but without
   amortizing later site/A refreshes or changing learning-rate phases. The
   reported hours are projections; full per-epoch timing remains authoritative.
4. The full suite needs sufficient free scratch space for the existing DeiT
   checkpoint exports and a separate resolution of the obsolete plateau source
   assertion. Neither unrelated code path was changed for this VNA task.
5. Correction records live in JSONL, not checkpoints. An attached latest.pt
   alone resumes numerical state/counters but cannot reconstruct historical
   per-correction records; retain the archive's JSONL for historical analysis.


## Stage 2a-fix2

### Branch tracking and provenance

Both notebook builders now accept `--branch` (default `vna-phase0`) and emit
`REPO_BRANCH`, cloning the branch tip through the existing token/askpass path.
The pinned-commit preflight and regeneration instructions are removed; required
module existence checks remain. Each bootstrap prints its resolved HEAD and
atomically appends a UTC timestamp/branch/commit record to
`OUTPUT/code_commit.json`. All three notebooks have been regenerated.

Stage-2a `latest.pt`, epoch history, and final records carry `code_commit` and
`code_history`. A changed-commit resume prints a prominent old/new warning and
the first new epoch (`checkpoint epoch + 1`), then appends
`{"from_epoch": e, "commit": c}` and continues. Same-commit resumes do not add
entries. Even an already completed arm records a changed session boundary
without reevaluating the official test set. Protocol and split validation stay
strict. Final notebook results include each arm's commit set. Legacy checkpoints
without provenance use `unknown`, rather than attributing past work to today's
commit.

Phase-0 recipe checkpoints/history also record provenance and warn on code
transitions. Every newly computed diagnostic completion marker records its
own producing commit. Completed diagnostics retain their original provenance
when reused across revisions. Source-file hashes are no longer experiment
identity: legacy `identity.source` is excluded during compatibility/merge checks,
while all configuration/schema and model-digest checks remain strict.

### Smooth correction and smoke reporting

`--lambda-mode {per_step,time_avg,spread}` now defaults to `spread`. An accepted
direction computed at step t adds lambda times that same stale direction at
steps t through t + VNA_EVERY - 1, including across epoch boundaries. The sum of
these gradient additions equals the former time_avg impulse; this does not
claim identical model trajectories under momentum or changing gradients.
The same path serves VNA, norm-matched NGD, and random-A VNA. Pending directions
and remaining counts are serialized in the VNA resume state. Rejected or
nonfinite replacements clear their site's state and leave no spread residue.
Explicit per_step and time_avg behavior remains available.

Compact `steps.jsonl` records loss, scheduled correction-step status, and actual
per-site applied-delta/gradient ratios, including stale applications. The saved
previous loss makes these step records identical across CPU resumes. After each
VNA-type two-epoch smoke, the notebook prints median/p95 applied ratios, skipped
correction fraction, and maximum train-loss jumps at scheduled correction steps
versus other steps. The ratio distribution uses actual applications (skipped
attempts are reported separately); missing samples produce null statistics.
Thresholds are pre-registered and report-only: p95 > 3 gives IMPULSE_RISK;
median < 0.05 gives LIKELY_NULL. No configuration changes follow these flags.
Loss is measured before each update, so the loss-jump comparison is descriptive
and does not attribute causality to the correction computed at that same step.

### Pre-registered batch choice and total budget

At training damping, real-checkpoint validation measures cos(d16,d128),
cos(d64,d128), and disjoint-batch cosines for 16/64/128 at the selected sites.
The smaller comparison batches are nested subsets of the same augmented
128-example reference, while the disjoint comparison uses a separate sample
pool. All directions at a site use the same selected A and absolute mu.

Cell A prints and applies the minimum cosine across sites/epochs:
cos16 >= 0.5 selects (batch 16, every 10); otherwise cos64 >= 0.5 selects
(64,20); otherwise (128,20). `VNA_BATCH_OVERRIDE` and `VNA_EVERY_OVERRIDE`
(default None) independently win. Without validation measurements the rule's
fallback is (128,20); the existing explicit solver-override requirement remains.
After selecting the batch/every pair, Cell A remeasures validation costs at that
pair and only then chooses the solver and its total correction overhead. The
chosen values are also the full training command/protocol values.

The notebook documents 30 GPU-hours as the TOTAL projected full-training cost
across all arms of one seed. After all smoke measurements, the budget policy
prints the projection table and drops sgd_cm first, then vna_tiny_U if needed.
If retained arms still exceed the budget, it raises with the table unless
ALLOW_OVER_BUDGET=True. The explicit allowance bypasses the final rejection;
the specified drop order still applies. Dropped arms are excluded from restore,
launch, and results. Successful smoke outputs are deleted before full training.

### Verification

- Required suite: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest
  -o addopts='' -q --basetemp=/tmp/vna-stage2afix2-final tests/test_vna*.py`
  → **85 passed, 0 failed, 0 errors, 0 skipped, 39 warnings in 55.55 s**.
  Warnings come from upstream Torch/Gromo deprecations and pseudo-inverse use.
- New coverage exercises branch session append records; Phase-0 producing
  commits and cross-commit reuse; strict config rejection; changed/same-commit
  training resumes and completed-arm transitions; bitwise CPU spread resumes
  with a pending direction for VNA/NGD/random-A; exact summed gradient effect;
  skipped/nonfinite residue clearing; threshold boundaries; worst-site/epoch
  batch choice and independent overrides; executed Cell A command ordering;
  smoke isolation; ordered budget drops, rejection table, and allowance.
- All three checked-in notebooks equal default builder output, all code cells
  parse, and all 12 changed/new Python files compile. `git diff --check` passes.
  Machine-readable counts: `docs/vna/stage2afix2_test_results.json`.
- No commit or push was made; `.git` was only read. The pre-existing .DS_Store
  modification and supplied task file were left untouched. Only this task's
  scratch directories/logs under /tmp were removed after verification.

Open questions, recorded without asking:
1. Real CIFAR checkpoint batch agreement, solver convergence/overhead, and the
   projected total GPU-hours still need T4 measurements. This workspace has no
   CUDA GPU or attached real Phase-0 checkpoints; mocked notebook execution
   validates decisions and commands, not GPU timings.
2. Epoch-2 extrapolation estimates full arm-training costs, excluding validation
   and smoke calibration itself. Should the 30-hour policy later include those
   setup costs as well? The requested post-smoke arm-drop policy uses full-run
   projections; actual per-arm timings remain authoritative.
3. With changed remote code, existing diagnostic markers intentionally retain
   the original producing revision. Scientific interpretation of a mixed-commit
   run must use the per-marker provenance and per-arm code_history.
4. Kaggle executes the remote branch tip. These local changes are uncommitted as
   requested, so the remote branch must contain the implementation when those
   notebooks run; no pinned-hash regeneration is needed afterward.
5. A latest.pt-only restore preserves numerical spread/provenance state but
   cannot reconstruct old correction/step JSONL. Retain output archives for
   complete historical smoke/correction analysis. The unrelated full repository
   suite was not rerun; the required three VNA files all pass, and previous
   full-suite disk-exhaustion issues remain outside this task's scope.
