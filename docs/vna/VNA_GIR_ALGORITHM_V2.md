# VNA–GIR algorithm v2 — opt-in, H1–H4 unchanged

## Scope and controls

Three *opt-in algorithmic* paths are implemented on top of the correctness-reviewed code. `original` remains the default for every registered protocol and existing result. Do not compare these paths against historical runs without matched checkpoints, budget, data, seeds and relevant hardware.

### 1. H1/H4: `candidate_strategy='pareto_minimax'`

For two gradients `g_new,g_old`, a fixed PSD virtual kernel `K_E`, and a shared SPD metric `A`, let `z_i=H J A^{-1} g_i`, `v_i=J_E^T z_i`. Compute the closest-to-origin convex combination `p=t v_new+(1-t) v_old` with

`t = clip((||v_old||^2 - v_new^T v_old) / ||v_new-v_old||^2, 0, 1)`.

For `||p||>0`, build the new physical candidate with RHS `J^T H K_E (t z_new+(1-t)z_old)/||p||`; `A.solve` and the **original KKT projection** continue unchanged. The norm-one virtual feature direction maximizes the worst first-order gradient gain in the virtual feature Euclidean unit ball (MGDA geometry); no claim is made about exact model losses. An approximate Gram matrix is checked for symmetry/PSD. If the virtual convex hull contains zero (or the margin is numerically too small), fall back to the original total-gradient candidate **with its original KKT projection**, avoiding an automatic zero correction. In `unified` mode, native/gn/gradient kernels, and cases without selected virtual sites, retain the original candidate construction; the diagnostic records which path actually ran.

- H1 config: `configs/standard_cl/h1_cifar100_bn_pareto_v2.json`
- H4 config: `configs/standard_cl/h4_rotated_mnist_pareto_v2.json`
- Algorithm verification: `tests/test_vna_algorithm_v2.py`
- No new objective variables, dataset, features or sensor inputs.

### 2. H2: optional exact measured replay-feature drift gate

`GeometryCorrection.step(..., feature_drift_budget=value)` evaluates the **actual** eval-mode replay feature change for each physical trial and rejects candidates exceeding the budget. The existing quadratic feature penalty is unchanged. The gate is optional; default `None` preserves historical H2. A candidate may be rejected even when train/eval loss gates pass. If no replay reference is available the drift gate is marked not applicable; it does not synthesize an old-task reference.

- H2 separate config: `configs/standard_cl/h2_online_cifar100_drift_guard_v2.json`
- The number `0.1` in that example is an **untuned absolute feature norm**, not an empirically validated choice. The scale depends on replay window size and feature dimensions; calibrate from validation data before full benchmark.

### 3. H3: `gir_selection='certified_plasticity'`

The original fixed-capacity candidate generation and exact physical train/eval gates stay intact. A candidate is now admissible only if:

1. Its **upper** bound on exact damped-GN surrogate is below the identity candidate's **lower** bound, using the PCG solution error interval already calculated.
2. Its **lower** bound on predicted GN recovery is higher than the identity's **upper** bound.

This is a proof-aware **local surrogate / plasticity proxy** guard, not a proof of future-task accuracy. No new solves are added, but conservative gating can select identity more often.

- H3 config: `configs/standard_cl/h3_imagenet_small_certified_plasticity_v2.json`

## Safety invariants

- No default config or original benchmark is mutated.
- Original componentwise train/eval acceptance remains active; any new guard adds checks rather than relaxing them.
- Existing PCG residual thresholds and retries remain unchanged. **Goal-oriented early stopping is not included**: its mathematical feasibility proof alone does not certify the same inverse-solve approximation and it would alter the registered numerical protocol.
- Early-feasible projection, zero RHS and large/tiny gradient normalization corrections from the earlier correctness patch are preserved.
- Virtual objectives may conflict. A degenerate virtual margin is not a proof that all useful physical updates are impossible; hence the fallback to original projection.
- Integration uses the complete repository and its pinned PyCIL/LPR/LoP sources. CPU native-runner tests are run with the real Avalanche dependency; GPU accuracy remains unverified.

## Reproduce focused verification

From the source directory:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q -o addopts='' \
    tests/test_vna_algorithm_v2.py \
    tests/test_vna_correctness_integration.py \
    tests/test_standard_cl_math.py \
    tests/test_standard_safety_revision.py
```

**No GPU or original-data accuracy benchmark has been performed.** The new methods are candidates for ablation, not established accuracy improvements.

H2 calibration trials use the same feature_drift_budget as the applied correction.
The eight existing main/ablation notebooks are repinned to this source but retain
their original configs. The four *_v2.json files require explicit selection;
repinning alone does not activate algorithm V2.

## Review update: opt-in nominal physical-size pairing

CorrectionConfig adds physical_update_norm=None and physical_budget_policy=cap.
An explicit step(update_norm=...) overrides the configured norm. Default cap
keeps min(LR, budget/||d||); matched uses (budget/||d||)*(1-16*eps_dtype)
for trial0. For k>=1 use min(s0/2**k, lr_limited_step/2**(k-1)), where
lr_limited_step=min(LR,budget/||d||). This reaches at most the initial cap step
by trial1 when needed while preserving near-LR dyadic trials. Matched requires
a budget; max_trials=1 tests trial0 only and has no LR fallback.
Trials log nominal_update_norm, measured physical_norm and physical_budget_fill_ratio.
The measured norm gate remains strict: roundoff can make trial0 exceed the cap
and require backtracking. Neither accepted norm nor compute is thereby matched.

Separate h1/h4_physical_matched_v1/v2.json configs differ only in candidate
strategy. Their absolute norm0.01 is an unvalidated example. Existing notebooks
retain their original configs; choose these paired configs explicitly.

H3 candidate logs include certified_plasticity_diagnostics (surrogate and recovery
separation, eligibility and uncertainty) and aggregate plasticity_certificate_counts.
Eligibility still requires strict interval separation in both original V2 gates.
Uncertainty denotes overlapping/touching comparison intervals or nonfinite bounds;
these are conditional local surrogate statements, not future-plasticity guarantees.

Matched rows log lr_limited_reference_step. The trial limit is unchanged;
matching nominal trial0 still does not match accepted size or total compute.
