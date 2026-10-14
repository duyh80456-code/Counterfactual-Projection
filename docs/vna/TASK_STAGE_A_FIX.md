# Stage A fix (review round 1)

Same rules as TASK_STAGE_A.md (no commit, `.venv/bin/python`, all tests pass).

## 1. Damping convention (spec: μ = μ_rel · tr(K)/n)
Stage A currently uses the Stage 2 convention μ = μ_rel·tr(F)/p (p = number of O parameters). The
Stage A/B spec and SPEC.md define μ = μ_rel·tr(K)/n with n = number of output rows (batch × C), as in
Phase 0. For the joint model p/n ≈ 1700, so the two conventions differ by three orders of magnitude.
- Add `GGNConfig.damping_norm ∈ {"params", "outputs"}`; default `"params"` so the training engine and
  every existing test are bit-identical. `"outputs"`: μ = μ_rel · tr(F)/n_out (tr(F) = tr(H̄K)).
- Stage A (and the gate defaults) use `"outputs"`. Every row records `mu`, `mu_rel`, `damping_norm`,
  `trace_F`, `n_out`, `p`, and `mu_rel_equiv_params = mu / (trace_F/p)`, so either convention can be read.
- Same change in `joint_correction` and every Stage A call site. Test: with `"outputs"`, the joint
  single-site result still matches the single-site GGN correction; `"params"` reproduces current numbers.
- State the convention in STAGE_A_REPORT.md as a decision pending supervisor confirmation.

## 2. A5 on the real checkpoint
Add subcommand `a5`: for epochs 90, 100, 110, all 8 sites, r=16, reproduce the **Phase 0 code path
exactly** (`signal_curve` objects: VNAConfig, `tiny_candidate`, `random_A_like` with the same generator
purposes, same batches) and dump raw `V`, `random_V`, their ratio, both CG records (iterations,
converged, relative residual, breakdown flag), gain-guard flags, finite flags, ‖g‖, tr(K), μ, and the
checkpoint's epoch/LR/scheduler state → `a5.csv`. Then rerun the same rows with the RHS-normalised CG
and add those columns. The notebook runs `a5` first on GPU0. The report's A5 section states which
column explains the zero once the CSV exists (leave the decision logic in `summary`: it marks A5
`resolved` only if the old-path ratio is 0 and the fixed-path ratio is nonzero at epoch 100).

## 3. Cost
`extension_trace` streams n = batch·C coordinate VJPs (25,600 at stat batch 256), and random_trace
calls it twice per (site, rank, epoch, draw). Measure it in the smoke on the real model size (CPU OK,
extrapolate) and, if exact trace makes A3 exceed 30 min on a T4, replace it for CIFAR-scale inputs with
a Hutchinson estimate of tr(K_E) using ≥ 64 Rademacher probes **shared between reference and random A**
(common random numbers), recording the estimator and its std. Keep the exact version for tests/toy.
After the 2-epoch-equivalent smoke, the notebook prints projected GPU-hours per subcommand and stops if
the total exceeds `GPU_HOURS_BUDGET = 3` (constant at top of the notebook) unless `ALLOW_OVER_BUDGET`.

Report: append "Review round 1" to STAGE_A_REPORT.md. Regenerate the notebook. Run all tests; report
counts. List changed files.
