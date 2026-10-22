# Same-commit Gram M1s0 and CG-IR T4 profile

Notebook: `notebooks/vna4_gram_m1s0_ir.ipynb`.
Inputs: CIFAR-100; the old full/noBN/superclass/s0 task-1 checkpoint or
`vna4_main.tar.gz`; previous output archives to resume. The old archive also
contains the completed baseline and its full-precision epoch history.
GitHub Secret `github_token` and T4 x2 are required by the shared bootstrap.
No source ZIP. Both active lanes import the one pinned implementation commit;
8c56d02 is cloned solely for the fixed-system Gram regression oracle; its old LM/GIR decisions are not required to match the new policy.

| Lane | Sequence |
|---|---|
| GPU0 | Gram/power10 K0 from three task-2 points; wait for fixed-system PASS; full seed0 20 tasks; baseline verification/continuation |
| GPU1 | Same-commit Gram/power10 with fixed-system 8c56d02 check; same-commit CG-IR/exact with capacity; idle |

Output archive: `/kaggle/working/vna4_gram_m1s0_ir.tar.gz`.
Output paths include the pinned commit prefix. Per-job logs and atomic native
checkpoints are archived every 15 minutes and after stages. Exit 75 pauses
remaining launches; attach the output and rerun to continue. A CG-IR failure is
reported without cancelling the approved Gram run. CG-IR never trains M1 here.

Reports under `profiles/`:
- `gram_compatibility.json`: fixed-system direction regression against the actual 8c56d02 direction and OutputSolve source, strict rel tol 1e-5/abs tol 0 PASS required, plus completed current Gram profile. This gate covers frozen theta, Gram, metric, proposals and damping, not changed LM/GIR decisions;
- `cg_ir_vs_gram.json`: strict cross-backend comparison, with explicit curvature
  change, full differences and lm_numeric_audit; do not treat a pending check as PASS;
- each `diagnostic.json`: gamma_precision, application time/peak, timers,
  ir_summary, candidate/swap details, actual solves, capacity_profile and streamed
  downstream_capacity. Capacity is a 100-class/repeated-window proxy, not actual task20.

At task end activity remains 256 samples; B is CG-IR on the first 128. The
128-sample normalization is preserved and logged. Gram construction has no
600-second stop. Deadline checks and all loss/window/finite guards remain.

The baseline gate uses full-precision checkpoint epochs (150 entries, 100/50),
not six-decimal console output and not the full arm's losses. It trains a fresh
baseline prefix after the full arm. Exact match reuses the validated historical
20-task result with original commit/hashes; mismatch continues the new run.
On reuse, `summary.json` for verification says REUSED_BASELINE/tasks_run=2;
`reused_baseline.json` points to the real 20-task result. Never read the partial
verification metrics as a final baseline. A missing reference runs a full baseline.

## Previously excluded tests

- `test_adaptive_schedule_iterations_skips_and_budget_exceeded` used removed
  K_base/rho cadence and scalar cost projection. It now checks the registered
  K0/geometric cadence, task-specific cost and budget guard.
- `test_real_sgd_momentum_and_weight_decay_remain_native_after_correction`
  previously failed at an absolute 1e-16 expected-SGD comparison. It checks
  optimizer momentum preservation and the next native step; it is rerun here.
- `test_gir_rechecks_actual_dead_slot_revived_by_upstream_correction`
  previously failed its synthetic revival expectation. It is rerun unchanged;
  do not infer an engine correctness result by simply excluding this assertion.

The two latter tests passed in the isolated rerun; the broader affected run is
recorded after integration. Hardware compatibility PASS, capacity and runtime
must come from the notebook. Local CPU tiny-model bitwise equality is useful
regression evidence but does not replace the T4 checkpoint check.

## Final local validation

169 affected tests passed, zero deselected. The three named tests above all
ran successfully; the obsolete K_base test was updated, the other two unchanged.
Two additional old test fixtures were corrected: interventions artifacts for
all active arms/updated arm count, and the saved mid-task-2 learner index in the
native resume test. Native final weights/tasks/RNG resume checks remain intact
and passed. No production resume checks were removed.

The actual downloaded baseline archive passed the existing full-result validator
and supplied all 150 full-precision reference losses (1050 total epochs).
Two-GPU dispatch, compatibility failure gating, CG-IR failure isolation, exact
baseline comparison and delayed reuse are covered by focused tests. Notebook
cells parse and contain one implementation pin. No T4 run/full suite was launched.

## Current policy and notebook refresh (2026-10-22)

All twelve generated `vna4_*.ipynb` files are refreshed together by
`python -m scripts.build_kaggle_vna_parallel --commit <full source commit> --variants`.
The immutable source commit contains the engine changes and updated diagnostic
CLI; the later notebook packaging commit is deliberately not the runtime pin.
No source ZIP is required for the published notebooks.

For the authorized Gram seed-0 run, use `vna4_gram_m1s0_ir.ipynb` on T4 x2.
The engine is `schur_gamma_bound_rho25_v2`: dead-only GIR, live ablation off,
residual-bound gates, rho >= 1/4, no CIL interventions at task1, and stability
logs including 20 subsequent native SGD losses. Each current profile finishes
those 20 steps before declaring PROFILE_COMPLETED. Historical input theta is
for common-checkpoint diagnostics only; full training starts fresh or resumes
its matching current-policy checkpoint. The previous 8c56d02 whole-application
gate is replaced with an explicit fixed-system gate; it is not silently waived.

On session2 attach the previous output archive. Matching completed profiles
are validated by diagnostic identity and returned from cache; they are not
profiled again. The fixed-system PASS is read from that matching Gram profile
and checked again before training. A mismatched policy/commit/checkpoint
cannot supply the gate or K0. Main runs resume native checkpoints, and completed
main jobs return their validated results. Old-policy checkpoints remain separate.

The standalone `vna4_exact_diagnostics.ipynb` profiles current CG-IR/exact on
GPU0 with capacity, and current Gram/power10 (fixed-system reference check)
then current CG-IR/power10 on GPU1. Its strict solver comparison is between
current-policy power10 applications. Exact-vs-power10 is descriptive. Both
comparisons remain explicit even when they FAIL; this notebook does not train
M1. Its `MODE='smoke3'` uses a fresh current-policy CG-IR preflight.

Generic main/competitor outputs now live under
`outputs/<source-commit-prefix>/<RUN_TAG>/`, keeping the collector's
`<RUN_TAG>/<job>/<artifact>` layout intact. M3 has a separate `_random` schedule
filename and random-order preflight. Baseline-only/competitor policies and
tuning rules are not altered. Existing generic M1/M1s0 holds remain; no CG-IR
M1 is enabled by this packaging update. The dedicated Gram notebook is the
authorized M1s0 entrypoint.
