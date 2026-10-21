# Same-commit Gram M1s0 and CG-IR T4 profile

Notebook: `notebooks/vna4_gram_m1s0_ir.ipynb`.
Inputs: CIFAR-100; the old full/noBN/superclass/s0 task-1 checkpoint or
`vna4_main.tar.gz`; previous output archives to resume. The old archive also
contains the completed baseline and its full-precision epoch history.
GitHub Secret `github_token` and T4 x2 are required by the shared bootstrap.
No source ZIP. Both active lanes import the one pinned implementation commit;
8c56d02 is cloned solely for the requested historical regression oracle.

| Lane | Sequence |
|---|---|
| GPU0 | Gram/power10 K0 preflight; wait for compatibility PASS; full seed0 20 tasks; baseline verification/continuation |
| GPU1 | 8c56d02 Gram; same-commit Gram; strict compatibility gate; same-commit CG-IR/exact with capacity; idle |

Output archive: `/kaggle/working/vna4_gram_m1s0_ir.tar.gz`.
Output paths include the pinned commit prefix. Per-job logs and atomic native
checkpoints are archived every 15 minutes and after stages. Exit 75 pauses
remaining launches; attach the output and rerun to continue. A CG-IR failure is
reported without cancelling the approved Gram run. CG-IR never trains M1 here.

Reports under `profiles/`:
- `gram_compatibility.json`: historical/current Gram, strict PASS required;
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
