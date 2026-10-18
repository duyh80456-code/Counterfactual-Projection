# Phase 1 cleanup

Deleted unused online ER setup and obsolete pilot artifacts:

- `experiments/pilot_v3_online.py` and its `OnlineCE` test; CCL-DC is not part of the registered setup.
- `notebooks/kaggle_vna_parallel_O1.ipynb`, `O2.ipynb`, `O3.ipynb` and their `-R` variants; the online ER setup is out of scope.
- `third_party/patches/CCL-DC/0001-remove-unused-cgi-import.patch` and the optional CCL-DC fetch entry; they had no remaining user.
- The O1/O2/O3 presets, online runner branches, and `*_lf` arms in the v3 pilot builders/tests; these were obsolete claims and step-rule variants.

Phase 2 removed the v3 μ-grid, norm/K cap, τ certificates, exact run checks, fixed refresh cadence, old GIR residual/bias choices, and clipped adaptive schedule. The engine now uses T26 trust-region damping, T27 residuals, T28 net-gain swaps, and a measured geometric K0 schedule. `correction_step` remains importable only as a compatibility helper for historical diagnostic tests; production code does not call it.

The v4 algorithm keeps the matrix-free CG, JVP/VJP products, BetaRule, and BN/RNG-preserving functional measurement paths shared with v3. Live-slot swaps use the T28 measured net-gain primitive and the four-slot per-site compute budget.

## GradMax NaN root cause

The SVD/eigh failure is downstream of non-finite model parameters. GradMax row proposals scale new rows to the mean norm of `[W, b]` (including the bias), then add a projected bias shift. A large bias therefore feeds back into larger new units; large hidden activations then make plain SGD on the output layer diverge. Phase 1 guards non-finite eigensolver inputs and catches `LinAlgError`, preserving native CBP initialization and logging `gir_fallback=true`; parameter non-finiteness is handled as run divergence by the training status path.

Phase 2 fix 2 consumes GIR directions with S_x orthogonal projection and rescoring after each accepted swap, computes each slot's measured jumps once, and restores engine caches during rejected trials. Loss specs provide per-sample Hessian blocks for T25 range energy. Task-end B5/PM probes use fixed seeded current-task observations (up to 256), are checkpointed, and log diagnostics for every arm.

## Integration cleanup (2026-10-19)

- Deleted `experiments/v4_competitor_setup.py`: duplicate `class_order`,
  `SUPERCLASSES`/`COARSE_NAMES`, `SkipScale`, `skipinit`, `read_only`,
  `task_metrics`, and `feature_accuracy`. Class order/SkipInit now come from
  `adapters/cil_setup.py`; task measurement, feature probes and read-only context
  come from `experiments/main_cil.py`. The tiny `FixtureBackbone` moved to the
  shared adapter and pools to 4×4 only for local synthetic tests.
- Deleted `diagnostics/v4_competitors.py`, including `Logits` and its independent
  `diagnose`: all arms now use `methods.vna_v3.diagnose` with shared sites.
- Deleted `scripts.collect_parallel.report_v4`: all v4 CIL records feed
  `experiments.main_cil_results.table` and the same stability–plasticity figure.
- Replaced the parallel builder's old P/Q presets with nine main-run presets.
  Historical B5 checkpoint restoration still has an active caller; its legacy
  execution cell moved into `build_kaggle_vna_b5_seed0.py`, rather than dropping
  original-source resume handling. Previously generated historical notebooks
  are retained; this phase adds the nine v4 artifacts.
- Removed the phase-3 diagnose spy from tests. Main-run fixtures now execute
  the real diagnostics and public T26–T28 engine, including authority preflight.

## Cleanup 2: v4 and in-flight B5 only (2026-10-19)

Audited imports, runtime notebook cells, and callers before deletion. Kept the nine
v4 notebooks, B5 seed-0/group A–C notebooks, their entrypoints and dependencies,
`docs/` history, and `third_party/`. No commits were made.

Deleted files:

- `adapters/pilot_lop.py`
- `adapters/repdistiller_hooks.py`
- `diagnostics/lop_measure.py`
- `diagnostics/signal_curve.py`
- `diagnostics/stage_b_smoke.py`
- `diagnostics/vna_solver_validation.py`
- `experiments/phase0_discovery.py`
- `experiments/pilot_b3.py`
- `experiments/pilot_b5.py`
- `experiments/run_vna_training.py`
- `experiments/stage_b12.py`
- `experiments/stage_b4.py`
- `experiments/train_recipe_checkpoints.py`
- `experiments/vna_compare.py`
- `experiments/vna_training_config.py`
- `experiments/vna_training_kaggle.py`
- `experiments/vna_training_support.py`
- `methods/vna_kfac.py`
- `notebooks/kaggle_vna_b3_quick.ipynb`
- `notebooks/kaggle_vna_cifar_pilot.ipynb`
- `notebooks/kaggle_vna_parallel.ipynb`
- `notebooks/kaggle_vna_parallel_P-B5.ipynb`
- `notebooks/kaggle_vna_parallel_P-PM.ipynb`
- `notebooks/kaggle_vna_parallel_P10.ipynb`
- `notebooks/kaggle_vna_parallel_P7.ipynb`
- `notebooks/kaggle_vna_parallel_P8.ipynb`
- `notebooks/kaggle_vna_parallel_P9.ipynb`
- `notebooks/kaggle_vna_parallel_Q1.ipynb`
- `notebooks/kaggle_vna_parallel_Q2.ipynb`
- `notebooks/kaggle_vna_parallel_Q3.ipynb`
- `notebooks/kaggle_vna_pilot_b5.ipynb`
- `notebooks/kaggle_vna_signal_curve.ipynb`
- `notebooks/kaggle_vna_stage_a.ipynb`
- `notebooks/kaggle_vna_stage_b_nb1.ipynb`
- `notebooks/kaggle_vna_stage_b_nb2.ipynb`
- `notebooks/kaggle_vna_stage_b_nb3.ipynb`
- `scripts/build_kaggle_vna_b3_quick.py`
- `scripts/build_kaggle_vna_cifar_pilot.py`
- `scripts/build_kaggle_vna_pilot_b5.py`
- `scripts/build_kaggle_vna_signal_curve_notebook.py`
- `scripts/build_kaggle_vna_stage_a_notebook.py`
- `scripts/build_kaggle_vna_stage_b_nb1.py`
- `scripts/build_kaggle_vna_stage_b_nb2.py`
- `scripts/build_kaggle_vna_stage_b_nb3.py`
- `scripts/build_kaggle_vna_stage_b_rest.py`
- `scripts/build_kaggle_vna_training_notebooks.py`
- `scripts/fetch_teachers.py`
- `scripts/recompute_gates.py`
- `scripts/smoke_measure_efficiency.py`
- `scripts/smoke_pilot_gir_vna.py`
- `tests/test_competitors.py`
- `tests/test_pilot_b3.py`
- `tests/test_pilot_gir_vna.py`
- `tests/test_vna_split3.py`
- `tests/test_vna_stage_a.py`
- `tests/test_vna_stage_b12.py`
- `tests/test_vna_stage_b4.py`
- `tests/test_vna_training.py`
- `training/__init__.py`
- `training/datasets.py`
- `training/trainer.py`

Deleted functions/classes from retained modules:

- `adapters/pilot_icarl.py`: `ICaRLPilot`.
- `diagnostics/stage_a.py`: `measurement_entry`, `metadata`, `load_json`, `read_csv`, `write_csv`, `Measurement`, `checkpoint_hash`, `merge_measurement`, `recipe_scheduler_state`, `a5_decision`, `summarize`, `_run`, `run`, `real_trace_smoke`, `cost_identity`, `cost_smoke`, `parser`, `main`, `restore_archives`.
- `experiments/cifar_pilot.py`: `LimitedLoader`.
- `experiments/competitors.py`: `task_accuracy`, `run_cil`, `ReDoHook`, `run_b5_seed`, `run_b5`, `parser`.
- `experiments/pilot_gir_vna.py`: `scale_rows`, `slice_rows`, `exact_cache`, `PilotEngine`, `choose_interval`.
- `experiments/shared_protocol.py`: `seed_everything`, `make_train_loader`, `make_eval_loader`, `architecture_label`, `build_optimizer_scheduler`, `rebase_scheduler_from_theta150`, `rebase_scheduler_from_fork`, `assert_fork_protocol_compatible`, `train_epoch`, `save_shared_checkpoint`, `load_shared_checkpoint`, `sha256_file`.
- `experiments/stage_b5.py`: `run`, `_run`, `parser`, `main`, `finalize`, `merge_json`.
- `experiments/vna_kaggle.py`: `restore_outputs`, `archive_with_status`.
- `experiments/vna_provenance.py`: `resume_code_history`.
- `experiments/vna_resume.py`: `file_hash`.
- `experiments/vna_support.py`: `setup_data`, `fast_train_epoch`.
- `methods/vna_v3.py`: `correction_step`.

Removed `V3Engine.last_refresh`, `Calibration.mu_trials`, and the production
`_direction(exact=...)` compatibility branch. The high-accuracy comparison now
uses a copied engine with a tighter tolerance inside its test. Preflight still
runs; its engine protocol/calibration output remains authoritative. Legacy
checkpoint state loaders, model/optimizer/RNG restoration, λ=0 behavior,
deadlines/exit 75, result validation, and the B5 original-source runner remain.

Moved the used bootstrap/setup/runtime cells into `build_kaggle_pilot_common.py`
and removed obsolete pilot templates. M1, seed-0, and group-A generated cells
match the pre-cleanup output except for the obsolete signal-curve module check;
removed that check from all 13 kept notebooks too. The parallel CLI defaults
to M1 and still emits exactly nine variants. Updated README commands and removed
`training*` from package discovery. Retained shared native/measurement helpers
where v4, pilot_v3, B5, or remaining tests use them. Deleted tests solely for
retired code and removed obsolete cases from mixed test files. B5 regression
measurement/resume coverage now uses the existing real `actual_measure` helper.

Validation results: see "Cleanup 3" below.


## Cleanup 3: revised prompt §1 (2026-10-19)

Executed the revised §1 (deletion list from snapshot `d8e4be4`, re-checked on HEAD `d9d2591`);
313 tracked files deleted (old measurement stack, v1/v2 training path, online CIL, old
adapters/probe, old builders, every old notebook, committed outputs, old docs and logs, retired
tests). RepDistiller, One-Shot-TAS-CCIL and CCL-DC are no longer fetched or referenced.

**§1.0 import graph** (`ast` imports only) from `experiments/pilot_v3.py`,
`experiments/competitors.py`, `scripts/build_kaggle_vna_parallel.py`,
`scripts/collect_parallel.py`, the kept tests, plus the v4 runners added after the snapshot
(`experiments/main_cil.py`, `experiments/v4_competitors.py`, `experiments/v4_b5_width.py` —
discrepancy with the prompt's entry list: these implement §3/§4).

REACHABLE: adapters, adapters.cil_setup, adapters.icarl_hooks, adapters.lop_hooks, adapters.v4_competitors, adapters.vna_v3_hooks, diagnostics.vna_align, experiments, experiments.competitors, experiments.io_utils, experiments.kaggle_checkpoint_discovery, experiments.main_cil, experiments.main_cil_results, experiments.parallel_validity, experiments.pilot_v3, experiments.runners, experiments.runners.lop, experiments.runners.pycil, experiments.v4_b5_width, experiments.v4_competitors, experiments.vna_provenance, experiments.vna_resume, methods, methods.gromo_adapter, methods.vna, methods.vna_ggn, methods.vna_losses, methods.vna_select, methods.vna_sites, methods.vna_v3, projection, projection.cg, projection.functional, scripts.build_kaggle_pilot_common, scripts.build_kaggle_vna_parallel, scripts.collect_parallel
UNREACHABLE: diagnostics, experiments.vna_kaggle

`experiments/vna_kaggle.py` is not reached by `ast` imports but is imported by the generated
notebook cells (`scripts/build_kaggle_pilot_common.py`), so it is kept. `diagnostics` is only
the package of `vna_align`. `probe/gromo_adapter.py` moved to `methods/gromo_adapter.py` for
the TINY-init GIR ablation (§4 round 3).

**§1.1 runner split, bitwise check.** Utilities moved to `experiments/runners/pycil.py`,
`experiments/runners/lop.py`, `experiments/io_utils.py`; `NoMeasurement` no longer inherits
`B3Measurement`. `icarl` and `b5_bp` with `--fake-data` (2 tasks) on `d9d2591` versus the
cleaned tree: model state bitwise identical (b5_bp 6 tensors, icarl checkpoint 865 tensors);
`tasks.json`, `vna_log.csv`, `continual_eval.csv` byte-identical. Only the removed
measurement fields (`reproduction`, `gate`, `exact_vs_cg`) are absent from summary.json.
(Local numpy 1.26 lacks `np.trapezoid`; the comparison used a `np.trapz` alias shim for both
trees. Kaggle's numpy 2.x has it.)

**Bug found during verification.** `V3Engine` T26 prediction computed
`op.H(Jx) @ Jx` with `Jx` in the net dtype (float32) against a float64 metric output, raising a
dtype error at the first application of every float32 run; `Jx` is now cast to float64
(`methods/vna_v3.py`). Caught by `tests/test_pilot_v3.py::test_active_resume_preserves_fifo_and_solver_state`.

**§1.7 old Kaggle kernels (user to delete/archive):** `kaggle_vna_parallel_O1`, `O2`, `O3`,
`O1-R`, `O2-R`, `O3-R`, `Q1`, `Q2`, `Q3`, `P7`, `P8`, `P9`, `P10`, `P-B5`, `P-B5_seed0`,
`P-B5_groupA`, `P-B5_groupB`, `P-B5_groupC`, `P-PM`, `kaggle_vna_parallel`,
`kaggle_vna_stage_a`, `kaggle_vna_stage_b_nb1`, `nb2`, `nb3`, `kaggle_vna_signal_curve`,
`kaggle_vna_cifar_pilot`, `kaggle_vna_b3_quick`, `kaggle_vna_pilot_b5`. New notebooks:
`vna4_M1`, `vna4_M2`, `vna4_M3`, `vna4_C1`–`vna4_C5`, `vna4_B5w`.
