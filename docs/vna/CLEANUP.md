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
