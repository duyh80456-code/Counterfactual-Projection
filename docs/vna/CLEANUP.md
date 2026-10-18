# Phase 1 cleanup

Deleted unused online ER setup and obsolete pilot artifacts:

- `experiments/pilot_v3_online.py` and its `OnlineCE` test; CCL-DC is not part of the registered setup.
- `notebooks/kaggle_vna_parallel_O1.ipynb`, `O2.ipynb`, `O3.ipynb` and their `-R` variants; the online ER setup is out of scope.
- `third_party/patches/CCL-DC/0001-remove-unused-cgi-import.patch` and the optional CCL-DC fetch entry; they had no remaining user.
- The O1/O2/O3 presets, online runner branches, and `*_lf` arms in the v3 pilot builders/tests; these were obsolete claims and step-rule variants.

Phase-2 replacements are intentionally left importable and marked by the existing v3 paths: μ-grid/preflight selection (`V3Engine.preflight` and calibration reports), the old norm/K step cap (`correction_step`), exact run checks (`_direction(exact=...)`), power-iteration certificates/tau fields, LIVE_SWAP/measured branches, the old GIR residual/bias/refresh rules, and the `0.3/rho`/clipped K schedule. Phase 2 owns their algorithmic replacement.

## GradMax NaN root cause

The SVD/eigh failure is downstream of non-finite model parameters. GradMax row proposals scale new rows to the mean norm of `[W, b]` (including the bias), then add a projected bias shift. A large bias therefore feeds back into larger new units; large hidden activations then make plain SGD on the output layer diverge. Phase 1 guards non-finite eigensolver inputs and catches `LinAlgError`, preserving native CBP initialization and logging `gir_fallback=true`; parameter non-finiteness is handled as run divergence by the training status path.
