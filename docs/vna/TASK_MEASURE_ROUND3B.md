# Round 3b (supervisor): schedule-based exact-vs-CG check, resume tests, Cholesky test

Branch `vna-measure-round2` (HEAD 1a05179). Do not commit. `.venv/bin/python`,
env `REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

1. In `diagnostics/lop_measure.py` and `diagnostics/icarl_measure.py` remove the `self._measured` flag.
   The exact-vs-CG comparison runs only at the first point of the measurement schedule:
   - B5: the first task in the schedule, its `first` point;
   - B3: the first measurement point of task 1.
   Pass the needed schedule info through `ctx` from `adapters/lop_hooks.py` / `adapters/icarl_hooks.py`
   (no instance state). On resume the comparison is never recomputed: rows restored from the checkpoint
   stay as they are and are not duplicated.
2. `tests/test_vna_stage_b5.py::test_scalar_regression_full_measurement_alignment_and_resume` must pass
   **without modifying that test**. Add an analogous resume test for B3 in `tests/test_vna_stage_b3.py`
   (straight run vs stop+resume across the first point: identical rows; exact-vs-CG present exactly once).
3. Create `tests/test_output_cholesky.py` (placeholder until the supervisor's own file arrives): tiny
   float64 model, `OutputFactor(factor_mode='cholesky')` vs `eigh` vs CG (tol 1e-12) for CE, KD, MSE and
   both μ, comparing z and d (relative error ≤ 1e-8); plus a principal-submatrix group factor vs a freshly
   built one.
4. Run ONLY: the B5 test above, the new B3 resume test, `tests/test_output_cholesky.py`.
5. Rebuild nb1/nb2/nb3 with `--commit` of the current HEAD.
