# v3 fixes, round 3 (supervisor, 2026-10-18)

Branch `vna-v3-pilot`, HEAD 76cf34f. Follow AGENTS.md. Do not commit.

- The exact CG check (`_direction(..., exact=True)` in `V3Engine._apply`, methods/vna_v3.py) runs only in
  preflight (`--preflight-only`). Add an engine flag (e.g. `self.preflight`, set in
  `experiments/pilot_v3.make_engine` from `args.preflight_only`, kept in state_dict).
- The tol / power_iters reached in preflight are written to the schedule (already via `finalize_calibration`;
  verify `cg_tol`, `tol`, `power_iters`). Full arms load them via `configure_schedule` and keep them fixed: in
  a full run nothing changes `self.tol`, `self.solver_tightened` or `self.power_iters`.
- In a full run, no exact check at all (exact fields None, `exact_check_seconds` 0).
- One short test: engine configured from an existing schedule (non-preflight), `_apply` never calls
  `_direction(exact=True)` and leaves tol / power_iters unchanged. Run only that test (and the existing
  `test_nonconverged_exact_check_is_logged_but_not_counted`, adjusted to set preflight=True if needed).
- Append one dated line to the "V3 fixes" section of docs/vna/PREREGISTRATION_CHANGES.md.
