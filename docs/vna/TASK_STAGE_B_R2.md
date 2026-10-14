# Review round 2 (supervisor-approved decisions)

Branch `vna-phase0`, HEAD f96ec89. `.git` read-only: do NOT commit. `.venv/bin/python`.
The three upstream clones are now present under `third_party/` at the pinned commits and their
Python deps (`tqdm`, `tensorboard_logger`) are installed in `.venv`. No network inside your sandbox.

## 1. Damping: `outputs` everywhere (decision final)
- `GGNConfig.damping_norm` default → `"outputs"`. Training engine (`experiments/vna_training_*`,
  `run_vna_training.py`, `vna_training_config.py`, K-FAC path `vna_kfac.py`, solver validation) uses
  it by default. Bump the protocol/config fingerprint so a resume across this change is rejected with a
  clear message. Keep `"params"` selectable (tests that lock the old numbers pass it explicitly).
- Update STAGE2_REPORT.md / STAGE_A_REPORT.md: convention confirmed by supervisor 2026-10-15.

## 2. CG: RHS normalisation by default in every solver
- Move RHS normalisation into `projection/cg.py::conjugate_gradient` itself (`normalize_rhs=True`
  default; solution, residual norms and history rescaled back; relative residual unchanged). Remove the
  now-redundant per-call-site wrappers, or make them delegate. Phase 0 reproduction in Stage A `a5`
  passes `normalize_rhs=False` explicitly for the "old path" columns only.
- Tests: the small-RHS regression now passes through the default path; an existing-scale problem gives
  the same solution as before to ≤ 1e-12 relative.
- Stage A `a5`: epochs {90, 100, 110, 160, 200}. Regenerate the Stage A notebook builder output.

## 3. Upstream tests must run (no skip)
- Fix test bugs found by actually running them: the PyCIL class is `iCaRL` (not `ICarl`) — read the
  real sources in `third_party/` for every symbol you reference; RepDistiller entry points need env
  `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (tensorboard_logger's old protobuf stubs); set it in
  the test subprocess env and in any future runner that imports RepDistiller.
- Add `requirements-upstream.txt` (tqdm, tensorboard_logger, plus anything else the pinned entry
  points need; read their imports) and make the Kaggle builders `pip install -r` it and run
  `scripts/fetch_third_party.sh` before tests.
- Env `REQUIRE_UPSTREAM=1` turns every upstream skip into a failure; the notebooks set it. Locally run
  `REQUIRE_UPSTREAM=1 pytest tests/test_vna_stage_b_upstream.py tests/test_vna_stage_b_losses.py` —
  all must pass, zero skips. Loss-equality tests compare against the **imported** upstream
  `_KD_loss` (PyCIL) and `DistillKL` (RepDistiller).
- Make `STAGE_AB_IMPLEMENTATION.md` and `third_party/SOURCES.md` consistent: fetched and
  license-verified 2026-10-15; list deps and the protobuf env var.

## 4. Drop B6
Remove B6/PINN from the Stage B plan in STAGE_AB_IMPLEMENTATION.md and from any builder/config;
SOURCES.md keeps the "blocked: no license" note and adds "B6 dropped from this round (supervisor)".

## Report
Append "Review round 2" to STAGE_AB_IMPLEMENTATION.md. Run the FULL suite with `REQUIRE_UPSTREAM=1`;
report counts (skips must only be the pre-existing CUDA/optional-matplotlib ones; name each). List
changed files.
