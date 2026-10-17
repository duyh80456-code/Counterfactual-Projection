# v3 fixes before running (supervisor, 2026-10-18)

Branch `vna-v3-pilot`, HEAD d8e4be4. Follow AGENTS.md. Do not commit. Env: `.venv/bin/python`,
`REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
Main files: `methods/vna_v3.py` (V3Engine), `experiments/pilot_v3.py` (engine_arm, arms),
`adapters/vna_v3_hooks.py` (LoPV3 / ICaRLV3 adapters, FIFO window), `docs/vna/PREREGISTRATION_CHANGES.md`.

1. **kernel_self detection.** `engine_arm('b5_bp_kself_lf')` returns `kernel_self_lf`, which neither contains
   `'kself'` nor equals `'kernel_self'`, so that arm currently runs VNA. Detect kernel_self with
   `self.arm.startswith('kernel_self')` everywhere (grep every `kself` / `kernel_self` test in the engine and
   adapters; use one helper). Add a test mapping all 12 B5 arm names (`experiments.pilot_v3.B5_ARMS`), plus
   PM / iCaRL / online arms, through `engine_arm` and asserting the engine's kernel (K vs K_E), lf step rule,
   gir / vna / rand / b0 flags for each.
2. **Exact check solves.** When `_direction(..., exact=True)`, CG uses max 2000 iterations (tol 1e-8). An exact
   check whose records are not all converged must not count toward `exact_checks`, `false_skips`, the cos<.99
   tol tightening or the power_iters change; log it with `exact_cg_converged=False`.
3. **GIR-only arms** (`gir`, i.e. `b5_bp_gir`; not `*_vna_gir`): do not compute d, certificates, s or the
   step; each application computes only the split (z, z_O; `split_only=True`) and then runs GIR. GIR must
   not depend on `skipped` for these arms. Row still logged (d fields None / 0, skip_reason None).
4. **prepare_gir:** do not call `_direction` when `residual` is given. If not given, use
   `_direction(..., split_only=True)` with `warm={}` (do not touch the training warm-start cache).
5. **B5/PM window:** reset the FIFO at each task boundary; the first application of a task happens only once
   the FIFO holds at least 32 samples of the current task (then K schedule as before). Record the window
   definition change in `protocol.json` window string and as a new dated entry in PREREGISTRATION_CHANGES.md
   (before any v3 results).
6. **kernel_self protection:** kernel_self uses the same β_rule / d_rep + d_N split as VNA (only the kernel
   differs: K instead of K_E). Only `vna_b0` (and settings without old classes) stays unprotected — it is the
   ablation of protection. Update the T-kself test so that with β = 0 (no old classes) d = (FG)²g still holds,
   and add a test that with old classes kernel_self goes through β_rule. Record in PREREGISTRATION_CHANGES.

Tests: write/adjust the tests above, then run `tests/test_vna_v3.py tests/test_vna_v3_certificates.py
tests/test_pilot_v3.py tests/test_parallel_pilot.py` once. Report diff summary and the pytest tail.
