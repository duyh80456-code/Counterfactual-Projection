# Phase 2: algorithm T26–T28 (§2.2–§2.6) for B5/PM and iCaRL

Read `docs/vna/TASK_V4_COMMON.md`, prompt §2, and theory `theory/s9b_closure.tex` (T26–T28),
`theory/s9_unified.tex` (T25), `theory/s7_select_gir.tex` (T7, T13, T14, T17). Phase 1 left
`# v4-phase2` markers and `docs/vna/CLEANUP.md`; remove those leftovers now and update CLEANUP.md.

Implement in the existing v3 engine (`methods/vna_v3.py`, `methods/vna_ggn.py`,
`adapters/vna_v3_hooks.py`, `adapters/icarl_hooks.py`/`adapters/pilot_icarl.py` — reuse the
existing CG, JVP, BetaRule, statistics, GGNProducts, install_gir; do not duplicate them):
- §2.2 T26 trust region: state `mu`, `nu` checkpointed with the engine (resume must stay bitwise);
  mu0 = 1e-2*tr(F)/p by 8-probe Hutchinson once per run, written to protocol.json; rho from one
  train-mode forward with BN buffers restored bitwise (A1); LM update per TASK_V4_COMMON item 1,
  effective from the next application; rho <= 0 -> skip VNA (`skip_reason=untrusted`) but still run
  GIR with this application's r_perp; step s = min(s_F*, Delta/||d_hat||); exact acceptance
  T26(f): after applying theta' (outside momentum, as now) forward L(theta') (+ L_KD for CIL); if
  either increases restore theta bitwise, mu <- 2 mu, log `rejected=true`; S4 = sign filter only
  (skip if g_hat <= 0, or CIL h_hat < 0), no tau. All §2.2 log fields.
- §2.3 T27: x_O, delta_O = -O J x_O, e_beta, r_perp^beta via one CG on A_beta warm-started from
  BetaRule's solve; candidates (top-k generalized eigvecs of (N N^T, S_x); T14 with BN), sign,
  bias per §2.4, drop phi == 0; score ell_beta with exact small lambda_max. B5: beta = 0 and the
  r_perp computation must be bitwise identical to the old B5 rule (test 7.2c).
- §2.4 T28 net-gain swap for dead and live slots exactly as prompt §2.4 (live_candidates = 4 per
  site budget, protocol.json; exact jumps j_L, j_KD by forward; Gamma via one CG on the
  temporarily installed net; swap iff Gamma > 0 and both jumps <= 0, else restore bitwise;
  after swap: optimizer state reset (T13d), BN gamma/beta/running stats (T17), keep r_perp and
  S_x-orthogonal next direction, drop CG warm start). Unlimited swaps; A refreshed every
  application. All §2.4 log fields incl. contribution_norm at ages 1, 5, 10 tasks.
- §2.5 schedule: geometric K0, 2K0, 4K0... within each task (first application at the first
  iteration; B5/PM after FIFO >= 32). Preflight measures seconds/application (3 points on task 1
  and 2) and picks the smallest K0 meeting budget 1.0x baseline training time; writes `K0`,
  `seconds_per_application` to `_schedules/`. No mu search, no exact check; must finish < 15 min.
  Replace the old K schedule and the `0.3/rho`, clip constants.
- §2.6 T25 diagnostics per application: B, ||w||^2 = e^T H^+ e (per-sample c x c on range),
  reach/unreach, lambda_E exact, V, lambda_E*B, ratio. A read-only `diagnose(model, batch)`
  (dead + dormant tau=0.025 per site, reach/unreach/B) callable for every arm at task end on a
  fixed 256-sample probe batch; must not change params/buffers/RNG (test 7.8).

Tests (tiny, float64, local only; mirror check_t25.py / check_t26_28.py): 7.1 a–c, 7.2 a–c,
7.3 a–c, 7.4, 7.8. Keep existing resume/bitwise tests that cover the touched code passing (run
only those files). Do not commit.

## Phase-1 review follow-ups (do these too)
- `adapters/vna_v3_hooks.py` divergence check appends a `nonfinite_params` event on every step after
  divergence: record it once (first step only), keep the run going (prompt §0.4).
- Phase 1 left the replaced v3 paths importable (see `docs/vna/CLEANUP.md`); delete them here and
  move `_direction(exact=...)` into a test-only helper.
- Add the four new clones (pins in TASK_V4_COMMON) to `scripts/fetch_third_party.sh` and
  `third_party/SOURCES.md` (note: C-Flat and warm_start ship no LICENSE file).
