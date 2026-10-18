# Phase 2 fix: review findings (supervisor). The current working-tree diff is incomplete.

Read `TASK_V4_COMMON.md`, `TASK_V4_P2_ALGORITHM.md`, prompt §2, theory `theory/s9b_closure.tex`,
`theory/s9_unified.tex` (T25), `theory/s7_select_gir.tex` (T7, T13, T14, T17). Build on the
current uncommitted diff in `methods/vna_v3.py` / `adapters/vna_v3_hooks.py`. Each item below is
a concrete defect; fix all, then report item-by-item.

1. LM failure branch (T26d) and rejection (T26f) must do `mu <- 2*mu` (TASK_V4_COMMON item 1);
   code currently does `mu *= nu; nu *= 2`. Keep `nu` logged, fixed at 2.
2. T28 is missing. `net_gain` is defined but unused; GIR still swaps one dead slot (`units[:1]`)
   and `break`s after the first site. Implement prompt §2.4 fully: per valid site, dead slots
   (f_u = 0 at theta') swap when ell_beta(v*) > 0; live slots in increasing ||f_u||_{H_beta}
   order, `live_candidates`=4 per site; exact jumps j_L, j_KD by forward (reject if either > 0);
   Gamma by T28(a) with ONE matrix-free CG on the temporarily installed net (never dense
   `torch.linalg.solve` on J^T H J of the real net — dense is allowed only in tests); swap iff
   Gamma > 0 and both jumps <= 0, else restore params + optimizer state + BN buffers bitwise;
   unlimited swaps; after a swap reset slot optimizer state (T13d), BN gamma/beta/running stats
   (T17), keep r_perp^beta and take the next direction S_x-orthogonal, drop CG warm start.
   Candidates: top-k generalized eigvecs of (N_{r_perp} N_{r_perp}^T, S_x) (T14 with BN), sign
   ±1, bias argmax ell_beta over {0} ∪ {-quantile_q(sigma X a_k), q in .1,.3,.5,.7,.9} (on beta_BN
   with BN), drop phi ≡ 0, refresh every application. Log every swap with the §2.4 fields,
   including `contribution_norm` of the installed neuron at ages 1, 5, 10 tasks (track installs
   in engine state, checkpointed).
3. T27 delta_O: compute x_O = G g_O from the S1 solve (reuse, no extra solve) and
   delta_O = -O J x_O using the real old-logit selector O (known/old class count from ctx /
   learner), not `eold.ne(0)` column detection. e_beta = e - beta O^T W delta_O with W the
   per-sample KD Hessian already in code; r_perp^beta solve on A_beta warm-started from
   BetaRule's solution. B5 (beta = 0): r_perp bitwise identical to the old rule (test 7.2c).
4. §2.5 schedule is broken: preflight passes `iteration_seconds=0` so no K is feasible and K0 =
   max candidate. Measure the real baseline iteration time and seconds/application (3 points on
   task 1 and on task 2), K0 = smallest K whose geometric schedule K0,2K0,4K0,… within a task
   keeps total algorithm time <= 1.0 x baseline training time; write `K0`,
   `seconds_per_application`, `budget` to `_schedules/` and protocol.json; first application
   at the first eligible iteration of each task (B5/PM: FIFO >= 32). Remove `K_base`/0.3/rho/
   clip remnants (keep reading old schedule files only where `b5_seed0_resume.py` needs them).
5. `diagnose()` is a stub (B = ||e||^2, reach = 1, lambda_E = 0). Implement T25 exactly:
   B = g^T G g via one CG on F + mu, ||w||^2 = e^T H^+ e per-sample on range (c x c pinv), reach =
   B/||w||^2 clipped [0,1], unreach = 1 - reach, plus dead/dormant(ReDo tau=0.025) per site.
   Per application also log V, lambda_E = lambda_max(J_E^T H J_E) exact, lambda_E*B, ratio.
   Read-only: params, buffers, RNG, optimizer bitwise unchanged (test 7.8).
6. Delete the v3 leftovers: tau/power-iteration certificate fields and code, exact-check path in
   runners (move `_direction(exact=...)` to a test-only helper), `correction_step` / norm cap /
   `_lf` faithful branch, `mu_rel` grid, `relative_damping` use in training, refresh-every-10,
   `0.3/rho`. Update `docs/vna/CLEANUP.md`.
7. mu state: `mu`, `nu`, install-age tracking must be in `state_dict`/checkpoint; add a
   resume-bitwise test mid-task after at least one rejected step and one swap (tiny MLP).
8. Tests (tiny float64; each < 30 s): 7.1 a,b,c; 7.2 a,b,c; 7.3 a,b,c; 7.4; 7.8; plus item 7.
   Use `docs/vna/check_t26_28.py` / `check_t25.py` formulas as dense references.
Run only these tests. Do not commit.
