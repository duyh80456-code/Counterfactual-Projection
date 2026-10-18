# Phase 2 fix 2 (supervisor review of the current uncommitted diff)

Read `TASK_V4_COMMON.md`, `TASK_V4_P2_FIX.md`, AGENTS.md. Build on the working tree. Do not commit.

1. `_run_gir`: `proposals` are computed once per site and reused for every unit, so the same
   neuron can be installed into several slots (duplicates). After each accepted swap, remove the
   used direction and make the remaining candidate directions S_x-orthogonal to it (T13 cheap
   rule, prompt §2.4 "ứng viên kế tiếp S_x-trực giao"), keeping r_perp^beta. Add a test: two dead
   slots on a tiny MLP never receive the same incoming row.
2. `_slot_jump` is recomputed inside the proposal loop; compute j_L, j_KD once per unit.
3. ||w||^2 = e^T H^+ e is built by N*c full `op.H` applications (O((Nc)^2)); for iCaRL N=128,
   c=100 this is far too slow. Compute each per-sample c x c block directly from the loss spec
   (CE + KD student metric, as in theory §2.1 H_i), with pinv on range; keep a dense test.
4. Tests (tiny float64, each < 30 s; dense references from `docs/vna/check_t26_28.py`,
   `check_t25.py`): 7.1 a,b,c; 7.2 a,b,c; 7.3 a,b,c; 7.4; 7.8; mid-task resume-bitwise after one
   rejected step and one swap (mu, nu, install ages, model, optimizer, RNG); item 1 test. Update
   or delete the stale exact-check test that expects removed fields.
5. Wire `diagnose()` for B5/PM: at each task end on a fixed probe batch (seeded, from the task's
   data), logged per task for every arm (baselines included), read-only.
Run only these tests. Report item-by-item with test names and pass/fail.
