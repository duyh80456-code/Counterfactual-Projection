# Phase 1: cleanup (§1), CG solver fix (§2.1), GradMax guard (§2.7)

Read `docs/vna/TASK_V4_COMMON.md` first, then prompt §0, §1, §2.1, §2.7.

1. §1 cleanup exactly as the prompt table says. For every item: grep all references (methods/,
   adapters/, experiments/, diagnostics/, scripts/, tests/, notebooks builders), delete code, its
   tests, presets and generated notebooks (O1/O2/O3 and -R variants, `experiments/pilot_v3_online.py`,
   CCL-DC online adapter and its patch/fetch entry if only used there). No dead flags.
   Keep: B5/PM LoP parts, iCaRL adapter, checkpoint/resume, `experiments/b5_seed0_resume.py`
   (existing B5 runs must still resume bitwise with their pinned commits — those run old code from
   their own checkouts, so deleting code here is fine, but do not break the loader/selector).
   Things the later phases replace (mu grid/preflight mu, step cap, `*_lf`, tau certificate,
   LIVE_SWAP, `r = e_O` GIR residual, S=1, {0,-median} bias, refresh-every-10, K schedule
   `0.3/rho`, `clip(1/4,4)`): delete them now only if nothing else needs them to keep the code
   importable; otherwise leave a `# v4-phase2` marker comment so phase 2 removes them. List both
   groups in `docs/vna/CLEANUP.md` (file/function, reason, "deleted" or "phase 2").
   The exact-check diagnostic (CG 2000 iters, cos_cg_exact) becomes one test-only helper, never
   called from notebooks/runners.
2. §2.1 safe warm start in the shared CG used by training-time VNA (`projection/cg.py` and/or the
   v3 engine wrapper — fix it once in the shared function): compute the warm-start residual; if
   ||r0|| >= ||b|| restart from 0; drop warm starts at task change and after every GIR swap; stop
   at relative residual <= 1e-3 or 200 iterations; log `warm_reset`, `relative_residual`,
   `iterations`, `converged`; if the z solve ends with relative residual > 0.1 skip that
   application with `skip_reason=cg_z`. Test 7.9 (deliberately bad warm start -> restart).
3. §2.7 GradMax: catch `LinAlgError` and non-finite inputs (check `torch.isfinite` on the inputs
   of eigh/svd) inside `prepare_gir(gradmax=True)` -> keep native CBP init, log
   `gir_fallback=true`. Plus the non-finite-parameter detection / `DIVERGED` status from
   TASK_V4_COMMON. Write the NaN root cause (from TASK_V4_COMMON) into `docs/vna/CLEANUP.md`.
   One tiny test: NaN input to prepare_gir(gradmax=True) -> fallback, no exception.

Run only the tests you add or touch. Do not commit.
