# Stage 2a-fix task: review fixes before any Kaggle training run

Context: `docs/vna/DESIGN_V2.md`, `docs/vna/STAGE2_REPORT.md`. Branch `vna-phase0`, HEAD = Stage 2a.
`.git` read-only: do NOT commit. Use `.venv/bin/python`. Do not touch Phase 0 code paths.

## Must-fix
1. `diagnostics/vna_solver_validation.py`: only the exact rows at the TRAINING damping
   (`--damping-rel`, default 1e-2) must converge; other μ rows are recorded with convergence flags
   and never abort. Summary exposes `exact_converged_at_train_mu`.
2. `choose_solver` (experiments/vna_training_kaggle.py): evaluate ONLY rows at the training μ.
   kfac if min over (epoch, site) of whole-vector cos(kfac, exact) ≥ 0.9 AND all kfac realized gains
   > 0. Otherwise pick the cheapest of {exact, kfac, lite} whose TOTAL overhead
   n_sites × per-correction-cost(SGD units at the training batch size) / vna_every ≤ 1.0; among
   candidates prefer exact > kfac > lite by fidelity; if none fits, raise with the table (user must
   set SOLVER_OVERRIDE). If no Phase-0 checkpoint is attached and no override is set, raise
   (no silent exact default).
3. BN buffers: add a test that every model buffer (running stats, num_batches_tracked) and all modes
   are bit-identical before/after `engine.correction(...)` and after refresh, with the model in
   train() mode as during training. Fix if it fails.
4. Validation config must equal the training config: validation reads rank, vna_batch,
   damping_rel, cg settings, selection, max_sites from the same defaults/args as
   `run_vna_training.py` (single shared source of defaults); the notebook passes identical values.

## Strength / schedule
5. Effective strength: new `--lambda-mode {per_step,time_avg}` default `time_avg` meaning the
   applied coefficient is λ·vna_every (so time-averaged preconditioner ≈ I + λΠΠᵀ). Apply the same
   to the ngd norm-matched control. Log per correction ‖applied delta‖/‖g_batch block grad‖ and its
   per-epoch mean/max.
6. `--vna-start-epoch` (default 5): no refresh/correction before it; first refresh at that epoch,
   then every `--refresh-epochs`. K-FAC factors (and μ/trace) refresh every epoch once active
   (`--factor-refresh-epochs`, default 1); A/sites keep `--refresh-epochs`.
7. vna_batch check: validation adds, at the training μ and selected sites, cos(d_b, d_128) for
   b = training vna_batch and cos between d_b on two disjoint batches; report them. `choose_solver`
   output also prints a warning (not a failure) if the disjoint-batch cosine < 0.3, recommending
   `VNA_BATCH=64` and larger vna_every. Expose `VNA_BATCH`, `VNA_EVERY`, `LAMBDA_VNA` as notebook
   constants passed to both validation and training.

## Minor
8. Notebook runs `tests/test_vna.py tests/test_vna_stage2.py tests/test_vna_training.py` on CPU.
9. Accumulate train loss/correct on device; one sync per epoch.
10. Corrections log: write per-epoch aggregates + a compact per-correction record (no CG residual
    histories) to JSONL; do NOT store the correction log inside latest.pt (store only the counters
    needed for resume).
11. BN tangent projection: log realized gain before AND after projection per correction (eval-mode
    BN is not scale-invariant, so they may differ). No equality assumption.

## Smoke-then-full in the notebook
12. Cell B first runs every arm for `SMOKE_EPOCHS=2` in a separate smoke directory (not resumed into
    the real run), measures epoch-2 seconds, prints projected GPU-hours per arm (×200, ×260 for
    sgd_cm) and total, and stops if total > `GPU_HOURS_BUDGET` (default 30) unless
    `ALLOW_OVER_BUDGET=True`. Queue order: sgd, vna_tiny_V, ngd, vna_random, vna_tiny_U, sgd_cm
    (sgd_cm last). Then the full run proceeds.

## Report
Append "Stage 2a-fix" to `docs/vna/STAGE2_REPORT.md`. Run all VNA tests and the full suite; report
counts. Regenerate notebooks (default commit HEAD; supervisor re-pins).
