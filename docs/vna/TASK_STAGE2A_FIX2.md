# Stage 2a-fix2: unpinned code, smooth correction, pre-registered batch rule, budget policy

Branch `vna-phase0`. `.git` read-only: do NOT commit. Use `.venv/bin/python`. Keep all tests passing.

## 1. Track the branch instead of a pinned commit (user decision)
In BOTH builders (`scripts/build_kaggle_vna_training_notebooks.py`,
`scripts/build_kaggle_vna_signal_curve_notebook.py`): replace `REPO_COMMIT` with
`REPO_BRANCH = "vna-phase0"` (builder arg `--branch`, default `vna-phase0`). Clone the branch tip,
print and save the resolved commit to `OUTPUT/code_commit.json` (append a record per session with
timestamp). Remove the "regenerate with committed hash" preflight; keep the existence checks for the
required modules.
Traceability: each training arm's `latest.pt`/history and each Phase-0 diagnostic completion record
store the code commit. On resume with a different commit: print a prominent warning with old/new
commits and the epoch boundary, append `{"from_epoch": e, "commit": c}` to the arm's
`code_history`, and continue (protocol/config identity checks stay strict; only the code commit
may change). Final results table shows the commit set per arm. Tests for both behaviours.

## 2. Smooth correction (default)
`--lambda-mode {per_step,time_avg,spread}`, default `spread`: when a correction d is computed at
step t, add λ·d to the gradient at each of steps t..t+vna_every−1 (stale d reused; total effect
equals time_avg). Applies identically to `ngd` (norm-matched before scaling, as now) and
`vna_random`. Spread state is part of resume state (bitwise CPU resume test). Skipped/nonfinite
corrections leave no spread residue.

## 3. Smoke diagnostics (report-only, pre-registered thresholds)
After the 2-epoch smoke, print per VNA-type arm: median and p95 of ‖applied delta‖/‖g_block‖,
fraction of corrections skipped, and loss-spike check (max train-loss jump at correction steps vs
non-correction steps). Flags: `IMPULSE_RISK` if p95 > 3; `LIKELY_NULL` if median < 0.05. Print only;
do not change config automatically.

## 4. Pre-registered batch rule
Validation (Cell A) measures, at the training μ and selected sites, cos(d_b, d_128) for b ∈ {16, 64}
plus the disjoint-batch cosine. Rule, applied BEFORE solver choice and printed:
cos16 ≥ 0.5 → (VNA_BATCH 16, VNA_EVERY 10); else cos64 ≥ 0.5 → (64, 20); else (128, 20).
Notebook constants `VNA_BATCH_OVERRIDE` / `VNA_EVERY_OVERRIDE` (default None) win. Solver choice
(cost/overhead) is then evaluated at the chosen batch/every. Use min over sites/epochs of the cosine.

## 5. Budget policy
`GPU_HOURS_BUDGET=30` is the TOTAL across all arms of this seed (document in the notebook). After
the smoke projection: if over budget drop `sgd_cm`; if still over drop `vna_tiny_U`; if still over
raise with the table (unless `ALLOW_OVER_BUDGET=True`). Print which arms were dropped.

## Report
Append "Stage 2a-fix2" to `docs/vna/STAGE2_REPORT.md`. Regenerate all three notebooks. Run all
`tests/test_vna*.py` and report counts. Clean up your own /tmp outputs when done (disk is nearly full).
