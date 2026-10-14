# B5 measurement fixes (supervisor review)

Branch `vna-phase0`. No commit. Do NOT run pytest; allowed checks: pyflakes on touched files and one
`python -m experiments.stage_b5 regression --arm bp --smoke --fake-data` (< 2 min).
**Touch only:** `adapters/lop_hooks.py`, `diagnostics/lop_measure.py`, `experiments/stage_b5.py`,
`scripts/build_kaggle_vna_stage_b_nb1.py`, `tests/test_vna_stage_b5.py`, and a short section in
`docs/vna/STAGE_AB_IMPLEMENTATION.md`. Another agent is editing other files concurrently.

## 1. "First" point measured on the old task (real bug)
With online benchmarks (batch 1) the `within == 1` point uses a 64-sample buffer holding 63 old-task
samples. Fix:
- The measurement buffer is **task-scoped**: cleared at each task boundary (declared benchmark
  boundary: `flip_after` / `change_after` / ImageNet task length — already tracked as `task_steps`).
- The "first" point = the step at which the buffer first holds m samples of the current task (step m
  of the task; ImageNet m = one minibatch, so step 1 as now). "mid" unchanged (step task_steps//2,
  buffer = last m samples of the current task). Record `point`, `within_task_step`, and
  `buffer_task_purity` (fraction of buffer samples from the current task; must be 1.0) in every row.
- Update the T-align/resume test expectations accordingly (write, don't run).

## 2. GIR swap biased towards TINY (real bug)
`tiny_V` can return fewer directions k than the m units to replace; the code tiles them, and
duplicated directions inflate V. Fix: swap_tiny = the k TINY directions **plus (m − k) directions taken
from the random-init draw** used by swap_random (the first m − k rows of the same random tensor, i.e.
common random numbers); never duplicate. Same init-norm rescaling for both arms. Record
`{criterion}_tiny_directions` = k and `{criterion}_m`. If k == 0 the tiny arm equals the random arm and
the ratio is 1 (record it, do not skip). Add a tiny test (write only) that the swap A never contains
duplicate rows and that with k < m the last m − k rows equal the random arm's rows.

## 3. μ default wiring
nb1: if a Stage A `summary.json` is attached (search `/kaggle/input` for `stageA/summary.json`), set
`MU_REL_DEFAULT = summary['mu_rel_default']`, else 1e-2 with a printed warning; pass it to every B5
run and record its source in the run summary.

Report in your final message: the changed lines' intent per item, files touched.
