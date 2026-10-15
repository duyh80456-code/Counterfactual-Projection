# Full resume + observability for nb1 (Stage A + B5)

Run after TASK_RESUME_NB23 (reuse its deadline / heartbeat / logging helpers; do not duplicate them).
No commit. Do NOT run pytest. Allowed: pyflakes, rebuilding notebooks, and ONE fake-data CPU round trip
per runner (< 3 min) comparing straight vs stop+resume bit-for-bit (print the comparison). Measurement
semantics unchanged.

## Evidence from the Kaggle run (nb1 @ 96d91ac, 3 h)
- Regression: all five cbp arms finished (~14 min each, ~1 ms/step).
- MNIST cbp (GPU1): smoke projected 5.78 ms/step. Afterwards the archived output size stayed **exactly
  constant from t=4419 s to t=10873 s**, although `resume.pkl` should be rewritten (and grow) at every
  task boundary (~6 min). So no MNIST checkpoint reached the archive for 1.8 h: hung, far slower than
  the smoke estimate, or checkpoint/archive broken. Find out which by reading the code paths
  (`LearnHook.after_or_replay` / `_checkpoint`, `stage_b5.run`, `prepare_data`, nb1 archive thread,
  `archive_outputs` budget/exclusions) and report the most likely cause; fix it if it is a bug.
- Stage A (GPU0): after 3 h, no completed unit and no cost line printed.

## 1. Stage A
- Per-unit progress lines: subcommand, epoch, site, rank/μ, seconds, CG iterations + converged; per
  null replicate (kind, i/8, seconds). Print the Stage A cost projection right after its smoke.
- Resume already exists at unit granularity; make each unit smaller if a unit can take > 15 min (e.g.
  `null` per (epoch, site) rather than per epoch), so a deadline stop loses little.

## 2. B5 (regression, MNIST)
- One line per task: benchmark, arm, task, steps, ms/step over that task, online accuracy/error, and
  "checkpoint written (bytes)". Also a line every 10 000 steps within a task (step, ms/step).
- Checkpoint at every task boundary as now, and additionally every 15 min of wall time inside a task
  (mid-task resume by the existing prefix-replay mechanism must stay bit-identical; check in the round
  trip with a mid-task stop).
- Timing check: in the smoke, also time 2000 MNIST steps with the hook **disabled** (plain upstream
  `learn`) and print hook overhead vs upstream ms/step.
- Each runner writes `status.json` (job, task/epoch, step, last update time) every ≤ 60 s; the heartbeat
  prints it, so "no progress" vs "slow" is visible.

## 3. Deadline and resume across sessions
Same `VNA_DEADLINE` / exit-75 / final-archive behaviour as nb2/nb3 (SESSION_HOURS = 11.5). On rerun with
the previous output attached: restore Stage A completed units and B5 `resume.pkl`, resume automatically,
skip finished arms (regression cbp arms already done in a previous session must not rerun). bp arms wait
for Stage A's summary as now; if Stage A is restored complete, they start immediately.

Rebuild nb1 and the standalone Stage A notebook. Short section in STAGE_AB_IMPLEMENTATION.md. Final
message: cause found for the frozen archive size, files touched, round-trip results.
