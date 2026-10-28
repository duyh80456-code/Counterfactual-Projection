# Agent rules (adapted from DietrichGebert/ponytail, MIT)

Solve the whole task with the least new code. End with one or two lines: what you skipped or did not check.

## Before writing
- Read the task spec and the code it touches. Grep every caller of each function whose signature,
  return value or side effects you change (adapters, experiments, diagnostics, tests, notebook builders).
- Trace the real flow for training hooks: what runs per step / per task / at preflight, what mutates
  model parameters, optimizer state, RNG, logs and counters. A "measure only" call must not update any.

## Smallest complete change
1. Reuse what exists (`methods/`, `diagnostics/`, `experiments/vna_resume.py`, upstream hooks) the way
   surrounding code does. 2. Stdlib / torch / installed deps. 3. Minimum new code.
- No new abstraction, option, config or "for later" code nobody asked for. Keep existing structure.
- Bug fix: fix the root cause once in the shared function, not in one caller.
- Comment only the why, in one line.
- Counters/cadences ("every N steps"): make them robust to skipped steps (compare against the last
  event, not `% N`).

## Never cut
Resume bitwise-exactness, λ=0 / baseline bitwise identity, RNG isolation, deadline/exit-75 handling,
validation of results, anything in the task spec or PREREGISTRATION_CHANGES.md.

## Tests
New non-trivial logic gets one small focused test. Run only the tests the spec names plus the affected
test files; run the full suite once at most, and only when the spec asks.
Env: `.venv/bin/python`, `REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

## Workspace and delivery
- Work directly in `/Users/pxokq/expanding-guide model`. Use this directory explicitly for repository commands.
- Do not create or switch to a separate worktree, checkout, or copied repository folder unless the user explicitly requests it.
- Commit completed requested changes to the current branch. The user grants standing authorization to commit and push without asking again.
- Stage only files belonging to the requested change; preserve unrelated local edits and untracked files.
- Update repository code directly. Do not create or update source ZIP overlays unless explicitly requested.
- New notebooks must pin a real source commit containing their required code.

## Current layout
- Main runs: `experiments/{pilot_v3,main_cil,v4_competitors,v4_b5_width}.py`.
- Native runners: `experiments/runners/{pycil,lop}.py`; atomic artifacts: `experiments/io_utils.py`.
- Build/collect: `scripts/{build_kaggle_vna_parallel,collect_parallel}.py`; ten `notebooks/vna4_*.ipynb` (M1s0 = seed-0 subset of M1).
- Authority and history: theory PDF/source, main-run prompt, PREREGISTRATION_CHANGES.md,
  CLEANUP.md, V4_PRERUN_REPORT.md and pilot_v3_summary.md in `docs/vna/`.
- Retired Stage A/B measurement and old B5 resume entrypoints are removed. In-flight Kaggle runs
  retain their pinned historical sources; do not restore retired modules for new runs.
