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
test files; run the full suite once at most, and only when the spec asks. Do not commit.
Env: `.venv/bin/python`, `REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
