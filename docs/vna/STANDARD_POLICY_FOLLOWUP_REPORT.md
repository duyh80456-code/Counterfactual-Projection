# H2/H3 follow-up — 2026-10-29

Implemented calibration fallback without stopping the benchmark, separate geometry
evidence cohorts, fixed-lambda numerical ablations plus a matched reference,
and H3 checkpoints after intervention epochs. No numerical certificates changed.

Actual Python 3.10 / Torch 2.2.2 / Avalanche 0.4 CPU environment:
- Broad affected tests: 32 passed, no skips (STANDARD_POLICY_FOLLOWUP_TESTS.log).
- Final authority/fallback/collector, locked H2 native integration, H3 VNA→GIR
  interruption/resume and export tests: 9 passed, no skips
  (STANDARD_AUTHORITY_RESUME_TESTS.log).

Synthetic fixtures verify state/event restoration and protocol enforcement; they
do not establish real-data solver convergence, accuracy or GPU timing.
No GPU run was performed, as requested. The existing notebook pins are unchanged
until notebook rebuilding. AGENTS.md explicitly permits committing and pushing
completed requested changes.
