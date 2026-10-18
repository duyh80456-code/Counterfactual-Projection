# Phase 5: notebooks (§6), PREREGISTRATION (§8), pre-run report (§9)

Read `docs/vna/TASK_V4_COMMON.md`, prompt §6, §8, §9.
- Builder `scripts/build_kaggle_vna_parallel.py`: replace presets with M1 (noBN superclass,
  icarl+icarl_full, 3 seeds, authority preflight K0 -> `_schedules/icarl_noBN.json`), M2 (BN,
  own preflight), M3 (noBN random; reuse M1 schedule if attached else self preflight with
  `schedule_source=self`), C1 (sweeps seed 0 noBN: CBP, ReDo, S&P, UPGD -> best_config.json),
  C2 (PODNet, WA, GEM, C-Flat x 3 seeds x 2 backbones), C3/C4 (CBP/ReDo/S&P/UPGD best config,
  attach C1), C5 (C-Flat reproduction), B5w (CPU width sweep). Kaggle T4x2: one lane per GPU.
  Keep existing checkpoint/resume/deadline/exit-75/archive machinery. All notebooks pinned to
  one REPO_COMMIT placeholder argument (the supervisor regenerates with the real hash).
  Final cell per notebook: §5 table + flag rates (`gir_fallback`, `warm_reset`, `rejected`,
  `skip_reason`) + stability–plasticity figure.
- Generate notebooks into `notebooks/` with `--commit PLACEHOLDER` (supervisor repins).
- `docs/vna/PREREGISTRATION_CHANGES.md`: new dated section (2026-10-19) with all 8 items of §8
  plus the discrepancies from TASK_V4_COMMON and earlier phases' reports, and the GradMax note.
- `docs/vna/V4_PRERUN_REPORT.md` per §9: CLEANUP summary, `git diff --stat` of methods/ and
  adapters/, test results 7.1–7.9 (pass/fail each), competitor repo+commit+reimplemented flag,
  estimated preflight time.
Builder test: each notebook builds, is valid JSON, and contains the expected jobs. Do not commit.
