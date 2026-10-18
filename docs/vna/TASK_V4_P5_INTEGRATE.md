# Integration (before phase 5) — supervisor findings after merging phases 2, 3, 4

Read `TASK_V4_COMMON.md`, `CIL_SETUP.md`, `COMPETITORS_V4.md`, AGENTS.md. Do not commit.
1. Phase 4 built its own setup (`experiments/v4_competitor_setup.py`, `adapters/v4_competitors.py`,
   `diagnostics/v4_competitors.py`) in parallel with phase 3's `adapters/cil_setup.py`,
   `experiments/main_cil.py`, `experiments/main_cil_results.py`. Unify: every arm (icarl,
   icarl_full, all competitors) must use ONE setup (class order, backbones, epochs/milestones,
   memory, probe batch, metrics, files) — keep phase 3's as the source of truth, make the
   competitors call it, delete the duplicate code. One results/table path in
   `scripts/collect_parallel.py` (`report_v4` and `main_cil_results.table` must not diverge).
2. Wire `icarl_full` to the phase-2 engine (V3Engine T26–T28, r=16, §2.5 schedule) and call the
   real `diagnose()` at task end for every arm (remove the spy fixture from tests).
3. GEM: add `quadprog` to `requirements-upstream.txt` and the notebook install cell.
4. Run only the targeted tests of the touched files (tests/test_main_cil.py,
   tests/test_v4_competitors.py, tests/test_vna_v4_fix2.py); each < 60 s.
Then execute `docs/vna/TASK_V4_P5_NOTEBOOKS.md`.
