# Split each seed's training into 3 notebooks (2 arms each, one per GPU)

Branch `vna-phase0`. `.git` read-only: do NOT commit. Use `.venv/bin/python`.

1. `scripts/build_kaggle_vna_training_notebooks.py` generates, per seed in {0,1},
   `notebooks/kaggle_vna_seed{S}_part{1,2,3}.ipynb` (keep `--commit` pinning, default HEAD):
   part1 = [sgd (GPU0), vna_tiny_V (GPU1)]; part2 = [ngd, vna_random]; part3 = [vna_tiny_U, sgd_cm].
   Delete the old `kaggle_vna_training_seed{0,1}.ipynb` files and update tests/docs referring to them.
2. Each part runs: setup/tests, Cell A (validation + pre-registered batch rule + solver choice,
   unchanged logic, deterministic), smoke 2 epochs for ITS arms only, then the full run of its 2 arms
   in parallel (one per GPU). Each part saves `run_config.json` = {seed, solver, vna_batch,
   vna_every, lambda, rank, damping, code commit, ...} and its SHA256 `config_hash`.
3. Budget: `GPU_HOURS_BUDGET` is per notebook (default 15). Only part3 may auto-drop arms
   (sgd_cm first, then vna_tiny_U); parts 1–2 raise when over budget unless `ALLOW_OVER_BUDGET=True`.
4. Output dir per part: `/kaggle/working/vna_seed{S}_part{P}`; archive name likewise. Resume restores
   only that part's arms (but tolerates other parts' archives being attached).
5. Comparison: new `experiments/vna_compare.py` that scans a root (and /kaggle/input incl. archives)
   for finished arms of the given seed from any part, requires identical `config_hash` across all
   VNA-family arms and identical recipe protocol across all arms (else raise with a diff), and builds
   the table (test acc, Δ vs sgd, cost vs sgd, params, commits) + curves. The last cell of every part
   calls it and shows whatever arms are available (clearly listing missing ones).
6. Tests: notebook generation (6 files, correct arm pairs/GPUs), config-hash mismatch is rejected,
   compare works with partial results, per-part budget/drop policy.
7. Append a short "Split into 3 notebooks" section to `docs/vna/STAGE2_REPORT.md` with run order.
   Run `tests/test_vna*.py`; report counts. Remove any /tmp files you create.
