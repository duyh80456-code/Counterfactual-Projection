# Training-time VNA solver (supervisor, 2026-10-17) — make B3 VNA feasible

Branch `vna-pilot-gir-vna`, HEAD 84d0137 (parallel runs already implemented: P1–P6 notebooks,
`experiments/parallel_validity.py`, `scripts/collect_parallel.py`, tiny_V_euclid). `.git` read-only: do NOT
commit. `.venv/bin/python`, `REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

## Why (real T4 numbers)
| | Task 1 | Task 2 | Extrapolated task 10 |
|---|---|---|---|
| n_out (128-sample window) | 1,280 | 2,560 | 12,800 |
| One VNA computation | 125 s | 195 s | ≈1,250 s |
| One train iteration | 0.068 s | | |
The K rule needs K ≥ 1250/0.068 ≈ 18,000 iterations (≈450 epochs) but a task has 170 epochs → NO_FEASIBLE_K
(NGD too, since it norm-matches to d). At task 1, M build ≈37 s; ≈90 s is tiny_V on all 15 sites + d.

## Changes (computation only — the formula for d is unchanged)
Record all of this in `docs/vna/PREREGISTRATION_CHANGES.md` BEFORE any run.
1. **Training solver:** for the VNA/NGD corrections applied during training (B3 and B5 pilots), solve with
   matrix-free CG, `tol=1e-3`, `max_iter=50`, warm-started from the previous solution of the same system
   (cache per arm/site set; reset at task boundaries if dimensions change). Stage B measurement keeps the
   exact output-space solve unchanged.
2. **Do not reselect A every time:** compute tiny_V (and the A for `random_frob` / `tiny_V_euclid`
   selections) at the start of each task and after every 10 applied VNA steps; in between recompute only z
   and d with A fixed. Log `A_refreshed` per VNA computation in `vna_log.csv`.
3. **Extended K set:** {50, 100, 200, 400, 800} ∪ {1, 2, 5, 10} × iterations_per_epoch (currently 40).
   Pick the smallest K with total VNA cost ≤ 1 × total train time. If none is feasible, choose the largest K,
   record `budget_exceeded: true`, and still run (no more NO_FEASIBLE_K skip).
4. **Preflight timing with the new solver:** measure t_VNA for a computation WITH A reselection and one
   WITHOUT, at task 1 and task 2 (extrapolate in n_out to the last task as now); the K rule uses the
   weighted mean (1 reselect per 10 applications → weights 1/10 and 9/10). Save it in the schedule JSON.
5. NGD arm uses the same solver, same K, same window, and norm-matches to the d computed at the same step.

## Tests (write and run only these, plus the affected existing pilot tests)
- On a tiny model: cosine(d_CG(tol 1e-3, warm start), d_exact) ≥ 0.99 for CE and iCaRL losses.
- λ = 0 still reproduces iCaRL bitwise (existing test must still pass) and B5 BP bitwise.
- A is reused between refreshes (assert tiny_V call count) and refreshed at task start / every 10 steps.
- K rule: extended set, weighted t_VNA, `budget_exceeded` path.
Then run the full suite once.

## Notebooks
Rebuild the parallel notebooks (P1–P6 and the shared one) with `--commit` of current HEAD; the supervisor
repins after committing. Append a "Training solver" section to `docs/vna/PARALLEL_REPORT.md`: diff summary,
test results, and the expected preflight output fields.
