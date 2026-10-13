# Stage 1C task: fixes before the Kaggle Phase 0 run

Context: `docs/vna/SPEC.md`, `docs/vna/STAGE1_REPORT.md`. Branch `vna-phase0`, HEAD has Stage 1/1B.
`.git` is read-only for you: do NOT commit; leave changes in the working tree. Use `.venv/bin/python`.
Do not modify unrelated existing runners. If you must touch a shared helper, list it in the report.

## 1. Random-A control must never pass the go test by being empty (BUG)
`diagnostics/signal_curve.py` computes the random-A direction through the guarded
`vna_block_correction`; when the guard skips it, the correction is `{}`, `random_derivative` becomes
0 and `|vna| >= 3*|random|` passes trivially. Fix: add an explicit `apply_gain_guard: bool = True`
argument to `vna_block_correction`; controls (random-A, and any other comparison direction) are
computed with `apply_gain_guard=False`, and their realized gain / skip flag is logged. Any row where
a direction needed by a criterion is empty or nonfinite is EXCLUDED from that criterion and counted
as `excluded_*`. Test: a forced-negative random direction cannot make the criterion pass.

## 2. G1 noise floor at matched sample count
Primary significance statistic = `lambda_full` (stat_samples) vs the shuffled-label null computed
on the SAME number of samples (use ≥3 independent shuffles; log mean and max), and also the
split-half cross-validated λ (already implemented). Keep sub-batch mean+2std only as a logged extra
column. Summary reports, per checkpoint×site, `significant_vs_shuffled` and `significant_split_half`.

## 3. Cheap variant `vna_lite`
correction_lite = λ · Jᵀ K_E K g / c,  c = (tr(K)/n)²  (tr(K)/n from the same Hutchinson estimate).
No CG. Products: Jᵀg (VJP), w = J(Jᵀg) (JVP), J_Eᵀ w (ext VJP), s = J_E(·) (ext JVP), d = Jᵀ s (VJP).
- Implement as `vna_lite_correction(...)` in `methods/vna.py` sharing the `_Products` machinery,
  same invariance guarantees, same diag keys where meaningful (U, V_lite = (Kg)ᵀK_E(Kg)/c, ρ_lite,
  realized gain, counts, time).
- BN consistency: all of Jᵀg, J, J_E must be evaluated in the SAME BN mode. In diagnostics use the
  configured `--vna-bn` mode for all products. Document in the report that reusing the SGD batch's
  train-mode gradient in Stage 2 is only valid with `bn_mode="train"` (copied buffers) or by
  recomputing Jᵀg in eval mode — do not silently mix.
- Tests: descent identity gᵀJ d = λ (Kg)ᵀK_E(Kg)/c ≥ 0 to tight tolerance; finite-difference check;
  invariance; equals the full VNA chain with R replaced by I/sqrt(c) on a dense small oracle.
- signal_curve: log U/V/ρ/realized gain/held-out cosine+directional derivative/time for vna_lite
  next to full VNA, and the same for random-A lite.

## 4. Rank sweep
At each diagnosed checkpoint, choose the top-3 sites by full-VNA V at default r_max using a
SEPARATE selection batch (dedicated generator, not the measurement batch). For those sites run the
counterfactual over-expansion with requested r ∈ {4, 16, 64, width_of_site} (width = current hidden
channels of that block). For each r log: requested and Gromo-effective rank, λ spectrum summary,
U, V, η, ρ (full and lite), held-out derivative and cosine (full, lite, random-A of the same
effective rank, norm-matched plain-gradient LR-boost), realized gains, and time. If Gromo cannot
return the requested rank, log the effective rank and continue (no failure).

## 5. Diagnostic stride
Diagnostics process every 20th epoch (20, 40, …, 200) by default (`--diagnostic-stride 20`);
training still saves every 10 epochs. Watch mode and the two-GPU drain must respect the stride.

## 6. Cost on GPU
Per row, report per-step cost of full VNA (cold/warm) and vna_lite in units of one SGD step at batch
64 on the same device, with CUDA synchronization.

## 7. Go/no-go summary (report only; do not tune)
Evaluate PER RANK r, on rows before epoch 150, excluding empty/nonfinite rows:
- TINY V ≥ 3× random-A V (same effective rank);
- held-out VNA directional derivative negative and |·| ≥ 3× random-A's;
- held-out VNA derivative better (more negative) than the norm-matched LR-boost direction;
- the same three for vna_lite;
- decay rule already defined (epochs ≥180 vs pre-150).
Print a compact table: rows = r, columns = criteria (full / lite), plus cost ratios.

## 8. Notebook
Regenerate `notebooks/kaggle_vna_signal_curve.ipynb` via the builder (default commit = current HEAD;
the supervisor will re-pin after committing). Update final cells for the new columns/table. Keep
resume identity strict: diagnostics produced before this change must be rejected, not merged.

## 9. Report and tests
Append a "Stage 1C" section to `docs/vna/STAGE1_REPORT.md`: what changed, files touched (incl. any
shared helper), new tests, CPU smoke timings for full vs lite, updated T4 runtime estimate for the
stride-20 + rank-sweep plan. Run `.venv/bin/python -m pytest -o addopts='' -q tests/test_vna.py`
and the full suite; report counts.
