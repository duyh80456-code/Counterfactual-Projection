# v4 main run: rules shared by every phase (read first)

Spec: `docs/vna/VNA_agent_prompt_main_run.md` (the "prompt"). Authority: the theory
`docs/vna/vna_gir_theory.pdf` / `docs/vna/vna_gir_theory.tex` + `docs/vna/theory/*.tex`
(T26–T28 in `theory/s9b_closure.tex`, T25 in `theory/s9_unified.tex`). Reference numerics:
`docs/vna/check_t25.py`, `docs/vna/check_t26_28.py` (both pass: 0 violations).
Competitor survey: `docs/vna/survey_competitors.md`. Repo rules: `AGENTS.md`.

Branch `vna-v4-main` is already checked out. Do NOT commit, do NOT create branches.

## Environment
- Python: `.venv/bin/python`; env `REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
- No network in your sandbox. Third-party clones already present (gitignored) under `third_party/`:
  PyCIL f3509b8 (has icarl, podnet, wa, gem), loss-of-plasticity a6b7958,
  upgd b75e90ad4b09c28971ac9dbb902a8fd86709b28c (MIT),
  C-Flat f57a557523136834bdc28a4902cde9842859b2a8 (no LICENSE file),
  warm_start 896176b2cf89f259bcde8e241a8987e6019741d0 (Ash & Adams S&P official, no LICENSE file),
  dopamine 5873f5494ee0c2d7c016d0ab2ad530354fec59d0 (Apache-2.0, sparse: dopamine/labs/redo).
  Add each to `scripts/fetch_third_party.sh` and `third_party/SOURCES.md` with these pins.
- Tests: run ONLY the targeted tests you write/touch, tiny CPU float64 models. Never run the full
  suite. No smoke notebooks. A test that would take > 60 s must be shrunk.

## Resolved spec/theory discrepancies (theory wins; record each in PREREGISTRATION_CHANGES.md)
1. T26(d) failure branch: theory says `mu <- 2*mu` when rho <= 0 (and T26(f) rejection counts as
   rho < 0). Prompt §2.2 uses the Madsen nu-doubling. Implement theory: `mu <- 2*mu`; keep `nu`
   in state/logs fixed at 2 for schema compatibility. Success branch identical in both.
2. T26(e): s_F* = g_hat / (d_hat^T F d_hat) (F, not F+mu); infinity if denominator 0.
3. GIR bias set: prompt §2.4 quantiles {0.1,0.3,0.5,0.7,0.9} plus 0 — consistent with theory §10.4.
4. CG max iterations 200 (prompt §2.1) replaces the old 50 (training solver amendment 2026-10-17).
Any further discrepancy you find: do not guess; implement the theory and list it in your final
report under "DISCREPANCIES".

## GradMax NaN root cause (already diagnosed by the supervisor; for §2.7 / CLEANUP.md)
Reproduced locally: `b5_cbp_gradmax` seed 0 crashes at ~step 1.7k (Kaggle seed 2 at ~30k) with
`linalg.svd: input matrix contained non-finite values`. SVD is NOT the cause: all four model
parameter tensors are already NaN before `prepare_gir` runs. Weights blow up across installs
(layer-0 bias max 4 -> 47 -> 1066 -> 3.6e3 -> 3.4e8 -> 3.8e31 -> NaN; output weights 1 -> 877 ->
3490 -> 2e9). An installed GradMax unit had |w|=20, bias=-47 versus native CBP init |w|~0.8:
`rows_proposal` scales new rows to the mean row norm of [W, b] (bias included) and the bias then
gets `-median(projected)` added, so large biases feed back into larger new units; large hidden
activations then make plain SGD on the output layer diverge. The earlier `eigh` failure (pre
3c41ecf) had the same cause. §2.7 fallback alone therefore cannot rescue a run whose weights are
NaN: in addition to the §2.7 guard, detect non-finite parameters after each intervention, log
`nonfinite_params=true` with the step, and mark the run status `DIVERGED` in summary/status
(no crash, prompt §0.4). Do not change the GradMax formula (prompt §0.1).

## Final report (end of each phase, in your last message)
Files changed/added/deleted; what each targeted test checks and pass/fail; DISCREPANCIES; anything
from the phase you could not do and why. Keep it short.
