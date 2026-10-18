# Phase 4: competitor wrappers (§4 Đợt 2, prepare Đợt 3) on the phase-3 setup

Read `docs/vna/TASK_V4_COMMON.md`, prompt §0.2–0.3, §4, `docs/vna/survey_competitors.md`.
Every competitor runs on the exact phase-3 setup (same backbone variants, superclass/random order,
epochs, memory, seeds, metrics, files). Only the method differs. Thin wrappers that call the
original code; config changes only. `reimplemented=true` + paper equation/section in
protocol.json only when there is no official code.

Đợt 2 (must work now):
- `podnet`, `wa`, `gem`: PyCIL originals (`experiments/competitors.py` style), original configs.
- `icarl_cbp`: GnT from `third_party/loss-of-plasticity` applied to the iCaRL backbone;
  grid replacement rate {1e-6,1e-5,1e-4,1e-3}, maturity 100.
- `icarl_redo`: port `third_party/dopamine/dopamine/labs/redo/weight_recyclers.py` dormant score
  and reset line by line to PyTorch; `reimplemented=true` citing the file/functions;
  grid tau {0,0.025,0.1}, reset every {1,5} epochs.
- `icarl_snp`: `third_party/warm_start` is the official S&P code; call/port its shrink-perturb
  routine (cite file/function); grid shrink {0.4,0.8}, sigma {1e-4,1e-2}, at each task start.
- `icarl_upgd`: optimizer from `third_party/upgd` replacing SGD in iCaRL; grid = paper's
  3 noise x 3 utility-decay values (read them from the repo/paper configs, cite).
- `icarl_cflat`: `third_party/C-Flat` on iCaRL, original config; plus a reproduction runner for
  their original protocol (random order, BN, full epochs, 1 seed) comparing to their published
  number (record the number and its source file/README line).
- Sweep runner: seed 0, noBN, superclass, select by average incremental accuracy ->
  `best_config.json` per method; best config then runs 3 (later 5) seeds x 2 backbones.
Đợt 3 (code prepared, not run): L2-Init, Forget-Forgetting (reimplemented from paper),
FIRE, SWR, MRFA, BiC, DER/FOSTER/MEMO (with param counts), ablations `+gir`, `+vna`,
`full_beta0`, `full_Arand`, `kernel_self`, GIR init {random, GradMax, TINY}. If an official repo
is needed and absent (no network), add its URL to `scripts/fetch_third_party.sh` and leave the
wrapper raising a clear "fetch first" error; list these in the report.
B5 width sweep: widths 5/20/100 x {bp, bp_vna, bp_gir, bp_vna_gir} x 3 seeds on the v4 algorithm.

Test 7.7: each Đợt 2 wrapper runs 2 tasks x 1 epoch on ~1% fake data on CPU locally, and with the
method disabled (λ=0 or equivalent) is bitwise equal to `icarl`. Keep each under ~60 s. Do not commit.
