# V4 pre-run report — 2026-10-19

No real training or notebooks were executed. No commit or branch was created;
branch remains `vna-v4-main`. The supervisor must regenerate with the real
shared immutable commit before publication. All nine v4 notebooks currently
use **PLACEHOLDER**, as requested.

## Cleanup and integration

Full deletion inventory and GradMax diagnosis: [CLEANUP.md](CLEANUP.md).
This phase deletes `experiments/v4_competitor_setup.py` and
`diagnostics/v4_competitors.py`, removes `report_v4`, and replaces P/Q presets.
Native competitor method hooks remain in `adapters/v4_competitors.py`.
Historical B5 original-source resume behavior is retained in its own builder.

All main/competitor arms share phase-3 class ordering, SkipInit conversion,
epoch/milestone settings, dynamic memory, fixed probe selection, real public
diagnostics, metrics, zero-initialized feature probes and table/figure functions.
Cosine PODNet sites now receive the same activity diagnostics. `icarl_full`
uses the actual T26–T28 engine (`K0`, r=16) and public task reset. Preflight
uses six read-only measurements on the main setup; damping initialization is
written from the engine's protocol. Collection rejects mixed noBN LR scales.
`quadprog` was already present in requirements-upstream.txt and is explicitly
installed by the generated notebook install cell.

Current `git diff --stat -- methods/ adapters/`:

```text
 adapters/cil_setup.py   | 27 ++++++++++++++++++++++-----
 adapters/icarl_hooks.py |  3 ++-
 2 files changed, 24 insertions(+), 6 deletions(-)
```

There is no methods/ diff. The two adapter changes consolidate SkipInit and
its tiny fixture, and include native cosine residual blocks in site discovery.

## Changed-file inventory

- Modified: `adapters/cil_setup.py`, `adapters/icarl_hooks.py`;
  `experiments/main_cil.py`, `experiments/main_cil_results.py`,
  `experiments/parallel_validity.py`, `experiments/v4_competitors.py`;
  `scripts/build_kaggle_vna_parallel.py`, `scripts/build_kaggle_vna_b5_seed0.py`,
  `scripts/collect_parallel.py`; `tests/test_main_cil.py`,
  `tests/test_v4_competitors.py`, `tests/test_vna_v4_fix2.py`,
  `tests/test_parallel_pilot.py`; `docs/vna/CIL_SETUP.md`,
  `docs/vna/COMPETITORS_V4.md`, `docs/vna/CLEANUP.md`,
  `docs/vna/PREREGISTRATION_CHANGES.md`; `notebooks/kaggle_vna_parallel.ipynb`.
- Added: this report and `notebooks/kaggle_vna_parallel_{M1,M2,M3,C1,C2,C3,C4,C5,B5w}.ipynb`.
- Deleted: `experiments/v4_competitor_setup.py`, `diagnostics/v4_competitors.py`.
- Pre-existing user modification: `.DS_Store`, untouched by this phase.

## Targeted checks

Command environment: `.venv/bin/python`, `REQUIRE_UPSTREAM=1`,
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

The final combined run of `tests/test_main_cil.py`,
`tests/test_v4_competitors.py`, `tests/test_vna_v4_fix2.py`, and
`tests/test_parallel_pilot.py` passed **72 tests**, skipped **1**, in 153.62 s.
The slowest individual test was 14.41 s. A subsequently added independent
read-only preflight test passed; focused follow-ups for the generated notebooks,
main authority calibration, mixed-scale collector, and reproduction-separated
paired table also passed. Thus **73
unique targeted tests passed, 1 skipped**. No individual test exceeded 60 s.
Earlier interrupted/failed exploratory runs were superseded by these checks.

| Prompt check | Result / evidence |
|---|---|
| 7.1 T26 | PASS: constrained SLSQP match, trust radius, bitwise parameter/BN rejection rollback |
| 7.2 T27 | PASS: dense protected residual, Schur gain, beta=0 bitwise identity |
| 7.3 T28 | PASS: direct BFGS net gain, dead-slot gain, rejected-swap state rollback, distinct orthogonal replacements |
| 7.4 T25 | PASS: dense reach and V <= lambda_E B |
| 7.5 Superclass | PASS: 20 groups of five, coverage, seeded order, official pickle/meta validation |
| 7.6 noBN | PASS: float64 ResNet-32 finite forward/backward and parameter count |
| 7.7 Competitors | PASS: seven enabled native wrappers, all eight disabled identities, two tasks × one epoch; GEM enabled SKIPPED because quadprog is absent locally (actionable dependency error verified) |
| 7.8 diagnose | PASS: real task-end diagnostics, unchanged parameters/buffers/gradients/modes/RNG; separate preflight also preserves optimizer and training state |
| 7.9 CG warm start | PASS: deliberately bad warm start resets and matches zero-start solution/history bitwise |

Additional PASS: coefficient-zero main identity; exact exit-75 task replay;
measurement-deadline recovery; real six-point authority calibration; paired
main/competitor tables; sweep provenance and deterministic ties; all nine
notebooks build as JSON with valid cell structure, parseable code and expected
job counts; required flag reports and figures; retained legacy B5 builder parses;
`git diff --check` clean.

Notebook job counts: M1/M2/M3 = 6 each; C1 = 4 sweeps (23 registered trial
configs); C2 = 24; C3/C4 = 12 each; C5 = 1; B5w = 36. The generic parallel
notebook duplicates M1 as the default. Each GPU has one sequential lane.
M3 imports M1's authority or records self-calibration. C3/C4 import C1 selections
with full source/dataset/grid provenance. B5 schedules and tables are separated
by width. The final cell reports seed statistics, flags and a figure; B5 uses
explicitly labelled regression proxies instead of undefined class accuracies.

## Competitor provenance

Existing local clones were verified against these pins by wrapper execution;
no third-party source was edited or fetched during this phase.

| Arms | Official repository | Commit | reimplemented |
|---|---|---|---|
| iCaRL, PODNet, WA, GEM | https://github.com/LAMDA-CL/PyCIL | f3509b8ca3f20660ce4aa13f19d5283de81b4b35 | false |
| CBP | https://github.com/shibhansh/loss-of-plasticity | a6b79580d85f3025bdb601566d3627c5f489f13b | false |
| ReDo | https://github.com/google/dopamine | 5873f5494ee0c2d7c016d0ab2ad530354fec59d0 | true — original reset/score port, NHWC to NCHW |
| S&P | https://github.com/JordanAsh/warm_start | 896176b2cf89f259bcde8e241a8987e6019741d0 | true — native initialization port |
| UPGD | https://github.com/mohmdelsayed/upgd | b75e90ad4b09c28971ac9dbb902a8fd86709b28c | false |
| C-Flat | https://github.com/WanNaa/C-Flat | f57a557523136834bdc28a4902cde9842859b2a8 | false |

## Preflight estimate and discrepancies

Planning allowance: **15 minutes per backbone**, 30 minutes for M1+M2, with
M3 needing no new calibration when M1 is attached. This is an unverified T4
allowance, not a measured production runtime. Actual work is one first-task
epoch, three task-2 optimizer steps, six measured applications, and native
first-task evaluation/exemplar construction. The tiny CPU fixture measured
about 0.34 s/application; it cannot predict full ResNet-32 T4 runtime. The
production <15-minute target remains to be measured on Kaggle.

All eight registration items and discrepancies are consolidated in
[PREREGISTRATION_CHANGES.md](PREREGISTRATION_CHANGES.md), dated 2026-10-19:
fixed nu=2 with mu doubling; undamped F step denominator; five bias quantiles
plus zero; CG=200; UPGD grid union; native S&P perturbations; C-Flat Inc5 and
unidentified published head; phase-3 feature/forgetting conventions; GradMax
nonfinite-parameter root cause. This integration makes no new formula changes.
B5's figure uses explicitly labelled native regression loss proxies because
classification forgetting and learning accuracy are undefined for B5.

Not checked: full suite, smoke notebooks, GPU/real-data training, actual sweeps,
full C-Flat reproduction, B5 width training, production preflight timing, or
Round-3 training. Enabled GEM remains untested locally because quadprog is not
installed; installation is wired into Kaggle. The inherited source-blocked
Round-3 entries/ablations remain outside this integration phase. `.DS_Store`
was already modified and was left untouched. No commits were made.
