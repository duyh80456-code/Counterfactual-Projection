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
| FIRE (round 3) | https://github.com/isaac7778/FIRE | 3f73d7848c42dbd42e7770cb3aa61b4fd3ffe3e1 | false |
| SWR (round 3) | https://github.com/JFernando4/collas_2025_swr_paper | 6637c351e5e34d3c7f7f9fe71d26e2c70bc6a2ac | true — init table for resnet32 only |
| MRFA (round 3) | https://github.com/bwnzheng/MRFA_ICML2024 | 2435c35d7d923e79ca5f564739b4f69963fce117 | not wired (separate PyCIL fork) |

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

## Preserved CIL setup and competitor operating notes

# Main CIL runner (phase 3)

`experiments/main_cil.py` reuses pinned PyCIL iCaRL training, the existing
`ICaRLHook` optimizer interception and native epoch/task checkpoints. It does
not modify `methods/vna_v3.py` or `adapters/vna_v3_hooks.py`.

Merge Phase 2 before training: the runner requires its public
`methods.vna_v3.diagnose(model, batch)` function. `icarl_full` uses the engine's
protected `vna_gir` arm through `apply`, `state_dict`, and `load_state_dict`.
Task-local geometric cadence is checkpointed alongside the engine. Damping
state is retained across tasks; task warm starts and cached directions reset.
Real full runs consume the matching backbone authority schedule with `K0`.

```bash
REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
.venv/bin/python -m experiments.main_cil --arm icarl \
  --backbone resnet32_nobn --class-order superclass --seed 0 \
  --out outputs/main/icarl__nobn__superclass__s0

REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
.venv/bin/python -m experiments.main_cil --arm icarl_full \
  --backbone resnet32_nobn --class-order superclass --seed 0 \
  --schedule outputs/_schedules/icarl_noBN.json \
  --out outputs/main/icarl_full__nobn__superclass__s0
```

The default input root is `/kaggle/input/datasets/nlnk1607/cifar100`; use
`--data-dir` to override it. Repeat for seeds 1/2, BN (`--backbone resnet32`),
and the noBN random control (`--class-order random`). The random control
reuses the noBN schedule. Resume by repeating the identical command with the
same output directory; deadline pauses propagate exit 75. A completed run is
returned without training again. Existing checkpoints with different setup
or code identity are rejected.

If task-1 loss is NaN on noBN, the summary records
`nobn_lr_halving_required=true`, and the run/status become `DIVERGED`.
Restart every noBN arm with `--nobn-lr-scale 0.5` in fresh output directories;
the flag is part of the strict comparison and resume protocol. It is never
changed independently for one arm during training.

Each run writes summary, task and continual-evaluation metrics, fixed-probe
diagnostics, protocol and status; full arms also write interventions.
Parameter counts are in task rows, protocol and summary. Feature probes run
on all seen training examples at tasks 5/10/15/20, evaluating all seen test
examples. Their frozen-feature, zero-initialized LBFGS recipe is recorded in
protocol.json and PREREGISTRATION_CHANGES.md.

```bash
.venv/bin/python -m scripts.collect_parallel outputs --run-tag main \
  --json-out main_results.json --stability-figure stability.svg
```

The collector rejects synthetic/preflight/incomplete runs and missing v4
artifacts. Tables report mean ± sample standard deviation, seed-paired deltas
against matching iCaRL, seeds won, and GPU-hour ratios. The SVG includes
standard-deviation error bars on both axes. Helpers are also callable from
notebooks as `experiments.main_cil_results.table(records)` and
`stability_plasticity(records, path)`.

Local checks use embedded CIFAR labels, one float64 noBN batch, a tiny
linear-probe fixture and tiny two-task iCaRL training. They call the real
Phase-2 diagnostics and engine, check coefficient-zero bitwise identity and
exact deadline replay, and build a six-measurement authority schedule.

Authority preflight uses the same runner with `--arm icarl_full
--preflight-only --schedule-output outputs/_schedules/icarl_noBN.json`.
BN uses `--backbone resnet32` and its own schedule. Generated main notebooks
perform this before their full jobs; M3 imports M1's schedule when available.
Competitors share this module's training settings, measurements and table.

# V4 competitor runner

Run from the repo root with `.venv/bin/python`, `REQUIRE_UPSTREAM=1` and
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`. No third-party source is edited.
`methods/vna_v3.py` and `adapters/vna_v3_hooks.py` are deliberately untouched.

Two-task CPU float64 fixtures (never publish or tune real runs from these):

```bash
.venv/bin/python -m experiments.v4_competitors --method icarl_redo --fake-data --out outputs/check_redo
.venv/bin/python -m experiments.v4_competitors --method icarl_redo --fake-data --disabled --out outputs/check_off
```

The fixture has 8 images/class (~1.6% of CIFAR training size), a two-channel
residual backbone and memory 40. It keeps native head growth, training, KD,
exemplar selection, CNN/NME evaluation and task transitions. Production uses
the native ResNet-32 (PODNet's cosine variant), 2000 exemplars, 100/50 epochs,
proportional native milestones and superclass order. The noBN variant uses
SkipInit, biased convolutions and zero initial branch scales. The explicit
`--nobn-lr-scale .5` flag must be shared by all noBN arms if the baseline
first-task NaN rule has been triggered; default is 1. No sweep tunes this flag.
PODNet's later 50 epochs comprise 30 training + 20 native classifier finetuning.

Tune each of CBP, ReDo, S&P, UPGD on seed 0 / noBN / superclass:

```bash
.venv/bin/python -m experiments.v4_competitors --method icarl_cbp --mode sweep --device cuda:0 --out outputs/cbp
.venv/bin/python -m experiments.v4_competitors --method icarl_cbp --mode matched --best-config outputs/cbp/best_config.json --device cuda:0 --out outputs/cbp_best
```

`--seeds 5` extends the default three seeds. PODNet, WA, GEM and C-Flat have
one configuration; `--mode matched` requires no best-config file for them.
Synthetic selections, altered grids, commit/source mismatches and out-of-grid
configs are rejected. CNN average incremental accuracy selects configurations;
ties retain grid order. `--resume` restores a complete task boundary. Deadline
exit 75 replays the interrupted task from its original model/method/RNG state.
GEM requires `quadprog` from requirements-upstream.txt; no substitute QP is used.

C-Flat's full original-protocol comparison (not a local tiny test):

```bash
.venv/bin/python -m experiments.v4_competitors --method icarl_cflat --backbone bn --order random --seed 1993 --reproduction --device cuda:0 --out outputs/cflat_reproduction
```

It uses 200/170 epochs and overrides the original Inc10 config to B0 Inc5.
The README.md:20 results image, Table 1, reports **59.13%** for iCaRL+C-Flat.
The image does not specify CNN versus NME; reproduction_gate.json reports
both deltas and UNVERIFIED_HEAD rather than declaring an unsupported pass.

Every completed run writes summary/tasks/protocol/status JSON, continual_eval.csv,
class_order.json and diagnostics.json. Diagnostics use a fixed seen-class test
probe and the shared CE metric (including the loss Hessian pseudoinverse),
without running or mutating an intervention engine. Raw reach and CG residuals
are retained. CNN/NME metrics include per-class accuracies, old/new/total,
learning accuracy, prior-class forgetting, average incremental and last accuracy.
Feature accuracy at tasks 5/10/15/20 uses all seen train examples, frozen
features and a CE linear head (100 LBFGS iterations), with isolated RNG.
Collectors accept job names `METHOD__BACKBONE__ORDER__sSEED` and require all
20 task/diagnostic/evaluation rows; they reject disabled and fixture runs.
Their v4 table separates backbone/order and reports paired seed deltas/wins.

B5 width sweep is prepared and deliberately gated on persistent public mu/nu
state from Phase 2. It calls the existing native runner and public engine state
API, with no legacy algorithm fallback:

```bash
.venv/bin/python -m experiments.v4_b5_width --out outputs/b5_width --schedule-root outputs/_schedules --resume
```

Supply `b5_width5.json`, `b5_width20.json`, `b5_width100.json`; each must be an
authoritative schedule accepted by the native runner. This enumerates exactly
36 CPU runs (three widths, four arms, three seeds). No width training was run.

Round 3 L2-Init is available with `--method icarl_l2init --config '{"coefficient":0.01}'`;
the original initialization of existing classifier rows persists across head growth.
BiC/DER/FOSTER/MEMO use the pinned PyCIL factories and report parameter counts;
these entries were prepared, not trained. SWR/MRFA/FIRE fail with fetch-first
messages; source URLs and pin requirements are in third_party/SOURCES.md.
Forget-Forgetting is blocked on the full paper equations/supplementary code.
The ablation names are reserved but their implementations remain blocked on
Phase 2/3 integration: the current public API does not expose all requested
initialization variants. No guessed engine state edits or private calls are used.

Files added: adapters/v4_competitors.py; diagnostics/v4_competitors.py;
experiments/v4_competitors.py; experiments/v4_competitor_setup.py;
experiments/v4_b5_width.py; tests/test_v4_competitors.py; this guide.
Files updated: experiments/pilot_v3.py (native width override),
experiments/parallel_validity.py and scripts/collect_parallel.py (v4 validation
and paired tables), requirements-upstream.txt (quadprog),
scripts/fetch_third_party.sh, third_party/SOURCES.md and
PREREGISTRATION_CHANGES.md. No files deleted, no commits made.
C-Flat reproduction additionally retains its native torch initialization seed 1
and shuffle=True NumPy stream (class-order seed 1993); main matched runs use
their run seed for initialization. Both use native deterministic cuDNN settings.

Targeted verification: 19 passed, 1 skipped in 109.50 seconds, followed by
2 passing C-Flat checks in 40.50 seconds after the reproduction RNG correction.
Each wrapper check stayed below 60 seconds (slowest final C-Flat check: 13.15 s).

| Check | Result |
|---|---|
| 7.7 PODNet, WA, CBP, ReDo, S&P, UPGD, C-Flat: native two-task/one-epoch fixture plus disabled bitwise equality | PASS |
| GEM disabled equality and actionable dependency error | PASS; native enabled training SKIPPED (quadprog absent) |
| Superclass partition, seed changes, official train/meta pickle verification | PASS |
| Native noBN ResNet-32 finite forward/backward | PASS |
| ReDo normalized score, selected incoming reset, zero outgoing weights/momentum | PASS |
| Original ResGnT invocation, mature feature replacement, persisted ages | PASS |
| Diagnose parameter/buffer/gradient/mode/RNG isolation and reach bound | PASS |
| Deadline exit 75 followed by bitwise exact task replay | PASS |
| Registered grids, deterministic selection ties, synthetic selection rejection | PASS |
| Collector recipe/provenance validation and fixture rejection | PASS |
| B5 36-job width grid and native width override | PASS; no training |
| Nonfinite intervention produces DIVERGED summary/status | PASS |
| L2-Init retains old classifier-row initialization across head growth | PASS |
| Existing PM horizon/architecture and fixed CBP rate recipe regression | PASS |
| C-Flat original-protocol fixture: native initialization/order seeds, reference and invalid-synthetic gate | PASS |

Syntax compilation, shell syntax and git diff whitespace checks passed. No full
suite, notebook, GPU training, real hyperparameter sweep, full C-Flat reproduction,
B5 width training or Round 3 training was run. SWR/MRFA/FIRE and Forget-Forgetting
remain source-blocked; the requested ablations remain unimplemented pending
Phase 2/3 public-API integration. The two protected Phase 2 files have no diff.

Integration correction (2026-10-19): duplicate setup and diagnostic modules
listed in the historical phase report above are deleted. Competitors use
`adapters/cil_setup.py`, `experiments/main_cil.py` training/measurement helpers,
and the public `methods.vna_v3.diagnose`. Production probes use phase-3 seed 0
and size 256; local fixtures use size 4 and 4×4 pooled features, memory 20.
Metrics and the zero-initialized feature probe now follow phase 3 exactly.
The collector has one v4 table path for main arms and competitors. `quadprog`
was already listed in requirements; v4 notebook installation includes it.

## Round-3 preparation (added 2026-10-19)

Own ablations (8 arms in `experiments/main_cil.py`) and FIRE/SWR competitors are wired and pass
two-task fake-data runs (`tests/test_main_cil.py::test_round3_ablation_arms_run_with_engine_variant`,
`tests/test_competitors.py::test_77_native_wrapper_and_disabled_identity[icarl_fire|icarl_swr]`).
MRFA needs a dedicated runner; Forget-Forgetting has no public code. Definitions:
PREREGISTRATION_CHANGES.md "Round-3 code preparation".

## Current execution plan (2026-10-23; supersedes original job counts)

1. `vna4_gram_m1s0_ir.ipynb`: attach the failed/paused Gram seed0 output and
   CIFAR-100; resume full seed0, retaining the historical baseline verification.
2. `vna4_C1.ipynb`: CIFAR-100; four LR levels x registered method configs,
   92 full seed0 trials. Attach its previous output to continue its native jobs.
3. `vna4_C2.ipynb`: CIFAR-100 plus old GEM output if available; noBN only,
   16 seed0 LR trials followed by eight selected-config seed1/2 jobs. Compatible
   old GEM seed0 supplies its exact LR candidate without retraining it.
4. `vna4_C3.ipynb`: CIFAR-100 plus this version's C1 output containing four
   best_config.json files. Eight seed1/2 jobs; combine with C1 winners for all seeds.
5. `vna4_M1.ipynb`: CIFAR-100 plus Gram seed0 output containing the frozen
   gram_power10.json authority. Four seed1/2 jobs, Gram/power10, no hold.

All generated notebooks share one source commit. M1 does not repeat seed0 or
recompute K0. The seed0 repair accepts registered source80608ab in addition to
29cc735/1bf6d7a; it preserves native checkpoint state and calibration provenance.

After gathering completed outputs (install the project's diagnostics extras
for matplotlib when running locally):

```bash
.venv/bin/python -m scripts.collect_cil_main seed0.tar.gz M1.tar.gz C1.tar.gz C2.tar.gz C3.tar.gz --out results/cil
```

The output includes main_cil_table.csv/.md, average_accuracy_by_task.csv,
PNG/SVG curves and provenance.json. It reports partial seed coverage explicitly
and refuses to pool mismatched solver/curvature/LR/configuration. Include both
C1 and C3 outputs: C3 does not rerun the winning seed0 trials. See the latest
PREREGISTRATION_CHANGES entry for exact trial counts and reuse validation.

Local affected checks: 83 tests PASS (competitor wrappers, joint LR selection,
validated GEM reuse, M1/C2 dispatch and sweep resume, native checkpoint transfer,
fixed Gram math, aggregation/curves, main CIL and diagnostic GPU lanes). Notebook
cells are also parsed after generation. No real T4 run or new seed0 run archive
was available for validation.

### Fixed-LR override (latest user decision)

All noBN jobs now use nobn_lr_scale=0.5 (lr=0.05). The preceding four-LR counts
are superseded: C1 has 23 method-config trials and C2 has four fixed-LR seed0
jobs, then eight matched jobs. All notebook pins are refreshed together.
Current best_config.json must carry lr_scales=[0.5]; regenerate selections
instead of importing a winner chosen at a different LR. Seed0 native resume
also accepts source359da84 without changing saved training state or K0.
