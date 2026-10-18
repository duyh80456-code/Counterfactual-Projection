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
