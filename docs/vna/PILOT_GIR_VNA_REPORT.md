# GIR/VNA pilot implementation — 2026-10-16

Implemented against `a77a57bef6a8e5eadf5049dbd00a73f5ee90fae8` without committing.
The task context block governs this report: all local performance smokes use
synthetic data on CPU. Real Dohare regression and CIFAR-100 performance,
GPU-hours, the 64.42 ± 1.5 NME reproduction gate, and the real B3 K decision
remain for the user to run on Kaggle. Synthetic errors/accuracies are wiring
checks, not pilot results.

## Implementation and protocol decisions

- `experiments/pilot_b5.py` runs all nine requested regression arm names,
  with seeds 0/1/2, paired data and native `expr`, BP/CBP/GnT implementations.
  CBP's JSON lists five rates. All five are explicit separate settings; the
  notebook reports each rather than selecting a rate from performance.
- `adapters/pilot_lop.py` intercepts only `GnT.gen_new_features`; upstream
  selects units, consumes its ordinary initialization draws, corrects the
  outgoing bias, zeros outgoing weights, resets ages/utilities and optimizer
  state. GIR substitutes the incoming rows afterwards. Its upstream-random
  identity mode substitutes nothing. The normalized random-direction control
  is a separate mode and shares the TINY row norm.
- Pools retain positive generalized eigenvalues only and consume rows once.
  Task changes invalidate the pool. Regression refreshes lazily at replacement
  events, using available current-task observations, on exhaustion or after
  1000 steps. The default pool capacity is eight. Short pools use untouched
  upstream random initialization for remaining selected units and count fill.
- VNA corrections run after the native update, every K=100 updates, using the
  last 128 current-task samples, r=1, mu_rel=.01, lambda=.1 or 1, rho_min=0.
  The first scheduled point in each regression task can have fewer than 128
  samples; it is explicitly recorded as a window-warmup skip. GIR-consumed
  directions 1..m are reserved, and correction selects m+1..m+r. The scalar
  regression selector can have no unreserved direction; that is an explicit
  skip, not a duplicated GIR direction.
- `experiments/pilot_gir_vna.py` uses existing `SolveCache`, `OutputFactor`,
  `tiny_V`, and `joint_correction`. The training damping trace comes exactly
  from `diag(M).sum() = tr(F)`. Preparing the factor is deferred until that
  damping is known. `OutputFactor` and `choose_solver` themselves are unchanged:
  eigendecomposition through 4096 outputs, per-mu Cholesky above that limit.
- `ngd_control` has an output-factor option for exactly `(F+mu I)^-1 g`, using
  the same loss, parameter scope, factor, damping and data as VNA. It matches
  the concurrent joint VNA norm. Its existing callers retain their behavior.
- Reset row norms are RMS initialization norms: LoP `b*sqrt(fan_in/3)` for
  uniform(-b,b), zero random bias; PyCIL
  `sqrt(fan_in * 2/(kernel_height*kernel_width*out_channels))` for its native
  normal conv initializer. Bias is included when normalizing an affine TINY
  direction.
- `experiments/pilot_b3.py` runs the six requested arms. Class-order seed is
  1993; the pilot overrides PyCIL's hardcoded initialization seed 1 with the
  requested seed 0. Each task uses a private, seeded sample of 128 examples
  from its actual training dataset, including exemplars. Sampling and
  transforms have a stream independent of direction draws and upstream RNG.
- B3 corrections use CE plus native KD (T=2, known classes), r=4 at all fifteen
  sites, current optimizer LR, lambda=1, mu_rel=.01 and rho_min=0. BN is eval
  during computations and prior modes are restored. Starting with task 2,
  GIR resets channels whose relative mean post-ReLU activity is <=.1, zeros
  the corresponding conv_b input channels and restores selected BN channels
  to initialization. Activity/rank is recorded for all arms, including icarl.
- The existing resume hooks also persist pools, consumption positions, window
  data, private RNG, intervention events, calibration and task activity.
  Regression resumes at safe pre-learn boundaries; iCaRL resumes task/epoch
  prefixes, moving restored window tensors to the active device. Elapsed cost
  accumulates across sessions, and completed-run reuse preserves smoke timings.

Incoming changes with zero outgoing weights are function preserving; this is
verified in float64 to 1e-12. Removing an *active* channel with a nonzero
outgoing weight is generally not function preserving: native CBP supplies only
mean-activation bias correction. That upstream behavior is deliberately kept.

## B3 quick and K preregistration

`experiments.stage_b3 --measure-profile {full,quick}` defaults to full. Quick
uses the same numerical measurement implementation with 128 samples split
64/64, mu_rel=.01, rank16, the last block of each stage, all eight N1/eight N2
null draws, and exemplar/new groups. Its full-run points are mid-task 2/5/10.
It verifies d and compares exact/CG at task 2 only. Rows include activity,
rank, energies, null p-values, subgroup rho, subspace similarity and timings.
A point over 600 seconds warns and continues. `gate_quick.json` is labeled
exploratory and never overwrites `gate.json` or its formal per-mu files.

The real-data smoke measures corrections at tasks 1 and 2 and extrapolates
linearly to 12,800 correction outputs. K is the smallest feasible value from
50/100/200/400/800/one epoch for `max(t_VNA*12800/n_out)/K <= mean(t_iter)`.
The one-epoch candidate uses the shorter smoke epoch and, when selected,
runs at epoch ends. An infeasible real schedule stops. The decision is written
to `schedule.json` and appended to `PREREGISTRATION_CHANGES.md` before full
training; a copy is archived with smoke outputs. Synthetic schedules cannot
authorize real full runs. No real-data K has been selected locally.

## Notebooks and reproduction commands

- `scripts/build_kaggle_vna_pilot_b5.py` builds
  `notebooks/kaggle_vna_pilot_b5.ipynb`: CPU only, at most four processes,
  all five CBP rates, three paired seeds, two-task smokes, resume/archive and
  per-arm mean ± sample SD with paired differences against BP and CBP.
- Updated nb2 builder/notebook: GPU0 icarl → icarl_vna → icarl_gir_vna;
  GPU1 icarl_ngd → icarl_gir → icarl_gir_randdir. It runs real-data smokes,
  prints K and per-lane/total GPU-hour estimates, sets ALLOW_OVER_BUDGET=True,
  and automatically starts full runs if smoke succeeds and the slower lane
  projects below three 11.5-hour sessions. Deadline pauses, archives and
  heartbeat use existing machinery. The final table includes CNN/NME average,
  last accuracy, forgetting and differences from icarl.
- Both notebooks are built with the current full HEAD hash above. That hash
  does not contain these uncommitted files. The supervisor must repin after
  committing before running the notebooks on Kaggle, as the task context says.

Local synthetic smoke command:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 REQUIRE_UPSTREAM=1 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
.venv/bin/python -m scripts.smoke_pilot_gir_vna --part all --out /tmp/vna_pilot_cpu
```

Synthetic B5 uses two 500-step tasks. Synthetic B3 keeps the native ResNet32,
two epochs per task, 20 fake classes and bounded exemplar memory; its training
and quick windows are four samples solely to bound local CPU matrix size.
Real notebook runs use the registered 128-sample windows and unmodified full
training lengths.

## Validation and local timings

See the adjacent `pilot_gir_vna_logs` directory for the diff, pytest log/XML,
per-arm smoke logs and machine-readable timing summaries. The tables below
are populated from completed local smokes. Timings include CPU contention
with checks and are illustrative extrapolations, not Kaggle forecasts.

### B5: two-task synthetic CPU smoke, seed 0

CBP entries below use rr=1e-3. Each full-length estimate is `smoke seconds / 1000 * 1,000,000`; seeds 1/2 are assumed to have the same runtime, not measured locally.

| Arm | Smoke seconds | ms/step | GIR refresh / correction seconds | Estimated hours for each seed 0 / 1 / 2 |
|---|---:|---:|---:|---:|
| bp | 10.05 | 10.05 | 0.00 / 0.00 | 2.79 / 2.79 / 2.79 |
| bp_ngd_l0.1 | 20.82 | 20.82 | 0.00 / 4.93 | 5.78 / 5.78 / 5.78 |
| bp_ngd_l1 | 18.44 | 18.44 | 0.00 / 6.25 | 5.12 / 5.12 / 5.12 |
| bp_vna_l0.1 | 16.29 | 16.29 | 0.00 / 5.69 | 4.53 / 4.53 / 4.53 |
| bp_vna_l1 | 17.64 | 17.64 | 0.00 / 4.57 | 4.90 / 4.90 / 4.90 |
| cbp | 13.78 | 13.78 | 0.00 / 0.00 | 3.83 / 3.83 / 3.83 |
| cbp_gir | 15.57 | 15.57 | 2.87 / 0.00 | 4.32 / 4.32 / 4.32 |
| cbp_gir_randdir | 22.90 | 22.90 | 3.90 / 0.00 | 6.36 / 6.36 / 6.36 |
| cbp_gir_vna_l1 | 29.11 | 29.11 | 3.22 / 5.90 | 8.09 / 8.09 / 8.09 |

All nine completed two tasks. Per-task errors, rho values, warmup skips, GIR use/fill and timings are in `pilot_gir_vna_logs/b5_smoke_timings.json`. These synthetic errors are not used to select anything.

The nine-arm, one-rate, three-seed proxy totals **34.29 hours on four ideal cores**. Retaining all five rates for the four CBP arms gives **102.09 hours on four ideal cores**, if each rate costs the measured rr=1e-3 time. This latter number is only an explicit proxy: GIR cost scales with replacement events, and actual rate-dependent estimates come from the notebook’s two-task smoke at every registered rate. Process startup, scheduling, varying window warmup fractions, replacement counts and CPU contention make these short-run extrapolations imprecise; no Kaggle CPU speedup is assumed.

### Targeted checks

The 17 numerical/native-interception checks in the two new pilot test files
passed. Two additional schedule-provenance cases passed separately. Two
resume-focused checks also passed after the timing-accounting fix. They cover
native CBP random-source and BP zero-lambda bitwise identity (logs, parameters,
optimizer and global RNG), iCaRL zero-lambda/no-dormant identity, exact NGD
against a dense solve, equal correction norms/nonnegative gradient product,
GIR selected count/outgoing zeros/neutral incoming replacement, positive
nonreused pools, selected conv/BN resets, private RNG/mode restoration, active
GIR+VNA resume, quick/full numerical equivalence, first-point-only verification,
official gate preservation, completed-run reuse, schedule provenance and
notebook syntax/pinning/lanes. Existing affected upstream measurement/loss/
Cholesky files also passed; their log is retained.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 REQUIRE_UPSTREAM=1 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
.venv/bin/python -m pytest -o addopts='' -q \
  --junitxml=docs/vna/pilot_gir_vna_logs/pytest_full.xml
```

The shared real B3 schedule is sourced only from `icarl_vna` smoke. Other
correction-arm smokes retain their own timing records but cannot substitute
another K into the full comparison.

### B5 intervention costs

| Arm | GIR refresh count / mean seconds | Computed correction count / mean seconds | Random fill |
|---|---:|---:|---:|
| bp | 0 / 0.000 | 0 / 0.000 | 0 |
| bp_ngd_l0.1 | 0 / 0.000 | 8 / 0.616 | 0 |
| bp_ngd_l1 | 0 / 0.000 | 8 / 0.781 | 0 |
| bp_vna_l0.1 | 0 / 0.000 | 8 / 0.711 | 0 |
| bp_vna_l1 | 0 / 0.000 | 8 / 0.571 | 0 |
| cbp | 0 / 0.000 | 0 / 0.000 | 0 |
| cbp_gir | 6 / 0.479 | 0 / 0.000 | 0 |
| cbp_gir_randdir | 6 / 0.649 | 0 / 0.000 | 0 |
| cbp_gir_vna_l1 | 6 / 0.536 | 8 / 0.737 | 0 |

Per-event durations and every correction rho are retained in the `b5_*_summary.json` files. Warmup skips are excluded from the computed-correction mean.

The first full-suite run exposed a pre-existing annotation bug in
`MeasurementHook`: a callback returning a plain string was treated as rows
and indexed as a dict. Metadata annotation now applies only to dictionary
rows in a list; arbitrary callback return values remain intact. The existing
alignment tests verify this fix. The final full-suite run uses the corrected
wrapper and the 19 pilot cases collected for that run.

Forgetting now includes the current accuracy in the historical maximum, as
PyCIL’s native matrix formula does. An improved old-class accuracy therefore
has zero forgetting instead of a negative value. The additional focused
regression case passed (`pytest_forgetting.log`); it was run separately from
the full-suite collection.
