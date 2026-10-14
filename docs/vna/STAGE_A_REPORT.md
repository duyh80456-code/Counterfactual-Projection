# Stage A implementation report

2026-10-14, branch `vna-phase0`. No commit or Git metadata write. The supplied
Stage-A docs and existing user changes are retained. Results on real Phase-0
checkpoints are **TODO**: this workspace has no epoch-100 ResNet-18 checkpoint,
CIFAR-100 data, or CUDA device. CPU fixtures are implementation validation.

## Implementation and conventions

`diagnostics/stage_a.py` supplies `null`, `mu`, `selection`, `joint`, `solver`,
`a5`, `summary`, `all`, `cost`, and `trace-cost`. Use `.venv/bin/python -m diagnostics.stage_a all
--checkpoint-dir PATH/recipe --data-root DATA --device cuda:0`. Defaults are
rank 16, 64-example corrections, four 64-example statistics batches, eval BN,
cached block inputs, and relative damping 0.01. Each measurement has private
named generators. Statistics and corrections draw from disjoint dataset-index
pools; heldout uses the fixed Phase-0 `heldout` stream and 512 eval images.
Site ranking uses Phase-0's `rank-selection-statistics` and `rank-selection`
streams, with normalized A and the Phase-0 V statistic. No official test data
is used for selection or diagnostics.

Cotangent TINY converts the supplied mean-batch logit derivative to the sum
loss convention consumed by native Gromo, then uses the pinned projected
TINY generalized singular-value solve. It never needs labels. Native eigenvalues
and row subspaces match the float64 `tiny_candidate` oracle. Gromo's lazy
convolution identity needs explicit native-dtype initialization on the private
statistics clone for float64. The original model is never changed.

`prepare_A` is the Stage-A normalization boundary. It records row scales and
an optional final matching scale. Random Frobenius controls exactly match the
normalized tiny_U proposal. Random trace controls reuse
`trace_matched_random_A`, accepting a normalized input proposal. Exact trace
now streams coordinate probes rather than allocating a logit-dimension-square
identity; this bounds memory but can be expensive on real CIFAR statistics
batches. Existing helper defaults and numerical trace definitions remain intact.
Selection reports effective ranks as well as requested ranks. Production
`greedy_V` uses a matrix-free ridge group-OMP objective, numerically checked
against the existing dense objective on float64. This avoids the outgoing
feature Gram's prohibitive size at CIFAR rank 64; selection CG failures are
recorded and exclude the row from decisions. The existing dense default remains
available for old callers and small smoke fixtures.

Joint products concatenate only virtual outgoing B scopes at B=0 and include
all trainable original parameters by default. JVP/VJP through the full native
model gives K_E equal to the sum of the independent site kernels. There are two
shared inverses for the correction chain, rather than independent inverses per
site. Production Stage A uses the output-normalized GGN convention: Hbar = (diag(p)-pp^T)/batch, F=J^T Hbar J,
mu=mu_rel*tr(F)/number_of_output_rows (batch × classes), and
V=||J_E^T Hbar J (F+mu I)^-1 J^T g||^2. The parameter inverse avoids CG on the
nonsymmetric Hbar K operator. Single-site measurements use the block scope;
A4 uses the full model scope. BN sphere projection is applied to d after the
raw V/rho calculation, and realized gains are measured afterward.

There is a convention conflict in the task: it requests both GGN and equality
to `vna_block_correction`, which is Phase-0 Euclidean VNA. An explicit
`VNAConfig` compatibility mode implements the Euclidean audit; `GGNConfig`
selects GGN. Float64 tests lock equality to both corresponding single-site
solvers (direction, V and rho). `vna_block_correction` now preserves the model
input dtype so the required double oracle runs; float32 Phase-0 arithmetic is
unchanged. `diagnostics/signal_curve.py` is unchanged.

Measurement entry points assert identical parameters, all buffers (including
nonpersistent registrations), gradients, module modes, global Torch/CUDA RNG,
and parameter counts. Solvers construct temporary neurons but never install
or apply them persistently. Recipe solver validation retains shared training
configuration and now reports lambda-direction/gradient ratios. Its Stage-A
mode saves unconverged rows without making a solver decision.

## A5 investigation and confirmed numerical defect

Source investigation rules out the gain guard as a mechanism for erasing the
stored raw V: the existing correction computes and records V before returning
an empty guarded direction. The random path does use `apply_gain_guard=False`,
so comparing direction validity can be asymmetric, but this does not by itself
explain a zero raw V. Neither scheduler nor learning rate participates in the
V calculation. Nonfinite or zero random denominators must remain missing,
rather than be converted into zero ratios.

A reproducible numerical failure exists in the old CG path: with rhs
`[1e-20, 2e-20]`, operator `2 I`, and epsilon `1e-30`, its absolute denominator
breakdown test exits before the first update and returns an unconverged zero
solution. Scaling down residuals therefore can produce spurious zero signals.
Stage-A shared solves normalize the RHS before CG and rescale the solution
afterward. A regression verifies the old failure and the new exact nonzero
solution. `diagnostic_ratio` also preserves tiny finite ratios and rejects
invalid controls. CG convergence is explicitly required for decisions.

**Historical A5 status remains unverified.** The regression proves this defect,
but without the actual epoch-100 checkpoint/raw completion file it cannot prove
that it caused the reported Phase-0 ratio. `summary.json` records that limitation
explicitly. No historical Phase-0 number was recomputed or changed, and no
claim is made that the historical zero has been explained conclusively.
The real-checkpoint investigation must inspect raw V, random_V, both CG residuals,
guard flags, finite flags and ratios before marking A5 resolved.

## Output and resume contracts

Every Stage-A CSV row and JSON artifact has `schema_version="stage1d-v1"`,
`code_commit`, seed, device and wall_seconds. Existing other-schema files are
rejected. Each `(measurement, epoch)` writes one atomic completion JSON;
identity includes configuration arguments, schema and checkpoint SHA256.
Completed units retain their producing commit and are skipped without rewriting.
CSV files are rebuildable from markers. Summary excludes unconverged points.

- `null.csv`: one row per epoch/site/null/statistic; eight replicate values,
  lambda1 or rho, means/maxima, empirical p and significance at p<=1/9.
  N0 additionally logs lambda/||g||^2. Null selection is repeated from the null
  residual, with real/null corrections sharing absolute mu.
- `mu_sweep.csv`: four damping values per top-three site plus the joint model,
  U/V/eta/rho, realized gain, negative heldout loss derivative, convergence and
  residual/iteration records. Rule constants are ratio 3 and fraction 0.5.
- `selection.csv`: three independent draws per site/rank/kind, mean and sample
  standard deviation, raw draw records, normalization scales, split-half row
  subspace overlap and random expectation. Heldout derivative uses direction -d.
- `joint.csv`: joint rho using all original parameters, sum/max of block-local
  site rho, two relative dampings, raw/projection gains and shared CG records.
- `solver.csv`: shared-training-config recipe audit at training damping;
  K-FAC/exact cosine and group cosines, nested d16/d128 and disjoint-16 cosines,
  overhead = number_of_selected_sites*sgd_units/vna_every,
  lambda_d_over_g, and solver records. Full sweeps remain in `solver_audit/`.
- `summary.json`: default damping and raw counts/fractions; converged tiny_V vs
  random_frob wins per rank; significant N1 fractions for lambda1 and rho;
  explicit A5 status.

`diagnostics/vna_gate.py` locks thresholds (0.3, 0.5, 0.5, 0.7, 0.99, rank16).
B1/B2 use epoch<210; B4 epoch<150; B3 task>=2 excluding task ends; B5 uses the
second half of tasks and records train accuracy as n/a. Empty or unconverged
measurement sets fail. The caller may supply final train accuracy explicitly.

The generated Kaggle notebook pins an immutable `--commit`, tests all
`tests/test_vna*.py` first, and runs GPU0 A5→two-checkpoint cost smoke→A1→A3, GPU1 A4→A2→A6 after the budget check. It restores
Stage-A directories and tar/ZIP archives, rejects unsafe paths/conflicts, and
archives `outputs/stageA/` hourly using the existing 5 GB peak disk cap. The
last cell computes the summary, displays tables, renders rho(mu), selection V
and null p-value plots, and archives again. The current local changes must be
made available at the pinned remote commit before Kaggle use.

## Verification

Required full suite: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python
-m pytest -o addopts='' -q --basetemp=.stage-a-test-tmp tests/test_vna*.py`
→ **115 passed, 1 skipped, 39 warnings in 266.61 seconds**. The skip is the
existing unavailable-Matplotlib rendering test; all existing VNA tests are
unchanged. After the final smoke fixture and cache changes, the affected Stage-A
and lemma files were rerun: **15 passed, 36 warnings in 191.94 seconds**,
including cached/uncached single-site parity. Compilation, `git diff --check`,
notebook/builder equality, and parsing every notebook Python cell pass.

The final CPU `all --smoke --fake-data` test passed its <120-second assertion;
measurement units took **116.56 seconds** combined (null 19.42, mu 3.71,
selection 75.29, joint 1.17, solver 16.96). Smoke uses rank1, stat4,
correction1, heldout2, 30 CG iterations at 1e-4, three selection draws and all
eight null replicates. Its recipe solver audit also trims rank/statistics/batch
sizes and CG; production uses the unchanged shared training configuration.
Production sweeps retain 1000/1e-6. Earlier overlapping test processes failed
a smoke timing assertion (150.96 seconds); an isolated run remained slow, so
the fixture was reduced further and verified successfully. Timing is sensitive
to host contention. These CPU fixture timings are not T4 estimates.

All task-owned scratch outputs/logs, including the two default pytest scratch
directories under the system temporary directory, were removed after recording
these results. No commit was made. The smoke test requires completion under
120 seconds, validates all CSVs/markers, reruns without rewriting markers,
and rejects changed checkpoints and schemas. Additional tests cover null
invariants, empirical p-values, normalization, cotangent/native parity, single
site GGN and Euclidean equality, a dense full-model GGN oracle, gate constants
and windows, numerical small-RHS regression, archive restore, and the nested-GN
lemma (zero-damping equality and approximately linear O(mu) gaps).

## Results

TODO: Phase-0 checkpoint measurements and tables.

TODO: rho(mu), V by selection kind, null p-values, joint rho comparison and
solver fidelity/cost figures from the real T4 run.

TODO: conclusively identify the historical epoch-100 A5 cause from its actual
checkpoint and raw Phase-0 diagnostics.


## Review round 1

### Damping decision (confirmed by supervisor 2026-10-15)

Stage A and gate defaults now use `damping_norm="outputs"`, namely
mu=mu_rel*tr(F)/n_out with tr(F)=tr(Hbar K). `GGNConfig` defaults to
`"params"` to preserve existing training arithmetic; both single-site correction
and filtered cotangent honor the switch, as does the joint solve. Stage-A solver
audits explicitly opt into outputs. Every measurement row records `mu`,
`mu_rel`, `damping_norm`, `trace_F`, `n_out`, `p`, and
`mu_rel_equiv_params=mu/(trace_F/p)`. Selection aggregates retain per-draw values
and complete sample records rather than substituting a mean trace into the
nonlinear conversion. The output convention is the implementation decision
confirmed by the supervisor on 2026-10-15.

A5 and the A6 Euclidean control are explicit historical-path exceptions: their
mu uses Phase-0 tr(K)/n_out, with `damping_metric="phase0_euclidean_trace_K"`.
Their actual mu and the GGN trace used for the equivalent-parameter conversion
are recorded separately; the GGN trace must not be read as their damping trace.
Historical Phase-0 results remain historical evidence. Review round 2 makes
RHS normalization the default in generic CG and Phase-0 corrections; A5 old-path
columns explicitly disable it. Breakdown telemetry remains selectable.

### A5 checkpoint evidence

`a5` writes `a5.csv` for epochs 90, 100, 110, 160 and 200, all eight sites, requested rank16.
It uses Phase-0 `VNAConfig` sizes (stat1024/correction128), native `tiny_candidate`
and `extract_tiny_A` without Stage-A normalization, and `random_A_like` with
purpose `random-A:{site}:16`. Statistics and corrections use exactly the
`statistics` and `vna` generators from `signal_curve.compute`. The checkpoint's
Phase-0 `gpu_data`/`channels_last` protocol is respected, and cached block inputs
match the Phase-0 notebook. Tiny retains its gain guard; random disables it
and shares tiny's damping cache. Both use the same cached logits and residual.
The paired fixed path changes only RHS normalization in both preconditioned CG
solves. Raw `V`, `random_V`, `ratio` and all original diagnostics alias the
`old_` columns; `fixed_` columns contain the normalized-path results. All four
solves per path record iterations, convergence, relative residual and explicit
breakdown flags. Gain guard, finite flags, residual norm, tr(K), and mu are saved.

`summary` marks A5 `resolved` only when an epoch-100 row has `old_ratio==0` and
a finite positive `fixed_ratio`. `A5_evidence` names the relevant columns. If
`old_cg_z_breakdown` or `old_cg_q_breakdown` is true and fixed V recovers, that
column identifies absolute CG breakdown as the numerical explanation. A gain
flag alone cannot explain zero raw V; inspect finite flags and both raw V
columns when the breakdown evidence is absent. Until a real CSV exists, the
historical cause remains **unresolved**. This workspace has no real checkpoints,
CIFAR-100 data, or CUDA, so no historical Phase-0 number was recomputed.

Periodic recipe checkpoints deliberately omit optimizer and scheduler state.
A5 records missing raw `checkpoint_lr`/`checkpoint_scheduler_state` as null and
`checkpoint_state_present=False`; it separately records
`recipe_scheduler_state_reconstructed` and `recipe_lr_reconstructed` by replaying
the pinned MultiStepLR recipe (milestones100/150, gamma0.1). A full checkpoint's
actual state is retained whenever supplied. Reconstructed state is not presented
as saved checkpoint evidence.

### Trace and budget

`extension_trace` remains exact by default for existing callers and toy tests.
Stage-A random_trace uses a timed automatic choice at CIFAR output sizes:
measure four coordinate VJPs, extrapolate both traces over the 135 A3
site/rank/epoch/draw points including full and split-half statistics, and switch
when exact trace alone exceeds 1800 seconds. The replacement is 64 Rademacher
probes, with identical output probes for reference and random A. Normalization
records the estimator, probe seed/count, common-random-number flag, each estimate's
sample standard deviation and standard error, and the exact-trace cost projection.
Selection CSV also exposes estimator and std columns.

Local real-size CPU smoke used a genuine 11,220,132-parameter ResNet-18,
rank16, CIFAR32 inputs, batch1, and four timed coordinates after warmup:
**0.150521 seconds per coordinate VJP** (100 output rows). Extrapolating coordinate
cost and output count to stat256 gives **532,685,638 seconds** of exact A3 trace
work, including split halves. This deliberately simple CPU extrapolation is
not a T4 measurement; hardware and batching affect it substantially. The
`trace-cost` CLI reproduces the bounded measurement without downloading data.
Actual production matching remeasures on its own device and full input batch.

The notebook runs A5 first on GPU0, then `cost` on epochs90/100 using full
model/data/batch shapes and production CG settings. Each cost unit samples one
site/null replicate/selection draw; A3 times rank64 and conservatively charges
that timing to all three requested ranks. Joint units retain all eight sites.
Per-subcommand projections use the slower of the two checkpoints, scaled to
scheduled counts (A1 ×64, A3 ×27, A2 ×2, A4 ×1, A5 ×8, A6 ×3 per sampled epoch).
This is a projection; varying site/epoch costs remain a limitation. Cost markers
are isolated under `cost_units/` and do not enter scientific summaries. The
notebook prints every subcommand's GPU-hours, adds measured smoke GPU-hours,
archives, and stops above top-level `GPU_HOURS_BUDGET=3` unless
`ALLOW_OVER_BUDGET=True`. There is no local T4 projection claim.

Resume identities include `implementation_revision="review-round-1"`, so old
completion markers cannot silently substitute measurements with the previous
convention on measurement resume. Notebook regenerated; no commit made.

### Review verification and files

Required suite: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python
-m pytest -o addopts='' -q --basetemp=/tmp/vna-stage-a-fix-final tests/test_vna*.py`
→ **125 passed, 1 skipped, 39 warnings in 582.21 seconds**. The skip remains
the existing unavailable-Matplotlib rendering test. All previous VNA test files
except the task-owned Stage-A tests are unchanged. Stage-A tests now cover both
damping conventions and cached/uncached single-site equality, bitwise default
parameter-normalization equality, gate convention filtering, old/fixed CG
breakdown, exact Phase-0 A5 oracle values and absent checkpoint state, shared
trace probes with uncertainty, the A5 evidence rule, and executed notebook
budget rejection/allowance and A5-first command ordering.

The final six-measurement `all --smoke --fake-data` test passed its <120-second
assertion and resume/schema/checkpoint-identity checks. Combined measurement
cost was **88.37 seconds**: null11.97, mu4.42, selection55.34, joint1.75,
solver13.79, A5 1.11. The CPU smoke uses two selection draws (production still
uses three), and A5 trims stat2/correction1/rank1/CG3/one trace probe. Production
A5 remains the Phase-0 configuration. The first suite exposed a nonpersistent
lazy-buffer registration by native TINY on the original A5 model; selecting
on a private clone fixed that invariance violation. The final suite passes.

Python compilation and `git diff --check` pass; the regenerated notebook equals
its default builder, and every notebook Python cell parses. Hourly archiving
starts before A5 and cost preflight, with archive/stop cleanup on budget rejection.
Additional check: `tests/test_cg.py` plus the two Stage-A notebook tests
→ **4 passed in 6.15 seconds**, covering both generic CG cases and the latest
regenerated notebook. All task-owned `/tmp` scripts, logs, trace/A5 fixtures,
and pytest output directories were removed after recording these results.
No commit or Git metadata write was made.

Files changed in this review:
`methods/vna_ggn.py`, `methods/vna_joint.py`, `methods/vna.py`,
`methods/vna_select.py`, `projection/cg.py`, `diagnostics/stage_a.py`,
`diagnostics/vna_solver_validation.py`, `diagnostics/vna_gate.py`,
`tests/test_vna_stage_a.py`, `scripts/build_kaggle_vna_stage_a_notebook.py`,
`notebooks/kaggle_vna_stage_a.ipynb`, and this report. Supplied task documents,
other prior Stage-A files, and unrelated user changes are retained.

## Review round 2 — 2026-10-15

The supervisor confirmed the outputs convention: `mu = mu_rel * trace(F) / n_out`.
`GGNConfig` and training/solver-validation defaults now all select `outputs`;
`params` remains explicitly selectable for historical comparisons. CG normalizes
the RHS by default in every solver, scales warm starts with it, and returns
solutions, residual norms and history in original units. Only A5's `old_`
columns explicitly disable normalization. A5 now measures epochs
90, 100, 110, 160 and 200. Stage A schema is `stage1d-v2`, rejecting stale
completion markers across the solver change; its notebook has been regenerated.
Historical checkpoint conclusions still require actual checkpoint measurements.
