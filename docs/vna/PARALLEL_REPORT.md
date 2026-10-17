# Parallel CIL/B5 pilot implementation — 2026-10-17

## Validity review follow-up

The earlier CPU suite established implementation identities only. The full
experiment pipeline is **held pending real T4 smoke review**. Updated P1–P6
run two-task, real-data GPU calibration into `_preflight/<job>/`, archive it,
and stop before any full CPU/GPU job. They do not treat synthetic timing as
approval. No T4 or CIFAR dataset is available in this local environment;
real feasibility and matched scheduling remain unverified.

Baseline reuse now requires completed ordered tasks, explicit real/full flags,
seed and initialization seed, CIFAR-100 B0 Inc10, exact pinned upstream/native
recipe, the implementation commit, and SHA-256 hashes of train/test/meta.
Existing historical artifacts without these proofs are rejected. This means
the old seed1993 result cannot currently be reused automatically; recovering
verifiable metadata or rerunning a matching baseline is necessary before a
complete paired comparison can be authorized.

Collector accepts only completed job-root artifacts, rejects missing metadata,
mixed implementation commits, synthetic/truncated CIL results, mismatched
comparison protocols, and conflicting duplicates. Nested B5 rate artifacts
cannot overwrite the aggregate job summary. Deliberate arm/seed/device
differences are excluded from comparison signatures; dataset, native recipe,
upstream, initialization, and worker differences are retained.

Targeted validation log: `parallel_logs/pytest_validity.xml`. The previously
recorded full suite below describes the earlier implementation revision.
The affected CIL/validity run passed 24 tests in 143.90 seconds. After adding
rate and preflight guard coverage, the final validity contracts passed all 14
tests in 18.19 seconds (`parallel_logs/pytest_validity_contracts.xml`).

The runtime pin for the common notebook and every P1–P6 variant is
`84d0137c325df9acadf30cf820faee2a5e580870`. It includes the validity fixes
and smoke-only gate. The training-solver amendment supersedes the old
infeasible-calibration behavior by selecting the largest registered K and
recording `budget_exceeded`.

## 0.1 Range(H) investigation

`ICaRLPyCIL.metric_apply` agrees with the autograd Hessian-vector product of
the actual pinned PyCIL loss: full-logit mean CE plus `_KD_loss` on the known
classes at T=2.  Because upstream `_KD_loss` does not multiply its objective
by T squared, its logit Hessian correctly contains the existing `1/T^2`
factor.  The residual also agrees with the autograd gradient of that exact
loss.

The task-10 failure was therefore not a missing KD term.  `OutputFactor`
derived `range(H)` from a relative eigenvalue cutoff.  Saturated softmax
curvature has valid eigenvalues far below that cutoff, even though CE keeps
the mathematical Hessian range equal to the complete zero-sum logit tangent
space.  The implementation now factors H in an explicit orthonormal basis of
that analytic tangent space for CE/KD losses.  It removes exactly the shared
constant-logit null direction and retains tiny nonzero directions.  Other
losses keep the numerical fallback.  The reported
`residual_in_range_H` threshold is now relative error <= 1e-6; the numerical
error remains in every row.  A saturated task>=2 regression test requires
error <= 1e-8 and checks both the upstream residual and Hessian.

### 0.1(c) local remeasurement

No real quick checkpoint is present in this workspace.  A complete search
found no `latest.pt`, `task_*_epoch_*.pt`, or `gate_quick.json` outside ignored
dependencies.  Consequently task 2/5/10 cannot be reconstructed without
training, and no fabricated replacement gate is included.

The new command is:

```sh
python -m experiments.pilot_b3 remeasure --from PATH_TO_ICARL_OUTPUT
```

It refuses a directory without immutable measurement-point states.  Future
measured `icarl` quick runs save only the model, frozen old model, loss/task
state, buffer and point metadata under `measurement_checkpoints/` before measuring.
Remeasure writes `measurements.csv` and a new exploratory `gate_quick.json`
without training.

## Direction selection

`tiny_V_euclid` reuses TINY's exact N statistic and selects leading
eigenvectors of N N^T.  It then applies the same incoming-neuron mean-norm
normalization and true-ReLU sign comparison as `tiny_V`.  Measurement rows
now include `V_tiny_V_euclid`.  Training accepts
`--selection tiny_V|tiny_V_euclid|random_frob`; correction logs record the
selection.  The post-hoc diagnostic status is recorded in
`PREREGISTRATION_CHANGES.md`.

## CIL protocols and outputs

`experiments.pilot_b3` now accepts `--seed`, `--init-cls {10,50}`,
`--increment {5,10}`, `--selection`, and `--auto-schedule`.  Inline automatic
scheduling times the first correction of each task and chooses the smallest
registered K satisfying the existing cost rule; every decision is stored as
an event, and every correction row records task, K, rho and duration.

B50 Inc10 uses PyCIL's native first-task 200 epochs and subsequent-task 170
epochs.  The pinned PyCIL documentation/resource set does not publish a
numeric iCaRL B50 Inc10 reference, so `reproduction_gate.json` records
`reference: none` rather than borrowing the B0 Inc10 number.  Only baseline
`icarl` jobs write that gate.

Every parallel job directory contains `summary.json`, `tasks.json`,
`protocol.json`, and `vna_log.csv`; baseline iCaRL also contains
`reproduction_gate.json`.  A logical CBP B5 job preserves all five registered
replacement rates and stores their individual results below `rates/`, plus a
top-level aggregate manifest. The full-run CPU queue is configured for at
most four B5 processes, but remains blocked by the preflight hold.

## Notebooks and collection

Builder:

```sh
python scripts/build_kaggle_vna_parallel.py \
  --commit 2d4114b0bfa23e74847c23d15725515ac64397bc --variants
```

Generated files:

- `notebooks/kaggle_vna_parallel.ipynb`
- `notebooks/kaggle_vna_parallel_P1.ipynb` through `P6.ipynb`

All six populated notebooks are byte-equivalent after the first cell.  The
first cell contains only the selected `JOBS_GPU0`, `JOBS_GPU1`, `JOBS_CPU`,
shared `RUN_TAG`, and common immutable session settings. Direct inspection
of all six first cells confirms the runtime pin above; Python cells parse
successfully, and all cells after the first are identical across P1–P6.

They run pytest once, then real-data two-task GPU preflight on T4 x2, archive
the results, and stop. Full CPU/GPU training remains blocked even though
`ALLOW_OVER_BUDGET=True` is retained in the configuration. Review real T4
timings, feasibility, matched schedules and baseline provenance before
deciding whether to enable the P1–P6 full-run queues.

Results are merged with:

```sh
python scripts/collect_parallel.py ARCHIVE_P1 ... ARCHIVE_P6 \
  --run-tag vna_parallel_20261017
```

The collector reads directories, zip files or tar archives and prints the
Inc10, Inc5, B50 Inc10, B5 paired, and VNA cost tables requested in the task.

## Validation

## Training solver

The training-time B3/B5 VNA and NGD corrections now use matrix-free CG
(`tol=1e-3`, `max_iter=50`) with per-arm/site warm starts. Exact output-space
solves remain in the Stage-B measurement path. A selection matrix is reused
for nine applications and refreshed at task start and every tenth applied
step; `vna_log.csv` records `A_refreshed` for every correction.

Preflight schedules use the extended K set `{50,100,200,400,800}` plus
`{1,2,5,10} * iterations_per_epoch`, and the weighted cost
`0.1*t_with_A + 0.9*t_without_A`. Expected schedule fields include
`K`, `budget_exceeded`, `iterations_per_epoch`, `t_vna_with_A`,
`t_vna_without_A`, `weighted_vna_seconds`, `projected_vna_seconds`,
`iteration_seconds`, `refresh_weight`, `reuse_weight`, `smoke_points`, and
the budget rule. An infeasible budget selects the largest K and continues.

- Notebook Python syntax, full commit pin, P1-P6 first-cell-only difference,
  two GPU lanes, four CPU workers and shared full pytest cell are tested.
- Numerical selection tests verify the top eigendirection of N N^T.
- CLI and collector artifact contracts are covered by local tests.
- Affected tests: 48 passed, 0 failed, 0 skipped in 317.396 seconds;
  JUnit log: `parallel_logs/pytest_affected.xml`.
- Full suite: 296 passed, 0 failed, 1 optional skip (297 collected) in
  1542.901 seconds;
  JUnit log: `parallel_logs/pytest_full.xml`.

All local tests use synthetic data/CPU.  They validate code paths and
identities, not CIFAR-100 or Dohare performance.

An additional synthetic `icarl_vna --auto-schedule` smoke reached inline
calibration and exercised the budget-exceeded schedule path. Synthetic timing
does not authorize a real schedule; T4/CIFAR calibration remains required.
