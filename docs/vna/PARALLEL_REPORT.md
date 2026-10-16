# Parallel CIL/B5 pilot implementation — 2026-10-17

Implemented from branch `vna-pilot-gir-vna`, base commit
`018789b422f271021384d2254b0ff1c68ccf4e7b`.  The generated notebooks are
pinned to implementation commit
`11886859f98ee596b27b72e4f8d603d6c60fd849` and can fetch the implemented code.

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
top-level aggregate manifest.  At most four B5 processes run concurrently.

## Notebooks and collection

Builder:

```sh
python scripts/build_kaggle_vna_parallel.py \
  --commit 11886859f98ee596b27b72e4f8d603d6c60fd849 --variants
```

Generated files:

- `notebooks/kaggle_vna_parallel.ipynb`
- `notebooks/kaggle_vna_parallel_P1.ipynb` through `P6.ipynb`

All six populated notebooks are byte-equivalent after the first cell.  The
first cell contains only the selected `JOBS_GPU0`, `JOBS_GPU1`, `JOBS_CPU`,
shared `RUN_TAG`, and common immutable session settings.  They run pytest
once, then start full jobs directly with resume, hourly archive, heartbeat,
`ALLOW_OVER_BUDGET=True`, two GPU lanes and a four-worker CPU queue.  The P6
queue naturally runs four jobs followed by the remaining two.

Results are merged with:

```sh
python scripts/collect_parallel.py ARCHIVE_P1 ... ARCHIVE_P6 \
  --run-tag vna_parallel_20261017
```

The collector reads directories, zip files or tar archives and prints the
Inc10, Inc5, B50 Inc10, B5 paired, and VNA cost tables requested in the task.

## Validation

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
calibration and stopped with `no registered K meets the correction budget`.
This is the preregistered guard (synthetic timing must not authorize a real
schedule), not a performance result; feasibility must be decided on T4/CIFAR.
