# Stage A/B implementation

## Stage B core

This implements `TASK_STAGE_B_CORE.md` shared infrastructure only. It adds no
setting-specific runner, applies no VNA direction to baseline training, and
makes no commit. Stage A continues using `GGNConfig(damping_norm='outputs')`.
Review round 2 updates solver defaults and resume fingerprints as recorded below.

### Loss interface and normalization

`methods/vna_losses.py` defines `LossSpec`, `CE`, `MSE`,
`KDRepDistiller(alpha=.1, beta=.9, T=4)` and
`ICaRLPyCIL(T=2, known_classes=...)`. For real measurements, supply
`ctx={'targets': targets, 'loss_fn': upstream_callable, ...}` where
`upstream_callable(z, ctx)` calls the original upstream loss object. Teacher
logits and old-class count are passed in the same context. Residuals are
obtained by autograd on this callable, evaluated in float64 on a detached
output copy. No baseline gradients accumulate.

Both residual rows and metric actions contain **one batch factor 1/n**:

| Loss | Residual | H action |
|---|---|---|
| CE | `(p-one_hot(y))/n` | `(diag(p)-p p.T)v/n` |
| MSE, batchmean | `2(z-y)/n` | `2v/n` |
| MSE, element mean | `2(z-y)/(n*d)` | `2v/(n*d)` |
| RepDistiller KD | `.1*g_CE + .9*T*(p_student_T-p_teacher_T)/n` | `[.1*Cov(p)+.9*Cov(p_student_T)]v/n` |
| PyCIL iCaRL | `g_CE + (q_student_T-q_teacher_T)/(T*n)` on old classes | `[Cov(p)+Cov(q_student_T)/T² on old block]v/n` |

First-task iCaRL has CE only. PyCIL's KD is cross entropy with soft teacher
probabilities, with no T²; its scalar differs from KL by the teacher entropy.
MSE uses the supplied callable's actual quadratic Hessian action when present,
so an upstream element-mean reduction cannot silently receive a batchmean H.
Built-in defaults and copied reference formulas are for tests and compatibility;
real integrations pass their original loss through `loss_fn`.

`GGNProducts`, exact/lite/NGD solvers, filtered cotangents, joint correction,
Euclidean block/lite corrections, selection and residual-oriented diagnostic
helpers accept the generic loss. `tiny_V` also accepts one context per batch.
GGN H is the true output-loss Hessian, never empirical Fisher. Default/explicit
CE with no context preserves the original float64 softmax-minus-onehot rounding
after autograd evaluation; supplied upstream callables use their autograd
residual directly. A bitwise test locks the legacy CE residual, H and gradient.

### Declaring a site

```python
site = HookSite(model, 'block.conv1', 'block.conv2', 'conv',
                between=('block.bn2', 'block.relu2'))
# MLP: HookSite(model, 'fc1', 'fc2', 'linear', between=('relu',))
```

`between` lists native module paths in actual execution order. For WRN this
means conv1 -> BN -> ReLU -> conv2, after the block's initial BN/ReLU. The
original normalization executes normally. On new virtual coordinates norms
are identity, matching pinned Gromo's extension convention: new units do not
inherit old-channel affine parameters or running statistics. Incoming A uses
the first layer's geometry; external B uses the second layer's geometry and
starts at zero, including its bias if present. B's output is added by a
forward hook to the native second-layer output, before its downstream norm,
activation and residual addition. Grouped convolutions are rejected explicitly.

O defaults to the first/second layers and trainable between modules. Native
Gromo declarations reuse its full residual-path scope, including both norms
and excluding shortcuts. `o_params` can declare an explicit scope.
`HookProducts` supplies the existing J/J_E, adjoint, kernel and correction
contracts. Hooks exist only during a forward call; virtual tensors are never
registered on the original model. Native Gromo cached suffixes remain available;
generic branching graphs replay their full native forward for exact suffix
propagation. All contexts restore modes/buffers and assert unchanged parameter
count and state, including exceptional exits.

Native Gromo HookSites reuse its TINY statistics/solve. Generic linear/conv
sites form input-patch covariance and output-cotangent cross statistics, with
optional projection off existing second-layer features, then use a whitened
SVD. All selected maps pass the same normalization/sign-selection boundaries.
`tiny_U_from_cotangent`, `tiny_V`, `normalize_A`, block products/correction and
single/multiple-site joint correction work with HookSites. Gromo remains the
Phase 0/Stage A resolver.

### Alignment and measurement data

`diagnostics/vna_align.py::MeasurementHook(measure, seed=..., purpose=...)`
accepts `(model, batch_the_loop_already_has, loss_spec, ctx)`, calling
`measure(..., generator=dedicated_stream)`. It saves/restores original parameter
and buffer registrations/objects/values, gradients, module graph, each module's
train/eval mode, Python/NumPy and global CPU/CUDA RNG state, even on exceptions.
It asserts the parameter count at each measurement. Measurement code uses the
provided dedicated stream; it never receives or iterates a dataloader.

`assert_bitwise_alignment(run_fn, steps, seed)` calls the user's unchanged
CPU loop twice with `with_hook=False/True`, identical seeds and deterministic
algorithms. `run_fn` returns `(model.state_dict(), optimizer.state_dict())`.
The helper compares tensor bytes (including optimizer state), and restores
caller RNG/determinism settings. A toy Adam/dropout/BN loop passes; a raw hook
that draws from global RNG fails. `MeasurementBuffer(m=64)` clones/detaches
only batches already supplied by the loop and retains their last m samples.

### Upstream sources

Pins, URLs, declared licenses, intended imports and fetch status are recorded
in `third_party/SOURCES.md`. `scripts/fetch_third_party.sh` is strict and
idempotent: clone, obtain/check the exact commit, detached checkout, then check
and apply each sorted patch (already-applied patches are recognized). All three
new clones are gitignored. **No upstream code was modified and no patches were
created.** All three clones were fetched and license-verified on 2026-10-15
at the listed pins. Tests verify pins/licenses, clean patch application,
native imports/entrypoint help and network one-step smoke, and loss equality
against imported PyCIL `_KD_loss` and RepDistiller `DistillKL`.
`REQUIRE_UPSTREAM=1` makes missing upstream clones fail instead of skip.
Install `requirements-upstream.txt`: torch, torchvision, numpy, scipy, Pillow,
tqdm, tensorboard_logger, protobuf and six. RepDistiller imports require
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`; notebook environments and
entrypoint test subprocesses set it. Future RepDistiller runners must set it
before importing tensorboard_logger.

The Stage B plan covers B1–B5 only. The supervisor dropped B6 from this round.
The source/license block remains documented in `third_party/SOURCES.md`.


### Original Stage B core verification (before review round 2)

Final command: `.venv/bin/python -m pytest -rA`.
**174 passed, 11 skipped, 0 failures; 185 collected; 126.22 seconds.**
Ten skips are upstream checks blocked by absent clones; the remaining skip is
the existing optional Matplotlib diagnostics test. All existing VNA/Stage A
cases pass. `git diff --check` and `bash -n scripts/fetch_third_party.sh` pass.
No checkpoint measurements or setting-specific training results are claimed.

Coverage includes float64 analytic gradients and
finite-difference Hessians for all four losses; MSE reduction plumbing; all
eight Phase 0 ResNet-18 blocks' K, K_E, V, rho and d equivalence (1e-8 relative);
MLP/CNN and WRN pre-activation adjoints/finite differences; LoP 5-unit regression
and 3x2000 MLP shapes; B=0 output, modes, buffers, module graph, parameter-count
and RNG invariance; multi-site kernel addition and loss-generic joint equality;
positive/negative bitwise alignment and exception restoration; and the complete
existing test suite. Native Phase 0 equivalence uses four shared CG iterations
to bound CPU cost; it compares the same computed direction, without asserting
solver convergence as a training/measurement result.

### Created/modified files

Created:

- `methods/vna_losses.py`
- `methods/vna_sites.py`
- `diagnostics/vna_align.py`
- `scripts/fetch_third_party.sh`
- `third_party/SOURCES.md`
- `tests/test_vna_stage_b_losses.py`
- `tests/test_vna_stage_b_sites.py`
- `tests/test_vna_stage_b_align.py`
- `tests/test_vna_stage_b_upstream.py`
- `docs/vna/STAGE_AB_IMPLEMENTATION.md`

Modified:

- `.gitignore`
- `methods/vna.py`
- `methods/vna_ggn.py`
- `methods/vna_joint.py`
- `methods/vna_select.py`
- `diagnostics/signal_curve.py`
- `diagnostics/stage_a.py`

The pre-existing `.DS_Store` change was left untouched. Temporary test logs are
removed after verification; test scratch artifacts use pytest-managed temp
paths, and upstream subprocess fixtures use automatic temporary-directory cleanup.

## Review round 2

Implemented supervisor-approved decisions from `TASK_STAGE_B_R2.md`, dated
2026-10-15, without committing.

- Outputs damping is now the `GGNConfig` and training/solver-validation default,
  including K-FAC and both factor/selection refreshes. Explicit `params` remains
  available through `--damping-norm params` and configuration objects. The
  supervisor confirmed `mu = mu_rel * trace(F) / n_out` on 2026-10-15.
- Generic CG normalizes RHS and warm starts by default and rescales solutions,
  absolute residuals and history; relative residuals stay unchanged. The
  Phase-0 wrapper delegates, joint correction and functional projection use
  generic CG directly. Regression checks cover tiny RHS via the default path,
  original residual units and existing-scale/warm-start agreement within
  1e-12 relative. Historical audit tests explicitly select their old settings.
- Training protocol is format 3, including damping_norm and cg_normalize_rhs.
  Generated run-config hashes also include solver_protocol_version=3. Old
  resumes fail with named protocol differences. Stage A completion schema is
  stage1d-v2 and signal-curve identity is stage1c-v2, preventing stale numerical
  outputs from being resumed across this change.
- A5 measures epochs {90, 100, 110, 160, 200}; only its old-path columns disable
  RHS normalization. The Stage A and signal-curve notebook artifacts were
  regenerated. All Kaggle VNA builders install requirements-upstream.txt,
  fetch pinned sources before tests, set REQUIRE_UPSTREAM=1 and the protobuf
  Python implementation, and include upstream tests.
- Actual pinned upstream imports/help, native LoP predict/optimizer steps,
  clean patch checks and imported `_KD_loss`/`DistillKL` equality run successfully.
  PyCIL uses iCaRL; RepDistiller subprocesses use the required protobuf env var.
  All three upstream licenses were verified on 2026-10-15. B6 was dropped from
  this round by the supervisor; the blocked-license source note is retained.

The local NumPy/PyTorch bridge was repaired offline using cached NumPy 1.26.4
and SciPy 1.16.3 wheels. requirements-upstream.txt bounds NumPy below 2 and
SciPy below 1.18 to keep that pairing compatible with the installed PyTorch 2.2.2.
No upstream source edits or patches were needed.

### Verification

Full suite command:
`REQUIRE_UPSTREAM=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -ra`.
**187 passed, 1 skipped, 0 failures; 188 collected; 229.41 seconds.**
The sole skip is the pre-existing
`tests/test_vna_split3.py::test_render_comparison_curves` (optional Matplotlib
unavailable). There were no upstream skips and no CUDA skips in this run.

The explicitly required targeted command
`REQUIRE_UPSTREAM=1 .venv/bin/python -m pytest tests/test_vna_stage_b_upstream.py tests/test_vna_stage_b_losses.py -ra`
also passed: **19 passed, zero skips**. All these cases pass in the final full suite.
`git diff --check`, `bash -n scripts/fetch_third_party.sh`, and compilation of
Stage A, signal-curve and all six training notebook variants pass. Notebook setup
checks verify dependency installation, pinned-source fetch, required upstream
mode and protobuf environment before tests. No actual checkpoint measurements
or new training results are claimed.

### Changed files

- `diagnostics/signal_curve.py`
- `diagnostics/stage_a.py`
- `diagnostics/vna_solver_validation.py`
- `docs/vna/STAGE2_REPORT.md`
- `docs/vna/STAGE_AB_IMPLEMENTATION.md`
- `docs/vna/STAGE_A_REPORT.md`
- `experiments/run_vna_training.py`
- `experiments/vna_training_config.py`
- `experiments/vna_training_support.py`
- `methods/vna.py`
- `methods/vna_ggn.py`
- `methods/vna_joint.py`
- `notebooks/kaggle_vna_signal_curve.ipynb`
- `notebooks/kaggle_vna_stage_a.ipynb`
- `projection/cg.py`
- `projection/functional.py`
- `requirements-upstream.txt`
- `scripts/build_kaggle_vna_signal_curve_notebook.py`
- `scripts/build_kaggle_vna_stage_a_notebook.py`
- `scripts/build_kaggle_vna_training_notebooks.py`
- `tests/test_cg.py`
- `tests/test_vna_stage_a.py`
- `tests/test_vna_stage_b_losses.py`
- `tests/test_vna_stage_b_upstream.py`
- `tests/test_vna_training.py`
- `third_party/SOURCES.md`

The pre-existing `.DS_Store` change and user task files (`TASK_STAGE_B_R2.md`,
`TASK_STAGE_B5.md`) were left untouched. `.git` was not written; no commit was made.

## Stage B5

**Status:** Stage B5 implementation is complete, including regression measurements
under the supervisor's §6 N2 decision. Full benchmark baseline results require
the Kaggle runs; local synthetic/smoke outputs remain explicitly invalid.

N2 centering is a property of `LossSpec`: CE, RepDistiller KD and PyCIL iCaRL
have `residual_is_zero_sum=True`; MSE has False. N2 samples Gaussian rows,
centers only the softmax losses, and rescales to the real per-sample norm.
Scalar MSE therefore uses random sign times the absolute residual, including
exactly zero for zero residuals. Measurements retain eight N1 and eight N2
values, and rows record `null_kind="N2"` and `n2_centered`. Stage A retains its
existing centered default. Tests cover all four loss properties, vector/scalar
norm preservation, CE zero sums, and the complete scalar regression pipeline.

### Upstream hooks and data

- `adapters/lop_hooks.py` wraps the class-level `learn` methods of upstream
  `Backprop`, `ContinualBackprop`, and `ConvCBP`, calling the original method
  first and restoring all three methods on context exit, including exceptions.
  `experiments/stage_b5.py` imports and invokes `expr`, `online_expr`, and
  `repeat_expr` unchanged, with each arm's run directory as cwd. There are no
  upstream source edits or patches.
- The schedules are regression `flip_after`, MNIST `change_after / batch_size`,
  and ImageNet `examples_per_epoch * num_showings / batch_size`. Registered
  tasks {1,2,5,10,20,50,100,150,200} are measured after the first step and at the
  task midpoint. Buffer capacity is 64 for regression/MNIST and 100 for
  ImageNet. Actual available sample count is also logged; the first online
  step cannot provide 64 already-seen samples. Only learner inputs and targets
  enter this buffer, never future or test data.
- `Prediction` forwards directly to native `predict`; it introduces no model
  parameters. Sites cover each adjacent Linear/activation/Linear pair. CNN
  sites cover conv1/conv2, conv2/conv3, conv3/fc1 (including pooling and native
  flattening), fc1/fc2, and fc2/fc3. The pooled TINY proposal statistics use
  the channel-averaged *native max-pool switches* as their linearization;
  virtual features, sign scores, JE, and counterfactual gains use actual ReLU
  and max pooling. No upstream activation or pooling operation is replaced.
- The existing `MeasurementHook` already isolates Python, NumPy, torch CPU and
  CUDA RNG and restores model values, registrations, modes, buffers and grads.
  Every point asserts unchanged parameter count. CE and MSE residuals call the
  learner's actual loss object. MSE targets are cast to the diagnostic output
  dtype to support its float64 autograd reduction.
- Real regression data is generated by upstream `generate_problem_data`.
  Real MNIST uses upstream `load_mnist.mnist`, or an attached pickle in its
  exact format. Fake fixtures preserve native input shapes and the MNIST 60k
  sample permutation. ImageNet uses per-class `.npy` files and upstream class
  order; missing data writes `skipped: data unavailable` and continues.

### Pre-registered recipes (§5)

| Benchmark | BP step size | BP cfg | CBP cfg / rates |
| --- | --- | --- | --- |
| Regression | 0.01 | `slowly_changing_regression/cfg/sgd/bp/relu.json` | `cfg/sgd/cbp/relu.json`: all {1e-6,1e-5,1e-4,1e-3,1e-2} |
| Permuted MNIST | 0.003 | `permuted_mnist/cfg/bp/std_net.json` | `cfg/cbp.json`: 1e-4 |
| ImageNet-32 | 0.01, momentum 0.9 | `imagenet/cfg/bp.json` | `cfg/cbp.json`: 3e-4 |

BP step-size sources are the corresponding CBP cfgs, as specified by the new
§5. All other training values retain their upstream cfg meaning. Every CSV row
records the cfg path, learning-rate source, chosen rate/momentum, effective
maturity, complete runtime cfg, schema `stage1d-v1`, benchmark/arm/task/step,
commit, seed, device and timing. Measurements run only on BP. Regression
baseline selection uses the CBP arm with minimum mean online error across the
whole run; all five registered CBP utility criteria are also exposed for GIR.

### Measurements, gates and resume

`diagnostics/lop_measure.py` calls the existing VNA selection/GGN/joint code.
K and filtered cotangents use the whole original network. Each rank/µ pair
gets one joint correction solve across all eligible sites. Per-site rows
include U, V, eta, rho, rho_joint, lambda1, eight N1/N2 values and empirical
p-values, tiny_U, tiny_V/random_frob, normalization, split-half subspace
similarity, CG diagnostics, dormant fraction (tau=0.1), activation effective
rank, all layer weight magnitudes, and exact output-space GGN eigenvalues and
residual energy below µ. Regression uses ranks {1,2,4}; other sites use
{4,16,64} capped by their width. µ includes the CLI Stage-A default and 1e-2.
GIR uses imported GnT/ConvGnT utility/maturity and ReDo unit masks. Its model
copies place incoming rows at the selected units, scale to the recorded
initial row norm, and zero the corresponding outgoing columns. It reports
before/after eta and tiny/random swap gain; the training model is untouched.

`b5_baseline_gate.json` records early (tasks 1–10) and late (last ten) online
means, a fixed CBP tolerance of 0.02, and the selected regression CBP arm.
Fewer than 20 tasks, fake/smoke data, or a failed baseline cannot validate
measurements. The existing gate thresholds are unchanged; B5 uses its
second-half-task window and train-accuracy n/a. Regression's gate rank is four,
its registered maximum, since rank sixteen cannot exist at width five. Gate
entries remain INVALID until the baseline passes. Independent GPU workers
serialize JSON merges with file locks.

At task boundaries, resume stores model, optimizer, CBP/shadow state, original
log arrays, buffer, dedicated measurement generator states, and all global RNG
states. The unchanged loop replays the prefix without optimizer updates and
then restores its saved boundary. Packed replay stores precisely the loss and
softmax-argmax consumed by the native loops rather than millions of Python
objects/full logits. CPU tests compare final parameters, optimizer and CBP
internal state, original returned logs, measurement outputs and data streams.

### Notebook and discrepancies

`scripts/build_kaggle_vna_stage_b_nb1.py --commit <full hash>` builds
`notebooks/kaggle_vna_stage_b_nb1.ipynb`, with no A or B6 runs. It installs
upstream requirements, fetches pinned third-party sources, runs the entire test
suite with REQUIRE_UPSTREAM and the Python protobuf implementation, and
schedules ImageNet BP/CBP on GPU0, then regression BP/all five CBPs and MNIST
BP/CBP on GPU1. Native regression remains on CPU because `expr` hardcodes its
CBP device; this is recorded accurately in outputs. Two-task smokes separate
training time per step from diagnostic time per registered point for the
GPU-hour projection and enforce GPU_HOURS_BUDGET. Archives run hourly and at
completion, retain the existing 5,000,000,000-byte peak cap, serialize archive
writers, and exclude regenerable data and in-progress upstream files. Stage-B
archives restore task checkpoints safely. The last cell prints baseline/G
results, loudly flags invalid measurements, plots rho versus task, and shows
GIR ratios.

Observed upstream details preserved:

- Regression CBP cfg supplies `maturity_threshold=100`, but `expr` only reads
  `mt`, so its actual default is **10**. The runner records 10 and does not fix
  or reinterpret the upstream training recipe.
- The native regression generator writes an extra flip interval and only
  computes complete 10k chunks. A real-data tiny smoke generates at least 10k
  points via that same function and consumes only its declared first 100.
- MNIST's rank-log arrays have zero rows for sequences shorter than 60k. CPU
  fixtures and smoke pad only those zero-sized diagnostic allocations to one
  row; samples, permutations, models and optimizer steps stay upstream. Full
  runs do not use this allocation fixture.
- Fake ImageNet smoke sets native data-size constants to 100 train and 50 test
  images/class: two native 100-example batches per task. Constants are restored
  on exit; the training/evaluation loops are unchanged.
- No full baseline or GPU measurements have been claimed from local fake data.
  No network was used. The notebook is pinned to the existing HEAD; because
  the supervisor requested **no commit**, this pin cannot contain the new
  uncommitted implementation. Regenerate with its eventual published commit
  before running on Kaggle.

### Verification and changed files

Verification is recorded below after the required full suite completes.
Changed files for this stage:

- `adapters/__init__.py`, `adapters/lop_hooks.py`
- `diagnostics/lop_measure.py`, `diagnostics/vna_align.py`, `diagnostics/vna_gate.py`,
  `diagnostics/stage_a.py`
- `experiments/stage_b5.py`, `experiments/vna_kaggle.py`
- `methods/vna_sites.py`, `methods/vna_losses.py`, `pyproject.toml`
- `scripts/build_kaggle_vna_stage_b_nb1.py`
- `notebooks/kaggle_vna_stage_b_nb1.ipynb`
- `tests/test_vna_stage_b5.py`
- `docs/vna/STAGE_AB_IMPLEMENTATION.md`

The pre-existing `.DS_Store` and supervisor-edited `TASK_STAGE_B5.md` remain
untouched. No commit was made.

Relevant iteration tests: **62 passed**, covering B5, Stage-B losses, and
Stage A null behavior. Offline regression BP CLI smoke completed 100 steps
and wrote 12 measurement rows with eight uncentered N2 nulls each, while
correctly flagging the synthetic baseline INVALID. Notebook cells compile,
`compileall` and `git diff --check` pass; the pinned LoP checkout is clean.

Final required suite (run **once** after implementation and relevant checks):
`REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -ra`
— **215 passed, 1 skipped, 0 failures; 216 collected; 458.21 seconds**.
The sole skip is the pre-existing optional diagnostics dependency check in
`tests/test_vna_split3.py::test_render_comparison_curves`; there are no upstream
skips. The suite includes 28 B5 tests. All changes remain uncommitted.

Continuation checks include real scalar-regression measurements at all four
points of two native tasks, all three registered ranks and five GIR CBP
criteria. Hook-on/off final parameters, optimizer state and upstream logs
match bit-for-bit; task-boundary resume compares all numerical measurement
rows, excluding wall-clock timing. The notebook setup no longer refers to
Phase-0's undefined CIFAR `DATA_ROOT`; regenerated cells compile and the B5
notebook test rejects CIFAR setup references.
