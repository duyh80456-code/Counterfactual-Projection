# Standard CL completion report — 2026-10-28

Starting local/remote HEAD: `6bf844ee51642f72a0eeac8256f923231f4cdfbf`.
Branch: `codex/gir-live-task1`; remote HEAD was verified again before commit.
Final implementation commit: `7aa40e8292d05341b5f71e9216e9e08beb6783d9`.
The final delivery commit adds the recorded implementation SHA to this report;
its exact SHA is the latest commit touching this report (`git log -1 --format=%H
-- docs/vna/STANDARD_CL_COMPLETION_REPORT.md`) and is supplied with the delivery.

## Scope and completion matrix

PASS denotes implemented and executed software within the CPU/synthetic scope
below. No benchmark or scientific hypothesis is represented as passed.

| Component | Implemented | CPU tested | End-to-end tested | GPU tested | Benchmark completed |
|---|---|---|---|---|---|
| H0 | PASS | PASS | PASS | NOT RUN | NOT RUN |
| H1 | PASS | PASS | PASS | NOT RUN | NOT RUN |
| H2 | PASS | PASS | PASS | NOT RUN | NOT RUN |
| H3 | PASS | PASS | PASS | NOT RUN | NOT RUN |
| H4 | PASS | PASS | PASS | NOT RUN | NOT RUN |

* H0: fixed-checkpoint/window CLI, shared damping/norm cap, unconstrained and
  constrained native/virtual controls, multiple random draws, distinct GN and
  unfiltered standard-BN/cloned-buffer optimizer references. The synthetic CLI
  produced nine rows: five accepted corrections, two rejected corrections and
  two raw references. These numbers are execution checks, not method evidence.
* H1: full native PyCIL CIL integration, explicit pooled/separate-mean reductions,
  iCaRL KD/reference variants, certified QP, finite physical acceptance, BN/SGD
  intervention and independently configured unified mode, native herding/NME,
  task/trajectory metrics, exact epoch rollback/resume and failure artifacts.
  Two-task synthetic tests use the real PyCIL orchestration and a tiny BN fixture.
  A separate synthetic-image smoke initializes the actual full ResNet32 BN,
  performs native SGD and a tiny_V correction at its final eligible site.
* H2: independent executable CIL branch, fixed feature-layer configuration,
  JVP/VJP feature penalty in both solves, consistent cotangent, finite acceptance,
  train/eval feature drift and linearization error, no persistent feature hooks,
  independent configs/modes and native/random/GN controls with shared acceptance.
  Lambda is a fixed penalized variant, not a calibrated drift-budget multiplier.
* H3: actual existing tiny_V candidate generation, all eligible live/dead slots
  plus identity, serial fixed-capacity evaluation, measured functional/loss jump,
  net post-removal GN recovery objective and separate VNA compatibility diagnostic,
  exact validated-state commit, optimizer/BN resets and interrupted-swap rollback.
  Independent LoP regression runner records online learning curves, task learning
  speed/slope, plasticity decay, gradient RMS, replacement counts and costs.
  Native BP/CBP, supervised GnTredo hook and native S&P execute in synthetic tests;
  their optimizer/RNG and CBP/ReDo auxiliary-state resume paths are verified.
* H4: deterministic recorded rotations/permutations, fixed ten-label classifier,
  reservoir, disjoint train/test splits, evaluation of all seen domains, forgetting
  and BWT, exact finite BN weight/buffer drift diagnostics, configs and epoch resume.
  Integration uses two short synthetic domains; no real MNIST performance is claimed.

Historical M1/P1 runners, notebooks, policies, baseline files and checkpoint
formats are unchanged. There is no automatic conversion of historical noBN
checkpoints into new BN baselines.

## Actual execution and logs

CPU environment: macOS, Python3.12.7, torch2.2.2, torchvision0.17.2,
NumPy1.26.4, SciPy1.16.3; `torch.cuda.is_available() == False`.
Pinned PyCIL and LoP checkouts were available in the original workspace and
linked locally for verification; their Git worktrees were clean. Those local
links are not part of the delivered source. No dataset was downloaded.

Every pytest command used:

```bash
REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest ... -o addopts='' -q -p no:cacheprovider
```

The first reproduction/expanded attempts omitted the thread-limit variables;
the full suite and final checks used them. The actual interpreter was
`/Users/pxokq/expanding-guide model/.venv/bin/python` while cwd was this worktree.

| Command scope | Actual result | Saved log |
|---|---|---|
| Original math/pipeline/audit files, freshly reproduced | 48 passed, 36 warnings, 11.03s | standard_cl/CPU_COMPLETION_REPRODUCE48.log |
| First new completion-file attempt | 3 failed, 8 passed; fixed | standard_cl/CPU_COMPLETION_INITIAL_FAILURES.log |
| Second completion-file attempt | 1 failed, 17 passed; fixed | standard_cl/CPU_COMPLETION_SECOND_FAILURES.log |
| Initial plasticity subset after dtype fix | 5 passed, 13 deselected, 6.86s | standard_cl/CPU_COMPLETION_PLASTICITY.log |
| Expanded math/pipeline/audit/completion snapshot | 68 passed, 36 warnings, 244.87s | standard_cl/CPU_COMPLETION_EXPANDED.log |
| Entire repository snapshot, once | 384 passed, 53 warnings, 407.00s; no skips | standard_cl/CPU_COMPLETION_FULL_SUITE.log |
| Four standard-CL files after mode/config fixes | 75 passed, 36 warnings, 183.12s | standard_cl/CPU_COMPLETION_FOCUSED.log |
| Auxiliary-state/parallel-gradient/collector additions | 6 passed, 20 deselected, 11.01s | standard_cl/CPU_COMPLETION_COLLECTOR_RESUME.log |
| Final H2/H3/RNG/collector selection | 17 passed, 26 deselected, 65.59s | standard_cl/CPU_COMPLETION_H2_H3.log |
| Final strong-baseline/unified selection | 4 passed, 24 deselected, 39.96s | standard_cl/CPU_COMPLETION_BASELINES.log |
| H2 lambda0 physical/optimizer/RNG bitwise identity | 1 passed, 28 deselected, 40.38s | standard_cl/CPU_COMPLETION_LAMBDA0.log |

Do not add these counts: they overlap and correspond to successive revisions.
The full-suite snapshot preceded the last focused additions and logging/RNG
fixes; the affected portions were rerun after those changes. No second full
repository run is claimed. The final standard-CL files contain 77 test cases;
they were covered collectively by the listed focused runs, not by a claimed
single fresh 77-test run. Warnings are preserved in the logs (upstream Python
AST deprecations and legacy NumPy empty-slice fixture warnings).

Exact scopes for the full and final focused runs:

```bash
.venv/bin/python -m pytest -o addopts='' -q -p no:cacheprovider
.venv/bin/python -m pytest tests/test_standard_cl_math.py tests/test_standard_cl_pipeline.py tests/test_standard_cl_audit.py tests/test_standard_cl_completion.py -o addopts='' -q -p no:cacheprovider
.venv/bin/python -m pytest tests/test_standard_cl_audit.py tests/test_standard_cl_completion.py -k 'h2 or H2 or data_generation or plasticity or collector or h3' -o addopts='' -q -p no:cacheprovider
.venv/bin/python -m pytest tests/test_standard_cl_completion.py -k 'strong_controls or agem_projection or unified' -o addopts='' -q -p no:cacheprovider
.venv/bin/python -m pytest tests/test_standard_cl_completion.py -k zero_penalty -o addopts='' -q -p no:cacheprovider
```

Standalone H0 execution:

```bash
.venv/bin/python -m diagnostics.cl_mechanism --fake-data --random-replicates 2 --out /tmp/standard-cl-h0-smoke.json
```

Output is preserved as `standard_cl/H0_COMPLETION_SMOKE.json`. Tests/smoke ran
on the uncommitted completion worktree over the starting SHA; that starting SHA
alone is not falsely represented as containing the new code. Final source is
recorded by the implementation commit above.

The integration failures exposed and fixed: an unpickleable torch.Generator
in learner snapshots (persist its tensor state), a BN measurement between
forward/backward invalidating parameter versions (move it before forward), and
an H3 mixed-precision dot product (explicit double promotion). Final auditing
also isolated environment-generation RNG from learner initialization, included
feature diagnostics in H2 total cost/JVP/VJP accounting, recorded ER-ACE's actual
summed objective, and kept baseline nonfinite-loss coordinates explicit.

## Mathematical checks and boundaries

Executed checks cover direct/generalized VNA systems and noncommuting PSD
metrics, g^T d = z^T K_E z, component-gradient and direction linearity, unequal
2/3 incoming/replay means and autograd Hessians, native-versus-virtual
counterexamples, true CG residual failures, zero/opposing/rank-deficient/nearly
parallel QP cases, SciPy KKT oracle, float32/float64 paths, actual physical
acceptance and complete rejected-state rollback. H2 checks explicit feature
Jacobian versus matrix-free penalty, consistent penalized identity, small-step
drift and hook cleanup, plus lambda0 bitwise physical/optimizer/RNG identity.
H3 checks Schur added-column gain and the noncommuting VNA derivative against
finite differences, generated live/dead candidates, identity fallback and exact
state commit. H4 checks transform determinism, labels/head, memory, normalization
and evaluation/resume.

These are finite numerical tests, not universal floating-point proofs. Error
bounds assume the declared SPD/PSD operators and lower bounds. Virtual projection
remains in range(J); a PSD virtual kernel is not universally superior to native
or exact GN at the same surrogate optimum. A local accepted window step does
not prove retained test accuracy. Dead-on-window is not global deadness. GN gain
and VNA gain are distinct. Train-mode constraints do not protect eval BN effects.

## Commands, inputs and resume

Install instructions and complete comparison commands are in
[standard_cl/README.md](standard_cl/README.md). Required inputs are official
CIFAR Python `DATA/cifar-100-python/{train,test,meta}` and torchvision MNIST
`DATA/MNIST/raw/` train/test files. H3 generates its pinned native regression
stream into `OUT/data/regression.pkl`; no external checkpoint is needed.

```bash
.venv/bin/python -m experiments.standard_smoke
.venv/bin/python -m experiments.standard_cl --config configs/standard_cl/h1_cifar100_bn.json --data-dir DATA --tasks 3 --device cuda:0 --arm h1 --seed 0 --out outputs/h1/s0
.venv/bin/python -m experiments.standard_cl --config configs/standard_cl/h1_cifar100_bn_unified.json --data-dir DATA --tasks 3 --device cuda:0 --arm h1 --seed 0 --out outputs/h1_unified/s0
.venv/bin/python -m experiments.standard_h2 --data-dir DATA --tasks 3 --device cuda:0 --seed 0 --out outputs/h2/s0
.venv/bin/python -m experiments.standard_plasticity --arm h3 --seed 0 --out outputs/h3/s0
.venv/bin/python -m experiments.standard_domain --config configs/standard_cl/h4_rotated_mnist.json --data-dir DATA --device cuda:0 --seed 0 --out outputs/h4/s0
.venv/bin/python scripts/collect_standard_cl.py outputs/h1 outputs/h1_unified outputs/h2 outputs/h3 outputs/h4
```

These benchmark launch commands are **not executed**. Replace H1 arm with
baseline/native/random/gn/gradient/vna for controls; use a new directory per
arm. `--baseline er` is ER with herding; `--baseline agem/er_ace --arm baseline`
selects the declared adaptations. H3 arms bp/cbp/redo/snp use native adapters.
H4's permuted config changes the stream; `--arm baseline --weight-decay VALUE`
is the separately identified L2 control. Full CIL is `--tasks 20`, not an
unannounced truncation. GPU profiling uses the same commands and recorded
process CUDA high-water/phase timings; CUDA behavior itself remains untested.

Resume repeats the exact command/output directory with `latest.pt` present.
Protocol mismatches raise; no seed substitution, automatic clipping or silent
baseline reuse occurs. Exit75 restores a partial epoch or interrupted H3 native
step/swap and checkpoints the resumable state. FAILED_DIVERGED, resource and
other failures remain distinct in artifacts/collector; failed seeds are not
converted into finite benchmark values. Collector verifies per-run manifests,
completed horizons/metrics and paired data/transform identities, keeping source,
objective, normalization, integration-mode, budget and synthetic cohorts separate.

## Remaining limitations and unsupported comparisons

* CUDA/T4 compatibility, memory peaks, deterministic GPU replay and performance:
  NOT RUN. Real CIFAR-100/MNIST/LoP benchmark training: NOT RUN.
* Authentic LPR comparison: BLOCKED. Its separate pinned source/environment is
  not installed (Hydra/OmegaConf/Avalanche/wandb absent). H2 is not LPR.
* A-GEM/ER-ACE: implemented/tested **adaptations** with PyCIL task-boundary
  herding and expanding linear head; not official matched-protocol reproductions.
  Sequential BN forwards differ from pooled correction forwards. See
  [baseline source audit](standard_cl/BASELINES.md) for primary source revisions.
* CBP/S&P defaults are declared, not tuned best settings. The ReDo supervised
  hook calls original GnTredo on native features; it is labelled as a hook.
* Fixed H2 penalty is not a solved drift-budget calibration. Virtual GroupNorm
  and stochastic dropout corrections are unsupported; optional native GroupNorm
  controls are implemented but NOT RUN in this completion audit.
* Concurrent engines within one process are not validated. The benchmark setup
  uses separate processes; selector accounting temporarily wraps the shared
  product factory and always restores it. Large feature-layer windows may be
  expensive even with matrix-free products and serial candidate evaluation.
* No physical checkpoint from old M1 is required, reused or converted. This task
  does not resolve the old s1 epoch43 replay or validate historical GPU outcomes.

All hypotheses remain **UNVERIFIED**: virtual-kernel superiority, H1 stability
gap, H2 representation retention, H3 plasticity versus CBP/ReDo, H4 domain
accuracy and favorable GPU-hours. Code completion does not provide this evidence.

## Modified files

* `README.md`
* `configs/standard_cl/h1_cifar100_bn.json`
* `configs/standard_cl/h1_cifar100_bn_unified.json`
* `configs/standard_cl/h2_cifar100_bn.json`
* `configs/standard_cl/h2_cifar100_bn_unified.json`
* `configs/standard_cl/h3_regression.json`
* `configs/standard_cl/h4_permuted_mnist.json`
* `configs/standard_cl/h4_rotated_mnist.json`
* `diagnostics/cl_mechanism.py`
* `docs/vna/PREREGISTRATION_CHANGES.md`
* `docs/vna/STANDARD_CL_COMPLETION_REPORT.md`
* `docs/vna/standard_cl/AUDIT.md`
* `docs/vna/standard_cl/BASELINES.md`
* `docs/vna/standard_cl/CPU_AUDIT.md`
* `docs/vna/standard_cl/CPU_COMPLETION_BASELINES.log`
* `docs/vna/standard_cl/CPU_COMPLETION_FOCUSED.log`
* `docs/vna/standard_cl/CPU_COMPLETION_FULL_SUITE.log`
* `docs/vna/standard_cl/CPU_COMPLETION_H2_H3.log`
* `docs/vna/standard_cl/CPU_COMPLETION_INITIAL_FAILURES.log`
* `docs/vna/standard_cl/CPU_COMPLETION_LAMBDA0.log`
* `docs/vna/standard_cl/CPU_COMPLETION_REPRODUCE48.log`
* `docs/vna/standard_cl/CPU_COMPLETION_SECOND_FAILURES.log`
* `docs/vna/standard_cl/H0_COMPLETION_SMOKE.json`
* `docs/vna/standard_cl/MATHEMATICS.md`
* `docs/vna/standard_cl/PROTOCOL.md`
* `docs/vna/standard_cl/README.md`
* `docs/vna/standard_cl/STATUS.md`
* `experiments/standard_cl.py`
* `experiments/standard_domain.py`
* `experiments/standard_h2.py`
* `experiments/standard_plasticity.py`
* `experiments/standard_smoke.py`
* `methods/cl_baselines.py`
* `methods/cl_h2.py`
* `methods/cl_h3.py`
* `methods/cl_h4.py`
* `methods/cl_replay.py`
* `scripts/collect_standard_cl.py`
* `tests/test_standard_cl_completion.py`
* `docs/vna/standard_cl/CPU_COMPLETION_PLASTICITY.log`
* `docs/vna/standard_cl/CPU_COMPLETION_EXPANDED.log`
* `docs/vna/standard_cl/CPU_COMPLETION_COLLECTOR_RESUME.log`
