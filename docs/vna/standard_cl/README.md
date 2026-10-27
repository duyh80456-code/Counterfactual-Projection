# Standard CL VNA: code-only implementation, 2026-10-28

This is a separate research path. Historical methods, training runners,
notebooks and source pins are unchanged. No test, toy experiment, benchmark,
GPU profiling or scientific evaluation was executed for this delivery.
All research hypotheses remain **INCONCLUSIVE**. Written tests are not passed
tests; the pilot is not implementation-validated or ready for paper claims.

Read [audit](AUDIT.md), [mathematics](MATHEMATICS.md),
[protocol](PROTOCOL.md) and [status](STATUS.md) before launching anything.
The new source paths are:

| Responsibility | Implementation |
|---|---|
| JVP/VJP, loss Hessian, virtual features | existing `methods/vna.py`, `vna_ggn.py`, `vna_sites.py`, `vna_v3.training_products` |
| ProjectionOperator / ProjectionSolver | `methods/cl_projection.py` |
| Two-halfspace ConstraintSolver | `methods/cl_projection.py` |
| Model/optimizer/BN/RNG transactions | `methods/cl_state.py` |
| LossComponent, CandidateUpdate, AcceptanceCertificate, H1 | `methods/cl_replay.py` |
| H0 fixed-state controls | `diagnostics/cl_mechanism.py` |
| Independent H2 penalty | `methods/cl_h2.py` |
| Independent H3 finite-candidate search | `methods/cl_h3.py` |
| Independent H4 data/head/replay | `methods/cl_h4.py` |
| ContinualLearner / ExperimentLogger | `experiments/standard_cl.py` |
| H4 training | `experiments/standard_domain.py` |
| Strict result grouping | `scripts/collect_standard_cl.py` |

## Reproduction commands (instructions only, not executed)

Install the repository's existing dependencies and fetch its pinned upstreams
with `scripts/fetch_third_party.sh` before using PyCIL. The active worktree
does not contain a populated PyCIL checkout; a checkout in the original
workspace is not silently substituted. Use the official CIFAR-100 Python
files and unmodified PyCIL transforms. No data or checkpoints are downloaded
automatically by this new code.

After mathematical/integration validation, a paired BN pilot is expressed as:

```bash
.venv/bin/python -m experiments.standard_cl --data-dir /path/to/cifar100 \
  --config configs/standard_cl/h1_cifar100_bn.json --tasks 3 \
  --baseline icarl --arm baseline --seed 0 --out outputs/h1/icarl/baseline/s0
.venv/bin/python -m experiments.standard_cl --data-dir /path/to/cifar100 \
  --config configs/standard_cl/h1_cifar100_bn.json --tasks 3 \
  --baseline icarl --arm h1 --seed 0 --out outputs/h1/icarl/h1/s0
```

Repeat the same locked configuration for seed 1 and for `native`, `gn`,
`gradient`, `random`, `vna`. `native` means native-kernel candidate followed
by the same constrained projection. `gradient` is the gradient-candidate
constrained control; `gn` is the damped GN-candidate constrained control.
`vna` disables the halfspace projection and checks only total window loss.
ER is a separate `--baseline er` family using PyCIL herding, explicitly
labelled **ER with herding**, not a reservoir ER reproduction.

`--reference kd` is a separate iCaRL-style experiment: the first constraint
then covers all CE, the second only old-logit KD. It is rejected for ER.
This field is part of the protocol and cannot be merged with replay-CE+KD.

For unified updates, copy the configuration, explicitly set
`intervention_every=1`, and use `--mode unified`. Candidate = actual SGD
parameter step divided by LR + virtual correction; project that **whole**
candidate, then line-search the physical update. The staged optimizer state
is committed only on acceptance; a rejected/uncertified unified step is a
no-op. It is a distinct algorithm and table from intervention mode.

The later phases (20 tasks, B50 Inc10 and held-out seeds) require a new
registered config and successful pilot validation. They are not scheduled
or executed by this change. No full-scale job launches at import time.

H4 uses local torchvision MNIST files, with no automatic download:

```bash
.venv/bin/python -m experiments.standard_domain --data-dir /path/to/mnist \
  --kind rotated --arm h1 --seed 0 --out outputs/h4/rotated/h1/s0
.venv/bin/python -m experiments.standard_domain --data-dir /path/to/mnist \
  --kind permuted --arm baseline --seed 0 --out outputs/h4/permuted/baseline/s0
```

The optional GroupNorm control currently supports baseline/native/GN/gradient
candidates; virtual GroupNorm construction explicitly refuses, rather than
silently replacing it with a different virtual normalization. Frozen-BN is
a labelled loss diagnostic, not an unregistered training-mode change.

After runs finish, collect expanded directories with
`scripts/collect_standard_cl.py ROOT [ROOT ...]`. Different task protocols,
BN modes, integration modes, budgets, references and code commits remain
separate. Failed/OOM/diverged runs stay in the outcome table; only completed
runs supply finite metrics, and paired-seed counts are reported explicitly.

## Independent branches

H2's `GeometryCorrection.step(..., replay_inputs=..., geometry_weight=...)`
recomputes both cotangent and correction with the feature penalty. It is
not LPR and is not enabled by the H1 CLI. H3 accepts a finite set of in-place
`SwapCandidate`s and an optional scalar GN-surrogate evaluator, always with
identity available and componentwise measured loss checks. Its search is
only optimal over the supplied admissible finite list. A complete new
plasticity benchmark/candidate generator is not claimed. H4 shares the
replay correction but never expands the ten-class head or uses domain ID
to select an inference classifier.

## Validation to perform later

The written checks are `tests/test_standard_cl_math.py` and
`tests/test_standard_cl_pipeline.py`. They include direct systems, generalized
metrics, numerical QP reference, rank-deficient gradients, KD invariance,
BN rollback, float32/float64 paths, state resume, Schur/derivative identities,
dead/live effects, chunked trace, domain replay and collector integrity.
The SciPy QP reference is explicitly skipped if SciPy is absent; a skip is
not a numerical verification. None of these checks was executed here.
