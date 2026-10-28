# Standard continual learning VNA

H1–H4 are independent executable research branches. Historical VNA–GIR,
M1/P1 policies, notebooks and checkpoint formats are unchanged. Read the
[completion report](../STANDARD_CL_COMPLETION_REPORT.md), [mathematics](MATHEMATICS.md)
and [protocol](PROTOCOL.md). CPU smoke results are software checks, not CL
performance experiments. Every superiority/cost-benefit hypothesis is **UNVERIFIED**.

| Branch | Entry point | Configuration |
|---|---|---|
| H0 fixed-state kernel audit | `diagnostics.cl_mechanism` | explicit checkpoint/window and norm/time budgets |
| H1 replay interference | `experiments.standard_cl` | `configs/standard_cl/h1_cifar100_bn.json` |
| H2 feature geometry | `experiments.standard_h2` | `configs/standard_cl/h2_cifar100_bn.json` |
| H3 tangent reconfiguration | `experiments.standard_plasticity` | `configs/standard_cl/h3_regression.json` |
| H4 fixed-label Domain-IL | `experiments.standard_domain` | `configs/standard_cl/h4_{rotated,permuted}_mnist.json` |

## Installation and inputs

From the repository root, use Python 3.12 and a compatible torch/torchvision
pair (the CPU audit used torch 2.2.2). Install the package and existing upstream
requirements; SciPy supplies the independent numerical QP oracle.

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[vision,test,diagnostics]' scipy
.venv/bin/python -m pip install -r requirements-upstream.txt
bash scripts/fetch_third_party.sh
export REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The fetch script checks the existing registered dependency pins. The local
verification linked the already populated dependency checkouts from the original
workspace; those machine-specific links are not committed. No dataset download
or GPU training is performed by import.

CIFAR input: `DATA/cifar-100-python/{train,test,meta}` from official CIFAR-100
Python files. MNIST input: `DATA/MNIST/raw/` containing torchvision's extracted
train/test images and labels. Train/test splits remain separate. H3 creates
`OUT/data/regression.pkl` with the pinned upstream slowly-changing regression
generator. It contains `(inputs, targets, target_network)`; generated inputs
and targets are fingerprinted. No historical checkpoint is required.

## CPU verification

```bash
.venv/bin/python -m pytest tests/test_standard_cl_math.py tests/test_standard_cl_pipeline.py tests/test_standard_cl_audit.py -o addopts='' -q
.venv/bin/python -m pytest tests/test_standard_cl_completion.py -o addopts='' -q
.venv/bin/python -m experiments.standard_smoke
.venv/bin/python -m diagnostics.cl_mechanism --fake-data --random-replicates 5 --update-norm .01 --out outputs/h0_smoke.json
```

`standard_smoke` uses the actual full ResNet32 BN, synthetic images, one SGD
step and a tiny_V correction at the final eligible virtual site. Locked damping10/norm0.01 are declared
smoke settings, not the benchmark's damping defaults. `--fake-data` on the CIL
runner replaces only the declared smoke dataset/backbone/size controls with a
tiny BN fixture. Its source records `synthetic_smoke=true` and effective settings.
The Domain-IL fixture has disjoint random train/test images and a width16 head;
H3 has 32 synthetic online samples. Their metrics must never be used as benchmark
results or merged with real-data runs.

## Launch commands (not executed on benchmark data)

```bash
# H1 native iCaRL-style control and H1, respectively; --tasks 20 is full B0 Inc5.
.venv/bin/python -m experiments.standard_cl --data-dir DATA --tasks 3 --device cuda:0 --arm baseline --baseline icarl --seed 0 --out outputs/h1/icarl/baseline/s0
.venv/bin/python -m experiments.standard_cl --data-dir DATA --tasks 3 --device cuda:0 --arm h1 --baseline icarl --seed 0 --out outputs/h1/icarl/h1/s0
# Native constrained, random, GN, gradient and unconstrained virtual controls:
# repeat the identical recipe with --arm native/random/gn/gradient/vna.
# ER with herding is a separate --baseline er family.
.venv/bin/python -m experiments.standard_h2 --data-dir DATA --tasks 3 --device cuda:0 --seed 0 --out outputs/h2/h2/s0
.venv/bin/python -m experiments.standard_h2 --data-dir DATA --tasks 3 --device cuda:0 --arm native --seed 0 --out outputs/h2/native/s0
.venv/bin/python -m experiments.standard_cl --data-dir DATA --tasks 3 --device cuda:0 --arm baseline --baseline agem --seed 0 --out outputs/controls/agem/s0
.venv/bin/python -m experiments.standard_cl --data-dir DATA --tasks 3 --device cuda:0 --arm baseline --baseline er_ace --seed 0 --out outputs/controls/er_ace/s0
.venv/bin/python -m experiments.standard_plasticity --arm h3 --seed 0 --out outputs/h3/h3/s0
# Repeat H3 with --arm bp/cbp/redo/snp for the pinned upstream adapters.
.venv/bin/python -m experiments.standard_domain --config configs/standard_cl/h4_rotated_mnist.json --data-dir DATA --device cuda:0 --seed 0 --out outputs/h4/rotated/h1/s0
.venv/bin/python -m experiments.standard_domain --config configs/standard_cl/h4_permuted_mnist.json --data-dir DATA --device cuda:0 --arm baseline --seed 0 --out outputs/h4/permuted/baseline/s0
.venv/bin/python scripts/collect_standard_cl.py outputs/h1 outputs/h2 outputs/h3 outputs/h4
```

Unified H1/H2 configurations are `h1_cifar100_bn_unified.json` and
`h2_cifar100_bn_unified.json`; each sets mode=unified and cadence=1. For example:

```bash
.venv/bin/python -m experiments.standard_cl --config configs/standard_cl/h1_cifar100_bn_unified.json --data-dir DATA --tasks 3 --device cuda:0 --arm h1 --out outputs/h1/unified/s0
```

Resume by repeating the exact command and output directory. Immutable protocol
identity is checked before loading `latest.pt`. H1/H2 and H4 roll back a partial
epoch on deadline exit75. H3 checkpoints online position, upstream auxiliary
state and RNG; an interrupted swap rolls back its enclosing native step before
saving, so that slot is attempted again. Completed H1/H2/H4 runs return their
existing verified summary. No seed replacement or clipping is automatic.

H1/H2 synthetic startup example: append `--fake-data --tasks 2 --device cpu` and
use a new output directory. H4 synthetic startup: `--fake-data --domains 2
--batch-size 4 --device cpu`. H3 synthetic startup: `--fake-data`. Synthetic
fixtures are shortened **only** under that explicit flag.

## Baseline and normalization limits

A-GEM and ER-ACE implement their update laws with a declared **task-boundary
PyCIL herding / expanding linear-head adaptation**. They are not official
end-to-end Mammoth reproductions. Sequential native BN forwards really update
running statistics. H1/H2's functional pooled-forward loss decomposition is a
different normalization protocol and must be interpreted accordingly. See
[baseline sources](BASELINES.md). Authentic LPR is **BLOCKED** in this environment:
its Avalanche/Hydra/OmegaConf environment is absent; H2 is never labelled LPR.

H3 calls native LoP Backprop/ContinualBackprop and GnTredo directly. S&P uses
upstream Backprop's perturbation and weight decay. The ReDo supervised hook calls
GnTredo after native SGD, preserving its default threshold/period. No upstream
algorithm source is edited. Default CBP rate/S&P scale are declared config values,
not claimed tuned best configurations.

H4 supports ER, native/GN/gradient controls and L2 through `--weight-decay`.
GroupNorm is available for those controls; virtual GroupNorm and stochastic
dropout corrections explicitly refuse. Domain identity selects a data transform,
never a classifier. Train-mode parameter certificates do not imply eval-mode BN
protection. GPU profiling remains to be performed with the same launch commands;
summary/intervention artifacts contain wall time, JVP/VJP/CG accounting and CUDA
peak allocation when CUDA is actually used. No GPU readiness claim is made.


## Main seed-zero notebook pairs (supersedes the earlier LPR block)

2026-10-28 dataset correction: the main notebooks now use CIFAR10020×5 (H1),
original **online** LPR Split-CIFAR100/SlimResNet18 (H2), original **Continual
ImageNet32 binary H3-small100×50** stream (H3), and **20 rotated MNIST** domains (H4). Old offline
H2, regression H3 and permuted H4 commands below/above remain separate experiments.
Use STANDARD_COMPARE_S0.md for the current inputs, compatibility pins and scope.

See [STANDARD_COMPARE_S0.md](../STANDARD_COMPARE_S0.md) for the four registered-horizon
notebooks. Main H2 uses actual Avalanche0.4 online training and verified official
LPR callbacks; its notebook installs the isolated compatible environment. The
older ER/PyCIL callback adaptation remains separate. H3 main
uses `--arm vna_gir --config configs/standard_cl/h3_imagenet_small.json` (VNA -> GIR,
bounded16-candidate search), preserving exhaustive H3 in its separate full config.
Official pairs are H1 iCaRL, H2 LPR, H3 CBP and H4 ER. Paper references justify
baseline selection only. GPU/benchmark validation is still pending.
