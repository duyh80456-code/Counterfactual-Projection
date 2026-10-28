# H1–H4 benchmark comparisons, seed 0 — 2026-10-28

Notebook source pin: `8cb50ca7114513864a44e0dc0eb7e1034d46ddd0`.

This registration supersedes the earlier offline-H2/B5-H3/five-domain notebook
plan. Papers justify baseline choice; published accuracy is not a gate. The
four directions are different benchmark problems, evaluated within each pair.

| Notebook | Baseline / full arm | Dataset and horizon | Input |
|---|---|---|---|
| standard_cl_H1_s0.ipynb | iCaRL / Replay-Aware VNA | CIFAR100,20×5 classes, PyCIL ResNet32 BN, memory2000,200/170 epochs | Python train/test/meta |
| standard_cl_H2_s0.ipynb | LPR / Geometry-Aware VNA | Online Split-CIFAR100, SlimResNet18 nf20,20 experiences, minibatch10 repeated3 times, validation5%, class-balanced replay2000 | Python train/test/meta |
| standard_cl_H3_s0.ipynb | CBP / BP + VNA → GIR | Continual ImageNet32,5000 binary tasks,250epochs/task, batch100, original ConvNet | Original 0.npy …999.npy, each700×3×32×32 |
| standard_cl_H4_s0.ipynb | ER / ER + Domain-Adaptive VNA | MNIST,20 rotations0,15,…,285degrees; fixed10 labels, one epoch/domain, MLP256 BN, reservoir2000 | Four IDX files, plain/gzip |

All notebooks use T4×2: baselineGPU0/fullGPU1. No permuted-MNIST or regression
jobs are queued. Those old runners remain separate experiments. Seed0 is
descriptive, not a significance claim.

## Original protocol sources

H1 keeps PyCIL `f3509b8ca3f20660ce4aa13f19d5283de81b4b35`.
H2 follows LPR `76c44f8f2481f4e23881b2463eb58aaf94affabb`:
`config/experiment/split_cifar100.yaml`, `benchmark/{nc_scenario,split_cifar100}.yaml`,
`optimizer/sgd.yaml`, `strategy/{er,method_defaults}.yaml` and `experiments/main.py`.
It uses actual Avalanche0.4 OnlineCLScenario, Naive, ReplayPlugin,
ClassBalancedBuffer and IncrementalClassifier. Original LPR callback bodies
are verified and unchanged. Hydra, asynchronous evaluation and external wandb
logging are omitted. Evaluation is isolated from training RNG. Online experiences
are materialized before training for deterministic resume; their original
partition permutation consumes RNG at that boundary. Optimizer is SGD0.1,
momentum0, weight_decay0, no scheduler. Geometry-VNA shares the online protocol,
using its existing lambda1 feature penalty and cadence512, replacing LPR rather
than silently stacking onto it.

H2 creates Python3.10 because Avalanche0.4 does not support Kaggle's Python3.13.
Torch2.2.2/torchvision0.17.2, NumPy1.26.4, torchmetrics0.11.4, pytorchcv0.0.67
are pinned. Avalanche requires typing-extensions4.4 for its Protocol constructors. Keep that
original version and wandb0.15.12/protobuf4.25.8; override torch2.2's newer typing
metadata by installing Avalanche/typing4.4 without dependency resolution last.
The training uses torch.func without torch.compile. Algorithm source is unchanged.
This is a compatibility adaptation, not exact reproduction of the old dependency
stack. Local Python3.12 toy tests emulate only typing4.4's no-op Protocol
constructors; the production Python3.10 environment uses typing4.4 unchanged.
Avalanche retains orphan optimizer entries when its classifier grows; only
empty/None entries are pruned for torch2.2 serialization. Any nonempty orphan
state raises. A toy comparison checks bitwise equality with native online LPR.

H3 reads unchanged LoP `a6b79580d85f3025bdb601566d3627c5f489f13b`
`lop/imagenet/cfg/cbp.json` and class_order[0]. It retains SGD0.01 momentum0.9,
CBP replacement3e-4, maturity100, decay0.99, adaptable_contribution,600 train and
100 test images/class, raw float conversion, repeated class order and zero head
at every task without clearing momentum. Baseline calls ConvCBP.learn. Full
calls Backprop.learn then existing physical VNA→GIR at cadence10000/window128.
VNA/GIR scope is fc1→fc2 and fc2→fc3. Pool/flatten sites lack compatible training
statistics and are explicitly excluded, not assigned an invented kernel. Both
engine states, CBP utilities/ages, optimizer, epoch ordering, head reset and RNG
resume. H3 reports classification error in percentage points, not regression MSE.

H4's20-angle grid is registered here, not claimed to be a unique Avalanche paper
protocol. Domain identity is absent at inference; all domains share the10 labels.

## Inputs and sessions

Enable Internet and Kaggle Secret github_token. Fresh runs need only the dataset
in their row; no source ZIP, Gram authority, old baseline or best_config. Set
DATA_DIR only if several compatible roots are attached. Source must be committed
and notebooks regenerated with its exact40-character commit before uploading;
never pin an older implementation to run new code.

Eleven-hour deadline and exit75 resume stay enabled. Attach that notebook's
previous Output and same dataset, then rerun the same source pin/run tag. Only
matching run-tag output is restored. Local checkpoints take precedence; conflicting
attachment versions raise. Archives every300s, heartbeat30s. Datasets and the H2
environment are excluded from archives and rebuilt when needed. No notebook
automatically tests/smokes, clips, truncates or substitutes seeds.

H3 has **15 million native optimizer updates per arm**, before VNA/GIR overhead;
many sessions are required. Exact T4 hours/session counts are unmeasured. Other
branches may also need multiple sessions. No numerical runtime estimate is
invented. Collector keeps branches separate, validates paired identity and retains
failures. Whole-method differences do not isolate virtual-kernel benefits.

Only focused synthetic CPU checks are authorized here; no real dataset/GPU run.

Focused verification:17 tests passed (33.77s), including original online LPR
bitwise endpoint equality, LPR/Geometry resume, native Slim feature equality,
ImageNet split/native CBP/full-hook resume and affected H2/H4 math. The added
mid-correction/classifier-expansion rollback check passed in a subsequent2-test
rerun (10.39s). See STANDARD_BENCHMARK_CPU_TESTS.log. Local tests use Python3.12
and the documented Protocol-constructor shim; the production Python3.10 stack,
Kaggle install, CUDA, real datasets and timing have not been exercised.
