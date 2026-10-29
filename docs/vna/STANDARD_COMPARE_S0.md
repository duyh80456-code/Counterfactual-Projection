# Final seed-0 H1–H4 Kaggle comparisons — 2026-10-29

Source pin: 11d61bb7a1eafcc11f9d78a422a28351c2f0309a. Four independent notebooks, GPU0 baseline and
GPU1 full method on T4×2. H3 additionally queues BP on GPU0 and VNA-only/GIR-only after full on GPU1. No source ZIP, old baseline, Gram schedule or old VNA
checkpoint is required. Papers motivate baseline choice, not a paper-score gate.

| Notebook | Matched baseline / full method | Native protocol | Fresh input |
|---|---|---|---|
| standard_cl_H1_s0.ipynb | iCaRL / Replay-Aware VNA | CIFAR10020×5, PyCIL ResNet32 BN, memory2000,200/170 epochs | Optional CIFAR100 Python files; auto-download when absent |
| standard_cl_H2_s0.ipynb | official LPR / Geometry-Aware VNA | Online Split-CIFAR10020 experiences, SlimResNet18 nf20, batch10×3 repeats, replay2000, validation5%, SGD0.1 | Optional CIFAR100 Python files; auto-download when absent |
| standard_cl_H3_s0.ipynb | official ConvCBP / BP + VNA→GIR | H3-small, five arms:100 binary ImageNet32 tasks×50 epochs, batch100, native ConvNet, SGD0.01/momentum0.9 | Authorized original files listed in H3_SMALL_REQUIRED_FILES.json |
| standard_cl_H4_s0.ipynb | ER / prior-domain constrained VNA | MNIST20 rotations0,15,…,285°, MLP256 BN, fixed10-class head, reservoir2000,1epoch/domain | Optional four MNIST IDX files, plain/gzip; auto-download when absent |

Enable Internet and Kaggle Secret github_token. Set DATA_DIR only if several
compatible dataset roots are attached. CIFAR files are train/test/meta in the
original Python format. MNIST files are train-images-idx3-ubyte,
train-labels-idx1-ubyte,t10k-images-idx3-ubyte,t10k-labels-idx1-ubyte.
Official torchvision download/integrity checks are used, never fake fallback.

H3-small needs200 original class files: seed0's first100 pairs from upstream
class_order. Each file is700×3×32×32, numeric and finite: first600 train, last100
test. No label file or all1000 classes is needed. The exact filenames and pairs
are in H3_SMALL_REQUIRED_FILES.json. Missing/malformed inputs block training.
H3 automatically downloads the archive linked by the pinned upstream CBP README when attached class files are absent. It extracts only the 200 required classes and deletes the archive after successful extraction. Internet and enough Kaggle disk for the complete compressed archive plus selected classes are required; Google Drive quota failures stop setup. Existing attached data still takes priority.

## H2 baseline and environment

LPR source76c44f8f2481f4e23881b2463eb58aaf94affabb and plugin SHA256
07d0aef6077bac8b77f4e52194b79e155696da956a14fa9cc55381d9ac28e7d6 are checked.
Use the pinned README Split-CIFAR100 example: omega0=4,beta=2,every_iter=30.
Earlier1/1/None came from generic defaults and is superseded for these notebooks.
No experiment outcome was used to choose these values. Other online settings
come from the pinned experiment, benchmark, model and optimizer configurations.

The notebook creates an isolated Python3.10 environment, installs torch2.2.2,
torchvision0.17.2 and requirements-h2.txt, then Avalanche0.4/typing4.4 last.
Only torch2.2's typing>=4.8 metadata requirement is overridden: original
Avalanche0.4 requires typing4.4's Protocol constructor behavior. This exact
exception is tested on real Python3.10 CPU, without any constructor shim.
grpcio1.62.3 prevents a second typing conflict. The dependency gate rejects
missing packages or any other unresolved conflict. Exact versions are printed.
Required APIs are constructed/executed, not merely imported. See
H2_AVALANCHE_FINAL_ENVIRONMENT.json and STANDARD_VALIDATION_COMPLETION.md.

Baseline retains native ReplayPlugin/ReplayDataLoader. Full uses the same loader
with an explicit per-sample incoming/replay boolean before the final task label;
no positional constraint mask. Origins survive permutation. Original untagged
experiences feed ClassBalancedBuffer; baseline loader behavior is unchanged.
Both loss gradients come from one shared physical forward and sum to total CE.
Full retains cadence512. Geometry weight starts at1, then a bounded seen-validation
grid{0,0.1,1,10,100} from task2 selects and freezes it. Up to3 eligible windows
are allowed; exhaustion freezes FALLBACK_ER and continues native ER training,
retaining any earlier correction prefix. It is not a pure-ER trajectory. Trials restore all
state, share fixed held-out tensors and use only classes already observed.
No test data selects the weight. This is one-step calibration, not full-run
hyperparameter optimality; its external experience-prefix metadata is disclosed.
LPR and VNA have different algorithm-specific compute, reported in wall/GPU hours;
their native training budgets match, not their elapsed runtime.

## H3-small bounded search

configs/standard_cl/h3_imagenet_small.json explicitly overrides num_tasks100 and
num_showings50. CLI --num-tasks/--num-showings also enter the protocol and resume
identity. The pinned upstream cfg is never edited. Native CBP hyperparameters,
600/100 split and task-specific zero-head reset without clearing momentum remain.
The full model is not reinitialized at task boundaries. The original class file
has14000 sequence entries/run; use its actual prefix, not a fabricated repeated
1000-class permutation. The5000×250 config stays available separately, unqueued.

Full: physical VNA then fixed-capacity GIR, first full128-sample window of each
task then every200 updates. Planned local steps2,202,402 give300 opportunities.
The4000 schedule remains a separate ablation. Supported sites
remain fc1→fc2 and fc2→fc3. Generate tiny_V proposals normally. Rank all resets by
R=-gᵀδθ_reset-(μ/2)||δθ_reset||² at the current network, using the same μ as the
exact identity objective. This is a parameter-only first-order LM proxy, not a
Schur or global guarantee. Ties break by stable candidate name, without RNG.
Evaluate at most16 swaps with the unchanged full transactional objective and
componentwise loss checks. Identity is always included. Commit the exact stored
winning state; rejects/failures restore model, optimizer and RNG. Candidate-local
solver failures are skipped; only identity/background solve failure retries the
whole search at a new shared damping. Logs contain
all preliminary scores, shortlist/pruning reasons, exact trials, failures and
actual swaps. Optimality is only within the evaluated shortlist plus identity.
This reduced pilot does not reproduce published full-horizon CBP results.
Native updates per arm:100×50×12=60000, before correction overhead.

## Preflight, resume and output

Before either lane starts, a separate process checks dependency/API availability,
data structure/labels/task order, native backbone forward and (H2) real replay
batches, LPR callbacks, gradient decomposition and actual geometry projection.
CPU fixtures are explicitly labelled and never reported as benchmark results.
The isolated interpreter must see two CUDA GPUs and execute a CUDA operation;
no success is inferred from installation. Any gate failure blocks both lanes.

Eleven-hour deadline, exit75 exact resume, heartbeat30s and archive300s remain.
Attach previous Output and original data, keep the same source/config/run tag.
Canonical CIFAR train/test/meta and MNIST's four raw IDX files determine data
identity; optional compressed copies do not change it. H3-small has its own tag.
No seed substitution, clipping or unregistered horizon truncation is introduced.

Artifacts include protocol/tasks/epochs(or H4 trajectory)/summary, interventions,
checkpoints, costs/VRAM, comparison.csv, curves.csv and accuracy_curves.png.
H2 adds online.json, incoming pre-update accuracy and replay/feature diagnostics.
H3 adds per-task learning speed, last50 error, candidate counts and actual swaps.
Collection validates protocol and completed horizon; paused/failed rows retain
their status, partial curves and costs but no completed-run performance claim.

T4 runtime is unmeasured. More than one session may be necessary; the registered
horizon remains unchanged. No real dataset training or Kaggle/CUDA run occurred
in this coding task. See the focused CPU test log and environment report.


## Current correction and validation scope

Shared V2 residual and approximate-solution error certificates remain unchanged.
Seeded operator audits, variance-shrunk Jacobi and fixed-mu iterative refinement
are used; train/eval gates have dtype-scaled roundoff allowances and separate
failure reasons. Neutral accepted updates are not evidence of improvement.
H3 uses single current-task CE and H4 uses explicit prior-domain reference loss.
H4 has a shared fixed backbone/head and training-only domain IDs.

All four main and four `*_ablations_s0.ipynb` notebooks share the pin above and
fresh output tags. Numerical H2 ablations require the main selected geometry_selection.json
(seed0, identical source and dataset fingerprints). They share a frozen weight
with a new fixed-weight reference; neither retunes. Compare numerical effects
against that reference, since the calibrated main has a different prefix.
Fallback cannot supply an authority. Zero-weight and fallback cohorts are
reported separately and are not evidence of geometry regularization benefit. H3 old schedules contrast phase/frequency together. Native baseline
training is unchanged; H3 has five queued arms and substantially more correction
work, which can require multiple sessions. No automatic truncation occurs.

H2 exports geometry_selection.json and selected geometry fields in summary and
comparison.csv. H1/H2/H4 export accuracy matrices; acceptance_by_task.json/CSV
separate attempts, certificates, candidates, effective/neutral/improved updates
and failures. See STANDARD_SIX_FOLLOWUPS.md and STANDARD_VALIDATION_COMPLETION.md
for exact tests and methodological limits. Real Python3.10/Avalanche0.4 tests
have run without skips in that environment. GPU checkpoint verification was
explicitly skipped by the user and must not be described as passed.

The notebooks now pin the committed follow-up source shown above. H3 saves at the end of each epoch containing an intervention; an
interrupted unsaved epoch replays from its prior checkpoint. This preserves
logical events, not computation already spent. More frequent saves add I/O.
