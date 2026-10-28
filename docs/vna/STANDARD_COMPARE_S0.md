# H1–H4 main comparisons — seed 0

Source pin: `SOURCE_PIN_TO_RECORD`. All notebooks use this immutable code commit;
no source ZIP is attached. These are full-horizon runs, with no automatic pytest,
smoke, task truncation, seed substitution or clipping in the notebook.

| Notebook | Baseline | Full algorithm | Fresh data input | Resource |
|---|---|---|---|---|
| standard_cl_H1_s0.ipynb | iCaRL | iCaRL + Replay-Aware VNA | CIFAR100 Python | T4 x2 |
| standard_cl_H2_s0.ipynb | LPR on shared ER/PyCIL setup | Geometry-Aware VNA | CIFAR100 Python | T4 x2 |
| standard_cl_H3_s0.ipynb | native CBP | BP + VNA -> GIR (`vna_gir`) | none; native regression generator | CPU, four cores |
| standard_cl_H4_s0.ipynb | ER | ER + Domain-Adaptive VNA | MNIST raw IDX, plain or gzip | T4 x2 |

Paper/framework references are **baseline-choice rationale only**. There is no
paper-number reproduction gate. H4 uses our explicit reservoir ER implementation;
Avalanche is a rationale/framework reference, not a claim that its trainer runs.

## Protocol

H1/H2: CIFAR100, BN ResNet32, PyCIL random class order seed0, B0 Inc5 full20 tasks,
memory2000, 200 epochs first task/170 later, original standard-CL LR and milestones,
final-epoch models. H1 uses the existing CE+KD iCaRL objective. H2 deliberately
shares an ER objective/herding memory between LPR and Geometry-Aware VNA; it does
not silently combine LPR with iCaRL KD. H2's fixed lambda1 is the existing penalty
variant, not an exact calibrated feature-drift constraint.

The LPR source is the official pinned callback implementation, with its original
preconditioner formulas and default options. A small interface adapter runs it
without the Avalanche/wandb stack: no callback algorithm body is rewritten; the
unused framework superclass and disabled logger imports are adapted. A PyCIL
linear-head view preserves its exact F.linear output, Parameter identities and
initialization RNG while allowing the official nn.Linear checks to include it.
Matrices/update cadence persist across task/epoch resume. This is an explicit
framework/protocol adaptation, not a full official experiment reproduction.

H3 uses the native LoP B5 stream (width5 ReLU, SGD0.01, 100 tasks x10000 points).
The earlier `h3` arm remains reconfiguration-only; it is not relabelled as full.
The new main `vna_gir` actually attempts VNA and then evaluates GIR with its own
loss certificate at cadence10000, window128. Resume restores both engines and
rolls back the enclosing native step if an intervention is interrupted. CBP's
registered rate1e-4/effective upstream maturity10 are preserved; no tuned-best
baseline or reproduction of published scores is claimed.

H4 runs rotated and permuted MNIST separately, five domains each, fixed10 labels,
MLP width256 BN, reservoir2000, existing SGD0.01 and one epoch per domain. Each
GPU lane queues the two streams. Their tables remain separate.

## Kaggle use

Enable Internet and grant the existing Kaggle Secret `github_token`. H1/H2/H4:
select GPU T4 x2; baseline is on GPU0 and full on GPU1. H3: select CPU, two lanes
with two threads each. GPU acceleration is unnecessary for that native runner.

CIFAR input must expose `cifar-100-python/{train,test,meta}` (or those files in a
single directory). MNIST must expose train/test IDX images and labels, optionally
gzipped. The notebook stages these paths without changing labels/splits. If
multiple dataset roots are visible, set DATA_DIR in the first code cell to the
intended directory; the notebook refuses to guess between different roots.
No Gram output, old baseline, best_config.json or source ZIP is required.

Runtime on T4 is **unmeasured**. More than one eleven-hour session may be needed,
particularly for the correction arms; this warning is printed before execution.
The horizon stays full. On exit75, attach that notebook's latest Output together
with its original dataset and rerun the same notebook/pin/run-tag. Matching
`standard_outputs/<RUN_TAG>/` alone is restored. Existing local checkpoints are
kept on an in-session rerun; conflicting attached versions raise. Archives are
written periodically and at exit; heartbeat appears every30 seconds. Data and
source checkouts are excluded from archives and are restaged/regenerated.

Final tables list completed-run metrics and leave paused/failed metrics blank.
The collector retains failures and pairs H2 to LPR and H3 to CBP. No kernel-only
benefit can be inferred from comparing complete algorithms without the H0 controls.
Seed0 gives a descriptive paired comparison, not a significance result.

## Build and verification

```bash
.venv/bin/python -m scripts.build_kaggle_standard_compare --commit SOURCE_PIN_TO_RECORD
REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_standard_compare_notebooks.py -o addopts='' -q
```

New focused tests passed4/4. A combined affected-path check passed16 tests,
17 deselected (29.19s), including old H1/H2/H3 resume and collector paths.
Log: STANDARD_COMPARE_CPU_TESTS.log. Source syntax/build checks are local;
Kaggle authentication, live archive transport, CUDA and real benchmark training
have not been executed. All scientific outcomes remain UNVERIFIED.
