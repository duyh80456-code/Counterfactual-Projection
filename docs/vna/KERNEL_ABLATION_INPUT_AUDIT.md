# P1 kernel-ablation input audit — 2026-10-27

Inspected the user's downloaded archives without training. No s1 replay or
kernel result is claimed from this inspection.

| Artifact | Available | Missing / incompatible |
|---|---|---|
| p1_ablation_bundle.zip | s0 baseline protocol/tasks/train log; s0 full protocol, epochs, interventions, tasks, schedule; s1 full interventions; PARAMS.json | No .pt checkpoints. No primary s1 protocol.json or epochs.json. PARAMS is not a checkpoint or substitute for those primary artifacts. |
| vna4_main.tar.gz | s0 baseline checkpoints at task20, step35750; s0 full checkpoint at completed1/epoch100/step2000 from d88abc4 | Baseline checkpoints are too late for the anchor. The full checkpoint contains task-1 GIR swaps and is not accepted as a baseline-equivalent source. No s1. |
| vna4_gram_m1s0_ir.tar.gz | Same historical s0 checkpoints; profiles/task1_source.pt at completed1/epoch100/step2000; later Gram full output | The task1_source is the same d88abc4 full source, not a verified matching baseline checkpoint. No s1 anchor. |

Concrete provenance discrepancy: the d88abc4 full checkpoint records task1
epoch100 train_loss=0.49428418427705767, train_acc=0.8164 and 29 GIR swaps
at it1025. The supplied target s0 full epoch trace (29cc735) instead records
train_loss=0.4936316817998886, train_acc=0.8144. These are not the same task-1
trace. The new runner refuses to silently reuse the historical full checkpoint.
The two archived copies of that task-1 source have identical SHA256:
`76c9a041b71c994483ef1ab942c476f84dd9d75de7b0a03bc15fbb0584af2058`.

Target metadata from the supplied bundle: s0 commit29cc735, s1 commit6be6543,
normal schur_gamma_bound_rho25_v2, Gram/power10, Inc5, K0=512, memory2000,
noBN LR0.05. s0 first intervention s=0.17132912554034105,
mu=41.392471440725124; s1 s=0.2296649942700549,
mu=0.5078344679654524. These are original observations, not new measurements.

Original s1 task2 has interventions at it2001 and it2513, with 0 and 4 swaps.
The bundle reports a prior replay failure at epoch43 but includes no replay
trace or full-precision s1 epochs. It cannot establish the cause. The runner
records actual deterministic/CuDNN/TF32/CUBLAS/device flags and compares event
decisions and epoch losses; missing original flags remain explicitly unrecorded.
Pinned upstream PyCIL sets cudnn.deterministic=True and benchmark=False, but
that source statement alone does not prove the original runtime's full settings.

Required to execute: matching original baseline end-task1 (or native task2
epoch0) checkpoint for each of s0 and s1, with learner/teacher/exemplars,
optimizer/scheduler and RNG; original full s1 protocol.json and epochs.json.
The existing s1 interventions.json can be reused. If a full-arm checkpoint is
supplied instead, its matching baseline checkpoint is needed for bitwise proof.
The first task2 SGD step is reconstructed natively, its pre-intervention losses
must equal the original it2001 event, and the complete resulting anchor/window
is compared bitwise across all three arms. No substitute checkpoint or seed is
selected when these gates fail.
