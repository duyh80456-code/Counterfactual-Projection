# Fixed-checkpoint kernel ablation — 2026-10-27

Registered before any experiment execution. Exploratory seeds: 0,1 only.
Seeds 2–7 are reserved for subsequent confirmation; do not use them for tuning.

Resume the original P1 noBN/superclass/Inc5 checkpoint at the end of task 1
(100 epochs, 2000 SGD steps), or its native task-2 epoch-0 checkpoint. Take
the original first task-2 SGD step and capture it2001 immediately before the
intervention. Preserve learner, teacher, exemplars, optimizer, scheduler,
window, gradients, modes and RNG. A baseline checkpoint is admissible directly;
a full-arm checkpoint requires bitwise comparison with the matching baseline.
Require original protocol, dataset fingerprint and class order, and reject
task-1 live-GIR/GIR-only amendments. Missing baseline task1 checkpoints may
be reconstructed under the recovery registration below.

Use the 6be6543 normal policy (the inherited default-off path is separately
checked against that source). Three arms differ only in projection kernel:
tiny_V K_E; native K; random A with the same per-site tr(K_E) as tiny_V.
tiny_V and random use matching shapes with up to 16 incoming directions/site.
This is NOT a rank-matched comparison with native K. Incoming feature count
is not the matrix rank of K_E; empirical ranks of both kernels are unmeasured.
All use the same seed, native Gram solver, power10, mu/LM rule, original Gram
trust radius, certificate and acceptance checks. Native K is untruncated;
do not claim its empirical matrix rank is necessarily full. Actual update norm is an
observation only; do not clamp it or install a different sharpness policy.

On copies of the same it2001 state, attempt k=1..8 with GIR disabled. Recompute
each direction. Rejections keep the current state; do not stop merely because
lambda falls below 40. Record k=0 and every attempt: s, norm of the actual
parameter difference before/after VNA, Delta, window CE+KD parts, independent
held-out CE+KD parts, V_O, native lambda_max GGN (power10), full seen-test
old/new/total accuracy. The independent loss batch is a fixed balanced batch
of up to 128 official test examples, disjoint from the training window; it
never affects acceptance, selection, LM or hyperparameters. Measurements
must leave model/optimizer/engine/RNG bitwise unchanged.

Save the state after attempt 1, including a rejected/no-op first attempt. After
finishing the copied chain, restore that first-attempt state and continue the
native training loop to task 5. Subsequent interventions use VNA–GIR with the
same arm's kernel and original schedule. Keep teacher/exemplars/SGD intact.
Report old/new CNN/NME and average incremental accuracy after each task.
Mark nonfinite-loss divergence honestly; do not replace seeds or shorten tasks.

Before interpreting s1, replay its original full arm through task 2 epoch 43.
Compare all interventions (s, mu, iteration, accepted/rejected and swaps) and
full-precision epoch losses against original artifacts; report the first
discrepancy. Record deterministic-algorithm, CuDNN, TF32, CUDA/PyTorch/device
and CUBLAS workspace settings. Missing original flags are 'unrecorded', not
assumed matching. Missing flags do not veto an otherwise matching replay.
Replay mismatch blocks conclusions; it does not justify
silently changing GPU flags or the algorithm. No causal attribution without
supporting evidence.

Notebook: T4x2, GPU0 seed0, GPU1 seed1; each lane runs vna, kernel_self,
vna_rand sequentially (s1 replay gate first). Native epoch checkpoints handle
continuation resume. Chain state is saved between attempts; deadline exit 75
preserves the last safe native epoch and copied-chain progress. Artifacts
record source/checkpoint hashes, identity and measurement protocol. No ZIP
overlay, no new preflight, no seed search. GPU runtime is not yet measured;
no claim that all work fits in a single Kaggle session.

## Review clarifications — 2026-10-27, before GPU execution

Check the actual VNA parameter displacement against logged Delta in both
the copied chain and native continuation, before GIR changes any slots.
Exclude this measurement time from algorithm compute-budget accounting.
Do not clamp or change the 6be6543 update; preserve raw results, but set
interpretation_allowed=false if an applied step exceeds Delta or its audit
is missing. A no-op with an undefined radius is not a detected violation.
This checks the logged legacy radius, not a proof that its Gram computation
equals a separately reconstructed exact LM radius.

Keep strict kernel_variant checking for ablation resumes. Old engine states
are admitted only through explicit import after source validation, retaining
mu/nu, warm data, counters and adapter state. Baseline sources without engine
state use the normal fresh engine for the original task-2-only policy.

Random-A is one seeded trajectory per seed, with new dedicated draws at
each intervention. It is an exploratory comparator, not a repeated-random
estimate. Do not claim tiny_V statistically dominates random-A. No extra
random replicates or held-out seeds are run under this registration.

Report V and V_O as pre-VNA quantities. For kernel_self, the corresponding
kernel spectral estimate is native GGN before the update; the engine's legacy
lambda_E instead describes virtual scaffolding and must not be substituted.
For virtual kernels lambda_E describes J_E.T H J_E before the update.
Power10 is an estimate, not a certified upper bound: ratio_estimate is purely
descriptive, bound_certified=false. Post-VNA and post-VNA+GIR measurements
are labelled separately. No spectral-bound conclusion is inferred from them.

## Checkpoint recovery and missing-record exception — 2026-10-27

When discovery finds no admissible task1 checkpoint, run --mode make_task1
on the same lane before replay/ablation. Train icarl task1 for 100 epochs on
unmodified historical source 29cc735ab8770b3602c54c3c1f157649d02f807a (s0)
or 6be654310c2e4123391fd7d8d31716d25c89c6fb (s1), in a separate process.
Use the native noBN/superclass/Inc5/0.05 recipe, original PyCIL pin, exemplars,
optimizer/scheduler/RNG and native epoch resume/deadline exit75. Export
end_task1/task1.pt only after checking all 100 train_loss/train_acc/lr values
for exact equality with original epochs.json; any mismatch rejects recovery.
Retain reference epoch hashes and historical commit provenance.

Search attached primary s1 records, including
nlnk1607/adadad/vna4_main (2)/outputs/6be654310c2e/vna_v4_main/icarl_full__nobn__superclass__s1/.
If protocol.json or epochs.json is still absent, allow s1 ablation without
replay under interpretation_scope=no_replay_audit. This is an explicit
exception to original-trace verification: do not claim equality for absent
epochs. A recovered checkpoint is provisional until exact anchor validation.
Require the original interventions record and reconstructed it2001 state to
have L_before=2.853116586477115 and KD_before=1.2374861653019964. Available
epoch traces remain mandatory checks; missing protocol setup is derived
transparently from the validated baseline and frozen s0 schedule authority,
not labelled as a recovered original protocol. No new K0 preflight is run.

For an available replay, conclusion_allowed = not differences. Record
original_flags='unrecorded' plus current flags(); do not compare current
flags against a protocol field the original run did not record. Runtime
flags must still match on subsequent resumes of the new experiment.
