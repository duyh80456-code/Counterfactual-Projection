# Common noBN gradient clipping study

## Inputs and order

Run `notebooks/vna4_clip_AB.ipynb` first with T4 x2 and CIFAR-100.
Attach the four current historical Outputs (archives or expanded files):

| Download filename | Contents |
|---|---|
| `vna4_gram_m1s0_ir.tar.gz` | completed Gram/power10 full s0 and original baseline s0 |
| `vna4_main (2).tar.gz` | M1 baseline/full s1 and interrupted s2 |
| `vna4_main.tar.gz` | C3 original-default CBP/ReDo/S&P/UPGD checkpoints |
| `vna4_main (3).tar.gz` | C2, including completed GEM s0 |

Names are descriptive, not a discovery requirement. Each source is separately
namespaced; conflicting checkpoints are not merged or silently relabeled.
Actual current inputs contain all four safe end-task-1 checkpoints needed for A.
S&P s0 failed in task11 at step118; UPGD s2 completed all20 tasks and receives
the completed-run prefix check.

GPU0 runs A: baseline s2, full s2, CBP s1, ReDo s1, each at
None/1/2/5/10/20/50 for60 task2 updates. GPU1 runs B: baseline s0 and s1,
all20 tasks at None. Full always uses the original frozen K0, Gram/power10;
no recalibration or LR/head/epoch/method-policy change is permitted.
Reports include CE/KD, preclip norm, head logit maxima, parameter norm, LR,
and reset counts. A failed pre-update record is retained.

The driver writes `CLIP_STUDY_REPORT.md` and `grad_clip_selection.json`.
Selection requires both full baseline replays to match every original
full-precision epoch loss exactly (1050 each), finite nonempty norm monitoring,
and all28 A cases to finish or declare divergence. The selected threshold is
the smallest of1/2/5/10/20/50 with tau >=2*G_max and60 successful updates in
all four A runs. Missing evidence or no feasible threshold stops before C;
the grid is never extended. No tau or measured G_max has been chosen locally.

Then run `notebooks/vna4_clip_C.ipynb`, attaching the A/B Output plus the same
four historical Outputs and CIFAR-100. It recomputes selection from the saved
A/B evidence, checks hashes and the common source commit, and requires the
output preregistration written before C. Completed diagnostic jobs are skipped
only when their cached identities match. The two C lanes share a queue.
After divergence a lane takes the next queued job; infrastructure errors are
reported separately rather than labeled successful divergence.

Archives are refreshed periodically and on exit, with separate job logs and
existing deadline/exit75 handling. The extracted `_sources` cache is excluded
from archives, so historical Inputs must be reattached for later sessions.
Native checkpoints for interrupted training remain archived. Diagnostic jobs
are restarted from their last native safe checkpoint, never from a partial
optimizer update. Attached old Outputs remain unchanged.

## Phase C proof scope

Baseline s0/s1 reuse needs the entire B replay and its maximum norm <=tau.
Completed full s0/s1, CBP s0, ReDo s0, S&P s1/s2, UPGD s0/s1/s2 and GEM s0
receive an exact/no-clip task1–4 check (250 epoch losses). This is the user's
registered **prefix** reuse rule, not a claim that all20 tasks were replayed.
Where an old competitor checkpoint has no full-precision epoch log, first
replay its historical pinned producer for that prefix to regenerate the
reference; rounded notebook logs cannot certify equality.

Interrupted runs may import the safe native checkpoint only after every epoch
before that checkpoint matches exactly with zero clipping. Otherwise rerun
from scratch. Model, optimizer, scheduler, RNG, teacher, exemplar memory and
VNA state remain native; the verified norm history is added to the clipping
state. The original producer identity and summary remain in the reuse package;
a separate certificate records the effective common threshold, approving
study, consumer commit and proof scope. Collection excludes A/B/prefix and
source-cache runs from final scores.

WA/PODNet/C-Flat seeds0–2 and GEM s1/s2 train fresh. All C jobs use the same
selected tau and noBN LR0.05. Full remains Gram/power10. The table reports
reuse/resume/fresh action, equivalence scope, n_clipped and clip rate.

## Clipping arithmetic and local checks

`--grad-clip` defaults to None. Norms are always measured on noBN training
updates. Multiplication occurs only when norm >tau, with scale
tau/(norm+1e-6). Inactive clipping never writes gradients. GEM is observed
after projection; C-Flat after final gradient aggregation at its base SGD;
UPGD after noise and utility weighting, using the pinned native arithmetic
and RNG draw order. VNA/GIR parameter edits and diagnostic optimizers are not
clipped. Checkpoints persist the threshold and norm history and reject a
threshold mismatch.

`grad_clip.json` contains per-task/run counts, quantiles and clipped step/norm
pairs. A nonfinite norm is explicitly counted; max is null in that case,
with a separate finite maximum and quantiles labeled finite-only.

Related local tests check inactive bit identity, active bounds, UPGD state and
RNG identity, C-Flat aggregation placement, VNA exclusion, native wrappers,
strict epoch coverage, closed-grid selection, stale proof rejection, restore
exclusions and notebook syntax. Real T4 replay, G_max, selected tau and final
A/B/C numerical results require running the notebooks; none is fabricated.

Local affected-file run: **121 passed** (test_grad_clip, test_clip_study,
test_main_cil, test_competitors, test_cil_run_plan, test_gram_m1_ir,
test_exact_gpu_lanes). This is a CPU validation result, not a T4 replay result.
