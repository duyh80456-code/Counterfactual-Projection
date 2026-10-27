# Locked pilot configuration and comparison rules

This registration describes code defaults before any new run. All results,
runtime, memory and superiority claims are currently INCONCLUSIVE.

## H1 pilot

* CIFAR-100 B0 Inc5, tasks 1–3, paired seeds 0 and 1. ResNet32 **with BN**.
* PyCIL f3509b8ca3f20660ce4aa13f19d5283de81b4b35 head, herding and NME.
  Memory 2000, fixed_memory=false; explicit random class order determined
  by the paired seed. Same data order/augmentation mechanism across arms.
* Default task1 200 epochs, later 170, LR0.1, momentum0.9, initial weight
  decay0.0005/later0.0002; milestones [60,120,170] / [80,120], decay0.1.
  This follows the PyCIL iCaRL training recipe, not old P1's 100/50 noBN
  study. Different protocols must not be averaged together.
* Final epoch model, no warmup, no clipping, no test-set tuning. Native
  loss remains ER mean CE or iCaRL-style mean CE + old-only KD (T2).
* Default correction cadence512 optimizer steps, compared to the **last
  attempted** intervention, not modulo of current steps. Fixed parameter
  r=16 per virtual site, relative damping0.01*estimated tr(F)/p with floor
  1e-12; eight fixed-seed Hutchinson trace probes. CG tolerance1e-6,
  max200 iterations, true-residual check. These are registered numerical
  controls, not claimed parameter-free theorem consequences.
* No GIR, no old beta rule, no hidden radius/utility/shock trigger. Initial
  physical correction step is the declared current LR, with at most12
  dyadic line-search trials. Reject if the specified actual window losses
  increase or if a numerical certificate fails.

Every arm has its own output and immutable protocol. CLI selects native
baseline, gradient/native-kernel/GN constrained controls, random virtual
constrained candidate, unconstrained VNA-only, or H1 virtual constrained
candidate. The KD-only constraint variant is a distinct experiment.

Matched-update budget here locks the number/cadence of proposed correction
slots across correction arms. Actual accepted/nonzero updates may differ
and are reported. Native baseline has no extra correction, so it is **not**
claimed to have identical physical-update count. Compare native-correction
controls at those same slots before attributing gains to virtual kernels.
Matched-compute is a separate table: correction wall time may use at most
the registered ratio of accumulated native training time. If already above
that allowance, the next slot is a logged resource skip. One intervention
may overshoot its allowance before its true cost is known; report that
overshoot, never claim an exact wall-time cap. All probe/acceptance costs
remain included. No speed claim is made from JVP counts alone.

Unified mode is separate, requires cadence1, and includes actual optimizer
candidate, momentum and weight decay in the projected physical update.
A skipped/rejected/uncertified unified slot is a no-op, not uncertified SGD.
BN's native forward has already updated running statistics; the certificate
is about the whole **parameter** step at that buffer state, not inference
function preservation or every stochastic state transition.

## Metrics and persistence

H1/H2 have separate `*_unified.json` configurations with mode=unified and cadence1; CLI mode is recorded independently.

Report CNN total/old/new, old-only versus global classification, NME total
when supported, class/task accuracy matrix, average incremental accuracy,
final accuracy, task-consistent forgetting and BWT. The old set is fixed
within each task's trajectory. The pre reference is global old accuracy
immediately after head expansion but before new SGD; compare only that
same old label set when computing stability gap/recovery. Record early
steps1..10,20,50,100,200 and every epoch. No diagnostic chooses a step.

Log incoming/reference losses, gradient norms/interference, gains, curvature,
CG residual/iterations/error bounds, KKT, candidate/actual loss changes,
physical norm, acceptance trials, eval functional drift, actual native and
counterfactual BN drift, time, product calls, parameter count and peak VRAM.
VRAM is a process high-water mark since runner initialization; it is not
claimed as a separately measured per-candidate peak. Wall GPU-hours include
data/training/rehearsal/evaluation while the experiment clock is active.

Use separate protocol, epochs, interventions, trajectory, tasks and summary
JSON plus native-state `latest.pt`/previous checkpoints. Exact epoch resume
replays the same task/loader after restoring all RNG and learner/teacher/
exemplar/optimizer/scheduler/engine counters. Deadline exit75 rolls back a
partial epoch. Resource OOM and nonfinite loss have distinct failure codes;
no rerun with a different seed and no clipping occurs automatically.

## H0, H2, H3 and H4

H0's `compare_local` checks model/optimizer/BN/gradients/modes/freeze flags/RNG
and window/teacher state at each arm, computes damping once and passes that
same mu into every correction. Tiny_V, native K and random form the kernel
group; GN is a separate candidate control. The unfiltered optimizer has two
explicit references: true standard-BN forward and private/frozen buffers.
Neither is eligible for kernel attribution or reported as a matched arm.

For direction quality, pass `update_norm` as a common physical norm budget.
Initial s is capped by both the declared LR and radius/||d||; every rounded
trial displacement must fit the budget. Acceptance can reduce actual norms,
so actual norms are not claimed equal. Per-unit-norm first-order gains are
also reported. With no norm budget, kernel-attribution eligibility is false.
This is a constraint on trial steps along the direction, not a trust-ball
QP/KKT solution and not a change to the default H1 training algorithm.

Optional `compute_seconds` gives each correction the same wall-time allowance.
Checks occur at operation boundaries; an indivisible operation may overrun.
Overrun restores the state, logs SKIPPED_RESOURCE_BUDGET and is censored from
matched-compute conclusions rather than counted as method rejection. Actual
overshoot remains recorded. All arms report full-arm seconds_total; correction
seconds includes snapshots, kernel construction, projection, acceptance and
diagnostics. Native update_seconds measures forward/backward/optimizer only,
and its total includes reference diagnostics. Shared damping/preparation time
is a separate one-time cost, not a per-arm speed advantage or equal-time claim.

There are at least two random draws (default five), changing draw index but
not metric seed/damping. Random trace is normalized per site against tiny_V
with the same virtual dimensions. These are exploratory draws, not a
significance test. No automatic old-checkpoint conversion into BN exists.

H2 is a penalty variant with an explicitly supplied nonnegative feature
weight. Both solves change consistently, and feature drift/cosine is
measured on replay in eval mode separately from the train penalty. It is
not an implementation of LPR, nor an automatic H1 add-on. `standard_h2` is its
independent CIL entry point. Default lambda=1, layers=[features] (the pooled
ResNet32 representation). Fixed supported alternatives stage_1/2/3 use the
corresponding PyCIL `fmaps` flattened without rescaling; selected tensors are
concatenated. There are no persistent feature hooks. H2 native/random/GN controls
use the same metric and unconstrained total-loss acceptance. Lambda is a fixed
penalty, not a fitted multiplier or an asserted exact feature-drift budget.
Predicted train-feature drift, actual train-feature drift/linearization error,
and eval drift/cosine are reported separately. Both cotangent and correction
use the same feature penalty.

H3's `TangentReconfiguration` generates existing tiny_V incoming rows with
real-ReLU sign selection. It serially evaluates each row in every eligible live
and dead slot, plus identity, with fixed model capacity. Installation reuses
V3's outgoing-zero, SGD-coordinate and BN reset logic. It minimizes post-swap
window loss minus damped GN recovery over the candidates whose actual component
losses do not increase. Both removed and inserted contributions are included
by relinearizing the actual swapped network, with the same damping. This is a
net local surrogate, not the added-column Schur theorem. VNA gain is separately
measured; it does not enter an arbitrary weighted mixture with GN gain. The
validated winning state is committed once. No global optimality or whole-history
deadness is claimed.

The independent H3 runner uses pinned LoP slowly-changing regression: ReLU B5,
20 inputs, target features100, flipping bits15, beta0.7, 100 tasks of10000 online
samples, SGD0.01. Window128, intervention every10000 samples. It reports task
learning curves, first/last-quarter errors, learning speed/slope, gradient RMS,
replacement counts, local tangent diagnostics and runtime. Native BP/CBP/ReDo/S&P
adapters are separate arms. Effective CBP maturity10 follows upstream expr's
`mt` lookup, as recorded by the existing recipe. Rate1e-4 and S&P perturbation1e-6
are declared settings, not a tuned-baseline claim. Historical runners are untouched.

H4 defaults to five rotated (15-degree increments) or independently seeded
permuted MNIST domains, one epoch/domain, 128 incoming plus up to128 replay,
reservoir2000, fixed MLP width256/ten-class head, SGD0.01/momentum0.
Domain ID defines the data transform, never the inference classifier.
BN train/eval and frozen-BN diagnostics remain distinct. Report accuracy
for every seen domain, average domain accuracy, forgetting, BWT, buffer and
functional drift and cost. GroupNorm virtual construction is unsupported
and explicitly refuses; GroupNorm native controls are available.

## Required further evidence

Do not launch phase3/full20 or held-out-seed tables until numerical/toy and
pilot gates are actually run. Compatible strong GEM/A-GEM/ER-ACE/LPR
comparisons remain required before comparative claims. Existing competitor
implementations are not silently adapted or assigned equal-protocol labels;
their default schedules/memory/samplers may differ. No claimed superiority,
novelty, cost advantage or statistical confirmation is established here.


## Completion registration, before benchmark execution (2026-10-28)

`--loss-reduction pooled` preserves native mean CE weights even with unequal
incoming/replay counts. `--loss-reduction separate_mean` explicitly changes the
objective to the sum of subset CE means; KD remains the original whole-window
term. The latter is a different experiment identity, not an iCaRL reproduction.
Both components come from the same pooled BN forward. `--reference kd` instead
protects all CE and old KD separately; it is another distinct identity.

H0 now emits unconstrained tiny/native/random controls, a distinct GN control,
and constrained tiny/native controls, all from the same checkpoint/window/mu.
Acceptance groups remain separate. Standard-BN SGD and cloned-buffer references
are still unfiltered raw references. Fixed per-site virtual width is not a
matched empirical kernel-rank claim.

H4 logs exact finite eval changes at old/current running buffers, separating
weight-only and buffer-only output changes. These norms need not add and are
not a first-order protection theorem. The transform sequence, permutation
hashes or rotation angles, split sizes, data hashes and fixed head are recorded.

The strict collector supports H1–H4, retains failed outcomes and counts paired
completed seeds. Different code commits/objectives/normalizations/modes/budgets,
synthetic flags and base protocols stay separate. Seed-derived transform/data
fingerprints are kept in each run and checked for equality across arms of the
same seed before pooling seeds. H3 uses task error/plasticity metrics rather
than fabricated classification accuracy. Runtime/parameter/VRAM records remain
available per run. Adapted A-GEM/ER-ACE are separately labelled comparison
families; their normalization/sampling differences cannot establish a virtual
kernel-only benefit. LPR integration is blocked, not silently approximated.
