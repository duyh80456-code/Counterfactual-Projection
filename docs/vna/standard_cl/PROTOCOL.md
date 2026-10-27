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

H0's `compare_local` uses an unchanged supplied checkpoint/window and fixed
metric seed. Controls are tiny_V, native K, GN, actual optimizer and at least
two random draws (default five). Random draw index changes A, not damping
probe seed. Normalize random trace per site against tiny_V with the same
virtual dimensions. These are exploratory draws, not a significance test.
No automatic old-checkpoint validation/conversion into a BN learner exists.

H2 is a penalty variant with an explicitly supplied nonnegative feature
weight. Both solves change consistently, and feature drift/cosine is
measured on replay in eval mode separately from the train penalty. It is
not an implementation of LPR, nor an automatic H1 add-on.

H3 optimizes over caller-supplied, fixed-capacity, in-place candidates and
identity. A supplied GN surrogate can rank dead replacements whose immediate
loss is unchanged. All candidates also face actual component-loss checks.
Measure live output jump and optimizer reset; no global optimality beyond
that finite set or whole-history deadness is claimed. Full new candidate
generation and standard plasticity benchmark integration remain conditional
on evidence of a suitable failure mode. Existing CBP/ReDo/S&P runners remain
untouched, and their old numbers are not relabelled as new H3 results.

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
