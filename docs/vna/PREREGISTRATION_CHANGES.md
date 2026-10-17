# Stage B measurement amendments (2026-10-15)

## Training solver amendment (2026-10-17, before the revised pilots)

Training-time VNA and NGD corrections in B3 and B5 use matrix-free conjugate
gradient with tolerance `1e-3`, at most 50 iterations, and warm starts from
the previous solve of the same system. Warm starts are cached per arm/site
set and reset when a task changes dimensions. The Stage-B measurement path
continues to use its exact output-space solve; the formula for `d` is
unchanged.

TINY-V, random-Frobenius, and Euclidean-TINY selection matrices are built at
task start and after every ten applied VNA steps. Between refreshes only `z`
and `d` are recomputed. Each correction records `A_refreshed` in
`vna_log.csv`.

The registered K set is `{50, 100, 200, 400, 800}` union
`{1, 2, 5, 10} * iterations_per_epoch`. The smallest K whose weighted VNA
cost fits the measured training budget is used, with one reselection weighted
at 1/10 and a reused-A computation weighted at 9/10. If no K fits, the
largest registered K is used and the schedule records `budget_exceeded: true`.
The schedule stores separate with-A and without-A timings for tasks 1 and 2,
their weighted means, the extrapolation, and the selected K. NGD uses the
same solver, K, window, and step-wise d target as VNA.

Because training no longer builds the output-space matrix M, the damping
mu = 0.01 * tr(F) / n_out uses the 8-probe Hutchinson estimate of tr(F)
(the registered `GGNConfig` estimator) instead of the exact trace of M.
The with-A/without-A preflight timings come from two solves at the same
step; the second is timing-only (no parameter update, no log row, no
application count). A is reselected once at least ten VNA steps have been
applied since the last reselection, so rho-gated skips do not force repeated
reselection.

Round 2 changes the N1 gate statistic from rho = V / ||J^T e||² to
nu = V / ||e||². Permuting residuals preserves ||e||² but can reduce
||J^T e||², inflating null rho. We retain rho and lambda p-values as diagnostics;
real and null rows now also report V, e_norm2, nu and nu p-values for N1/N2.

This decision follows inspection of regression B5 data and precedes all CIFAR
measurement data. Regression B5 will be rerun for all six arms and reported
under the new gate; the old regression gate is superseded.

The listed 55, 60, …, 100 grid supplies twenty second-half points. Task 51
is added (first and mid) to satisfy the simultaneous requirement of 22.
The second-half B5 schedule now provides at least twenty points (22 for the
100-task regression). Fewer than ten points in the gate window is INVALID.
B3 measures the middle of all ten tasks and the end of tasks 2, 5 and 10;
end-of-task points remain reference-only.

Solver selection is auto: exact output space for n_out <= 4096, CG otherwise.
This is a computational choice with the same damping and mathematical target.

## GIR/VNA performance pilot (2026-10-16, before pilot outcomes)

B5 uses post-SGD corrections every K=100 updates on the current task's last
W=128 examples. The two fixed coefficients are 0.1 and 1.0; mu_rel=0.01,
mu=mu_rel*tr(F)/n_out, rank r=1, rho_min=0. The existing exact OutputFactor
and joint_correction implement z and d. NGD uses (F+mu I)^-1 g, matched to
the concurrently computed VNA norm. There is no Stage-B measurement pass.
GIR pools hold at most eight positive generalized eigenvectors per layer,
refresh on task change, exhaustion or age 1000 steps, and consume each row
once. The small regression network refreshes lazily at replacement events.
Directions 1..m already consumed from the current GIR pool are reserved; subsequent r directions
for VNA; if no unreserved direction exists, record a skipped correction.
A task boundary invalidates pools; their first usable refresh uses available
current-task observations, never observations from the preceding task.

Regression's pinned CBP config specifies a list of five replacement rates,
not one scalar. Retain all five (1e-6, 1e-5, 1e-4, 1e-3, 1e-2) for each CBP
arm and report each rate separately; do not select a rate using pilot outcomes.
Seeds 0,1,2 share deterministic data and initialization streams across arms.
Upstream expr reads mt (default 10), not maturity_threshold (100 in JSON);
preserve that effective upstream behavior.

Reset norms are sqrt(E||w||^2): LoP uniform(-b,b) gives b*sqrt(fan_in/3)
with zero bias. PyCIL's conv initialization is normal with variance
2/(kernel_height*kernel_width*out_channels), so its expected squared row
norm is fan_in*2/(kernel_height*kernel_width*out_channels). Random-direction
controls use the same row norm. Random fill follows the upstream distribution.
GIR consumes upstream LoP initialization draws before substituting its rows,
retaining selection, mean-activation bias correction, utility, ages and
optimizer handling. A separate upstream-random identity mode substitutes
nothing and is tested bitwise.

B3 performance windows sample 128 seeded examples without replacement from
exactly the task training dataset (current classes plus exemplar memory).
CE plus native iCaRL KD (T=2, known_classes) and eval BN apply during
corrections, with previous modes restored. r=4 at all fifteen sites,
lambda=1, mu_rel=.01, rho_min=0. Dormancy uses mean absolute post-ReLU
activation divided by its layer mean, threshold <=.1, before the first epoch
of tasks >=2. Reset conv_a rows, zero conv_b input channels and reset the
corresponding BN running means/variances and affine values to initialization.

B3 quick is exploratory, decided before any CIFAR pilot data: 128 samples,
64 stat/64 eval, midpoint tasks 2,5,10; mu_rel=.01; rank16; the final block
of each stage; unchanged eight N1 and eight N2 draws and exemplar/new groups.
It uses the full measurement code with reduced configuration. Correction
verification and exact/CG comparison run only at the first point (task2).
Only gate_quick.json is written, labeled exploratory, with the point minimum
waived. The official B3 gate remains full and is deferred to a later session.
Smoke measures quick at tasks 1 and 2 and linearly extrapolates to 6400
outputs at task10. Warn above 600 seconds per point and continue.

Before full B3 correction runs, real two-task smoke writes schedule.json and
appends the measured K decision below. The extended K set and weighted budget
rule are recorded in the training-solver amendment above. If no registered K
fits, the largest one is used with `budget_exceeded: true`. Shared schedule
and windows apply to NGD.
Fake CPU calibration is labeled synthetic and never registers a real K.

## Euclidean TINY diagnostic (2026-10-17, before parallel runs)

After the B3 quick result showed `tiny_V` below its matched random control,
add `tiny_V_euclid` as a diagnostic selection variant.  It reuses exactly the
same N statistic as TINY but selects the leading eigenvectors of N N^T,
maximizing V under the Euclidean norm of A rather than the generalized S norm.
It uses the same true-ReLU sign choice and incoming-neuron mean-norm
normalization.  This post-hoc diagnostic is reported separately and does not
replace the preregistered `tiny_V` arm.

## Parallel full-run scheduling (2026-10-17, before parallel runs)

The six-account prompt removes the separate smoke phase.  Correction arms
therefore calibrate inside the full run, before observing task outcomes: the
first post-SGD correction of each task is timed, and the smallest K in
{50,100,200,400,800,iterations_per_epoch} satisfying the existing projected
cost rule is used for the remainder of that task.  The calibration correction
is itself the first VNA/NGD update, so no extra optimization step is introduced.
Each task's decision and timing are written to the event stream and every
correction row records the active K.
