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


## VNA–GIR v3 decisive pilot (2026-10-18T03:13:29+07:00)

Registration scope: before any full training results from the new v3 VNA or
`kernel_self` implementations. Historical two-epoch VNA/NGD smoke results
already exist and are disclosed below; they are not full v3 training results.
The following setup supersedes the old P1–P6 GIR/VNA setup. Competitors
P7–P10 retain their upstream algorithms.

1. **Training solver.** Matrix-free parameter-space CG, relative true residual
   against ||b||, tolerance 1e-3, maximum 50 iterations, task/site/system warm
   starts. First three applications of tasks 1 and 2 also compute CG tol 1e-8
   checks; cosine below .99 tightens training tol to 1e-4. Measurement remains
   exact output-space factorization. μ=max(μ_rel tr(F)/n_out,1e-12), eight
   Hutchinson probes; preflight evaluates μ_rel in {.001,.01,.1} and logs
   failed selection criteria explicitly. Full runs consume the fixed shared
   calibration artifact, starting from fresh native initialization/RNG.
2. **Joint correction.** One global Jacobian and summed virtual kernels across
   selected sites replace independent per-site correction solves. Equality of
   joint and separate formulas is checked with identical scope and damping.
3. **Push-through.** Exact measurement z uses R(M+μI)^−1 R K e, without
   pseudo-inverse or residual-range exclusion. Range error remains diagnostic.
4. **Iteration schedule.** First application is after iteration one of each
   task. K_next=round(K_base clip(.3/rho,.25,4)), at least one, capped at
   remaining iterations; skips use K_base. K_base is the smallest candidate
   in {50,100,200,400,800} union {1,2,5,10} times task-2 iterations/epoch
   fitting projected correction time per K <= baseline iteration time.
   Infeasibility selects the maximum candidate with budget_exceeded=true.
   Task-2 mean cost is extrapolated linearly in class count; raw timings are
   retained. Families share the authority VNA schedule; isolated Kaggle
   notebooks consume its archived artifact instead of recalibrating per arm.
5. **Step s — user decision 2026-10-18.** Main rule is
   min(s*,lr||g||/||d||), s*=gᵀd/[dᵀ(F+μI)d]. B5 _lf variants use
   min(s*,K lr/(1−momentum)). This explicitly overrides unified_gn.md §5's
   preference for λ-faithful as the primary setup, following the pilot prompt.
   Compute s only after certificates; nonpositive gᵀd yields s=0.
6. **Protection and certificate.** Implement v3 repair/new split, χ² from
   student probabilities, β-rule branches a/b/c and four log bisection steps,
   alternating A update, direct gᵀd and g_Oᵀd, outside-momentum correction,
   no correction weight decay, train-mode frozen BN buffers. Per user decision:
   “τ_h, τ_g là ước lượng; chuẩn phổ bị ước lượng thấp nên quy tắc bỏ qua
   chặt hơn (bảo thủ). Bảo đảm KD tăng bậc một ≤ ε·λ·τ_dùng đúng chính xác
   cho mọi bước được áp dụng, vì ĥ tính chính xác trên vector áp dụng. Điều
   có thể mất là tính chất ‘không bỏ nhầm bước có h* ≥ 0’ của T18.”
   Use five power iterations times 1.1; log skip rate and false skips when
   h_exact>=0. Above 10% false skips in preflight use 20 power iterations and
   retain the transition in the artifact. No claim of a rigorous upper bound
   for these estimated spectral norms is made.
7. **Kernel control.** New kernel_self is exactly K_E→K, d=(FG)^2g, with
   common certificate and step rule. Audit found that the old output-exact NGD
   branch used Gg, so this is an implemented formula correction, not merely a
   relabeling of the historical control. Historical results remain separate.
8. **GIR.** Cotangent is r_perp=r−z(r), r=e for B5/PM and r=e_old for KD.
   Check exact dead slots after the correction, valid conv1 slots only, S=1
   for standalone GIR. New outgoing weights are zero; reset corresponding
   optimizer elements to zero and BN slot statistics/affine parameters.
   ReLU candidates use four leading directions, both signs and median shifts;
   remove zero features, score the actual virtual Jacobian. CBP retains native
   selection/cadence/RNG draws and changes only initialization. For small-q
   multiple insertions deflate the complement. Historical B5 TINY 10/13 result
   used z, not the v3 complement. LIVE_SWAP remains off.
9. **Controls.** Add native-selection cbp_gradmax (Euclidean N_e eigvecs)
   and per-site exact virtual-kernel trace matched random control. With BN,
   trace matching scales the virtual feature/kernel, because scaling incoming
   a alone is cancelled by BN; the resulting kernel scale is logged.
10. **Settings and decision criteria.** Prioritize B5, then small Permuted
    MNIST, then official CCL-DC online CIFAR-100, then offline PyCIL iCaRL.
    B5 Q-A/B require all three paired seed wins; PM Q-A/B/C require >=2/3;
    offline CIL Q-A/B require >=.5pp CNN average; online require >=.5pp final
    and learning accuracy. GIR must beat CBP and GradMax >=2/3 seeds. β
    protection compares forgetting and stability-gap depth/recovery. Single
    CIL seed is pilot evidence only; no post-result hyperparameter changes.
11. **Disclosed prior observations, unused for selection.** Two-epoch smoke
    CNN VNA15.05/NGD21.3; β=0 quick rho_joint .26/.42/.58 and tiny_V losing
    random 7/9 rows; unified_gn toy results. consistency_setup.md was referenced
    but not supplied; its content/results are not fabricated or re-created.
12. **Metrics and architecture.** Main tables include anytime AUC, learning
    accuracy, iterations to half initial new-task loss, final performance,
    paired effects and cost. Offline continual evaluation: epochs1–10 then
    every10, old/new class accuracy and stability gap. Follow new-slot output
    contributions at 1/5/20 tasks. B5/PM MLPs remain plain ReLU; PM uses three
    hidden layers of ten units and full native task horizon. Online retains
    native backbone, with nf20 variants only where supported upstream.
13. **Removed from this pilot.** tiny_V_euclid is diagnostic only, LIVE_SWAP
    off. No full-run review hold follows successful preflight. Optional C-Flat
    may remain unintegrated and Q3 GPU1 empty, as the prompt permits.
14. **CBP rate — user decision 2026-10-18.** All B5 CBP-family arms and all
    three seeds use replacement_rate=1e-3. Add seed0 baseline checks
    b5_cbp__rate1e-6__s0, b5_cbp__rate1e-5__s0,
    b5_cbp__rate1e-4__s0, b5_cbp__rate1e-2__s0: total40 jobs.
    Record rate and check identities in protocol.json. If a check beats1e-3,
    report it and never change the selected rate during this pilot.
15. **Loss normalization.** B5 uses the native torch mean squared error:
    residual and H have coefficient2/N for a scalar output, not1/N. This
    resolves the pilot shorthand H=I/N by following authoritative loss
    derivatives and the upstream objective without altering it.
16. **Provenance (amended 2026-10-18, before any run).** The v3 code is
    committed on `vna-v3-pilot`; notebooks pin that commit directly, so no
    source overlay is used (source_digest = null). Notebooks without the
    authority schedule run their unscheduled baselines and skip scheduled arms
    until the Q2/O1 archive is attached; they never recalibrate.

## V3 fixes (2026-10-18, before any v3 results)

The B5/PM V3 intervention window is now the FIFO of at most 128 samples from
the current task. It is reset at every task boundary, and the first
application is delayed until at least 32 current-task samples are present;
subsequent applications retain the registered K schedule. GIR-only `gir`
applications compute only the split and GIR replacement. `kernel_self` uses
the same protected β-rule and `d_rep + d_N` split as VNA, differing only by
using K instead of K_E; only `vna_b0` and settings without old classes are
unprotected. Exact checks use up to 2000 CG iterations and affect adaptive
solver counters only when every exact solve converges.
