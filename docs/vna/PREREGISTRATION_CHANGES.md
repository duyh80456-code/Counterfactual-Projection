# Stage B measurement amendments (2026-10-15)

## B5 seed-0 preview and numerical repair (2026-10-18)

### Legacy authority schedule loading repair (2026-10-18)

The group notebooks previously rejected the real 8ea105f B5 preflight solely
because its commit predates the GradMax repair. The B5 authority arm is
bp_vna_gir, which does not use the repaired Euclidean GradMax eigensolver;
the regression FFNN has no aliased native layers affected by the MNIST fix.
Registered real v3 B5 calibrations are now accepted with their original
code_commit and values intact. Fresh jobs use the compatible repaired runner,
and protocol/preflight fields distinguish consumer code from calibration source.
This exception applies only to the registered B5 regression setup; MNIST,
iCaRL and online schedule commit checks retain their exact-match requirement.
The loader also reads schedule.json and embedded preflight artifacts from
protocol/summary files and prints rejection reasons instead of hiding them.

### Separate seed-1 and seed-2 notebooks (2026-10-18)

Superseded by three arm-group notebooks at the user's request. Groups A/B/C
each contain four distinct arms and both seeds 1 and 2: eight jobs/group,
24 jobs total, with four CPU workers per independent session. The partition
balances slower and faster arms using the available seed-0 task timings;
it does not change algorithms, seeds, horizons, rates or the shared schedule.
The two per-seed notebook files are removed; the seed-0 notebook is retained.
Each group restores only its own jobs from the same previous archives, records
source provenance and plots its four arms separately for seeds 1 and 2.

Each notebook runs the same 12 registered arms for its own seed and full
100-task horizon with four CPU workers. The four alternative CBP rate checks
remain seed-0 only; all main CBP arms keep replacement_rate=1e-3.
Both notebooks require the same real seed-0 authority schedule as Input
and reuse it with compatible repaired code for fresh trajectories, without
running another preflight. Existing completed results are skipped after validation;
unfinished checkpoints resume only for the requested seed, at their original
source commit. Missing checkpoints start new trajectories. The two proposed
additional GIR controls are not included in this scheduling-only update.

### Reuse of existing seed-0 work (2026-10-18)

The resume preview accepts old P-B5 archives and standalone resume.pkl files.
It selects a validated completed result, or the furthest compatible checkpoint,
for each arm/rate/seed-0 job. Original source directories and code identities
are retained. Incomplete trajectories execute their original pinned source
in a separate checkout, with the same upstream, schedule and checkpoint state.
No identity check is bypassed and no old result is relabelled with a new commit.
A full checkpoint lacking final artifacts replays its logged prefix without
optimizer steps to export the missing results. RESUME_APPLIED is logged only
after model, optimizer, RNG and hook state restoration actually returns.
Pre-repair GradMax trajectories are excluded and restarted with the repaired
implementation. The preview reports original commit/tag per row, so the
result is explicitly a provenance-labelled preview rather than a uniform
reproduction on the revised commit. The three-seed protocol is unchanged.

The seed-0 preview runs all 12 registered B5 arms and the four baseline CBP
rate checks for the full 100 tasks, with four CPU workers. It reports partial
progress separately from validated final scores; it does not replace the
three-seed experiment. Main CBP replacement_rate remains 1e-3.

GradMax obtains its Euclidean directions from the left singular vectors of
the cross-statistic N directly, with scalar normalization before SVD, rather
than eigendecomposing NN^T. These are the same nonzero eigendirections, up
to sign and bases within tied subspaces; the direct factor avoids the failed
Gram-matrix eigensolve observed in the earlier seed-0 run. GradMax also omits
the unused residual CG solve. GIR and VNA selection rules are unchanged.
New training uses a new run tag and code pin; reused trajectories keep their
original directories and provenance as described above. CPU heartbeat tolerates
an absent nvidia-smi. Existing checkpoint, deadline and archive handling remains.

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
17. **Preflight-only exact checks (2026-10-18, before any v3 results).** Full-run arms load and keep the calibrated `cg_tol`, `tol`, and `power_iters` without running exact checks.
18. **V4 algorithm (2026-10-19).** Theory §9b resolves the prompt’s ν-doubling wording: T26 failure and rejection use `mu <- 2*mu`, while checkpointed `nu` remains fixed at 2 for schema compatibility. The implementation records both fields and uses the geometric K0 schedule; no μ grid or exact run check remains.

## V4 phase 3 setup (2026-10-19, before any main results)

- CIFAR-100 B0 Inc5: 20 superclass tasks, five fine labels per task; both superclass
  order and within-superclass order permuted by seed. Random control retains PyCIL's
  seeded permutation. Root `/kaggle/input/datasets/nlnk1607/cifar100`.
- PyCIL iCaRL configuration, memory 2000 with `fixed_memory=False`, native SGD,
  weight decay, augmentation and T=2. First/later epochs 100/50; milestones scaled
  from 200/170 epochs to [30,60,85]/[24,35]. Initialization and data randomness use
  the paired run seed. BN ResNet-32 and SkipInit noBN (biased convs, zero residual
  scalar); parameter counts recorded after each task.
- Arms `icarl` and `icarl_full`; r=16, live candidate budget 4/site, geometric
  application gaps K0,2K0,4K0 within each task. Main full runs require the Phase 2
  backbone authority schedule, budget 1.0x baseline train time. No tuning here.
- `nobn_lr_scale` is 1 unless a task-1 NaN loss requires 0.5; the runner records
  `nobn_lr_halving_required` and a DIVERGED status. The notebook/runner must restart
  **all** noBN arms with the same recorded scale, in fresh output directories.
- CNN/NME total, old, new and learning accuracies; forgetting averaged over all
  seen fine classes using max strictly earlier accuracy minus current, including zero
  forgetting for newly seen classes; average incremental and final accuracy.
  Probe: deterministic seed-0 selection of 256 seen-class test samples (all samples
  if the tiny fake fixture has fewer); read-only public Phase 2 `diagnose()`.
- Feature accuracy at tasks 5/10/15/20: frozen features from all seen training
  samples under test transforms, multinomial linear probe, zero initialization,
  unregularized cross entropy, LBFGS with strong Wolfe and max_iter=100; eval on
  all seen test samples. This fixes unspecified probe optimizer details in §5.
- Paired seeds 0,1,2: preregister Q1 mean learning accuracy increases, Q2 final
  forgetting does not increase, Q3 average incremental accuracy increases. Predict
  a smaller full-versus-baseline benefit for random than superclass order. Report
  mean±sample std, paired deltas and seeds won separately for CNN/NME and setup.
- Trap checks (descriptive, never stop training): superclass/noBN iCaRL final dead
  or dormant >=10%, learning accuracy drops >=5pp from first to second half,
  final forgetting >10pp, and unreach increases over tasks.
- Phase 3 consumes the public Phase 2 API and does not implement T26–T28. Its
  local smoke substitutes a diagnose API spy until the Phase 2 branch is merged;
  no numerical diagnostic claim is made from that fixture.

## V4 phase 4 (2026-10-19, before competitor results)

Competitors use B0 Inc5, memory 2000, 100/50 epochs, seed-matched superclass
order and BN/SkipInit variants. Tune on seed 0 / noBN by CNN average
incremental accuracy; ties retain grid order. Fake fixtures never select a
configuration for real runs. Disabled competitors dispatch directly to iCaRL.
ReDo ports dopamine weight_recyclers.py (estimate_neuron_score,
NeuronRecycler.recycle_dead_neurons/create_masks, weight_reinit_random/zero,
reset_momentum), NHWC axes transposed to NCHW; only the first convolution of
residual branches is recycled, as in the upstream ResNet rule. CBP calls
ResGnT on those same unbranched conv_a -> conv_b pairs, supplying neutral
bias/BN compatibility objects where the backbone lacks them.
S&P ports warm_start/run.py:shrink_perturb: perturbations are a fresh native
initialization, not Gaussian parameter noise; fresh initialization buffers
are retained. UPGD calls FirstOrderGlobalUPGD from the official repository.
DISCREPANCY: label_permuted_cifar10.py:19 has two utility decays, not three.
The registered 3x3 grid uses the union with label_permuted_emnist.py:19:
{0.99,0.999,0.9999} x sigma {0.01,0.1,1.0}; it does not invent a fourth value.
C-Flat uses its original iCaRL training class and optimizer, rho=lamb=0.2;
reproduction uses random order, BN, seed 1993, 200/170 epochs, B0 Inc5.
Reference: 59.13% average accuracy, C-Flat/resources/cflat_results.png,
Table 1, iCaRL w/ C-Flat, CIFAR-100 B0_Inc5 (README.md:20 embeds image).
Its exps/icarl.json defaults to Inc10; reproduction overrides to Inc5 to
match that published column. Task-boundary checkpoints replay an interrupted
task from its initial model, method state and RNG, preserving optimizer
updates exactly without serializing ephemeral hooks or closures.
No real sweeps or long training runs are authorized by the local tiny-test rule.
PyCIL's unused convnet.fc is omitted from all v4 arms (including the baseline)
after initialization: its forward never uses it and GEM/UPGD assume every
listed parameter has a gradient. This changes parameter counting, not logits,
SGD updates or initialization RNG. PODNet retains its native cosine head;
its 50 later-task epochs comprise 30 native training + 20 classifier finetuning
(the one-epoch fixture disables the finetuning round). Feature probes use
frozen features, all seen train data, and a 100-iteration LBFGS CE linear head;
probe RNG, modes, buffers and training gradients are isolated.
The C-Flat reference image labels only average accuracy, without identifying
CNN or NME. The reproduction runner records both differences from 59.13%
and UNVERIFIED_HEAD until the published classifier is established. It never
chooses the closer head after observing results. B5 widths are 5/20/100 with
width-specific schedules; the launcher rejects legacy engine state without
persistent mu/nu and does not execute the old algorithm as a v4 fallback.
C-Flat reproduction retains trainer.py:_set_random's torch initialization
seed 1 and DataManager's original shuffle=True / seed-1993 NumPy permutation;
main matched runs seed initialization by the run seed. Both paths enable the
native deterministic cuDNN settings. The initialization seed is recorded.

## V4 integration and notebook registration (2026-10-19, before main results)

1. **Algorithm and cleanup.** The main full arm calls the merged public V3Engine
   T26–T28 implementation with fixed r=16 and four live candidates/site. Removed
   duplicate phase-4 setup/diagnose implementations; see CLEANUP.md for these
   and the earlier phase-1/2 deletions. The unused native backbone classifier is
   omitted consistently for main and competitor arms after native initialization.
2. **Main setup.** All arms share phase-3 class ordering, SkipInit conversion,
   100/50 epochs, [30,60,85]/[24,35] milestones, dynamic memory 2000, seed-0
   256-example seen-test probe, CE diagnostics, CNN/NME per-class metrics, and
   zero-initialized LBFGS feature probes. PODNet keeps its native cosine head and
   splits later-task training into 30 training + 20 classifier finetuning epochs.
   Synthetic tests alone use a four-example probe and pooled tiny residual model.
3. **Random control.** M3 uses noBN/random, paired seeds 0/1/2. Predict smaller
   full-minus-baseline benefit than superclass order. Reuse M1's noBN schedule
   when attached; otherwise calibrate on superclass data and record
   `schedule_source=self`. Collectors reject different backbone authority timings
   when combining results: rerun with one attached authority for pooled claims.
4. **Round-1 questions.** Paired by seed: Q1 mean learning accuracy(full) > base;
   Q2 final forgetting(full) <= base; Q3 average incremental accuracy(full) > base.
   Tables use mean ± sample standard deviation, paired deltas, wins and GPU-hour
   ratios separately for backbone/order and CNN/NME.
5. **Trap checks, descriptive only.** For superclass/noBN iCaRL, final dead or
   dormant >=10%; learning accuracy falls >=5 percentage points between sequence
   halves; final forgetting >10 points; unreach increases across tasks. Record
   outcomes without stopping a run.
6. **Competitor selection.** C1 tunes seed 0, noBN, superclass by CNN average
   incremental accuracy; ties retain grid order. CBP rates {1e-6,1e-5,1e-4,1e-3},
   maturity 100; ReDo tau {0,.025,.1} × reset epochs {1,5}; S&P shrink {.4,.8}
   × perturbation scale {1e-4,1e-2}; UPGD utility decay {.99,.999,.9999} × noise
   {.01,.1,1}. C3/C4 require C1 selections with matching source/dataset/grid
   provenance. PODNet/WA/GEM/C-Flat use native configurations without a sweep.
7. **Compute schedule.** Budget 1.0× ordinary iCaRL training time. K0 is the
   smallest measured feasible geometric gap; applications start at the first
   task iteration, then gaps K0, 2K0, 4K0. M1/M2 use the actual main setup for
   three read-only application measurements in each of tasks 1/2. Preflight
   trains one first-task epoch, then stops after task-2 measurement three; its
   projection uses the production 100/50-epoch task lengths. It runs neither an
   exact solver audit nor a damping search. BN and noBN have separate schedules.
8. **Damping initialization.** mu0 = .01 tr(F)/p (eight Hutchinson vectors) is
   initialization only, never tuned; record it from the engine protocol.

**Resolved theory/prompt discrepancies and earlier-phase clarifications:**
T26(d/f) failure/rejection doubles mu and keeps nu=2, rather than doubling nu;
T26(e) uses d^T F d, excluding damping, and infinity for zero denominator.
GIR biases include zero and the five registered quantiles. Production CG has
200 iterations, replacing the former 50. UPGD's registered three decay values
are the union of its two original experiment grids (CIFAR alone has two).
S&P uses the official fresh-native-initialization perturbation, rather than
inventing Gaussian noise. C-Flat reproduction overrides its Inc10 JSON to the
published B0 Inc5 column, retains torch initialization seed 1 and class-order
seed 1993, and records both CNN/NME deltas to 59.13%; the reference image does
not identify the head, so its gate remains UNVERIFIED_HEAD. Feature-probe
optimizer/initialization and all-seen-class forgetting follow phase 3's
registered definitions. No new algorithm formula changes were introduced.

**GradMax:** SVD/eigh NaNs follow exploding model parameters; bias-inclusive row
norms feed back into larger installed units. Keep its original formula, fall
back to CBP initialization on invalid proposals, log `gir_fallback`, and log
nonfinite intervention parameters/step with DIVERGED status (CLEANUP.md).

Nine v4 notebooks M1/M2/M3/C1/C2/C3/C4/C5/B5w are generated with one
`REPO_COMMIT=PLACEHOLDER` argument for supervisor repinning, one lane per T4,
existing exact resume/deadline/exit-75/archive handling, unified result tables,
flag rates and figures. B5 is regression: its figure explicitly labels native
last-50 online error / negative task loss AUC as stability/plasticity proxies;
class forgetting/learning accuracy are undefined there. Historical B5 resume
notebooks retain their separate original-source restoration path. No notebook
or real training was executed in this phase.

## Cleanup 3 clarification (2026-10-19)

 (theory wins; record each in PREREGISTRATION_CHANGES.md)
1. T26(d) failure branch: theory says `mu <- 2*mu` when rho <= 0 (and T26(f) rejection counts as
   rho < 0). Prompt §2.2 uses the Madsen nu-doubling. Implement theory: `mu <- 2*mu`; keep `nu`
   in state/logs fixed at 2 for schema compatibility. Success branch identical in both.
2. T26(e): s_F* = g_hat / (d_hat^T F d_hat) (F, not F+mu); infinity if denominator 0.
3. GIR bias set: prompt §2.4 quantiles {0.1,0.3,0.5,0.7,0.9} plus 0 — consistent with theory §10.4.
4. CG max iterations 200 (prompt §2.1) replaces the old 50 (training solver amendment 2026-10-17).
Any further discrepancy you find: do not guess; implement the theory and list it in your final
report under "DISCREPANCIES".

## GradMax NaN root cause (already diagnosed by the supervisor; for §2.7 / CLEANUP.md)
Reproduced locally: `b5_cbp_gradmax` seed 0 crashes at ~step 1.7k (Kaggle seed 2 at ~30k) with
`linalg.svd: input matrix contained non-finite values`. SVD is NOT the cause: all four model
parameter tensors are already NaN before `prepare_gir` runs. Weights blow up across installs
(layer-0 bias max 4 -> 47 -> 1066 -> 3.6e3 -> 3.4e8 -> 3.8e31 -> NaN; output weights 1 -> 877 ->
3490 -> 2e9). An installed GradMax unit had |w|=20, bias=-47 versus native CBP init |w|~0.8:
`rows_proposal` scales new rows to the mean row norm of [W, b] (bias included) and the bias then
gets `-median(projected)` added, so large biases feed back into larger new units; large hidden
activations then make plain SGD on the output layer diverge. The earlier `eigh` failure (pre
3c41ecf) had the same cause. §2.7 fallback alone therefore cannot rescue a run whose weights are
NaN: in addition to the §2.7 guard, detect non-finite parameters after each intervention, log
`nonfinite_params=true` with the step, and mark the run status `DIVERGED` in summary/status
(no crash, prompt §0.4). Do not change the GradMax formula (prompt §0.1).


The revised §1 supersedes Cleanup 2: old B5 resume code/notebooks are deleted here; in-flight Kaggle runs use their independently pinned old sources. Entrypoints also include main_cil, v4_competitors, v4_b5_width (added after the snapshot). Nine vna4_ notebooks use PLACEHOLDER for supervisor repinning. V4_PRERUN_REPORT and pilot_v3_summary are additional retained deliverables. Historical local synthetic pilot numbers are preserved in pilot_v3_summary; completed v3 B5/O1 Kaggle numbers are absent locally.

### Round-3 code preparation (2026-10-19, not run in rounds 1–2)

- **Own ablations** are `experiments/main_cil.py` arms sharing the main setup, the backbone K0
  authority schedule, r=16 and everything else of `icarl_full`; each changes one component:
  `icarl_gir` (GIR only), `icarl_vna` (VNA only), `icarl_full_beta0` (protected split kept,
  beta fixed at 0 instead of BetaRule), `icarl_full_Arand` (random virtual A, same trace),
  `icarl_kernel_self` (kappa = K, T19(c)), and GIR initialization variants on the same slots and
  the same T28 swap rule: `icarl_gir_random` (seeded Gaussian incoming rows),
  `icarl_gir_gradmax` (Euclidean top singular directions of N on the full residual e, theory
  §12.2 "reset kiểu GradMax: hướng theo gradient trên e thay vì r_perp"), `icarl_gir_tiny`
  (TINY's generalized eigenproblem (N N^T, S_x) on e instead of r_perp^beta; TINY's small
  amplitude gamma is not used, the swap's B_v = 0 already preserves the function).
- **FIRE** (`icarl_fire`): unmodified `FIRE/vision/interventions/fire.py:fire`, iter_num 10
  (their default), applied at the start of every task after the first, matching their
  "new data arrives" trigger.
- **SWR** (`icarl_swr`): unmodified `SelectiveWeightReinitialization` with the authors'
  incremental-CIFAR config (gradient utility, threshold pruning, reinit_freq 128,
  reinit_factor 5e-4, resample); `reimplemented=true` only for the per-parameter init table,
  ported from their ViT/MLP helper to PyCIL resnet32. Rebuilt per task (growing head), step
  counter carried across tasks.
- **MRFA** pinned but not wired (separate PyCIL fork). **Forget-Forgetting**: no public code.
