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

### Pre-run review fixes (2026-10-19, before any v4 result)

- **LM failure rule now follows the spec** (supersedes item 18 above): rho <= 0 or an exact-check
  rejection gives `mu <- mu*nu, nu <- 2*nu`; rho > 0 gives `mu <- mu*max(1/3, 1-(2rho-1)^3),
  nu <- 2`. rho is measured at S1 on x_GN for every application with a trustworthy split
  (before the S4 sign filter), as in prompt §2.2 step 2.
- **GIR is independent of the VNA certificate (T28):** swaps run unless the S1 split itself is
  unusable (`cg_z`, non-finite x_GN). Negative certificate, invalid beta or untrusted rho only
  skip the VNA step.
- **ell_beta uses H_beta (T27)** for lambda_max(J_v^T H_beta J_v).
- **`icarl_gir` ablation** runs the same T28 search as `icarl_full` with beta = 0
  (e_beta = e, H_beta = H, r_perp from the S1 split); no VNA step.
- **Cost (no formula change):** lambda_E and lambda_max(J_v^T H_beta J_v) by matrix-free power
  iteration (20 and 10 steps, dedicated generators) instead of explicit Jacobians; s_v = J_v^T r
  by one VJP. Gamma (one CG) is solved only for the best `gamma_candidates`=2 proposals by
  ell_beta (logged in protocol); live-slot exact jumps are checked before any proposal is built;
  q_hat*(J) is recomputed only after an accepted swap. lambda_E is therefore a lower estimate
  (Rayleigh quotient); the logged T25 ratio V/(lambda_E B) can exceed 1 by that error.
- **Preflight times a complete application** (VNA step, exact check, GIR swaps) and restores
  model, optimizer, RNG and engine state bitwise, so K0 includes GIR cost.
- **Schedule:** no forced application at the last iteration of a task; once the next geometric
  gap exceeds the remaining iterations, the task gets no further application.
- **Runtime compute-budget guard (§2.5, schedule only):** after each application the engine
  compares cumulative algorithm seconds with iterations x baseline seconds/iteration (schedule
  `iteration_seconds` for CIL; live per-iteration timing for B5/PM); above the 1.0x budget the
  next gap is doubled once more and `budget_guard=true`, `budget_ratio` are logged. This is
  needed because K0 is calibrated on tasks 1–2, where GIR is cheapest. Being wall-clock based,
  it makes the realized schedule hardware-dependent (resume restores its state; the realized
  K sequence is logged per application).
- **Uncertified beta is not used by GIR:** with `invalid_beta` the VNA step is skipped and GIR
  uses beta = 0 (e_beta = e, H_beta = H).
- S_x, N and the site input are computed once per site per application (unchanged by B_v = 0
  swaps). `icarl_gir` applications are no longer logged as skipped.
- GEM was run locally with quadprog (2 tasks x 1 epoch, enabled): completes.

### noBN learning-rate rule and GIR cost fixes (2026-10-19, after the first M1 attempt, before any v4 result)

- **Observed:** in the first M1 attempt every noBN run (all arms, including baseline `icarl`)
  produced a non-finite loss in epoch 1–2 of task 2 at lr 0.1; the old rule only reacted to a
  task-1 NaN. Those runs are discarded (no result was used).
- **Rule (replaces the "halve lr on a task-1 NaN" rule of §3):** lr_noBN = 0.1 x scale with the
  largest scale in {1, 0.5, 0.2, 0.1} (lr 0.1, 0.05, 0.02, 0.01) such that baseline `icarl`,
  noBN, superclass order, seed 0, has a finite training loss for the first 3 tasks. Chosen once
  as the first step of notebook `vna4_M1` (four `--max-tasks 3` probes, two per GPU, status
  PROBE/DIVERGED, never results), written to `_schedules/nobn_lr.json` in M1's output (other
  noBN notebooks attach M1's output; a noBN notebook started without it runs the same probe
  itself, and the collector rejects results that mix noBN lr scales), and passed as
  `--nobn-lr-scale` to every noBN run of
  every arm and competitor (including the M1 K0 preflight and the C1 sweeps). BN runs keep the
  original PyCIL lr.
- **GIR cost (no formula change).** In the first M1 attempt one application took ~2000 s: with
  SkipInit alpha = 0 every unit of every block has f_u = 0, so all ~560 units were treated as
  dead and the candidate set was rebuilt per unit, although s_v = J_v^T r = 0 and no swap can
  happen. Now: (i) the candidate set (a_v, B_v = 0) is built once per site and rebuilt only
  after an accepted swap (next direction S_x-orthogonal; at most the ~4 generalized directions
  per site per application); (ii) a site with s_v = 0 for every candidate is skipped before any
  slot work; (iii) dead slots swap on ell_beta > 0 without solving Gamma, since T28(b) gives
  Gamma = gain >= ell_beta (logged as `gain_lower_bound`, `Gamma=None`); live slots keep the
  exact-jump check and the top-2 Gamma solve.
- **Gamma CG warm start (no formula change):** the live-slot Gamma solve warm-starts from the
  base q_hat*(J) solution of the same window (the temporarily installed system differs in one
  slot); the CG warm-start guard restarts from zero if that start is worse. Local CPU timing on
  real PyCIL resnet32 noBN (batch 16): alpha = 0 application 94.5 s with GIR 0 s (every site
  skipped, s_v = 0); the remaining time is the S1 x_GN CG solve. Production timing comes from
  the M1 preflight on T4.
- **noBN lr fixed (2026-10-19):** the 3-task baseline probe selected scale 0.5, i.e. lr 0.05
  for every noBN run (init_lr = lrate = 0.05; BN runs keep 0.1). The value is now a registered
  constant (`NOBN_LR_SCALE = 0.5` in the notebook builder, passed as `--nobn-lr-scale 0.5` to every
  noBN job, preflight and sweep); the in-notebook probe step and `--max-tasks` were removed.


## T29–T31 exact solver amendment (2026-10-20, before numerical tests)

- Branch `vna-v4-exact`, base `d88abc4`. Replace truncated parameter CG in the engine and diagnose with the exact output Gram solver T29. This restores the stated mathematical model; it is not outcome tuning. Other users of `projection/cg.py` retain CG.
- Replace between-application damping with the inner LM loop T30 (40 trials); one accepted mu_star is used throughout VNA, BetaRule, residual and GIR. Old M1 damping lagged the observed curvature by approximately 10^8.
- Compute GIR Gamma in one frozen linear model with sequential Gram/residual updates and Woodbury removal/addition (T31); retain top-2 proposals and four live slots. Real loss/KD jumps still use the sequential real network.
- Reject nonfinite rho; restore trial state and roll back nonfinite application outcomes. Stop divergent training with explicit FAILED_DIVERGED metadata and retain the previous safe epoch checkpoint.
- Remove reach/unreach from current reports. B, V, lambda_E, V/(lambda_E B), norm_w2 and mu_diag remain diagnostic logs.
- Old d88abc4 M1 seed-0 full results are diagnostic evidence, not comparison data. Baseline reuse requires epoch-by-epoch loss identity through the first two tasks; this has not yet been checked. No M1 rerun is authorized before review of the profiling/NaN report.
- Keep noBN lr=0.05, B0 Inc5, 20 tasks, epoch/milestone/memory/superclass settings unchanged. No fallback, Gram jitter or PSD projection is introduced. A 12800-output Gram exceeding 600 seconds or CUDA memory stops profiling for user review.
- Source discrepancy: the supplied PDF prints the new source section s9c_exact as section 11, whereas the prompt names section 9c; formulas T29–T31/R1 agree. LM trial loss necessarily needs J^T y at each trial, in addition to the eventual real parameter step; no parameter-space solve is used.
- Local float64 tests follow prompt section 8 only. T4 profiling and A–D experiments must report actual GPU measurements, never local timing projections as T4 results.

- Capacity discrepancy found in existing main code: the fixed diagnostic probe has 256 samples, so task-20 diagnosis has m=25600, not the engine window's m=12800. Keep the probe and its validation unchanged; apply the same 600-second resource stop to m>=12800, and report actual GPU memory/time before any main run. No unapproved probe reduction or alternative solver.
- The supplied archive confirms task-2 mu=1.3814802088148892e-7 (the task-1 value was 6.907401044074446e-8); the prompt's approximate 6.9e-8 describes task 1. Preserve the checkpoint's carried mu when replaying D.


## 2026-10-21: historical checkout bootstrap repair

The Kaggle profile-B setup used a shared clone of a filtered checkout, leaving missing d88abc4 blobs and falling back to current source imports. Fetch the historical SHA directly with authenticated lazy blob checkout in an independent clone; rebuild incomplete/shared source caches, verify a clean historical tree, and validate all core import origins before GPU work. Record module paths with the diagnostic report. This changes source preparation/validation only; T29–T31, checkpoint data, SGD, RNG, schedules, comparison rules and the M1 hold are unchanged.


## 2026-10-21: discover the diagnostic checkpoint by metadata

Remove the exact-filename/single-archive requirement. Scan attached input directories (following mounted symlink directories without cycles), loose .pt/.pth files and TAR/ZIP members. Select only the registered d88abc4 full/noBN/superclass/seed-0 checkpoint with completed=1, mid=False, epoch=100. Deduplicate identical checkpoint bytes by SHA256; reject distinct compatible states instead of choosing arbitrarily. All A–D branches consume the same selected raw checkpoint bytes. No training, RNG, solver or schedule change.


## 2026-10-21: two-GPU diagnostic execution requested by the user

The user explicitly requests two-GPU diagnostic execution, replacing the initial serial scheduling requirement. Use independent subprocess lanes with physical GPU isolation via the existing CUDA_VISIBLE_DEVICES helper. With two GPUs, profile B/D concurrently; for NaN experiments assign A,C to GPU0 and B,D to GPU1. With one GPU retain serial B,D or A,B,C,D. Prepare the historical checkout and select the shared checkpoint once before launching either lane. Preserve per-branch output directories, validation, RNG isolation, safe resume, exit-75/deadline handling and the M1 hold. Log concurrency and visible GPU identity in each report so timing context is explicit. No solver, training recipe or K0 change.

## 2026-10-21: bound derivative batch memory after the T4 OOM

The actual profile_D traceback fails in the first 64-vector batched VJP used
to build the Gram: 12.77 GiB allocated and 14.47 GiB total process usage on a
14.56-GiB T4. At task 2, N=128 and C=10 give a 12.5-MiB float64 Gram; the
failure occurs in derivative intermediates rather than final matrix storage.

Select chunk=4 instead of 64 in the shared Gram and GIR Jacobian-column
helpers, as allowed by exact-solver prompt section 4.1. Record gram_chunk
in Gram metadata. This bounds concurrent derivative vectors without changing
the window, columns, precision, exact formulas, solver, or GPU lane mapping.
Keep the existing resource stop rules; introduce no OOM retry or approximation.
Actual T4 memory and timing after this change still require measurement.

## 2026-10-21: native GIR product dtype repair

The profile_D traceback reaches GIR curvature estimation after Gram construction,
then fails because the native JEv output is float32 while OutputSolve.metric
stores float64 blocks. Convert the input vector to the block dtype/device in
the shared metric method before einsum. The metric result remains float64;
JET already converts its input to native parameter dtype at the VJP boundary.
No formulas, model precision, windows, RNG, counters or lane scheduling change.

## 2026-10-21: exact-solver implementation speed pass, before M1

The user approved replacing GIR's ten power iterations with the exact largest
symmetric eigenvalue of J_v^T H_beta J_v, and explicitly accepts reporting
ranking/swap changes. This is not promised to reproduce finite-iteration
power estimates. The T27 score formula, shared mu_star, frozen theta and H_beta,
sequential G/residual updates, actual L/KD gates, top-2 proposals and four live
slots remain unchanged. No candidate limits are lifted.

Candidate features are concatenated as independent virtual channels with zero
outgoing weights in one training_products context per site preparation. One VJP obtains
all candidate gradients; vmap JVP chunks interleave groups of four candidates, then
float64 eigvalsh gives its exact curvature. Only the top-2 candidate Jacobians
are retained and reused for Gamma. Native slot Jacobians use vmap chunks of
16, with explicit native-column and Gamma timing/call accounting.

Gram vector blocks increase from 4 to 64. Because the actual 64-vector VJP
already OOMed on T4, backward and forward microbatches are limited to 16, pending T4 memory
validation. These three sizes are logged separately; 64 simultaneous VJPs are not
reintroduced. All products/columns and float64 G are retained. No OOM retry or
approximate solver is added to training. The m>=12800 / 600-second stop stays.

The TASK log already omitted reach before this pass; a regression check now
covers it. Scalar loss logging detaches tensors without changing the loss.

The profile notebook now compares reference D at b97d25c with optimized D on
the same task-1 checkpoint, on two isolated GPU lanes with serial fallback.
Reports are namespaced by the optimized code pin, retaining strict checkpoint
and source identities and prior outputs. Reports compare mu_star, rho, L/KD,
rankings, ordered Gamma and swap slots, and record final actual L/KD after GIR.
A 1.0x K0 estimate includes sample count and a labelled constant-task-2-cost
20-task scenario; it is not an authoritative schedule or final-size cost model.

A separate read-only capacity measurement extends a cloned head to 100 classes
and repeats the window for probes 256/128. It records actual allocation and
reserved peaks, Gram times at m=25600/12800, OOM/resource stops and proxy
provenance. This is not actual task-20 data. The main diagnostic probe remains
256 until hardware results justify a preregistered reduction to 128. M1 stays
held. No T4 speed/VRAM target or numerical equivalence on T4 is claimed locally.

## 2026-10-21: separate speed-only power10 validation from exact curvature

Add --curvature {power10,exact} to main_cil and exact diagnostics; default exact.
M1 commands and authority preflights explicitly select exact. Curvature is
recorded in the main protocol, schedule identity and engine checkpoint state;
mode-mismatched schedules/resume identities cannot be reused. A nondefault
power10 main experiment has distinct comparison metadata, while default exact
retains compatibility with the registered competitor comparison protocol.

power10 uses the original native JEv -> H_beta -> JET action, invokes
power_lambda_max with iterations=10 and the same dedicated generator purpose
v4-ell:{task}:{iteration}:{accepted_proposal_count} as b97d25c. Preserve that
implementation's early exit for zero/nonfinite action norms and its accepted
proposal counter after skips; do not replace the native action with a dense
small-matrix power approximation. Only top-2 candidate Jacobians are built
for reuse by Gamma in this mode. The exact mode retains grouped Jacobians
and exact eigvalsh for the existing T27 lower-bound ranking formula.

MODE=profile is now a two-stage validation on the identical task-1 checkpoint:
1. R (unchanged b97d25c D, GPU0) versus optimized D-power10 (GPU1), or serial
   on one GPU. Require relative tolerance 1e-5, absolute tolerance zero for
   numerical mu_star, rho, L/KD, candidate scores/curvatures and Gamma; compare
   generator seeds/iterations exactly, full candidate ordering, installed
   candidate parameters, swap slots and final L/KD. Candidate IDs identify
   incoming direction ordinal, sign and bias quantile, not floating byte hashes.
   Write stage1_power10_vs_b97d25c.json and stop on FAIL/incomplete results.
2. Only after PASS, run D-exact from the same original checkpoint and write
   stage2_exact_vs_power10.json, including actual trajectory/swap/Gamma/final
   L/KD changes. Audit native power10 on every accepted D-exact candidate at
   that exact frozen state. Report each lambda error (signed and relative),
   top-{1,2,4,8} overlap and Kendall tau-b of ell rankings on that same set.
   Report undefined tau as null (ties/singleton), not an invented numeric value.

The extra exact-run audit is diagnostic-only and does not select candidates,
mutate model/optimizer/global RNG or provide cached Jacobians. Record its
wall time separately and subtract it from application/prepare/GIR timings
used for speed comparisons and provisional K0 estimates. Report the unadjusted
wall time too. D-exact performs the existing separately labelled capacity
proxy after the application. Main runs do not enable this audit. Keep top-2,
four live slots, probe=256 and the M1 hold until real T4 report review. Local
reference checks are not a substitute for the requested T4 runs.


## 2026-10-21: independent exact and power10 profiling lanes

Supersedes the strict stage-1 gate above. In MODE=profile, GPU0 immediately
runs current D-exact with --capacity-profile. GPU1 runs clean b97d25c D
(power10), then current D-power10, on the same discovered task-1 checkpoint.
Keep the relative tolerance 1e-5 strict comparison and both comparison reports.
Power10 FAIL, missing report or child-process failure is advisory: record it
explicitly without gating exact or the M1 decision. Exact failures remain fatal
after reports/output are archived. One GPU runs exact first, then both power10
checks serially. Deadline/exit-75 pauses still stop remaining launches.

The same-state exact curvature audit, capacity proxy, M1 exact setting and
M1 hold are unchanged. Actual T4 timing/equivalence/capacity remain unmeasured
locally. The notebook embeds the new dispatch and pins the already committed
solver source; this scheduling change does not modify solver mathematics.

## 2026-10-21: residual-verified matrix-free CG after supplied T4 OOM

Supersedes production Gram solves and the exact-curvature default in earlier
entries. Supplied notebook `vna4-exact-diagnostics (4).ipynb` shows proxy
m=25600 OOM and m=12800 GRAM_LIMIT=602.763 s; actual task 20 was not run.
Current power10=1142.843 s/application; exact=3240.354 s after audit subtraction.
Strict power10 comparison FAIL remains recorded; see CG_OOM_REVIEW.md.

Engine and diagnose use the SPD parameter system J^T M J+mu I with native
network products (float32 on T4), float64 iterates/reductions, residual tolerance
1e-8 and a total 200-iteration budget per system. Verify true residual again in
caller units. Refinement is internal to the same solve/iteration budget; no
cross-solve warm state, hidden fallback, looser tolerance or dtype substitution.
Nonconverged LM trials fail and advance the same mu/nu sequence; unsuccessful
post-LM solves restore the entire application. Failed diagnostic solves log a
null B/explicit flag. Dense Gram and signed Woodbury remain small references.

Gamma now uses CG for frozen J' via coordinate masking/replacement. Native
slot columns are no longer materialized on the CG path; prior replacements
are updated correctly at shared/adjacent sites. Preserve mu*, BetaRule,
trust-region/window checks and candidate/slot limits. power10 is the only engine
ranking mode, with the same dedicated generator and ten power iterations.
Estimated ell_beta is a ranking score, not a certified spectral lower bound.
Numerical residual bounds do not promise bitwise equality or identical swaps.

The exact-diagnostics notebook now compares current CG/power10 (GPU0) with
8c56d02 Gram/power10 (GPU1), same task-1 source and first task-2 application.
Capacity reports cover both 128/256 probes at 100-class shape, using mu*;
application and capacity peaks are separately recorded. Keep probe=256.
MODE=smoke3 is an explicit fresh three-task baseline/full seed-0 rehearsal,
with fresh CG authority preflight, native 100/50/50 epochs and separate
max_tasks=3 identity. This reinstates --max-tasks only for the requested
rehearsal; default M1 remains 20 tasks. Deadline/exit-75 still governs launches.
M1 remains held; compute-matched budget/competitor/BN/order experiments follow
only after the requested real profile and rehearsal review.

## 2026-10-22: true-double CG-IR, shared downstream and local warm starts

Supersedes the previous native-residual/power10-only/no-cross-solve-warm policy.
The selected frozen problem is A64=J64^T M J64+mu I, b64=J64^T c: current native
weights are promoted into one actual double snapshot. Frozen M/c are unchanged;
replacement columns are recomputed in double. Outer relative residual 1e-10,
inner float32 CG correction target 1e-4, at most 200 total CG iterations and
8 corrections, with explicit stagnation/failure handling. Double-native problems
use the outer target directly. Nonconverged LM trials remain failures; post-LM
failure rolls back the application. Never loosen tolerance to accept a result.

Warm starts/caches exist only within one application: adjacent mu/beta,
LM -> z, baseline -> Gamma and accepted swaps -> next residual. Version/RHS
identity invalidates cached solves. No warm state enters engine checkpoints or
crosses application boundaries. Native SGD, RNG isolation, exact resume,
coefficient-zero identity, deadlines/exit-75 and real-loss guards remain required.

For sample-separable noBN models, construct one float64 downstream D_site at a
time. Reuse it for candidate gradients and Jacobian columns, and compute the
exact small lambda_max(J_v^T H_beta J_v) by eigvalsh. Default curvature=exact;
BN/stochastic/cross-sample-coupled models fall back to full products/power10.
A VJP coupling check supplements the BN/dropout exclusion. Power10 remains an
explicit reference mode with its existing generator and ten iterations; noBN
may reuse D only for its double Gamma columns without changing power10 ranking.
Exact ranking can change swaps. Candidate grid, top-2/four-live limits and probe
256 remain unchanged. No output Gram/native J_u columns/Woodbury in production.

Reports include true-double residual histories, correction iterations/timing,
warm resets/cache hits, per-mu numerical vs model failures, double D bytes/time
and suffix VJP counts. Gamma's bilinear solve-error bound is diagnostic-only;
there is no extra acceptance gate and no universal margin versus near-zero Gamma.
At N128/C100/stage1, double D alone is 1,677,721,600 bytes. Neither constant
runtime across tasks nor the proposed 300–350 s/application has been measured.

Profile lanes: GPU0 current IR/exact + capacity; GPU1 clean 8c56d02 Gram/power10,
then current IR/power10. Preserve strict 1e-5 comparison in
ir_power10_vs_gram.json and separate curvature comparison in
ir_exact_vs_power10.json. Retain failures and per-mu reference-rho audit;
positive-reference numeric rejections and unknown trials prevent an unsupported
claim that mu* is unaffected. Capacity adds first-stage double D alongside
native/double frozen VJP graphs at N128/C100, in addition to diagnose at
m12800/m25600. This is still a proxy, not actual task-20 training or a full
application memory proof. Reference failures are advisory, primary IR failure
fatal after archiving. Three-task rehearsal and M1 hold remain unchanged.

Theory and limits: IR_DOWNSTREAM_REVIEW.md, theory/s9b_closure.tex and
theory/s9c_exact.tex. A sufficient contraction assumption is
(delta+eta*kappa)/(1-delta)<1, not kappa*u32<1 alone. No arbitrary small-mu
cutoff and no claim of bitwise equality across different warm starts.

## 2026-10-22: stream downstream by sample blocks

Supersedes retaining a whole site's D and constructing candidate J_v while
ranking. Downstream now uses fixed blocks of eight samples. Within each block,
reuse D^T r and accumulate s_v and K_v=J_v^T H_beta J_v for all candidates in
float64, then release D and block Jacobians before constructing the next block.
Metric blocks retain the original full-window loss normalization. Dense full-
window J_v is allocated only when that candidate actually reaches a Gamma trial,
by streaming another pass. The power10 fallback likewise defers dense candidate
columns until Gamma; its score/generator/ten iterations are unchanged.

The frozen Downstream object retains model/inputs and small validation state,
not D. One full-window random-seed suffix VJP checks block separability; it does
not construct full D and its graph is freed before block processing. Batch-
coupled models retain the existing full-products fallback. This is a numerical
reduction-order change, not a promise of bitwise identity to the full-D sum.

Log bytes as peak D block bytes, block_samples, block count, total suffix VJPs
and retains_full_D=false. Capacity profiling streams the same blocks alongside
native/double frozen VJP graphs. At N128/C100/stage1 (O16,32x32), D allocation
is bounded by 104,857,600 bytes (100 MiB) instead of 1,677,721,600 bytes; this
is a D-only bound, not total process VRAM. Recomputing blocks for scoring/Gamma
may change runtime; measure on T4. No change to candidate limits, mu/IR policy,
loss/window checks, warm scope, resume, K0 authority or M1 hold.

## 2026-10-22: same-commit Gram M1s0 and CG-IR profile

Explicitly authorizes the new vna4_gram_m1s0_ir notebook's Gram-only M1s0 lane;
other M1 notebooks and CG-IR M1 remain held. Both active lanes pin one source
commit containing --solver gram|cg_ir. Solver identity is included in protocol,
preflight authority and engine state; cross-backend schedules/resume are refused.

Gram uses the 8c56d02 output Gram/OutputSolve/signed Woodbury math, its power10
score/generator and grouped top-2 derivative construction. No IR or D_site is
used in its engine. Remove the 600 s Gram-construction stop, retain all real-
loss/window checks, finite checks, native checkpoints and deadline/exit-75.
Preflight derives K0 from actual Gram full-application costs at budget 1.0x.
The new lane runs icarl_full, noBN, lr scale .5, superclass, seed0, 20 tasks,
registered 100/50 epochs. Native resume is automatic (no separate resume flag).

Task-end dead/dormant still use the seeded 256-sample test probe. B/norm_w use
CG-IR on the first 128 samples of that same probe, with that 128-sample loss
normalization. Protocol and diagnostic records label b_size=128,
b_solver=cg_ir, activity_probe_size=256. This changes diagnostic B values, not
SGD, engine loss/window or active intervention batch. Historical reused baseline
B values retain their old provenance and are not relabelled as this diagnostic.

GPU1 first profiles clean 8c56d02 Gram/power10 and current Gram/power10 from the
same d88abc4 task-1 checkpoint (first task-2 application). Strict 1e-5 relative,
zero absolute comparison must PASS before GPU0 begins full training. Gram K0
preflight may overlap this check. GPU1 then profiles current CG-IR/exact with
capacity at m12800/m25600, streamed D capacity, gamma_precision, solve/timer logs
and lm_numeric_audit, against that same-commit Gram result. This strict second
comparison permits the intentional curvature-mode difference but keeps all
numerical/ranking/swap checks; it can legitimately FAIL. Exact vs power10 is
not misrepresented as solver-only equivalence. GPU1 stops after this profile;
its failure does not cancel the validated Gram lane. Archive each stage and
every 15 minutes, with separate per-lane/job logs.

After full completes, GPU0 starts a fresh native baseline replay. Reuse the
completed d88abc4 seed0 baseline only if all 100+50 epoch losses match exactly
(full-precision checkpoint values, no rounded log comparison). Validate data
fingerprint, order/recipe/upstream/seed and old 20-task artifacts. Record every
mismatch and hashes in baseline_loss_check.json/reused_baseline.json. PASS stops
the replay and references the historical 20-task result without changing its
commit identity; the 2-task verification is never labelled a final baseline.
FAIL continues the fresh run to 20 tasks. Missing/invalid reference runs a fresh
full baseline. Resume identity includes reference digest; references cannot be
silently substituted in a resumed run. Current epochs.json is saved atomically
alongside native checkpoints; it is evidence, not a replacement resume format.

## 2026-10-22: dead-only GIR, direct Schur gain, rho >= 1/4, CIL task >= 2

Explicit user-requested algorithm revision `dead_schur_rho25_v1`, applying to
both solver backends. Native GIR uses dead slots by default; `--gir-live`
retains live-slot trials only as a labelled ablation. Existing candidate
limits remain. Deadness must hold on the frozen and current post-VNA loss
window, and on the CG-IR double target; the seeded 256-sample probe comparison
is observational. No rule based on probe agreement is introduced.

Dead gain is evaluated directly as 0.5 s^T S^-1 s, using new-coordinate
coefficients of the augmented solve, without subtracting q and q'. Accept only
finite positive gain strictly exceeding `linear_solve_error_bound`, derived
from both verified linear-system residuals. Existing actual-loss/KD and finite
checks remain. The bound covers the represented frozen linear system, not
Jacobian/Gram assembly rounding or future SGD; derivation and logs are in
DEAD_SCHUR_POLICY_REVIEW.md. Live ablation retains the old q-difference path.

LM accepts only finite rho >= 0.25 and shares its accepted mu_star with GIR.
This is a deliberately stricter acceptance policy: Nocedal–Wright Algorithm
4.1's radius-shrink threshold is 1/4, but its eta acceptance threshold may be
smaller. No claim of a universal 1/4 acceptance requirement is made.

Main CIL skips interventions and intervention preflight throughout task 1.
Its K0 now uses three real task-2 points; task1 ordinary SGD, optimizer and RNG
remain baseline-identical. LoP regression/MNIST task1 is not disabled. This
change does not implement or silently authorize cross-protocol checkpoint
import. New engine checkpoints and Main schedules carry policy/live identity;
old policy cannot resume under the new one. Historical task1 full checkpoints
can seed diagnostic theta, with fresh new-policy engine state.

Log 10-step GGN spectral estimates before/after each attempted application,
window-vs-probe dead-mask disagreement, and 20 subsequent native SGD losses
for each effective Main CIL application. Followups are checkpointed with
step-based duplicate protection; incomplete task-end followups are labelled.
These logs introduce no extra stability acceptance rule. Diagnostic profiles
record immediate final L/KD and then 20 native SGD losses before completion.
Observational timing is separate from K0/correction budget; wall-clock and
exit-75/deadline accounting still include it.

The already pinned a9e04b9 notebook/source retains its historical policy and
8c56d02 compatibility gate. This entry does not repin that notebook or waive
its gate. The requested changes intentionally preclude a new-policy whole-
application identity claim against 8c56d02; fixed-system Gram direction math
remains checked against that source. New-policy strict profiles require equal
policy/live identity. No T4 profile or M1 outcome is asserted here.


### Follow-up: gate live-slot Gamma by its solve error as well

Policy `schur_gamma_bound_rho25_v2` replaces the preceding v1 identity. With
`--gir-live`, require finite positive Gamma strictly exceeding the sum of the
base and swapped q linear-solve error bounds, for both Gram and CG-IR. Dead
slots retain their direct Schur bound. Log the two q bounds; do not add a
stability threshold. This gives the ablation the same numerical acceptance
principle. The bound is for represented frozen systems, excluding jump and
operator-assembly rounding. Main preflight remains three task-2 points, with
no task-1 intervention points. Focused tests cover perturbed-solution gain
error <= bound and rejection of positive live Gamma below its bound.


## 2026-10-22: refresh every vna4 notebook for the current policy

Explicit user request to update all notebooks/builders after the v2 engine
commit. All generated notebooks pin one immutable implementation commit;
no source ZIP is needed. Runtime pin precedes the notebook packaging commit.
The dedicated Gram M1s0 notebook remains the authorized 20-task seed0 route,
with Gram/power10 preflight on three task2 points, native resume and unchanged
baseline 150-epoch verification. GPU1 profiles same-commit Gram/power10 with
a strict fixed-system 8c56d02 direction/OutputSolve check, then CG-IR/exact with
capacity and stops. Training waits for the completed current Gram profile and
that fixed-system PASS. The old whole-application gate cannot apply across the
requested LM/GIR policy change; this replacement explicitly narrows the claim.

The reference check reuses the exact same frozen Gram, theta, residual/metric,
proposals and accepted damping, at rel tol 1e-5/abs tol 0. Reference source must
be a clean 8c56d02 checkout, and historical OutputSolve is loaded from it.
The check is read-only and separately timed as diagnostic audit overhead.
Its status/fields/scope are logged. It does not assert LM/swap equivalence.
No profile mismatch or pending report is reclassified as a whole-application
PASS. Current-policy CG-IR/exact vs Gram/power10 stays a strict, explicit
comparison permitting only the labelled curvature change; it does not gate
the authorized Gram lane or authorize CG-IR M1.

Standalone diagnostic GPU1 uses current-policy Gram/power10 plus the fixed
reference check, then current CG-IR/power10. Thus strict solver-only comparison
no longer mixes policy versions; GPU0 still runs CG-IR/exact/capacity.
Matching completed diagnostic identities are reused, with fixed PASS checked
again. Native K0/checkpoint policy checks remain. Generic outputs are isolated
as outputs/<source-commit-prefix>/<RUN_TAG>/<job>; M3 has its own random-order
preflight and schedule filename. Existing generic M1/M1s0 holds are preserved.
No T4 runtime/capacity or M1 performance result is asserted by this refresh.


## Metric-trace OOM repair and narrow continuation from 29cc735

The uploaded Gram full-run traceback fails in dead_gain's vmap(metric) at
allocation of 5.49 GiB, not in Gram construction. Candidate-batched metric
application can materialize candidate x sample x class x class intermediates.
Both Gram and CG-IR dead gains now share metric_trace: reshape replacement
columns as sample x class x coordinate, multiply the actual metric blocks by
bmm over at most 16 samples, and accumulate the scalar trace. The frozen
Schur formula, LM/GIR policy, solve tolerance and candidate limits are unchanged.
Floating-point summation order changes; equality is checked to 1e-12 relative
on dense metrics, including m12800/m25600 and coordinate count576. At C100,
k576 the output temporary is at most 16*100*576 doubles (~7.03 MiB), rather
than the 5.49 GiB broadcast in the report; this is not total process VRAM.

The dedicated Gram notebook can transfer only unfinished icarl_full, seed0,
noBN/superclass, Gram/power10 from source29cc735, policy
schur_gamma_bound_rho25_v2. It copies into the new commit namespace and leaves
the original untouched. Full native learner/optimizer/scheduler/RNG/hook state
is preserved; only implementation identity, schedule consumer commit and the
protocol artifact are adjusted. Old K0/timing remain unchanged and retain
calibration_code_commit=29cc735. The transfer manifest records both commits,
source checkpoint hashes, completed tasks/epoch and the original schedule hash.
Main's exact protocol validation still runs before any training; changed
recipe/data/order/seed/policy/solver cannot pass it. Missing old checkpoint
starts normally; an existing new target is not overwritten. Completed old
results and old diagnostic PASS/timing are not imported as current results.
Current-source profile gates rerun unless a matching current report exists.
This is a declared exception for this memory-only fix, not general cross-
commit compatibility. Native continuation may replay work since the last safe
epoch checkpoint. No hardware runtime or universal OOM guarantee is asserted.


## 2026-10-23: preserve metric dtype/device in blocked trace

metric_trace now casts columns with columns.to(blocks) and creates its scalar
accumulator with blocks.new_zeros(()), matching the metric's dtype/device.
The engine's existing blocks.double() contract makes accumulation fp64.
This does not promote an arbitrary standalone fp32 metric to fp64. Sample
blocking and the Schur formula remain unchanged. A mixed fp32-column/fp64-
metric test checks bmm dtype, block size and agreement with a dense reference
computed from the same quantized fp32 columns cast to double.

All notebook pins are refreshed. The narrowly registered checkpoint continuation
now also accepts 1bf6d7a (the preceding memory-only repair), in addition to
29cc735. If both are attached, select the greatest saved native hook step
count, preferring the newer registered source on a tie. Keep the original
calibration_code_commit and K0, model/optimizer/scheduler/RNG/hook state and
source files; retain existing strict recipe/seed/policy checks and migration
hashes. This permits continuity through the dtype-only repair without general
cross-commit checkpoint acceptance. Local affected validation: 77 tests PASS;
no T4 run or real failing-session archive validation was performed.

## 2026-10-23: consistent Gram M1 seeds and joint competitor LR selection

This entry supersedes earlier generic M1/C2/C3 dispatch and sweep counts.
M1 is Gram/power10, noBN/superclass, lr=0.05, budget 1.0x. Its notebook
runs only full and baseline seeds 1/2 (four jobs), without MAIN RUN HELD.
It imports the frozen Gram seed0 K0 from attached output, preserving calibration
producer identity; missing or conflicting authorities fail instead of recalibrating.
The dedicated gram_m1s0_ir notebook remains the seed0 native continuation path,
including the 1bf6d7a bounded-memory repair and 80608ab dtype repair. Registered
unfinished seed0 sources are 29cc735, 1bf6d7a and 80608ab. Model, optimizer,
scheduler, RNG, counters and original K0 retain the preceding transfer contract.
80608ab already contains columns.to(blocks) and blocks.new_zeros(()).

User decision: C1 and C2 both select noBN lr from {0.1, 0.05, 0.02, 0.01}.
Each method-config/LR pair runs the complete seed0 20-task protocol. Maximize
finite CNN average incremental accuracy; ties take the first registered LR,
then first registered method config. Diverged/invalid/nonfinite trials cannot
win. Persist per-trial status/score and the winning LR/config/path/provenance.
This is the same LR grid, not equal total tuning compute: C1 has 92 trials
(4 LRs times 4 CBP + 6 ReDo + 4 S&P + 9 UPGD configs); C2 has 16 trials
(4 LRs times PODNet, WA, GEM and C-Flat, one config each).

C2 contains no BN jobs. It finishes its sweep stage on two GPU lanes before
launching matched seeds1/2 with the selected LR/config. The winning seed0 trial
supplies seed0: no redundant matched seed0 job. An attached completed GEM seed0
result can supply only its identical LR/config candidate, after recipe, dataset,
order, training and all result artifacts pass validation. Its producer identity
is retained with source hash and consumer metadata. It cannot replace another
LR trial; if another LR wins, the earlier GEM result is excluded from the main
aggregate. C3 similarly runs eight matched jobs (four methods, seeds1/2), using
current C1 best_config.json; C1 winners supply seed0. C4/C5/M2/M3/B5w and exact
diagnostics remain available as deferred experiments.

scripts.collect_cil_main accepts expanded outputs and tar/zip archives. It
validates full 20-task artifacts, uses only selected sweep winners, checks
summary/final-task consistency and rejects conflicting duplicate seeds or mixed
solver/curvature/policy/K0/LR/config within a method. Producer commits remain in
provenance; registered formula-preserving memory repairs can be combined when
the complete mathematical configuration agrees. Outputs: CNN/NME A-bar and last
accuracy CSV/Markdown, sample standard deviation, taskwise A-bar CSV and PNG/SVG
curves. Incomplete seed coverage is explicitly PARTIAL_SEEDS; n=1 has no estimated
std. BN reproduction results are excluded from this main table.

No new real T4 training, runtime estimate or actual seed0 archive validation is
claimed by these local implementation checks.

## 2026-10-23: user override — fixed noBN learning rate for every method

Supersedes the preceding four-LR selection plan. Every new noBN run uses
nobn_lr_scale=0.5 (lr=0.05), including main arms, competitors, preflight and
profiles. C1 selects only the registered method configurations at this fixed LR:
23 seed0 trials total. C2 has one fixed-LR seed0 candidate per method (four
total), followed by the same eight seed1/2 jobs. There is no LR sweep. Selected
configuration artifacts must declare lr_scales=[0.5]; artifacts from the earlier
four-LR selection cannot silently change the fixed LR. The main collector
rejects noBN results at any other scale. BN retains its upstream training LR.

The command-line default is 0.5 for noBN. Main BN normalizes the inapplicable
noBN scale to 1.0. Historical validators retain the original rate metadata;
existing checkpoint identity and K0 are not rewritten to change learning rates.
The formula-preserving Gram seed0 continuation also registers source359da84
so its unfinished lr0.05 run can resume with original calibration provenance.

## 2026-10-23: S&P expanded-head device ordering repair

Before applying S&P in CompetitorHook.install, move the original network to
learner._device, matching the device move in native iCaRL _train. On later tasks,
update_fc has just created a CPU head while the backbone remains on CUDA; the
hook runs before native _train and must perform that move before S&P arithmetic.
Fresh-network initialization, S&P coefficients, task cadence and optimizer
creation remain unchanged. No RNG draws are added. CPU execution is unaffected.
No DataLoader worker change is included. Gram seed0 source336dded is also
registered for the existing strict native continuation, with lr0.05 and frozen
K0; this S&P-only fix does not change Gram training or solver formulas.

# 2026-10-26 — task-1 live GIR pilot (registered before execution)

Base: 6be654310c2e4123391fd7d8d31716d25c89c6fb. Exploratory amendment motivated
by the reported d88abc4 seed-0 log: VNA skipped (cg_z/untrusted), 29 live swaps
at iteration 1025, and subsequent Gram continuation from that task-1 checkpoint.
This is a new run, not a reproduction of that checkpoint or its swap count.

Flag `--gir-live-task1` defaults off. Task 1 alone enables interventions and the
d88abc4 T28 selection: zero function-contribution slots first, then the four
smallest nonzero H-norm contributions; exact CE removal must not increase loss;
test the two highest ell proposals and accept strictly positive Gamma. Dead
slots retain the old positive-ell lower-bound shortcut (Gamma is not measured).
There is no KD in task 1. Live Gamma uses the actual candidate network Jacobian
and the same pre-VNA CE metric/residual, solved with Gram rather than old CG.
GIR can run after a skipped VNA when that separate Gram system is finite and
solvable; a failed GIR factorization is logged and does not install its trial.
Task 2 onward retains 6be6543 policy schur_gamma_bound_rho25_v2 unchanged.

Fixed K0=512 each task, geometric cadence and compute-budget guard remain.
The guard uses measured native SGD iteration time, excluding intervention time.
Backbone noBN SkipInit ResNet32, superclass Inc5, memory 2000, LR .05,
100/50 epochs, milestones 30/60/85 and 24/35, no warmup, no clipping.
N=8 planned seeds 0..7; this separate pilot notebook runs only 0 and 1,
one full arm per GPU. No baseline/competitor reruns or new K0 preflight.
Loss NaN/Inf is reported FAILED_DIVERGED without replacing seeds.
New output tag; record elapsed GPU-hours and per-event task-1 live/dead counts.
No source overlay. CPU tests only for the new selection, isolation and defaults;
GPU performance and the historical 29-swap observation are not validated here.

## 2026-10-27 — task-1 GIR only for --gir-live-task1

Supersedes the task-1 VNA attempt in the preceding pilot amendment. Historical
rationale: original s0 had no accepted VNA step in task 1 (cg_z/untrusted),
but did accept live GIR swaps. This observation motivates the design; it is
not a claim that the new Gram run reproduces its exact trajectory.

With --gir-live-task1, task 1 performs GIR only. Compute its native Gram,
CE split/residual and damping, but do not run the VNA LM search, selection,
certificate or parameter update. Log s=0 and vna_applied=false. Preserve the
historical live/dead GIR rule and Gram candidate solves. Task 2 onward keeps
the existing full VNA+GIR path unchanged; the flag-off path is also unchanged.
K0=512, power10, compute-budget guard and planned seeds remain unchanged.
Record task1_gir_policy=d88abc4_T28_gir_only_gram and use a new output tag
vna4_m1_gir_only_task1_s01. Do not resume pre-amendment live-GIR checkpoints:
they may contain an accepted task-1 VNA step. Baselines/competitors are reused.
