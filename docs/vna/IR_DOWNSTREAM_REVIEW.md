# CG-IR, shared downstream and application-local warm starts

## Analysis of the proposal

All three optimizations are useful, with these qualifications:

1. The previous solver checked residual using native float32 JVP/VJP. Tightening
   that residual cannot remove the operator error. CG-IR now evaluates RHS and
   residual using a real double snapshot of the frozen network, including double
   replacement columns after swaps. Casting old float32 output to double is
   insufficient. The output metric M and cotangent c remain the frozen inputs of
   the selected problem; the target is explicitly A64=J64^T M J64+mu I and
   b64=J64^T c. Native SGD, real-loss rho and window checks keep their native model.
2. Carson–Higham supplies a mixed-precision IR framework, not an unconditional
   guarantee for an arbitrary matrix-free CG correction. Let delta=||A64^-1||||E||
   for A32=A64+E and let eta bound the correction's relative residual. A sufficient
   contraction bound is (delta+eta*kappa)/(1-delta)<1. With a genuinely close
   operator, kappa around seven and eta=1e-4, a few outer corrections are plausible.
   They must be measured. ReLU mask differences can also make the native/double
   Jacobians differ more than a simple roundoff estimate.
3. For lambda_max=232, kappa≈1+232/mu. The threshold kappa*u32≈1, using
   u32=2^-24, corresponds to mu≈1.38e-5, not 2e-3. A larger empirical safety
   threshold may reflect operator error/constants, but is not that identity.
   No hard mu cutoff is introduced. Numerical failure is logged separately from
   measured negative/nonfinite rho. Reference comparison flags numerical skips
   at positive reference rho and mu values absent from the reference trajectory.
   This can support a statement about that checkpoint, not all future tasks.
4. A verified relative residual does not universally imply relative q error of
   the same size or a margin of 1e5 against every Gamma. Use the mu/RHS-dependent
   bilinear error bound in the existing lemma. Diagnostics log its ratio to
   |Gamma|; it is not an added decision gate. Errors in the supplied frozen
   metric, residual or finite output shift are outside that linear-solve bound.
5. J_v=D_site Phi_v is the exact chain rule at B_v=0. It supports exact small
   eigenvalues on a sample-separable network. Changing power10 to eigvalsh changes
   a numerical ranking approximation; it can change ranks/swaps. A missing BN
   alone does not prove separability. The implementation excludes BN/dropout and
   checks a random-seed VJP identity, falling back to full products if it fails.
6. At N=128, C=100, stage-1 O=16 and 32x32 spatial shape, D has 209,715,200
   entries: 838,860,800 bytes in float32, **1,677,721,600 bytes in float64**.
   True double replacement columns use double D. The updated implementation
   retains only an eight-sample block (100 MiB at this stage-1 shape), then
   discards it. A pass costs C*ceil(N/8) suffix VJPs, plus one full-window
   random-seed VJP when first checking separability; no full D is stored.
   The 50 s, 300–350 s and task-independent cost figures are unverified estimates.
7. Warm start changes the initial iterate, not the exact linear problem. Finite
   tolerance solutions need not be bitwise identical across starts. Cache/warm
   state is application-local and absent from engine checkpoints. Native exact
   resume still follows the same deterministic sequence within a run.

Primary source: Carson & Higham (2018),
https://research.manchester.ac.uk/en/publications/accelerating-the-solution-of-linear-systems-by-iterative-refineme/
DOI: 10.1137/17M1140819. The CG adaptation's sufficient bound is derived in the
updated theory text; it is not attributed verbatim to the LU/GMRES-IR paper.

## Supplied run evidence and OOM scope

The downloaded `vna4-exact-diagnostics (4).ipynb` predates CG-IR. Its
8c56d02 capacity proxy recorded CUDA OOM at N256/m25600 while building Gram
(allocated peak 14,659,228,160 bytes); N128/m12800 hit GRAM_LIMIT at
602.763 s. These were repeated-window/extended-head probes, not actual task 20.
The measured power10 application was 1142.843 s with prepare_gir 658.358 s;
exact was 3240.354 s excluding its separate audit, with prepare_gir 2757.444 s.
Thus both the quadratic Gram allocation and repeated candidate derivatives
are real targets. The old strict power10 comparison also failed; it is not
proof that later implementations preserve every discrete decision.

Matrix-free CG removes the global m-by-m allocation, but double frozen graphs,
replacement columns and D blocks still consume memory. The new N128/C100 D
capacity phase checks this additional cost. OOM is not declared eliminated
until the new reports and a complete application at the intended shape pass.
See CG_OOM_REVIEW.md for the historical report extraction and limitations.

## Implementation

- One double frozen snapshot per native operator/application; all high residual
  matvecs use its real double JVP/VJP. Float32 native products correct the
  solution with inner tolerance 1e-4. Float64 native models solve directly at
  the outer target. Outer relative tolerance is 1e-10, with 200 total CG
  iterations, at most eight corrections, and a stagnation check. Failure is
  explicit; tolerances are never loosened to force a result.
- Warm starts cover adjacent LM mu values, accepted LM -> z, adjacent beta
  values, current baseline -> candidate Gamma, and accepted swap -> next solve.
  Repeated identical systems reuse verified results. Span/metric/RHS versions
  invalidate caches. None of this state is stored in V3Engine.
- The noBN exact path streams eight-sample D blocks, computing D^T r once
  per block/residual. For every candidate it accumulates s_v and K_v in double
  with the original full-window metric normalization, then discards the block
  and its candidate Jacobians. Small eigvalsh gives lambda_max only after all
  sample blocks are summed. No full-window J_v is built during ranking.
  Only candidates actually tested for Gamma construct dense double J_v, by
  replaying the blocks; this deferral also covers power10 native columns.
  The native incoming/outgoing column materialization for J_u was already
  removed by the previous coordinate-masking CG implementation and remains
  absent. No Woodbury and no output Gram return.
- BN/sample-coupled fallback retains full products and power10 ranking. For
  accurate Gamma IR, double candidate columns come from full double products
  in this branch, with bounded derivative batching. Its speed is not claimed
  to match the noBN optimization.
- Power10 remains an explicit comparison mode. For noBN it may share D only
  for accurate double Gamma columns while keeping the registered power10 score
  and generator. Candidate counts, bias grid, top-2/four-live limits, mu update,
  BetaRule and real-loss guards are unchanged.

## What the profile notebook measures

GPU0: current IR/exact noBN D_site path plus capacity profile.
GPU1: old 8c56d02 Gram/power10, then current IR/power10.
All use the same task-1 checkpoint and replay the first task-2 application.

- ir_power10_vs_gram.json: strict relative tolerance 1e-5 and zero absolute
  tolerance. It retains failures and the per-mu numerical-failure audit.
- ir_exact_vs_power10.json: separate comparison of the curvature change.
- Reports retain high/inner timing, outer residual histories, inner iterations,
  warm resets, cached requests, Gamma linear-solve margins, peak D-block bytes/time/VJPs and block counts,
  actual final L/KD and application VRAM.
- Capacity measures diagnose at m=25600/12800 and, at the 128-sample engine
  shape, the first stage's streamed double D block alongside native/double
  frozen VJP graphs.
  This remains a task-2-weight/100-class-head proxy, not a complete task-20 run.
- MODE=smoke3 remains a separate explicit three-task native-epoch rehearsal
  after review. M1 remains held. No authoritative K0 or T4 speedup is inferred
  from local CPU measurements.

## Small local numerical example (not a T4 timing)

A float32 MLP with a genuine float64 operator reference, mu=.3, condition
number 4.2025033: three outer corrections, 14 total inner CG iterations.
Double residual history:
1 -> 2.3407400e-5 -> 2.0662344e-9 -> 6.1452890e-14.
Relative solution error against a dense float64 solve: 8.9807372e-14.
This validates the intended mechanism on a small problem, not the user's T4
condition estimate, mu* trajectory, 1e5 Gamma margin or proposed runtime.

## Local validation (2026-10-22)

148 affected tests passed, 5 deselected; no full-suite run. Files:
`test_vna_ir_downstream`, `test_vna_cg_output`, `test_vna_exact`,
`test_vna_v4_fix2`, `test_vna_v3`, `test_vna_v3_certificates`, `test_theory_v3`,
`test_exact_gpu_lanes`, `test_exact_profile_report`,
`test_exact_notebook_bootstrap`, `test_main_cil`, `test_cg`.
The five previously failing tests excluded from this run are:
- adaptive_schedule_iterations_skips_and_budget_exceeded (retired K_base);
- real_sgd_momentum_and_weight_decay_remain_native_after_correction (roundoff);
- gir_rechecks_actual_dead_slot_revived_by_upstream_correction (old expectation);
- collector_and_paired_table (fixture missing interventions);
- two_task_smoke_identity_and_resume (stale task-index pause assertion).
They were not repaired or claimed passing in this task.

All 11 generated notebook variants passed code-cell AST checks in memory;
`git diff --check` passed. No notebook source pin was changed: the current
AGENTS.md says not to commit, so existing checked-in notebooks still load the
previous committed solver. The updated builder defines the new IR lanes but
requires packaging against a future committed source before use on Kaggle.
No T4/profile/M1, theory PDF render or full task-20 memory check was run locally.

## Sample-block follow-up validation

The follow-up suite passed 131 tests (3 previously failing tests deselected;
first three in the list above). It adds block sizes 1/2/8, an uneven final
block, and weak-reference assertions that the preceding D block is freed before
the next is constructed. Block s_v, K_v and eigenvalues match full double
references for MLP/conv. The full float32 GIR preflight asserts that dense
columns are requested only from the Gamma loop and restores model/optimizer/RNG.
Diagnostic capacity logs now report peak block bytes/count, not full-site bytes.
Full-window metric normalization and power10 scores/generators are preserved.
Actual T4 timing/VRAM and notebook source packaging remain unchecked/unchanged.

## Selectable solver / notebook integration

The subsequent same-commit request authorizes committing/publishing the source
and packaging a new notebook; the earlier no-commit/pending-pin note is historical.
See GRAM_M1_IR_NOTEBOOK.md and the latest PREREG entry. Both Gram and CG-IR now
exist behind --solver, with separate authority/resume identity. The integrated
169-test affected run passes without exclusions, including the three previously
excluded tests. Only the new notebook authorizes Gram M1s0; cheap M1 stays held.
