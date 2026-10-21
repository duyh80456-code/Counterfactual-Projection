# T4 OOM review and matrix-free replacement — 2026-10-21

Source: downloaded `vna4-exact-diagnostics (4).ipynb`, output cells 5–6.
The current/reference reports identify their commits and the same task-1
checkpoint; the application being replayed is in task 2.

## What the run actually established

| Report | Application seconds, audit excluded | Gram m=1280 seconds | prepare_gir, 40 calls | Gamma, 58 trials |
|---|---:|---:|---:|---:|
| b97d25c power10 | 1215.697 | 81.005 | 537.087 | 136.260 |
| 8c56d02 power10 | 1142.843 | 71.595 | 658.358 | 375.591 |
| 8c56d02 exact | 3240.354 | 69.327 | 2757.444 | 375.840 |

The exact application wall time was 3771.064 seconds, including 530.710 seconds
of separately timed power10 audit. The candidate/JVP batching was **not** a
successful T4 speed improvement overall: grouped exact candidate Jacobians are
far more expensive; native slot column batching also increased Gamma time.
A larger vmap width is not a demonstrated solution.

All three applications completed, with mu*=37.08382698082, eight LM trials,
and the same 25 swap slots. Nevertheless strict power10 comparison **FAILS**:
candidate values, ordered Gamma and installed candidate weights do not all
meet relative tolerance 1e-5, absolute tolerance zero. Two of 58 Gamma values
fail: maximum absolute discrepancy 1.3091488781924454e-8, for approximately
7.0158e-4. No tolerance has been relaxed to turn this into a PASS.
Final actual L/KD are respectively (3.1291733894748175, 1.3002662095412456)
and (3.129173424789118, 1.3002662204032305) for old/current power10.

The **capacity proxy**, using task-2 weights with an extended 100-class head
and repeated inputs, reports:

- N=256, m=25600: CUDA_OOM during Gram construction after 0.431 seconds;
  allocated peak 14,659,228,160 bytes, reserved peak 15,328,083,968 bytes.
  The CUDA error states a 14.56 GiB device and only 106.81 MiB free.
- N=128, m=12800: GRAM_LIMIT at 602.763 seconds, allocated peak 8,372,708,352
  bytes. This is a time guard, not an engine OOM.
- Neither is an actual task-20 checkpoint/run. Existing reports merge overall
  peaks with capacity peaks, so their peak is not an isolated engine peak.
  New reports preserve application_peak_cuda_bytes separately.

A single float64 Gram at m=25600 needs 5,242,880,000 bytes. Global metric
factors, factorizations, derivative batches and workspace add substantially.
Reducing the diagnostic probe or vmap width does not remove the engine's
quadratic storage or Gram construction cost.

## Chosen implementation

Use the existing CG implementation on the SPD **parameter** system
(J^T M J + mu I)x = J^T c. Do not apply CG to nonsymmetric mu I + M G.
Keep network products in native dtype (float32 on T4); vectors, metrics and
reductions are float64. The returned parameter solution avoids the cancellation
in reconstructing an output cotangent by division by a tiny mu, and avoids
squaring errors in an eigendecomposed square root when BetaRule makes beta large.
Gram/Cholesky/Woodbury remain only for small reference tests/historical runs.

The fixed relative residual requirement is 1e-8, with at most 200 iterations
per system and no warm start shared between solves. Check the residual again in
original RHS units, since native float32 products need not scale exactly.
If necessary refine within that same total iteration budget; never loosen the
tolerance. An unsuccessful LM solve is a failed trial and increases mu using
the unchanged Nielsen trial sequence. A failed later solve cancels and restores
the entire application, including prior swaps, optimizer state and engine state.
End-task diagnose returns B=null and cg_not_converged rather than a false score.

For Gamma, construct a frozen J' action by masking removed native coordinates
and assigning J_v columns to the outgoing coordinates. Incoming columns of the
new B=0 neuron are zero. Coordinate replacement handles adjacent/shared sites
without subtracting columns twice. This is the same J'J'^T as signed column
updates, but needs neither native J_u materialization nor Woodbury. Retain only
current replacement columns, not a history of all native removed matrices.
Log Gamma_cg for new runs; comparison accepts historical Gamma_woodbury fields.

Only power10 candidate ranking remains in the engine, using the registered
10 iterations and generator. Its estimated ell_beta is a ranking score, **not**
a certified spectral lower bound. Keep the existing 4 directions, 2 signs,
bias quantiles, top-2 Gamma candidates and four live slots per site.

This removes the identified global O(m^2) allocations. It is not a promise
that every configuration fits T4 or that the application now costs <400 s.
The measured prepare_gir=658 seconds alone is a remaining performance risk.
CG can change discrete decisions near ties/zero; float32 matvec and an absolute
1e-8 residual target are not automatically compatible at small damping.

## Next runs, in order

1. MODE=profile: GPU0 current D-CG power10 + capacity proxy at m=12800/25600;
   GPU1 unchanged 8c56d02 Gram power10. Same original checkpoint, first task-2
   application. Save strict comparison, all residuals/iterations, per-component
   time, application VRAM, proxy VRAM and actual final L/KD. A mismatch remains
   FAIL; missing/failed data remain INCOMPLETE. No M1 is started.
2. Review the actual results. MODE=smoke3 then runs fresh baseline/full seed 0
   with a fresh CG authority preflight, full 100/50/50 epochs, distinct max_tasks=3
   result identity, native resume and deadline handling. Full on GPU0, baseline
   on GPU1; one GPU uses serial fallback. It does not require the historical
   task-1 checkpoint and does not truncate the main M1 horizon.
3. Inspect applications, time, NaN, dead/dormant and convergence failures before
   deciding M1: two arms, 20 tasks, seeds 0–2, budget 1.0x. Recalibrate K0 from CG;
   do not reuse the old Gram schedule or invent K0 from local CPU timing.
4. Only after M1: compute-matched 2–3x budget ablations, C1–C5, component
   ablations, M2 BN and M3 random order. No reduced candidate counts or changed
   probes are enabled to force a faster result.

## Limits of the error statement

For exact SPD A >= mu I and true r=b-Ax, ||x-x*|| <= ||r||/mu. For the bilinear
q=-b^T x/2 used in Gamma, |q-q*| <= ||b||||r||/(2 mu). The two q errors add.
Thus O(epsilon) for Gamma has constants depending on mu and RHS norms; it
cannot guarantee unchanged signs/swaps near zero. Full quadratic objective
error would be O(epsilon^2), but that is a different evaluated quantity.
For an inexact matvec A+E, the ideal-operator bound also needs ||Ex||/mu.
True-loss rho and window checks remain necessary.

No new T4 run, real task-20 capacity result or three-task training result was
produced locally. The review records supplied measurements, not new timings.

## Local validation

134 related tests passed, with five previously documented failing tests excluded:
retired adaptive K_base schedule; strict native-SGD roundoff at 1e-16; old
revived-slot expectation; incomplete collector fixtures; old task-prefix pause
assertion. No full repository suite was run. The unchanged numerical theory
checker reports zero T29/T31/R1 violations over 200 problems and zero T30
nontermination cases (median/max seven retries). The LaTeX fragments were edited;
a PDF was not built. Local CPU checks are not substitute T4 timings.
