# Matrix-free T29/T31 follow-up protocol

Authority: CG_OOM_REVIEW.md, PREREGISTRATION_CHANGES.md and theory/s9c_exact.tex.
Do not reinterpret the old run's capacity proxy as actual task-20 training.

Implementation acceptance:
- Production engine and diagnose use residual-verified CG on J^T M J + mu I;
  no global output Gram/factor and no nonsymmetric CG. Native float32 products,
  float64 iterates/reductions, relative tolerance 1e-8, 200 total iterations.
- Nonconverged LM trial fails and increases mu; no invalid direction/Gamma is
  accepted. Later failure rolls back the application. No cross-solve warm state.
- Frozen J' uses exact column-coordinate replacement; shared sites are covered.
  Native J_u columns and Woodbury are absent on the CG path.
- power10 only: same generator and 10 iterations; ell_beta ranks candidates.
  Preserve top-2/four-live limits and all loss/window checks.
- Preserve baseline/coefficient=0 identity, RNG/model/optimizer isolation,
  native exact resume, provenance, deadline and resumable exit-75 behavior.
- Local reference checks: CG vs Gram on small MLP/conv/BN and singular metrics;
  sequential/shared swaps, nonconvergence, LM retry, no dense Gram/native columns,
  read-only measurement, exact resume and mode/schedule identity.

Kaggle acceptance, without automatically launching M1:
1. One MODE=profile notebook: GPU0 D-CG power10/capacity; GPU1 D-Gram power10
   at 8c56d02. Same checkpoint, first task-2 application. Report strict rel tol
   1e-5 comparison of mu*, rho, L/KD, candidates, Gamma and installed swaps,
   plus time, CG residuals/iterations and memory at m=12800/25600. Never hide FAIL.
2. After review, MODE=smoke3: fresh baseline/full seed 0, three complete tasks,
   fresh CG K0 preflight, native resume, full training epochs. Report actual
   application counts, elapsed time, NaN and dead/dormant.
3. Decide M1 only from those actual results. If still >400 s, investigate
   prepare_gir before proposing any change to candidate sets or algorithm.
