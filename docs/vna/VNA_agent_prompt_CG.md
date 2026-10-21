# Matrix-free T29/T31 follow-up protocol

Current authority: IR_DOWNSTREAM_REVIEW.md, the 2026-10-22 entry in
PREREGISTRATION_CHANGES.md and theory/s9{b,c}_*.tex. CG_OOM_REVIEW.md describes
historical measurements, not the current precision or curvature defaults.

Implementation acceptance:
- Engine and diagnose solve A64=J64^T M J64+mu I, b64=J64^T c. Use a genuine
  double frozen model and double replacement columns for RHS/residual, native
  float32 correction CG at 1e-4, double iterates, outer target 1e-10, total
  200 CG iterations and at most eight corrections. Detect stagnation. Double
  native models use the outer target directly. Never accept nonconvergence.
- Failed LM trials advance mu; later failure restores the whole application.
  Application-local warm starts/cache only; no state across applications.
- Frozen J' uses column-coordinate replacement, including shared sites.
  No global Gram, native J_u columns or Woodbury in production.
- Default exact noBN: sample-separable double D in blocks of eight samples;
  accumulate every candidate's s_v and K_v then discard the block. Exact small
  eigvalsh curvature after accumulation; dense full-window J_v only at Gamma. BN/stochastic/coupling fallback
  uses full products/power10. Keep explicit power10 comparison mode/generator.
- Preserve candidate grid, top-2/four-live limits, native real-loss/window checks,
  coefficient-zero identity, RNG isolation, exact resume and deadline/exit-75.
- Local checks cover actual double reference accuracy, shared D for MLP/conv/
  residual/inplace models, coupling fallback, local warm/cache, overlapping
  replacement columns, nonconvergence and read-only full float32 GIR preflight.

Kaggle acceptance, without automatically launching M1:
1. MODE=profile: GPU0 current IR/exact/capacity; GPU1 8c56d02 Gram/power10,
   then current IR/power10. Strict 1e-5 comparison ir_power10_vs_gram.json;
   separate ir_exact_vs_power10.json. Preserve FAIL and numeric-failure audit.
   Log true64 residuals, corrections, high/inner time, Gamma error margins,
   D time/bytes/VJPs, L/KD and application peak memory. Capacity covers
   m12800/m25600 diagnose plus D/frozen graphs at N128/C100. Label proxy scope.
2. Review whether positive-rho reference mu trials were rejected numerically,
   and whether unknown reference trials remain. No generic mu* equivalence claim.
3. After review, MODE=smoke3: fresh baseline/full seed 0, three complete tasks,
   fresh IR/exact K0 preflight, native resume/epochs; actual application counts,
   elapsed time, NaN and dead/dormant. M1 remains held pending measured results.
4. Do not reduce candidates or probe size on an unmeasured runtime/memory guess.

The explicit same-commit Gram M1s0 / CG-IR profile notebook is now specified in
GRAM_M1_IR_NOTEBOOK.md and the latest PREREG entry. --solver selects gram or
cg_ir. Only that Gram lane is authorized for M1s0; CG-IR M1 remains held.
The numerical IR policy above describes cg_ir, not the archival Gram backend.
