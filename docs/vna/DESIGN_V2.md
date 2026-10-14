# VNA design decisions v2 (2026-10-14) — supervisor-condensed

Authoritative decisions for Stage 2. Earlier SPEC.md remains valid where not overridden here.

## Formulas (GGN metric, one 1/n normalization throughout)
- g = (p − y)/n per sample (mean CE), H_i = diag(p_i) − p_i p_iᵀ (true GGN; never empirical Fisher).
- F = Jᵀ H J (block params, residual-path scope), C_OE = Jᵀ H J_E, C_EO = C_OEᵀ, ∇θ = Jᵀ g.
- d = (F+μ)⁻¹ C_OE C_EO (F+μ)⁻¹ ∇θ   (parameter space, symmetric PSD preconditioner ⇒ descent).
- Function-space equivalent: d = Jᵀ (HK+μ)⁻¹ H K_E H K (HK+μ)⁻¹ g. HK+μ is NOT symmetric: a
  function-space exact solver must use the symmetric per-sample H^{1/2} form or solve in parameter
  space. Do not define K_H = J H Jᵀ.
- At B = 0 the A-part of J_E is exactly zero ⇒ E parameters Q = params(B) only.

## Solver variants
- `kfac` (default for training): C_EO u = J_Eᵀ H (J u) and C_OE v = Jᵀ H (J_E v) computed EXACTLY by
  JVP+VJP; only (F+μ)⁻¹ approximated by K-FAC/EK-FAC (per conv: input-patch ⊗ output-grad factors,
  true-Fisher via MC-sampled labels or exact GGN; EK-FAC eigenbasis gives exact Tikhonov damping;
  BN affine params use a diagonal block). Factors via EMA, eigendecomposition at refresh only.
  Never factor C_OE block-diagonally (that zeroes the conv1 coupling).
- `exact` (reference): exact (F+μ)⁻¹ via CG in parameter space on the GGN (matrix-free JVP/VJP),
  enough iterations to converge on the toy; optional output-sketch variant later.
- `lite`: scalar damping, d = Jᵀ H K_E H K g / c with c = (tr(HK)/n)² — cheap baseline.
- Old Euclidean-logit full VNA stays available for comparison only.

## Selection rules
- `tiny_U`: GroMo default (current).
- `tiny_V`: TINY closed form with cotangent z (z = filtered residual from the exact/kfac solve:
  z = (HK+μ)⁻¹… consistent with the chosen metric; document precisely) instead of g; averaged over
  several batches; generalized eigenproblem (N_z N_zᵀ, S) with thresholded S^{-1/2}; NO
  existing-weight projection by default (ablation flag). Linearized-ReLU approximation, documented.
- `greedy_V`: pool = random ∪ top eigvecs of tiny_V ∪ sign flips; score c_j = J_{E_j}ᵀ z with true
  ReLU; greedy OMP / pivoted Cholesky on cᵀ(G+νI)⁻¹c until θ ∈ [0.5,0.8] of pool gain.
- Fix ‖a_k‖ to the mean norm of existing neurons (V ∝ ‖A‖²). Score a and −a, keep better.

## Required controls / fixes
- NGD control `ngd`: same solver, same schedule, target −K g ⇒ d_ctrl = (F+μ)⁻¹ F (F+μ)⁻¹ ∇θ
  (GGN form), norm-matched to the VNA correction at each step.
- `random_A`: K_E trace-matched random A. Optional `spectrum_matched` S = h(K).
- BN effective-LR: for every BN-followed conv weight project d ← d − (⟨d,W⟩/‖W‖²) W (default on,
  flag to disable); log ‖W‖ per layer every epoch in every arm.
- BN eval mode (folded running stats) in all products; never update running stats.

## Validation before GPU
cos(d_kfac, d_exact) and cos(d_lite, d_exact) on a tiny Gromo ResNet across μ sweep
{1e-4,1e-3,1e-2,1e-1}·tr(F)/p; ablation dropping the conv1 cross term must collapse the W1 cosine.

## Training protocol (Stage 2a)
Arms: sgd, sgd_cm (equal wall-clock, i.e. ~more epochs at same schedule shape), vna_kfac_tinyV,
vna_kfac_tinyU, vna_random, ngd. Same LR/WD for all; final model; official test set once; curves
logged; cost columns (train/deploy params, GPU-h, peak memory, extra passes/step). Seeds 0,1 first.
Later (not now): SWA/SAM/DBB baselines, multi-block joint K_E, shadow-B (VNA+).
