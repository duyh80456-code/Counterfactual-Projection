# Stage 2-core task: GGN/K-FAC VNA solvers, controls, projection-aware selection (CPU-validated)

Read `docs/vna/DESIGN_V2.md` (authoritative), then `docs/vna/SPEC.md`, `docs/vna/STAGE1_REPORT.md`.
Branch `vna-phase0`. `.git` is read-only: do NOT commit. Use `.venv/bin/python`. Do NOT change the
Phase 0 notebook, `diagnostics/signal_curve.py` behaviour, or its resume identity (a Phase 0 run is in
progress on Kaggle pinned to an older commit; keep that code path working). New code goes in new
modules (e.g. `methods/vna_ggn.py`, `methods/vna_kfac.py`, `methods/vna_select.py`) reusing
`methods/vna.py` `_Products` machinery. No training runner or notebook in this stage.

## 1. GGN products and exact reference
- Matrix-free products with H applied per sample (C×C, from softmax p, fp64 where cheap):
  F v = Jᵀ H J v, C_OE v = Jᵀ H J_E v, C_EO u = J_Eᵀ H J u, ∇θ = Jᵀ g. One 1/n normalization.
- `vna_exact_ggn(...)`: d = (F+μ)⁻¹ C_OE C_EO (F+μ)⁻¹ ∇θ with parameter-space CG (fp64 vectors),
  configurable iterations (converged on toy). μ relative to tr(F)/p (Hutchinson).
- `ngd_control(...)`: d_ctrl = (F+μ)⁻¹ F (F+μ)⁻¹ ∇θ via the same solver; helper to norm-match it to a
  given VNA correction.
- `vna_lite_ggn(...)`: d = Jᵀ H K_E H K g / c, c = (tr(HK)/n)², no CG.
- Tests: dense small-model oracle (explicit J, J_E, H) for all three; function-space equivalence
  d = Jᵀ(HK+μ)⁻¹HK_E HK(HK+μ)⁻¹g on the dense oracle; descent gᵀJ d ≥ 0; finite differences for the
  H-weighted products; invariance (params/buffers/grad/modes/global RNG unchanged).

## 2. K-FAC variant
- `KFACBlockPreconditioner`: per conv in the block's residual path, factors A (unfolded input patches
  covariance) and G (output-gradient covariance) from TRUE Fisher (MC-sampled labels from p with a
  dedicated generator, or exact GGN — pick one, document), EMA, EK-FAC eigenbasis with exact
  Tikhonov damping; BN affine params use a diagonal Fisher block. Refresh/eigendecompose on demand
  only; state_dict/load_state_dict.
- Prefer a maintained library for factor computation if it pins cleanly and works with Gromo convs
  and torch 2.2 (curvlinops or ASDL; record commit + license in third_party/REFERENCES.md); otherwise
  implement minimal factors yourself and say why.
- `vna_kfac(...)`: d = M C_OE C_EO M ∇θ with M = (F_kfac+μ)⁻¹ and C_OE/C_EO EXACT (JVP+VJP). Never
  factor C_OE block-diagonally.
- Tests: symmetric PSD preconditioner ⇒ gᵀJ d ≥ 0 for random batches; EK-FAC inverse equals dense
  inverse of the Kronecker approximation on a tiny layer.

## 3. BN tangent projection
- For each BN-followed conv weight W in the scope: d ← d − (⟨d,W⟩/‖W‖²)W (default on, flag off).
  Apply to VNA corrections and the ngd control alike. Helper to log per-layer ‖W‖.
- Test: ⟨d_proj, W⟩ = 0; BN affine params untouched by the projection.

## 4. Selection rules (`methods/vna_select.py`)
- `tiny_V`: TINY closed form with cotangent z instead of g. Implement by running Gromo statistics
  with a pseudo-loss whose logit gradient equals z (verify the sign convention against TINY), so the
  pinned upstream solve is reused; NO existing-weight projection by default (flag for ablation);
  average N_z over several batches; generalized eigenproblem (N_z N_zᵀ, S), thresholded S^{-1/2}.
  Define z exactly per DESIGN_V2 for the GGN metric and document the derivation in the report.
  Test: with z := g and existing-weight projection ON it reproduces tiny_U's A (up to sign/order).
- `greedy_V`: pool = random ∪ top eigvecs of tiny_V ∪ sign flips; true-ReLU scores c_j = J_{E_j}ᵀ z,
  Gram G; greedy OMP / pivoted Cholesky maximizing cᵀ(G+νI)⁻¹c until θ (default 0.6) of pool gain or
  r_max. Test: no duplicate picks; monotone gain; reaches θ.
- Norm: rescale each a_k to the mean norm of existing neurons; score ±a_k, keep better.
- Random-A trace-matched (K_E trace) control.

## 5. CPU validation script `diagnostics/vna_solver_validation.py`
On a tiny Gromo ResNet checkpoint (fake data OK): for μ ∈ {1e-4,1e-3,1e-2,1e-1}·tr(F)/p report
cos(d_kfac, d_exact), cos(d_lite, d_exact), cos(old Euclidean VNA, d_exact), per-parameter-group
cosines (W1, W2, BN), the conv1-cross-term-dropped ablation (W1 cosine must collapse), realized
gains, and per-correction cost in SGD-step units and pass counts. Also V for tiny_U vs tiny_V vs
greedy_V vs random at equal rank. Save CSV/JSON; summarize in the report.

## 6. Report
Append "Stage 2-core" to `docs/vna/STAGE1_REPORT.md` (or new `docs/vna/STAGE2_REPORT.md`): files,
formulas as implemented, the z definition, library decision, validation table, costs, open
questions. Run tests/test_vna*.py and the full suite; report counts.
