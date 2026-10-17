# Unified projection: how the GN step, the base optimizer and VNA fit into one mechanism (2026-10-18)

Basis: `THEORY_FINAL_v3.md` (notation §0, algorithm §2) and `consistency_setup.md`. New checks: `check_unified_gn.py` (math: 15/15 pass, with negative controls and witnesses; toy: §8). Status tags as in v3: proven / assumption / empirical / interpretation.

**Bottom line.**
1. The claimed decomposition "projection of (K + K_E)z = x_K + d" is **true** (exact, T19). But **x_K is not the GN step**. x_K = (FG)²g is a damped *gradient*; x_GN = Gg is a Newton-type step, and they differ (median cos 0.50).
2. VNA's own dynamics is the **grown z-flow**, v = −(K + K_E)z. Its K-part equals gradient flow up to an exact O(μ) term (T20), so **the K-part is the base SGD step that the algorithm already takes**. The algorithm "SGD every step + λ·d every K steps" is therefore already the discretised Dirac–Frenkel projection of one grown dynamics. **No change to the update vector**, so every v3 theorem carries over unchanged.
3. The GN step x_GN has a different, exact role: it is the **splitter**. Step 1 computes it, and it splits the residual exactly into a reachable part z = HJx_GN, which VNA acts on, and an unreachable part r_⊥ = e − z, which only GIR can act on (T20, T23).
   - Base optimizer, VNA and GIR thus have disjoint, linked roles: K-part, K_E-part on z, new degrees of freedom on r_⊥.
4. "SGD + λ(x_GN + d)" is **not** the projection of any single grown dynamics. The grown damped-GN dynamics has K-part x_GN but a different virtual term, (1/μ)GJᵀHK_E r_⊥ (T22). Do not use the mixed update. The GN step stays an external "base-optimizer swap" baseline (free: x_GN is stored in step 1), not an ablation.
5. The ablations are members of one update family; the kernel control (old "NGD" arm) is exactly the K_E → K member (T19c).

---

## 1. Projection operator

For a target output velocity t ∈ ℝⁿ, define the damped Dirac–Frenkel projection

  P(t) := argmin_x ‖Jx − t‖²_H + μ‖x‖² = G JᵀH t.

**Lemma P (proven).** The normal equations give (JᵀHJ + μ)x = JᵀHt. Uniqueness follows from μ > 0. Check U1: error ≤ 1.5e-13 against a stacked least-squares solve; the NEG case GJᵀt (H dropped) is off by ≥ 590×.

Note: for an exactly representable velocity t = Jy, P(Jy) = GFy ≠ y. Damping shrinks every direction by FG ⪯ I. The undamped (minimum-norm, μ → 0) projection would return the representable part of y.

## 2. Three candidate grown dynamics

The grown model has output Jacobian [J, J_E] and kernel K + K_E, with K_E = J_EJ_Eᵀ at w = 0 (v3 §0).

| Dynamics | Output velocity of the grown model | Projection onto the fixed model | Verdict |
|---|---|---|---|
| D1: grown gradient flow | −(K + K_E)e | GFg + GJᵀHK_E e | The virtual part is **not** sign-definite: gᵀGJᵀHK_E e < 0 in 10/200 problems (U5-WIT). It has no descent certificate. Rejected. |
| D2: grown **z-flow** (VNA) | −(K + K_E)z, with z = HJx_GN | x_K + d (T19) | Both parts are descent directions (T21). The K-part is gradient flow up to O(μ) (T20). **Adopted.** |
| D3: grown damped GN | −K′(HK′+μ)⁻¹e, K′ = K + K_E | K-part = x_GN; first-order virtual part = (1/μ)GJᵀHK_E r_⊥ (T22) | Consistent in itself, but its virtual term is not VNA's d. Valid only while ‖HK_E‖ ≪ μ. |

**Why z (D2), not e (D1), drives the virtual neurons (proven, T21).** With cotangent z, the virtual part is a PSD preconditioner: gᵀd = zᵀK_Ez = V ≥ 0. With e this fails. In D2 the old parameters still effectively follow gradient flow (T20), so D2 differs from D1 only in feeding the virtual neurons the *reachable* part of the residual.

## 3. Theorems

### T19. Decomposition of the projected grown velocity (proven)

(a) P((K + K_E)z) = P(Kz) + P(K_Ez) = x_K + d, with d = GJᵀHK_Ez (VNA as in v3) and x_K := GJᵀHKz. *Proof:* P is linear. U3: 7.5e-16.

(b) x_K = GF·(Jᵀz) = GF·FGg = (FG)²g = GF²Gg. *Proof:* Kz = J(Jᵀz) and Jᵀz = JᵀHJGg = FGg. F and G commute. U2: 2.3e-12.

(c) x_K is exactly the existing "NGD same-norm" arm, i.e. d with K_E replaced by K. *Proof:* substitute K_E → K in d = GJᵀHK_Ez. U2.

(d) x_K ≠ x_GN := Gg in general (U2-WIT: relative difference ≥ 1.16, median cos 0.50). As μ → 0 on range(F), x_K → P_{range F}g (gradient) while x_GN → F⁺g (Gauss–Newton). **The faithful K-part of VNA's dynamics is x_K, not x_GN.** ∎

### T20. Exact split of the residual by the GN step (proven)

Let x_GN = Gg, which is the first solve of step 1 (v3 §0: x_z). Then:

(a) z = HJx_GN = HK(HK+μ)⁻¹e, and r_⊥ := e − z = μ(HK+μ)⁻¹e.
*Proof:* push-through, J(JᵀHJ+μ)⁻¹Jᵀ = K(HK+μ)⁻¹. Hence z = HK(HK+μ)⁻¹e and e − z = (I − HK(HK+μ)⁻¹)e = μ(HK+μ)⁻¹e.

(b) Ke − Kz = μ·Jx_GN.
*Proof:* Ke − Kz = μK(HK+μ)⁻¹e = μJGJᵀe. U4: 2.9e-11. The NEG case "= μKe" is rejected (≥ 0.62).

(c) Consequently the K-part of the z-flow is gradient flow minus μ times the GN velocity. ‖Ke − Kz‖/‖Ke‖ → 0 as μ → 0 (U7: 5.2e-1, 2.1e-1, 5.4e-2, 6.9e-3, 7.1e-4 for μ_rel = 1, …, 1e-4).

(d) *Interpretation.* With Δ\* = −Jx_GN, the output change of the fixed model's GN step:
- z = −HΔ\* is the part of the residual that this step removes;
- r_⊥ = e + HΔ\* is the gradient of the GN quadratic model eᵀΔ + ½ΔᵀHΔ left at Δ\*, i.e. what the fixed model cannot remove. ∎

### T21. Gain decomposition (proven; β = 0)

Per unit time, the first-order decrease of the projected z-flow is

  gᵀ(x_K + d) = ‖FGg‖² + V,  with ‖FGg‖² = gᵀx_K ≥ 0 and V = gᵀd = zᵀK_Ez = ‖J_Eᵀz‖² ≥ 0.

*Proof:* gᵀGF²Gg = ‖FGg‖². gᵀGJᵀHK_EHJGg = zᵀK_Ez. U5 over 200 problems.

**Consequence for ρ.** Since ‖FGg‖² ≤ ‖g‖², the existing statistic ρ = V/‖g‖² is a lower bound on the **virtual share of the grown model's descent rate**, ρ_proj := V/‖FGg‖² ≥ ρ, and ρ_proj → ρ as μ → 0 on range(F). Gate G's ρ therefore measures exactly "how much of the grown model's progress comes from the virtual neurons".

With β > 0, gᵀd\* is no longer sign-definite (v3 T5(g)), and the v3 skip rule applies unchanged. ∎

### T22. The grown damped-GN dynamics, and why "x_GN + d" is not one mechanism (proven)

The grown GN θ-step is δ′_θ = −Jᵀ(HK′+μ)⁻¹e. At K_E = 0 it equals −x_GN. Its first-order change in K_E is

  ∂δ′_θ/∂K_E [K_E] = GJᵀHK_E(HK+μ)⁻¹e = (1/μ)·GJᵀHK_E r_⊥.

*Proof:* (HK′+μ)⁻¹ = (HK+μ)⁻¹ − (HK+μ)⁻¹HK_E(HK+μ)⁻¹ + O(K_E²), Jᵀ(HK+μ)⁻¹ = GJᵀ, and T20(a). U6: central finite difference matches to 1.4e-6.

This term is **not** VNA's d. U6-NEG: relative difference ≥ 43, median |cos| 0.76. Its validity needs ‖HK_E‖/μ ≪ 1.

Hence x_GN + d combines the K-part of D3 with the virtual part of D2, and is not the projection of any single grown dynamics. **Rejected as the unified update.** ∎

*Remark (interpretation).* If one ever switched the base optimizer to damped GN, the consistent virtual correction would be T22's term, driven by r_⊥. That is the same cotangent GIR uses. This is recorded as a possible future variant and is not part of the method.

### T23. GIR consistency with the split (proven; restates v3 T7b in split form)

Enlarging the span by a new neuron J_v lowers the GN-model value by

  q\*_e(J) − q\*_e([J, J_v]) = ½ sᵀS⁻¹s,  with s = J_vᵀr_⊥ and S = J_vᵀH_⊥J_v + μ.

So the swap gain is driven by **exactly the unreachable part r_⊥ of T20**. U8: 6.2e-12; the NEG case with z in place of r_⊥ is rejected (≥ 0.21).

After the swap, K′ = K + J_vJ_vᵀ ⪰ K. At the next application, step 1 recomputes the split with K′. The part of r_⊥ that the installed neuron can reach migrates into z′, and the base optimizer (the K-part) trains the installed neuron like any other. This is consistent with v3 T7(g), which runs the swap after the correction. ∎

### T24. Step size along d (proven)

For the update θ ← θ − s·d, the GN-model decrease s·gᵀd − ½s²·dᵀ(F+μ)d is maximised at s\* = gᵀd / dᵀ(F+μ)d. U9 confirms this.

---

## 4. Final update (unchanged vector; reinterpreted)

```
every iteration:   θ ← θ − ε m_t                         # K-part of the projected grown z-flow (base optimizer)
every K iterations (v3 §2 steps 1–7, unchanged):
   step 1 solves x_GN = G g  →  split e = z + r_⊥          # T20: the splitter
   d̂ = d_rep + d_N^β  (K_E-part acting on z, with v3 protection)
   certificates, skip rule, momentum, BN: as v3
   θ ← θ − ε λ d̂
   GIR on r_⊥ (dead slots), after the correction           # T23
λ (setup rule; see §5):  λ = min(λ_faithful, s*/ε),  λ_faithful = K
```

**What changes relative to v3:**
- **Nothing in the update vector, the certificates or the costs.** All of v3 T1–T18 carry over verbatim, because the applied vector is the same d̂.
- **Interpretation only:**
  - the base step is the K-part;
  - ρ is the virtual share of the descent rate (T21);
  - the GN solve is the splitter (T20);
  - GIR's r_⊥ is the complement of VNA's z (T23).
- **Arm naming:**
  - the old "NGD same-norm" arm is renamed `kernel_self` (K_E → K, which equals x_K, T19c);
  - `sgd_gn` (x_GN) is an *external* base-optimizer baseline, not an ablation.
- **λ rule:** §5. This is a setup choice and must be pre-registered.

## 5. λ: what the projection prescribes (assumption + proven safeguard)

- In the continuous z-flow, the K-part and the K_E-part enter with equal weight per unit time.
- The base optimizer applies the K-part every step. The virtual part, applied once every K steps, must therefore carry K steps' worth: **λ_faithful = K**.
- This needs **A13 (lazy correction):** d̂ varies little over the K-step interval. A13 is the same staleness assumption used for K-FAC and Shampoo preconditioners. It is **assumption, interpretive** and is not used by any certificate.
- Safeguard: cap the step at the GN-model optimum s\* (T24), giving λ = min(K, s\*/ε).
- Apply the same rule to every control (`kernel_self`, `sgd_gn`).
- Momentum: if the base optimizer has momentum m, its effective per-step K-part is ε/(1−m)·g, so λ_faithful = K/(1−m). Report which convention is used.
- **Effect on v3 theorems.** The first-order certificates (signs of ĝ, ĥ, the leak test) are scale-invariant in λ > 0. The KD-increase bound ελτ_h and v3's second-order step-size conditions (T5(e, f)) depend on λ only through the effective step ελ‖d̂‖, and must be evaluated with the chosen λ. No theorem statement changes.

The earlier norm-matching rule λ = ‖g‖/‖d‖ is **not** implied by the projection. It is a fairness convention and can be reported as a secondary setting.

## 6. One update family; every arm is a special case

U(λ, kernel, β, selection, S): every K steps add −ελ·P(kernel·z) with the v3 protection split; every step the base optimizer runs.

| Arm | Setting | Meaning |
|---|---|---|
| baseline (iCaRL / BP / CBP) | λ = 0 | K-part only |
| `kernel_self` | kernel = K | x_K: extra K-part, no virtual capacity (T19c) |
| VNA | kernel = K_E (tiny_V), β = 0 | virtual part on z |
| VNA + protection | β = β_cert | v3 |
| random-A control | kernel = K_E(random A), same trace | tests the growth criterion |
| GIR | S > 0 on dead slots, r_⊥ | new degrees of freedom |
| `sgd_gn` (outside the family) | base optimizer changed to lazy damped GN | external baseline, free byproduct |

The decisive comparisons inside the family are VNA vs `kernel_self` and tiny_V vs random-A. Both differ only in the kernel fed to the **same** projection, so a reviewer cannot read them as "a different optimizer won".

## 7. Consistency with the existing lemmas

- **Nested lemma (v3 T2).** Jx_K, Jd ∈ range(K). VNA acts only on the reachable part z; r_⊥ ∉ reach. This is now *structural*: VNA re-speeds reachable directions, GIR is the only component that reduces r_⊥ (T23).
- **Interpolation lemma.** If e → 0 then z, r_⊥ → 0 (T20a), so all three components switch off together. Phase 0 (no gain on an interpolating ResNet-18) is the predicted negative control.
- **GIR.** It uses r_⊥, the exact complement of VNA's z (T20, T23), and runs after the correction (v3 T7g). The installed unit joins the K-part from then on.
- **Protection (v3 T3).** It acts on the K_E-part only. The K-part (base optimizer) is unprotected, as in v3. The variant "protect the whole update" remains outside the method (consistency_setup §3).

## 8. Cost (proven by counting; v3 §2 table unchanged)

- Method: identical solve counts to v3. x_GN is the first solve of step 1 and only needs to be stored.
- `kernel_self`: one solve for x_K = G(F(Jᵀz)), replacing the d solve. Same count as VNA at β = 0.
- `sgd_gn`: 0 extra solves.

## 9. Toy (empirical; 12 seeds, float64, same toys as consistency_setup §4)

All arms run with the same certificate skip rule (gᵀv ≥ 0 and, for toy A, g_Oᵀv ≥ −1e-12, applied at β = 0) and the same norm-matched step ‖λv‖ = ‖g‖. The `_ls` variants add the cap s\*/ε. That cap **never bound** on these toys, so the `_ls` numbers equal the plain ones. Raw data: `unified_gn_toy_*.json`; summary: `unified_gn_toy_summary.txt`.

**Toy A (CIL-like). Means:**

| Arm | Avg acc | Forget | Loss at step 50 | Skips /60 |
|---|---|---|---|---|
| sgd | 0.6537 | 0.349 | 1.062 | 0 |
| kernel_self (x_K) | 0.6531 | 0.352 | 1.003 | 20.0 |
| VNA (d) | **0.6559** | **0.348** | 1.034 | 25.5 |
| sgd_gn (x_GN) | 0.6542 | 0.350 | 0.994 | 0.3 |
| x_GN + d (rejected mix) | 0.6546 | 0.349 | 0.995 | 0.5 |

**Toy A, paired average accuracy:**

| Comparison | Effect | Wins |
|---|---|---|
| VNA vs sgd | +0.22 pp | 10/12 |
| VNA vs kernel_self | +0.28 pp | 8/12 |
| VNA vs sgd_gn | +0.17 pp | 8/12 |
| x_GN + d vs VNA | −0.13 pp | 5/12 |

GN is faster early: VNA's loss at step 50 is +0.040 higher than GN's, and VNA wins on that metric in 0/12 seeds.

**Toy B (plasticity, no dead units). Final loss over the last 3 tasks, lower is better:**

| Arm | Final loss |
|---|---|
| sgd | 0.3705 |
| kernel_self | 0.3527 |
| VNA | 0.3613 |
| sgd_gn | 0.3524 |
| x_GN + d | 0.3503 |

| Comparison | Effect | Wins |
|---|---|---|
| VNA vs sgd | −0.009 | 9/12 |
| VNA vs kernel_self | +0.009 | 4/12 |
| VNA vs sgd_gn | +0.009 | 3/12 |
| sgd_gn vs sgd | −0.018 | 11/12 |

**Reading (empirical, small toys, directional only):**
- **Toy A:** VNA's virtual kernel gives the best final average accuracy inside the family, ahead of both its kernel control and the GN baseline.
- **Toy B:** the virtual kernel does **not** beat the K-kernel or GN.
- The rejected mix x_GN + d is never better than its better component on toy A.
- The skip rate is high for VNA (25.5/60) and kernel_self (20/60). That is the β = 0 KD certificate failing, as v3 T5 predicts, and the reason CIL needs β_cert.
- None of this changes the theory. The decisive VNA vs kernel_self and tiny_V vs random-A comparisons on ResNet32 and B5 remain open.

## 10. Status table (new items)

| Claim | Status | Check |
|---|---|---|
| P(t) = GJᵀHt | proven | U1, U1-NEG |
| P((K+K_E)z) = x_K + d; x_K = (FG)²g = kernel_self arm | proven | U2, U3 |
| x_K ≠ x_GN | proven (witness) | U2-WIT |
| e = z + r_⊥; z = HJx_GN; Ke − Kz = μJx_GN | proven | U4, U4-NEG |
| K-part of the z-flow → gradient flow as μ → 0 | proven | U7 |
| gᵀx_K, gᵀd ≥ 0 (β = 0); e-cotangent fails | proven / witness | U5, U5-WIT |
| ρ ≤ ρ_proj = virtual share of the descent rate | proven | T21 |
| Grown-GN virtual term = (1/μ)GJᵀHK_E r_⊥ ≠ d; "x_GN + d" is not a single projection | proven | U6, U6-NEG |
| Swap gain driven by r_⊥ | proven (= v3 T7b) | U8, U8-NEG |
| s\* maximises the GN-model decrease | proven | U9 |
| The base SGD step is the K-part | interpretation (exact up to O(μ) via T20b, ignoring minibatch and momentum differences) | — |
| λ_faithful = K | assumption A13 (lazy correction) | — |

**Corrections to the framing proposed before this analysis:**
- "K_E = 0 gives the GN step" is false. K_E = 0 gives the K-part (≈ SGD); K_E → K gives x_K.
- "Update = SGD + λ(x_GN + d)" is rejected (T22).
