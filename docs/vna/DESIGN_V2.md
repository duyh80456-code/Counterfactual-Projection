# VNA pre-coding design decisions (v2, 2026-10-14)

Sources:

- Four survey agents covered: selection, efficient compute, baselines and code, and theory verification.
- "Derived" marks agent or own algebra that still needs a written proof.
- [unverified] marks items not confirmed from the primary source.

## 0. Core alignment: expansion ⇄ projection

- The DF/VNA update transfers exactly V = ‖J_Eᵀ z‖² = zᵀ K_E z of first-order gain, where **z = R K g**.
- z is g filtered to the directions O can move: z_i = λ_i/(λ_i+μ) · g_i in K's eigenbasis.
- TINY maximizes the raw expansion gain U (with g, after removing what the same layer can do). This is **misaligned** with what the projection delivers.
- **Fix: a projection-aware selection rule.** Use TINY's closed form with the cotangent g replaced by z, i.e. build N_z from z.
  - V(A) is a generalized Rayleigh quotient: max Σ a_kᵀ N_z N_zᵀ a_k / (a_kᵀ S a_k). The top-r generalized eigenvectors of (N_z N_zᵀ, S) give the answer.
  - Cost: one extra backward pass with z as the cotangent, plus the d×d eigenproblem (d = C_in·k²).
  - Do NOT apply TINY's existing-weight projection by default; R already handles O's overlap. Keep it as an ablation.
- SENN's expansion score measures the *complement* (the part of g outside O's tangent space). Log V/U and the SENN score side by side.
- This gives a candidate technical contribution: **a growth criterion that is optimal for transfer rather than for installation**.

### Selection rules to implement and ablate

| Rule | Definition |
|---|---|
| (a) `tiny_U` | GroMo default |
| (b) `tiny_V` | N_z with z = R K g, averaged over several batches; generalized eigenproblem with S (pseudo-inverse sqrt with threshold) |
| (c) `greedy_V` | Pool = random ∪ top eigenvectors of (b) ∪ sign flips. Score c_j = J_{E_j}ᵀ z using the true ReLU; Gram G_ij. Greedy OMP / pivoted Cholesky on cᵀ(G + νI)⁻¹c until it reaches θ (0.5–0.8) of the pool gain. Plain top-k by V picks duplicates. |

ReLU handling:

- Score both a and −a under the true ReLU and keep the better one. Optionally use antisymmetric pairs (exact linear reproduction at 2× neurons).
- Fix ‖A‖ to match existing neurons, since V ∝ ‖A‖².

## 1. Efficient computation (current CG version ≈170× an SGD step; too slow)

Exact parameter-space identity (derived, via push-through J^T R = (F + μ)⁻¹ J^T):

  **d = (F + μ)⁻¹ C_OE C_EO (F + μ)⁻¹ ∇θ_block**

where F = JᵀJ and C_OE = Jᵀ J_E.

| Variant | Method | Est. cost per correction | Use |
|---|---|---|---|
| A. K-FAC (default for training) | F ≈ A⊗G per layer; C_OE ≈ cross-covariances of layer inputs and backpropagated grads between O and E (same statistics TINY uses); EMA; eigendecomposition every T corrections | ~1–2 SGD steps → k ≥ 5–10 | main arm |
| B. Exact with output sketch (accuracy reference) | s = 4–8 Monte Carlo / top-k Fisher outputs per sample, batch 64 → n′ ≈ 256–512; vmapped VJP gives J and J_E (bf16, chunked); Cholesky of K + μ | ~3–6 SGD steps → k ≈ 20–50 | reference / ablation |
| C. Matrix-free | Nyström-preconditioned CG (rank 20–50, reused ~10 corrections), warm start, 5 iterations | ~10–15 SGD steps → k ≥ 50 | fallback |
| lite | scalar R | ~2 SGD steps | cheap baseline (biased toward large-eigenvalue directions) |

Other decisions:

- **Output metric:** move from Euclidean logits to the **GGN/Fisher metric** H = diag(p) − ppᵀ. This is consistent with K-FAC and with cross-entropy.
- **BatchNorm:** eval mode, or batch statistics frozen as constants via `functional_call`. Never update running statistics. LayerNorm (DeiT) needs nothing.
- **Libraries:** torch.func for B and C; curvlinops or ASDL for the K-FAC factors, plus custom O–E cross-factor hooks.
- **Validation step:** before going to GPU, compare the cosine of d from A and B against 30-iteration CG on the CPU toy model.

## 2. Theory to state (corrected)

1. **DF error bound** (Zhang, Chen, Vanden-Eijnden, Peherstorfer, arXiv 2404.01145, Prop. 1):
   ‖u − û‖ ≤ e^{Ct}‖u₀ − û₀‖ + e^{Ct}∫e^{−Cs} ε(s) ds.
   - Lasser–Lubich (arXiv **2002.00624**, *not* 1910.04102), Lemma 3.10 is the unitary case with no exponential factor.
   - Derived improvement: for convex loss with K_tot = K + K_E positive definite, the one-sided Lipschitz constant is 0 in the K_tot⁻¹ inner product, so error ≤ ∫ε ds.
   - Derived defects:
     - DF: ε ≤ μ/(λ_min(K)+μ)·‖v*‖.
     - VNA: ε ≲ 2μ‖K_E‖/(λ_min(K)+μ)·‖g‖.
   - Valid only for the **frozen-kernel (lazy) reference flow**.
   - On the batch with full row rank and μ → 0 the bound is trivial; the real content is μ > 0 and off-batch.
2. **Fictitious Space Lemma** (Xu / Zikatanov; Chen's note): B = Π B̃ Πᵀ gives κ(BA) ≤ c₁/c₀. For P = I + ΠΠᵀ (derived):
   - κ(PH) < κ(H) ⇔ κ(H + MMᵀ) < κ(H), with M = H^{1/2}Π.
   - Low-rank limit: λ_min ≤ λ_{rC+1}(H).
   - The cleanest form to state is κ_eff(K + K_E) < κ_eff(K).
   - H is singular, so use effective condition numbers or Park–Xu's semidefinite theory (2509.07179).
3. **Convergence:**
   - Bounds: I ⪯ P ⪯ (1 + ‖K_E‖/(4μ))I, so **μ > 0 is required**.
   - Under PL, linear rate with a time-varying bounded preconditioner (Radhakrishnan–Belkin–Uhler 2009.08574; Bottou–Curtis–Nocedal 1606.04838 Assump. 4.3).
   - Pitfall: P and g computed on the same minibatch violates the uncorrelatedness assumption. Prefer refreshing A and μ on separate batches, and note the bias.
4. **Implicit bias / off-batch:**
   - Lazy, fixed P: converges to the min-‖θ − θ₀‖_{P⁻¹} interpolant.
   - The predictor kernel is K_P(x, x′) = J(x) P J(x′)ᵀ, which equals K(x, X)(I + R K_E R K) on batch inputs. This is a **Nyström extension of K_E through K, not K + K_E**. Do not claim the grown model's generalization.
   - With refreshed P there is no clean characterization.
5. **Folklore correction:**
   - "Zero first-order gain at a stationary point" holds for *splitting-type* growth (Liu et al. 2019, Thm 2.2 expansion), **not** for zero-fan-out growth (GradMax).
   - Our old negative statement concerns the *projected update in range(Jᵀ)*, which is still correct. Phrase it precisely.
   - The cross-entropy saturation bound is elementary: ‖p − e_y‖ ≤ √2(1 − p_y) ≤ √2·ℓ.

## 3. Baselines and protocol (CIFAR-100, ResNet-18, T4×2)

Reference numbers [unverified for our exact recipe]:

- ResNet-18 (CIFAR stem), 200 epochs SGD: 76–78. Our runs reached about 77.3 validation.
- ResNet-20: 68.8. ResNet-56: 72.6. MobileNetV2 x1.0: 74.2. DeiT-Tiny from scratch: about 65–70 and recipe-sensitive.
- Seed std 0.2–0.4 (CNN), ≥ 0.5 (ViT). Use ≥ 3 seeds; use 5 seeds if the gain is < 0.5 pp.

Baseline set, with official code where possible:

1. SGD.
2. SGD compute-matched (equal wall-clock).
3. SWA/EMA (`torch.optim.swa_utils`, free; run `update_bn`).
4. SAM ρ 0.05–0.1 (davda54/sam, MIT; about 2× cost, so also needs an equal-wall-clock SGD).
5. DBB / ACNet (DingXiaoH repos; merge before test).
6. Muon (KellerJordan/Muon) or Shampoo (facebookresearch/optimizers).
7. CS-KD (alinlab/cs-kd), and KD reported separately.
8. TINY real growth (gromo) as the ceiling, reported apart.
9. **Ablations of our method:**
   - K_E = 0 (same solver, plain GN/NGD);
   - random-A;
   - spectrum-matched S = h(K) (Geifman-style);
   - `tiny_U` vs `tiny_V` vs `greedy_V`;
   - lite vs A vs B.
- Skip, as related work only: RepOpt (VGG-only code), NetAug (code unverified), Bypass / NORTH* / SENN / Geifman (no code), K-FAC-PyTorch (stale, no license).
- Report: equal epochs, equal wall-clock, time-to-target, extra passes per step. Final model, official test set.
- Budget: about 1.5–2.5 h per ResNet-18 run on a T4. 3 seeds × 8 arms ≈ 25–35 GPU-hours, and SAM roughly doubles its share.

## 4. Coding order (proposed)

1. Finish Phase 0 (running). Then add `tiny_V` and greedy_V diagnostics on the same checkpoints (cheap).
2. Implement variant B (exact with sketch, GGN metric) and variant A (K-FAC with cross-factors). Validate both against CG on the CPU toy.
3. Stage 2a training arms:
   - sgd;
   - sgd_cm;
   - vna (A, `tiny_V`);
   - vna (A, `tiny_U`);
   - vna_random;
   - gn (K_E = 0).
   - Run seeds 0 and 1, then expand.
4. Add the baselines (SWA free, SAM, DBB) for the go case.

## 5. Review round (4 agents, 2026-10-14): required changes before coding

### 5.1 Must-fix

1. **Wrong control.** K_E = 0 makes d = 0, so it controls for nothing. Replace it with an **NGD/GN control**:
   - Same solver, same k, target −K g, i.e. d_ctrl = J^T R K g.
   - **Norm-matched** to the VNA correction at each step.
   - Also keep random-A (with ‖K_E‖ / trace matched) and the spectrum-matched S = h(K).
2. **Effective-LR confound.** In a BN ResNet, conv weights are scale-invariant, and d is not orthogonal to W. VNA can therefore change weight norms and, with them, the effective LR.
   - Fix (default): for every BN-followed conv weight, project d onto the tangent space of the sphere, d ← d − (⟨d, W⟩/‖W‖²) W.
   - Log ‖W‖ per layer in every arm.
   - Give SGD and VNA the same LR/WD grid.
3. **GGN-consistent formulas** (identity verified numerically, max error 8e-15 / 6e-15):
   - F = J^T H J; C_OE = J^T H J_E; g = p − y; H_i = diag(p_i) − p_i p_iᵀ.
   - d = (F+μ)⁻¹ C_OE C_EO (F+μ)⁻¹ J^T g.
   - Function-space form: d = J^T (HK+μ)⁻¹ H K_E H K (HK+μ)⁻¹ g.
   - Do NOT define K_H = J H J^T. No H^{±1/2} is needed.
   - Use one normalization (1/n) throughout.
   - Use the true Fisher / GGN (curvlinops type-2, or MC-sampled labels). Never the empirical Fisher.
4. **K-FAC structure.**
   - At B = 0 the A part of J_E is exactly 0, so Q = params(B).
   - Block-diagonal factoring of C_OE drops all coupling to conv1, which makes d exactly 0 on W1.
   - **Compute C_EO u = J_E^T H (J u) and C_OE v = J^T H (J_E v) exactly** with JVP + VJP. Use K-FAC/EK-FAC only for (F+μ)⁻¹. EK-FAC gives exact Tikhonov damping in the eigenbasis.
   - Cost: about 3–4 SGD steps per correction, plus factor building at refresh. Use eval-mode BN, folded.
   - Validation: cos(d_KFAC, d_exact-CG) on a tiny ResNet across a μ sweep. Ablate dropping the conv1 cross term; the W1 cosine should collapse.

### 5.2 Strongly recommended

5. **Mechanism experiment** (answers the "growth is empty on-batch, so benefit must come off-batch" critique):
   - Over short windows, compare VNA's function velocity on **held-out** inputs with the velocity of a really installed TINY-grown model: cosine and loss gap.
   - Also run a wide-net check (e.g. WRN-28-10). If VNA's gain shrinks there, that supports the expressivity story; if not, it points to preconditioning.
6. **Multi-block joint K_E.** Use K = Σ K_l and K_E = Σ K_E^l with **one shared solve**, and allocate r_l in proportion to the measured V_l. This is likely the best V per FLOP.
7. Report curves, not only final accuracy, to show whether the gain is acceleration or persists to the end.

### 5.3 Promising extension (VNA+; decide after Stage 2a)

8. **Shadow-B with periodic absorb.**
   - Keep virtual B (and optionally A) as shadow parameters trained on O's residual during training. Never deploy them.
   - Every T_a steps, project the shadow output h into O (dθ = J^T R h) and reset or decay B.
   - With T_a = 1 this is (unsymmetrized) VNA [derivation to verify]. T_a > 1 adds co-adaptation of the virtual features, fixing the B ≡ 0 first-order limitation.
   - Trade-off: it adds train-time parameters (like DBB or NetAug), which weakens "never grow". Present it as a VNA+ variant; sweep T_a ∈ {1, 4, 16, 64}; log the absorbed fraction.
9. Ablations rather than main method:
   - A selection: TINY vs GradMax vs OMP vs learned A (a few steps maximizing V).
   - Infinite-width (arc-cosine) K_E.
   - Adam/momentum placement of d.
   - Linear-expansion VNA (a pure preconditioner, linked to RepOpt/ExpandNets).

### 5.4 Experimental protocol updates

- Main CNN table: **5 seeds**, paired by seed. Report mean ± std, the paired difference with 95% CI, and Holm-corrected tests. Ablations: 3 seeds. DeiT: 8–10 seeds, or drop it.
- Hyperparameters chosen on a 45k/5k split; final model; official test set once. Consider batch 128.
- Strong-recipe check: cosine schedule plus label smoothing.
- An equal-compute SGD arm (1.3× epochs, the SAM-paper convention) is cheap. Include it in the main table, or at least in the appendix.
- Cost columns (RepOpt/NetAug style): train params, deploy params, GPU-hours, peak memory.
- Closest optimizer-side competitor is RepOpt (VGG code). Consider RepOpt-VGG on CIFAR-100 as a head-to-head.
- Scale: CIFAR-10/100 × 3+ CNNs (+ ViT) plus Tiny-ImageNet / ImageNet-100 (128 px, pre-resized, about 2–4 h per run) is a solid Q1 package. A top venue realistically expects ImageNet-1k.
- Reviewer-simulated scores (ICLR scale):
  - +0.5pp, CIFAR only: 3.
  - +1.5pp, CIFAR only, with clean controls: 5–6.
  - +1.5pp CIFAR plus ≥ 1pp on ImageNet-100, multiple seeds: 6–7.
  - 8 additionally needs the off-batch theory, or evidence that VNA closes most of the gap to TINY growth.

## 6. Phase 0 result (seed 0, ResNet-18 / CIFAR-100, batch 64, commit 44d1990; 2026-10-14)

- **Training.** SGD reached 75.5% on the 5k holdout, with train accuracy 99.95% and train loss 0.0145. About 49 s per epoch on T4 fp32.
- **Formal decision: no-go.**
  - `significant_vs_shuffled` was false at all 80 checkpoint/site pairs.
  - `rho > 0.3` was never reached.
  - The full V and derivative criteria passed 0–1 of 21 rows.
- **The shuffled-label null is miscalibrated.** It compares raw λ₁, but shuffled labels have a much larger residual. At epoch 200, λ_full ≈ 0.035 vs λ_shuffled ≈ 0.62. The null must be residual-norm matched before G1 can be trusted:
  - normalize λ by ‖g‖²; or
  - permute residual rows across samples; or
  - use a random residual with the same per-sample norm in the softmax tangent.
- **ρ is small, and the signal exists only in a window.**
  - Before epoch 100 (LR 0.1): ρ < 0.01.
  - Epochs 100–150 (LR 0.01): ρ peaks at 0.02–0.05 for r = 16 and 0.12 for r = 512 (site stages.3.blocks.0, epoch 140).
  - After the second LR drop at epoch 150: U and V go to about 0. This is consistent with the CE-saturation bound at interpolation.
- **TINY A is worse than random A.** V_TINY / V_random is about 0.1–0.6 at almost all points, and exceeds 1 at only one point (stages.3.blocks.1, r = 4, epoch 180).
  - Phase 0 used the tiny_U selection.
  - At epoch 100 the ratio is 0 for every site. This is suspicious; check for a bug.
- **Cost per correction per site**, in batch-64 SGD-step units:
  - exact, cold start: about 47;
  - exact, warm start: about 38;
  - lite: about 2.7.
  - Lite V and ρ are on a different scale (hundreds to thousands), so they are not comparable.
- **Rank.** V keeps growing up to r = 512 (roughly √r), with no saturation at 16.
- **Implications.**
  1. Stage 2a, as configured (start at epoch 5, r = 16, μ_rel = 1e-2), would very likely return a null result. Hold it.
  2. ResNet-18 on CIFAR-100 interpolates (100% train accuracy). There is no persistent expressivity bottleneck, so VNA's premise holds only during the epoch 100–150 window. A capacity-limited setting fits the story better: a thin ResNet (for example width/4), ImageNet-100, or other regimes where training does not reach interpolation.
- **Cheap follow-ups, on existing checkpoints only, before any training run:**
  - a residual-matched null;
  - a sweep of ρ over μ_rel ∈ {1e-2, 1e-3, 1e-4} at epochs 100–160;
  - tiny_V vs tiny_U vs random A at epochs 120–140;
  - joint K_E over all blocks;
  - the epoch-100 ratio bug;
  - repeat Phase 0 on a width/4 ResNet-18.

## 7. Điều chỉnh thuật toán cho bối cảnh CL / mất plasticity (2026-10-14)

### 7.0 Phát hiện lý thuyết: lemma chiếu lồng nhau (đã kiểm tra bằng số, `scratchpad/nested.py`)

- **Phát biểu.** Chiếu Dirac–Frenkel bước **Gauss–Newton** của model đã grow lên model cố định cho ra đúng bước GN của model cố định. Lý do: range(J) ⊆ range([J, J_E]), nên P_J · P_[J,J_E] = P_J.
- **Kiểm tra bằng số.** Sai lệch tương đối khoảng 1e-7–1e-10 khi μ → 0. Khi μ > 0, sai lệch chỉ ở mức O(μ).
- **Hệ quả.**
  1. VNA chỉ có nghĩa với động lực **bậc nhất** (gradient flow: SGD, momentum). Mục tiêu của nó là −(K + K_E)g. Với optimizer kiểu GN/NGD chính xác, neuron ảo **không thêm gì**.
  2. VNA không phải optimizer nhanh hơn GN. Nếu VNA hơn được GN/NGD (ENGD, K-FAC, OCAR, ARROW), lợi thế chỉ có thể đến từ **implicit bias hoặc khả năng tổng quát hoá**, tức hướng nào được học trước, và từ chi phí. Lợi thế đó không đến từ tốc độ giảm loss trên batch.
  3. **PINN bị hạ ưu tiên.** Ở đó mục tiêu là đưa residual tới độ chính xác cao, và ENGD/K-FAC-PINN (tức GN) là state of the art. Theo lemma, VNA không có lợi thế cấu trúc nào trước họ. B6 chỉ giữ lại như phép đo phụ.
  4. **CL: phải có hai loại đối chứng NGD.**
     - (a) NGD cùng norm, J^T R K g: cô lập phần đóng góp của K_E.
     - (b) Optimizer GN/K-FAC đầy đủ (kiểu OCAR), so ở cùng compute.
     Paper phải nói rõ VNA nhắm vào chế độ bậc nhất.
  5. **GIR quan trọng hơn VNA preconditioner.** GIR thật sự cài thêm hướng mới vào range(J), nên không bị lemma này giới hạn.

### 7.1 Các thay đổi cụ thể cho CL

1. **Refresh theo sự kiện, không theo epoch.**
   - Phát hiện task mới mà không cần nhãn task: loss trung bình trượt (EMA) vượt EMA dài hạn cộng kσ, hoặc dùng ranh giới task của benchmark (phải ghi rõ dùng cách nào).
   - Refresh A ngay sau ranh giới, rồi mỗi N bước trong task.
2. **Bật/tắt theo ρ.** Chỉ áp correction khi ρ (ước lượng ở lần refresh gần nhất) > τ. Kỳ vọng correction dày ở đầu task và tự tắt khi residual giảm.
3. **Dữ liệu cho phép tính kernel ở benchmark online.** Dùng một cửa sổ trượt m mẫu gần nhất (32–128) của luồng hiện tại, và khai báo rõ. Các đối chứng NGD và GIR dùng cùng cửa sổ đó. Không replay task cũ.
4. **Solver exact, gộp mọi layer.** Mạng nhỏ nên không cần K-FAC. K_E gộp tất cả các cặp lớp ẩn và giải chung một lần.
5. **Ghép với optimizer.**
   - Benchmark dùng SGD: như hiện tại.
   - Benchmark dùng Adam: cộng λ·d vào gradient trước Adam, với cùng quy tắc khớp norm cho đối chứng NGD. Ghi rõ đây là heuristic; đặc tính PSD chỉ đảm bảo trong không gian tham số trước Adam.
6. **Lớp chuẩn hoá.**
   - MLP không norm: bỏ phép chiếu lên mặt cầu.
   - LayerNorm: trọng số đứng trước LN bất biến theo tỉ lệ của toàn ma trận, nên chiếu d lên tiếp tuyến của ‖W‖_F.
   - Luôn log độ lớn trọng số, vì đây là một nguyên nhân đã biết của mất plasticity.
7. **GIR (growth-informed reset):**
   - Chọn unit theo đúng tiêu chí và nhịp của CBP (utility, maturity threshold, replacement rate), để ablation chỉ khác nhau ở bước "reset thành gì".
   - Trọng số vào = A từ `tiny_V`, lấy m vector riêng đầu, **đổi thang về đúng phân phối khởi tạo** (cùng norm hàng với He/Kaiming init), tránh làm activation bão hoà. Trọng số ra = 0. Bù bias theo kiểu CBP: cộng (activation trung bình × trọng số ra cũ) vào bias lớp sau.
   - **Ablation chính:** cùng tiêu chí chọn unit, reset ngẫu nhiên so với reset bằng `tiny_V`.
8. **Kết hợp VNA + GIR.** GIR cài m hướng tốt nhất. VNA dùng các vector riêng tiếp theo (từ m+1 đến m+r) làm neuron ảo, để hai phần không trùng nhau.
9. **Phần giữ nguyên:**
   - công thức lõi d = J^T R K_E R K g;
   - metric GGN dùng Fisher thật;
   - `tiny_V`;
   - đối chứng NGD cùng norm, random-A và tiny_U;
   - null có cùng độ lớn residual.

### 7.2 Thay đổi câu chuyện

- **Cũ:** "VNA chữa các hướng chậm khi kernel sụp."
- **Mới:** "Tín hiệu grow, dùng mà không grow, theo hai cách:
  - (i) GIR thay capacity đã chết bằng đúng những neuron mà phương pháp grow sẽ thêm vào;
  - (ii) VNA cho phần capacity còn sống được train như model đã grow (trong chế độ bậc nhất).
  Kèm lemma chiếu lồng nhau, xác định rõ khi nào neuron ảo vô dụng (GN chính xác) và khi nào cần cài thật (kernel sụp hẳn)."
