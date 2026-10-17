# Prompt cho agent: cài thuật toán VNA–GIR bản cuối (lý thuyết v3) và chạy pilot quyết định

**Base:** HEAD hiện tại (sau commit 2d4114b, preflight bị dừng vì `NO_FEASIBLE_K`). Làm trên branch mới `vna-v3-pilot`.

**Tài liệu kèm theo** (copy vào `docs/vna/`):
- `THEORY_FINAL_v3.md`: tài liệu có thẩm quyền. Giả thiết A1–A12, thuật toán §2, định lý T1–T18.
- `unified_gn.md`: định lý T19–T24. Đổi tên arm `kernel_self`; quy tắc λ ở §5.
- `final_check_v3.py`, `check_unified_gn.py`: các phép kiểm số tham chiếu.

Khi prompt này và tài liệu khác nhau, **tài liệu thắng**. Ghi lại mọi chỗ lệch.

**Prompt này thay thế** P1–P6 của `VNA_agent_prompt_parallel_run.md` và phần GIR/VNA của `VNA_agent_prompt_pilot_b5.md`. Các arm đối thủ P7–P10 (`VNA_agent_prompt_competitors.md`) giữ nguyên và có thể chạy song song, vì chúng không dùng VNA.

---

## 0. Nguyên tắc bắt buộc

1. **Không đổi công thức.** Mọi đại lượng tính đúng như §0 và §2 của THEORY_FINAL_v3. Nếu thấy phải đổi, dừng lại và báo; không tự sửa.
2. **Không có `assert` nào được làm sập một run dài.** Lỗi kiểm lúc chạy thì ghi cờ vào log và đi tiếp. Riêng test (pytest) thì phải fail thật.
3. **λ = 0 phải cho kết quả trùng bitwise** với baseline (iCaRL, BP, CBP), ở mọi arm.
4. **Phép đo Stage B giữ nguyên** (giải chính xác, `OutputFactor`). Chỉ đường train dùng solver xấp xỉ.
5. **Mọi thay đổi so với spec cũ** ghi vào `docs/vna/PREREGISTRATION_CHANGES.md` (danh sách ở §8) trước khi bấm chạy, kèm thời điểm. Hiện chưa có kết quả train VNA hay `kernel_self` nào.
6. **Không cần smoke riêng.** Pytest chạy local trên dữ liệu giả. Preflight trên Kaggle là một phần của notebook full (§6) và không được raise.

---

## 1. Sửa lỗi đã biết (làm trước)

### 1.1 Correction phải là d joint, không tính riêng từng site
- **Hiện trạng:** `run_vna_training.py` (đường Stage 2a) tính một correction riêng cho mỗi site, mỗi site hai lần giải. Với 15 block của ResNet32 là 30 lần giải. Tổng các d riêng đúng về toán nhưng tốn gấp 15 lần.
- **Sửa:** mọi đường train dùng `vna_joint.joint_correction`: một z dùng chung, K_E = Σ_site K_E,site, một lần giải cho d.
- **Test T-joint:** trên model nhỏ ở float64, d joint bằng tổng các d riêng, sai lệch tương đối ≤ 1e-10.

### 1.2 Lỗi range(H): bỏ pseudo-inverse, dùng dạng push-through
- **Nguyên nhân:** khi softmax bão hòa, ngưỡng trị riêng theo từng mẫu cắt mất hướng. Sai lệch tăng từ 5e-13 (logit scale 5) lên 0,26 (scale 20), khớp với mức trôi 1e-5 → 0,11 quan sát từ task 2 đến task 10.
- **Sửa:** cả trong `OutputFactor` (phép đo) lẫn đường train, dùng dạng của T1(c, d), không cần điều kiện e ∈ range(H):
  - z = e − μ(HK + μI)⁻¹e
  - d = Jᵀ(HK + μI)⁻¹H·J_E·w
  - giải bằng LU hoặc CG, không pseudo-inverse.
- **Bỏ** cờ loại điểm `residual_in_range_H`; vẫn ghi giá trị lệch làm chẩn đoán.
- **Test T-push:** ở các logit scale {1, 5, 10, 20, 30}, z và d theo dạng mới khớp với lời giải bình phương tối thiểu xếp chồng (CG tol 1e-12), sai lệch tương đối ≤ 1e-8. Đối chứng âm: dạng pseudo-inverse cũ phải lệch > 1e-2 ở scale 20.

### 1.3 K tính theo iteration, không theo epoch
Số iteration mỗi epoch đổi theo task (40 ở task 1, khoảng 55 từ task 2 vì có thêm exemplar). Mọi K trong code, log và `protocol.json` đều tính theo **iteration**.

### 1.4 `choose_interval` không được raise
Thay bằng quy tắc ở §4.4. Không còn `ValueError`. Cũng không còn `RuntimeError: FULL RUN HELD`: preflight chỉ ghi kết quả rồi chạy tiếp.

---

## 2. Solver cho đường train (sửa `NO_FEASIBLE_K`)

- **Solver:** mọi hệ (F + μI)x = b và (F + βB + μI)x = b giải bằng CG trong không gian tham số.
  - Matrix-free, dùng JVP/VJP.
  - Dừng theo residual tương đối ρ = ‖b − Ax̂‖/‖b‖ ≤ 1e-3 (A8), tối đa 50 vòng.
  - Warm start từ nghiệm cùng loại của lần áp dụng trước, cùng site set, cùng task.
- **Log mỗi lần giải:** ρ cuối và số vòng. Hai giá trị này là đầu vào cho dung sai τ ở bước 4 (§3).
- **μ:** μ = max(μ_rel·tr(F)/n, 1e-12), trong đó tr(F) ước lượng bằng Hutchinson với 8 vector.
  - Ở preflight, chọn μ_rel nhỏ nhất trong {1e-3, 1e-2, 1e-1} sao cho ở 3 lần áp dụng đầu, Δ̂ (bước 4) ≤ 0,1·|ĝ| / ‖g‖.
  - Ghi μ_rel đã chọn vào `protocol.json`. Giá trị này cố định cho cả run.
- **Kiểm độ chính xác lúc chạy** (không fatal): ở 3 lần áp dụng đầu của task 1 và task 2, tính thêm d bằng CG tol 1e-8 rồi log `cos_cg_exact` = cos(d̂, d_exact).
  - Nếu cos < 0,99: tự hạ tol xuống 1e-4 cho phần còn lại của run, và log `solver_tightened = True`.

---

## 3. Thuật toán: cài đúng §2 của THEORY_FINAL_v3

Mỗi iteration: bước optimizer gốc như bình thường (iCaRL: SGD có momentum của PyCIL; BP/CBP: SGD của upstream).

**Mỗi lần áp dụng** (lịch ở §4.3), trên cửa sổ 𝒲 hiện tại:

**Bước 0. Cửa sổ.**
- CIL: đúng minibatch hiện tại, gồm dữ liệu mới, exemplar và augmentation, forward ở train mode.
- B5: N_w = 128 mẫu gần nhất (FIFO) nếu vòng train là online. Nếu code đo Stage B của B5 đã có định nghĩa cửa sổ thì dùng đúng định nghĩa đó.
- Ghi định nghĩa cửa sổ vào `protocol.json`.
- Lưu buffer running-stat của BN trước khi làm và khôi phục sau đó. Test A1 kiểm trùng bitwise.

**Bước 1. Residual.**
- e = e_N + ẽ_O, theo đúng §0 của tài liệu:
  - KD là `_KD_loss` của PyCIL: lấy trung bình, **không nhân T²**;
  - p_s là xác suất của **student**;
  - H và W cũng dựng từ xác suất student.
- Tính g = Jᵀe, g_O = Jᵀẽ_O.
- Giải z = z(e) và z_O = z(ẽ_O), rồi đặt z_N = z − z_O.
- B5 (MSE, không có lớp cũ): ẽ_O = 0, H = I/N, chỉ giải z.
- Lưu x_z = GJᵀe. Bước 6 dùng nó, và đây cũng là phần "tách" e = z + r⊥ của T20.

**Bước 2. Chọn A.** Làm mới A ở đầu mỗi task và sau mỗi 10 lần áp dụng; ngoài các lúc đó giữ A cũ.
- A₀ = tiny_V(z): top-r eigvec của bài toán suy rộng (N_z N_zᵀ, S_x), với S_x có ridge ρ_rel = 1e-6 (L9). Nếu có BN theo sau, dùng thống kê đã trừ trung bình (T14).
- Khi β > 0 (chỉ CIL): làm thêm bước luân phiên của T6, gồm β₀ = β_rule(A₀), y = HJG_{β₀}g, M_S. A = các eigvec suy rộng dương lớn nhất của (M_S, S_x).
- Dấu chọn bằng ReLU thật. Chuẩn hóa về chuẩn trung bình của neuron, như hiện tại.

**Bước 3. β_rule.** Chỉ CIL; B5 luôn β = 0. Làm đúng các nhánh a/b/c của tài liệu:
- χ² dạng đóng (L6);
- ε\* = V_O²/χ²;
- nếu leak(0) > ε\* thì bisection trên log β trong [β_suf·10⁻⁸, β_suf], n_b = 4 bước, warm start;
- β = đầu mút trên đã kiểm leak(β) ≤ ε\*.

Sau đó d̂ = d_rep + d_N.

**Bước 4. Chứng chỉ trên đúng vector sẽ áp dụng.**
- Tính ĥ = g_Oᵀd̂ và ĝ = gᵀd̂ trực tiếp, không qua solve.
- Tính dung sai τ_h, τ_g theo T9(e). Trong Δ̂ có hai chuẩn phổ, ước lượng mỗi cái bằng 5 vòng lặp lũy thừa.
- Nếu ĥ < −τ_h (CIL) hoặc ĝ < −τ_g: **bỏ qua** lần áp dụng này và log `skipped` kèm lý do.

**Bước 5. Áp dụng.**
- θ ← θ − s·d̂, **cộng ngoài momentum buffer** (A9; ghi d̂ vào buffer là sai, T10f).
- s tính theo quy tắc ở §4.2, và **chỉ tính sau khi đã qua bước 4** (T24 yêu cầu ĝ > 0).
- Weight decay không áp lên d̂.

**Bước 6. GIR.** Chạy sau bước 5. Chỉ xét slot hợp lệ (A12): kênh ra của conv1 trong basic block là hợp lệ; kênh đi vào phép cộng residual thì không.
- **Phần bù:**
  - r = ẽ_O ở CIL, r = e ở B5;
  - r⊥ = r − z(r), tính tại θ, dùng lời giải có sẵn từ bước 1.
  - Ghi chú: code B5 cũ dùng z; **đổi sang r⊥**, xem §8.
- **Ứng viên:**
  - hướng a_k = top-k eigvec của (N_{r⊥} N_{r⊥}ᵀ, S_x), với k = 4;
  - dấu σ ∈ {+1, −1};
  - bias b ∈ {0, −median(σXa_k)}, hoặc nếu có BN thì γ_v = σ, β_v ∈ {0, −median(σ x̂_k)} (T17b);
  - loại ứng viên có φ ≡ 0 trên 𝒲.
- **Điểm:** ℓ(v) = ½‖J_vᵀr⊥‖² / (λ_max(J_vᵀHJ_v) + μ), theo T13g.
- **Slot chết:** đếm lại A5 tại θ′, tức sau bước 5 (T13a). Với mỗi slot còn chết, tối đa S slot mỗi lần áp dụng:
  - v = argmax ℓ;
  - ghi a_v (cùng b_v, hoặc γ_v, β_v), đặt B_v = 0;
  - nếu có BN: running_mean = 0, running_var = 1;
  - **đặt bằng 0** trạng thái optimizer của mọi phần tử thuộc slot (a, b, γ, β, B);
  - deflation chính xác khi q nhỏ (B5), bản rẻ khi conv (T13c–e).
- **Slot sống:** `LIVE_SWAP = off`, không cài vào đường chạy pilot.
- **Mặc định S:** B5 S = 1. CIL S = 1, nhưng GIR gần như không kích hoạt vì 0% đơn vị chết; log số slot chết để xác nhận.

**Bước 7. Log** (`vna_log.csv`, mỗi lần áp dụng một dòng). Các cột:
- task, iteration, K đang dùng, K_next, `refresh_A`;
- ρ = V/‖g‖², `rho_proj` = V/‖FGg‖² (T21), V, V_O;
- ĝ, ĥ, τ_g, τ_h, Δ̂, skipped, `skip_reason`;
- β, nhánh của β_rule, β/β_suf, leak, ε\*, χ², κ = V_NS/V, Φ_N(d_N);
- s, s\*, `cap_binding` (∈ {s\*, norm, none});
- ρ và số vòng CG của từng lần giải, `cos_cg_exact` (khi có);
- `dead_before_step5`, `dead_after_step5`, `swaps`, ℓ(v), gain;
- thời gian tách theo phần: solve, tiny_V, chứng chỉ, GIR.

---

## 4. Setup (rút từ lý thuyết; đăng ký trước; áp **giống hệt** cho mọi arm cùng họ)

### 4.1 Hằng số
| Tên | Giá trị |
|---|---|
| r (số neuron ảo) | 16 |
| n_b | 4 |
| k_lo | 8 |
| S | 1 |
| ρ_rel (ridge của S_x) | 1e-6 |
| Làm mới A | đầu task, rồi mỗi 10 lần áp dụng |

**Site:**
- CIL: 15 block của ResNet32. Mỗi site là kênh ra của conv1 trong block, tức bản hiện tại.
- B5: lớp ẩn.

### 4.2 Bước s (thay cho λ cố định)
Bước áp dụng là θ ← θ − s·d̂, với:
- **Quy tắc chính (mọi arm):** s = min(s\*, ε‖g‖/‖d̂‖). Ở đây s\* = ĝ / d̂ᵀ(F + μI)d̂ là cực đại của mô hình GN (T24; d̂ᵀFd̂ là một JVP), còn số hạng thứ hai giới hạn hiệu chỉnh không dài hơn một bước SGD.
- **Quy tắc phụ (chỉ các arm `_lf` của B5):** s = min(s\*, K·ε/(1 − m)), với m là momentum của optimizer gốc (B5 dùng SGD không momentum, nên m = 0). Đây là λ_faithful của `unified_gn.md` §5, dựa vào giả thiết A13.

### 4.3 Lịch áp dụng (K thích nghi)
- Lần áp dụng đầu của mỗi task: ngay sau iteration đầu tiên của task.
- Sau đó: K_next = K_base · clip(ρ_ref / ρ_last, ¼, 4), với ρ_ref = 0,3 (ngưỡng của cổng G) và ρ_last là ρ của lần áp dụng vừa xong.
- Nếu lần vừa rồi bị bỏ qua: K_next = K_base.
- Làm tròn K_next thành số nguyên ≥ 1. K không vượt quá số iteration còn lại của task.

### 4.4 Chọn K_base (preflight, không raise)
- Đo t_VNA (wall-clock của một lần áp dụng với solver CG ở §2, trung bình 3 lần áp dụng đầu của task 1 và task 2) và t_iter (trung bình 200 iteration).
- Ngoại suy theo số lớp: t_VNA,task10 ≈ t_VNA,task2 × (c₁₀ / c₂) với c là số lớp, và ghi rõ đây là ngoại suy tuyến tính.
- **Tập ứng viên:** {50, 100, 200, 400, 800} ∪ {1, 2, 5, 10} × (iteration mỗi epoch của task 2).
- **Quy tắc:** K_base = ứng viên nhỏ nhất thỏa t_VNA,task10 / K ≤ t_iter, tức tổng thời gian không quá 2× baseline.
  - Nếu không ứng viên nào thỏa: chọn ứng viên lớn nhất, đặt `budget_exceeded = True`, **và vẫn chạy**.
- Ghi tất cả số đo và K_base vào `protocol.json`.
- Mọi arm trong cùng họ (VNA, `kernel_self`, random-A) **dùng chung K_base**, lấy theo arm đắt nhất (VNA có β), để lịch áp dụng giống nhau.

---

## 5. Các arm

Mọi arm là trường hợp riêng của cùng một cập nhật (`unified_gn.md` §6). Chúng chỉ khác nhau ở những gì ghi trong bảng; mọi thứ khác (cửa sổ, solver, lịch, quy tắc s, chứng chỉ, log) giữ nguyên.

| Tên | Khác biệt |
|---|---|
| baseline | λ = 0 (không áp dụng gì) |
| `kernel_self` | thay K_E bằng K: d = G·JᵀH·K·z = (FG)²g (T19c). Đây là arm "NGD cùng chuẩn" cũ, đổi tên. Cùng chứng chỉ, cùng quy tắc s. |
| `vna` | K_E từ tiny_V (bước 2) |
| `vna_rand` | A ngẫu nhiên Gaussian, chuẩn hóa sao cho tr(K_E) theo từng site bằng của tiny_V tại cùng lần áp dụng; làm mới cùng lịch |
| `vna_b0` | như `vna`, nhưng ép β = 0 (chỉ CIL) |
| `gir` | chỉ bước 6 trên slot chết (λ = 0 cho phần d); cần bước 1 để có r⊥ |
| `cbp_gir` | CBP của upstream giữ nguyên cách **chọn** đơn vị và lịch thay thế; chỉ thay **giá trị khởi tạo** của đơn vị mới bằng ứng viên GIR (r⊥, ℓ(v), B = 0, đặt 0 trạng thái optimizer). Tính r⊥ trên cửa sổ hiện tại tại đúng thời điểm CBP thay. |
| `cbp_gradmax` | như `cbp_gir`, nhưng hướng vào = top eigvec **Euclid** của N_e N_eᵀ (cotangent e, không chia S_x, không dùng r⊥), theo tinh thần GradMax (Evci et al., ICLR 2022). Cùng quy tắc dấu/bias, B = 0. |
| `*_vna_gir` | `vna` cộng bước 6 (phương pháp đầy đủ) |
| `*_lf` | quy tắc s phụ (§4.2) |

---

## 6. Pilot quyết định

### 6.1 B5 regression (CPU)
Dohare slowly-changing regression, cfg upstream `bp/relu`, giữ nguyên mọi tham số. Seed 0, 1, 2. Tổng 12 arm × 3 seed = 36 job, chạy tối đa 4 job song song.

| # | Arm |
|---|---|
| 1 | `b5_bp` |
| 2 | `b5_bp_kself` |
| 3 | `b5_bp_vna` |
| 4 | `b5_bp_vna_rand` |
| 5 | `b5_bp_gir` |
| 6 | `b5_bp_vna_gir` |
| 7 | `b5_cbp` |
| 8 | `b5_cbp_gir` |
| 9 | `b5_cbp_gradmax` |
| 10 | `b5_cbp_vna_gir` |
| 11 | `b5_bp_vna_lf` |
| 12 | `b5_bp_kself_lf` |

**Metric chính:** sai số trung bình của 50 task cuối.

**Ghi thêm:**
- tỉ lệ đơn vị chết theo task;
- số swap;
- với mỗi đơn vị vừa lắp: phần đóng góp ‖f_u‖ của nó ở 1, 5 và 20 task sau. Cột này dùng để theo dõi xem neuron lắp vào có bị ghi đè hay không.

### 6.2 CIL (GPU)
PyCIL iCaRL, CIFAR-100, ResNet32, B0 Inc10, seed (thứ tự lớp) 1993, memory 2000. Epoch đúng config gốc: task đầu 200, các task sau 170. Chạy đủ 10 task, có resume.

| Notebook | GPU0 | GPU1 |
|---|---|---|
| Q1 | `icarl` (λ = 0 qua runner mới) | `icarl_kself` |
| Q2 | `icarl_vna_b0` | `icarl_vna` |
| Q3 | `icarl_vna_rand` | `icarl_cflat` (tùy chọn, xem dưới) |

- **`icarl`:** chạy lại bằng runner mới với λ = 0. Kết quả phải trùng bitwise với `icarl__inc10__s1993` đã có (test T-bitwise). Nếu trùng thì job có thể dừng sớm và dùng lại số cũ; ghi rõ trong `protocol.json`.
- **`icarl_cflat`:** dùng code gốc C-Flat (https://github.com/WanNaa/C-Flat) gắn lên iCaRL của PyCIL, giữ siêu tham số gốc của họ cho iCaRL, chỉ ghi đè setup chung. Nếu tích hợp chưa xong trước khi bấm chạy, để GPU1 của Q3 trống và làm sau; không được chặn các arm khác.
- **Continual evaluation:**
  - accuracy trên **lớp cũ** (test) ở mỗi epoch của 10 epoch đầu mỗi task, sau đó mỗi 10 epoch;
  - accuracy trên lớp mới cùng các thời điểm đó;
  - ghi vào `continual_eval.csv`.
  - Từ đó tính min-acc và độ sâu cùng thời gian hồi phục của stability gap (De Lange et al., ICLR 2023).
- **Metric chính:**
  - average incremental accuracy (CNN và NME);
  - accuracy ở task cuối;
  - forgetting trung bình;
  - accuracy lớp cũ và lớp mới theo task;
  - GPU-giờ, và hệ số so với `icarl`.

### 6.2b Permuted MNIST, mạng nhỏ
- Cfg upstream loss-of-plasticity (online permuted MNIST) với **mạng nhỏ: 3 lớp ẩn, mỗi lớp 10 unit**, theo SWR (arXiv 2508.00212). Giữ nguyên mọi tham số upstream khác. Seed 0, 1, 2.
- Arm: `pm_bp`, `pm_bp_kself`, `pm_bp_vna`, `pm_bp_vna_rand`, `pm_cbp`, `pm_cbp_gir`, `pm_cbp_gradmax`, `pm_cbp_vna_gir`.
- Cửa sổ: FIFO 128 mẫu gần nhất. β = 0 (không có lớp cũ theo nghĩa KD).
- **Metric:** accuracy online trung bình của các task trong nửa sau; tỉ lệ đơn vị chết; số swap.

### 6.2c Online CIL (single pass)
Đây là setting chuẩn mà lý thuyết dự đoán thuận lợi nhất cho VNA: model không bao giờ fit hết dữ liệu, và ngân sách bước rất ít.
- **Protocol chuẩn:** CIFAR-100, 10 task, mỗi mẫu chỉ thấy một lần. Stream batch 10, replay batch 64. Bộ nhớ 2000 (và 1000 nếu kịp). Backbone ResNet-18 đúng như codebase gốc dùng. Seed 0.
- **Code gốc:** dùng codebase chính thức của CCL-DC (Wang et al., CVPR 2024), vốn có sẵn ER, ER-ACE, OCM, OnPro. Agent kiểm URL repo và commit, rồi ghi vào `protocol.json`. Nếu không dùng được, dùng codebase chính thức của OCM hoặc OnPro. Chỉ ghi đè setup chung; không sửa thuật toán của họ.
- **Arm:** `er`, `er_kself`, `er_vna`, `er_vna_rand`, cùng `erace`, `erace_vna`.
  - ER không có KD, nên β = 0 và không có phần d_rep riêng.
  - Cửa sổ là batch stream gộp với batch replay (74 mẫu).
- **Lịch:** như §4.3 và §4.4. Mỗi task chỉ có khoảng 500 iteration nên K_base sẽ nhỏ; quy tắc ngân sách vẫn áp dụng.
- **Metric:** accuracy cuối, accuracy trung bình theo task, forgetting, learning accuracy (plasticity, theo CCL-DC), và GPU-giờ.

### 6.2c' Metric bổ sung cho mọi setting (đăng ký trước)
Lý thuyết dự đoán VNA chủ yếu **tăng tốc học ngay sau mỗi lần chuyển task**. Vì vậy ngoài metric cuối, mọi setting ghi thêm:
- **Anytime accuracy / AUC:** diện tích dưới đường accuracy (hoặc loss) theo iteration, tính trong từng task và trung bình qua các task. Đây là metric chuẩn của online CL (continual evaluation).
- **Learning accuracy:** accuracy trên task hiện tại ngay khi task đó kết thúc. Đây là metric plasticity, theo CCL-DC.
- **Tốc độ đầu task:** số iteration cần để loss của task mới giảm được 50%.

Các metric này phải nằm trong bảng chính cạnh các metric cuối, không chỉ ở phụ lục.

### 6.2c'' Các ràng buộc cấu hình thuận cho thuật toán (giữ trong phạm vi chuẩn)
- **Không thêm LayerNorm hay BatchNorm** vào các mạng MLP của B5 và Permuted MNIST. Dùng đúng kiến trúc upstream: ReLU trần, nơi đơn vị chết xuất hiện.
- **Permuted MNIST:** chạy đủ số task của cấu hình upstream. Không cắt ngắn, vì mất plasticity chỉ lộ ra ở chuỗi dài.
- **Online CIL:** giữ đúng backbone của codebase gốc. Nếu codebase có sẵn biến thể **reduced ResNet-18 (nf = 20)** theo protocol online chuẩn, chạy thêm biến thể đó làm cấu hình thứ hai và ghi rõ nguồn. Đây là mạng hẹp, đúng tiêu chí 2 của lý thuyết.
- **Online CIL, nếu còn GPU:** thêm biến thể **20 task** của CIFAR-100. Đây là cấu hình chuẩn có nhiều lần chuyển task hơn.

### 6.2d Thứ tự ưu tiên
1. B5 regression
2. Permuted MNIST (mạng nhỏ)
3. Online CIL
4. PyCIL offline (Q1–Q3)

Mục 4 được trình bày là phần "tính tổng quát". Lý thuyết dự đoán lợi nhỏ ở đó, vì model gần như fit hết dữ liệu trong phần lớn mỗi task. Báo kết quả của mục 4 trung thực như một đối chứng đã được dự đoán.

### 6.3 Tiêu chí quyết định (đăng ký trước, ghi vào PREREGISTRATION_CHANGES)

| Câu hỏi | So sánh | "Có tín hiệu" khi |
|---|---|---|
| Q-A: kernel ảo có giá trị? | `vna` vs `kernel_self` | B5: `b5_bp_vna` tốt hơn `b5_bp_kself` ở cả 3 seed (so cặp). CIL: `icarl_vna` hơn `icarl_kself` ≥ 0,5 điểm average incremental accuracy (CNN). |
| Q-B: tiêu chí grow có giá trị? | `vna` vs `vna_rand` | như Q-A |
| Q-C: GIR có giá trị? | `b5_cbp_gir` vs `b5_cbp` và vs `b5_cbp_gradmax` | tốt hơn cả hai ở ≥ 2/3 seed |
| Q-D: bảo vệ β có tác dụng? | `icarl_vna` vs `icarl_vna_b0` | forgetting thấp hơn, và stability gap nông hơn |

**Cách áp các câu hỏi cho các setting thêm:**
- **Permuted MNIST:** Q-A, Q-B, Q-C dùng cùng tiêu chí như B5 (so cặp theo seed; tốt hơn ở ≥ 2/3 seed).
- **Online CIL:** Q-A, Q-B so theo accuracy cuối và learning accuracy, ngưỡng ≥ 0,5 điểm.

CIL chỉ có 1 seed, nên kết quả CIL **chỉ là tín hiệu pilot**, không phải kết luận. Không đổi bất kỳ siêu tham số nào sau khi đã thấy kết quả pilot. Mọi thay đổi sau đó phải thành một mục mới trong PREREGISTRATION_CHANGES, kèm lý do.

---

## 7. Test bắt buộc (pytest, local, float64, model nhỏ)

1. **T-bitwise:** λ = 0 trùng bitwise baseline cho iCaRL (1 task, dữ liệu giả), BP và CBP (B5, vài task).
2. **T-joint** (§1.1) và **T-push** (§1.2), kèm đối chứng âm.
3. **T-cg:** CG tol 1e-3 có warm start; cos(d_CG, d_exact) ≥ 0,99 trên MLP nhỏ và ResNet8 giả; ρ cuối được log đúng.
4. **T-kself:** d của `kernel_self` = G·JᵀH·K·z = (FG)²g, sai lệch ≤ 1e-10.
5. **T-rand:** tr(K_E) của `vna_rand` bằng của tiny_V theo từng site, sai lệch ≤ 1e-8.
6. **T-beta:**
   - đủ ba nhánh a/b/c của β_rule, có dựng trường hợp χ² = 0, V_O = 0 và leak(0) > ε\*;
   - leak đơn điệu không tăng theo β trên lưới;
   - tại β đã chọn, leak ≤ ε\*.
7. **T-cert:**
   - ĥ, ĝ tính trên đúng vector được áp dụng;
   - dựng một trường hợp ĝ < −τ_g và kiểm lần áp dụng đó bị bỏ qua, model không đổi;
   - dựng trường hợp V_O = 0 với CG không chính xác và kiểm **không** bị bỏ qua nhầm (T18).
8. **T-step:**
   - s ≤ ε‖g‖/‖d̂‖ và s ≤ s\*;
   - s chỉ tính khi đã qua bước 4;
   - arm `_lf` dùng đúng K·ε/(1 − m).
9. **T-momentum:** dùng `torch.optim.SGD` thật (momentum 0,9, như PyCIL). Sau khi áp dụng d̂, momentum buffer không đổi; bước SGD kế tiếp giống hệt khi không có hiệu chỉnh, trừ phần do θ đã dịch.
10. **T-bn:** buffer BN trùng bitwise trước và sau một lần áp dụng (A1).
11. **T-gir:**
    - đếm lại slot chết sau bước 5;
    - tại lúc chèn với B_v = 0, output trên cửa sổ không đổi (sai lệch ≤ 1e-12);
    - trạng thái optimizer của slot bằng 0;
    - slot vào phép cộng residual bị loại (A12);
    - ứng viên có φ ≡ 0 bị loại;
    - trong arm `gir`, cotangent là r⊥ chứ không phải z (đối chứng âm: dùng z cho ℓ khác).
12. **T-gradmax:** hướng vào bằng top eigvec Euclid của N_e N_eᵀ; khác với hướng của GIR trên một ví dụ dựng sẵn.
13. **T-cbp:** `cbp_gir` và `cbp_gradmax` chọn đúng các đơn vị mà CBP gốc chọn, vào đúng thời điểm (so với CBP gốc cùng seed); chỉ giá trị khởi tạo khác.
14. **T-schedule:**
    - lần áp dụng đầu ở iteration 1 của mỗi task;
    - K_next theo đúng công thức §4.3;
    - K tính theo iteration;
    - preflight chọn K_base đúng quy tắc §4.4 và không raise khi không ứng viên nào khả thi.
15. **T-resume:** dừng giữa task rồi resume cho kết quả trùng bitwise với chạy liền, kể cả trạng thái warm start của CG, A hiện tại, K_next và FIFO của B5.
16. **T-theory:** port một tập con nhanh (dưới 60 giây) của `final_check_v3.py` và `check_unified_gn.py` vào `tests/test_theory_v3.py`, gồm cả các đối chứng âm, và chạy trong pytest.

---

## 8. PREREGISTRATION_CHANGES.md: các mục phải thêm (trước khi bấm chạy)

1. **Solver train:** CG tol 1e-3, tối đa 50 vòng, warm start, có kiểm cos và tự siết tol. Phép đo vẫn giải chính xác.
2. **d joint** thay cho tổng các correction riêng từng site. Đồng nhất về toán; lý do là chi phí.
3. **Dạng push-through** thay pseudo-inverse. Sửa lỗi range(H) do softmax bão hòa; bỏ cờ loại điểm.
4. **Lịch áp dụng:** K tính theo iteration; K thích nghi theo ρ (ρ_ref = 0,3); áp ngay đầu task; K_base chọn theo quy tắc ngân sách, không raise.
5. **Quy tắc bước s:**
   - chính: min(s\*, ε‖g‖/‖d̂‖);
   - phụ (`_lf`, chỉ B5): min(s\*, Kε/(1 − m)).
6. **Thuật toán bảo vệ và chứng chỉ theo THEORY_FINAL_v3:** β_rule (β_cert), chứng chỉ ĥ, ĝ với dung sai từ CG, quy tắc bỏ qua, cộng ngoài momentum, khôi phục BN, chọn A luân phiên khi β > 0.
7. **Đổi tên arm:** "NGD cùng chuẩn" thành `kernel_self`. Công thức không đổi (T19c); lý do là trọng tài phát hiện đây không phải natural gradient.
8. **GIR:**
   - cotangent đổi từ z sang r⊥ = r − z(r) (T20, T23);
   - chỉ slot chết, đếm lại sau bước 5;
   - đặt 0 trạng thái optimizer của slot;
   - thêm biến thể `cbp_gir` (CBP chọn, GIR khởi tạo).
   - Ghi rõ: kết quả B5 cũ (TINY thắng random 10/13) được đo với z.
9. **Control mới:** `cbp_gradmax` và `vna_rand` (trace khớp theo từng site).
10. **Tiêu chí quyết định ở §6.3**, cùng các setting và thứ tự ưu tiên ở §6.2b–d. Lý do chọn setting rút từ lý thuyết: residual tồn tại lâu (bổ đề nội suy), mạng hẹp, ngân sách bước ít, có đơn vị chết. PyCIL offline được giữ làm đối chứng đã dự đoán.
11. **Đã thấy nhưng không dùng để chọn:**
    - accuracy smoke 2 epoch (VNA 15,05 / NGD 21,3 CNN);
    - phép đo quick ở β = 0 (ρ_joint 0,26/0,42/0,58; tiny_V thua random 7/9 dòng);
    - các kết quả toy trong `unified_gn.md` §9 và `consistency_setup.md`.
11b. **Metric bổ sung và ràng buộc cấu hình ở §6.2c' và §6.2c''.**
- Lý do: chúng đo đúng cơ chế mà lý thuyết dự đoán (tăng tốc sau mỗi lần chuyển task).
- Tiền lệ cho việc chọn cấu hình thuận nhưng vẫn chuẩn, kèm lý do từ cơ chế:
  - SWR (Permuted MNIST với mạng 10 unit);
  - CCL-DC (CVPR 2024, online CL cho plasticity).
12. **Bỏ khỏi đợt này:** arm `tiny_V_euclid` (chỉ để chẩn đoán) và swap slot sống (`LIVE_SWAP = off`).

Mỗi mục ghi thời điểm, và câu "trước khi có bất kỳ kết quả train VNA/`kernel_self` nào".

---

## 9. Notebook

- **Dùng chung `kaggle_vna_parallel.ipynb`.** Cell đầu có `JOBS_GPU0`, `JOBS_GPU1`, `JOBS_CPU`, `RUN_TAG`; các cell khác giống hệt nhau giữa các notebook.
- **Thứ tự chạy:** pytest một lần, preflight (nằm trong job, không dừng), rồi full.
- **Các notebook:**
  - **P-B5:** chỉ CPU, 36 job của §6.1.
  - **P-PM:** 24 job của §6.2b, chạy CPU (hoặc GPU nếu upstream dùng GPU).
  - **O1:** GPU0 `er`, GPU1 `er_vna`.
  - **O2:** GPU0 `er_kself`, GPU1 `er_vna_rand`.
  - **O3:** GPU0 `erace`, GPU1 `erace_vna`.
  - **Q1–Q3:** như bảng ở §6.2.
- **Đầu ra mỗi job:** `outputs/<RUN_TAG>/<job>/`, gồm `summary.json`, `tasks.json`, `vna_log.csv`, `continual_eval.csv` (CIL), `protocol.json` (gồm μ_rel, K_base, định nghĩa cửa sổ, quy tắc s, commit, mọi số đo preflight) và `reproduction_gate.json` (`icarl`).
- **`scripts/collect_parallel.py`:** in các bảng của §6.2 và §6.3 (so cặp theo seed cho B5), cùng tổng thời gian VNA so với thời gian train, tỉ lệ bỏ qua và phân bố K theo task.

---

## 10. Báo cáo trước khi bấm chạy

1. Diff theo file, và log pytest của toàn bộ test ở §7.
2. Nội dung PREREGISTRATION_CHANGES.md đã thêm.
3. Notebook đã pin commit, và cell đầu đã điền sẵn cho P-B5 và Q1–Q3.

**Không có bước giữ lại chờ duyệt.** Preflight nằm trong mỗi job và tự chạy tiếp sang full. Mỗi job in ra ngay đầu log các số: t_VNA (task 1, task 2), giá trị ngoại suy, K_base, cờ `budget_exceeded`, μ_rel, `cos_cg_exact`. Gửi các số đó về ngay khi có, để supervisor kiểm trong lúc job vẫn chạy.
