# Prompt cho agent: giải chính xác (T29), vòng LM bên trong (T30), GIR một mô hình (T31), chẩn đoán NaN

**Base:** commit `d88abc4` trên branch `vna-v4-main`. Làm trên branch mới `vna-v4-exact`.

**Tài liệu kèm theo** (copy vào `docs/vna/`, thay bản cũ):
- `vna_gir_theory.pdf` cùng source LaTeX. Mục mới là **§9c** (T29, T30, T31, R1).
- `check_t29_31.py`: phép kiểm số tham chiếu.

Khi prompt này và tài liệu khác nhau, **tài liệu thắng**. Ghi lại mọi chỗ lệch.

---

## 0. Bối cảnh: lần chạy M1 vừa rồi cho thấy gì

Run `icarl_full__nobn__superclass__s0` có 3 lần áp dụng:

| Lần | VNA | GIR | Thời gian |
|---|---|---|---|
| task 1, it 1 | skip `cg_z` (z: 200 vòng CG, residual 0.124) | 0 swap | 80 s |
| task 1, it 1025 | skip `untrusted` (ρ = −23.9) | 29 swap slot sống | 3059 s, trong đó GIR 2987 s |
| task 2, it 2001 | ρ = **NaN** nhưng không skip; window check bác (L 5.93 → 7.49) | 26 swap slot sống | 2879 s, trong đó GIR 2529 s |

Sau đó, ở task 2 epoch 3, loss thành NaN và job chết.

- **μ sai thang.** μ = 6.9e-8 trong khi λ_E ≈ 30, nên ‖x_GN‖ lên tới 289–656. LM chỉ nhân μ với 2 và để giá trị mới có hiệu lực ở lần sau, mà mỗi task chỉ có 2–3 lần áp dụng.
- **Γ là nhiễu.** Γ được tính với μ đó, và CG cho Γ có residual 0.01–0.05. Ứng viên nào cũng có Γ > 0.
- **Neuron chết có thật.** Baseline có 0 neuron chết ở task 1 nhưng 3% ở task 2 và 7–8% từ task 7, dormant 11–15%. Setup giữ nguyên.

---

## 1. Nguyên tắc bắt buộc

1. **Không đổi công thức** ngoài những gì §9c quy định. Thấy cần đổi thì dừng lại và báo.
2. **Không chạy pytest toàn bộ, không chạy smoke notebook.** Chỉ chạy các test ngắn ở §8, trên local, mô hình nhỏ.
3. **Không có `assert` nào được làm sập một run dài.** Lỗi kiểm lúc chạy thì ghi cờ vào log rồi đi tiếp.
4. Mọi thay đổi ghi vào `docs/vna/PREREGISTRATION_CHANGES.md` (§9) **trước** khi chạy.
5. Setup chính (§3 của prompt trước) **giữ nguyên**: noBN, lr 0.05, B0 Inc5, 20 task, epoch, milestone, memory, thứ tự superclass. Không đổi lr.

---

## 2. Sửa lỗi ngay (nhỏ, làm đầu tiên)

1. **Cổng ρ.** Mọi giá trị ρ không hữu hạn (NaN, ±inf) phải tính là thất bại.
   - Thay `if reason is None and rho <= 0` bằng điều kiện "chỉ chấp nhận khi `math.isfinite(rho) and rho > 0`".
   - `_measure_ratio`: nếu loss tại θ − x không hữu hạn thì trả về ρ = −inf và phải khôi phục tham số. Thêm một kiểm tra cho việc khôi phục.
2. **Kiểm tra trạng thái sau mỗi lần áp dụng**, dù thành công, bị skip hay bị revert:
   - mọi tham số và mọi tensor trong optimizer state phải hữu hạn;
   - log `param_norm`, max |θ|, và chuẩn của từng site trước và sau.

   Nếu có giá trị không hữu hạn thì đặt cờ `nonfinite_after_apply`, khôi phục snapshot đầu lần áp dụng, rồi đi tiếp.
3. **NaN trong lúc train.** Không để job chết im lặng. Khi loss không hữu hạn:
   - ghi `diverged` vào `summary.json` cùng task, epoch, step;
   - lưu checkpoint của epoch trước đó;
   - đánh dấu job `FAILED_DIVERGED`. Notebook phải in trạng thái này ra.

---

## 3. Profiling trên commit hiện tại (trước khi sửa thuật toán)

Đây là cơ sở để chọn phương án ở §4.4, và để đối chiếu sau khi sửa. Chạy **một** lần áp dụng trên T4, noBN, task 2, iteration đầu (dùng resume từ checkpoint cuối task 1). Ghi vào `outputs/profile_d88abc4.json`:

- thời gian từng phần: preliminary (z, z_O), tiny_V, certificate (d_rep, d_N theo từng β), `gir_residual`, `baseline_gain`, mỗi lần CG của Γ, `_slot_jump`, `_slot_contribution`, `prepare_gir`;
- với mỗi CG: số vòng, residual, giây;
- thời gian một JVP và một VJP trên cửa sổ, đo ở cả float32 và float64. Ghi cả dtype thực tế của tham số và của vector trong CG;
- số hệ được giải trong lần áp dụng đó.

---

## 4. T29: giải trong không gian output (thay **mọi** CG trong engine và `diagnose`)

### 4.1 Dựng G = JJᵀ

- **Quy mô.** Cửa sổ có N = 128 ảnh, C = số lớp đã thấy, nên m = N·C (640 ở task 1, 12800 ở task 20).
- **Cách dựng.** Dùng cùng hàm forward như `HookProducts` hiện tại (BN ở train-mode, cùng tham số). Chia các vector cơ sở cotangent thành khối k cột (k tùy bộ nhớ, ví dụ 64):
  - `V = vmap(VJP)(E_k)`, cỡ k × P;
  - `G[:, khối] = vmap(JVP)(V)`, cỡ m × k.
- **Độ chính xác.** Forward/backward chạy float32, nhưng cộng dồn G bằng **float64**. Sau đó đối xứng hóa G ← (G + Gᵀ)/2.
- **Lưu trữ.** G float64 trên GPU (1.3 GB ở m = 12800). Ghi `gram_seconds`, `m`, `gram_dtype` vào log.

### 4.2 Metric M và các nghiệm

- **Metric.** M_β = H + β·H_old. Cả hai là khối C×C theo từng mẫu; H_old chỉ khác 0 trên các logit cũ, như `old_parts` hiện tại.
- **Phân rã metric.** Tính L theo từng mẫu bằng `eigh` của từng khối, bỏ trị riêng ≤ 0 (theo ngưỡng số). Ghép thành L có cỡ m × r.
- **Nghiệm.** Dùng dạng đối xứng của T29(b):

  y = (1/μ)·[c − L·(μI + LᵀGL)⁻¹·LᵀGc]

  Hệ r × r giải bằng Cholesky float64. Nếu Cholesky thất bại thì ghi cờ `chol_fail` và skip lần áp dụng.
- **Ánh xạ từ các hệ hiện tại sang c:**

  | Hệ hiện tại | c |
  |---|---|
  | `z` (vế phải g = Jᵀe) | e |
  | `z_O` | e_old |
  | `d` / `d_rep` / `d_N` (vế phải JᵀH·u) | H·u |
  | `gir_residual` | e_β |
  | diagnose B | e |

- **Các đại lượng tính từ y** (T29(c)):
  - Jx = G·y;
  - z = M·G·y;
  - ‖x‖² = yᵀGy, nên Δ = ‖x_GN‖;
  - xᵀFx = yᵀ·G·H·G·y (với F dùng H, không phải M_β);
  - B = cᵀGy;
  - r⊥^β = e_β − M_β·G·y_β.
- **Ánh xạ về tham số.** Chỉ khi cần tham số (bước VNA thật) mới gọi một VJP: x = Jᵀy.
- **Xóa:** `self.warm`, mọi warm start, `cg_tol`, `cg_iters`, `warm_reset`, lý do skip `cg_z`. Trong `projection/cg.py`, chỉ giữ hàm CG nếu còn nơi khác dùng (kiểm import graph); không còn thì xóa.

### 4.3 BetaRule

- Mỗi giá trị β cần một Cholesky của (μI + L_βᵀ·G·L_β), dùng chung G.
- Giữ đúng logic nhánh và bisection hiện tại.
- Log thêm `leak` của từng β. Với nghiệm chính xác, leak phải đơn điệu giảm theo β; vi phạm thì ghi cờ.

### 4.4 Khi m quá lớn

Không tự cài phương án dự phòng. Nếu profiling cho thấy `gram_seconds` ở m = 12800 lớn hơn 10 phút, hoặc vượt bộ nhớ, thì dừng lại và báo số đo. Mình sẽ quyết giữa Nyström PCG và dựng G theo khối.

---

## 5. T30: vòng LM bên trong mỗi lần áp dụng

- **Vòng lặp.** Bắt đầu từ μ = `self.mu` (mang sang từ lần trước; lần đầu dùng μ₀ hiện tại):
  1. Giải x(μ) theo §4.
  2. Tính pred = gᵀx − ½xᵀFx, đo ρ bằng một forward tại θ − x, rồi khôi phục tham số.
  3. Nếu ρ hữu hạn và ρ > 0 thì dừng với μ* = μ.
  4. Ngược lại đặt μ ← ν·μ, ν ← 2ν, rồi quay lại bước 1.
- **Giới hạn.** Tối đa 40 lần thử. Hết số lần thử thì skip lần áp dụng với lý do `lm_budget`. Theo T30(b), μ₀ = 1e-8 cần khoảng 7 lần thử.
- **Sau khi dừng:** μ cho lần sau = μ*·max(1/3, 1 − (2ρ − 1)³), ν = 2.
- **Dùng chung μ\*.** Toàn bộ phần còn lại của lần áp dụng dùng μ*: `_direction`, BetaRule, Δ = ‖x(μ*)‖, `prepare_gir` (ℓ_β, power iteration), r⊥^β, Γ.
- **Khi skip.** Nếu lần áp dụng bị skip (`lm_budget`, `chol_fail`, g = 0, non-finite) thì **GIR cũng không chạy**. Riêng dead slot: theo T28(b), swap dead slot không đổi hàm mạng, nhưng vẫn cần một mô hình đáng tin để chọn v, nên cũng skip.
- **Log:** `lm_tries`, `mu_trajectory`, `rho_trajectory`, `mu_star`.

---

## 6. T31: mọi quyết định GIR trong một lần áp dụng dùng một mô hình

- **Cố định đầu lần áp dụng:** θ, G, H_β, e_β, μ*. Không gọi `training_products` trên mạng đã sửa để tính Γ.
- **Swap dead slot** (T29(e), cập nhật chính xác):
  - cần J_{B_v}, tức Jacobian theo trọng số ra của v khi cài với B_v = 0. Tính bằng vmap JVP trên |B_v| vector cơ sở;
  - cập nhật G ← G + J_{B_v}J_{B_v}ᵀ.
- **Swap slot sống:**
  - lấy J_u (cột theo trọng số vào, bias và trọng số ra của u) và J_v tại θ, bằng vmap JVP;
  - với Ĵ = LᵀJ, gain bớt và thêm theo T29(d). Γ = −jump + q̂*(J) − q̂_sau theo T28(a), tính bằng cập nhật Woodbury hạng |u| + |v| trên Cholesky của (μ*I + LᵀGL);
  - f_u vẫn là đóng góp thật (forward có và không có u). e′ = e_β − H_β·f_u;
  - **test bắt buộc:** Γ theo Woodbury phải bằng Γ theo giải trực tiếp trên mô hình nhỏ (§8).
- **Tuần tự (T31(a)).** Sau mỗi swap được nhận, cập nhật trạng thái mô hình:
  - G ← G − J_uJ_uᵀ + J_vJ_vᵀ;
  - độ dịch output cộng thêm −f_u;
  - e ← e − H_β·f_u.

  Các bước nhảy j_L, j_KD vẫn đo trên mạng thật đã sửa, như hiện tại.
- **Ứng viên.** Ứng viên v vẫn dựng bằng `prepare_gir` tại θ, với residual r⊥^β cập nhật theo trạng thái mô hình hiện tại (dùng G đã cập nhật), trực giao theo S_x như cũ.
- **Hai giới hạn top-2 và `live_candidates = 4`.** Giữ nguyên trong lần này. Sau khi đo chi phí mỗi Γ, báo lại xem bỏ chúng thì tốn bao nhiêu. Mình sẽ quyết.
- **Log thêm cho mỗi swap:**
  - ‖a_v‖ so với trung vị của ‖a‖ các neuron cùng site;
  - tỉ lệ vị trí mà neuron mới kích hoạt trên cửa sổ;
  - `Gamma_woodbury`;
  - `n_jvp`.

---

## 7. R1: chỉ số reach

- Bỏ `reach` và `unreach` khỏi bảng chính và khỏi report.
- Giữ trong log: B, V, λ_E, tỉ số V/(λ_E·B), và `norm_w2` (chỉ để chẩn đoán).
- `diagnose()` cho baseline và đối thủ dùng solver ở §4 với μ = μ_diag. μ_diag là μ* của lần áp dụng gần nhất cùng task nếu có, ngược lại là μ₀. Ghi giá trị μ đã dùng.

---

## 8. Test ngắn (local, float64, mô hình nhỏ, chỉ các test này)

1. **T29 so với giải trực tiếp.** MLP và conv nhỏ, có và không BN, có H_old, β ∈ {0, 0.5, 5}. So các đại lượng sau với giải dày (dense) hoặc CG có tol 1e-12, sai lệch tương đối phải < 1e-8:
   - x, z, Δ, xᵀFx;
   - B, d_N(β), r⊥^β.
2. **T29(e).** Cài dead slot với B_v = 0. Logits và G tính lại từ đầu phải trùng với G + J_{B_v}J_{B_v}ᵀ (sai lệch < 1e-10).
3. **Γ Woodbury.** Γ theo §6 phải bằng Γ giải dày trên mô hình nhỏ (sai lệch < 1e-8).
4. **T30.** Với μ₀ = 1e-10, vòng lặp dừng và ρ* > 0. Ép L(θ − x) = NaN ở lần thử đầu thì phải tính là thất bại.
5. **Cổng NaN.** ρ = NaN không bao giờ cho `skipped = false`.
6. **Không còn import CG trong engine**, nếu §4.2 xóa được CG.

---

## 9. PREREGISTRATION_CHANGES.md: các mục phải thêm trước khi chạy

- **Solver** đổi từ CG cắt cụt sang giải chính xác trong không gian output (T29). Lý do: tính chính xác, để các định lý đúng cho code. Không phải tinh chỉnh theo kết quả.
- **μ** chuyển sang vòng LM bên trong (T30), VNA và GIR dùng chung μ*. Lý do: log M1 cho thấy μ chậm hơn yêu cầu 10⁸ lần.
- **Γ** được tính trong một mô hình (T31), đúng như phát biểu của T28. Code cũ tuyến tính hóa lại mạng cho từng ứng viên.
- **Cổng ρ** coi NaN là thất bại (bug fix).
- **Reach** bị bỏ khỏi chỉ số chính (R1).
- **Kết quả M1** ở `d88abc4` (seed 0, full arm chết ở task 2) được giữ làm bằng chứng chẩn đoán, không dùng để so sánh. Baseline của run đó dùng lại được nếu commit mới không đổi code của baseline. Phải kiểm bằng trùng loss từng epoch ở 2 task đầu.

---

## 10. Chẩn đoán NaN (sau khi §2–§6 xong)

Từ checkpoint cuối task 1 của `icarl_full__nobn__superclass__s0` cũ: nếu checkpoint có trong archive thì dùng lại, nếu không thì train lại task 1 của baseline seed 0, vốn trùng loss tới epoch 51. Chạy 5 epoch đầu task 2 với 4 nhánh:

| Nhánh | Mô tả |
|---|---|
| A | baseline |
| B | code cũ `d88abc4`: một lần áp dụng ở it 2001 rồi SGD (tái hiện NaN) |
| C | code cũ, nhưng **không** reset optimizer state khi swap |
| D | code mới (T29–T31) |

**Log mỗi bước SGD:**
- loss, ‖∇‖, max |θ|;
- chuẩn tham số theo site;
- chuẩn trọng số ra của các neuron mới cài;
- 10 vòng power iteration cho λ_max của Hessian loss (GGN) mỗi 10 bước, so với 2/lr = 40.

**Báo lại:**
- bước đầu tiên có giá trị không hữu hạn, và tensor nào;
- λ_max có vượt 2/lr trước lúc đó không;
- nhánh C và D có hết NaN không.

---

## 11. Notebook và thứ tự chạy

1. §2 → §8 local.
2. Notebook profiling (§3), một T4.
3. Chẩn đoán NaN (§10).
4. Gửi báo cáo §12. **Chưa** chạy lại M1 trước khi mình xem báo cáo.

Mọi notebook pin cùng commit. Dừng mọi notebook M1 đang chạy ở `d88abc4`.

---

## 12. Báo cáo

1. Diff tóm tắt theo file và danh sách code đã xóa.
2. Kết quả 6 test ở §8.
3. `profile_d88abc4.json` và profile của commit mới trên cùng lần áp dụng:
   - thời gian từng phần;
   - `gram_seconds` theo m;
   - số lần thử LM;
   - chi phí mỗi Γ.
4. Bảng chẩn đoán NaN (§10).
5. Ước lượng thời gian cho một run 20 task với ngân sách 1.0×: K0 mới và số lần áp dụng mỗi task.
6. Mọi chỗ prompt lệch với tài liệu.
