<!-- Supervisor spec round 2 (2026-10-15), verbatim below the context block. -->
## Context and supervisor-approved adjustments (read first; these override the spec where they conflict)
- Branch `vna-measure-round2` from 20af3dd (pins 60b7b5b). `.git` read-only: do NOT commit.
  `.venv/bin/python`; upstream tests need `REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
- `verify_energy` already has no assert (commit 350f1aa); keep it that way.
- **Logging (user request):** print progress per EPOCH (B1–B4) / per TASK (B5), never per step.
  Remove our per-10k-step lines and set `TQDM_DISABLE=1` in every runner subprocess env so upstream
  tqdm bars (LoP, PyCIL) do not flood Kaggle logs. Keep heartbeat (10 min) and one summary line per
  measurement point.
- **§B solver scaling:** M is n_out × n_out with n_out = samples × classes (CIFAR-100, 256 eval → 25,600:
  M = 5.2 GB fp64, fp64 factorization takes hours on a T4). Implement `--solver {auto, output_exact, cg}`,
  default `auto` = output_exact when n_out ≤ `OUTPUT_EXACT_MAX` (CLI, default 4096) else cg, and print the
  choice and reason per measurement point. Everything in §B (identity, docstring, range(H) flag, reuse
  of the factorization, tests vs CG) is still required for the output_exact path. The smoke must print,
  per setting, n_out, the chosen solver, and for output_exact the M build/factor seconds and memory; for
  cg the seconds per solve. Also print the projected cost if output_exact were forced (n_out³ estimate),
  so the supervisor can decide on a smaller measurement batch.
- Testing: write the required tests; run only the new/affected test files while iterating, then the
  full suite once at the end. Do not run full Kaggle-scale jobs. Rebuild nb1/nb2/nb3 per §E.3 with
  `--commit` of the current HEAD (supervisor repins after committing).

# Prompt cho agent: hoàn thiện phép đo Stage B, vòng 2 (chưa train VNA/GIR)

Base: commit `60b7b5b` (đang pin trong nb1–nb3). Làm trên branch mới, commit và pin lại cả 3 notebook khi xong.

## Nguyên tắc (giữ nguyên từ vòng trước)

- **Không đổi thuật toán.** Giữ nguyên công thức d, tiny_V, GIR, damping "outputs", cả hai μ, N1/N2 và 8 null.
- **Mọi thay đổi về số phải là đồng nhất toán học**, và phải có test đối chiếu chứng minh.
- **Không có `assert` nào được làm sập một run dài.** Nếu `verify_energy` vẫn còn assert thì đổi thành ghi `V_consistency` vào dòng. Khi lệch quá 1e-4 thì đặt `cg_converged_real = False` để điểm đó bị loại có đếm.
- **Gửi `git diff 90a18f7..60b7b5b`** kèm một câu giải thích lý do sửa damping.

---

## A. Lỗi làm sập run

**A1. MNIST: `torch.zeros` và JVP.** `stage_b5.py` tạm thay `torch.zeros` khi chạy upstream. Lần JVP đầu tiên của phép đo nạp `torch._decomp.decompositions_for_jvp`, và bước nạp đó `torch.jit.script` gặp hàm `zeros` đã bị thay nên lỗi.
- Sửa: `import torch._decomp.decompositions_for_jvp` ở đầu module, trước mọi lần thay `torch.zeros`.
- Test bắt buộc: chạy đúng đường production (thay `torch.zeros` rồi gọi một phép đo có JVP), **không** nạp sẵn trong fixture.

**A2. Hết bộ nhớ GPU trong `hook_tiny`** (`methods/vna_sites.py:252`). Code đang `torch.cat` toàn bộ X, Y, R (đặc trưng conv đã unfold, float64), khoảng 5 GB với resnet8x4.
- Sửa: cộng dồn theo chunk các thống kê mà bài toán trị riêng dùng (S = Σ x xᵀ, N = Σ x vᵀ, cùng các tổng khác nếu có), không giữ X, Y, R đầy đủ. Bước chọn dấu bằng ReLU thật cũng phải chạy theo chunk.
- Test: kết quả chunked bằng bản cũ với sai lệch tương đối ≤ 1e-10 (float64) trên model nhỏ.
- In ra mức bộ nhớ đỉnh của `tiny_V` cho resnet8x4, wrn_16_2 và ResNet32 với batch thật.

**A3. B3 sập khi ghi summary.** Metadata có object `torch.device` lấy từ args của PyCIL. Sửa: `write_json` dùng `default=str`, hoặc chuyển device sang chuỗi trước khi ghi. Rà lại mọi `write_json` khác.

---

## B. Chi phí giải: giải chính xác trong không gian output (đồng nhất toán học)

Smoke trên T4: mỗi lần giải CG mất 127–188 giây; một điểm đo B3 chạm timeout 2 giờ ngay trong phần giải z dùng chung; riêng B3 ước tính 80 GPU-giờ. Thay CG trong không gian tham số bằng giải chính xác trong không gian output.

**B1. Đẳng thức.** Gọi R = H^{1/2} (H khối chéo theo mẫu, nửa xác định dương) và M = R K R với K = J Jᵀ (kích thước n_out × n_out). Với residual nằm trong range(H) (CE/KD có residual tổng bằng 0; MSE có H = I), ta có

- (F+μI)⁻¹ Jᵀ R s = Jᵀ R (M+μI)⁻¹ s;
- z = H J (F+μI)⁻¹ Jᵀ e = R M (M+μI)⁻¹ s, với s = R⁺ e;
- d = (F+μI)⁻¹ Jᵀ H J_E w = Jᵀ R (M+μI)⁻¹ (R J_E w).

Yêu cầu:
- Viết dẫn xuất vào docstring.
- Assert khi chạy (không làm sập run; ghi cờ) rằng e nằm trong range(H), với sai lệch tương đối ≤ 1e-10. Khi H suy biến thì dùng pseudo-inverse.

**B2. Dựng M một lần cho mỗi tập dữ liệu ở mỗi điểm đo.**
- Hàm `spectrum` đã dựng H^{1/2}KH^{1/2} từng cột một. Hãy vector hóa (vmap theo chunk cột) cho nhanh.
- Ở mỗi điểm đo: một M cho tập stat, một M cho tập eval.
- Nhóm exemplar/new của B3 là tập con của eval, nên M của nhóm lấy đúng ma trận con của M_eval; không dựng lại.
- Mỗi clone swap GIR của B5 có F riêng, nên cần M riêng.

**B3. Phân tích một lần, giải nhiều lần.**
- Phân tích phổ M = QΛQᵀ một lần (hoặc Cholesky cho từng μ), ở float64.
- Mọi lần giải dùng lại phân tích đó: z thật, 32 null, các nhóm, cả hai μ, và bước kiểm chứng d.
- M được giữ ở GPU hay CPU tùy bộ nhớ. In `n_out`, bộ nhớ, thời gian dựng và thời gian phân tích.

**B4. Giữ đường CG làm đối chứng.** Thêm `--solver {output_exact, cg}`, mặc định `output_exact`.

Test bắt buộc trên model nhỏ, float64, CG với tol 1e-12:
- z, d, V, ρ, ρ_joint, null_rho và null V khớp giữa hai solver, sai lệch tương đối ≤ 1e-8;
- chạy cho CE, KD và MSE.

**B5. Ghi lại cho mỗi điểm đo:** `solver`, `n_out`, `M_build_seconds`, `factor_seconds`, `min_eig`/`max_eig` của M, `residual_in_range_H`.

---

## C. Thống kê của cổng

**C1. Null theo năng lượng chuẩn hóa bằng ‖e‖².** Hoán vị residual (N1) giữ nguyên ‖e‖² nhưng làm ‖g‖² = ‖Jᵀe‖² nhỏ đi, nên ρ của null bị thổi phồng. Dữ liệu regression xác nhận điều này: ở nhiều điểm p_λ = 0.111 nhưng p_ρ = 1.0.
- Với dòng thật và từng null, lưu `V`, `e_norm2 = ‖e‖²` (residual trên eval) và `nu = V / e_norm2`.
- Thêm `p_nu_N1`, `p_nu_N2`, `significant_nu_N1`, `significant_nu_N2`; vẫn ghi `p_rho_*` và `p_lambda_*`.
- Cổng: điều kiện N1 đọc `significant_nu_N1`, và ghi `gate_null_statistic: 'nu'`.
- Viết vào `docs/vna/PREREGISTRATION_CHANGES.md`:
  - lý do (mẫu số ‖g‖² thay đổi dưới null);
  - thời điểm: sau khi thấy dữ liệu regression B5 và trước mọi dữ liệu CIFAR;
  - regression B5 sẽ được chạy lại và báo cáo theo cổng mới.

**C2. Lịch đo của B5 dày ở nửa sau.**
- Regression (100 task): đo ở task {1, 2, 5, 10, 20, 30, 40, 50} và **mỗi 5 task từ 55 đến 100**. Mỗi task 2 điểm (first, mid), tức 22 điểm trong cửa sổ.
- MNIST: tương tự. Đo thưa ở nửa đầu; ở nửa sau, đo đủ dày để có ít nhất 20 điểm trong cửa sổ `task > tasks_run/2`.
- Ghi `points_in_window` vào gate.json. Nếu dưới 10 thì trả INVALID.

**C3. Lịch đo của B3:** điểm giữa task cho cả 10 task, cộng điểm cuối task ở các task 2, 5, 10. Điểm cuối task chỉ để tham khảo; cổng đã tự loại chúng. Tổng 13 điểm, thay cho 40.

---

## D. GIR khi gắn vào CBP trong phép đo B5

Swap theo `cbp_rr_*` luôn rỗng (m = 0), vì CBP thay neuron theo số tích lũy dần nên chụp tại một thời điểm hiếm khi có neuron nào.
- Sửa: với mỗi criterion của CBP, lấy **m neuron có utility CBP thấp nhất** (tính bằng shadow utility, đúng công thức upstream) với m = max(1, số neuron ReDo-dormant của layer đó tại điểm đo).
- Ghi `cbp_selection: 'lowest_utility_m'` và m.
- Swap theo ReDo giữ nguyên.

---

## E. Kiểm tra cuối và báo cáo

1. Toàn bộ test pass, kể cả test mới A1, A2, B4, C1 và test đường production của MNIST.
2. `scripts/smoke_measure_efficiency.py` chạy được với `--solver output_exact`.
3. Notebook:
   - nb1 (B5): regression chạy lại cả 6 arm với lịch C2; MNIST BP và CBP với resume nhiều phiên. In ước tính GPU-giờ theo từng phiên.
   - nb2: **chỉ chạy B3** (B4 tạm gác).
   - nb3: tạm gác (B1, B2 chạy sau B3).
4. **Không chạy full.** Gửi về:
   - diff theo file và kết quả test;
   - với mỗi setting trong smoke: n_out, thời gian dựng M, thời gian phân tích, giây mỗi điểm đo, ước tính GPU-giờ cả run;
   - bộ nhớ đỉnh.

   Chạy full chỉ sau khi supervisor đã review.
