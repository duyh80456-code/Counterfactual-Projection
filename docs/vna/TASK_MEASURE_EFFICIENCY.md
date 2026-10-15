<!-- Supervisor spec (2026-10-15), saved verbatim below the context block. -->
## Context for this repo state (read first)
- Work on branch `vna-measure-efficiency`, created from HEAD 566eafc (NOT 688b61d): commits after 688b61d
  add pilot mode, lane plan, Phase-0 discovery, both-mu measurement, full resume/deadline/heartbeat
  logging (`experiments/vna_resume.py`, status.json, nvidia-smi heartbeat) and a Python 3.13 fix. Keep all
  of them. §4.1/§4.3 extend the existing heartbeat/deadline machinery instead of duplicating it.
- The measurement files changed only trivially since 688b61d, so test 5.1 compares against the
  688b61d versions: load them from `git show 688b61d:<path>` into a temporary module (do not check out).
- `.git` is read-only for you: do not commit. Use `.venv/bin/python`; upstream tests need
  `REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
- Testing: run the new tests 5.1–5.4 and the full suite once at the end (5.5). Keep new tests small/CPU.
- Smoke on Kaggle is the supervisor's step; locally only fake-data CPU smokes to print solve counts.
- Report exactly the items in "Báo cáo khi xong".

# Prompt cho agent: sửa phép đo Stage B (giữ nguyên thuật toán, chạy full)

Repo: `expanding-guide model`, base commit `688b61d`. Làm trên branch mới, pin commit vào notebook khi xong.

## Nguyên tắc (bắt buộc)

1. **Không thay đổi thuật toán và không nới số.** Giữ nguyên công thức d, tiny_V, GIR, damping "outputs", CG `cg_tol=1e-6`, `cg_iters=1000` cho mọi arm. Giữ đủ mọi rank (B3: 4/8/16; B5: như hiện tại; B1/B2/B4: 4/16/64), cả N1 và N2, 8 null mỗi loại, cả hai μ.
2. Chỉ được phép:
   - (a) bỏ tính lặp bằng **đồng nhất toán học**;
   - (b) sửa các lỗi làm sai lệch cổng G;
   - (c) sửa cấu hình cho khớp những gì đã chốt;
   - (d) thêm cơ chế an toàn khi chạy.
3. Mọi thay đổi loại (a) phải qua test đối chiếu số (mục 5) trước khi merge.

## 1. Bỏ tính lặp: chỉ đồng nhất toán học (`diagnostics/icarl_measure.py`, `diagnostics/lop_measure.py`)

F được tính trên toàn mạng (`names` = mọi tham số trainable) và không phụ thuộc neuron ảo. Vì vậy, với một residual và một μ cố định, z = H J (F+μI)⁻¹ Jᵀe là **một vector duy nhất**, dùng chung cho mọi site và mọi rank.

- **1.1 Một trace mỗi (điểm đo, μ).** Tính `trace = op.trace(cfg)` một lần (với B3: trên tập stat), suy ra `mu` một lần. Dùng μ đó cho mọi lần giải, kể cả null. Ghi `trace` vào mỗi dòng.
- **1.2 Một hàm `solve_z(residual, data, ctx)`** trả về `(z, cg_record)`. Gọi hàm này đúng một lần cho mỗi residual sau:
  - g thật trên stat;
  - g thật trên eval (B3);
  - mỗi null.

  Cache các kết quả theo (μ, residual id).
- **1.3 Rút null một lần cho mỗi (điểm đo, μ).** Rút 8 N1 + 8 N2 trên stat, cùng 8 N1 + 8 N2 tương ứng trên eval (B3), với generator cố định theo seed và điểm đo. Giải z cho từng null một lần. Dùng chung các null này cho mọi site và mọi rank.
  - Lưu `null_draw_id` vào dòng.
  - Ghi chú trong docstring: p-value của từng dòng vẫn hợp lệ; p-value giữa các site của cùng điểm có tương quan.
- **1.4 Hàm `virtual_energy(model, sites, A_by_site, data, z, e)`** trả về `V = ‖J_Eᵀz‖²` và `U = ‖J_Eᵀe‖²` chỉ bằng VJP, không giải CG. Dùng `joint_products(...).JET`, giống hệt cách `joint_correction` đang tính V và U.
  - `rho = V / ‖Jᵀe‖²`, với cùng mẫu số như `joint_correction`.
  - Mọi arm (tiny_V, random_frob, tiny_U, null_A) đều tính V/ρ qua hàm này.
- **1.5 `rho_joint = Σ_site V_site / ‖g‖²`** cho các site có proposal, cùng z. Đây là đẳng thức chính xác vì z dùng chung: ‖[J_E1,…]ᵀz‖² = Σ‖J_Esᵀz‖². Bỏ lời gọi `joint_correction` nối mọi site chỉ để lấy ρ.
  - Nhóm exemplar/new (B3): giải z riêng trên từng tập con (mỗi nhóm một lần CG), rồi tính ΣV theo 1.4.
- **1.6 Vẫn kiểm chứng d.** Với mỗi (site, rank, μ), gọi `joint_correction` **một lần** cho arm tiny_V thật. Lời gọi này giải đủ 2 lần CG như cũ và trả `raw_gain`, `realized_gain`, `finite`.
  - Assert `|V_from_1.4 − V_joint_correction| / V ≤ 1e-6`; ghi lệch này vào cột `V_consistency`.
  - Truyền `mu` của 1.1 vào để hai đường dùng cùng μ.
- **1.7 B5 swap (GIR) ra khỏi vòng rank.** Swap không phụ thuộc rank: tính một lần cho mỗi (site, μ, criterion) rồi gắn vào mọi dòng rank của site đó.
  - `before_swap` dùng z chung qua 1.4, không cần CG.
  - Mỗi clone (swap_A, swap_random) giải z riêng một lần vì F của clone khác, rồi tính V qua 1.4.
- **1.8 tiny_V của null** vẫn chạy cho từng (site, rank), giống code cũ. Chỉ phần giải z được dùng chung.

## 2. Sửa lỗi làm sai lệch cổng G

- **2.1 B5 tách stat/eval.** Buffer đo 256 mẫu, hoán vị theo generator của điểm đo, chia 128 stat / 128 eval, giống B3.
  - tiny_V, tiny_U và null_A được chọn trên stat.
  - Mọi V/ρ được tính trên eval với residual eval.
  - Null trên eval rút theo cùng quy tắc như B3.
  - `subspace_sim` vẫn chia đôi tập stat.
  - Ghi `stat_m`, `eval_m`.
- **2.2 B3/B1/B2/B4: bỏ hard-code `buffer_m: 128`.** Buffer đo 512 mẫu chia 256/256; ghi giá trị thật vào dòng.
- **2.3 Tách cờ hội tụ** thành `cg_converged_real` (z thật, z eval, d) và `cg_converged_null` (mọi null của điểm). `cg_converged` = AND của hai cờ.
  - Trong `vna_gate.py`, ghi thêm `excluded_points` và `excluded_fraction`.
  - Nếu `excluded_fraction > 0.10` thì `status = 'INVALID'`, không để n giảm âm thầm.

## 3. Cấu hình cho khớp đã chốt

- **3.1** `vna_gate.py`: `train_acc_applicable = setting not in ('B5', 'B3')`.
- **3.2** μ: `self.mus = [1e-3, 1e-2]` cho mọi Measurement. Cổng đọc ở `mu_rel = 1e-2`.
- **3.3 Điều kiện N1 của cổng:** dùng `significant_rho_N1` (p-value của ρ, đại lượng VNA dùng) thay cho `significant_N1` (p-value của λ₁).
  - Vẫn ghi đủ cả `p_lambda_*` và `p_rho_*`.
  - Ghi `gate_null_statistic: 'rho'` vào gate.json.

## 4. An toàn khi chạy

- **4.1 Heartbeat.** Một thread daemon ghi `heartbeat.json` mỗi 60 giây, gồm:
  - setting, task/epoch, điểm đo;
  - μ, site, rank;
  - tên phép giải đang chạy và số vòng CG hiện tại;
  - thời gian từ đầu điểm đo.

  `conjugate_gradient` nhận một callback tùy chọn để báo số vòng.
- **4.2 Faulthandler.** `faulthandler.dump_traceback_later(timeout_s, repeat=True, file=...)` cho mỗi điểm đo; mặc định 3600 giây, cấu hình được qua CLI.
- **4.3 Timeout mềm** cho mỗi điểm đo (mặc định 2 giờ, cấu hình được). Khi quá hạn: ghi dòng có `status='timeout'` kèm heartbeat cuối, checkpoint, rồi chạy tiếp điểm sau. Dòng timeout tính vào `excluded_fraction`.
- **4.4 Thời gian.** Ghi `seconds` và `cg_iters_used` cho từng lần giải vào CSV phụ `solves.csv`. Smoke phải in tổng số lần giải CG và thời gian trung bình mỗi lần giải.

## 5. Test bắt buộc (`tests/`)

- **5.1 Đối chiếu bản cũ.** Trên model nhỏ (ResNet nhỏ CIFAR-shaped, 32 mẫu, float64, seed cố định), chạy code cũ (`688b61d`) và code mới.
  - Với cùng A, cùng residual và cùng null residual (truyền vào tường minh), các đại lượng V, U, ρ, η, `V_tiny_V`, `V_random_frob`, `V_tiny_U`, ρ_joint, null_rho, null_lambda phải khớp với sai lệch tương đối ≤ 1e-6.
  - Để so được, code cũ cần một hook test-only cho phép truyền null residual vào.
- **5.2 Kiểm tra đẳng thức ρ_joint.** `rho_joint` tính qua ΣV phải khớp `joint_correction(all sites)` với sai lệch ≤ 1e-6.
- **5.3 B5 split.** A được chọn hoàn toàn không đọc các chỉ số eval. Dùng mock để assert điều này.
- **5.4 Cổng.** Có unit test cho: B3 train_acc n/a; N1 đọc ρ; `excluded_fraction > 0.1` → INVALID.
- **5.5** Toàn bộ test cũ (1–16) vẫn pass.

## 6. Notebook

- Tạo lại nb1, nb2, nb3 theo lane đã chốt.
- Pin commit mới.
- Smoke chạy 1 điểm đo mỗi setting và in tổng số lần giải CG, số giây mỗi lần giải và ước tính thời gian cả run.
- **Gửi code + log smoke để review trước khi chạy full.**

## Báo cáo khi xong

- Diff theo file.
- Kết quả test 5.1–5.5.
- Bảng chi phí smoke: số lần giải CG mỗi điểm đo trước và sau, giây mỗi điểm đo.
- Ước tính GPU-giờ cho từng notebook.
