<!-- Supervisor spec (2026-10-16), verbatim below the context block. -->
## Context (read first)
- Branch `vna-pilot-gir-vna`, HEAD 018789b. `.git` read-only: do NOT commit. `.venv/bin/python`,
  `REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
- "VNA_agent_prompt_pilot_b5.md" = `docs/vna/TASK_PILOT_GIR_VNA.md` in this repo.
- 0.1(a) hypothesis to test first: the 0.11 range error at task 10 is the pseudo-inverse threshold
  (`vals > max*c*eps`) dropping tiny-but-nonzero eigenvalues of a saturated iCaRL H, while the KD part of
  e puts mass there (teacher probabilities on classes the student nearly zeroes). Verify analytically and
  numerically (null(H) should be span(1) only); fix the threshold/projection accordingly, not the loss,
  unless you find metric_apply or the residual actually wrong. Write the requested test.
- 0.1(c): the Kaggle checkpoints are not available locally. Provide a CLI subcommand
  `python -m experiments.pilot_b3 remeasure --from <icarl output dir>` that re-runs the quick measurement
  on whatever per-point checkpoints exist; if the existing resume checkpoints cannot reproduce the
  task-2/5/10 mid-task states, say so in the report and make future `icarl` quick runs save those
  measurement-point checkpoints (small: model + old model + buffer) so remeasurement works.
- Testing: run only new/affected tests while iterating, full suite once at the end.
- Notebook: `scripts/build_kaggle_vna_parallel.py` -> `notebooks/kaggle_vna_parallel.ipynb`, plus six
  ready first-cell variants for P1–P6 in `docs/vna/PARALLEL_JOBS.md` (or 6 prebuilt notebooks
  `notebooks/kaggle_vna_parallel_P{1..6}.ipynb` that differ only in the first cell). Rebuild with
  `--commit` of current HEAD; the supervisor repins after committing.
- Report: `docs/vna/PARALLEL_REPORT.md` with everything §4 asks for (0.1(c) as far as possible locally).

# Prompt cho agent: đợt chạy song song (6 notebook T4×2), pilot CIL và B5

**Base:** HEAD hiện tại (đã có `icarl` quick, iCaRL tái lập NME 64.64). Thay cho danh sách arm và bố trí lane trong `VNA_agent_prompt_pilot_b5.md`. Các định nghĩa thuật toán ở đó (VNA, GIR, NGD cùng chuẩn, chọn K tự động, test bitwise) **giữ nguyên**.

## 0. Sửa trước khi chạy (bắt buộc)

1. **Kiểm tra range(H)** (`methods/vna_output.py`).
   - Ngưỡng hiện tại 1e-10 làm loại cả 3 điểm đo B3 quick: lệch 1e-5 ở task 2, 8e-5 ở task 5, **0.11 ở task 10**.
   - (a) Tìm nguyên nhân lệch 0.11 ở task 10: so residual e với range của H cho loss `ICaRLPyCIL` (CE + KD, T = 2, `known_classes`). Kiểm tra `metric_apply` có đúng là Hessian theo logits của **toàn bộ** loss không (cả phần KD trên các lớp cũ, có hệ số 1/T²), và residual có khớp gradient của đúng loss đó không. Viết test: với iCaRL loss ở task ≥ 2, e phải nằm trong range(H) với sai lệch ≤ 1e-8.
   - (b) Sau khi sửa (a), đặt ngưỡng `residual_in_range_H` thành sai lệch tương đối ≤ 1e-6, và ghi giá trị lệch vào dòng như hiện tại.
   - (c) Chạy lại phép đo quick trên checkpoint `icarl` đã có (task 2, 5, 10), không train lại. Gửi `gate_quick.json` mới.
2. **Biến thể chọn hướng `tiny_V_euclid`** (`methods/vna_select.py`).
   - Cùng thống kê N như tiny_V, nhưng lấy vector riêng lớn nhất của N·Nᵀ, tức tối đa V trên chuẩn Euclid của A, thay vì bài toán suy rộng N·Nᵀα = λSα.
   - Cùng chọn dấu bằng ReLU thật, cùng chuẩn hóa về chuẩn trung bình của neuron.
   - Thêm vào phép đo: cột `V_tiny_V_euclid`. Thêm vào VNA: tham số `--selection {tiny_V, tiny_V_euclid, random_frob}`.
   - Ghi vào `PREREGISTRATION_CHANGES.md` là biến thể chẩn đoán, thêm vào sau khi thấy tiny_V thua random ở B3 quick.
3. **Tham số cho iCaRL runner:** `--seed` (thứ tự lớp của PyCIL), `--init-cls {10, 50}`, `--increment {10, 5}`, `--arm`.
   - Với B50 Inc10: task đầu 50 lớp, sau đó 5 bước mỗi bước 10 lớp, epoch theo đúng cấu hình PyCIL (task đầu 200, các task sau 170).
   - Thêm số tham chiếu iCaRL B50 Inc10 từ tài liệu PyCIL vào `reproduction_gate.json`, ghi rõ nguồn. Nếu PyCIL không công bố số này thì ghi `reference: none`.

## 1. Notebook dùng chung: `kaggle_vna_parallel.ipynb`

- Cell đầu có `JOBS_GPU0`, `JOBS_GPU1`, `JOBS_CPU` (danh sách tên job) và `RUN_TAG`. Ngoài ba biến này, các cell khác giống hệt nhau giữa các notebook.
- Pytest chạy một lần ở đầu, rồi chạy full luôn (không smoke riêng). Có resume và archive như hiện tại; `ALLOW_OVER_BUDGET = True`.
- Mỗi job ghi vào `outputs/<RUN_TAG>/<job>/` gồm: `summary.json`, `tasks.json`, `reproduction_gate.json` (chỉ cho arm `icarl`), `vna_log.csv` (ρ, K, thời gian của mỗi lần tính VNA), `protocol.json`.
- Cuối notebook, in bảng các job đã xong: average incremental accuracy (CNN, NME), last accuracy, forgetting.

## 2. Phân job (copy notebook ra 6 tài khoản, chỉ đổi cell đầu)

| Notebook | GPU0 | GPU1 | CPU (4 lõi) |
|---|---|---|---|
| P1 | `icarl_vna__inc10__s1993` | `icarl_ngd__inc10__s1993` | `b5_bp__s0`, `b5_cbp__s0`, `b5_cbp_gir_vna__s0` |
| P2 | `icarl_vna_random__inc10__s1993` | `icarl_vna_euclid__inc10__s1993` | `b5_bp__s1`, `b5_cbp__s1`, `b5_cbp_gir_vna__s1` |
| P3 | `icarl__inc10__s1994` | `icarl_vna__inc10__s1994` | `b5_bp__s2`, `b5_cbp__s2`, `b5_cbp_gir_vna__s2` |
| P4 | `icarl__inc10__s1995` | `icarl_vna__inc10__s1995` | `b5_cbp_gir__s0`, `b5_cbp_gir__s1`, `b5_cbp_gir__s2` |
| P5 | `icarl__inc5__s1993` | `icarl_vna__inc5__s1993` | `b5_cbp_gir_randdir__s0`, `__s1`, `__s2` |
| P6 | `icarl__b50inc10__s1993` | `icarl_vna__b50inc10__s1993` | `b5_bp_vna__s0`, `__s1`, `__s2`, `b5_bp_ngd__s0`, `__s1`, `__s2` |

Ghi chú:
- `icarl__inc10__s1993` **đã có** từ lần chạy quick, nên không chạy lại.
- Arm `icarl` mới **không** chạy phép đo quick, chỉ train, để rẻ hơn.
- Các arm VNA dùng `--selection tiny_V`; riêng `icarl_vna_random` dùng `random_frob` và `icarl_vna_euclid` dùng `tiny_V_euclid`.
- B5 regression: arm và định nghĩa theo phần A của prompt pilot. `cbp_gir_vna`, `bp_vna` và `bp_ngd` dùng λ = 1, K = 100. P6 có 6 job CPU; chạy tối đa 4 job cùng lúc, số còn lại xếp hàng.

## 3. Gộp kết quả: `scripts/collect_parallel.py`

- Đọc nhiều archive output (mỗi notebook một archive), gộp theo `RUN_TAG`.
- In:
  - **Bảng CIL Inc10:** mỗi arm, giá trị seed 1993; với `icarl` và `icarl_vna`, thêm trung bình ± độ lệch chuẩn qua 3 seed, và chênh lệch ghép cặp theo seed.
  - **Bảng CIL Inc5:** `icarl` so với `icarl_vna`.
  - **Bảng CIL B50 Inc10:** `icarl` so với `icarl_vna`, kèm số tham chiếu nếu có.
  - **Bảng B5:** sai số trung bình 50 task cuối, trung bình ± độ lệch chuẩn qua 3 seed; chênh lệch ghép cặp so với `cbp`.
  - Với các arm VNA: K đã chọn ở từng task, ρ trung bình, tổng thời gian VNA so với thời gian train.

## 4. Báo cáo trước khi bấm chạy

- Diff, log pytest (local, dữ liệu giả).
- Kết quả 0.1(c): `gate_quick.json` mới.
- Notebook đã pin commit, kèm 6 cell đầu đã điền sẵn cho P1–P6.
