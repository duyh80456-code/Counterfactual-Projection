<!-- Supervisor spec (2026-10-16), verbatim below the context block. -->
## Context (read first; overrides the spec only where stated)
- Branch `vna-pilot-gir-vna` from a77a57b (round 3 done, Cholesky test in). `.git` read-only: do NOT commit.
  `.venv/bin/python`; upstream tests need `REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.
- Output-space solve: use the existing `methods/vna_output.OutputFactor` and `choose_solver` as they are
  (eigh up to 4096, per-mu Cholesky above); "solve directly up to 12,800" in Part B means the same exact
  solve — do not add a new path.
- Reuse existing machinery: `adapters/lop_hooks.py`, `adapters/icarl_hooks.py`, `diagnostics/*_measure.py`,
  `diagnostics/measure_efficiency.py`, `methods/vna_joint.joint_correction`, `ngd_control`, `vna_resume`
  (checkpoint/deadline/heartbeat), notebook builders. Logs per task (B5) / per epoch (B3) only.
- Any dedicated RNG for VNA/GIR must not touch upstream global RNG (bitwise tests).
- Testing: write all tests in §6 and B3/B6; run new/affected test files while iterating and the full suite
  once at the end. Smoke runs are done by the user on Kaggle; locally only fake-data CPU smokes to print
  timings for the report.
- Notebooks: create `scripts/build_kaggle_vna_pilot_b5.py` -> `notebooks/kaggle_vna_pilot_b5.ipynb`
  (Part A, CPU lanes) and update the nb2 builder for Part B lanes; rebuild with `--commit` of current HEAD
  (the supervisor repins after committing).
- Report: `docs/vna/PILOT_GIR_VNA_REPORT.md` with everything §7 and B5 ask for.

# Prompt cho agent: cài GIR và VNA vào vòng train; pilot performance trên B5 regression (phần A) và CIL B3 (phần B)

Base: HEAD sau khi đã hoàn tất vòng 3 (resume exact-vs-CG, `tests/test_output_cholesky.py`, toàn bộ pytest pass).

**Mục tiêu:** lần đầu đo **performance thật** của thuật toán trên B5 regression (Dohare slowly-changing regression). Phép đo Stage B vẫn chạy như đã cài. MNIST và CIFAR chưa chạy pilot.

## Nguyên tắc

- **Đi qua upstream, can thiệp tối thiểu.** Dùng `Backprop`, `ContinualBackprop` và `GnT` của loss-of-plasticity, chỉ chèn hook ở hai chỗ:
  - lúc GnT khởi tạo lại neuron (cho GIR);
  - sau bước SGD (cho VNA).
- **Thuật toán giữ nguyên định nghĩa đã chốt:**
  - tiny_V chọn hướng theo z = HJ(F+μI)⁻¹Jᵀe, giải chính xác trong không gian output;
  - μ = μ_rel·tr(F)/n_out;
  - d = (F+μI)⁻¹ C_OE C_EO (F+μI)⁻¹ g;
  - hướng 1..m dành cho GIR, hướng m+1..m+r dành cho VNA.
- **Hai test bitwise bắt buộc:**
  1. **CBP+GIR với nguồn hướng = random** (cùng RNG như upstream) phải tái lập **bitwise** CBP upstream: log, trọng số cuối, RNG.
  2. **BP+VNA với λ = 0** phải tái lập **bitwise** BP.

## 1. GIR (Growth-Informed Reset)

Hook vào chỗ GnT sinh neuron mới (`gen_new_features` hoặc tương đương). Với các neuron `features_to_replace` ở layer i:

1. Chọn neuron **giữ nguyên logic CBP upstream** (utility, maturity, replacement rate, cộng dồn).
2. Lấy hướng vào từ **pool tiny_V** của layer i (mục 1a). Mỗi hướng chỉ dùng một lần; không bao giờ nhân bản một hướng.
3. Chuẩn hóa mỗi hướng về **chuẩn kỳ vọng lúc init** của layer đó, tức chuẩn của phân phối init upstream (ghi rõ công thức).
4. Trọng số ra = 0. Bias correction, reset utility và reset age **giữ nguyên upstream**.
5. Nếu pool cạn: dùng init ngẫu nhiên đúng như upstream, và ghi `gir_random_fill` (số neuron).

**1a. Pool tiny_V.**
- Tính trên **cửa sổ trượt** 128 mẫu gần nhất của task hiện tại, dùng cùng buffer với phép đo Stage B.
- z được giải bằng `OutputFactor` (giải chính xác), μ_rel = 1e-2.
- Số hướng trong pool: `--gir-pool` (mặc định 8 mỗi layer). Chỉ giữ hướng có trị riêng suy rộng > 0.
- Làm mới pool khi: (a) bắt đầu task mới; (b) mỗi `--gir-refresh-steps` bước (mặc định 1000); (c) pool cạn.
- Với regression (5 neuron, rr nhỏ): có thể tính lại pool ở mỗi sự kiện thay neuron nếu rẻ hơn. Ghi rõ đã chọn cách nào.
- Ghi số lần làm mới, thời gian, số hướng dùng / số neuron được thay.

**Arm đối chứng `cbp_gir_randdir`:** cùng mọi bước trên, nhưng hướng là vector ngẫu nhiên được chuẩn hóa về **cùng chuẩn** với hướng TINY. Arm này tách hiệu ứng của *hướng* khỏi hiệu ứng của *chuẩn*.

## 2. VNA (bước correction định kỳ)

Upstream học online với minibatch rất nhỏ, nên GGN trên một batch là suy biến. Vì vậy định nghĩa VNA cho B5 như sau (ghi vào `docs/vna/PREREGISTRATION_CHANGES.md` **trước khi chạy**):

- Mỗi `K` bước SGD, trên cửa sổ trượt W gồm 128 mẫu:
  1. Tính g_W và z_W bằng `OutputFactor`.
  2. Chọn tiny_V với m+r hướng; dùng các hướng m+1..m+r (với BP+VNA thì m = 0).
  3. Tính d_W bằng đường `joint_correction` có sẵn, kèm `output_factor`.
  4. Cập nhật θ ← θ − η·λ·d_W, với η là step size upstream.
- Cũng như phép đo Stage B, **cổng ρ**: nếu ρ < ρ_min (mặc định 0) thì bỏ qua bước correction. Ghi ρ của mỗi lần.
- Regression có 1 output nên rank tối đa là 1: dùng r = 1.
- Lưới tham số đặt trước, chỉ có 2 cấu hình: (K, λ) ∈ {(100, 0.1), (100, 1.0)}, μ_rel = 1e-2.

**Arm đối chứng `bp_ngd`:** cùng lịch K, cùng cửa sổ W.
- Bước correction dùng d_NGD = (F+μI)⁻¹ g_W, chuẩn hóa về **đúng chuẩn của d_W** tính ở cùng thời điểm (dùng `ngd_control(norm_match=True)` có sẵn).
- Chỉ khác VNA ở hướng, không khác ở độ lớn.

## 3. Các arm của pilot regression

| Arm | Mô tả |
|---|---|
| `bp` | upstream |
| `cbp` | upstream, rr lấy theo cfg upstream của regression |
| `cbp_gir` | CBP với hướng tiny_V |
| `cbp_gir_randdir` | CBP với hướng ngẫu nhiên cùng chuẩn |
| `bp_vna_l0.1`, `bp_vna_l1` | BP cộng VNA |
| `bp_ngd_l0.1`, `bp_ngd_l1` | BP cộng NGD cùng chuẩn |
| `cbp_gir_vna_l1` | đầy đủ |

- **3 seed** (0, 1, 2), dùng cùng chuỗi dữ liệu cho mọi arm ở cùng seed (ghép cặp theo seed).
- Pilot không chạy phép đo Stage B, trừ các phép ghi ở mục 4.

## 4. Metric (ghi cho mọi arm)

- Sai số online theo task (như upstream).
- **Chính:** sai số trung bình 50 task cuối.
- Phụ:
  - sai số trung bình toàn bộ;
  - mỗi 10 task: tỉ lệ neuron dormant và effective rank.
- Chi phí: giây cho mỗi bước train, giây cho mỗi lần làm mới GIR và mỗi lần correction VNA, tổng thời gian.
- Ghi `gir_random_fill`, ρ của mỗi lần VNA, và số lần bị cổng ρ bỏ qua.

## 5. Notebook

- Thêm lane pilot vào nb1, hoặc tạo `kaggle_vna_pilot_b5.ipynb`.
- Chạy song song tối đa 4 process trên CPU. GPU vẫn để phép đo MNIST của nb1.
- Có resume và archive như các notebook hiện có.
- Cuối notebook, in bảng:
  - mỗi arm: trung bình ± độ lệch chuẩn qua 3 seed của sai số 50 task cuối;
  - chênh lệch ghép cặp so với `cbp` và so với `bp`.

## 6. Test

- Hai test bitwise ở phần Nguyên tắc.
- GIR thay đúng số neuron CBP chọn; trọng số ra = 0; output không đổi ngay sau khi thay (sai lệch ≤ 1e-12, float64).
- Bước VNA và bước NGD có cùng chuẩn; gᵀd ≥ 0.
- Pool không dùng một hướng hai lần.
- Toàn bộ pytest pass.

## 7. Báo cáo (phần A)

Gửi về:
- diff và log pytest;
- smoke 2 task cho mỗi arm;
- thời gian ước tính cho mỗi arm × seed;
- thời gian ước tính của cả pilot trên 4 lõi CPU.

---

# Phần B: pilot CIL (B3, PyCIL iCaRL, CIFAR-100 B0 Inc10, ResNet32), chạy full luôn

**Base:** cùng HEAD và cùng nguyên tắc như phần A: đi qua upstream PyCIL, hook tối thiểu, công thức thuật toán không đổi, giải chính xác trong không gian output.

## B1. VNA trong iCaRL

- **Cửa sổ W:** 128 mẫu lấy ngẫu nhiên, có seed, từ dữ liệu train của task hiện tại cộng exemplar memory, tức đúng tập mà iCaRL đang train. **Loss dùng đúng loss của iCaRL:** CE cộng KD với model cũ (`ICaRLPyCIL`, T = 2, `known_classes`).
- **Mỗi K iteration**, trên W:
  1. tính g_W và z_W bằng `OutputFactor` (với n_out ≤ 12.800 thì giải trực tiếp; lớn hơn thì Cholesky);
  2. chọn tiny_V với r = 4 hướng ở mỗi site (15 basic block của ResNet32), gom tất cả site lại như `rho_joint`;
  3. tính d_W;
  4. cập nhật θ ← θ − lr_hiện_tại·λ·d_W.

  BN ở chế độ eval trong lúc tính d; sau khi tính xong trả BN về train.
- **Chọn K tự động, quyết định trước khi xem kết quả:**
  - Smoke đo t_VNA (giây cho một lần tính ở task 1 và task 2, ngoại suy tuyến tính theo n_out lên task 10) và t_iter (giây cho một iteration train).
  - Chọn K nhỏ nhất trong {50, 100, 200, 400, 800, 1 epoch} sao cho tổng chi phí VNA ≤ 1× tổng thời gian train.
  - Ghi K và cách tính vào `PREREGISTRATION_CHANGES.md` **trước khi chạy full**.
- λ = 1, μ_rel = 1e-2, cổng ρ_min = 0.
- **Arm `icarl_ngd`:** cùng K và cùng W, d_NGD = (F+μI)⁻¹g_W, chuẩn hóa về chuẩn của d_W tính ở cùng thời điểm.

## B2. GIR trong iCaRL

iCaRL không có CBP, nên GIR dùng tiêu chí dormant của ReDo:
- Ở **đầu mỗi task** từ task 2, trước epoch đầu tiên của task đó, chạy một forward trên W.
- Neuron (kênh conv_a trong mỗi basic block) bị coi là dormant khi tỉ lệ activation trung bình so với trung bình của layer ≤ τ = 0.1 (đúng cách `activity_stats` đang đo).
- Thay trọng số vào của kênh dormant bằng hướng tiny_V của site đó (pool tính trên W). Chuẩn hóa về chuẩn init của layer.
- Trọng số ra (kênh vào tương ứng của conv_b) = 0. Đặt lại running stats của BN cho kênh đó về giá trị init.
- Nếu không có hướng nào thì dùng init ngẫu nhiên, và ghi `gir_random_fill`.
- **Arm `icarl_gir_randdir`:** như trên nhưng hướng ngẫu nhiên cùng chuẩn.
- Ghi số kênh dormant mỗi site ở mỗi task, kể cả với arm `icarl` (không thay).

## B3. Các arm (seed 1993 theo PyCIL; seed init 0)

| Arm | Mô tả |
|---|---|
| `icarl` | upstream, kèm phép đo **quick** (mục B6) |
| `icarl_vna` | iCaRL + VNA |
| `icarl_ngd` | iCaRL + NGD cùng chuẩn |
| `icarl_gir` | iCaRL + GIR |
| `icarl_gir_randdir` | iCaRL + GIR với hướng ngẫu nhiên |
| `icarl_gir_vna` | đầy đủ |

**Bắt buộc:** `icarl` phải tái lập con số gate reproduction hiện có (64.42 ± 1.5). Ngoài ra:
- VNA với λ = 0 phải tái lập bitwise `icarl`;
- GIR khi không có kênh dormant phải tái lập bitwise `icarl`.

## B4. Metric

Ghi cho mọi arm:
- Average incremental accuracy (CNN và NME);
- last accuracy;
- forgetting (CNN và NME);
- accuracy theo từng task;
- tỉ lệ kênh dormant và effective rank theo task;
- chi phí VNA và GIR;
- ρ của mỗi lần tính VNA.

Notebook in bảng so sánh các arm cùng chênh lệch so với `icarl`.

## B5. Notebook (nb2) và chạy

| Lane | Arm |
|---|---|
| GPU0 | `icarl` (kèm phép đo) → `icarl_vna` → `icarl_gir_vna` |
| GPU1 | `icarl_ngd` → `icarl_gir` → `icarl_gir_randdir` |

- Có resume nhiều phiên. `ALLOW_OVER_BUDGET = True`, nhưng in ước tính GPU-giờ sau smoke.
- Chạy smoke trước; nếu smoke pass và ước tính dưới 3 phiên Kaggle thì **tự chạy tiếp full**, không cần dừng chờ review.
- Gửi về log smoke, K đã chọn, và kết quả full khi xong.

## B6. Phép đo B3 quick (thay cho bản 13 điểm trong lần chạy này)

Mục đích là có tín hiệu thăm dò cùng lúc với kết quả performance, với phần đo chỉ thêm khoảng 5–15 phút vào run `icarl`. **Thuật toán giữ nguyên**: giải chính xác trong không gian output, cùng định nghĩa z, d, V, ν, cùng null. Chỉ thu nhỏ quy mô.

Thêm cờ `--measure-profile {full,quick}`, mặc định `full`. Cấu hình `quick`:

| Thông số | full (đăng ký chính thức) | quick |
|---|---|---|
| Buffer đo | 512 → 256 stat / 256 eval | **128 → 64 stat / 64 eval** (cùng cách lấy mẫu: task hiện tại + exemplar) |
| Điểm đo | 10 điểm giữa task + cuối task 2, 5, 10 | **3 điểm: giữa task 2, 5, 10** |
| μ_rel | 1e-3, 1e-2 | **1e-2** |
| Rank | 4, 8, 16 | **16** |
| Site | 15 basic block | **3: `stage_1.4`, `stage_2.4`, `stage_3.4`** (block cuối mỗi stage) |
| Null | 8 N1 + 8 N2 | **giữ nguyên** |
| Nhóm exemplar/new | có | **giữ nguyên** (ma trận con, gần như miễn phí) |
| Kiểm chứng d và so exact với CG | mọi site × rank / điểm đầu | **chỉ ở điểm đầu (task 2)** |

**Ghi cho mỗi dòng đo:**
- ρ, ρ_joint (trên 3 site), ν cùng p_ν của N1 và N2, p_ρ, p_λ;
- V_tiny_V, V_random_frob, V_tiny_U;
- `subspace_sim`, ρ của nhóm exemplar và nhóm new;
- `dormant_fraction` và `effective_rank` của từng site (dùng `activity_stats`, như B5);
- n_out, thời gian dựng ma trận, thời gian phân tích, số giây của điểm đo;
- `measure_profile='quick'`.

**Cổng thăm dò:**
- Tính bằng `evaluate_gate` với `rank=16`, μ = 1e-2 và cửa sổ B3. Bỏ qua yêu cầu tối thiểu số điểm, vì chỉ có 3 điểm.
- Ghi vào `gate_quick.json` với `status_label: 'exploratory'`.
- **Không** ghi đè `gate.json` chính thức.

**Thời gian:**
- In thời gian cho từng điểm đo.
- Nếu một điểm vượt 10 phút thì ghi cảnh báo và vẫn chạy tiếp.
- Smoke phải đo thử ở task 1 và 2 với profile quick, rồi in ước tính tổng thời gian đo cho 3 điểm, ngoại suy n_out lên task 10 (6.400).

**Test:**
- Profile `quick` cho ra cùng giá trị với `full` khi đặt cùng tham số nhỏ (cùng mẫu, site, rank, μ). Nói cách khác, quick chỉ là tập con cấu hình, không có đường code riêng.
- `gate_quick.json` không chạm tới `gate.json`.

`PREREGISTRATION_CHANGES.md`: ghi rằng quick là phép đo thăm dò, quyết định trước khi có dữ liệu CIFAR. Cổng chính thức B3 vẫn là profile `full`, chạy ở phiên sau.
