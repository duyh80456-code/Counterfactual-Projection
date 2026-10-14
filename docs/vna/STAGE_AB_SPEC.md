# Spec: Giai đoạn A (chẩn đoán) và B (sàng lọc bối cảnh) — giao cho agent code

Ngày 2026-10-14. Đọc kèm: `claude/VNA_run_plan.md`, mục 5–6 của `claude/VNA_design_v2.md`, và `claude/VNA_spec.md` (phần toán và quy ước).

## 0. Bối cảnh và mục tiêu

Phase 0 trên ResNet-18 / CIFAR-100 (seed 0, commit 44d1990) cho kết quả no-go:

- ρ khoảng 0.03, và tín hiệu tắt sau epoch 150;
- V_TINY / V_random < 1;
- đối chứng nhãn xáo trộn bị lệch thang đo.

Hai giai đoạn trong spec này **chỉ đo, không chạy VNA trong lúc train**:

- **A:** sửa và kiểm tra lại các thước đo trên checkpoint Phase 0 đã có.
- **B:** sàng lọc ba bối cảnh mới (model tí hon, KD, CIL) bằng ρ, rồi tính cổng G tự động.

Không thêm bất kỳ arm train nào có correction. Stage 2a tạm hoãn.

## 1. Quy tắc chung (bắt buộc)

1. **Pin commit.** Notebook ghim commit; đổi code thì tạo lại notebook với hash mới. Arm nào đang chạy dở khi đổi pin phải chạy lại từ đầu.
2. **Diagnostics không được đổi model.** Assert tham số, buffer BN và mode train/eval giống hệt trước và sau mỗi lần đo. Đã có test, phải mở rộng cho các kiến trúc mới.
3. **Tập test chính thức** chỉ dùng để báo cáo một lần ở cuối mỗi lần train. Không dùng nó cho bất kỳ quyết định nào. Quyết định dùng split 45k/5k.
4. **Metric:** dùng GGN / Fisher thật, H = diag(p) − ppᵀ (hoặc H tổng của loss KD, xem mục B2). Không dùng empirical Fisher.
5. **Không đổi số liệu Phase 0.** Mọi thay đổi chạm vào `signal_curve.py` hay `vna.py` đều phải giữ nguyên output của các test cũ, hoặc tăng `schema_version` lên `stage1d-v1` và từ chối trộn với output cũ.
6. **Resumable.** Archive mỗi giờ, giới hạn đĩa như notebook hiện tại.
7. Mỗi bảng ghi thêm `code_commit`, `seed`, `device` và thời gian chạy.

---

## Giai đoạn A. Chẩn đoán trên checkpoint Phase 0

**Đầu vào:** đính kèm `vna_phase0_seed0.tar.gz` (thư mục `recipe/` có checkpoint mỗi 10 epoch).
**Thư mục output:** `outputs/stageA/`.
**Mặc định:** batch 64 như Phase 0, site như Phase 0 (8 block), stat batch 64 × 4, `cache_block_input=True`.

### A1. Đối chứng null có cùng độ lớn residual (thay cho shuffled-label)

**Vấn đề.** Nhãn xáo trộn tạo ra residual lớn hơn nhiều, nên λ_null luôn cao hơn λ_real. Ví dụ ở epoch 200: 0.62 so với 0.035.

**Cài đặt ba loại null, cùng số mẫu với phép đo thật.** Gọi g_i ∈ ℝ^C là residual thật của mẫu i.

- **N0 (giữ để so):** shuffled-label như cũ. Báo thêm λ/‖g‖² để có thang so chuẩn hoá.
- **N1 residual-permute:** g̃_i = g_{π(i)}, với π là hoán vị ngẫu nhiên các mẫu và input giữ nguyên. Cách này bảo toàn phân phối norm và ‖g‖ tổng.
- **N2 tangent-gaussian:** lấy ξ_i ~ N(0, I_C), chiếu lên không gian tổng bằng 0 (ξ_i − mean(ξ_i)·1), rồi đổi thang để ‖g̃_i‖ = ‖g_i‖.

**Quy trình cho mỗi null.** Tính lại toàn bộ chuỗi chọn (A lấy từ residual null), rồi λ₁, V, ρ, với **cùng** μ và rank. Số lần lặp null là **n = 8**, mỗi lần một hạt giống riêng.
- p-value thực nghiệm: p = (1 + #{null ≥ real}) / (1 + n).
- `significant_N1 = p_N1 ≤ 1/9`, tức thật lớn hơn cả 8 null. Tương tự cho N2.
- Thống kê dùng cho kiểm định: λ₁ và ρ. Báo cả hai.

**Điểm đo:** epoch 20, 40, …, 200 × 8 site, r = 16, μ_rel = 1e-2 (giống Phase 0).

**Cột output:**
`epoch, site, null_kind, stat, real, null_mean, null_max, p_value, significant`

### A2. Quét μ

- **Điểm đo:** epoch 40, 80, 100, 120, 140, 160; top-3 site theo V, cộng thêm "joint" từ A4.
- **μ_rel:** {1e-1, 1e-2, 1e-3, 1e-4}. Solver exact, `cg_iters = 1000`, `cg_tol = 1e-6`.
- **Báo các cột:** U, V, η, ρ, realized_gain, đạo hàm trên held-out, số vòng CG, `cg_converged`, residual tương đối của CG.
- Hàng nào CG chưa hội tụ thì đánh dấu và **loại khỏi mọi quyết định**.
- **Quy tắc đặt trước:** nếu ở μ_rel = 1e-3 thì ρ ≥ 3 × ρ(1e-2) tại ≥ 50% số điểm (chỉ tính hàng hội tụ), đặt `mu_rel_default = 1e-3` cho giai đoạn B. Nếu không, giữ 1e-2. Ghi quyết định vào `summary.json`.

### A3. So sánh các cách chọn A

- **Cách chọn:**
  - `tiny_U`;
  - `tiny_V`;
  - `greedy_V`;
  - `random_trace` (A ngẫu nhiên, cùng tr(K_E));
  - `random_frob` (A ngẫu nhiên, cùng ‖A‖_F với A của TINY sau `normalize_A`).
- Mọi A đi qua **cùng một** hàm chuẩn hoá. Phải ghi rõ trong code và log hằng số chuẩn hoá của từng A.
- **Rank:** {4, 16, 64}.
- **Điểm đo:** epoch 40, 80, 120, 140, 160; top-3 site.
- **Lặp lại:** 3 lần rút stat-batch độc lập, báo trung bình ± độ lệch chuẩn.
- **Thước đo:**
  - V và ρ trên một correction batch tách biệt với stat batch;
  - đạo hàm loss trên held-out (5k eval split, một batch cố định 512 ảnh);
  - **độ ổn định split-half:** chia stat batch thành hai nửa rời nhau, tính A₁ và A₂, lấy Q₁, Q₂ là cơ sở trực chuẩn của không gian hàng của A₁, A₂ (theo chiều vào), rồi tính `subspace_sim = ‖Q₁ᵀQ₂‖_F² / r`. Giá trị ngẫu nhiên kỳ vọng xấp xỉ r / d_in, báo luôn con số này.

### A4. K_E gộp nhiều block

- K = toàn mạng; K_E = Σ K_E^site trên tất cả 8 site, mỗi site rank 16; **giải chung một lần**.
- Báo ρ_joint, so với Σ ρ_site và max ρ_site, ở epoch 100, 120, 140, 160 với hai mức μ (1e-2 và 1e-3).
- **Test bắt buộc:** joint với đúng một site phải ra kết quả trùng single-site (sai số tương đối ≤ 1e-6 khi chạy float64 trên tiny model).

### A5. Bug V_TINY / V_random = 0 ở epoch 100

- Tìm nguyên nhân. Các khả năng: random_V bị inf/NaN; V = 0 do scheduler hoặc trạng thái checkpoint ở mốc LR; lỗi chia.
- Viết test hồi quy và ghi nguyên nhân vào `docs/vna/STAGE_A_REPORT.md`.

### A6. Kiểm tra solver và batch

- Chạy `diagnostics.vna_solver_validation --mode recipe` trên checkpoint 40, 100, 160, với cấu hình **giống train** (dùng `vna_training_config.py`).
- Thêm cos(d₁₆, d₁₂₈) và cos giữa hai batch 16 rời nhau.
- **Báo các cột:** cosine K-FAC/exact theo site, overhead = n_site × cost / vna_every, ‖λd‖/‖g‖.
- Không ra quyết định ở bước này; chỉ ghi bảng.

**Đầu ra của A:**
- `stageA/null.csv`
- `stageA/mu_sweep.csv`
- `stageA/selection.csv`
- `stageA/joint.csv`
- `stageA/solver.csv`
- `stageA/summary.json`, gồm `mu_rel_default`, tỉ lệ `tiny_V > random_frob` theo điểm đo, và tỉ lệ điểm có `significant_N1`
- `docs/vna/STAGE_A_REPORT.md`, có bảng và hình: ρ(μ), V theo cách chọn, null.

---

## Giai đoạn B. Sàng lọc bối cảnh (train SGD hoặc KD, chỉ đo ρ)

Thư mục output: `outputs/stageB/<setting>/`.

### B0. Kiến trúc và site (làm trước)

- Thêm vào `models/cifar_repdistiller.py` các kiến trúc **giống hệt** định nghĩa trong repo RepDistiller (Tian et al., CRD; github.com/HobbitLong/RepDistiller): `resnet8x4`, `resnet32x4`, `wrn_16_2`, `wrn_40_2`, `resnet32`. Chép đúng cấu trúc và khởi tạo, ghi nguồn và license trong header.
  - **Test:** số tham số khớp RepDistiller (resnet8x4 khoảng 1.23M, wrn_16_2 khoảng 0.70M, resnet32x4 khoảng 7.43M, wrn_40_2 khoảng 2.25M). Agent kiểm tra con số chính xác bằng cách instantiate code gốc và so.
- **Site adapter tổng quát cho basic block:** conv3×3 → BN → ReLU → conv3×3 → BN; neuron ảo thêm kênh ẩn giữa hai conv, theo đúng quy ước của site ResNet-18 hiện tại (incoming A, outgoing B = 0).
  - Áp dụng cho mọi block của resnet8x4, wrn_16_2 và resnet32.
  - WRN dùng pre-activation (BN-ReLU-conv). Site là conv1 → (BN, ReLU) → conv2, nên phải cài đúng thứ tự.
  - **Test** cho từng kiến trúc mới: adjoint bằng sai phân hữu hạn (theo mẫu `test_extension_finite_difference_adjoint_and_native_alignment`); forward với B = 0 không đổi output; buffer BN không đổi.
- **Teacher:**
  - Tải checkpoint teacher công khai của RepDistiller (`resnet32x4`, `wrn_40_2`) bằng script tải của repo gốc. Lưu SHA256 trong code; tải lỗi hoặc hash sai thì dừng.
  - **Test:** accuracy trên test chính thức ≥ 79.0% (resnet32x4) và ≥ 75.0% (wrn_40_2). Công bố là 79.42 và 75.61.
  - Đây là ngoại lệ duy nhất được dùng tập test, và chỉ để xác nhận checkpoint teacher.

### B-recipe chung (recipe CIFAR-100 của CRD/DOT, không tune)

- 240 epoch; SGD momentum 0.9; weight decay 5e-4; batch 64.
- LR 0.05, nhân 0.1 ở epoch 150, 180, 210.
- Augmentation: random crop 32 với padding 4, horizontal flip; chuẩn hoá theo mean/std CIFAR-100.
- Train trên 45k, đánh giá trên 5k (split hiện có); test chính thức chỉ một lần ở cuối.
- Log mỗi epoch: train loss, **train accuracy** (cần để kiểm tra model có khớp hết dữ liệu không), eval loss/accuracy, LR, thời gian.

### B1. Model tí hon, train với CE

Hai cấu hình: `resnet8x4` và `wrn_16_2`, seed 0, chạy riêng.

### B2. KD (cài loss KD theo đúng RepDistiller)

Hai cặp: `resnet32x4 → resnet8x4` và `wrn_40_2 → wrn_16_2`, seed 0.

- **Loss:** L = 0.1 · CE(z, y) + 0.9 · T² · KL(softmax(z_t/T) ‖ softmax(z/T)), với T = 4. Đây là cấu hình `-r 0.1 -a 0.9 -b 0` của RepDistiller.
- **Residual** cho VNA: g = ∂L/∂z (dùng autograd hoặc công thức giải tích, hai cách phải khớp nhau).
- **Metric GGN** cho mỗi mẫu:
  H = 0.1 · (diag(p) − ppᵀ) + 0.9 · (diag(p_T) − p_T p_Tᵀ), với p = softmax(z), p_T = softmax(z/T).
  - **Test:** H khớp Hessian sai phân hữu hạn của L theo z (sai số ≤ 1e-5, float64).
- Logit teacher tính trên **cùng** batch đã augment, ở chế độ eval, không lấy gradient.

### B3. CIL: class-incremental learning bằng PyCIL, iCaRL

**Lưu ý.** "CCIL" trong repo tham chiếu (`One-Shot-TAS-CCIL`, nhánh `ccil-residual-capacity`) **không phải** class-incremental learning. Đó là pipeline **tăng capacity theo cấp** (AutoFormer/DeiT phân cấp, độ rộng 192 → 216 → 240 → 256, có acquisition/integration và giữ nguyên prefix cũ), tức là một phương pháp grow. Không dùng nó cho B3.

**Nguồn.** Cài từ PyCIL (github.com/G-U-N/PyCIL), pin một commit cố định, ghi nguồn và license. Chỉ chép những phần cần dùng: data manager, ResNet32 CIFAR, iCaRL, chọn exemplar bằng herding.

**Giao thức chuẩn của PyCIL** (agent phải đối chiếu lại với `exps/icarl.json` và `models/icarl.py` tại commit đã pin):

- CIFAR-100, `init_cls = 10`, `increment = 10` (B0 Inc10, tổng 10 task); `shuffle = true`; seed 1993 cho thứ tự lớp.
- `memory_size = 2000`, `fixed_memory = false`; backbone ResNet32.
- **Task đầu:** 200 epoch, LR 0.1, giảm ×0.1 ở epoch 60/120/170, WD 5e-4.
- **Các task sau:** 170 epoch, LR 0.1, giảm ×0.1 ở epoch 80/120, WD 2e-4.
- Batch 128.
- **Loss:**
  - task đầu: chỉ CE;
  - các task sau: CE trên mọi lớp đã thấy, cộng KD với T = 2 trên các lớp cũ. Theo `_KD_loss` của PyCIL: −Σ softmax(z_old/T) · log_softmax(z[:, :K_old]/T), lấy trung bình theo batch, **không nhân T²**.
- Baseline đối chiếu: số iCaRL đã công bố trong PyCIL cho CIFAR-100 B0 Inc10. Cần khớp trong khoảng ±1.5pp về accuracy trung bình qua các task.

**Residual và metric GGN cho VNA (sau task đầu):**

- g = ∂L/∂z; dùng autograd hoặc giải tích, hai cách phải khớp nhau.
- H = (diag(p) − ppᵀ) trên tất cả lớp đã thấy, **cộng** (1/T²) · (diag(q_T) − q_T q_Tᵀ) đặt vào khối các lớp cũ.
  - p = softmax(z) trên các lớp đã thấy;
  - q_T = softmax(z[:, :K_old]/T).
- **Test:** H khớp Hessian sai phân hữu hạn của L theo z.

**Dữ liệu cho đo:** stat batch và correction batch rút từ (dữ liệu task t ∪ buffer exemplar), theo tỉ lệ có trong dataloader thật. Ghi thêm ρ tính riêng trên phần exemplar và phần dữ liệu mới.

**Site:** mọi basic block của ResNet32. Kênh là 16/32/64, nên dùng rank {4, 8, 16}, và rank không được vượt quá độ rộng kênh.

**Điểm đo:** với mỗi task t, đo ở epoch 1, 10, giữa task và cuối task; đo cả task đầu.

**Log mỗi task:** accuracy trên mọi lớp đã thấy (split eval, tách từ train của từng lớp theo tỉ lệ 10%), accuracy trung bình qua các task tính đến thời điểm đó, forgetting, train loss và train accuracy.

**Cổng G cho CIL:** thay "trước lần giảm LR cuối" bằng "mọi điểm đo của task 2–10, trừ điểm cuối task". Ba điều kiện còn lại giữ nguyên.

**Chi phí ước tính:** 200 + 9 × 170 epoch trên khoảng 7k ảnh mỗi task (5k ảnh mới cộng 2k exemplar). Tương đương khoảng 250 epoch CIFAR đầy đủ với ResNet32 (0.46M tham số), tức khoảng 1.5–2.5 GPU-giờ, chưa tính phần đo.

### B4. Model hẹp: ResNet-18 độ rộng 0.25, dùng lại hạ tầng `dual_growth`

Repo tham chiếu đã có ResNet-18 CIFAR với độ rộng tùy ý (gồm bản 0.25) và Gromo adapter cho đúng model này (`dual_growth/backbones`, `dual_growth/adapters`). Dùng lại hai phần đó **qua import**, không chép code.

- Recipe giống Phase 0 (200 epoch, LR 0.1, giảm ở epoch 100/150, batch 64), seed 0.
- Đo giống B1.
- Mục đích kép:
  1. thêm một điểm cho đường cong gain theo độ rộng (1/4 so với 1 ở Phase 0);
  2. sau này so trực tiếp với TINY grow thật. Repo đã có launcher `tiny_figure5` chạy bản TINYpub chính thức, xuất phát từ model nhỏ, dùng cho thí nghiệm cơ chế ở D3. **Không chạy launcher đó trong giai đoạn này.**

### B5. Mất plasticity / continual learning không dừng — **ưu tiên cao nhất trong B**

**Lý do.** Mất plasticity tương ứng với việc kernel của model cố định bị sụp: residual của task mới rơi vào các hướng chậm. VNA cộng K_E đúng vào các hướng đó (xem `claude/VNA_setting_survey.md`).

**Nguồn code.** Repo công khai của Dohare et al. (Nature 2024; arXiv 2306.13812; trên GitHub thường được biết là `shibhansh/loss-of-plasticity`). Agent tự xác minh đúng repo chính thức, pin commit và ghi license. Dùng nguyên giao thức, kiến trúc và siêu tham số gốc.

**Benchmark, rẻ trước:**

1. **Slowly-changing regression**: learner 1 lớp ẩn, 5 unit; mục tiêu là một target network cố định 100 unit. Chạy trên CPU hoặc GPU, toàn bộ chuỗi.
2. **Online Permuted MNIST**: MLP 3 lớp ẩn theo đúng bài gốc. Lần sàng lọc chỉ chạy **200 task đầu**.
3. **Continual ImageNet 32×32**: các task nhị phân, CNN theo bài gốc, 250 epoch mỗi task. Lần sàng lọc chạy **200 task đầu**. Nếu Kaggle chưa có dataset ImageNet 32×32 thì báo supervisor và bỏ qua bước này.

**Residual và metric:**
- Phân loại: dùng CE, H = diag(p) − ppᵀ.
- Hồi quy: dùng MSE, H = (2/n)·I, g = f − y.

**Site:** mọi cặp lớp ẩn liên tiếp (lớp ẩn → phi tuyến → lớp kế tiếp). Neuron ảo thêm unit ẩn, A lấy từ TINY, B = 0. Rank {1, 2, 4} cho mạng 5 unit; {4, 16, 64} cho MLP/CNN.

**Điểm đo:** ở mỗi task thứ t ∈ {1, 2, 5, 10, 20, 50, 100, 150, 200}, đo ở bước đầu task và giữa task.

**Log thêm (theo Dohare et al. và bài Spectral Collapse):**
- accuracy hoặc lỗi trực tuyến theo task;
- tỉ lệ unit "chết" (dormant);
- effective rank của activation;
- độ lớn trọng số;
- phổ của K trên batch đo (top-k trị riêng và tỉ lệ năng lượng residual nằm trong các hướng có trị riêng < μ).

**Đo thêm cho biến thể GIR (growth-informed reset), chỉ đo, không thay unit thật.** Ở mỗi điểm đo:

1. Xác định các unit **chết hoặc có utility thấp** theo đúng tiêu chí của CBP (utility) và ReDo (dormant, τ = 0.1 theo bài gốc). Ghi số lượng và tỉ lệ.
2. Với m unit đó, tính **gain bậc nhất phản thực**, tức mức giảm loss nếu thay chúng bằng m neuron mới:
   - `V_swap_tiny`: A chọn bởi `tiny_V`, đặt vào đúng vị trí các unit đó, B = 0;
   - `V_swap_random`: A ngẫu nhiên theo phân phối khởi tạo gốc (cách CBP/ReDo làm), cùng chuẩn hoá;
   - và tỉ lệ V_swap_tiny / V_swap_random.
3. Ghi thêm phần residual mà unit hiện tại không giải thích được nhưng neuron thay thế giải thích được (η trước và sau khi thay, chỉ tính phản thực).

Các số này không tham gia cổng G. Supervisor dùng chúng để quyết định có đưa GIR vào giai đoạn C hay không.

**Cổng G cho B5:** điều kiện ρ ≥ 0.3 tính trên các task thuộc **nửa sau** của chuỗi đã chạy. Báo thêm xu hướng ρ theo t: kỳ vọng ρ tăng khi kernel sụp.

### B6. PINN — **hạ xuống ưu tiên thấp nhất, chỉ đo phụ**

Lý do hạ ưu tiên: theo lemma chiếu lồng nhau (`claude/VNA_design_v2.md` §7.0), VNA không có lợi thế cấu trúc nào trước ENGD/Gauss–Newton, vốn là state of the art của PINN. Chỉ chạy B6 nếu còn quota.

**Nguồn code.** Repo của Rathore et al., "Challenges in Training PINNs" (ICML 2024). Agent xác minh repo chính thức và pin commit.

**Cấu hình:** các PDE convection, reaction và wave cùng tham số khó trong bài gốc; MLP với 2–3 mức độ rộng lấy theo bài; optimizer Adam+L-BFGS theo bài. Chỉ đo, không chạy VNA.

**Residual và metric:** loss là bình phương tối thiểu của residual PDE và điều kiện biên/ban đầu, nên g = vector residual, H = (2/n)·I, K = JJᵀ. J là Jacobian của residual PDE theo tham số; dùng JVP/VJP lồng qua đạo hàm theo input.
- **Test:** J khớp sai phân hữu hạn.

**Điểm đo:** 10 điểm đều trong pha Adam, cộng 3 điểm trong pha L-BFGS. Báo ρ, ρ_joint, η, phổ của K.

Không thêm ENGD hay NNCG ở giai đoạn này; chúng là baseline của giai đoạn C.

### Đo trong B (dùng watcher như Phase 0)

- Đo ở epoch 20, 40, …, 240, cộng thêm **148, 178, 208** (ngay trước mỗi lần giảm LR).
- **Site:** tất cả block; rank {4, 16, 64}; μ_rel lấy theo `mu_rel_default` của A, kèm 1e-2 để so.
- **Thước đo:** U, V, η, ρ, ρ_joint (từ A4), λ₁ với null N1 và N2 (n = 8), `tiny_V` so với `random_frob`, `subspace_sim`, `cg_converged`, train loss/accuracy tại checkpoint.
- **Selection mặc định là `tiny_V`**, có `tiny_U` để so.

### Cổng G (tính tự động, ghi vào `stageB/gate.json`)

Một bối cảnh **PASS** khi thoả cả bốn điều kiện, với r = 16, μ = `mu_rel_default`, và chỉ tính các hàng có CG hội tụ:

1. ρ_joint ≥ 0.3 ở ≥ 50% số điểm đo **trước lần giảm LR cuối** (epoch < 210);
2. `significant_N1` đúng ở ≥ 50% số điểm đó;
3. V(`tiny_V`) > V(`random_frob`) ở ≥ 70% số điểm đó;
4. train accuracy cuối < 99%.

Ghi thêm các con số thô bên cạnh PASS/FAIL. **Không** thay đổi ngưỡng sau khi đã thấy dữ liệu.

## 2. Notebook Kaggle (T4×2, một phiên, khoảng 8–10 giờ)

Chia thành **ba notebook**, vì tổng thời gian vượt một phiên 12 giờ. Thứ tự ưu tiên: **A → B5 → B3 → B2 → B4 → B1 → B6**.

- **Notebook 1, rẻ và quan trọng nhất:**
  - GPU0: A, rồi B5 Continual ImageNet-32 (200 task).
  - GPU1: B5 slowly-changing regression, rồi B5 Permuted MNIST (200 task), rồi B6 (PINN) nếu còn thời gian.
- **Notebook 2:**
  - GPU0: B3 (CIL iCaRL).
  - GPU1: B2 cặp resnet32x4 → resnet8x4, rồi B4.
- **Notebook 3, chỉ chạy nếu còn quota:**
  - B1 resnet8x4 và wrn_16_2;
  - B2 cặp wrn_40_2 → wrn_16_2.
- Watcher đo dùng GPU còn rảnh, như Phase 0.
- Smoke 2 epoch cho mỗi cấu hình B, in thời gian dự kiến; nếu tổng vượt `GPU_HOURS_BUDGET` thì dừng trước khi chạy full.
- Cell cuối in bảng cổng G, các hình ρ theo epoch (theo setting), V theo cách chọn, ρ(μ), rồi archive.
- Chạy cả `tests/test_vna.py`, `tests/test_vna_training.py` và các test mới trước khi train.

## 3. Test bắt buộc (mới)

1. Null N1 bảo toàn tập norm theo từng mẫu và ‖g‖; null N2 có tổng 0 theo từng mẫu và norm khớp.
2. p-value và cờ significant đúng trên dữ liệu giả lập: một trường hợp dựng sẵn có tín hiệu, một trường hợp không có.
3. Joint K_E với một site trùng single-site.
4. Mọi cách chọn A đi qua cùng hàm chuẩn hoá; `random_frob` có ‖A‖_F đúng bằng của TINY.
5. `subspace_sim` của A với chính nó bằng 1; của hai A ngẫu nhiên độc lập xấp xỉ r / d_in.
6. Kiến trúc RepDistiller: số tham số khớp; site adapter đạt adjoint/FD và bảo toàn output khi B = 0.
7. KD: g khớp autograd; H khớp Hessian sai phân hữu hạn.
8. Kiểm tra teacher bằng hash SHA256 (dùng file giả trong test CPU).
9. Cổng G: dữ liệu giả lập cho ra PASS/FAIL đúng; ngưỡng là hằng số trong code, có test khoá giá trị.
10. Diagnostics không đổi tham số, buffer hay mode (cho mọi kiến trúc mới).
11. Test hồi quy cho bug A5.
12. CIL (PyCIL):
    - thứ tự lớp với seed 1993 khớp PyCIL;
    - herding chọn đúng 2000 / số lớp đã thấy exemplar cho mỗi lớp;
    - loss KD khớp `_KD_loss` của PyCIL trên dữ liệu giả lập;
    - H của loss CIL khớp Hessian sai phân hữu hạn;
    - ResNet32 có số tham số khớp PyCIL.
13. B4: model ResNet-18 ×0.25 và adapter import từ `dual_growth` khớp số tham số và đạt test adjoint.
14. B5:
    - site adapter cho MLP và CNN nhỏ: adjoint/sai phân hữu hạn, output không đổi khi B = 0;
    - metric MSE đúng (H = 2I/n);
    - thứ tự task, permutation và luồng dữ liệu khớp code gốc với seed cố định (so 3 task đầu bit-for-bit).
15. **Lemma chiếu lồng nhau.** Trên ma trận ngẫu nhiên float64, J·J⁺·(vận tốc GN của model đã grow, μ → 0) khớp vận tốc GN của model cố định (sai lệch ≤ 1e-6). Đây là test khoá tính đúng của các công thức.
16. B6:
    - Jacobian của residual PDE khớp sai phân hữu hạn;
    - K = JJᵀ khớp bản dense trên một MLP rất nhỏ.

## 4. Không được làm

- Không cộng correction VNA vào gradient trong A và B.
- Không tune recipe của student hay teacher; dùng nguyên recipe CRD/DOT.
- Không dùng tập test chính thức cho quyết định (trừ kiểm tra teacher ở B0).
- Không đổi ngưỡng cổng G hay quy tắc chọn μ sau khi đã chạy.
- Không dùng empirical Fisher. Không định nghĩa K_H = JHJᵀ ngoài công thức GGN đã chốt.
- Không trộn output schema cũ và mới.
- Không thêm rollback, line-search hay bất kỳ cơ chế chọn checkpoint nào.

## 5. Bàn giao

- Commit đã ghim, notebook đã tạo lại theo commit đó, và tất cả test pass trên bản checkout sạch.
- Sau khi chạy, gửi supervisor:
  - `stageA/summary.json` và `STAGE_A_REPORT.md`;
  - `stageB/gate.json`;
  - mọi file CSV;
  - log train (train/eval accuracy cuối, thời gian mỗi epoch).
