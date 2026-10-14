# PROMPT GIAO AGENT: VNA, giai đoạn A + B (chẩn đoán và sàng lọc bối cảnh)

Bạn là kỹ sư nghiên cứu, viết code thí nghiệm cho phương pháp **VNA (Virtual-NTK Augmentation)** trong repo `duyh80456-code/Counterfactual-Projection`. Đọc toàn bộ prompt này **trước khi** viết dòng code đầu tiên.

## 0. Tài liệu bắt buộc đọc

Supervisor sẽ đặt các file sau vào `docs/vna/`:

- `STAGE_AB_SPEC.md`: spec chi tiết của giai đoạn A/B. Nếu có mâu thuẫn, **spec thắng prompt này** về chi tiết số liệu; prompt này thắng về cách tổ chức code.
- `DESIGN_v2.md`: các quyết định toán học. Đọc kỹ §5 (công thức GGN, các đối chứng), §6 (kết quả Phase 0) và §7 (lemma chiếu lồng nhau, điều chỉnh cho CL).
- `RUN_PLAN.md` và `SETTING_SURVEY.md`: bối cảnh, để hiểu vì sao làm.

Đọc thêm code hiện có:

- `methods/vna.py`, `methods/vna_ggn.py`, `methods/vna_select.py`, `methods/vna_kfac.py`
- `diagnostics/signal_curve.py`, `diagnostics/vna_solver_validation.py`
- `experiments/vna_support.py`, `experiments/vna_training_config.py`
- `tests/test_vna*.py`

## 1. Mục tiêu của đợt này

1. **Giai đoạn A:** sửa và kiểm tra lại các thước đo VNA trên checkpoint Phase 0 (ResNet-18/CIFAR-100) đã có.
2. **Giai đoạn B:** đo tín hiệu VNA (ρ, V, null, cách chọn A, tiềm năng của GIR) trong các bối cảnh mới, **chạy trên code gốc của từng lĩnh vực**, rồi tự động tính cổng G.
3. **Chỉ đo.** Trong đợt này không cộng correction VNA vào gradient, không thay unit thật.

## 2. Nguyên tắc tổ chức code: "upstream-first, hook mỏng"

Mục tiêu là reviewer có thể tin rằng baseline **chính là** code gốc của từng lĩnh vực, và phần của chúng ta chỉ là một lớp móc vào bên ngoài.

### 2.1 Mỗi bối cảnh dựa trên một repo gốc, pin commit

| Bối cảnh | Repo gốc | Phần dùng |
|---|---|---|
| B5: mất plasticity | `github.com/shibhansh/loss-of-plasticity` (Dohare et al., Nature 2024; URL lấy từ footnote của arXiv 2306.13812) | Slowly-changing regression; Online Permuted MNIST (3 lớp ẩn × 2000 unit, học từng mẫu một, không mini-batch); Continual ImageNet-32 (conv 32/64/128 + FC 128/128/2, 250 epoch mỗi task, batch 100, SGD momentum 0.9); CBP; L2; Shrink & Perturb |
| B3: class-incremental | `github.com/G-U-N/PyCIL` | `exps/icarl.json` (CIFAR-100 B0 Inc10, ResNet32, 2000 exemplar, seed 1993), `models/icarl.py` |
| B1/B2: model tí hon và KD | `github.com/HobbitLong/RepDistiller` | `resnet8x4`, `wrn_16_2`, teacher `resnet32x4` (79.42) và `wrn_40_2` (75.61) tải qua `scripts/fetch_pretrained_teachers.sh`; loss KD |
| B4: ResNet-18 ×0.25 | `duyh80456-code/One-Shot-TAS-CCIL`, nhánh `ccil-residual-capacity`, package `dual_growth` | Backbone độ rộng 0.25 và Gromo adapter (import, không chép) |
| B6 (ưu tiên thấp nhất) | Repo của Rathore et al., ICML 2024 (tự xác minh repo chính thức) | Convection/reaction/wave |
| Lõi toán VNA | Repo của chúng ta: `methods/vna*.py` | **Nguồn duy nhất** cho K, K_E, R, CG, chọn A, null |

**Cách đưa repo gốc vào:**

- Clone vào `third_party/<tên>` ở **một commit được pin**. Ghi URL, commit và license vào `third_party/SOURCES.md`.
- Ưu tiên **import**. Nếu buộc phải sửa repo gốc, sửa tối thiểu và lưu mỗi chỗ sửa thành một file `.patch` trong `third_party/patches/<tên>/`, kèm một dòng giải thích lý do. Không sửa trực tiếp mà không có patch.
- **Không viết lại** vòng train, loss, data pipeline hay kiến trúc của repo gốc. Nếu thấy cần viết lại thì dừng và báo supervisor.
- Repo gốc nào phải xác minh URL hoặc license mà không xác minh được, hoặc có license không cho phép dùng, thì dừng và báo supervisor.

### 2.2 Hook mỏng

- Viết một lớp `adapters/<setting>_hooks.py` cho mỗi bối cảnh. Lớp này:
  1. nhận model, batch và hàm loss **lấy từ chính repo gốc**;
  2. khai báo các *site* (cặp lớp: lớp thứ nhất → phi tuyến → lớp thứ hai) theo quy ước VNA;
  3. gọi hàm đo trong `methods/vna*.py` và `diagnostics/`.
- Hook được gọi tại các điểm đo, qua callback hoặc qua một wrapper quanh vòng train gốc, ở chỗ ít xâm lấn nhất.
- **Residual g lấy bằng autograd trên đúng object loss của repo gốc**: g = ∂L/∂z, với z là logit, hoặc là output đối với hồi quy và PDE. Không tự viết lại loss. Có thể dùng công thức giải tích để test đối chiếu, nhưng khi chạy thật thì dùng autograd.
- Metric H (GGN, Fisher thật) cho từng loại loss, phải có test sai phân hữu hạn:
  - CE: diag(p) − ppᵀ.
  - MSE: (2/n)·I.
  - KD theo RepDistiller (0.1·CE + 0.9·T²·KL, T = 4): 0.1·(diag(p) − ppᵀ) + 0.9·(diag(p_T) − p_T p_Tᵀ).
  - iCaRL theo PyCIL (CE + KD với T = 2, không nhân T²): diag(p) − ppᵀ trên các lớp đã thấy, cộng (1/T²)·(diag(q_T) − q_T q_Tᵀ) trên khối các lớp cũ.

## 3. Hợp đồng căn chỉnh với thuật toán (bắt buộc, có test)

1. **Neuron ảo.** Unit ẩn thêm vào giữa lớp thứ nhất và lớp thứ hai của site. Trọng số vào A lấy từ TINY/`tiny_V`, trọng số ra B = 0. **Không bao giờ cài vào model.** Số tham số của model không đổi; assert điều này ở mọi điểm đo.
2. **Công thức.**
   - K = JJᵀ (gộp với H theo dạng GGN như trong `vna_ggn.py`), K_E là kernel của B tại B = 0, R = (K + μI)⁻¹;
   - correction d = Jᵀ R K_E R K g;
   - ρ = V / ‖Jᵀg‖²;
   - μ = μ_rel · tr(K)/n, đúng như code hiện có.
   - Không tự đặt định nghĩa mới. Không dùng empirical Fisher.
3. **Chỉ đo.** Tham số, buffer (BN/LN), mode train/eval và **RNG của repo gốc** phải giữ nguyên trước và sau mỗi lần đo. Mọi phép đo dùng generator riêng (`dedicated_generator`).
4. **Test căn chỉnh quan trọng nhất:** với mỗi bối cảnh, chạy vòng train gốc ngắn (vài chục bước, CPU, seed cố định) hai lần, **có hook đo và không có hook**. Tham số cuối phải **giống hệt bit-for-bit**. Test này chứng minh việc đo không làm nhiễu baseline.
5. **Dữ liệu cho phép đo** chỉ lấy từ những gì vòng train gốc được phép thấy tại thời điểm đó:
   - dữ liệu task hiện tại;
   - cửa sổ trượt các mẫu gần nhất (với benchmark online; khai báo rõ kích thước);
   - buffer exemplar (CIL).
   Không dùng tập test chính thức, không dùng dữ liệu task tương lai.
6. **Null.** Gồm N1 (hoán vị residual giữa các mẫu) và N2 (nhiễu Gauss có tổng bằng 0 theo từng mẫu, cùng norm), mỗi loại 8 lần. Chọn A lại từ residual null. Quy tắc p-value theo spec.
7. **Mọi cách chọn A** đi qua cùng một hàm chuẩn hoá. Random-A khớp ‖A‖_F với A của TINY.
8. **Lemma chiếu lồng nhau** (DESIGN §7.0) có test khoá, chạy trên ma trận ngẫu nhiên float64.

## 4. Cổng tái lập baseline (phải qua trước khi đo)

Ở mỗi bối cảnh, chạy code gốc **không có hook** và đối chiếu với số đã công bố. Không qua cổng thì không được đo, mà phải báo lại.

| Bối cảnh | Cổng |
|---|---|
| B5 | Bản tái lập ngắn cho thấy mất plasticity xuất hiện với backprop thường theo đúng xu hướng trong bài gốc. Ví dụ Permuted MNIST: accuracy trực tuyến giảm dần theo task. CBP giữ được accuracy. |
| B3 | iCaRL trên CIFAR-100 B0 Inc10 cho accuracy trung bình qua các task trong khoảng ±1.5pp so với số PyCIL công bố. |
| B1/B2 | Teacher tải về đạt ≥ 79.0% (resnet32x4) và ≥ 75.0% (wrn_40_2) trên test chính thức. Student tự train với recipe gốc trên 45k: ghi lại số; so với số gốc (resnet8x4 72.50, wrn_16_2 khoảng 73.3; số gốc train trên 50k) chỉ để tham khảo, sai khác > 2pp thì báo. |
| B4 | Số tham số khớp với `dual_growth`; test adjoint đạt. |

## 5. Thứ tự làm và Kaggle

**Thứ tự ưu tiên:** A → B5 → B3 → B2 (resnet32x4 → resnet8x4) → B4 → B1 → B2 (wrn) → B6.

**Cách chia notebook:** theo spec (ba notebook T4×2).

**Quy tắc chung cho mọi notebook:**

- Pin commit; mỗi lần đổi code thì tạo lại notebook với hash mới.
- Archive mỗi giờ, có resume, giới hạn đĩa như hiện tại.
- Smoke 2 epoch (hoặc 2 task) cho mỗi cấu hình, in GPU-giờ dự kiến, vượt `GPU_HOURS_BUDGET` thì dừng.
- Chạy toàn bộ test trước khi train.
- Ô cuối in cổng G và các hình theo spec.

## 6. Test bắt buộc

Gồm danh sách test 1–16 trong spec, cộng thêm:

- **(T-align)** Test hook-on/hook-off bit-for-bit cho từng bối cảnh (mục 3.4).
- **(T-loss)** g lấy bằng autograd trên loss gốc khớp công thức giải tích; H khớp Hessian sai phân hữu hạn, cho CE, MSE, KD và iCaRL.
- **(T-upstream)** Mỗi patch trong `third_party/patches/` apply sạch trên commit đã pin; chạy entry point gốc với `--help` hoặc với smoke nhỏ thì thành công.
- **(T-params)** Số tham số không đổi sau mọi phép đo, ở mọi bối cảnh.

## 7. Không được làm

- Không cộng correction VNA vào gradient. Không thay unit thật (GIR chỉ được đo phản thực).
- Không viết lại vòng train, loss hay kiến trúc của repo gốc. Không tune siêu tham số của repo gốc.
- Không dùng tập test chính thức cho bất kỳ quyết định nào, trừ cổng kiểm tra teacher.
- Không đổi ngưỡng cổng G hay quy tắc chọn μ sau khi đã thấy dữ liệu.
- Không dùng empirical Fisher. Không định nghĩa kernel mới ngoài GGN đã chốt.
- Không trộn output schema cũ và mới (`schema_version = stage1d-v1`).
- Không thêm rollback, line-search hay chọn checkpoint theo kết quả.
- Không âm thầm bỏ qua lỗi tải repo hoặc dữ liệu: dừng và báo.

## 8. Bàn giao

1. Một PR/commit gồm:
   - `third_party/SOURCES.md` và các patch;
   - các adapter hook;
   - code đo;
   - test;
   - notebook đã tạo lại theo commit pin.
   Tất cả test phải pass trên bản checkout sạch.
2. `docs/vna/STAGE_AB_IMPLEMENTATION.md` gồm:
   - mỗi bối cảnh dùng repo nào, commit nào, sửa gì và vì sao;
   - site được khai báo ở đâu;
   - bảng kết quả của cổng tái lập;
   - những chỗ code gốc khác với bài báo (nếu phát hiện).
3. Sau khi chạy, gửi supervisor:
   - `stageA/summary.json` và `STAGE_A_REPORT.md`;
   - `stageB/gate.json`;
   - mọi file CSV;
   - log train (accuracy cuối, thời gian mỗi epoch hoặc task);
   - bảng tiềm năng GIR (V_swap_tiny / V_swap_random).

Nếu có chỗ nào trong spec không thể làm đúng như mô tả với code gốc, ví dụ vòng train online không có chỗ móc hook, thì **đề xuất phương án và chờ duyệt**. Không tự chọn.
