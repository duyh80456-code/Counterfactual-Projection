# PROMPT CHO CODING AGENT — ETT (Expand–Train–Transplant) cho GIR / plasticity
(2026-11-01. Bản này THAY THẾ mọi prompt VNA/GIR/SCVE trước đó. Đọc hết trước khi viết dòng code nào.)

---

## 0. Bối cảnh ngắn và những gì đã BỎ

Dự án trước (VNA/SCVE: neuron ảo không được train, chiếu cập nhật qua Jacobian, bước GN/CG, certificate, β_cert, test LM) đã bị dừng: có định lý cho thấy phép chiếu chỉ chuyển được *lựa chọn hướng*, không chuyển được *năng lực*. Phương pháp mới là **ETT**:

> Định kỳ thêm k **neuron ảo** (trọng số ra = 0 nên hàm không đổi) → **train thật** mạng mở rộng h bước, cùng mọi tham số khác → **thu gọn** về đúng n unit bằng cách chọn tập giữ lại trong n + k unit (thật và ảo được xét như nhau), **bù least squares** lên trọng số ra của unit được giữ → kiểm chấp nhận → chép trạng thái optimizer/BN. Mô hình triển khai luôn có kích thước gốc.

**KHÔNG được dùng** (không import, không port từ code cũ): Gauss–Newton, CG, Jacobian theo từng mẫu, kernel K_E, certificate, β_cert, LM test, lookahead/meta-gradient, teacher eval-mode của SCVE. Code cũ chỉ được tái dùng cho: data loader, logging, checkpoint/resume.

**Prompt này tự đủ.** Không cần tài liệu nào khác: phát biểu các kết quả C1–C5 ở Phụ lục A, cài đặt tham chiếu (numpy, đã kiểm với lstsq và backward ngây thơ) ở Phụ lục B. Mã C# trong prompt chỉ để ghi vào comment; chứng minh đầy đủ nằm ở tài liệu riêng của tác giả, agent không cần đọc.

Nguyên tắc dự án (bắt buộc):
1. Mọi thành phần thuật toán hoặc đã có chứng minh (ghi mã C# trong comment), hoặc phải được **đo** trước khi đưa vào run chính. Không thêm heuristic tự chế. Nếu gặp chỗ prompt không quy định, chọn phương án đơn giản nhất, ghi vào `DECISIONS.md`, **không** tự thêm cơ chế mới.
2. Không đốt GPU: làm theo pha, có cổng dừng (gate). Không chạy grid lớn.
3. **Test:** chỉ các test nhắm đích ngắn liệt kê ở §8 (float64, CPU, tổng < 1–2 phút). **Không** chạy toàn bộ pytest suite, **không** smoke test riêng / run thử end-to-end để "xem có chạy không". Run thật đầu tiên của mỗi pha chính là kiểm tra.

---

## 1. Ký hiệu

- Lớp ẩn ℓ có n unit. Unit i: đặc trưng h_i(x) (sau norm + activation), cột ra c_i (cột của trọng số lớp ℓ+1 nối từ unit i), z_{ℓ+1} = Σ_i c_i h_i + c₀ là **pre-activation (trước norm) của lớp ℓ+1**.
- Mở rộng: U = tập n + k unit (n thật + k ảo). Thu gọn: chọn S ⊂ U, |S| = n; D = U ∖ S, |D| = k.
- A = [H_U, 1] (ma trận đặc trưng trên cửa sổ, cột cuối là intercept), G = AᵀA.
- Vì z_{ℓ+1} của mạng mở rộng **đúng bằng** A·[C, c₀]ᵀ, mọi đại lượng thu gọn chỉ cần **G và trọng số ra hiện tại** (không cần lưu Z):
  - g(D) = min_W ‖Z − A_{S∪1} W‖²_F = tr(C_D [(G⁻¹)_{DD}]⁻¹ C_Dᵀ) (C3c, OBS/OBC; G⁻¹ ở đây là nghịch đảo của G trên U∪{1}).
  - Nghiệm bù: W_S = C_Sᵀ + G_{SS}⁻¹ G_{SD} C_Dᵀ (S gồm cả intercept).

### 1.1 Unit hợp lệ (giả thiết A1, A2 — kiểm tra trong code, assert nếu vi phạm)
- **A1:** lớp ℓ không có norm, hoặc BatchNorm **theo kênh** (train hay eval đều được). **Không** áp dụng cho lớp có LayerNorm/GroupNorm (C1-NEG: phá tính chính xác). Gặp lớp như vậy → bỏ qua lớp đó, log.
- **A2:** unit chỉ ảnh hưởng phía sau qua cột c_i. MLP: mọi lớp ẩn. ResNet basic block: **chỉ kênh ra của conv1** (bn1 → relu → conv2). Không áp dụng cho kênh ra của conv2/shortcut (bị cộng residual), không áp dụng cho stem nếu nó đi vào residual stream.

---

## 2. Thuật toán ETT — đặc tả chính xác

Mỗi chu kỳ bắt đầu mỗi T bước train (và tuỳ chọn ở đầu mỗi task — siêu tham số `cycle_at_task_start`, mặc định false). Chu kỳ dùng **chính các batch của stream** (không thêm dữ liệu); số bước train tổng giữ nguyên như BP.

### Bước 1 — Expand (C4a, C4b)
Với mỗi lớp hợp lệ ℓ: thêm k_ℓ = max(1, round(ρ·n_ℓ)) unit ảo, ρ ∈ {0.05, 0.1} (đăng ký trước).
- Trọng số ra b (cột mới của lớp ℓ+1) = **0 chính xác** ⇒ hàm không đổi (test T1).
- BN của lớp ℓ cho unit ảo: γ = 1, β = 0, running_mean/var = giá trị đo trên batch hiện tại (để eval-mode cũng không đổi hàm vì b = 0 thì không quan trọng, nhưng tránh NaN).
- **Trọng số vào — `init=gradmax_pool` (mặc định):** lấy P = 8·k ứng viên từ đúng phân phối init gốc của lớp; nếu lớp **không** có BN, chuẩn hóa mỗi ứng viên về chuẩn kỳ vọng của phân phối init (để so ở cùng chuẩn, như GradMax); trên batch hiện tại tính δ = ∂L/∂z_{ℓ+1} (N×d) và H_cand (N×P, đặc trưng ứng viên, BN train-mode nếu có); ‖g_b,j‖ = ‖δᵀ H_cand[:, j]‖; chọn top-k. Lý do: C4b — lợi bậc một của bước đầu = η‖g_b‖². Với conv: g_b,j là gradient theo kernel ra 3×3 của kênh ứng viên (dùng unfold/conv, không cần vòng lặp).
- Ablation `init=random`: lấy k mẫu từ phân phối init.
- Unit ảo vào optimizer với cùng lr, weight decay, nhóm tham số như unit thật. Adam: trạng thái mới = 0 với **bộ đếm step theo từng hàng/kênh** (xem §4.3) — không dùng chung step của tensor.

### Bước 2 — Train mạng mở rộng h bước
Train bình thường h bước bằng optimizer gốc trên các batch kế tiếp của stream. Không có loss phụ, không đóng băng gì. h ∈ {0, 10, 100} (h = 0 là arm ablation).

### Bước 3 — Thống kê cửa sổ (chính xác, không xấp xỉ)
Sau h bước, với **mạng mở rộng cuối cùng E** (đã cố định), chạy `w_b` forward **no-grad** trên w_b batch gần nhất (lấy từ buffer các batch vừa train; mặc định w_b = 4; với MLP nhỏ có thể dùng toàn bộ h batch nếu h ≤ 16). Tích lũy cho mỗi lớp hợp lệ:
- MLP: G = Σ_batches AᵀA, kích thước (m+1)², m = n + k.
- Conv (ResNet conv1→conv2): A = unfold(h, kernel 3×3, padding/stride/dilation đúng như conv2), mỗi kênh i ứng với **nhóm 9 cột**; thêm cột intercept. G kích thước (9m+1)². Tích lũy bằng einsum, float64 cho G (đặc trưng có thể float32 trên GPU, cộng dồn vào float64).
- BN lớp ℓ trong lúc thu thập: dùng **train-mode** (đúng chế độ lúc train) nhưng **không** cập nhật running stats (lưu và khôi phục, hoặc momentum = 0). Lý do: tránh thu gọn làm bẩn thống kê.
- Không cần lưu Z (xem §1).
- Ridge: G_reg = G + ε·(tr(G_units)/(m·r))·I chỉ trên khối unit (không trên intercept), ε = 1e-6 (r = 1 cho MLP, 9 cho conv).

Lý do chọn forward riêng thay vì cộng dồn trong h bước train: C3 chính xác cho một mạng cố định; cộng dồn qua các bước làm A và Z thuộc các tham số khác nhau. Chi phí w_b forward được tính vào compute.

### Bước 4 — Chọn tập bỏ D (C3c–C3e)
Ứng viên có thể bị bỏ = tất cả unit ảo + unit thật **không được bảo vệ** (bảo vệ: tuổi < m, xem bước 7). Số cần bỏ = k_ℓ (luôn khả thi vì unit ảo không bao giờ được bảo vệ).
1. **Unit chết** (G_ii = 0, tức ‖h_i‖² < 1e-12·tr(G_units)/m trên cửa sổ; conv: cả nhóm 9 cột) có chi phí 0, loại trước (C1: chính xác). Nếu số unit chết ≥ k_ℓ thì chọn k_ℓ unit chết (ưu tiên unit thật chết; còn lại theo chỉ số), xong.
2. Trên phần còn lại, tính **hai** tập ứng viên:
   - **D_score:** k_ℓ unit có s_i nhỏ nhất, s_i = ‖A_i C_iᵀ‖_F = sqrt(tr(C_i G_ii C_iᵀ)) (MLP: = ‖h_i‖‖c_i‖). Đảm bảo κ(Ĝ)-xấp xỉ đã chứng minh cho MLP (C3d); với conv chỉ là ứng viên.
   - **D_back:** loại bỏ ngược có bù, bằng nghịch đảo + downdate (không giải lại LS):
     - M = G_reg⁻¹ (qua Cholesky, float64); W = nghiệm LS hiện tại trên toàn U∪1 = [C, c₀]ᵀ (vì Z = A[C,c₀]ᵀ chính xác).
     - Lặp k_ℓ lần: với mỗi ứng viên i (nhóm chỉ số I_i, kích thước 1 hoặc 9): cost_i = tr(W_{I_i}ᵀ (M_{I_i I_i})⁻¹ W_{I_i}). Chọn i* = argmin. Cập nhật (R = các chỉ số còn lại):
       W_R ← W_R − M_{R,I}(M_{I,I})⁻¹ W_I; M_{RR} ← M_{RR} − M_{R,I}(M_{I,I})⁻¹ M_{I,R}.
     - Tổng O(k(n+k)²) (conv: nhân 81).
   - Tính chính xác g(D_score) và g(D_back) bằng công thức C3c; **D = argmin của hai** (cả hai đều được đánh giá bằng hàm mục tiêu chính xác, nên lựa chọn này không phải heuristic).
3. Log: g(D_score), g(D_back), g(D), số unit ảo được giữ, số unit chết, và (chỉ khi m ≤ 20, k ≤ 3, mỗi 10 chu kỳ) OPT bằng vét cạn trên CPU để log tỉ số g(D)/OPT (kiểm thực nghiệm C3e).

### Bước 5 — Bù least squares (C3a)
W_S = (G_reg)_{SS}⁻¹ (G_reg)_{S,U∪1} [C, c₀]ᵀ (S gồm intercept). Ghi W_S thành trọng số ra mới của các unit giữ lại, phần intercept:
- MLP có bias lớp ℓ+1: ghi vào bias.
- Không có bias, theo sau là BN (conv2 → bn2): train-mode BN bất biến với hằng số cộng nên bỏ qua trong train; với eval-mode thì running_mean ← running_mean + c₀_new − c₀_old (hằng số cộng trước BN tương đương dịch running_mean). Ghi chú trong comment.
- Không bias, không BN: bỏ cột intercept khỏi A (giải LS không intercept).
Cập nhật này kết thúc bằng **một** tensor trọng số ra mới; không cập nhật trọng số vào.

### Bước 6 — Dựng lại lớp và kiểm chấp nhận
- **Thứ tự slot (Bổ đề 0: không ảnh hưởng hàm):** unit thật được giữ ở nguyên slot; unit ảo được giữ lấp các slot bị giải phóng theo thứ tự tăng dần.
- Chép theo từng unit giữ lại: hàng trọng số vào + bias, γ/β/running_mean/running_var của BN lớp ℓ, cột ra (đã bù), **và trạng thái optimizer tương ứng** (momentum, exp_avg, exp_avg_sq, step theo hàng). Không đặt lại về 0 cho unit ảo được giữ.
- **Chấp nhận:** trên một batch mới (batch kế tiếp của stream, chưa dùng ở bước 3), đo ΔC = L(C) − L(E) (cùng chế độ BN như lúc train; log thêm eval-mode). Nếu ΔC ≤ τ_rel·L(E) → chấp nhận. Ngược lại → **fallback:** D = toàn bộ unit ảo (giữ toàn bộ unit thật), vẫn bù LS (bước 5), log ΔC_fallback. Không có hoàn tác về snapshot trước h bước (sẽ vứt h bước train, không công bằng). τ_rel ∈ {0.01, 0.05}, đăng ký trước. Batch kiểm sau đó vẫn được dùng để train như bình thường (không tốn dữ liệu).

### Bước 7 — Hòa nhập sau thu gọn
- Mỗi unit có bộ đếm tuổi (bước). Unit ảo vừa được giữ có tuổi 0. Unit có tuổi < m (bước train) **không** được là ứng viên bỏ (giống maturity threshold của CBP). m ∈ {T, 5T} đăng ký trước.
- Không có cơ chế nào khác (không warm-up lr riêng, không loss phụ).

### 2.1 Nhiều lớp
Thực hiện bước 3–6 **tuần tự từ lớp nông tới sâu**, mỗi lớp dùng G đo trên mạng đã thu gọn ở các lớp trước (thu lại w_b forward cho lớp sau). Mặc định: tuần tự, ghi chú chi phí. Ablation (chỉ nếu Phase 1 dương tính): đồng thời mọi lớp với cùng G đo một lần.

### 2.2 Chi phí (C5)
Overhead ≈ (h/T)(k/n) cho phần train + w_b forward + đại số tuyến tính. Chọn T sao cho overhead đo bằng wall-clock ≤ 3% (log chính xác). Không dùng Jacobian, GN, CG, certificate.

---

## 3. Arm (cùng số bước train, cùng dữ liệu; compute log đầy đủ)

| Arm | Mô tả | Tách tác dụng của |
|---|---|---|
| BP | baseline | — |
| BP+compute | BP cộng thêm số bước SGD lặp lại batch hiện tại (xác suất p) sao cho tổng FLOPs = ETT đầy đủ | compute |
| CBP | Dohare et al. Nature 2024, code upstream `shibhansh/loss-of-plasticity` (giữ nguyên siêu tham số gốc theo setting) | reset ngẫu nhiên |
| ReDo | Sokar et al. 2023, τ dormant theo paper | reset unit ngủ |
| SNR **hoặc** SWR | chọn cái có code chính thức; ghi lý do trong DECISIONS.md | reset gần đây |
| S&P | shrink-and-perturb | — |
| GradMax-reset | như CBP (cùng tiêu chí chọn unit, cùng tần suất) nhưng unit mới có trọng số vào chọn bằng `gradmax_pool` của Bước 1, trọng số ra 0 | khởi tạo theo gradient |
| ETT h=0 | ETT với h = 0 | tác dụng của "train" |
| ETT-drop | ETT đầy đủ nhưng luôn D = toàn bộ unit ảo (chỉ mở rộng-train-bỏ, có bù) | tác dụng của "chọn/cấy" |
| **ETT** | đầy đủ | — |

Ablation (chỉ Phase 1, chỉ trên setting nhỏ, chỉ nếu ETT dương tính): h ∈ {10, 100}; ρ ∈ {0.05, 0.1}; m; τ_rel; D_score-only vs D_back-only vs min; chỉ-unit-chết vs mọi unit; không bù LS (chỉ chọn); `init=random`.

**Tinh chỉnh công bằng:** mỗi arm (kể cả baseline) được cùng số cấu hình thử (tối đa 4), chỉ trên seed 0; chọn theo metric chính của §6; seed 1–3 dùng cấu hình đã chọn, không chỉnh thêm. Ghi mọi cấu hình đã thử.

---

## 4. Chi tiết cài đặt bắt buộc

### 4.1 Cấu trúc
```
ett/
  core/units.py        # trừu tượng lớp hợp lệ: LinearUnitLayer, ConvUnitLayer (conv1→bn1→relu→conv2)
  core/expand.py       # Bước 1 (+ gradmax_pool)
  core/stats.py        # Bước 3 (G, unfold, ridge, BN no-update)
  core/select.py       # Bước 4 (dead, score, backward+downdate, exact g, brute-force OPT)
  core/compensate.py   # Bước 5
  core/rebuild.py      # Bước 6: dựng tensor, remap optimizer state + BN, tuổi unit
  core/ett.py          # điều phối chu kỳ, acceptance, logging
  optim/adam_rowstep.py# Adam với step theo hàng/kênh (§4.3), dùng cho MỌI arm có reset
  baselines/           # cbp (upstream), redo, snr|swr, sp, gradmax_reset
  envs/                # scr.py, pmnist.py, cimagenet32.py, icifar100.py
  phase0/measure_ge.py
  tests/test_ett_core.py  # chỉ các test §8
  configs/             # yaml đăng ký trước, commit trước khi chạy
  DECISIONS.md
```

### 4.2 Dựng lại tham số
Expand/contract đổi shape tensor. Dựng `nn.Parameter` mới và **remap** `optimizer.state` + `param_groups` (thay tham chiếu tham số cũ bằng mới; gather theo chỉ số hàng/cột cho mọi tensor trạng thái cùng shape với tham số). Không dùng "mask k slot cố định" (tốn FLOPs ngoài chu kỳ, phá công bằng compute). Các tham số không liên quan giữ nguyên object.

### 4.3 Adam theo hàng
Nếu setting dùng Adam: bộ đếm step theo unit — cho hàng trọng số vào (+ bias, γ/β BN) **và** cho cột trọng số ra của unit đó. Bias correction dùng step của hàng/cột tương ứng. Unit ảo bắt đầu step = 0 ở mọi tensor của nó. Lý do: tham số mới với step chung lớn làm bước Adam đầu tiên ≈ (1−β₁)/√(1−β₂)·lr ≈ 3.16·lr — sai ngữ nghĩa. Áp dụng **giống hệt** cho CBP/ReDo/SNR/SWR/GradMax-reset (nếu upstream CBP đã có AdamGnT thì dùng của nó và kiểm cùng ngữ nghĩa). SGD: momentum mới = 0.

### 4.4 Số học
Đại số tuyến tính thu gọn: float64 (CPU cho lớp nhỏ, GPU float64 cho conv lớn). Cholesky thất bại → tăng ε ×10 tối đa 3 lần, log; vẫn thất bại → bỏ chu kỳ đó ở lớp đó (log). Không im lặng nuốt lỗi.

### 4.5 RNG và tái lập
Seed riêng cho: data stream, init mạng, ETT (pool ứng viên). Các arm ghép cặp dùng chung seed data + init. Pin commit (ghi hash vào mỗi file kết quả). Checkpoint (mạng + optimizer + buffer + tuổi unit + RNG) mỗi N bước để resume trên Kaggle.

---

## 5. Setting

1. **SCR** — Slowly-Changing Regression của Dohare (repo upstream), learner nhỏ đúng như cấu hình upstream (mạng 1 lớp ẩn, 5 unit; không norm). Ghi rõ cấu hình dùng. CPU.
2. **PM-small** — Online Permuted MNIST, MLP 3×10 unit (như setting mạng nhỏ của SWR), không norm, SGD hoặc Adam theo paper tham chiếu. CPU.
3. **CI32** — Continual ImageNet 32×32 (task nhị phân) của Dohare, mạng conv upstream (không BN → ConvUnitLayer không norm; lớp FC ẩn → LinearUnitLayer). GPU. Kiểm khả năng tải dữ liệu trên Kaggle; nếu không có, báo lại, **không** tự thay dataset.
4. **iCIFAR100** (tuỳ chọn, chỉ khi 1–3 cho kết quả dương) — incremental CIFAR-100 của Dohare, ResNet-18 BN; lớp hợp lệ = conv1 trong mỗi basic block.

---

## 6. Các pha và cổng dừng

### Phase 0 — đo G_E (gate). Chỉ SCR và PM-small.
Mục đích: kiểm thực nghiệm phần lý thuyết **không** chứng minh được (C4d1: G_E ≥ 0 sai trong trường hợp tổng quát; chỉ đúng khi K, K_E giao hoán hoặc ηhλ nhỏ — C4d2, C4d3).
- Chạy BP, lưu checkpoint ở ≥ 20 thời điểm trải đều (gồm đầu, giữa, cuối — lúc plasticity đã giảm).
- Từ mỗi checkpoint θ, với **cùng chuỗi batch** và cùng RNG, h ∈ {1, 10, 100}:
  - (a) native: train h bước → L_N; (b) ETT: expand + train h bước → L_E; thu gọn + bù → L_C; (c) CBP: một lần thay unit như CBP tại θ rồi train h bước → L_CBP; (d) GradMax-reset tương tự → L_GM.
  - Đo tất cả trên **cùng** các batch đánh giá mới (current task, chưa dùng để train), ≥ 8 batch.
  - G_E = L_N − L_E; ΔC = L_C − L_E; net = L_N − L_C (= G_E − ΔC, C4c); cũng log L_CBP − L_C, L_GM − L_C.
  - h = 1: so G_E với dự đoán bậc một η‖g_b‖² (C4b) → log tỉ số.
  - Log tỉ lệ unit ảo được giữ, g(D_score)/g(D_back), tỉ lệ unit chết.
- **Cổng:** đi tiếp Phase 1 nếu trên **ít nhất một** setting, với **ít nhất một** h ∈ {10, 100}: median(net) > 0 **và** sign test ghép cặp qua checkpoint p < 0.05, **và** median(L_CBP − L_C) > 0. Nếu không → DỪNG, gửi báo cáo Phase 0 (không chạy Phase 1).
- Phase 0 chỉ dùng 1 seed, CPU. Không tinh chỉnh siêu tham số ETT ngoài h (ρ = 0.1, init gradmax_pool, τ không áp dụng — log ΔC thô).

### Phase 1 — run đầy đủ SCR + PM-small (CPU)
Mọi arm §3, seed 0 để chọn cấu hình (≤ 4 cấu hình/arm), seed 1–3 đánh giá. Ablation chỉ khi ETT đạt tiêu chí trên ≥ 1 setting.

### Phase 2 — CI32 (GPU, Kaggle T4×2)
Chỉ arm: BP, BP+compute, CBP, GradMax-reset, ETT h=0, ETT-drop, ETT (+ SNR/SWR nếu rẻ). Cấu hình ETT lấy từ Phase 1 (không tinh chỉnh lại trừ T để giữ overhead ≤ 3%). 3 seed.

### Phase 3 — iCIFAR100 (tuỳ chọn, phải hỏi lại trước khi chạy).

**Tiêu chí thành công (đăng ký trước, không đổi sau khi thấy kết quả):** ETT hơn CBP, GradMax-reset **và** ETT h=0 trên metric chính ở ≥ 2/3 seed ghép cặp, trên ≥ 2 setting, ở cùng compute (so cả với BP+compute).
**Metric chính:** trung bình online accuracy (PM, CI32) / online loss (SCR) trên 20% task cuối.

---

## 7. Logging (JSONL theo run + CSV tổng)
Mỗi task/khoảng: online acc/loss; plasticity gap (Dohare); tỉ lệ unit chết; effective rank đặc trưng; ‖w‖ trung bình.
Mỗi chu kỳ ETT, mỗi lớp: k, h, số chết, g(D_score), g(D_back), g(D), OPT (nếu có), ΔC (train + eval mode), chấp nhận/fallback, số unit ảo được giữ, churn (số lần mỗi slot bị thay, histogram cuối run), thời gian từng bước (expand, train, stats, select, compensate, rebuild, check).
Toàn run: wall-clock, FLOPs ước lượng, overhead % so với BP, commit hash, config.

---

## 8. Test nhắm đích DUY NHẤT được phép (float64, CPU, tiny MLP/conv, tổng < 2 phút)
Viết trong `tests/test_ett_core.py`, chạy **một lần** sau khi viết `core/` và sau mỗi lần sửa `core/` thì chỉ chạy test liên quan. Không chạy pytest toàn repo, không smoke test.
- **T1** expand bảo toàn hàm (MLP và conv, BN train + eval): max|f_E − f| < 1e-12.
- **T2** C1: unit bị bỏ đều chết trên cửa sổ ⇒ thu gọn chính xác (< 1e-12), BN train + eval.
- **T3** C2: sai số tại z_{ℓ+1} = −Σ_{D} c_i h_i (< 1e-12).
- **T4** Bổ đề 0: hoán vị slot của unit ảo được giữ không đổi output (< 1e-12).
- **T5** bù LS = `torch.linalg.lstsq` trên cửa sổ (cùng ridge = 0, N > m), MLP và conv (unfold) (< 1e-8).
- **T6** g(D) theo C3c = residual LS trực tiếp (< 1e-8 tương đối), cả nhóm 9 cột.
- **T7** backward có downdate = backward ngây thơ giải lại LS (cùng D) trên 20 bài ngẫu nhiên.
- **T8** remap optimizer state (SGD momentum, Adam theo hàng) + BN: tensor của unit giữ lại khớp bitwise với trước thu gọn.
- **T9** ETT với k = 0 (code path tắt) khớp BP bitwise sau 50 bước.
- **T10** h = 0 và D = toàn bộ unit ảo ⇒ tham số, state optimizer, output khớp bitwise với trước expand (vì b = 0 nên bù = 0).
T5–T7 so với cài đặt tham chiếu ở Phụ lục B (dùng sai số tương đối theo ‖W‖, vì G có thể điều kiện kém).

---

## 9. Báo cáo gửi lại (sau mỗi pha, dừng chờ trước khi sang pha sau)
1. Commit hash, DECISIONS.md, config đã đăng ký.
2. Kết quả T1–T10 (một dòng mỗi test).
3. Phase 0: bảng theo setting × h: median/IQR của G_E, ΔC, net, L_CBP − L_C, L_GM − L_C, tỉ lệ dương, p sign test; tỉ số G_E/(η‖g_b‖²) ở h = 1; tỉ lệ unit ảo được giữ; g(D_score) vs g(D_back); kết luận gate.
4. Phase 1/2: bảng metric chính theo arm × seed (ghép cặp), overhead %, đồ thị online acc/loss theo task, dead fraction, effective rank, ΔC theo chu kỳ, tỉ lệ fallback, churn.
5. Mọi bất thường (NaN, Cholesky fail, fallback > 30%, overhead > 3%) — báo, không tự sửa thuật toán.
6. Zip toàn bộ JSONL/CSV + đồ thị.

## 10. Những điều KHÔNG làm
- Không thêm loss phụ, distillation, lookahead, warm-up riêng cho unit mới, hay bất kỳ cơ chế nào không có trong §2.
- Không áp dụng ETT cho lớp vi phạm A1/A2.
- Không đổi tiêu chí thành công, metric chính, hay setting sau khi đã thấy kết quả.
- Không chạy Phase sau khi gate trước chưa qua; không chạy Phase 3 khi chưa hỏi.
- Không chạy full test suite hay smoke test.

---

## Phụ lục A — Các kết quả lý thuyết (phát biểu; dùng để ghi comment và hiểu vì sao từng bước như vậy)
Giả thiết: A1 (lớp ℓ không norm hoặc BN theo kênh), A2 (unit chỉ ảnh hưởng phía sau qua c_i). Ký hiệu như §1.
- **Bổ đề 0 (hoán vị):** z_{ℓ+1} = Σ_{i∈S} c_i h_i + c₀ không phụ thuộc unit nào nằm ở slot nào ⇒ chỉ tập giữ S quan trọng, không có bài toán gán.
- **C1 (thu gọn chính xác):** nếu mọi unit trong D đều chết trên cửa sổ (h_i ≡ 0) thì mạng thu gọn trùng mạng mở rộng trên cửa sổ, BN train hoặc eval. Sai với LayerNorm.
- **C2 (sai số chính xác):** z^C_{ℓ+1} − z^E_{ℓ+1} = −Σ_{i∈D} c_i h_i (trước bù).
- **C3a (bù tối ưu):** trọng số ra tối ưu cho S là nghiệm LS của Z trên [H_S, 1]; sai số còn lại = ‖(I − Π_S) Z‖.
- **C3c (OBS/OBC):** g(D) = tr(C_D [(G⁻¹)_{DD}]⁻¹ C_Dᵀ); bỏ một unit i khỏi S: chi phí ‖w_i‖²/[G_S⁻¹]_{ii} (nhóm: dạng ma trận như §2 bước 4), cập nhật bằng downdate.
- **C3d (điểm):** với cột chuẩn hóa, λ_min Σ_{D} s_i² ≤ g(D) ≤ λ_max Σ_{D} s_i² ⇒ chọn theo s_i nhỏ nhất là κ(Ĝ)-xấp xỉ.
- **C3e (backward):** tối ưu với k = 1; nhiều đầu ra: tối ưu nếu σ_min(A)·min_{i∈S*}‖w*_i‖ > (1+√2)‖E*‖_F. Ngoài điều kiện này không có bảo đảm ⇒ luôn so với D_score bằng g chính xác.
- **C3f:** ΔC ≤ ‖∇_z L‖·√g + (β_z/2)·g ⇒ g nhỏ thì bước nhảy loss nhỏ; kiểm chấp nhận đo trực tiếp ΔC.
- **C4a,b (bước đầu sau expand):** gradient các tham số chung của E bằng của mạng gốc; gradient trọng số vào của unit ảo = 0 ở bước đầu (vì b = 0); L_E⁺ = L_native⁺ − η‖g_b‖² + O(η²) ⇒ chọn trọng số vào cực đại ‖g_b‖ (gradmax_pool).
- **C4c (đẳng thức):** L_CBP − L_ETT = G_E + C_reset − ΔC, với G_E = L_N − L_E (lợi do train mở rộng), C_reset = L_CBP − L_N.
- **C4d (giới hạn đã biết):** G_E ≥ 0 **sai** trong trường hợp tổng quát với h > 1 (có phản ví dụ ngay cả ở chế độ kernel tuyến tính); đúng nếu K và K_E giao hoán và ηλ ≤ 1; tổng quát G_E ≥ ηh·rᵀK_E r − 4(ηhλ)²‖r‖² khi 2ηhλ ≤ 1. ⇒ **Đây là lý do Phase 0 phải đo G_E.**
- **C5 (chi phí):** overhead ≈ (h/T)(k/n).

## Phụ lục B — Cài đặt tham chiếu (numpy, float64). Code thật (torch) phải khớp các hàm này trong T5–T7.
Quy ước: cột của A = [cột của các unit theo nhóm (1 cột/unit với MLP, 9 cột/kênh với conv), cột intercept cuối]; W = [C, c₀]ᵀ có hàng tương ứng với cột của A; G = AᵀA (đã cộng ridge trên khối unit nếu cần).
```python
import numpy as np

def g_exact(G, W, D):
    """C3c: g(D) = sum over removed index set D of tr(W_D^T [(G^-1)_DD]^-1 W_D).
    G: (p,p) Gram of A=[H_U, 1]; W: (p,d) = [C, c0]^T (current outgoing, rows = columns of A)."""
    M = np.linalg.inv(G); D = list(D)
    return float(np.trace(W[D].T @ np.linalg.solve(M[np.ix_(D, D)], W[D])))

def compensate(G, W, S):
    """C3a: LS weights on kept index set S (must include intercept index) for target Z = A W."""
    S = list(S)
    return np.linalg.solve(G[np.ix_(S, S)], G[S] @ W)          # (|S|, d)

def backward(G, W, groups, cand, k):
    """C3c/C3e: backward elimination with downdate. groups[i] = list of column indices of unit i
    (len 1 for MLP, 9 for conv). cand = removable unit ids. Returns removed unit ids."""
    M = np.linalg.inv(G); Wc = W.copy(); alive = list(range(G.shape[0])); removed = []
    cand = list(cand)
    for _ in range(k):
        pos = {c: j for j, c in enumerate(alive)}
        def cost(u):
            I = [pos[c] for c in groups[u]]
            return np.trace(Wc[I].T @ np.linalg.solve(M[np.ix_(I, I)], Wc[I]))
        u = min(cand, key=cost); I = [pos[c] for c in groups[u]]
        R = [j for j in range(len(alive)) if j not in I]
        X = np.linalg.solve(M[np.ix_(I, I)], np.eye(len(I)))
        Wc = Wc[R] - M[np.ix_(R, I)] @ X @ Wc[I]
        M = M[np.ix_(R, R)] - M[np.ix_(R, I)] @ X @ M[np.ix_(I, R)]
        alive = [alive[j] for j in R]; cand.remove(u); removed.append(u)
    return removed, Wc, alive          # Wc = compensated weights on 'alive' columns

if __name__ == "__main__":
    rng = np.random.default_rng(0); bad = 0
    for t in range(50):
        r = 1 if t % 2 else 9; m, N, d, k = 7, 400, 5, 2
        H = rng.normal(size=(N, m * r)) @ (np.eye(m*r) + .5*rng.normal(size=(m*r, m*r)))
        A = np.c_[H, np.ones(N)]; W = rng.normal(size=(m*r + 1, d)); Z = A @ W; G = A.T @ A
        groups = {u: list(range(u*r, (u+1)*r)) for u in range(m)}
        D = [1, 4]; cols = sum((groups[u] for u in D), [])
        S = [j for j in range(m*r + 1) if j not in cols]
        res = np.sum((Z - A[:, S] @ np.linalg.lstsq(A[:, S], Z, rcond=None)[0])**2)
        bad += abs(g_exact(G, W, cols) - res) > 1e-8 * max(1, res)
        bad += np.abs(compensate(G, W, S) - np.linalg.lstsq(A[:, S], Z, rcond=None)[0]).max() > 1e-8
        rem, Wc, alive = backward(G, W, groups, range(m), k)
        # naive backward
        T = list(range(m)); nrem = []
        for _ in range(k):
            def c(u):
                cs = sum((groups[v] for v in nrem + [u]), [])
                Sx = [j for j in range(m*r+1) if j not in cs]
                return np.sum((Z - A[:, Sx] @ np.linalg.lstsq(A[:, Sx], Z, rcond=None)[0])**2)
            u = min([v for v in T if v not in nrem], key=c); nrem.append(u)
        bad += rem != nrem
        bad += np.abs(Wc - compensate(G, W, alive)).max() > 1e-6
    print("violations", bad)
```
Ghi chú: hàm `backward` đã được kiểm với backward ngây thơ (giải lại lstsq cho mỗi ứng viên) trên 50 bài (MLP và nhóm 9 cột): cùng tập bỏ; trọng số bù khớp `compensate` đến sai số làm tròn (sai số tuyệt đối tăng theo cond(G), nên test dùng sai số tương đối và ridge).