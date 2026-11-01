# PROMPT SỬA LẦN 1 — ETT Phase 0 (2026-11-01)
Áp dụng lên source 81a7d04. Bổ sung cho prompt ETT chính; mục nào không nhắc ở đây thì giữ nguyên.

## 0. Kết luận review
- Core (expand, Gram, OBS/backward downdate, bù, rebuild, remap optimizer, Adam theo hàng) đúng với đặc tả. Thứ tự cột conv (unfold ↔ permute(1,2,3,0)) khớp nhau.
- **Dấu running_mean của bạn đúng, prompt sai:** khi bỏ intercept c₀ trước BN thì running_mean ← running_mean − c₀. Giữ nguyên code.
- **Kết quả Phase 0 từ 81a7d04 KHÔNG hợp lệ và không được dùng để kết luận** (gate chưa được đánh giá, nên sửa protocol lúc này là hợp lệ). Lỗi nằm ở đặc tả cửa sổ trong prompt (tôi viết "w_b = 4 batch" vì nghĩ tới minibatch; ở SCR/PM batch = 1 mẫu) và ở việc cửa sổ chỉ gồm các bước của pha mở rộng:
  - G được tính trên **1 mẫu (h = 1) hoặc 4 mẫu (h = 10, 100)**, trong khi số cột p = n + k + 1 = 12 (PM) / 7 (SCR). Phép bù LS nội suy cửa sổ: log có `g_actual_data = 0.0` nhưng ΔC trên dữ liệu mới là 0.1–0.47. Đây là overfit của phép bù, không phải giới hạn của phương pháp.
  - "Unit chết" được xác định trên 1–4 mẫu ⇒ dead_fraction 0.2–0.6 ở PM: unit thật còn sống bị coi là chết và bị bỏ "chính xác" (virtual_kept = 1.0 ở mọi chu kỳ).
  - Pool GradMax xếp hạng theo gradient của **1 mẫu**.
  - Đánh giá trên 8 mẫu: quá nhiễu cho sign test.
  - Tỉ số bậc một ở h = 1 đang so G_E đo trên 8 mẫu **tương lai** với η‖g_b‖² tính trên **batch train**: C4b chỉ phát biểu cho loss trên chính batch đó, nên phép so này không kiểm được gì.
  - Sự cố Cholesky ở SCR có cùng gốc: các đặc trưng bằng 0 trên vài mẫu.

## 1. Các sửa bắt buộc

**F1. Buffer mẫu cuộn, độc lập với chu kỳ.** Duy trì buffer gồm N_buf mẫu **đã train** gần nhất của stream, cập nhật ở mọi bước, kể cả ngoài chu kỳ. Đặc trưng vẫn được tính lại trên mạng mở rộng cuối cùng E (forward no-grad như hiện tại), nên mẫu có trước lúc expand vẫn hợp lệ.
- MLP: N_buf = max(1000, 20·p), p = Σ cột của lớp (n + k + 1). PM: 1000; SCR: 1000.
- Conv (dùng cho Phase 2): số ảnh sao cho số hàng (ảnh × vị trí) ≥ 20·p, p = 9(n + k) + 1. Ghi giá trị vào DECISIONS.md.
- Bộ nhớ: lưu input + nhãn, không lưu đặc trưng.

**F2. Tách buffer, không rò dữ liệu.** 80% mẫu cũ hơn → tính G (bước 3–5). 20% mẫu mới nhất → tập kiểm ΔC (bước 6, acceptance). Cả hai đều là mẫu đã train, nên không dùng nhãn tương lai. Bỏ tham số `check_batch` lấy từ batch kế tiếp.

**F3. Unit chết** được xác định trên phần G của buffer (≥ 800 mẫu), vẫn theo quy tắc hiện tại (energy = 0 chính xác → nhánh C1; dưới ngưỡng → ứng viên dead). Không đổi ngưỡng.

**F4. Pool GradMax:** tính δ = ∂L/∂z_{ℓ+1} và H_cand trên phần G của buffer tại thời điểm expand (một forward/backward với mạng hiện tại), không dùng 1 mẫu. Áp dụng như nhau cho GradMax-reset.

**F5. Đánh giá Phase 0:**
- L_N, L_E, L_C, L_CBP, L_GM đo trên **1000 mẫu tương lai của task hiện tại** (bắt đầu sau anchor + 100). Dịch anchor sớm hơn nếu cần để 100 + 1000 mẫu nằm trong cùng task (SCR task = 10000, PM = 60000 nên vẫn đủ). Chỉ đo, không train trên các mẫu này trước khi đo.
- **Bổ sung hậu kiểm (secondary, không thuộc gate):** sau khi đo, train tiếp **mọi nhánh** (native, ETT đã thu gọn, CBP, GM) thêm 100 bước trên cùng chuỗi mẫu, rồi đo lại trên 1000 mẫu tương lai kế tiếp. Log `*_post` và `net_post = L_N,post − L_C,post`.
- **Kiểm bậc một (h = 1):** G_E,train = L_native(b₀ sau 1 bước) − L_E(b₀ sau 1 bước), đo trên **chính batch train b₀**. So với η‖g_b‖², g_b tính trên b₀ bằng mạng sau expand với các trọng số vào đã chọn. Log tỉ số; kỳ vọng ≈ 1 khi η nhỏ (C4b). Đây là kiểm cài đặt, không phải gate.

**F6. Gate:** giữ nguyên như đã đăng ký (net ngay sau thu gọn, h ∈ {10, 100}, ≥ 20 anchor, median > 0, sign p < 0.05, median(L_CBP − L_C) > 0). Báo cáo thêm net_post, nhưng **không** dùng nó để qua gate.

**F7. Tái dùng quỹ đạo BP.** Quỹ đạo native không đổi (cùng seed, lr, data), nên được phép tái dùng các checkpoint anchor đã lưu của 81a7d04. Kiểm hash state_dict, ghi vào protocol. Toàn bộ nhánh đo phải chạy lại với source mới và output tag mới. Đánh dấu output 81a7d04 là INVALID_WINDOW (giữ file, không xóa).

**F8. I/O:** trước khi chạy, kiểm dung lượng trống ≥ 2 GB; checkpoint anchor chỉ giữ model + optimizer + GnT history (không lưu buffer lặp lại); nếu thiếu dung lượng thì dừng sớm với thông báo rõ.

## 2. Test nhắm đích (chỉ các test này; không chạy lại T1–T10 trừ khi sửa file core tương ứng)
- **T11:** buffer chứa đúng N_buf mẫu gần nhất, kể cả mẫu trước lúc expand; phần G và phần kiểm không giao nhau và đều là mẫu đã train.
- **T12:** với N_buf ≥ 20p trên một MLP tiny có ngẫu nhiên, sai số bù ngoài mẫu (trên 1000 mẫu mới cùng phân phối) ≤ 2 × sai số trong mẫu + 1e-8. Đây là kiểm sanity cho F1; nếu fail thì báo, không tự đổi N_buf.
- Nếu F4 động vào `core/expand.py`: chạy lại T1 (expand bảo toàn hàm).

## 3. Báo cáo gửi lại
Như §9 của prompt chính, thêm: N_buf thực dùng, tỉ lệ unit chết trên buffer, phân phối ΔC và g_actual_data (trong mẫu) so với ΔC trên tập kiểm, tỉ số bậc một trên b₀, net_post. Dừng sau Phase 0.