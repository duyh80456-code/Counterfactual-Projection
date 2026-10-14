# Kế hoạch chạy thí nghiệm VNA (bản để public, 2026-10-14)

Mục tiêu: paper Q1, hoặc A\* nếu kết quả mạnh. Nguyên tắc:

- Chỉ so những gì các paper cùng hướng thường so.
- Cổng quyết định được đặt trước khi chạy.
- Baseline phải khớp con số đã công bố.

Chi phí là ước tính theo GPU-giờ T4, trừ khi có ghi chú là đã đo.

## 0. Bối cảnh được chọn và lý do

VNA chỉ có tác dụng khi đồng thời có:

1. residual kéo dài, tức model không khớp hết dữ liệu train;
2. layer hẹp, nên số neuron ảo chiếm tỉ lệ đáng kể so với neuron thật;
3. nút thắt có cấu trúc ổn định;
4. ngân sách train giới hạn.

Phase 0 trên ResNet-18 / CIFAR-100 không thoả (ρ ≈ 0.03, tín hiệu tắt sau epoch 150).

Ba bối cảnh ứng viên dưới đây đều dùng **giao thức chuẩn có sẵn**, nên baseline đối chiếu được với số đã công bố.

| Bối cảnh | Giao thức chuẩn | Model | Baseline đã công bố |
|---|---|---|---|
| **S1. Model tí hon** | Recipe CIFAR-100 của CRD/DOT: 240 epoch, LR giảm ×0.1 ở epoch 150/180/210, batch 64, SGD 0.9, WD 5e-4, chạy 3 lần | ResNet8×4, ShuffleNetV2, MobileNetV2 | ResNet8×4 72.50; ShuffleNetV2 71.82; VGG8 70.36 (DOT) |
| **S2. KD** | Cùng recipe, teacher ResNet32×4 (checkpoint công khai của CRD) | Các student giống S1 | KD và DOT (ICCV 2023) trên cùng các cặp |
| **S3. CIL (class-incremental)** | PyCIL iCaRL: CIFAR-100 B0 Inc10 / B50 Inc10, 2000 exemplar, seed 1993 cho thứ tự lớp; task đầu 200 epoch, các task sau 170 epoch, batch 128; ImageNet-100 B50 Inc5. Lưu ý: "CCIL" trong repo One-Shot-TAS-CCIL là pipeline tăng capacity (grow), không phải class-incremental | ResNet32 (CIFAR), ResNet18 (ImageNet-100) | iCaRL, WA, PODNet; DER, FOSTER, MEMO theo bộ nhớ ngang nhau (MEMO, ICLR 2023) |

S1 và S2 dùng chung student, nên một lần sàng lọc phục vụ được cả hai.

---

## Giai đoạn A. Chẩn đoán phương pháp trên checkpoint Phase 0 đã có (khoảng 2 GPU-giờ, một phiên)

| # | Việc | Đầu ra | Điều kiện đạt |
|---|---|---|---|
| A1 | Đối chứng nhãn xáo trộn có cùng độ lớn residual: chuẩn hoá λ theo ‖g‖², hoặc hoán vị residual giữa các mẫu | Kết quả G1 đã sửa, theo epoch | Ghi nhận; dùng làm thước cho B |
| A2 | Quét ρ theo μ_rel ∈ {1e-2, 1e-3, 1e-4} ở epoch 100/120/140/160 | ρ(μ) | Nếu μ nhỏ làm ρ tăng ≥ 3 lần → hạ μ mặc định |
| A3 | So tiny_U, tiny_V, greedy_V, A ngẫu nhiên (cùng trace) ở epoch 40/80/120/140 | V của từng cách chọn | Cần tiny_V > random; nếu không, xem lại bước chọn |
| A4 | Gộp K_E của tất cả 8 block, giải chung một lần | ρ_joint | Ghi nhận |
| A5 | Bug V_TINY/V_random = 0 ở epoch 100 | Bản sửa code | Bắt buộc sửa |
| A6 | Kiểm tra K-FAC so với exact: cosine, chi phí | Bảng solver | Cosine ≥ 0.9 và overhead ≤ 1 |

## Giai đoạn B. Sàng lọc bối cảnh, chỉ đo ρ, không chạy VNA (khoảng 6–8 GPU-giờ)

| # | Chạy | Chi phí |
|---|---|---|
| B1 | S1: ResNet8×4 và ShuffleNetV2 tự train, 240 epoch, đo ρ mỗi 20 epoch | khoảng 1.5 × 2 |
| B2 | S2: ResNet32×4 → ResNet8×4 với KD, đo ρ trên residual KD (so với soft target) | khoảng 1.5 |
| B3 | S3: ResNet32 trên CIFAR-100 B0 Inc10 với iCaRL (PyCIL), đo ρ ở đầu, giữa và cuối mỗi task | khoảng 2 |
| B4 | ResNet-18 độ rộng 0.25 (dùng lại `dual_growth` trong repo tham chiếu); sau này so với TINY grow thật (`tiny_figure5`) ở D3 | khoảng 1 |

Mọi lần chạy đều ghi train accuracy cuối, để xác nhận model không khớp hết dữ liệu.

**Cổng G (đặt trước khi chạy).** Một bối cảnh qua cổng khi đạt cả ba điều kiện:

- ρ ≥ 0.3 trong ít nhất 50% số điểm đo trước khi LR giảm lần cuối;
- vượt đối chứng A1;
- tiny_V tốt hơn random.

Nếu không bối cảnh nào qua cổng, dừng và xem lại hướng đi. Không chạy C.

## Giai đoạn C. Pilot trên bối cảnh qua cổng G (seed 0, khoảng 5–15 GPU-giờ)

Các arm:

- `sgd` (hoặc phương pháp nền: KD, iCaRL);
- `+vna_tiny_V`;
- `+ngd` (cùng norm, cùng solver);
- `+vna_random`;
- `+vna_tiny_U`.

**Đạt khi:** VNA tốt hơn baseline ≥ 0.5pp **và** tốt hơn ngd ≥ 0.3pp. Nếu chỉ hơn baseline thì lợi ích đến từ preconditioner, không phải từ neuron ảo.

## Giai đoạn D. Gói paper (chỉ làm khi C đạt)

**D1. Bảng chính.** 5 seed, ghép cặp theo seed, báo cáo trung bình ± khoảng tin cậy 95%, kiểm định paired t-test kèm hiệu chỉnh Holm.

- S1: 3 student × {SGD, NetAug, VNA}. NetAug là baseline trực tiếp vì cũng nhắm "model tí hon được lợi từ model lớn hơn"; code đã công khai.
- S2: 2–3 cặp teacher-student × {KD, KD+DOT, KD+VNA, DKD+VNA}. DOT cũng can thiệp ở mức optimizer nên là đối chứng sát nhất.
- S3: {iCaRL, WA, PODNet} ± VNA trên B0 Inc10 và B50 Inc10. Thêm đường cong accuracy theo bộ nhớ, so với DER, FOSTER, MEMO (giao thức MEMO).

Mỗi bảng có cột tham số, FLOP lúc train và GPU-giờ. Ở CIL, model size dùng cho suy luận của VNA bằng baseline.

**D2. Ablation** (3 seed, trên một bối cảnh chính):

- cách chọn A: tiny_V, tiny_U, random, GradMax-A;
- rank ∈ {4, 16, 64};
- μ;
- tần suất correction;
- số site: 1, 3, gộp tất cả;
- solver: exact, K-FAC, lite;
- đối chứng Geifman: S = h(K), cùng phổ;
- đối chứng nhánh tuyến tính kiểu RepOpt/Bypass.

**D3. Cơ chế, hình quan trọng nhất:**

1. **Gain theo độ rộng:** ResNet ở 1/8, 1/4, 1/2, 1 trên CIFAR-100. Gain đo được phải đi cùng ρ từ Phase 0, tức ρ dự báo được khi nào VNA có ích.
2. **VNA và grow thật:** so cosine vận tốc hàm số trên held-out giữa VNA và model được TINY grow thật; khoảng cách quỹ đạo.
3. **ρ theo thời gian / task:** minh hoạ lemma tín hiệu tắt khi khớp hết dữ liệu (ResNet-18 làm ví dụ âm).
4. **Khoảng cách off-batch (Nyström):** kernel tại điểm test của VNA so với K_E thật sau khi cài neuron.

**D4. Quy mô lớn hơn:**

- Tiny-ImageNet hoặc ImageNet-100 cho bối cảnh chính. Ví dụ ImageNet-100 B50 Inc5 cho CIL, hoặc MobileNetV2 ×0.35 ở 128px cho S1.
- ImageNet-1k (MobileNetV2-Tiny, giống NetAug: baseline 51.7 → NetAug 53.3 với 150 epoch) **chỉ khi có thêm tài nguyên**. Đây là điều kiện thực tế để lên A\*.

## Ước tính tổng chi phí

| Giai đoạn | GPU-giờ T4 |
|---|---|
| A | khoảng 2 |
| B | khoảng 6–8 |
| C | khoảng 5–15 |
| D1 (một bối cảnh chính, 5 seed) | khoảng 40–80 |
| D2 + D3 | khoảng 30–50 |
| D4: Tiny-ImageNet / ImageNet-100 | khoảng 40–100 |
| **Tổng cho gói Q1** | **khoảng 120–250**, tức 4–8 tuần quota Kaggle |
| ImageNet-1k (A\*) | vài trăm GPU-giờ cho mỗi lần chạy; cần tài nguyên ngoài |

## Những gì paper sẽ trình bày

1. **Lý thuyết:**
   - VNA là phép chiếu Dirac–Frenkel của quỹ đạo model đã grow;
   - cận sai số theo projection defect;
   - lemma tín hiệu tắt khi khớp hết dữ liệu;
   - ρ là thước dự báo.
2. **Bảng chính:** S1/S2/S3, 5 seed, kèm cột chi phí.
3. **Hình cơ chế:** gain–ρ–độ rộng; VNA so với grow thật.
4. **Ablation và giới hạn:** ResNet-18 chuẩn (không có lợi, như lý thuyết dự đoán); chi phí.

## Rủi ro

- Không bối cảnh nào qua cổng G. Khi đó dừng, có thể viết một paper ngắn về lemma tín hiệu tắt cùng ρ làm chẩn đoán.
- Ở CIL, VNA làm quên nhiều hơn. Cách xử lý: tính residual trên cả buffer, đo forgetting riêng.
- TINY-A không hơn random. Khi đó câu chuyện chuyển từ "growth criterion" sang "virtual kernel", và novelty giảm.
- Bị scoop bởi nhóm Inria TAU (CAGE-NAS) hoặc nhóm Xu. Nếu C dương, đưa preprint lên arXiv sớm.
EOF