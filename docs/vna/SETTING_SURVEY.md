# Survey: bài toán và setup hợp với VNA nhất (2026-10-14)

## Tiêu chí, suy ra từ công thức

ρ ≈ (g∥ᵀ K_E g∥) / (gᵀ K g). Bài toán phù hợp khi có:

1. **Residual kéo dài**: mục tiêu khó, hoặc thay đổi liên tục, nên model không bao giờ khớp hết.
2. **Model nhỏ hoặc hẹp**: vài neuron ảo đã chiếm phần đáng kể của kernel.
3. **Residual dồn vào các hướng "chậm"** của K, nơi K yếu, và neuron ảo bù được đúng những hướng đó.
4. **Ngân sách bước train có hạn**, vì VNA chủ yếu thay đổi tốc độ và quỹ đạo.
5. **Kiến trúc bị ràng buộc cố định**, và cộng đồng đó đã có sẵn các phương pháp "grow" để đặt cạnh làm so sánh.
6. Chạy được trên Kaggle T4, và có tiền lệ paper ở hội nghị A\*.

## Xếp hạng

| # | Bài toán | 1 | 2 | 3 | 4 | 5: đối thủ grow | Chi phí | A\* | Đánh giá |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Mất plasticity / continual learning không dừng** | ✓✓ (mỗi task mới tạo residual mới) | ✓ (MLP/CNN nhỏ) | ✓✓ (xem dưới) | ✓ (mỗi task có số bước cố định) | Plasticity injection (NeurIPS 2023), Neuroplastic Expansion (ICLR 2025) | Rất rẻ đến trung bình | Nature 2024, ICML 2026, ICLR 2026 | **Khớp nhất** |
| 2 | **PINN / scientific ML** | ✓✓ (residual PDE hiếm khi về 0) | ✓✓ (MLP rộng 50–400) | ✓ (ill-conditioning do toán tử vi phân) | ✓ | Adaptive Neuron Enhancement (Cai), Galerkin NN (Ainsworth–Dong), adaptive-growth RaNN | Rất rẻ | ICML 2024 (Rathore), ICLR 2024 (De Ryck) | Khớp lý thuyết nhất (Dirac–Frenkel / Neural Galerkin); cộng đồng khác |
| 3 | KD: khoảng cách capacity | ✓✓ | ✓ | ? | ✓ | Teacher-assistant (TAKD) | Trung bình | ICCV/AAAI | Tốt |
| 4 | Class-incremental (iCaRL, PyCIL) | ✓ | ✓ (ResNet32) | ? | ✓ | DER, FOSTER, MEMO | Trung bình | CVPR/ICLR | Tốt, nhưng thêm vấn đề quên |
| 5 | Model tí hon (NetAug) | ✓ | ✓ | ? | — | NetAug (augmented, không grow) | Cao nếu ImageNet | ICLR 2022 | Ổn |
| 6 | LoRA / PEFT | ✓ | ✓ (rank r) | ? | ✓ | Tăng rank, AdaLoRA | Trung bình | — | **Bị đi trước một phần**: LoRA-Pro (ICLR 2025) đã chiếu gradient full fine-tune lên cấu trúc low-rank |
| 7 | Mất plasticity trong RL | ✓✓ | ✓ | ✓ | ✓ | Plasticity injection, NE | Cao, nhiễu | ICML/NeurIPS | Để sau |
| — | ResNet-18 chuẩn trên CIFAR | ✗ | ✗ | ✗ | — | — | — | — | Phase 0 đã loại |

## Vì sao mất plasticity khớp nhất

- **Cơ chế trùng với lý thuyết của VNA.** *Spectral Collapse Drives Loss of Plasticity* (ICML 2026, arXiv 2509.22335) chỉ ra:
  - Mất plasticity xảy ra khi phổ Hessian/GGN sụp đổ ở đầu task mới. Họ chứng minh GGN có cùng các trị riêng khác 0 với Gram matrix của NTK, nên đây cũng là sự sụp đổ của NTK.
  - Residual của task mới rơi vào các hướng chậm, có trị riêng nhỏ, nên không giảm được trong ngân sách K bước.
  - **VNA cộng thêm K_E đúng vào những hướng đó mà không thêm tham số.** Đây là lời giải thẳng cho cơ chế mà paper đó mô tả. Họ dùng regularizer (L2 kết hợp effective rank); VNA can thiệp trực tiếp vào kernel.
- **Có sẵn đối thủ grow.** Plasticity injection (Nikishin et al., NeurIPS 2023) thêm head mới để khôi phục plasticity, nên tổng số tham số tăng. Neuroplastic Expansion (ICLR 2025) grow neuron trong RL. Câu chuyện "lợi ích của việc grow mà không grow" vì vậy rất tự nhiên ở đây.
- **Có sẵn baseline không grow:** L2, Shrink & Perturb, continual backprop (CBP), ReDo, L2-Init, L2-ER, SNR, isometry/AdamO (2606.09762).
- **Benchmark chuẩn, phần lớn rẻ** (Dohare et al., Nature 2024; code công khai):
  - Online Permuted MNIST: MLP 3 lớp ẩn.
  - Slowly-changing regression: mạng 1 lớp ẩn, 5 unit; cực rẻ, rất hẹp.
  - Continual ImageNet 32×32: các task nhị phân, CNN nhỏ.
  - Incremental CIFAR-100: ResNet-18, mỗi lần thêm 5 lớp, 200 epoch mỗi lần; đắt.
- **Novelty:** survey chưa tìm thấy paper nào dùng virtual neurons hoặc kernel augmentation làm preconditioner để chữa mất plasticity (độ tin cậy trung bình). Các hướng hiện có là reset, regularizer, normalization, grow và optimizer (AdamO).

## Vì sao PINN là lựa chọn thứ hai

- Khung Dirac–Frenkel / Neural Galerkin, vốn là lý thuyết chính của VNA, **ra đời từ chính cộng đồng này**.
- Loss là bình phương tối thiểu của residual PDE, nên H = 2I/n và K = JJᵀ chính xác, không cần GGN.
- **Đối chứng NGD ở đây chính là ENGD** (energy natural gradient, Müller–Zeinhofer, ICML 2023), một baseline mạnh và được công nhận. VNA được viết dưới dạng ENGD cộng kernel của neuron ảo.
- Model rất nhỏ, nên solver exact có thể chạy mọi bước.
- Rathore et al. (ICML 2024) có benchmark convection/reaction/wave và so Adam, L-BFGS, Adam+L-BFGS và NNCG. Họ còn quét độ rộng MLP, nên đường cong gain theo độ rộng gần như có sẵn (agent cần kiểm tra lại trong code gốc).
- **Rủi ro:** reviewer ML có thể coi đây là bài toán ngách. Tuy vậy, paper PINN optimization vẫn vào được ICML/ICLR. Nếu không, có thể gửi JCP hoặc CMAME (Q1).

## Đề xuất đổi giai đoạn B

Ưu tiên mới, rẻ trước, khớp nhất trước:

1. **B5: plasticity.** Slowly-changing regression, rồi Permuted MNIST (200 task đầu), rồi Continual ImageNet-32 (200 task đầu). Đo ρ ở đầu và giữa mỗi task, so task sớm với task muộn. Kỳ vọng: ρ tăng ở các task muộn, khi K sụp.
2. **B6: PINN.** Convection, reaction và wave theo code của Rathore et al., với 2–3 mức độ rộng.
3. B3: CIL (iCaRL).
4. B2: KD.
5. B4 và B1: model hẹp/tí hon.

Cổng G giữ nguyên tinh thần. Với plasticity, điều kiện ρ ≥ 0.3 được tính trên các task trong nửa sau của chuỗi.

## Tác động lên câu chuyện paper

- **Câu chuyện chính, nếu B5 qua cổng:** *"Mất plasticity là sự sụp đổ kernel của model cố định. VNA khôi phục kernel bằng neuron ảo được tiêu chí grow chọn ra, cho lợi ích của plasticity injection hay grow mà không thêm tham số nào."*
- Lý thuyết Dirac–Frenkel vẫn giữ nguyên. Lemma "tín hiệu tắt khi khớp hết dữ liệu" giải thích vì sao ResNet-18 trên CIFAR không được lợi.
- Nếu PINN cũng qua cổng, nó là bối cảnh thứ hai để chứng minh tính tổng quát.

## Phải trích dẫn và phân biệt

- **LoRA-Pro:** ý tưởng "chiếu gradient của model lớn hơn (full fine-tune) lên cấu trúc bị ràng buộc" đã có ở dạng tham số. VNA khác ở ba điểm: dùng neuron ảo phi tuyến do tiêu chí grow chọn, làm việc trong không gian hàm (NTK), và áp cho kiến trúc cố định bất kỳ.
- Plasticity injection, Neuroplastic Expansion, Spectral Collapse (L2-ER), AdamO/isometry, CBP, ReDo.
- ENGD, NNCG và operator preconditioning (De Ryck et al., ICLR 2024).

## Nguồn

- Dohare et al., Loss of plasticity in deep continual learning (Nature 2024), arXiv 2306.13812
- Spectral Collapse Drives Loss of Plasticity (ICML 2026), arXiv 2509.22335
- Nikishin et al., Deep RL with Plasticity Injection (NeurIPS 2023), arXiv 2305.15555
- Liu et al., Neuroplastic Expansion in Deep RL (ICLR 2025)
- Preserving Plasticity via Dynamical Isometry, arXiv 2606.09762
- Farias & Jozefiak, Self-Normalized Resets, arXiv 2410.20098
- Rathore et al., Challenges in Training PINNs (ICML 2024)
- De Ryck et al., Operator preconditioning perspective (ICLR 2024)
- LoRA-Pro (ICLR 2025), arXiv 2407.18242
- Cai et al., Adaptive Neuron Enhancement; Ainsworth & Dong, Galerkin NN
- Cho & Hariharan (ICCV 2019); Mirzadeh et al., TAKD (AAAI 2020)

## Kiểm tra novelty sâu trong hai lĩnh vực (2026-10-14, lượt 2)

### Continual learning / mất plasticity — **chưa thấy ai dùng neuron ảo** (độ tin cậy trung bình–cao)

Các nhóm phương pháp đang có:

- **Chẩn đoán cơ chế:**
  - Lewandowski et al., *Directions of Curvature* (2312.00246): curvature bị mất, đo bằng srank của Hessian; chỉ dùng để chẩn đoán, chữa bằng regularizer Wasserstein.
  - *Spectral Collapse* (ICML 2026): GGN/NTK sụp; chữa bằng L2-ER.
  - *InterpLayers* (ICML 2026): kiến trúc mới tránh NTK rank collapse.
- **Optimizer / preconditioner — gần VNA nhất về hình thức:**
  - ARROW (*Vision Transformers That Never Stop Learning*, 2603.07787): (αI + βC)⁻¹, với C là hiệp phương sai gradient low-rank; khuếch đại các hướng curvature thấp.
  - OCAR (Urettini & Carta, 2502.01866): Fisher thật dạng K-FAC làm preconditioner cho online continual learning.
  - TRAC; AdamO/isometry (2606.09762).
  → Cả nhóm này **chỉ biến đổi kernel/curvature sẵn có của model**. Không phương pháp nào đưa kernel của neuron chưa tồn tại vào.
- **Grow:** plasticity injection (NeurIPS 2023), Neuroplastic Expansion (ICLR 2025), *Plasticity of Growing and Elastic NNs* (2608.01475; thêm unit ngẫu nhiên). Tất cả đều cài neuron thật.
- **Reset / regularizer:** CBP, ReDo, Shrink & Perturb, L2-Init, SNR, Wasserstein-init.

**Khoảng trống:** chưa có phương pháp nào dùng tiêu chí grow để chọn neuron rồi đưa kernel của chúng vào update mà không cài vào model.

### PINN / scientific ML — **chưa thấy neuron ảo, nhưng rủi ro bị scoop cao hơn**

- **Natural gradient / Gauss–Newton, rất mạnh:**
  - ENGD (ICML 2023);
  - K-FAC cho PINN (Dangel et al., NeurIPS 2024);
  - Dual natural gradient (2505.21404);
  - SOAP / gradient alignment (2502.00604);
  - NNCG (Rathore et al., ICML 2024);
  - operator preconditioning (ICLR 2024).
- **Neural Galerkin / Dirac–Frenkel:** TENG (ICML 2024), RSNG (cập nhật thưa ngẫu nhiên). Kiến trúc cố định, không có neuron ảo.
- **Grow, cài neuron thật:**
  - **Orthogonal Greedy Algorithm** (Siegel, Hong, Jin, Hao, **Xu**; JCP 2023): chọn neuron mới bằng cách tối đa hoá tương quan với residual, gần như TINY cho PDE;
  - ANE (Cai);
  - Galerkin NN (Ainsworth–Dong);
  - adaptive-growth RaNN (2408.17225).
- **Rủi ro:** nhóm Xu có cả hai mảnh, chọn neuron theo residual (OGA) và lý thuyết auxiliary space (Park & Xu 2509.07179). Ghép hai mảnh thành "neuron ảo làm preconditioner" là bước tự nhiên tiếp theo của họ.

### Lưu ý kỹ thuật quan trọng cho cả hai lĩnh vực (phải nói thẳng trong paper)

- Vận tốc hàm số của VNA là J·d = K(I + R K_E R K)g, **vẫn nằm trong range(K)**. VNA không tạo ra hướng hàm số mới; nó chỉ **chọn lại cách phân bổ** giữa các hướng mà K có, theo trọng số mà model đã grow sẽ dùng.
- Hệ quả:
  1. Nếu kernel sụp **hẳn** (trị riêng bằng 0), VNA không chữa được, cũng như mọi preconditioner. Chỉ cài neuron thật mới chữa được.
  2. Nếu kernel sụp **mềm** (trị riêng nhỏ nhưng khác 0), NGD, ARROW và OCAR cũng khuếch đại các hướng đó. Lợi thế khả dĩ của VNA là **khuếch đại có chọn lọc**: chỉ những hướng mà neuron ảo do tiêu chí grow chọn sẽ dùng, thay vì mọi hướng nhỏ, vốn gồm cả nhiễu.
- Vì vậy **đối chứng quyết định** ở cả hai lĩnh vực là VNA so với NGD cùng norm (tương ứng ENGD hoặc K-FAC-PINN, và OCAR/ARROW). Thắng NGD thì câu chuyện đứng; không thắng thì VNA chỉ là một preconditioner.

### Đánh giá tiềm năng A\*

| | CL / mất plasticity | PINN |
|---|---|---|
| Novelty của VNA | Khoảng trống rõ ràng | Khoảng trống có, nhưng bị nhóm Xu áp sát |
| Độ khớp lý thuyết | Cao (kernel sụp) | Rất cao (Dirac–Frenkel / Galerkin) |
| Baseline phải thắng | NGD/OCAR/ARROW-style, plasticity injection (cùng 0 tham số thêm), CBP, L2-Init | ENGD, K-FAC-PINN, NNCG, Adam+L-BFGS |
| Độ khó để thắng | Trung bình | Cao (ENGD/K-FAC rất mạnh, đua tới độ chính xác máy) |
| Chi phí | Thấp | Rất thấp |
| Tiềm năng A\* | **Có**, nếu thắng NGD và đạt ngang plasticity injection mà không thêm tham số | Có, nhưng khó hơn; Q1 (JCP/CMAME) chắc hơn |

**Đề xuất:** dùng CL / mất plasticity làm câu chuyện chính, PINN làm bối cảnh thứ hai để chứng minh tính tổng quát. Đưa preprint lên arXiv sớm khi giai đoạn C ra kết quả dương.

## Lượt 3: reset neuron chết, venue của các paper, và tiềm năng A\* của nhánh CL (2026-10-14)

### Các paper liên quan và venue (đã kiểm tra hoặc ghi rõ chưa kiểm tra)

| Paper | Venue | Làm gì | Reset thành cái gì |
|---|---|---|---|
| Dohare et al., CBP | **Nature 2024** | Reset unit có utility thấp | Ngẫu nhiên; trọng số ra = 0 |
| Sokar et al., ReDo | **ICML 2023** | Reset unit "ngủ" (dormant) | Ngẫu nhiên; trọng số ra = 0 |
| Lyle et al., Understanding Plasticity | **ICML 2023** | Phân tích | — |
| Nikishin et al., Plasticity Injection | **NeurIPS 2023** | Grow head (thêm bản copy) | — |
| Neuroplastic Expansion | **ICLR 2025** | Grow neuron (RL) | — |
| FIRE (2602.08040) | **ICLR 2026, oral** | Trực giao hoá trọng số giữa các task | Không reset theo unit |
| Spectral Collapse (2509.22335) | **ICML 2026** | GGN/NTK sụp; chữa bằng L2-ER | — |
| InterpLayers | **ICML 2026** | Kiến trúc chống NTK rank collapse (RL) | — |
| Hernandez-Garcia, Dohare et al., SWR (2508.00212) | CoLLAs 2025 (không phải A\*) | Reset **từng trọng số** có utility thấp | Lấy mẫu lại từ phân phối khởi tạo, hoặc đặt bằng trung bình |
| ReGraMa (2505.24061) | Chưa xác nhận (có checklist NeurIPS) | Chọn unit theo độ lớn gradient (RL) | Phân phối khởi tạo; trọng số ra = 0 |
| SNR (2410.20098) | Chưa xác nhận | Reset theo kiểm định thời gian không kích hoạt | Reset |
| GXD (2605.06834) | Chưa xác nhận | Chọn unit theo attribution | Dùng cơ chế reset chung |
| Lewandowski et al., Curvature (2312.00246) | Chưa xác nhận | Curvature bị mất; regularizer Wasserstein | — |
| OCAR (2502.01866) | Chưa xác nhận (có thể CoLLAs) | Preconditioner K-FAC cho online CL | — |
| ARROW (2603.07787) | arXiv | Preconditioner dùng hiệp phương sai gradient | — |
| Luận văn TU Delft (Urbonavičiūtė) | Luận văn cử nhân | Thay reset trong CBP bằng Kaiming, nhiễu, hoặc rescale | Ngẫu nhiên hoặc rescale |

**Kết luận về reset:** tất cả các phương pháp reset đã kiểm tra, kể cả SWR và luận văn TU Delft, đều gán trọng số mới là **ngẫu nhiên, theo phân phối khởi tạo, hoặc bằng 0**. Chưa có phương pháp nào gán trọng số **theo residual hay tiêu chí grow**. Khoảng trống "growth-informed reset" (GIR) đứng vững, độ tin cậy trung bình–cao.

### Gói đóng góp đề xuất cho nhánh CL

1. **Lý thuyết.**
   - VNA là phép chiếu Dirac–Frenkel của động lực model đã grow lên model cố định.
   - Liên hệ với kernel sụp: VNA phân bổ lại tốc độ trong range(K) theo K_E. Mệnh đề giới hạn: VNA không vượt ra ngoài range(K), nên cần GIR để tạo hướng mới.
   - Lemma tín hiệu tắt khi khớp hết dữ liệu.
2. **VNA (preconditioner):** xử lý trường hợp kernel sụp mềm.
3. **GIR:** thay unit chết hoặc có utility thấp bằng neuron ảo; A chọn bởi TINY, B = 0, nên hàm số được giữ nguyên. Đây là cải tiến trực tiếp của CBP/ReDo ở bước "reset thành cái gì".
4. **Kích thước cố định:** so cùng hạng với CBP, ReDo, SWR, FIRE, và so cùng ngân sách tham số với plasticity injection.

### Baseline bắt buộc nếu nhắm A\*

- **Reset:** CBP, ReDo, SWR, cùng tiêu chí chọn unit nhưng reset ngẫu nhiên. Ablation quan trọng nhất là cô lập phần đóng góp của bước "reset thành gì".
- **Regularizer:** L2, L2-Init, Shrink & Perturb, L2-ER.
- **Optimizer/curvature:** NGD cùng norm (đại diện cho nhóm OCAR/ARROW), FIRE.
- **Grow:** plasticity injection. So theo hai cách: cùng số tham số huấn luyện, và cùng tổng số tham số.
- **Benchmark:**
  - Permuted MNIST, gồm cả biến thể mạng nhỏ (10 unit), nơi CBP và ReDo thất bại theo SWR. Đây là bối cảnh hẹp, rất hợp với VNA và GIR.
  - Slowly-changing regression; Continual ImageNet-32; incremental CIFAR-100 (ResNet-18 hoặc ViT).
  - **Ít nhất một bài RL** (ví dụ Slippery Ant), vì các paper A\* gần đây trong mảng này đều có.

### Đánh giá tiềm năng

- Chủ đề đang nóng, các paper chính đều ở A\* (Nature, ICML, ICLR, NeurIPS), nên **trần là A\***. Ngược lại, mặt bằng cạnh tranh rất dày, và FIRE (ICLR 2026 oral) đặt ra mức chuẩn cao.
- **Điều kiện để lên A\*:**
  1. GIR tốt hơn reset ngẫu nhiên với cùng tiêu chí chọn unit, trên nhiều benchmark;
  2. VNA hoặc VNA + GIR tốt hơn NGD cùng norm;
  3. ngang hoặc hơn plasticity injection mà không thêm tham số;
  4. có phần lý thuyết kèm hình cơ chế (phổ K theo task, phần residual nằm trong các hướng chậm).
- **Nếu chỉ đạt (1), hoặc chỉ có benchmark supervised:** Q1 (TNNLS, Neural Networks) hoặc CoLLAs là thực tế.
- **Lưu ý:** GIR có thể trở thành đóng góp chạy được và dễ hiểu nhất của paper, kể cả khi VNA preconditioner chỉ thắng NGD một chút. Nhờ vậy, câu chuyện "dùng tín hiệu grow cho model không grow" vẫn đứng vững.
