# Prompt cho agent: chạy các phương pháp đối thủ trên đúng setup của mình (4 notebook T4×2, P7–P10)

**Nguyên tắc:** dùng **code gốc** của từng phương pháp, ở đây là bản tái lập chính thức trong PyCIL (commit `f3509b8`, đã có trong `third_party/PyCIL`) và repo loss-of-plasticity (`third_party/loss-of-plasticity`). **Không sửa thuật toán của họ.** Chỉ đổi config cho khớp setup của mình. Dùng chung notebook `kaggle_vna_parallel.ipynb` (cell đầu chứa danh sách job) và chung script gộp `collect_parallel.py`.

## 1. CIL: setup dùng chung cho mọi phương pháp

- CIFAR-100, `memory_size` = 2000, seed (thứ tự lớp) = 1993, `convnet_type` = `resnet32` (hoặc biến thể resnet32 mà phương pháp đó bắt buộc, ví dụ `cosine_resnet32` cho PODNet, `memo_resnet32` cho MEMO).
- Hai cấu hình: **B0 Inc10** (`init_cls` = 10, `increment` = 10) và **B50 Inc10** (`init_cls` = 50, `increment` = 10).
- Mọi siêu tham số khác (epoch, learning rate, `fixed_memory`, loss riêng…) giữ **đúng file `exps/<method>.json` của PyCIL**. Chỉ ghi đè các trường ở trên, và ghi danh sách trường đã ghi đè vào `protocol.json`.
- **Lưu ý đã kiểm trong config gốc:**
  - `foster.json` và `beef.json` mặc định là B50 Inc10 (`init_cls` = 50, `fixed_memory` = True). Với B0 Inc10 chỉ đổi `init_cls`, giữ `fixed_memory` như gốc.
  - `tagfex.json` mặc định dùng **resnet18**. Chạy bằng resnet32 cho khớp backbone với mình, và ghi rõ việc ghi đè này.
- **Cổng tái lập** cho từng phương pháp: so với số PyCIL công bố (`resources/cifar100.png` hoặc paper gốc), dung sai ±1.5, metric theo đúng cách PyCIL báo cáo. Nếu không có số công bố cho cấu hình đó thì ghi `reference: none`.
- **Ghi thêm cho mọi run:** số tham số của model cuối (để so cùng bộ nhớ theo giao thức MEMO) và số GPU-giờ.
- **Joint (cận trên):** dùng `finetune.json` với `init_cls` = 100 và `increment` = 0, tức một task 100 lớp, cùng backbone và epoch. Nếu PyCIL không chạy được `increment` = 0 thì viết runner tối giản: train ResNet32 trên CIFAR-100 với đúng optimizer, lịch learning rate và epoch của task đầu trong `icarl.json`.

## 2. Mất plasticity (B5 regression): baseline của upstream

Chỉ dùng cfg có sẵn trong `lop/slowly_changing_regression/cfg/sgd/bp/`, giữ nguyên mọi tham số khác (ReLU, SGD, step size 0.01, 5 neuron ẩn):

| Arm | Cfg upstream | Siêu tham số |
|---|---|---|
| `b5_l2` | `relu_wd.json` | quét `weight_decay` ∈ {1e-6 … 1} như cfg |
| `b5_snp` (Shrink & Perturb) | `relu_snp.json` | `weight_decay` = 1e-2, quét `perturb_scale` ∈ {1e-6 … 1e-1} như cfg |
| `b5_redo` | `lop/algos/gntRedo.py` (bản ReDo của upstream) | threshold và `reset_period` mặc định của upstream |

**Chọn siêu tham số theo hai bước, công bằng với cách mình chọn cho phương pháp của mình:**
1. Quét trên seed 0. Chọn giá trị tốt nhất theo sai số trung bình 50 task cuối.
2. Chạy giá trị tốt nhất với seed 1 và 2.

Toàn bộ quy trình chạy tự động trong notebook; ghi giá trị đã chọn vào `protocol.json`. ReDo được wire vào vòng học supervised theo đúng cách upstream dùng `GnTredo`. Nếu upstream không có đường cho supervised thì viết hook tối giản, không đổi logic của `GnTredo`, và có test.

## 3. Phân job (P7–P10)

Mỗi GPU chạy job theo thứ tự; job sau chạy khi job trước xong, có resume qua nhiều phiên.

| Notebook | GPU0 (theo thứ tự) | GPU1 (theo thứ tự) | CPU |
|---|---|---|---|
| P7 | `finetune__inc10` → `joint` → `finetune__b50inc10` | `wa__inc10` → `wa__b50inc10` | `b5_l2` (quét + 3 seed) |
| P8 | `podnet__inc10` → `podnet__b50inc10` | `bic__inc10` → `bic__b50inc10` | `b5_snp` (quét + 3 seed) |
| P9 | `foster__inc10` → `foster__b50inc10` | `der__inc10` → `der__b50inc10` | `b5_redo` (3 seed) |
| P10 | `memo__inc10` → `memo__b50inc10` | `tagfex__inc10` → `tagfex__b50inc10` | (trống) |

- Tất cả seed 1993. Thêm seed chỉ chạy ở đợt bảng chính.
- DER, MEMO, TagFex mở rộng model theo task nên chạy lâu hơn. Ưu tiên xong `__inc10` trước; `__b50inc10` chạy tiếp ở phiên sau nếu cần.

## 4. Gộp kết quả

Mở rộng `collect_parallel.py`:
- **Bảng CIL**, mỗi cấu hình một bảng. Mỗi phương pháp một dòng, gồm: average incremental accuracy (CNN, NME), last accuracy, forgetting, số tham số cuối, GPU-giờ, trạng thái cổng tái lập.
  - Đặt chung với `icarl` và `icarl_vna` từ P1–P6.
  - Đánh dấu nhóm **cố định kích thước** (finetune, iCaRL, WA, PODNet, BiC, FOSTER sau khi nén) và nhóm **mở rộng model** (DER, MEMO, TagFex).
- **Bảng B5:** BP, CBP, L2, S&P, ReDo, cùng các arm GIR/VNA từ P1–P6. Trung bình ± độ lệch chuẩn qua 3 seed của sai số 50 task cuối, và chênh lệch ghép cặp so với CBP.

## 5. Test và báo cáo trước khi chạy

- Với mỗi phương pháp PyCIL: một test chạy 1 epoch trên dữ liệu giả, xác nhận config sau khi ghi đè đúng như mục 1, và `protocol.json` liệt kê đúng các trường đã ghi đè.
- Test ReDo: neuron dưới ngưỡng bị reset đúng logic `GnTredo`.
- Gửi về: diff, log pytest, và 4 cell đầu đã điền sẵn cho P7–P10.
