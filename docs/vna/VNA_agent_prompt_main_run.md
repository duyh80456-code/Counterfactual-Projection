# Prompt cho agent: cài bản thuật toán T26–T28, dọn code thừa, chạy so sánh trên setup chính

**Base:** HEAD hiện tại của branch `vna-v3-pilot`. Làm trên branch mới `vna-v4-main`.

**Tài liệu kèm theo** (copy vào `docs/vna/`):
- `vna_gir_theory.pdf` cùng source LaTeX (`theory/*.tex`): tài liệu có thẩm quyền.
  - Mục 9.1: định lý T25.
  - Mục 10: T26 (trust region), T27 (GIR có bảo vệ), T28 (swap theo lợi ròng).
- `check_t25.py`, `check_t26_28.py`: các phép kiểm số tham chiếu.
- `survey_competitors.md`: danh sách đối thủ, venue và repo đã xác minh.

Khi prompt này và tài liệu khác nhau, **tài liệu thắng**. Ghi lại mọi chỗ lệch.

---

## 0. Nguyên tắc bắt buộc

1. **Không đổi công thức.** Mọi đại lượng tính đúng như trong tài liệu. Nếu thấy phải đổi, dừng lại và báo; không tự sửa.
2. **Đối thủ dùng code gốc.**
   - Ưu tiên `git clone` repo chính thức, pin commit vào `third_party/`, rồi gọi hàm của họ qua wrapper mỏng.
   - Chỉ đổi config. Không viết lại thuật toán của họ.
   - Chỉ khi không có code chính thức mới được cài lại theo paper. Khi đó đánh dấu `reimplemented=true` trong `protocol.json` và ghi rõ phương trình, mục của paper đã theo.
3. **Mọi arm dùng chung setup ở §3**: cùng backbone, cùng thứ tự lớp, cùng số epoch, cùng memory, cùng seed. Chỉ khác phần phương pháp.
4. **Không có `assert` nào được làm sập một run dài.** Lỗi kiểm lúc chạy thì ghi cờ vào log và đi tiếp.
5. **Không chạy pytest toàn bộ, không chạy smoke notebook riêng.** Chỉ chạy các test ngắn có mục tiêu ở §7, trên local, mô hình nhỏ.
6. **Mọi thay đổi** so với spec cũ ghi vào `docs/vna/PREREGISTRATION_CHANGES.md` (mục §8) trước khi bấm chạy, kèm thời điểm.
7. **Mọi notebook pin cùng một commit.** Commit hash ghi trong `protocol.json` của từng run và phải giống nhau giữa các notebook của cùng một đợt.

---

## 1. Dọn code, notebook, output và tài liệu thừa (làm đầu tiên)

Mục tiêu: sau bước này, repo **chỉ còn** những gì cần cho thuật toán T26–T28 và cho các notebook ở §6. Không giữ cờ tắt cho code chết; lịch sử vẫn còn trong git. Mọi thứ đã xóa ghi vào `docs/vna/CLEANUP.md` (đường dẫn, lý do, commit).

Danh sách dưới đây được rút từ ảnh chụp repo ở commit `d8e4be4`. HEAD hiện tại mới hơn, nên **phải chạy lại phép kiểm ở §1.0** trên HEAD rồi mới xóa. Mục nào ở HEAD khác với danh sách thì báo lại.

### 1.0 Cách xác định code chết

- **Điểm vào sau bước này:**
  - `experiments/pilot_v3.py` (B5, PM, CIL chính);
  - `experiments/competitors.py`;
  - `scripts/build_kaggle_vna_parallel.py`, `scripts/collect_parallel.py`;
  - các test còn giữ ở §1.6.
- Dựng đồ thị import bằng `ast` (chỉ tính lệnh `import`, **không** tính khớp chuỗi) từ các điểm vào. Module nào không với tới được thì xóa.
- Ghi kết quả (danh sách với tới được và không với tới được) vào CLEANUP.md.

### 1.1 Tách phần runner khỏi stack đo lường cũ (làm trước khi xóa)

Hiện `pilot_v3.py` và `competitors.py` kéo theo toàn bộ stack đo lường Stage A/B chỉ vì vài tiện ích:
- từ `stage_b3`: `UPSTREAM_PIN`, `parser`, `recipe`, `run`, `runtime`, `upstream_imports`;
- từ `stage_b5`: `RATES`, `UPSTREAM`, `UPSTREAM_PIN`, `algos`, `atomic_pickle`, `prepare_data`, `recipe`, `run_native`, `upstream_imports`, `working_directory`, `write_csv`, `write_json`;
- lớp `NoMeasurement` trong `pilot_v3.py` kế thừa `diagnostics.icarl_measure.B3Measurement`.

Sửa:
- Chuyển các tiện ích này vào `experiments/runners/pycil.py`, `experiments/runners/lop.py` và `experiments/io_utils.py`. Giữ nguyên hành vi.
- Viết lại `NoMeasurement` độc lập, không kế thừa `B3Measurement`.
- **Test:** sau khi tách, chạy `icarl` và `b5_bp` local (2 task, 1 epoch, 1% dữ liệu). Kết quả phải trùng bitwise với trước khi tách.

Sau bước này, `stage_b3`, `stage_b5`, `icarl_measure` và cả stack đo lường sẽ không còn ai import.

### 1.2 Module dự kiến xóa (kiểm lại bằng §1.0)

| Nhóm | Module | Lý do |
|---|---|---|
| Đo lường Stage A/B, signal curve, efficiency | `diagnostics/{stage_a, stage_b_smoke, signal_curve, measure_efficiency, vna_gate, vna_solver_validation, icarl_measure, lop_measure}.py`; `experiments/{stage_b3, stage_b4, stage_b5, stage_b12, phase0_discovery}.py` | Phép đo của các vòng thiết kế cũ, đã kết thúc. Thuật toán v4 không dùng |
| Đường train VNA v1/v2 | `experiments/{run_vna_training, vna_training_kaggle, vna_training_config, vna_training_support, vna_kaggle, vna_compare, vna_support, train_recipe_checkpoints, cifar_pilot, pilot_b3, pilot_b5, pilot_gir_vna}.py`; `training/{datasets, trainer}.py` | Thay bởi `pilot_v3.py`, rồi v4 |
| Online CIL | `experiments/pilot_v3_online.py` cùng adapter CCL-DC | Setup online ER không kích hoạt thuật toán |
| Method cũ | `methods/{vna_joint, vna_kfac, vna_output}.py`; trong `methods/vna_select.py` chỉ giữ hàm còn được import (ví dụ `neuron_norms`, `tiny_V`) | Kiểm bằng §1.0. K-FAC chỉ được dùng ở Stage B |
| Adapter cũ | `adapters/{pilot_lop, pilot_icarl, repdistiller_hooks}.py` | Pilot cũ; teacher RepDistiller của Stage B12 |
| Probe | `probe/counterfactual_tiny.py`; `probe/gromo_adapter.py` | Xóa, **trừ khi** ablation GIR với khởi tạo TINY (§4, đợt 3) dùng `gromo_adapter`. Nếu giữ thì chuyển vào `methods/` |
| Script builder cũ | `scripts/build_kaggle_vna_{b3_quick, cifar_pilot, pilot_b5, stage_a_notebook, stage_b_nb1, stage_b_nb2, stage_b_nb3, stage_b_rest, training_notebooks, signal_curve_notebook}.py`; `scripts/{smoke_measure_efficiency, smoke_pilot_gir_vna, recompute_gates, fetch_teachers}.py` | Notebook cũ. Hàm `cell` mà `build_kaggle_vna_parallel.py` đang import từ `build_kaggle_vna_signal_curve_notebook` phải chuyển sang `build_kaggle_pilot_common.py` trước khi xóa |

**Trong các file còn giữ**, xóa các nhánh và tham số chết:

| Xóa | Lý do |
|---|---|
| Các arm `*_lf`, `kernel_self_lf`; quy tắc λ = K, K·lr/(1−m) | T26 thay quy tắc bước; A13 không còn cần |
| Cap bước `lr·‖g‖/‖d̂‖` | T26(e) |
| Lưới `mu_rel` và preflight chọn μ (`mu_trials`, `mu_condition_met`) | T26: μ thích nghi |
| Exact check trong run full (`exact_cg`, `cos_cg_exact`, `exact_check_seconds`) | Chỉ giữ một hàm chẩn đoán cho test §7 |
| τ từ power iteration (`tau_g`, `tau_h`, `power_iters`, `certificate_norms`) | T26(f) |
| `LIVE_SWAP` và nhánh `measured` | T28 |
| Residual GIR `r = ẽ_O` | T27 |
| `S = 1`, bias `{0, −median}`, "làm mới A mỗi 10 lần", hằng số lịch `0.3/ρ`, `clip(¼, 4)` | §2.4, §2.5 |
| Arm `icarl_vna_b0`, `icarl_vna_rand`, `icarl_kself`, `er_*`, `erace_*`, `*_nf20` | Arm của setup cũ. Ablation mới định nghĩa lại ở §4, đợt 3 |

### 1.3 Notebook và preset

Xóa **mọi** file trong `notebooks/` và mọi preset trong `PRESETS`, rồi dựng lại chỉ các notebook ở §6 (M1–M3, C1–C5, B5w). Lý do cho từng nhóm cũ:

| Nhóm cũ | Vì sao sai setup |
|---|---|
| `kaggle_vna_stage_a`, `stage_b_nb1..3`, `signal_curve`, `cifar_pilot`, `b3_quick`, `pilot_b5`, `kaggle_vna_parallel` (gốc) | Phép đo hoặc pilot của các vòng cũ |
| `O1..O3`, `O1-R..O3-R` | Online ER: không có KD, không có neuron chết |
| `Q1..Q3` | Inc10, thứ tự ngẫu nhiên (seed 1993), chỉ có BN, thuật toán cũ (cap lr, μ cố định, exact check làm treo run) |
| `P7..P10` | Đối thủ chạy trên Inc10 và B50 Inc10, thứ tự ngẫu nhiên. Không cùng setup chính, nên không so được với `icarl_full` |
| `P-B5`, `P-PM` | Thuật toán cũ. Thay bằng `B5w`; PM dựng lại với thuật toán v4 nếu cần |

### 1.4 Output và tài liệu

- **Xóa:** `outputs/measure_efficiency_smoke/` và mọi output khác đang commit trong repo.
- **`docs/vna/`: chỉ giữ**
  - `vna_gir_theory.pdf` và source (`theory/`);
  - các script kiểm số: `final_check_v3.py`, `check_unified_gn.py`, `check_t25.py`, `check_t26_28.py`;
  - `PREREGISTRATION_CHANGES.md` (giữ các mục cũ làm lịch sử, thêm mục mới ở §8);
  - `CLEANUP.md`, `survey_competitors.md`, và prompt này.
- **Xóa phần còn lại của `docs/vna/`:**
  - mọi `TASK_*.md`;
  - các báo cáo stage (`STAGE*_REPORT.md`, `MEASURE_*`, `PARALLEL_REPORT.md`, `PILOT_GIR_VNA_REPORT.md`, `CLEANUP_REPORT.md`);
  - `DESIGN_V2*.md`, `SPEC.md`, `STAGE_AB_*`, `RUN_PLAN.md`, `SETTING_SURVEY.md`, `PROMPT_STAGE_AB.md`;
  - `THEORY_FINAL_v3.md` và `unified_gn.md` (đã nằm trong bản PDF);
  - các thư mục log (`*_logs/`, `v3_logs/`, `stage2*_validation/`), `figures/` (sơ đồ thuật toán cũ), các file `*_results.json`, `stage1_*`, `split3_*`;
  - `__pycache__/`.
- **Kết quả pilot v3 đã chạy** (B5 3 seed, `er` và `er_vna` của O1): không xóa dữ liệu. Ghi một file tóm tắt `docs/vna/pilot_v3_summary.md` (bảng số, commit, ghi rõ "thuật toán trước T26, không dùng để so với v4"), rồi xóa log thô khỏi repo.
- Cập nhật `README.md` và `AGENTS.md` theo cấu trúc mới.

### 1.5 `third_party/`

- **Giữ:** `PyCIL` (pin commit mới nếu cần, §3), `loss-of-plasticity` (GnT cho CBP, B5).
- **Thêm theo §4:** `upgd`, `C-Flat`, `dopamine` (chỉ phần ReDo, để port), và các repo đợt 3 khi cài.
- **Xóa:** `RepDistiller`, `One-Shot-TAS-CCIL`, CCL-DC (clone lúc chạy và patch trong `patches/`), các patch không còn áp dụng. Với `gromo` thì làm như `probe/gromo_adapter.py` ở trên.
- Kiểm bằng `grep` rằng không còn code nào tham chiếu tới các repo bị xóa. Cập nhật `SOURCES.md`, `REFERENCES.md` và `scripts/fetch_third_party.sh`.

### 1.6 Test

- **Xóa** test của các module đã xóa: `test_measure_efficiency`, `test_measure_round2`, `test_output_cholesky`, `test_parallel_pilot`, `test_pilot_b3`, `test_pilot_gir_vna`, `test_vna_stage_a`, `test_vna_stage_b*`, `test_vna_stage2`, `test_vna_split3`, `test_vna_training`, `test_vna` (nếu chỉ kiểm đường v1/v2), `test_vna_lemma` (nếu module đã xóa).
- **Giữ:** `test_cg`, `test_functional`, `test_theory_v3`, `test_vna_v3`, `test_vna_v3_certificates` (cập nhật cho T26–T28), `test_competitors`, `test_kaggle_checkpoint_discovery` (nếu module còn), cộng các test mới ở §7.
- Theo nguyên tắc §0.5, chỉ chạy các test ở §7 cùng các test đã giữ có liên quan, không chạy toàn bộ.

### 1.7 Notebook trên Kaggle (người dùng tự làm, agent chỉ liệt kê)

Agent liệt kê vào CLEANUP.md tên các kernel Kaggle cũ không còn dùng: `kaggle_vna_parallel_{O*, Q*, P7..P10, P-B5, P-PM}`, `kaggle_vna_stage_*`, `kaggle_vna_signal_curve`, `kaggle_vna_cifar_pilot`, `kaggle_vna_b3_quick`, `kaggle_vna_pilot_b5`. Người dùng sẽ xóa hoặc lưu trữ chúng trên Kaggle. Notebook mới đặt tên có tiền tố `vna4_` để không lẫn với notebook cũ.

---

## 2. Thuật toán: cài T26–T28

Ký hiệu như tài liệu.
- Quy ước cập nhật: θ ← θ − x.
- G = (F + μI)⁻¹, G_β = (F + βB + μI)⁻¹.
- Mọi hệ tuyến tính giải bằng CG matrix-free như hiện tại.

### 2.1 Sửa solver (bắt buộc)

- **Warm start an toàn.**
  - Trước khi chạy CG, tính residual của điểm warm start.
  - Nếu residual đó ≥ ‖b‖ (tức tệ hơn x = 0), khởi tạo lại từ 0.
  - Bỏ warm start khi sang task mới, và sau mỗi swap của GIR.
  - Log `warm_reset` (bool).
- **Dừng:** residual tương đối ≤ 1e-3, tối đa 200 vòng. Log `relative_residual`, `iterations`, `converged`.
- Nếu z không hội tụ (residual > 0.1) thì bỏ qua lần áp dụng đó. Log `skip_reason=cg_z`.

### 2.2 T26: trust region thay cho μ và cỡ bước

Trạng thái lưu qua các lần áp dụng (checkpoint được): `mu`, `nu`.

**Khởi tạo** (chỉ một lần, đầu run):
- μ₀ = 1e-2 · tr(F)/p, với tr(F) ước lượng bằng Hutchinson 8 vector.
- ν = 2.
- Ghi μ₀ vào `protocol.json`. Đây là giá trị khởi tạo, không phải siêu tham số được tune.

**Mỗi lần áp dụng:**
1. **S1:** x_GN = G·g (đã có). Δ = ‖x_GN‖.
2. **Đo độ tin cậy (T26d)** bằng một forward trên cùng cửa sổ, ở train mode; khôi phục buffer BN sau đó (A1):
   - `pred = gᵀx_GN − ½‖J x_GN‖²_H`, tính bằng một JVP;
   - `actual = L(θ) − L(θ − x_GN)`;
   - ϱ = actual / pred.
3. **Cập nhật μ** theo quy tắc Levenberg–Marquardt (Madsen–Nielsen–Tingleff, 2004):
   - nếu ϱ > 0: μ ← μ · max(1/3, 1 − (2ϱ − 1)³), và ν ← 2;
   - ngược lại: μ ← μ · ν, và ν ← 2ν.

   μ mới có hiệu lực **từ lần áp dụng kế tiếp**.
4. **Nếu ϱ ≤ 0:** bỏ qua bước VNA lần này (`skip_reason=untrusted`). GIR (S6) vẫn chạy với r⊥ của lần này.
5. **S5, cỡ bước (T26e):** s = min(s_F\*, Δ/‖d̂‖), với s_F\* = ĝ / ‖J d̂‖²_H (một JVP). Nếu mẫu số bằng 0 thì s = Δ/‖d̂‖.
6. **Phép kiểm chính xác (T26f).**
   - Sau khi áp θ' = θ − s·d̂ (ngoài momentum, như cũ), forward trên cửa sổ để lấy L(θ') và, nếu là CIL, L_KD(θ').
   - Nếu L tăng, hoặc (CIL) L_KD tăng: **hoàn tác θ (bitwise)**, đặt μ ← μ·ν, ν ← 2ν, và log `rejected=true`.
7. **S4:** chỉ còn là bộ lọc dấu rẻ: bỏ qua nếu ĝ ≤ 0, hoặc (CIL) ĥ < 0. Không có τ.

**Log thêm:** `rho_tr` (ϱ), `mu`, `nu`, `Delta`, `s`, `s_F_star`, `cap_binding ∈ {s_F, Delta}`, `rejected`, `L_before`, `L_after`, `KD_before`, `KD_after`.

### 2.3 T27: GIR có bảo vệ

Dùng cho CIL. Với B5 thì β = 0 và mọi công thức tự trở về như cũ.

1. Từ S1 đã có x_O := G·g_O (nghiệm dùng để tính z_O). Đặt δ_O = −O·J·x_O. Đây là độ dịch logit cũ do bước GN trên KD; tính bằng một JVP rồi lấy các logit cũ.
2. β lấy từ BetaRule của lần áp dụng này. Không có tham số mới.
3. e_β = e − β·OᵀW·δ_O, với W là Hessian KD theo từng mẫu (đã có trong code).
4. r⊥^β = e_β − H_β·J·G_β·Jᵀe_β, với H_β = H + β·OᵀWO.
   - Một lần giải CG với ma trận A_β = F + βB + μI. **Đây chính là ma trận VNA đang giải**, nên warm start từ nghiệm của BetaRule.
5. **Ứng viên:**
   - Hướng a_k = top-k eigvec suy rộng của (N_{r⊥β}·N_{r⊥β}ᵀ, S_x). Có BN theo sau thì dùng T14.
   - Dấu σ ∈ {±1}.
   - Bias theo §2.4.
   - Bỏ ứng viên có φ ≡ 0.
6. **Điểm:** ℓ_β(v) = ½‖J_vᵀ r⊥^β‖² / (λ_max(J_vᵀ H_β J_v) + μ).
   - J_vᵀH_βJ_v có kích thước nhỏ, nên λ_max tính chính xác.
7. Thay mọi chỗ đang dùng `r = ẽ_O` hoặc `r = e` bằng r⊥^β.

### 2.4 T28: thay slot theo lợi ròng

Một quy tắc duy nhất cho mọi slot hợp lệ (A12). Chạy sau S5, như cũ.

**Slot chết** (f_u = 0 trên cửa sổ tại θ'):
- swap khi ℓ_β(v\*) > 0;
- không cần tính Γ, vì Γ = gain ≥ 0 (T28b).

**Slot sống:**
1. Thứ tự xét: tăng dần theo ‖f_u‖_{H_β} = sqrt(f_uᵀH_βf_u).
   - f_u là đóng góp của slot trên cửa sổ, tính bằng forward với B_u = 0.
   - Số ứng viên xét mỗi lần áp dụng là **ngân sách compute** `live_candidates` (mặc định 4 mỗi site). Ghi vào `protocol.json`. Đây không phải tiêu chí quyết định.
2. **Đo bước nhảy chính xác** j_u^L = L(f − f_u) − L(f) và j_u^KD (CIL), bằng cùng forward ở bước 1. Nếu một trong hai > 0 thì loại u.
3. **Tính Γ(u, v)** theo T28(a):
   - lắp tạm v vào slot u với B_v = 0, nên output = f − f_u và span = J′;
   - e′ = e_β − H_β f_u;
   - Γ = [Q̂(0) − Q̂(−f_u)] + ½(J′ᵀe′)ᵀ(J′ᵀH_βJ′ + μ)⁻¹(J′ᵀe′) − ½(Jᵀe_β)ᵀ G_β (Jᵀe_β).
   - Số hạng cuối đã có từ §2.3, bước 4. Số hạng giữa cần một lần giải CG trên mạng đã lắp tạm.
   - Q̂(y) = eᵀy + ½yᵀHy + (β/2)‖Oy − δ_O‖²_W.
4. Swap khi **Γ > 0 và cả hai bước nhảy ≤ 0**. Ngược lại thì gỡ v, trả slot về nguyên trạng (bitwise).

**Sau mỗi swap:**
- Đặt lại trạng thái optimizer của slot (T13d).
- Nếu có BN: đặt γ, β_BN và running stats của kênh (T17).
- Cập nhật phần bù theo T13(b), hoặc bản rẻ: giữ r⊥^β, ứng viên kế tiếp S_x-trực giao.
- Bỏ warm start CG.

**Các lựa chọn nhỏ:**
- **Số swap:** mọi slot thỏa quy tắc, không giới hạn S.
- **Bias** của neuron mới: chọn b cực đại ℓ_β trên tập {0} ∪ {−quantile_q(σ·X·a_k) : q ∈ {0.1, 0.3, 0.5, 0.7, 0.9}}. Có BN thì áp lên β_BN.
- **Làm mới A:** mỗi lần áp dụng.

**Log mỗi swap:**
- `slot_kind ∈ {dead, live}`, `f_u_norm_Hb`, `j_L`, `j_KD`, `Gamma`, `ell_beta`;
- `bias_q`, `site`, `unit`, `task`, `iteration`;
- `contribution_norm` của neuron mới ở tuổi 1, 5 và 10 task.

### 2.5 Lịch áp dụng (ngân sách compute, không phải thuật toán)

- **Ngân sách:** tổng thời gian của thuật toán ≤ 1.0 lần thời gian train của iCaRL cùng setup. Đây là tham số duy nhất, ghi vào `protocol.json`.
- **Lịch trong mỗi task:**
  - Lần áp dụng đầu ở iteration đầu tiên của task (với B5 và PM: sau khi FIFO có ≥ 32 mẫu, như hiện tại).
  - Các lần sau cách nhau K₀, 2K₀, 4K₀, … (hình học). Lý do: theo bổ đề nội suy, lợi tập trung ngay sau khi đổi task.
- **K₀** do preflight chọn: đo thời gian một lần áp dụng ở task 1 và task 2 (3 lần mỗi task), rồi chọn K₀ nhỏ nhất thỏa ngân sách.
- **Preflight** không còn exact check và không còn dò μ, nên phải chạy xong trong < 15 phút. Ghi `K0`, `seconds_per_application` vào `_schedules/`.

### 2.6 Chẩn đoán T25 (log ở mỗi lần áp dụng, và cho baseline)

- **B(K) = gᵀG·g**: đã có từ S1, không tốn thêm.
- **‖w‖² = eᵀH⁺e**: H⁺ tính theo từng mẫu (khối c×c, chỉ trên range).
- **Phần với tới được:** `reach = B/‖w‖² ∈ [0, 1]`. **Phần không với tới:** `unreach = 1 − reach`.
- **λ_E** = λ_max(J_EᵀHJ_E): ma trận nhỏ (r_tot × r_tot), tính chính xác.
- **Kiểm cận T25(a):** log `V`, `lambda_E * B` và tỉ số của hai giá trị. Phải ≤ 1, sai lệch chỉ do sai số CG.

**Với mọi arm baseline và đối thủ:** ở cuối mỗi task, trên một **probe batch cố định**, gọi hàm `diagnose(model, batch)` để log:
- `reach`, `unreach`, B;
- tỉ lệ dead (bằng 0 trên toàn batch) và dormant (điểm ReDo với τ = 0.025) theo từng site.

Hàm này **không** thay đổi model. Probe batch gồm 256 mẫu, lấy từ test set của các lớp đã thấy, seed cố định.

### 2.7 GradMax (B5)

- Bắt `LinAlgError` và mọi lỗi giá trị non-finite trong `prepare_gir(gradmax=True)`. Khi gặp lỗi thì giữ khởi tạo gốc của CBP và log `gir_fallback=true`.
- Kiểm `torch.isfinite` trên đầu vào của `eigh`/`svd`. Tìm nguồn NaN và ghi vào CLEANUP.md.

---

## 3. Setup chính (áp giống hệt cho mọi arm)

| Mục | Giá trị |
|---|---|
| Codebase | PyCIL (`third_party/PyCIL`). Nếu commit hiện tại thiếu model cần dùng (PODNet, WA, GEM) thì cập nhật lên commit mới nhất của `LAMDA-CL/PyCIL` và pin lại |
| Dataset | CIFAR-100, root `/kaggle/input/datasets/nlnk1607/cifar100` |
| Số task | **B0 Inc5: 20 task, 5 lớp mỗi task** (`init_cls=5`, `increment=5`) |
| Thứ tự lớp (chính) | **Theo superclass**: mỗi task là đúng 5 lớp fine của một superclass. Thứ tự 20 superclass được hoán vị theo seed. Trong mỗi superclass, 5 lớp xếp theo seed |
| Thứ tự lớp (đối chứng) | Ngẫu nhiên theo seed như PyCIL mặc định |
| Backbone | `resnet32` của PyCIL, hai bản: **BN** (gốc) và **noBN** |
| Bản noBN | SkipInit (De & Smith, NeurIPS 2020): bỏ BN, conv có bias, thêm vô hướng α học được khởi tạo 0 nhân vào nhánh residual. Nếu loss NaN ở task 1 thì giảm lr ×0.5 **cho mọi arm của bản noBN**, và ghi lại |
| Base | iCaRL theo config gốc của PyCIL (lr, batch size, weight decay, T = 2) |
| Epoch | Task đầu 100, các task sau 50. Milestone của lr scheduler co theo tỉ lệ từ config gốc (làm tròn). Áp cho mọi arm |
| Memory | 2000, `fixed_memory=False` (mặc định của PyCIL) |
| Seed | Đợt 1: 3 seed (0, 1, 2). Đợt 2: 3 seed, mở rộng lên 5 khi đủ GPU |
| Phần cứng | Kaggle T4×2, mỗi GPU một lane |

**Superclass:**
- Đọc `coarse_labels` từ file pickle `train` của CIFAR-100 (bản python).
- Nếu dataset không có trường này, dùng bảng ánh xạ fine → coarse chính thức của CIFAR-100. Kiểm lại với `meta['coarse_label_names']` và test ở §7.

**Thuật toán của mình:**
- r = 16 (cố định, đăng ký trước).
- Lịch theo §2.5.
- Không tune gì thêm.

---

## 4. Các arm

### Đợt 1: xác nhận cả hai phía (chạy trước)

| Arm | Mô tả |
|---|---|
| `icarl` | Base |
| `icarl_full` | iCaRL + VNA + GIR + β (T26–T28) |

Ma trận chạy:
- Hai arm × {noBN, BN} × thứ tự superclass × 3 seed: 12 run.
- Đối chứng: hai arm × noBN × thứ tự ngẫu nhiên × 3 seed: 6 run.

### Đợt 2: đối thủ bắt buộc (chạy song song với đợt 1, không phụ thuộc)

Mỗi đối thủ chạy trên **cả hai backbone** với thứ tự superclass.

| Arm | Nguồn code | Siêu tham số quét trên seed 0 (bản noBN), chọn theo average incremental accuracy |
|---|---|---|
| `podnet`, `wa`, `gem` | Có sẵn trong PyCIL | Config gốc của PyCIL, không quét |
| `icarl_cbp` | `third_party/loss-of-plasticity` (GnT, đã có) | replacement rate {1e-6, 1e-5, 1e-4, 1e-3}, maturity 100 |
| `icarl_redo` | Code gốc ReDo (Sokar et al., ICML 2023) là JAX trong `google/dopamine`. Clone về, port sang PyTorch **đúng từng dòng** của hàm tính điểm dormant và hàm reset; đánh dấu `reimplemented=true` | τ ∈ {0, 0.025, 0.1}; reset mỗi {1, 5} epoch |
| `icarl_snp` | Ash & Adams (NeurIPS 2020). Kiểm có repo chính thức không; nếu không thì cài theo paper | shrink ∈ {0.4, 0.8}, noise σ ∈ {1e-4, 1e-2}, áp ở đầu mỗi task |
| `icarl_upgd` | `mohmdelsayed/upgd`. Dùng optimizer UPGD của họ thay SGD trong iCaRL | Lưới trong paper gốc (3 giá trị cho noise, 3 cho decay của utility) |
| `icarl_cflat` | `WanNaa/C-Flat` (đã có bản PyCIL, đúng protocol B0 Inc5, ResNet-32) | Config gốc. **Phép kiểm tái lập:** chạy thêm đúng protocol gốc của họ (thứ tự ngẫu nhiên, BN, đủ epoch) với 1 seed, và so với số đã công bố |

Sau khi quét: chạy cấu hình tốt nhất trên 3 seed, rồi mở rộng lên 5 seed.

### Đợt 3 (chuẩn bị code, chưa chạy)

- **Nên chạy:** L2-Init (CoLLAs 2024), Forget Forgetting (ICLR 2026; cài lại theo paper), FIRE (ICLR 2026; kiểm code), SWR (CoLLAs 2025), MRFA (`bwnzheng/MRFA_ICML2024`), BiC (PyCIL).
- **Bảng cận trên:** DER, FOSTER, MEMO (PyCIL), báo cáo kèm số tham số.
- **Ablation của mình:**
  - `+gir`, `+vna`;
  - `full_beta0`;
  - `full_Arand`;
  - `kernel_self`;
  - các cách khởi tạo GIR: ngẫu nhiên, GradMax, TINY.

### B5 width sweep (CPU, chạy song song)

- 5, 20 và 100 neuron × {bp, bp_vna, bp_gir, bp_vna_gir} × 3 seed, bản thuật toán T26–T28.
- Mục đích: kiểm dự đoán của T25. Khoảng cách (vna_gir − gir) phải tăng theo độ rộng.

---

## 5. Chỉ số (log cho **mọi** arm)

**Sau mỗi task:**
- CNN top-1 và NME top-1: tổng, lớp cũ, lớp mới.
- **Learning accuracy:** accuracy trên các lớp của task vừa học, đo ngay sau task đó.
- **Forgetting:** trung bình theo lớp đã thấy của (max accuracy trong quá khứ − accuracy hiện tại).
- Average incremental accuracy, last accuracy.
- **Chẩn đoán (§2.6):** `reach`, `unreach`, dead và dormant theo site.

**Ở task 5, 10, 15, 20:**
- **Feature accuracy** (Kim et al., CVPR 2023): train linear probe trên đặc trưng đông cứng, dùng train set của mọi lớp đã thấy, rồi đo trên test.

**Chi phí:** GPU-giờ, tỉ lệ so với `icarl`.

**File mỗi run:** `summary.json`, `tasks.json`, `continual_eval.csv`, `diagnostics.json`, `interventions.json` (chỉ arm có thuật toán của mình), `protocol.json`, `status.json`.

**Bảng tổng hợp cuối notebook:**
- mỗi arm × backbone × thứ tự: trung bình ± độ lệch chuẩn qua seed;
- delta ghép cặp so với `icarl`;
- số seed thắng.

**Figure:** mặt phẳng stability–plasticity, trục x là forgetting cuối, trục y là learning accuracy trung bình. Mỗi điểm là một arm, có error bar theo seed.

---

## 6. Notebook

Mọi notebook pin cùng commit, dùng checkpoint và resume như hiện tại. Mỗi notebook ghi `_schedules/` và kết quả vào `outputs/`.

| Notebook | Nội dung | Ghi chú |
|---|---|---|
| **vna4_M1** | Đợt 1, noBN, superclass: `icarl`, `icarl_full`, 3 seed | Chạy **preflight K₀** ở đây (authority). Ghi `_schedules/icarl_noBN.json` |
| **vna4_M2** | Đợt 1, BN, superclass: `icarl`, `icarl_full`, 3 seed | Preflight riêng cho BN (authority của bản BN) |
| **vna4_M3** | Đợt 1, noBN, thứ tự ngẫu nhiên: `icarl`, `icarl_full`, 3 seed | Dùng schedule của M1 (attach output M1); nếu chưa có thì tự chạy preflight và ghi `schedule_source=self` |
| **C1** | Quét siêu tham số trên seed 0 (noBN): CBP, ReDo, S&P, UPGD | Ghi `best_config.json` cho từng phương pháp |
| **C2** | PODNet, WA, GEM, C-Flat: 3 seed × 2 backbone | Không cần quét |
| **C3, C4** | CBP, ReDo, S&P, UPGD với cấu hình tốt nhất: 3 seed × 2 backbone | Attach output C1 |
| **C5** | Phép kiểm tái lập C-Flat theo protocol gốc, 1 seed | So với số công bố |
| **B5w** | B5 width sweep | CPU |

**Báo cáo cuối của mỗi notebook:** bảng ở §5, cùng các cờ (`gir_fallback`, `warm_reset`, `rejected`, `skip_reason`) và tỉ lệ của chúng.

---

## 7. Test ngắn (local, float64, mô hình nhỏ, chỉ các test này)

1. **T26:**
   - (a) Trên MLP nhỏ, x_GN(μ) khớp nghiệm của bài toán có ràng buộc ‖x‖ ≤ ‖x_GN‖ (SLSQP), sai lệch ≤ 1e-6.
   - (b) ‖s·d̂‖ ≤ Δ.
   - (c) Khi phép kiểm từ chối, θ sau hoàn tác trùng bitwise với θ trước; buffer BN trùng bitwise.
2. **T27:**
   - (a) r⊥^β tính matrix-free khớp công thức dày μ(H_βK + μ)⁻¹e_β, sai lệch ≤ 1e-8.
   - (b) Gain Schur bằng hiệu q̂\* tính trực tiếp, và ≥ 0.
   - (c) Với β = 0, kết quả trùng bitwise với quy tắc GIR cũ của B5.
3. **T28:**
   - (a) Γ theo công thức khớp cực tiểu trực tiếp (BFGS) trên MLP nhỏ, sai lệch ≤ 1e-6.
   - (b) Với slot chết, Γ bằng gain của T27.
   - (c) Khi swap bị loại, mạng trùng bitwise với trước.
4. **T25:** V ≤ λ_E·B trên MLP nhỏ (dùng `check_t25.py` làm tham chiếu).
5. **Superclass:**
   - 20 task, mỗi task đúng 5 lớp fine cùng một coarse label;
   - hợp các task là đủ 100 lớp;
   - thứ tự đổi theo seed.
6. **noBN ResNet-32:** forward và backward không có NaN trên một batch; số tham số được log.
7. **Wrapper đối thủ:** mỗi wrapper chạy 2 task × 1 epoch trên 1% dữ liệu, ở **local**, không lên notebook.
   - Với λ = 0 (hoặc tắt phương pháp), kết quả phải trùng bitwise với `icarl`.
8. **`diagnose()`** không thay đổi tham số hay buffer (so bitwise).
9. **CG warm start:** khi warm start tệ hơn 0, solver phải tự khởi tạo lại (test có chủ đích).

---

## 8. PREREGISTRATION_CHANGES.md: các mục phải thêm trước khi chạy

1. Thuật toán chuyển sang T26–T28; danh sách đã xóa ở §1.
2. Setup chính: CIFAR-100 B0 Inc5 theo superclass, ResNet-32 BN và noBN (SkipInit), epoch 100/50, memory 2000.
3. Setup đối chứng: thứ tự ngẫu nhiên. **Dự đoán đăng ký trước:** lợi của `icarl_full` so với `icarl` nhỏ hơn ở thứ tự ngẫu nhiên.
4. **Câu hỏi quyết định của đợt 1**, ghép cặp theo seed:
   - Q1: learning accuracy trung bình của `icarl_full` > `icarl`?
   - Q2: forgetting cuối của `icarl_full` ≤ `icarl`?
   - Q3: average incremental accuracy của `icarl_full` > `icarl`?
5. **Chẩn đoán bẫy**, đăng ký trước: với `icarl` ở thứ tự superclass, noBN:
   - dead hoặc dormant ≥ 10% ở task cuối;
   - learning accuracy giảm ≥ 5 điểm từ nửa đầu sang nửa sau chuỗi task;
   - forgetting > 10;
   - `unreach` tăng theo task.

   Ghi kết quả là đạt hay không, nhưng không dừng run.
6. Quy tắc tune đối thủ: lưới ở §4, seed 0, noBN, chọn theo average incremental accuracy.
7. Ngân sách compute 1.0× và quy tắc chọn K₀.
8. μ₀ = 1e-2·tr(F)/p chỉ là giá trị khởi tạo.

---

## 9. Báo cáo trước khi bấm chạy

Gửi lại:
- danh sách file đã xóa (CLEANUP.md);
- diff của `methods/` và `adapters/`;
- kết quả §7 (pass hoặc fail từng mục);
- repo và commit đã clone cho mỗi đối thủ, kèm cờ `reimplemented`;
- commit hash chung của mọi notebook;
- thời gian preflight ước tính.

Sau đó bấm chạy theo thứ tự: **M1, M2, C1, C2 trước**; rồi M3, C3, C4, C5, B5w khi có GPU.
