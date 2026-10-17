# VNA–GIR: lý thuyết cuối cùng, bản v3 (có thẩm quyền), 2026-10-17

**Lịch sử thẩm định:** 4 vòng referee độc lập (referee_report, referee2, referee3, referee4). Vòng 4 không phát hiện mục nào sai; các sửa nhỏ của vòng 4 đã áp vào bản này.

Tài liệu này **thay thế** `THEORY_FINAL_v2.md`, `THEORY_FINAL.md` (v1) và toàn bộ phần toán trong `math.md`, `math_rep.md`, `unified.md`, `repair_swap.md`, `proofs_final.md`. Bản v3 sửa mọi mục mà `referee3_report.md` nêu ra (1 mục sai, 1 mục nói quá, 8 mục cần sửa nhỏ); danh sách đối chiếu ở §9. Lịch sử v2 (đối chiếu referee 2) giữ ở §9b.

**Quyết định chính của v3:** quy tắc swap **slot sống** của v2 (Net_KD < 0) là sai về logic và được thay bằng dự báo theo đúng giả thiết hồi phục A10 (T16). Kiểm trên chính thí nghiệm 198 cặp của referee 3 cho thấy dự báo này chỉ đáng tin trong chế độ gần tuyến tính (μ_rel = 10), còn ở damping dùng thật (μ_rel = 1e-2) thì ngang tung đồng xu. Vì vậy **swap slot sống bị hạ xuống "đo, không chứng nhận", và TẮT mặc định.** Swap slot chết vẫn có chứng chỉ (bảo toàn hàm chính xác).

Mọi định lý được kiểm số bằng `final_check_v3.py` (float64), kết quả trong `final_check_v3_out.txt`: **144/144 kiểm tra đạt** (29 đối chứng âm, 22 nhân chứng) (chạy `python3 -I` từ thư mục trống, khoảng 45 s). Có ba loại kiểm phụ, đặt tên khác nhau:
- **NEG (28 cái): đối chứng đột biến.** Áp đúng phép kiểm đó cho một công thức cố ý sai; chỉ đạt khi công thức sai bị loại với biên độ rõ.
- **WIT (22 cái): nhân chứng tồn tại.** Chứng tỏ một hiện tượng có xảy ra (ví dụ h không đơn điệu), nên một khẳng định phủ định là đúng, hoặc một quy tắc là cần thiết.
- **[structural]:** phép kiểm đúng theo cách dựng (ví dụ T12c, T16e, T13c); ghi rõ để không bị đọc như bằng chứng độc lập.

Thí nghiệm 198 cặp đầy đủ của referee 3 được chạy lại ở `live_slot_198.py` → `live_slot_198_out.txt` (T16 dùng bản 80 cặp cho nhanh).

Mọi con số trích trong tài liệu lấy từ đúng lần chạy ghi trong `final_check_v3_out.txt` (các phép kiểm T0–T15 cho đúng các số của v2, vì phần sinh ngẫu nhiên không đổi). Mã kiểm (ví dụ `T3e`) là tên phép kiểm, **không** trùng với chữ cái của mục định lý; bảng §6 cho ánh xạ.

Mỗi khẳng định có một trong bốn trạng thái:
- **proven**: có chứng minh dưới các giả thiết đã liệt kê;
- **assumption**: giả thiết được nêu rõ;
- **empirical**: chỉ có bằng chứng số;
- **interpretation**: diễn giải, không phải định lý;
- **measured**: đại lượng được đo chính xác lúc chạy, nhưng không có bảo đảm nào về hệ quả của nó.

---

## 0. Ký hiệu

**Cửa sổ.** 𝒲 gồm N mẫu (dữ liệu mới cộng exemplar, sau augmentation), forward ở train mode.
- f(θ) ∈ ℝⁿ: logit xếp chồng, n = N·c, với c là số lớp hiện tại.
- θ_S ∈ ℝᵖ: các tham số được hiệu chỉnh (có thể loại tham số BN).
- J = ∂f/∂θ_S ∈ ℝ^{n×p}, K = JJᵀ.

**Loss** (đúng như PyCIL iCaRL): L = L_CE + L_KD.
- L_CE = (1/N) Σᵢ CE(fᵢ, yᵢ), tính trên mọi logit.
- L_KD = −(1/N) Σᵢ p_{t,i}ᵀ log softmax(f_{i,old}/T). Đây là `_KD_loss` của PyCIL: lấy trung bình, **không** nhân T².

**Residual.** e = ∂L/∂f = e_N + ẽ_O.
- e_{N,i} = (pᵢ − 1_{yᵢ})/N.
- ẽ_O = Oᵀe_O, với e_{O,i} = (p_{s,i} − p_{t,i})/(NT) và p_{s,i} = softmax(f_{i,old}/T) là xác suất của **student**.
- O ∈ {0,1}^{m×n} chọn m = N·c_old logit cũ, với OOᵀ = I.

**Hessian theo output** (khối chéo theo mẫu): Hᵢ = (D(pᵢ) − pᵢpᵢᵀ)/N + OᵢᵀWᵢOᵢ.
- Wᵢ = (D(p_{s,i}) − p_{s,i}p_{s,i}ᵀ)/(NT²), cũng dùng xác suất **student**.
- W = blkdiag(Wᵢ) ∈ ℝ^{m×m}.

**Các ma trận dẫn xuất.**
- C = OJ, B = CᵀWC, F = JᵀHJ.
- g = Jᵀe, g_O = Jᵀẽ_O = Cᵀe_O.
- μ > 0; A_β = F + βB + μI; G = A₀⁻¹; G_β = A_β⁻¹.

**Lời giải z.** z(r) := HJGJᵀr; z := z(e), z_N := z(e_N), z_O := z(ẽ_O). Theo tuyến tính, z = z_N + z_O.

**Neuron ảo.**
- A = [a₁ … a_r] là các hướng vào. Mỗi neuron có trọng số ra wⱼ ∈ ℝ^q và Jacobian J_{aⱼ} = ∂f/∂wⱼ ∈ ℝ^{n×q} tại wⱼ = 0.
- K_E = Σⱼ J_{aⱼ}J_{aⱼ}ᵀ.
- **Đặc trưng của neuron ảo trên cửa sổ:** φ(a) ∈ ℝ^N (hoặc một bản đồ đặc trưng). Gọi X ∈ ℝ^{N×d_a} là ma trận đầu vào (hoặc patch đã unfold) của site.
  - Không có BN theo sau: φ(a) = D(a)·Xa, với D(a) là mặt nạ ReLU thực tế (diag 0/1). Nếu không có ReLU thì D = I.
  - Có BN train mode theo sau (kênh mới, γ = 1, β_BN = 0): φ(a) = D(a)·X̃a/σ(a), với X̃ = X − 1x̄ᵀ (trừ trung bình theo cửa sổ) và σ(a)² = aᵀX̃ᵀX̃a/N.
- **N_s:** với a cố định trong một vùng mặt nạ (xem A4′), J_aᵀs = N_sᵀa (không BN) hoặc N_sᵀa/σ(a) (có BN); N_s tuyến tính theo s và có range ⊆ range(X̃ᵀ).
- **S_x (định nghĩa, có ridge):** S_x := X̃ᵀX̃/N + ρ_x I, với ρ_x = ρ_rel·tr(X̃ᵀX̃)/(N·d_a) và ρ_rel = 1e-6 (X̃ = X khi không có BN). S_x ≻ 0 kể cả khi d_a > N (fan-in conv lớn hơn batch). Mọi kết quả chỉ cần S_x ≻ 0 (Bổ đề L9).
- u = K_Ez, u_N = K_Ez_N, u_R = K_Ez_O.
- V = zᵀK_Ez, V_O = z_OᵀK_Ez_O.

**Hiệu chỉnh.** d_rep := GJᵀHu_R, d_N^β := G_βJᵀHu_N, d\*(β) := d_rep + d_N^β.

**Đại lượng giám sát.**
- leak(β) := ‖Cd_N^β‖²_W = (d_N^β)ᵀBd_N^β.
- h(β) := g_Oᵀd\*(β).
- Φ(d) := ‖Jd − u‖²_H + μ‖d‖², và Φ_N(x) := ‖Jx − u_N‖²_H + μ‖x‖².
- ε\* := V_O²/‖e_O‖²_{W⁺}, và β_suf := ‖e_O‖²_{W⁺}‖u_N‖²_H/V_O².
- Λ là thừa số chính xác của W, với ΛᵀΛ = W (Bổ đề L7), và Ĉ := ΛC.

**Slot và GIR.**
- Slot u có trọng số vào a_u, trọng số ra B_u và các cột Jacobian J_u.
- **Slot hợp lệ (A12):** đầu ra của u chỉ đi tiếp qua các trọng số **riêng** B_u, đặt được về 0 mà không ảnh hưởng đơn vị khác. Ví dụ trong ResNet: kênh ra của conv1 trong một basic block (B_u = lát `conv2.weight[:, u]`) là hợp lệ; kênh ra của conv2 đi vào phép cộng residual nên **không** hợp lệ.
- **Slot có BN theo sau:** slot gồm thêm γ_u, β_u (affine của BN) và running_mean_u, running_var_u. Đặc trưng là φ = ReLU(γ_u·x̂ + β_u), với x̂ = (Xa − mean)/√(var + eps_BN) tính theo batch (train mode). Bias trước BN bị BN triệt tiêu.
- **Đóng góp của slot:** f_u := f(θ) − f(θ với B_u = 0), tính bằng một forward. Ở lớp ẩn cuối, f_u = h_u(x)B_uᵀ. Ở lớp sâu hơn (kể cả khi có BN phía sau), f_u vẫn được định nghĩa đúng bằng forward đó.
- "Chết trên cửa sổ" nghĩa là tiền kích hoạt ≤ 0 trên mọi mẫu của 𝒲, nên f_u = 0 và J_u = 0 (ReLU, đạo hàm tại 0 bằng 0 như PyTorch).
- Neuron mới v có trọng số vào a_v (và bias b_v) và trọng số ra B_v = 0; J_v := ∂f/∂B_v tại B_v = 0. **J_v được tính chính xác cho mọi a_v, b_v, mọi phi tuyến** (nó chỉ phụ thuộc vào đặc trưng thực tế φ(a_v, b_v)); không cần A4.
- **Bước nhảy khi bỏ slot sống u** (với một loss ℓ ∈ {L, L_KD}): j_u^ℓ := ℓ(f − f_u) − ℓ(f), tính chính xác bằng một forward.
- **Dự báo dưới hồi phục trên L (v3).** Cho một trạng thái (f, J) với residual e = ∇L(f), Hessian H = ∇²L(f), và các đại lượng KD e_K := ∇L_KD(f) = ẽ_O, H_K := ∇²L_KD(f) = OᵀWO, **cùng tính tại f**:
  - bước hồi phục x_L := −(JᵀHJ + μI)⁻¹Jᵀe (bước Gauss–Newton có damping trên **L**, đúng như A10), và Δ := Jx_L;
  - mô hình loss m_L(f, J) := eᵀΔ + ½ΔᵀHΔ, mô hình KD m_K(f, J) := e_KᵀΔ + ½ΔᵀH_KΔ.
  - Với một swap slot sống (u → v, B_v = 0), trạng thái sau là (f − f_u, J′ = [J₋ᵤ, J_v]) và mọi đại lượng sau được tính **tại f − f_u**. Dự báo: ΔL_pred := j_u^L + m_L(f − f_u, J′) − m_L(f, J) và ΔKD_pred := j_u^KD + m_K(f − f_u, J′) − m_K(f, J).
  - Trước swap, x_L = −x_z với x_z = GJᵀe là nghiệm của lần giải đầu tiên ở bước 1, nên m_L(f, J), m_K(f, J) không tốn thêm lần giải nào.

**Phần bù và hàm chiếu.**
- Với r ∈ ℝⁿ: r_⊥ := r − z(r).
- Với một Jacobian bất kỳ 𝒥: G(𝒥) := (𝒥ᵀH𝒥 + μ)⁻¹ và H_⊥(𝒥) := H − H𝒥G(𝒥)𝒥ᵀH. Viết tắt H_⊥ := H_⊥(J).
- q\*_r(𝒥) := min_x rᵀ𝒥x + ½xᵀ(𝒥ᵀH𝒥 + μ)x = −½(𝒥ᵀr)ᵀG(𝒥)(𝒥ᵀr), là mô hình Gauss–Newton (có damping) của mức giảm loss khi chỉ được đi trong span(𝒥).
- E(𝒥, t) := min_x ‖𝒥x − t‖²_H + μ‖x‖², gọi là hàm chiếu.

## 1. Giả thiết

| # | Giả thiết | Kiểm lúc chạy, hoặc lý do đúng |
|---|---|---|
| A1 | e, J, H và bước cập nhật dùng **cùng một** cửa sổ và cùng forward train-mode. Buffer running-stat của BN không bị các forward của VNA thay đổi. | So buffer trước và sau mỗi lần áp dụng (phải bằng nhau bitwise). |
| A2 | Loss có dạng CE (trung bình, mọi logit) cộng KD của PyCIL (trung bình, nhiệt độ T). Mọi xác suất > 0. | Đúng theo định nghĩa. Ở float32, xác suất có thể underflow về 0; các công thức T1(c, d) không cần range nên vẫn đúng. |
| A3 | μ > 0. | μ = max(μ_rel·tr(F)/n, 1e-12). |
| A4′ | **Vùng mặt nạ cố định.** Các hướng a được xét nằm trong cùng một vùng R_D = {a : D(a) = D} (mặt nạ ReLU thực tế không đổi; với BN thì thêm: kênh mới, γ = 1, β_BN = 0). Trên R_D, đặc trưng là φ(a) = DXa (không BN), hoặc φ(a) = DX̃a/σ(a) (có BN). | Mặt nạ đo được lúc chạy. A4′ **chỉ** dùng cho tính *tối ưu* của bước chọn (T6, T7d, T14): tối ưu là trong R_D. Tối ưu **toàn cục** theo a (khi mặt nạ đổi) là giả thiết A4-glob, không chứng minh. Mọi chứng chỉ (T3, T5, T9, T11), mọi công thức lợi chính xác của swap (T7b, T13), và việc bảo toàn hàm (T7f) **không** cần A4′. |
| A5 | Slot chết trên cửa sổ: f_u = 0 và J_u = 0, **tại đúng tham số lúc chèn** (tức là sau bước 5). | Đếm các mẫu có tiền kích hoạt > 0 tại θ′ = θ − ελd̂ (phải bằng 0). Phải đếm lại sau bước 5 (T13a: một đơn vị chết trước bước 5 sống lại sau đó ở 18/40 bài). |
| A6 | ∇L, ∇L_KD lần lượt L_s-, L_K-Lipschitz trong **không gian tham số**. | **Không** kiểm được lúc chạy. Chỉ dùng cho các phát biểu bậc hai theo tham số (T5e, T5f). |
| A6′ | ∂²L/∂f² là L₃-Lipschitz trong **không gian output**. | **Đúng, có dạng đóng, chặt** cho CE trung bình + KD của PyCIL: L₃ ≤ c₃·(1 + T⁻³)/N với c₃ = 2√2/(6√3) ≈ 0.2722, và hằng số c₃ đạt được (Bổ đề L10). Không còn là giả thiết. |
| A7 | **Tham chiếu tự trị:** mô hình grow được thay bằng một động lực tự trị trong không gian output, ẏ = U(y) (một xấp xỉ kernel đóng băng), với U là L_U-Lipschitz. | Chỉ dùng cho T8. Một mạng grow thật có động lực output phụ thuộc vào tham số của nó, nên T8 chỉ so với xấp xỉ này. |
| A8 | CG dừng theo ρ = ‖b − Ax̂‖/‖b‖, đo so với ‖b‖. | Tính trực tiếp ρ cuối cùng. |
| A9 | Hiệu chỉnh cộng **ngoài** momentum buffer. Weight decay áp riêng. Khi swap, trạng thái optimizer (momentum; với Adam là hai moment) của các phần tử thuộc slot được **đặt bằng 0** (giá trị 0, không phải xóa buffer). | Theo cách cài đặt; kiểm bằng `torch.optim.SGD` thật (T10f, T13b, T13b2). Phát biểu chính xác của T13(d) là cho momentum thường (nesterov = False, dampening = 0, như PyCIL); với nesterov và dampening, bước đầu là một bội dương của gradient (T13(d), T13b2). |
| A10 | **Hồi phục trên L:** sau một swap, mạng sống hồi phục bằng (tương đương với) một bước Gauss–Newton có damping **trên loss huấn luyện L**, từ trạng thái hiện tại, trong chế độ tuyến tính (f(θ + x) ≈ f + Jx). | **assumption, interpretive.** Dùng để diễn giải gain (slot chết) và ΔL_pred, ΔKD_pred (slot sống) như mức thay đổi loss *về sau*. **Không** đúng cho một quá trình train thật: T16d-WIT cho thấy dự báo khớp kém với 200 bước GD kể cả ở chế độ tuyến tính. Không ảnh hưởng tới các đại lượng chính xác: bảo toàn hàm (slot chết), bước nhảy j_u (slot sống), các chứng chỉ. A10 **không** bao gồm hồi phục chỉ trên KD; vì vậy đại lượng Net_KD của v2 (mô hình hồi phục trên KD) không được A10 hỗ trợ (§7 mục 30). |
| A12 | Slot hợp lệ: đầu ra của u chỉ đi qua trọng số riêng B_u (§0). | Kiểm theo kiến trúc. Kênh vào phép cộng residual bị loại. |
| A11 | Mọi bảo đảm là cục bộ cho mỗi lần áp dụng (bậc một, hoặc bậc hai khi có A6), trên loss thực nghiệm của cửa sổ. | Phạm vi. Không có phát biểu nào về accuracy trên test. |

## 2. Thuật toán (bản cuối, v3)

```
Siêu tham số: μ_rel, K (khoảng áp dụng VNA, tính theo iteration), λ (như spec, chuẩn hóa cùng norm với NGD),
              r (số neuron ảo), n_b (số bước bisection, ví dụ 12), k_lo (độ rộng khoảng log, ví dụ 8),
              S (số swap tối đa mỗi lần áp dụng), ρ_rel = 1e-6 (ridge của S_x),
              LIVE_SWAP ∈ {off (mặc định), measured}.
Không có β, γ, η, τ_V tự do. Dung sai τ_h, τ_g của bước 4 được tính từ CG, không phải siêu tham số.

Mỗi iteration: bước iCaRL bình thường, θ ← θ − ε·m_t (m_t là momentum buffer của SGD).
Mỗi K iteration, trên cửa sổ 𝒲 hiện tại (cùng batch, train mode, khôi phục buffer BN sau đó):
 1. e = e_N + ẽ_O từ loss.  g = Jᵀe,  g_O = Jᵀẽ_O  (backward).
    z = z(e),  z_O = z(ẽ_O),  z_N = z − z_O                                # 2 lần giải (G)
 2. Nếu đến lượt làm mới A (đầu task, sau đó mỗi 10 lần áp dụng):
      A₀ = tiny_V(z)                  # top-r của (N_z N_zᵀ, S_x), S_x có ridge (L9)
      β₀ = β_rule(A₀)   (bước 3)
      y = HJ G_{β₀} g                 # 1 lần giải (G_β)
      M_S = sym(N_z N_{z_O}ᵀ) + sym(N_y N_{z_N}ᵀ)
      A = các eigvec suy rộng dương lớn nhất của (M_S, S_x), tối đa r cái, S_x-trực chuẩn   # T6
    Ngược lại, giữ A cũ.
 3. β_rule (CIL; B5 không có lớp cũ nên β = 0):
      u_R = K_E z_O;  u_N = K_E z_N;  V_O = z_Oᵀ K_E z_O = ‖J_Eᵀz_O‖².
      d_rep = G JᵀH u_R                                                    # 1 lần giải
      d_N⁰ = G JᵀH u_N                                                     # 1 lần giải; cho leak(0)
      χ² := (1/N)Σᵢ χ²(p_t,i ‖ p_s,i)                                      # dạng đóng, L6
      a) nếu χ² = 0 hoặc e_O = 0 (kể cả c_old = 1):  β = 0                  # T3(g)(iv): h ≡ 0; ε* = 0/0 không bao giờ được tính
      b) ngược lại, nếu V_O = 0:                     β = 0                  # T3(g)(iii): h(0) = 0 chính xác (solve chính xác);
                                                                            # với CG, ĥ = O(sai số solve), xem bước 4
      c) ngược lại: ε* = V_O²/χ²;  nếu leak(0) ≤ ε*: β = 0.
         ngược lại: β_suf = χ²‖u_N‖²_H/V_O²; bisection trên log β trong [β_suf·10^(−k_lo), β_suf], n_b bước,
                    warm start; β = đầu mút trên đã kiểm leak(β) ≤ ε*                # T3e, T11g
      d_N = G_β JᵀH u_N  (mỗi bước bisection 1 lần giải; leak = ‖ΛC d_N‖², C d_N là một JVP;
                          nếu bisection không bao giờ dời đầu mút trên, phải giải thêm 1 lần cho d_N(β_suf))
      d̂ = d_rep + d_N.
      (Nếu p_s underflow ở float32 làm χ² vô hạn hay NaN: dùng nhánh c với β = β_suf hoặc bỏ qua; an toàn vẫn
       do bước 4 bảo đảm, vì ĥ được tính trực tiếp.)
 4. Chứng chỉ trên đúng vector sẽ áp dụng (không qua solve): ĥ = g_Oᵀd̂, ĝ = gᵀd̂.
    Dung sai từ CG (T9(e)): với mỗi chuỗi hai lần giải c (z rồi d), Δ̂_c = ‖JᵀHK_E‖‖HJ‖ρ₁‖b₁‖/μ² + ρ₂‖b₂‖/μ
      (ρ₁, ρ₂: residual tương đối cuối của hai lần giải của chuỗi). B5 (β = 0): một chuỗi, Δ̂ = Δ̂_c.
      CIL: 4 lần giải = hai chuỗi (d_rep: z_O rồi d_rep; d_N: z_N rồi d_N tại β đã chọn), Δ̂ = Δ̂_rep + Δ̂_N
      (bất đẳng thức tam giác, vì d̂ = d̂_rep + d̂_N); các chuẩn phổ ước lượng bằng vài vòng lặp lũy thừa; với β > 0 thay G bằng G_β, μ vẫn là cận dưới),
      τ_h = ‖g_O‖Δ̂,  τ_g = ‖g‖Δ̂.  (Solve chính xác: τ = 0, trừ sai số làm tròn.)
    Nếu ĥ < −τ_h (CIL) hoặc ĝ < −τ_g: bỏ qua lần áp dụng (log). Không áp dụng gì khác.        # T5, T9(e), T18
    Khi áp dụng: g_Oᵀd̂ ≥ −τ_h, nên KD tăng bậc một không quá ελτ_h (→ 0 khi ρ₁, ρ₂ → 0).
 5. θ′ = θ − ε λ d̂, cộng NGOÀI momentum buffer (ghi d̂ vào buffer là sai, T10f).
 6. GIR, chạy SAU bước 5 (vì swap làm K lớn lên, T7g). Chỉ xét slot hợp lệ (A12).
    Quy ước điểm tính: r_⊥ và q*_r(J) tính tại θ (có sẵn từ bước 1); A5, f_u, j_u, J₋ᵤ, H′ tính tại θ′ = θ − ελd̂.
      r = ẽ_O (CIL: phần lớp cũ, ưu tiên 1),  r = e (B5).   r_⊥ = r − z(r)   # tại θ; độ lệch O(ελ‖d̂‖), T7(h)
      Ứng viên neuron v: với các hướng a_k = top-k eigvec của (N_{r⊥}N_{r⊥}ᵀ, S_x), xét σ ∈ {+1, −1} và
         bias b ∈ {0, −median(σ X a_k)}  (không BN), hoặc, khi có BN theo sau slot,
         γ_v = σ, β_v ∈ {0, −median(σ x̂_k)} với x̂_k là tiền kích hoạt đã chuẩn hóa theo batch   # T17b
         (bias của lớp tuyến tính đặt 0, vì BN triệt tiêu nó); bỏ ứng viên có φ ≡ 0 trên 𝒲.
         Điểm của ứng viên: ℓ(v) = ½‖J_vᵀr_⊥‖² / (λ_max(J_vᵀHJ_v) + μ)    # cận dưới có chứng chỉ của gain (T13g)
      a) Slot chết: đếm lại A5 tại θ′ (T13a). Với mỗi slot còn chết, theo thứ tự, tối đa S slot:
           v = argmax ℓ(v);  đặt a_v (và b_v, hoặc γ_v, β_v) vào slot; B_v = 0;
           nếu có BN: đặt running_mean_u = 0, running_var_u = 1 (thống kê eval của kênh mới)            # T17
           đặt trạng thái optimizer của mọi phần tử thuộc slot (a, b, γ, β, B) bằng 0              # T7f, T13b
           cập nhật phần bù cho slot kế tiếp:
             - bản chính xác (khi q nhỏ, ví dụ B5): r_⊥ ← r_⊥ − H_⊥J_v S_v⁻¹ s_v,
               H_⊥ ← H_⊥ − H_⊥J_v S_v⁻¹J_vᵀH_⊥                                # T13(c, d): q + 1 lần giải
             - bản rẻ (conv, q lớn): giữ r_⊥, chọn hướng kế tiếp S_x-trực giao với các hướng đã dùng.
               Bảo đảm: lợi chung ≥ lợi lớn nhất của từng slot (T13e).
      b) Slot sống: CHỈ khi LIVE_SWAP = measured (mặc định off). Đây là thao tác *đo*, không *chứng nhận* (T16).
           Trước hết tính lại e, J, H và lời giải x_z tại θ′ (1 backward + 1 lần giải).
           Với ứng viên u có ‖f_u‖ nhỏ nhất (lọc trước bằng cận Taylor với L₃ dạng đóng, L10), và v như trên:
             j_u^L, j_u^KD  (một forward, chính xác)
             chứng chỉ tức thời (chính xác, T7(i.5)):  j_u^L ≤ 0  và  (CIL) j_u^KD ≤ 0
             dự báo dưới A10 (§0):  ΔL_pred < 0  và  (CIL) ΔKD_pred < 0,
               với mọi đại lượng sau swap (e′, H′, e_K′, H_K′, J′) tính tại f − f_u           # 1 lần giải cho mỗi cặp (T16a)
           Swap khi cả bốn điều kiện đạt; sau đó làm như a) (đặt lại optimizer, BN). Sau mỗi swap sống,
           phải tính lại e, J, r_⊥ và A5 trước khi xét cặp kế tiếp (đầu ra và độ chết phía sau đã đổi).
           Log: j_u, ΔL_pred, ΔKD_pred, và giá trị thực đo được ở các lần áp dụng sau.
    Không có slot nào đạt: không làm gì (đây là kết quả, không phải lỗi).
 7. Log: ρ, V, V_O, h = ĥ, gᵀd̂, leak, ε*, β, nhánh của β_rule, β/β_suf, Φ_N(d_N), κ, số lần bỏ qua, ρ của CG,
         cos(d̂, d_exact) ở các điểm kiểm, defect thực hiện δ_m (T8), −ελω θᵀd̂ (weight decay),
         số slot chết trước/sau bước 5, số swap (chết/sống), ℓ(v), gain, j_u, ΔL_pred, ΔKD_pred, τ_h, τ_g, Δ̂.
```

**Chi phí mỗi lần áp dụng** (số lần giải hệ tuyến tính, trường hợp xấu nhất; mỗi lần giải là một CG có warm start):

| Phần | CIL, không làm mới A | CIL, có làm mới A | B5 |
|---|---|---|---|
| z, z_O | 2 | 2 | 1 (chỉ z) |
| β_rule cho A₀ (d_N⁰, đầu mút dưới, n_b bước) | — | 2 + n_b | — |
| y = HJG_{β₀}g | — | 1 | — |
| d_rep, d_N⁰ cho A đang dùng | 2 | 2 | 1 (d) |
| bisection cho A đang dùng (chỉ khi leak(0) > ε*): đầu mút dưới, n_b bước, và có thể +1 cho d_N(β_suf) | 2 + n_b | 2 + n_b | — |
| **Tổng** | **6 + n_b** (tốt nhất 4) | **9 + 2n_b** | **2** |

- Các chứng chỉ, leak, ε*, χ², r_⊥, τ_h, τ_g không cần giải thêm hệ nào (các chuẩn phổ trong Δ̂: vài vòng lặp lũy thừa bằng JVP/VJP).
- GIR slot chết: ℓ(v) không cần giải (J_vᵀHJ_v là ma trận q×q dựng từ φ và các khối Hᵢ). Bản deflation chính xác tốn q + 1 lần giải cho mỗi slot thêm.
- GIR slot sống (chỉ khi LIVE_SWAP = measured): 1 backward + 1 lần giải tại θ′ (một lần), rồi mỗi cặp ứng viên tốn 1 forward (j_u^L, j_u^KD) và **1 lần giải trên mạng đã sửa** (Jacobian mới J′ = [J₋ᵤ, J_v] dùng qua JVP/VJP của mạng đã swap, H′ dạng đóng tại f − f_u). Cả ΔL_pred và ΔKD_pred dùng chung x′_L, nên không cần vế phải thứ hai (quy tắc v2 cần hai vế phải, cho L và cho L_KD). Sau mỗi swap sống: thêm 1 backward + 1 lần giải.
- So với 2 lần giải của VNA thuần, bản CIL đắt hơn nhiều khi bisection chạy: phải đo trên ResNet32 và đưa vào quy tắc chọn K.

**Thay đổi so với v2 (referee 3):**
1. **Quy tắc swap slot sống viết lại và TẮT mặc định.** Net_KD của v2 mô hình hóa hồi phục *chỉ trên KD*, trái với A10 (hồi phục trên L). Thay bằng ΔL_pred, ΔKD_pred dọc đúng bước hồi phục x_L trên L, với mọi đại lượng sau swap tính tại f − f_u (T16a chứng minh vì sao). Kiểm trên thí nghiệm của referee 3: dự báo chỉ đáng tin ở μ_rel = 10, còn ở μ_rel = 1e-2 thì ngang ngẫu nhiên (T16b–d, `live_slot_198_out.txt`). Nên swap sống là *đo*, không *chứng nhận*, và mặc định tắt.
2. **Bỏ cách nói "slot sống có chứng chỉ trên cửa sổ".** Bước nhảy j_u được *đo* chính xác; nó chỉ là chứng chỉ khi đòi thêm j_u^L ≤ 0 và j_u^KD ≤ 0 (T7(i.5)), và khi đó điều duy nhất được bảo đảm là: tại lúc chèn, L và KD trên cửa sổ không tăng.
3. **L₃ chặt:** c₃ = 2√2/(6√3) ≈ 0.2722 thay cho 0.708 (L10, có chứng minh đạt cận). T12a nay kiểm cả tính đúng lẫn tính chặt, nên bắt được hằng số sai.
4. **Slot có BN:** đặt lại γ, β (và running stats) của kênh; quy tắc dấu/bias tác động lên β_BN (T17).
5. **Bước 4 dùng dung sai từ CG** (τ_h, τ_g, T9(e)): quy tắc chặt ĥ < 0 bỏ qua nhầm 15/30 bước có chứng chỉ khi V_O = 0 (T18b-WIT).
6. **Chi phí:** +1 lần giải có thể cho d_N(β_suf); slot sống: 1 lần giải mỗi cặp trên mạng đã sửa, cộng tính lại tại θ′.
7. **Phạm vi của T13(d):** momentum thường; nesterov và dampening cho bội dương (1+ν) hoặc (1−δ) của gradient (T13b2).
8. **Phép kiểm:** T7f kiểm đúng quy tắc argmax ℓ; T12d (vốn là đồng nhất hiển nhiên) thay bằng so sánh trên mạng thật; thêm T12a-NEG, T12b-NEG (độ mạnh của phép kiểm L₃), T16–T18; gắn nhãn [structural] cho T12c, T13c, T16e.
9. **Slot hợp lệ (A12)** và quy ước θ/θ′ cho từng đại lượng ở bước 6 được nêu rõ.

**Thay đổi so với v1, do chứng minh hoặc referee bắt buộc (v2):**
1. **Bỏ quy tắc "V_O = 0 ⇒ bỏ qua" và ngưỡng τ_V chưa định nghĩa.** Khi V_O = 0, h(0) = 0 chính xác, nên β = 0 có chứng chỉ (T3(g)(iii), T11a). Quy tắc v1 vứt bỏ các bước an toàn (T11c-WIT: 60/60 bài).
2. **Trường hợp e_O = 0 (kể cả c_old = 1)** được xử lý riêng: ε* là 0/0, còn h ≡ 0 (T3(g)(iv), T11d, T11e).
3. **Slot sống được phép, có chứng chỉ trên cửa sổ** (v3: rút lại phần "có chứng chỉ" và tắt mặc định, xem §7 mục 30–32). v1 loại slot sống vì cho rằng L₃ không kiểm được. Thực ra L₃ (không gian output) có dạng đóng (L10), và bước nhảy j_u tính chính xác bằng một forward (T7(i), T12). Đây là thay đổi quan trọng cho CIL, nơi 0% đơn vị chết.
4. **GIR bước 6:** đếm lại A5 sau bước 5; đặt lại trạng thái optimizer của slot; quy tắc nhiều slot (deflation chính xác hoặc bản rẻ có cận); quy tắc dấu và bias của a_v cho ReLU (T13).
5. **Tính tối ưu của bước chọn chỉ được khẳng định trong một vùng mặt nạ cố định (A4′)**; với BN, bài toán chọn là bài toán trị riêng suy rộng với thống kê *đã trừ trung bình* (T14).
6. **S_x được định nghĩa, có ridge** (L9).
7. **Đếm chi phí được sửa** (bảng trên).

## 3. Bổ đề

**L1.** Với R = H^{1/2} và M = RKR: RJGJᵀR = M(M+μ)⁻¹ ⪯ I. Hệ quả: ‖JᵀHu‖²_G ≤ ‖u‖²_H.
*Chứng minh.* Push-through: (RJ)(JᵀR²J + μ)⁻¹(RJ)ᵀ = (RJJᵀR + μ)⁻¹(RJJᵀR) = M(M+μ)⁻¹, và các trị riêng thuộc [0, 1). Hệ quả: uᵀHJGJᵀHu = (Ru)ᵀM(M+μ)⁻¹(Ru) ≤ ‖Ru‖². ∎

**L2.** G_β ⪯ G ⪯ μ⁻¹I, vì B, F ⪰ 0.

**L3.** βG_β^{1/2}BG_β^{1/2} ⪯ I, vì βB ⪯ A_β.

**L4 (đồng nhất).** (Jᵀr)ᵀGJᵀH = z(r)ᵀ với mọi r, vì H và G đối xứng.

**L5 (range).** Khi mọi xác suất > 0:
- null(Hᵢ) = span(1_c) và null(Wᵢ) = span(1_{c_old});
- 1ᵀe_{N,i} = 0 và 1ᵀe_{O,i} = 0.

Do đó e_N, ẽ_O, e ∈ range(H) và e_O ∈ range(W).

*Chứng minh.*
- xᵀ(D(p) − ppᵀ)x = Var_p(x), bằng 0 khi và chỉ khi x là hằng (p > 0).
- Hᵢ là tổng của phần CE (bằng 0 khi và chỉ khi x hằng) và phần KD (bằng 0 khi và chỉ khi x_old hằng). Vậy null(Hᵢ) = span(1).
- Các residual có tổng bằng 0 vì Σp = Σ1_y = 1, và Σp_s = Σp_t = 1.
- Với ẽ_O: range(H) ⊇ range(OᵀWO) = Oᵀrange(W) ∋ Oᵀe_O. ∎

Kiểm: T0a (2.5e-11), T0b (2.1e-12), T0b-WIT (vector tổng quát không thuộc range(Hᵢ)).

**L6 (dạng đóng χ²).** ‖e_O‖²_{W⁺} = (1/N)Σᵢ χ²(p_{t,i} ‖ p_{s,i}), với χ²(p_t‖p_s) = Σⱼ(p_{s,j} − p_{t,j})²/p_{s,j}. Nếu KD có trọng số κ thì nhân thêm κ.
*Chứng minh.*
- Với v ⊥ 1, x = D⁻¹v thỏa (D − ppᵀ)x = v − p(1ᵀv) = v. Hình chiếu của x lên range khác x một bội của 1, nên vᵀ(D − ppᵀ)⁺v = vᵀD⁻¹v.
- Thay hệ số: eᵢᵀWᵢ⁺eᵢ = (NT)⁻²(NT²)Σⱼ(p_s − p_t)²/p_s = χ²ᵢ/N. ∎

Kiểm: T0c (2.8e-15) (T0c-NEG: dùng p_t ở mẫu số bị loại, lệch ≥ 1.7e-2).

**L7 (thừa số chính xác của W).** Λᵢ := (NT²)^{-1/2}(D(p_s)^{1/2} − √p_s p_sᵀ) thỏa ΛᵢᵀΛᵢ = Wᵢ và null(Λᵢ) = span(1).
*Chứng minh.*
- Đặt S = D^{1/2}. Vì S√p = p, ta có ΛᵀΛ ∝ (S − p√pᵀ)(S − √ppᵀ) = D − ppᵀ − ppᵀ + p(√pᵀ√p)pᵀ = D − ppᵀ.
- Λx = 0 ⇔ Sx = √p(pᵀx) ⇔ xⱼ = pᵀx với mọi j ⇔ x là hằng. ∎

Kiểm: T0f (ΛᵀΛ = W lệch 8e-16; Λ1 = 0).

*Hệ quả cài đặt:* mọi phép tính ‖·‖_W, và ràng buộc ở giới hạn cứng, dùng Λ. Không dùng căn bậc hai lấy qua trị riêng, vì các trị riêng xấp xỉ 0 của nó sinh hạng giả (lỗi số đã thấy ở T2g/T2i trước khi chuyển sang Λ).

**L8 (đạo hàm KD).** ∂L_KD/∂f = Oᵀe_O và ∂²L_KD/∂f² = OᵀWO, với p_s của **student**. Hessian dựng từ xác suất teacher là sai, trừ khi p_s = p_t.
Kiểm: T0d (2.2e-16), T0e (1.8e-16) (T0e-NEG: Hessian teacher lệch 0.69).

**L9 (S_x có ridge).** Cho X̃ ∈ ℝ^{N×d_a}, S₀ := X̃ᵀX̃/N, S_x := S₀ + ρ_xI với ρ_x > 0, và M ⪰ 0 hoặc đối xứng bất kỳ với range(M) ⊆ range(X̃ᵀ) (đúng cho mọi M dựng từ N_s, vì N_s = X̃ᵀ(·)). Gọi B là một cơ sở trực chuẩn của range(X̃ᵀ).
1. S_x ≻ 0 với mọi d_a, kể cả d_a > N.
2. Mọi eigvec suy rộng của (M, S_x) ứng với trị riêng khác 0 nằm **chính xác** trong range(X̃ᵀ).
3. Các trị riêng khác 0 đó bằng trị riêng của bút chì thu hẹp (BᵀMB, BᵀS₀B + ρ_xI), và hội tụ về trị riêng của (BᵀMB, BᵀS₀B) khi ρ_x → 0.
4. Với a S_x-chuẩn hóa nằm trong range(X̃ᵀ): aᵀS₀a = 1 − ρ_x‖a‖² ∈ [1 − ρ_x/λ⁺_min(S₀), 1], tức chuẩn đặc trưng thật lệch khỏi 1 không quá ρ_x/λ⁺_min(S₀).

*Chứng minh.*
- (1) S₀ ⪰ 0 và ρ_x > 0.
- (2) Viết a = a_r + a_n với a_r ∈ range(X̃ᵀ), a_n ∈ null(X̃). S₀ và I đều giữ hai không gian này. Từ Ma = λS_xa: chiếu lên null(X̃) cho 0 = λ(S₀a_n + ρ_xa_n) = λρ_xa_n, nên λ ≠ 0 ⇒ a_n = 0.
- (3) Thay a = Bc vào Ma = λS_xa và nhân Bᵀ. BᵀS₀B ≻ 0, nên trị riêng liên tục theo ρ_x tại 0.
- (4) aᵀS_xa = 1 ⇒ aᵀS₀a = 1 − ρ_x‖a‖², và ‖a‖² ≤ aᵀS₀a/λ⁺_min(S₀) ≤ 1/λ⁺_min(S₀) trên range(X̃ᵀ). ∎

Mọi định lý dùng S_x (T6, T7d, T14) chỉ cần S_x ≻ 0, nên giữ nguyên khi có ridge. Kiểm: T14c (phần ngoài range 1.2e-7, đúng bằng cond·eps; khoảng cách trị riêng ở ρ = 1e-8 là 1.9e-8), T14c-WIT (không có ridge thì Cholesky thất bại khi d_a > N).

**L10 (L₃ dạng đóng, chặt, trong không gian output).** Đặt c₃ := 2√2/(6√3) = √2/(3√3) ≈ 0.2722. Với L = CE trung bình (mọi logit) + KD của PyCIL (trung bình, nhiệt độ T, không nhân T²), với mọi f và mọi h ∈ ℝⁿ:
|D³L(f)[h, h, h]| ≤ c₃·(1 + T⁻³)/N · ‖h‖³.
Riêng CE: ≤ c₃/N·‖h‖³; riêng L_KD: ≤ c₃·T⁻³/N·‖h_old‖³. **Hằng số c₃ là chặt** cho từng thành phần (CE, KD với mọi T): đạt được ở giới hạn một mẫu, hai lớp mang xác suất q\* = (3 − √3)/6 và 1 − q\*, các lớp khác có xác suất → 0, h = (1, −1, 0, …)/√2.

*Chứng minh.*
- Với LSE(x) = log Σexp xⱼ và p = softmax(x): D³LSE(x)[h,h,h] = E_p[(X − E_pX)³], với X nhận giá trị hⱼ theo xác suất pⱼ (đạo hàm bậc ba của hàm sinh tích lũy là tích lũy bậc ba).
- **Cực đại của mô-men trung tâm bậc ba trên một khoảng.** Gọi Y = X − EX, nhận giá trị trong [−α, β] với α + β ≤ R := maxⱼhⱼ − minⱼhⱼ. Bài toán max E Y³ trên mọi phân phối của Y với EY = 0 là một bài toán mô-men tuyến tính theo độ đo, với hai ràng buộc (khối lượng 1, trung bình 0). Theo định lý Richter–Rogosinski, cực trị đạt tại một phân phối có nhiều nhất hai điểm. Với hai điểm −s < 0 < t (xác suất t/(s+t) tại −s), ta có E Y² = st và E Y³ = st(t − s), với s + t ≤ R. Biểu thức tăng khi phóng to (s, t), nên lấy s + t = R, s = Rx: E Y³ = R³x(1 − x)(1 − 2x), đạt max tại x = (3 − √3)/6 với giá trị R³/(6√3). Đối xứng cho min. Vậy |E Y³| ≤ R³/(6√3).
- R ≤ √2‖h‖ (vì (hᵢ − hⱼ)² ≤ 2(hᵢ² + hⱼ²)), nên |D³LSE[h,h,h]| ≤ (√2)³/(6√3)·‖h‖³ = c₃‖h‖³.
- CE của mẫu i là LSE(fᵢ) − f_{i,yᵢ} (phần tuyến tính không có đạo hàm bậc ba), chia N. KD của mẫu i là (LSE(f_{i,old}/T) − p_{t,i}ᵀf_{i,old}/T)/N, nên đạo hàm bậc ba có thêm hệ số T⁻³ và chỉ phụ thuộc h_{i,old}, với ‖h_{i,old}‖ ≤ ‖hᵢ‖.
- Cộng theo mẫu: |D³L[h,h,h]| ≤ c₃(1 + T⁻³)/N · Σᵢ‖hᵢ‖³ ≤ c₃(1 + T⁻³)/N · (Σᵢ‖hᵢ‖²)^{3/2}.
- **Đạt cận.** Với h = (1, −1, 0, …)/√2 ta có ‖h‖ = 1 và R = √2 = √2‖h‖; phân phối của X dồn về hai điểm ±1/√2 với xác suất q\*, 1 − q\*. Vậy cả hai bất đẳng thức đều thành đẳng thức ở giới hạn. Với KD dùng logit cũ f_old/T = (log q\*, log(1 − q\*), −∞, …). ∎

Cận cho tổng CE + KD với hệ số (1 + T⁻³) là đúng nhưng không nhất thiết chặt (hai phân phối softmax khác nhau); ở T = 1, khi các lớp mới có xác suất → 0, hai phân phối trùng nhau và cận tổng cũng đạt (T12b).

Hệ quả: A6′ không còn là giả thiết. Kiểm: T12a (cấu hình hai điểm đạt 1.0000 cận cho CE và cho KD ở T = 1, 2, 4; dò ngẫu nhiên tối đa 0.886 cận), T12a-NEG (hằng số đột biến bị loại: chuẩn hóa T⁻¹ cho KD hỏng tính chặt, hằng số ×0.1 hỏng tính đúng), T12a-WIT (hằng số 0.708 của v2 đúng nhưng lỏng 2.6 lần), T12a2 (CE + KD, nhiều mẫu, tỉ số tối đa 0.576; hai phía: cấu hình thẳng hàng N = 1, T = 1 đạt 1.0000 cận tổng), T12a2-NEG (chuẩn hóa max(1, T⁻³) thay cho tổng 1 + T⁻³ bị vi phạm, tỉ số 2.000).

## 4. Định lý

### T1. Dạng đóng, phân rã, các dạng không cần range

**Định lý 1.** Giả thiết A2, A3, β ≥ 0.

1. **(a)** Bài toán (P_β): min_d Φ(d) + β‖C(d − d_rep)‖²_W có nghiệm duy nhất d\*(β) = d_rep + d_N^β.
2. **(b)** Φ(d_rep + x) = Φ_N(x) + const. Vì vậy (P_β) tương đương min_x Φ_N(x) + β xᵀBx, và nghiệm là x = d_N^β.
3. **(c) Dạng output-space.** Lấy Ĵ = [J; C] và Ĥ = blkdiag(H, βW) = R̂ᵀR̂, với một thừa số bất kỳ (ví dụ R̂ = blkdiag(H^{1/2}, √βΛ)). Đặt t̂ = [u; Cd_rep] và M̂ = R̂ĴĴᵀR̂ᵀ. Khi đó d\* = ĴᵀR̂ᵀ(M̂ + μ)⁻¹R̂t̂. Không có pseudo-inverse.
4. **(d) z không cần range.** Với mọi thừa số H = ΛₕᵀΛₕ: z(r) = Λₕᵀ(ΛₕKΛₕᵀ + μ)⁻¹ΛₕK r = Λₕᵀ(ΛₕKΛₕᵀ + μ)⁻¹Λₕ(Jg_r), với g_r = Jᵀr.

*Chứng minh.*
- **(a)** Hessian của mục tiêu là 2(A_β) ≻ 0, nên nghiệm duy nhất. Điều kiện dừng: A_βd = JᵀHu + βBd_rep. Thay d = d_rep + x và dùng (F+μ)d_rep = JᵀHu_R, ta được A_βx = JᵀHu_N.
- **(b)** Khai triển. Số hạng chéo là −2xᵀ[JᵀHu_R − (F+μ)d_rep] = 0.
- **(c)** Mục tiêu là ‖R̂(Ĵd − t̂)‖² + μ‖d‖², nên d = (ĴᵀĤĴ + μ)⁻¹ĴᵀR̂ᵀR̂t̂. Push-through (XᵀX + μ)⁻¹Xᵀ = Xᵀ(XXᵀ + μ)⁻¹ với X = R̂Ĵ cho dạng đã nêu.
- **(d)** z(r) = ΛₕᵀΛₕJ(JᵀΛₕᵀΛₕJ + μ)⁻¹Jᵀr. Push-through với X = ΛₕJ cho Λₕᵀ(XXᵀ + μ)⁻¹XJᵀr. ∎

*Hệ quả cho lỗi cũ trong code.* Công thức cũ R M(M+μ)⁻¹R⁺e trùng (d) khi phép tính chính xác. Nhưng R⁺ có ngưỡng trị riêng làm sai khi softmax bão hòa.

Kiểm:
- T1a: 3.7e-12 (T1a-NEG: bỏ β thì lệch 6.0e-2);
- T1b: 4.0e-12;
- T1c: 1.4e-11 ở logit scale 1 đến 30 (T1c-NEG: công thức pinv có ngưỡng lệch 0.81 ở scale ≥ 20).
- Referee 2 kiểm thêm với một thừa số không vuông của H (độc lập): 1.9e-8 / 1.8e-9. Nhận xét "code cũ trùng (d) khi phép tính chính xác" cần e ∈ range(H), điều L5 cung cấp.

### T2. Rút gọn, giới hạn β → ∞, bổ đề chiếu lồng nhau

**Định lý 2.** Giả thiết A2, A3.

1. **(a) β = 0.** d\*(0) = GJᵀHK_Ez, đúng VNA hiện tại. Ngoài ra gᵀd\*(0) = V ≥ 0, và d\*(0) = Qg với Q = LᵀK_EL ⪰ 0, L = HJG.
2. **(b) Giới hạn cứng, với C hạng tùy ý.** Đặt A := A₀, b := JᵀHu_N, Ĉ = ΛC. Khi β → ∞:
   d_N^β → d_N^∞ := argmin{Φ_N(x) : Ĉx = 0} = A⁻¹b − A⁻¹Ĉᵀ(ĈA⁻¹Ĉᵀ)⁺ĈA⁻¹b,
   với ‖d_N^β − d_N^∞‖_A ≤ ‖Π A^{-1/2}b‖ / (1 + βσ²_min). Ở đây σ_min là giá trị suy biến khác 0 nhỏ nhất của ĈA^{-1/2}, và Π là phép chiếu lên không gian hàng của nó.
   - **Trường hợp suy biến:** nếu Ĉ có hạng cột đầy (cần p ≤ rank Ĉ ≤ N(c_old − 1)) thì null(Ĉ) = {0}, nên d_N^∞ = 0: giới hạn cứng đóng băng hoàn toàn việc học mới (T11f). Với ResNet32 thì p ≫ N(c_old − 1), nên trường hợp này chỉ xảy ra ở mạng rất nhỏ hoặc cửa sổ rất lớn.
3. **(c) Ràng buộc là ΛCx = 0, KHÔNG phải Cx = 0.** Vì null(W) = ⊕ᵢ span(1), giới hạn cứng vẫn cho phép mỗi mẫu dịch đều toàn bộ logit cũ; KD bất biến với phép dịch đó. `math.md` Prop 2 (Cx = 0) **sai**. (W^{1/2} và Λ có cùng null space, nên dùng Λ thay cho căn qua trị riêng chỉ là lời khuyên số học, không phải khác biệt toán học.)
4. **(d) Bổ đề chiếu lồng nhau, giới hạn cứng.** Với P = I − Ĉ⁺Ĉ:
   Jd_N^∞ ∈ range(K_NS), với K_NS := JPJᵀ = K − JĈᵀ(ĈĈᵀ)⁺ĈJᵀ (phần bù Schur theo khối [J; Ĉ]).
   Vậy ở giới hạn cứng, vận tốc học-mới nằm trong range(K_NS) ⊆ range(K).

*Chứng minh.*
- **(a)** L4 cho gᵀGJᵀH = zᵀ, nên gᵀd = zᵀK_Ez. Viết d = GJᵀHK_EHJGg = LᵀK_ELg.
- **(b)** Đổi biến y = A^{1/2}x. Bài phạt thành min ‖y − A^{-1/2}b‖² + β‖ĈA^{-1/2}y‖².
  - Lấy SVD ĈA^{-1/2} = UΣVᵀ (bản rút gọn). Phần của y ngoài range(V) không phụ thuộc β.
  - Tọa độ thứ i trong range là bᵢ/(1 + βσᵢ²) → 0.
  - Giới hạn chính là nghiệm KKT, đúng chính xác với (·)⁺, vì ĈA⁻¹b ∈ range(ĈA⁻¹Ĉᵀ) = range(ĈA^{-1/2}).
- **(c)** Ĉx = 0 ⇔ Cx ∈ null(Λ) = null(W).
- **(d)** Ĉd_N^∞ = 0, nên d_N^∞ = Pd_N^∞ và Jd_N^∞ ∈ range(JP) = range(JPJᵀ). Đồng nhất cho K_NS là P = I − Ĉᵀ(ĈĈᵀ)⁺Ĉ. ∎

*Ghi chú.* Khẳng định "Jd ∈ range(K) với mọi d" (`math_rep` 5(i)) đúng một cách hiển nhiên và không có nội dung; đã bỏ. Nội dung thật nằm ở (d).

Kiểm:
- T2a: 5.2e-16; T2b: 5.5e-13; T2c: Q ⪰ 0 (min eig/max eig = −2.7e-16);
- T2d: tỉ số với cận ≤ 0.900;
- T2e: tốc độ O(1/β), 48 cặp, lệch ≤ 1.8%;
- T2f: β = 1e7 lệch KKT 3.5e-7 (T2f-NEG: với ràng buộc Cx = 0, khoảng cách ≥ 0.265);
- T2g: C hạng thiếu (p < N·c_old), dư ràng buộc 4.1e-13;
- T2h (Schur): 1.6e-14; T2i: 1.1e-13;
- T11f (trường hợp suy biến): p ≤ N(c_old − 1), rank Ĉ = p: ‖d_N^β‖/‖d_N⁰‖ = 1.3e-8 tại β = 1e10.

### T3. Sửa, rò rỉ và chứng chỉ không quên thêm

**Định lý 3.** Giả thiết A2, A3; có lớp cũ (CIL).

1. **(a) Phần sửa đúng chiều.** g_Oᵀd_rep = V_O ≥ 0.
2. **(b)** h(β) = V_O + e_OᵀCd_N^β, và |e_OᵀCd_N^β| ≤ ‖e_O‖_{W⁺}·√leak(β).
3. **(c) Rò rỉ.** leak′(β) = −2(Bd_N^β)ᵀG_β(Bd_N^β) ≤ 0, và leak(β) ≤ min(leak(0), ‖u_N‖²_H/β).
   - leak giảm chặt khi còn dương, và → 0 khi β → ∞.
4. **(d) Mức quên bậc hai.** Với bước θ − εd:
   L_KD(θ − εd) − L_KD(θ) = −ε g_Oᵀd + ½ε²[‖Cd‖²_W + ẽ_Oᵀ∇²f[d,d]] + O(ε³).
   - ∇²f[d,d] là đạo hàm cấp hai của output theo hướng d; bằng 0 khi tuyến tính hóa.
   - Dấu của số hạng bậc một là **âm**. `math.md` Prop 3 ghi dấu dương, đó là lỗi.
5. **(e) Chứng chỉ.** Với V_O > 0: nếu leak(β) ≤ ε\* thì h(β) ≥ 0.
   - Tập {β : leak(β) ≤ ε\*} là một khoảng [β_cert, ∞).
   - β_cert ≤ β_suf, và β_cert = 0 khi và chỉ khi leak(0) ≤ ε\*.
6. **(f) Nhân tử KKT.** Với V_O > 0, d\*(β_cert) là nghiệm duy nhất của
   min Φ(d)  với ràng buộc  ‖C(d − d_rep)‖²_W ≤ ε\*,
   và β_cert là nhân tử Lagrange duy nhất (đối ngẫu mạnh). Ngân sách ε\* **không có tham số**: đó là ngân sách lớn nhất mà bất đẳng thức Cauchy–Schwarz ở (b) *đủ* để bảo đảm h ≥ 0. Nó là điều kiện đủ, không phải "đúng bằng lượng phần sửa bù lại": trên toy, β_cert gấp trung vị 4.5 lần β nhỏ nhất cho h ≥ 0 trên lưới (§8.5).
7. **(g) Biên và trường hợp suy biến.**
   - (i) Nếu β_cert > 0 thì 0 ≤ h(β_cert) ≤ 2V_O.
   - (ii) Ở giới hạn cứng: e_OᵀCd_N^∞ = 0, nên h(∞) = V_O.
   - (iii) **V_O = 0 (với e_O ≠ 0).** Khi đó J_Eᵀz_O = 0, nên u_R = 0, d_rep = 0, và h(0) = g_OᵀGJᵀHu_N = z_OᵀK_Ez_N = (J_Eᵀz_O)ᵀJ_Eᵀz_N = **0 chính xác**. Vậy β = 0 có chứng chỉ (ĥ = 0 khi giải chính xác) và qua được chứng chỉ giảm loss vì ĝ = V ≥ 0 (T2a). Không cần bỏ qua. (Công thức ε\* = 0 ở (e) chỉ là điều kiện đủ; nó đòi giới hạn cứng, trong khi β = 0 đã cho h = 0.) Với β > 0, h(β) = e_OᵀCd_N^β có thể âm, nên quy tắc chọn β = 0.
   - (iv) **e_O = 0** (teacher trùng student trên logit cũ, hoặc c_old = 1 khi đó W ≡ 0 và e_O ≡ 0). Khi đó g_O = 0, nên h ≡ 0 với mọi d; còn ε\* = 0/0 không xác định. Quy tắc: β = 0, chứng chỉ KD hiển nhiên đúng ở bậc một. Ở bậc hai, KD đang ở cực tiểu theo output trên cửa sổ, nên mọi bước đều làm KD tăng một lượng ½ε²‖Cd‖²_W ≥ 0 (T3d); đây là giới hạn cố hữu, không phụ thuộc thuật toán.
8. **(h) h không đơn điệu.** h(β) = V_O + Σᵢ cᵢ/(1 + βλᵢ) trong cơ sở riêng suy rộng của (A₀, B), và các cᵢ có thể trái dấu. Bisection trên h **không hợp lệ**; bisection trên leak thì hợp lệ.
9. **(i) Phạt về đích 0 làm mất phần sửa** (dạng `math.md` §1/§6, đã bị thay). Với phạt β‖Cd‖²_W và toàn bộ u: |g_Oᵀd_β| ≤ ‖e_O‖_{W⁺}‖u‖_H/√β → 0.

*Chứng minh.*
- **(a)** L4 với r = ẽ_O: g_OᵀGJᵀH = z_Oᵀ, nên g_Oᵀd_rep = z_OᵀK_Ez_O.
- **(b)** g_Oᵀx = ẽ_OᵀJx = e_OᵀCx. Theo L5, e_O ∈ range(W) = range(Λᵀ), nên e_O = Λᵀw, với ‖w‖ = ‖e_O‖_{W⁺} khi chọn w nhỏ nhất. Vậy |e_OᵀCx| = |wᵀΛCx| ≤ ‖w‖·‖ΛCx‖.
- **(c)** dG_β/dβ = −G_βBG_β, nên (d_N^β)′ = −G_βBd_N^β, và leak′ = 2dᵀBd′.
  - Cận 1/β: β·leak(β) ≤ Φ_N(d_N^β) + β·leak(β) ≤ Φ_N(0) = ‖u_N‖²_H.
  - Nếu leak > 0 thì Bd ≠ 0, nên leak′ < 0.
- **(d)** Taylor của L_KD∘f: gradient là g_O, Hessian là JᵀOᵀWOJ + Σⱼ(ẽ_O)ⱼ∇²fⱼ.
- **(e)** h ≥ V_O − ‖e_O‖_{W⁺}√leak ≥ V_O − ‖e_O‖_{W⁺}√ε\* = 0. Tập đó là một khoảng vì leak không tăng và liên tục. β_suf thuộc tập đó theo (c).
- **(f)**
  - Theo T1(b), bài toán là min Φ_N(x) với xᵀBx ≤ ε\*.
  - Điều kiện Slater thỏa tại x = 0. Mục tiêu lồi chặt, nên nghiệm duy nhất và đối ngẫu mạnh.
  - KKT: x = d_N^β với β ≥ 0 và β(leak(β) − ε\*) = 0.
  - Nếu leak(0) ≤ ε\* thì β = 0. Ngược lại, β là nghiệm duy nhất của leak = ε\*, tức β_cert, theo (c).
- **(g)**
  - Tại β_cert > 0: leak = ε\*, nên |e_OᵀCd| ≤ V_O.
  - Ở giới hạn cứng: ΛCd_N^∞ = 0 và e_O = Λᵀw, nên e_OᵀCd_N^∞ = 0.
  - (iii) V_O = ‖J_Eᵀz_O‖² = 0 ⇒ J_Eᵀz_O = 0 ⇒ u_R = K_Ez_O = 0. Theo L4, g_OᵀGJᵀH = z_Oᵀ, nên h(0) = z_OᵀK_Ez_N = 0.
  - (iv) e_O = 0 ⇒ g_O = Cᵀe_O = 0 ⇒ h(β) = g_Oᵀd\*(β) = 0. Và χ² = ‖e_O‖²_{W⁺} = 0.
- **(h)** Chéo hóa đồng thời (A₀, B). Phản ví dụ số ở T3d-WIT: 62/300 bài (referee 2: 37/150 trên bộ sinh độc lập).
- **(i)** Cùng lập luận với (b) và (c), với u thay cho u_N. ∎

Kiểm:
- T3a: 2.2e-13; T3b: tỉ số với cận ≤ 0.875 (T3b-NEG: cùng cận nhưng dùng ‖e_O‖_W thì vi phạm 12265 lần);
- T3c: leak không tăng, 0 vi phạm; T3d: tỉ số ≤ 0.246;
- T3e: 0 vi phạm trên 6900 điểm lưới (T3e-WIT: VNA thuần làm KD tăng ở 102/300 bài, nên chứng chỉ không rỗng);
- T3f: 300/300; T3g: h(β_cert)/V_O ≤ 1.856; T3h: 5.1e-11;
- T3i (SLSQP): 1.0e-7 (T3i-WIT: nghiệm không ràng buộc vi phạm ngân sách, leak₀/ε* ≥ 6.66);
- T3j: dấu đúng, sai lệch bậc một 5.2e-2 → 5.2e-3 khi ε giảm 10 lần. Referee 2 kiểm thêm **hệ số** bậc hai ‖Cd‖²_W + ẽ_Oᵀf″[d,d] với autograd, có và không có BN: 1e-8.
- T11a–g: các trường hợp suy biến của (g) và quy tắc β (xem T11).
- (i) được kiểm trong `check_rep.py`: tốc độ sửa 0.79 → 3.4e-4 khi β đi từ 0 đến 1e4; referee đã xác nhận.

### T4. Cái giá về plasticity của β

**Định lý 4.** Giả thiết A2, A3.

1. **(a)** β ↦ Φ_N(d_N^β) không giảm. Đây là độ trung thành của phép chiếu phần học-mới, gồm cả damping.
2. **(b)** 0 ≤ Φ_N(d_N^β) − Φ_N(d_N^0) ≤ β(leak(0) − leak(β)) ≤ β·leak(0).
3. **(c)** Riêng độ khớp fit_β = ‖Jd_N^β − u_N‖²_H **không** nhất thiết đơn điệu. Khẳng định "fit tăng đơn điệu" của `math_rep` 4d là sai.

*Chứng minh.*
- **(a)** Điều kiện dừng: ∇Φ_N(x_β) = −2βBx_β, với x_β = d_N^β. Ngoài ra x′_β = −G_βBx_β. Vậy dΦ_N(x_β)/dβ = 2β(Bx_β)ᵀG_β(Bx_β) ≥ 0.
- **(b)** Vì x_β cực tiểu Φ_N + β·leak: Φ_N(x_β) + βleak(β) ≤ Φ_N(x₀) + βleak(0).
- **(c)** Phản ví dụ tường minh (từ referee): F = [[2.077, .563], [.563, .265]], B = [[12.70, −.027], [−.027, 1e-4]], JᵀHu_N = [.278, .320], μ = 1. Khi đó fit = −0.1434, −0.1443, −0.1451 tại β = 0, 0.05, 0.2: **giảm**. Trong khi Φ_N = −0.0875, −0.0872, −0.0860: tăng. ∎

Kiểm: T4a (0 vi phạm), T4b (tỉ số ≤ 0.500), T4c-WIT, T4d.

### T5. Giảm loss, quy tắc bỏ qua, không có γ

**Định lý 5.** Giả thiết A2, A3; với (e) và (f) thêm A6.

1. **(a) Cận chuẩn.** ‖d_rep‖ ≤ ‖u_R‖_H/√μ và ‖d_N^β‖ ≤ ‖u_N‖_H/√μ, với mọi β.
2. **(b)** gᵀd\*(β) ≥ V − ‖g‖_G·min(√(β·leak(0)), 2‖u_N‖_H).
3. **(c) Không có γ.** Xét biến thể u_γ = K_E(z_N + γz_O) và β = 0. Nếu z_N và z_O độc lập tuyến tính, thì gᵀd ≥ 0 với mọi K_E ⪰ 0 **khi và chỉ khi γ = 1**. γ = 1 chính là VNA hiện tại.
4. **(d) Bậc một, có quy tắc bỏ qua.** Nếu gᵀd̂ ≥ 0 thì với mọi λ ≥ 0, bước p = g + λd̂ thỏa gᵀp ≥ ‖g‖².
5. **(e) Bậc hai.** Đặt ρ := λ‖d̂‖/‖g‖. Nếu gᵀd̂ ≥ 0 thì
   L(θ − εp) ≤ L(θ) − ε‖g‖²(1 − ½L_sε(1+ρ)²).
   Vậy L giảm chặt khi 0 < ε < 2/(L_s(1+ρ)²). Không cần ρ < 1.
6. **(f) So với bước baseline bất kỳ** (kể cả momentum). Gọi δ là bước baseline. Khi đó
   L(θ+δ−ελd̂) − L(θ+δ) ≤ −ελgᵀd̂ + ελL_s‖δ‖‖d̂‖ + ½L_sε²λ²‖d̂‖²,
   và bất đẳng thức tương tự cho L_KD với ĥ và L_K. Vậy với ĥ ≥ 0 và gᵀd̂ ≥ 0, hiệu chỉnh không làm xấu đi ở bậc một. Để **cải thiện chặt** khi ε đủ nhỏ thì cần ĝ = gᵀd̂ > 0 (và ĥ > 0 cho L_KD); khi ĝ = 0 hoặc ĥ = 0 thì chỉ có "không xấu đi ở bậc một".
7. **(g) Không có bảo đảm vô điều kiện khi β > 0.** gᵀd\*(β_cert) < 0 có thể xảy ra; khi đó một bước có thể làm L tăng.

*Chứng minh.*
- **(a)** ‖G_βJᵀHu‖² ≤ ‖G_β‖·‖JᵀHu‖²_{G_β} ≤ μ⁻¹‖JᵀHu‖²_G ≤ μ⁻¹‖u‖²_H, dùng L2 rồi L1.
- **(b)**
  - gᵀ(d_rep + d_N^0) = V theo T2a. Ngoài ra d_N^0 − d_N^β = βG_βBd_N^0.
  - |gᵀβG_βBd_N^0| ≤ ‖g‖_{G_β}·√β‖ΛCd_N^0‖, theo L3; và ‖g‖_{G_β} ≤ ‖g‖_G.
  - Cận thứ hai: |gᵀG_βJᵀHu_N| ≤ ‖g‖_G‖u_N‖_H (L1, L2), áp cho β và cho 0.
- **(c)**
  - Theo L4, gᵀd = zᵀK_E(z_N + γz_O). Với γ = 1, gᵀd = zᵀK_Ez ≥ 0.
  - Với K_E = vvᵀ, ta có gᵀd = (vᵀz)(vᵀ(z_N + γz_O)). Biểu thức này âm với một v nào đó, trừ khi z_N + γz_O = c·z, c ≥ 0.
  - Vì z_N, z_O độc lập, điều đó buộc c = 1 = γ.
- **(d)** gᵀp = ‖g‖² + λgᵀd̂.
- **(e)** Bất đẳng thức descent L-smooth, với gᵀp ≥ ‖g‖² và ‖p‖ ≤ (1+ρ)‖g‖.
- **(f)** Áp L-smooth tại θ+δ, và ‖∇L(θ+δ) − g‖ ≤ L_s‖δ‖.
- **(g)** Phản ví dụ số ở T5c-WIT và T5d-NEG. ∎

Kiểm:
- T5a: tỉ số ≤ 0.235; T5b: vi phạm tối đa 3.2e-13;
- T5c-WIT: gᵀd\* < 0 ở 8/200 bài tại β_cert;
- T5d: kiểm bổ đề descent L-smooth trên bài toán bậc hai (slack tối thiểu 12.4 > 0) (T5d-NEG: khi gᵀd < 0, L tăng). Đây là kiểm bổ đề, không phải kiểm thuật toán; bản kiểm ở mức thuật toán là T15;
- T5e: tìm kiếm đối kháng (Adam trên v, K_E = vvᵀ, d tính qua solve thật) không tìm được gᵀd < 0 khi γ = 1: min 9.7e-18 (T5e-NEG: cùng tìm kiếm tìm được giá trị âm với γ ∈ {0, 2, 5}: −3.4e-4, −2.0e-4, −4.5e-3);
- T15a, T15b: kiểm ở mức thuật toán trên MLP thật với CE + KD.

### T6. Chọn neuron ảo A

**Định lý 6.** Giả thiết A2, A3, A4′ (mọi aⱼ nằm trong vùng mặt nạ cố định R_D của neuron đó; N_s dựng với mặt nạ D thực tế và, nếu có BN, với X̃ đã trừ trung bình), cố định β, S_x ≻ 0 như §0 (có ridge, L9). Đặt y_β := HJG_βg và M_S(β) := sym(N_zN_{z_O}ᵀ) + sym(N_{y_β}N_{z_N}ᵀ).

1. **(a)** Với mọi A (trong R_D, không BN): gᵀd\*(β) = tr(AᵀM_S(β)A). Có BN: gᵀd\*(β) = Σⱼ aⱼᵀM_Saⱼ/σ(aⱼ)², và với chuẩn hóa σ(aⱼ) = 1 (tức aⱼᵀS₀aⱼ = 1 với S₀ = X̃ᵀX̃/N) thì trở lại dạng vết (T14).
2. **(b)** Trên các tập S_x-trực chuẩn có tối đa r neuron (trong R_D), giá trị lớn nhất là tổng các trị riêng **dương** lớn nhất (tối đa r) của bài toán suy rộng (M_S, S_x), đạt tại các eigvec tương ứng nếu chúng nằm trong R_D. Nếu eigvec rơi ra ngoài R_D (mặt nạ đổi), tổng đó chỉ là cận trên của giá trị trong R_D; tối ưu toàn cục theo a là giả thiết A4-glob.
3. **(c)** Với β = 0: M_S = N_zN_zᵀ, nên tiny_V(z) **chính xác** là tối ưu.
4. **(d)** Với β > 0, đặt Δ := sym(N_{y_β−z}N_{z_N}ᵀ). Độ dưới tối ưu của tiny_V(z) không vượt quá 2Σ_{j≤r}|λⱼ(S_x^{-1/2}ΔS_x^{-1/2})|. Ngoài ra y_β − z = −βHJG_βBGg.
5. **(e) Quy tắc xen kẽ.** A₁ = argmax M_S(β₀) tối ưu *có điều kiện* theo β₀. Sau đó β được tính lại, nên **không** có khẳng định tối ưu đồng thời theo (A, β). Khi A được giữ qua nhiều lần áp dụng, cũng không có khẳng định tối ưu. Mọi chứng chỉ (T3, T5) đúng với **mọi** A.

*Chứng minh.*
- **(a)** Theo L4: gᵀd_rep = zᵀK_Ez_O và gᵀd_N^β = y_βᵀK_Ez_N. Với sᵀK_Et = Σⱼ aⱼᵀN_sN_tᵀaⱼ (A4′: trong vùng mặt nạ cố định, J_{aⱼ} tuyến tính theo aⱼ; có BN thì chia thêm σ(aⱼ)², xem T14), lấy phần đối xứng.
- **(b)** Đổi biến U = S_x^{1/2}A rồi áp Ky Fan; chỉ giữ các trị riêng dương.
- **(c)** Khi β = 0 thì y₀ = z, và sym(N_zN_{z_O}ᵀ) + sym(N_zN_{z_N}ᵀ) = N_zN_zᵀ.
- **(d)** M_S = N_zN_zᵀ + Δ. Đặt Q = N_zN_zᵀ. Ta có Q(A\*) − Q(A_t) ≤ 0 vì A_t tối ưu cho Q. Phần Δ được chặn bởi Ky Fan cho trị tuyệt đối. Công thức y_β − z = HJ(G_β − G)g.
- **(e)** Theo định nghĩa. ∎

Kiểm:
- T6a: 4.2e-12 (T6a-NEG: dùng N_zN_zᵀ khi β > 0 thì lệch ≥ 2.1e-3);
- T6b: không A ngẫu nhiên nào vượt tối ưu;
- T6c: 1.7e-12; T6d: tỉ số ≤ 0.121 (T6d-WIT: dưới tối ưu tới 91% khi β > 0).
- Empirical (`gaps_out.txt` G1f): quy tắc xen kẽ tăng gᵀd̂ trung vị 22%, và chứng chỉ giữ ở 80/80 bài.

### T7. GIR: đặt neuron vào slot (chết: có chứng chỉ; sống: chỉ đo)

Gọi r ∈ range(H) là residual mà GIR nhắm tới: r = ẽ_O trong CIL (ưu tiên 1 là hàm cũ), r = e trong B5. Các r này thuộc range(H) theo L5. Đặt t_r := −H⁺r, và cho mỗi neuron v: s_v := J_vᵀr_⊥, S_v := J_vᵀH_⊥J_v + μI.

Lợi dài hạn của việc thêm cột J_v được định nghĩa là gain(v | J) := q\*_r(J) − q\*_r([J, J_v]).

**Định lý 7.** Giả thiết A2, A3, A12. (f), (g), (h) cần thêm A5 *tại tham số lúc chèn*. Riêng phần "v̂ = tiny_V(r_⊥)" của (d) cần A4′. Việc diễn giải gain, ΔL_pred, ΔKD_pred như mức thay đổi loss về sau cần A10 (interpretive).

1. **(a) Hàm chiếu chung (phát biểu cấu trúc).** gain(v | J) = ½[E(J, t_r) − E([J, J_v], t_r)]. Còn VNA ở β = 0 chính là nghiệm của E(J, u). Hai thành phần dùng **cùng một dạng** hàm chiếu E (khoảng cách theo metric H, có damping μ, tới span của các cột tiếp tuyến), nhưng **không cùng thang đo**:
   - **đích:** u là một *vận tốc* do dung lượng ảo đóng góp (tỉ lệ với λ và ε khi áp dụng); t_r = −H⁺r là một *độ dời Newton* trong không gian output cho residual r. Hai đại lượng có ý nghĩa vật lý khác nhau, nên cùng một μ có tác dụng khác nhau trong hai bài toán;
   - **biến tối ưu:** VNA tìm hệ số trên span cố định, còn swap mở rộng chính span.
   Vì vậy (a) chỉ là phát biểu cấu trúc, không phải một hàm mục tiêu chung có đơn vị chung.
2. **(b) Dạng Schur, đúng với mọi a_v, b_v và mọi phi tuyến.** gain = ½s_vᵀS_v⁻¹s_v, với r_⊥ = r − z(r) = μ(HK+μ)⁻¹r = H_⊥H⁺r và H_⊥ = μR(M+μ)⁻¹R ⪰ 0. Hệ quả: gain > 0 khi và chỉ khi J_vᵀr_⊥ ≠ 0. Công thức chỉ dùng J_v, mà J_v = ∂f/∂B_v tại B_v = 0 được tính chính xác cho mọi đặc trưng (ReLU, BN): **không cần A4′** (T13f).
3. **(c) Tốc độ ban đầu sau khi hồi phục.** Tại cực tiểu của mạng sống x\* = −GJᵀr, gradient theo B_v bằng s_v.
4. **(d) Chọn v.**
   - ½‖s_v‖²/(λ_max(J_vᵀHJ_v) + μ) ≤ gain ≤ ½‖s_v‖²/μ. Cận dưới ℓ(v) := ½‖s_v‖²/(λ_max(J_vᵀHJ_v) + μ) tính được không cần giải hệ, và là **chứng chỉ** cho gain (T13g).
   - Chọn v̂ = argmax ℓ(v) trên một tập ứng viên hữu hạn 𝒱 bất kỳ. Khi đó gain(v̂) ≥ μ/(c_H + μ)·max_{v∈𝒱} gain(v), với c_H := max_{v∈𝒱} λ_max(J_vᵀHJ_v). Không cần A4′.
   - Dưới A4′ (trong một vùng mặt nạ): ‖s_v‖² = aᵀN_{r_⊥}N_{r_⊥}ᵀa (không BN), nên hướng tốt nhất theo ‖s_v‖ là **tiny_V với cotangent r_⊥**; với BN, là eigvec suy rộng của (N_{r_⊥}N_{r_⊥}ᵀ, S₀) với thống kê đã trừ trung bình (T14).
   - **Quy tắc dấu và bias cho ReLU.** Eigvec chỉ xác định đến dấu, mà φ(−a) ≠ −φ(a) với ReLU, và một bias sai có thể làm φ ≡ 0 trên cửa sổ (gain = 0). Vì vậy 𝒱 gồm {σa_k + b : σ ∈ {±1}, b ∈ {0, −median(σXa_k)}} với a_k là vài eigvec đầu; bỏ ứng viên có φ ≡ 0. Bảo đảm ở ý trên áp cho 𝒱 này. Tối ưu trên toàn bộ không gian (a, b) là giả thiết A4-glob.
5. **(e) r_⊥ là phần bù theo metric H, không phải phần bù Euclid.** Khi μ → 0, r_⊥ tiến tới phép chiếu xiên của r lên null(HK), dọc theo H·range(J). Nó trùng với phần dư Euclid (I − JJ⁺)r chỉ khi H ∝ I, ví dụ MSE.
   - Tương ứng với bước chọn atom của OMP chỉ **chính xác** trong trường hợp H = I, μ → 0. Ngoài trường hợp đó chỉ là phép tương tự.
   - r_⊥ = e_O − z_O chính là đại lượng đúng theo (b). Nó **không** phải xấp xỉ của một "r_⊥ chính xác" nào.
6. **(f) Slot chết (A5 tại tham số lúc chèn).** Khi bỏ slot u và chèn v với B_v = 0:
   - hàm trên cửa sổ không đổi, **chính xác**;
   - giá bỏ u bằng 0, vì các cột bằng 0 không đổi q\*;
   - không có rò rỉ sang logit cũ tại lúc chèn;
   - các cột của a_v bằng 0 tại B_v = 0.
   Ngoài cửa sổ, hàm đổi một lượng −f_u(x) tại những điểm u kích hoạt (T12e-WIT).
   **A5 phải được kiểm lại sau bước 5.** Bước hiệu chỉnh đổi các trọng số phía trước, nên một đơn vị chết tại θ có thể sống tại θ′ (T13a-WIT: 18/40). Khi đó chèn sẽ làm đổi hàm (T13a-NEG: 0.108); quy tắc "kiểm lại, chỉ chèn nếu còn chết" giữ hàm chính xác (T13a: 0.0). Slot đã sống lại được xử lý như slot sống ở (i).
7. **(g) Swap không giao hoán với VNA.** Sau khi chèn, K′ = K + J_vJ_vᵀ: không gian tiếp tuyến lớn lên, đúng mục đích của swap. Vì vậy d̂ được tính và áp dụng **trước** swap.
   - d̂ không đụng tới tham số của slot chết: cột bằng 0 cho (d̂)_u = 0.
   - Lần áp dụng kế tiếp dùng K′.
8. **(h) Quyết định swap vào slot chết không cần chân trời.** Với slot chết, lợi ròng bằng gain(v | J) ≥ 0, và > 0 khi và chỉ khi s_v ≠ 0. Vậy swap vào mọi slot chết (tối đa S), chọn v theo (d).
   - A10 chỉ cần để lý giải vì sao *lợi dài hạn* (cotangent r_⊥) là tiêu chí đúng.
   - Nếu mạng sống không hồi phục (x = 0), tiêu chí tốc độ ban đầu là ‖J_vᵀr‖², tức cotangent r.
   - r_⊥ được tính tại θ, còn việc chèn xảy ra tại θ′ = θ − ελd̂. Việc này không ảnh hưởng tới các đại lượng chính xác (bảo toàn hàm, j_u); nó chỉ làm gain dự báo lệch một lượng O(ελ‖d̂‖) (có thể tính lại r_⊥ ở lần áp dụng sau).
   - **Lưu ý về CIL (r = ẽ_O).** gain(v | J) với r = ẽ_O là mức giảm của mô hình *KD* khi chỉ hồi phục trên KD (ưu tiên 1). A10 giả thiết hồi phục trên L, nên diễn giải "gain = mức giảm KD về sau" **không** suy ra từ A10. Đại lượng nhất quán với A10 là ΔKD_pred (§0) tính với J′ = [J, J_v]. Các phát biểu chính xác của (f) (bảo toàn hàm, không rò rỉ tại lúc chèn) không bị ảnh hưởng. Trong CIL nhánh này không kích hoạt (0% đơn vị chết, (j)), nên đây chỉ là ghi chú phạm vi; cần log ΔKD_pred nếu nó kích hoạt.
9. **(i) Slot sống: đo chính xác, không chứng nhận.** Với slot sống u, neuron mới v (B_v = 0) và một loss ℓ ∈ {L, L_KD}:
   - **(i.1) Thay đổi tại lúc chèn là chính xác:** ℓ(sau swap) − ℓ(trước swap) = j_u^ℓ = ℓ(f − f_u) − ℓ(f) trên cửa sổ, tính bằng một forward, không cần hằng số nào (T12c, [structural]). Đây là một phép **đo**: nó không cam kết gì về dấu.
   - **(i.2) Cận Taylor có hằng số dạng đóng, chặt:** |j_u^ℓ − (−rᵀf_u + ½f_uᵀH_ℓf_u)| ≤ (L₃/6)‖f_u‖³, với r = ∇ℓ(f), H_ℓ = ∇²ℓ(f) và L₃ ≤ c₃(1 + T⁻³)/N (L10). Dùng để lọc ứng viên trước khi chạy forward (T12b; cấu hình gần xấu nhất đạt 1.0000 cận).
   - **(i.3) Dự báo dưới A10 (thay cho Net của v2).** Mô hình nhất quán với A10 là ΔL_pred và ΔKD_pred của §0: bước hồi phục là x_L trên **L** (không phải trên KD), và mọi đại lượng sau swap (e′, H′, e_K′, H_K′, J′) tính **tại f − f_u**. Lý do chọn điểm khai triển là T16(a): mô hình khai triển tại đầu ra thật sau swap có sai số bậc ba (L₃/6)‖Δ′‖³, còn dùng H tại f thì có thêm một số hạng bậc hai ≤ ½L₃‖f_u‖‖Δ′‖².
   - **(i.4) Bao hàm trường hợp chết:** với slot chết, j_u = 0, f_u = 0, J′ = [J, J_v], nên ΔL_pred = m_L(f, [J, J_v]) − m_L(f, J); và hiệu mô hình q\* tính từ chính Jacobian của mạng sau swap bằng −gain(v | J) dạng Schur (T12d: 1.8e-14 trên mạng ReLU thật; T12d-NEG).
   - **(i.5) Chứng chỉ tức thời (chính xác).** Nếu j_u^L ≤ 0 và j_u^KD ≤ 0 thì tại lúc chèn, L và L_KD trên cửa sổ **không tăng**. Đây là điều duy nhất được chứng nhận cho swap slot sống (T16e: 0/37 vi phạm, [structural]).
   - **(i.6) Độ tin cậy của dự báo (empirical, T16).** Trên mạng ReLU thật (thí nghiệm của referee 3, 198 cặp): ở μ_rel = 10 (gần tuyến tính) dấu của ΔL_pred, ΔKD_pred khớp với bước GN thực hiện ở 194/198 và 193/198 cặp (tương quan 0.97, 0.99); ở μ_rel = 1e-2 (damping dùng thật) chỉ khớp 87/198 và 101/198 (tương quan −0.17, 0.07); so với hồi phục thật bằng 200 bước GD thì khớp kém ngay cả ở μ_rel = 10 (131/198, 116/198). **Kết luận: swap slot sống là "measured, not certified", mặc định TẮT.**
   - **Phạm vi:** khác slot chết, swap slot sống **không** bảo toàn hàm; nó làm đổi hàm trên cửa sổ đúng một lượng −f_u. Ngoài cửa sổ không có chứng chỉ.
10. **(j) GIR trong CIL (empirical).** Với ResNet32 có BN (CIL), tỉ lệ đơn vị dormant đo được là 0 (đây là số đo, không phải hệ quả toán học; referee 2 quan sát điều tương tự trên MLP có BN). Vì vậy nhánh slot chết (a) không kích hoạt trong CIL, và nhánh slot sống (b) mặc định tắt vì dự báo của nó không đáng tin (i.6). **Hệ quả: với cấu hình mặc định, GIR không hoạt động trong CIL**; nếu bật LIVE_SWAP = measured thì đó là một thí nghiệm, phải log ΔL_pred, ΔKD_pred và kết quả thực. Trong B5, nhánh slot chết là cơ chế chính.

*Chứng minh.*
- **(a)**
  - Vì r ∈ range(H), ta có HH⁺r = r. Do đó ‖𝒥x − t_r‖²_H = xᵀ𝒥ᵀH𝒥x + 2xᵀ𝒥ᵀr + rᵀH⁺r.
  - Suy ra E(𝒥, t_r) = 2q\*(𝒥) + rᵀH⁺r. Lấy hiệu.
  - Phần VNA: T1(a) ở β = 0.
- **(b)**
  - Nghịch đảo khối cho [J, J_v]: b̃ᵀÃ⁻¹b̃ = bᵀA⁻¹b + (b_v − J_vᵀHJGb)ᵀS_v⁻¹(·), với b = Jᵀr, b_v = J_vᵀr.
  - b_v − J_vᵀHJGJᵀr = J_vᵀ(r − z(r)).
  - Theo L1: H_⊥ = R(I − M(M+μ)⁻¹)R = μR(M+μ)⁻¹R.
  - Đồng nhất r − z(r) = μ(HK+μ)⁻¹r suy từ push-through: z(r) = HK(HK+μ)⁻¹r.
- **(c)** ∇_{B_v}q = J_vᵀ(r + HJx\*) = J_vᵀ(r − z(r)).
- **(d)** μI ⪯ S_v ⪯ J_vᵀHJ_v + μI. Khi đó gain(v̂) ≥ ½‖s_v̂‖²/(c_H+μ) ≥ ½‖s_{v\*}‖²/(c_H+μ) ≥ μ/(c_H+μ)·gain(v\*).
- **(e)**
  - HK đồng dạng (trên range) với RKR ⪰ 0, nên chéo hóa được với trị riêng không âm.
  - μ(HK+μ)⁻¹ → phép chiếu lên null(HK) dọc theo range(HK) = H·range(J).
  - Nếu H = I: null(K) = range(J)^⊥ và range(HK) = range(J), nên phép chiếu là trực giao.
- **(f)** Đóng góp của v là h_v(x)B_vᵀ = 0. Đạo hàm theo a_v tỉ lệ với B_v, nên bằng 0. Cột của u bằng 0 trên cửa sổ.
- **(g)** Cột mới là J_v; các cột của a_v bằng 0.
  - Với cột u bằng 0, F có hàng và cột u bằng 0, nên G_{uu} = μ⁻¹I, các khối chéo bằng 0, và (JᵀHy)_u = 0.
- **(h)** Từ (b) và (f).
- **(d), phần cận và chọn trên 𝒱.** ℓ(v̂) ≥ ℓ(v\*) với v\* tốt nhất trong 𝒱, và ℓ(v\*) = ½‖s\*‖²/(λ\*+μ) ≥ (μ/(λ\*+μ))·½‖s\*‖²/μ ≥ (μ/(c_H+μ))·gain(v\*). Kết hợp gain(v̂) ≥ ℓ(v̂).
- **(f), phần kiểm lại A5.** Chứng minh của (f) chỉ dùng f_u = 0, J_u = 0 tại tham số lúc chèn; nếu A5 không đúng tại θ′ thì f_u(θ′) ≠ 0 nói chung và hàm đổi đúng một lượng −f_u(θ′).
- **(i.1)** Sau khi bỏ u và chèn v với B_v = 0, đóng góp của v bằng 0, nên output là f − f_u (theo định nghĩa của f_u bằng forward). Do đó ℓ(sau) − ℓ(trước) = ℓ(f − f_u) − ℓ(f).
- **(i.2)** Taylor bậc ba của ℓ dọc đoạn [f, f − f_u] với phần dư Lagrange (1/6)D³ℓ(ξ)[−f_u]³, cộng L10.
- **(i.3)** Dưới A10, sau swap mạng ở đầu ra f − f_u (theo (i.1)) và đi một bước x′_L tính từ đúng trạng thái đó; không swap thì đi x_L từ f. Giá trị ℓ sau hồi phục, trong chế độ tuyến tính, là ℓ(f_state + J_state x_state), và khai triển Taylor bậc hai tại f_state cho ℓ(f_state) + m_ℓ(f_state, J_state) với phần dư ≤ (L₃/6)‖J_state x_state‖³ (L10). Lấy hiệu hai trạng thái và dùng ℓ(f − f_u) − ℓ(f) = j_u^ℓ. Lựa chọn điểm khai triển: T16(a).
- **(i.4)** Thay f_u = 0, J₋ᵤ = J vào (i.3); phần q\* là T7(b) vì Jacobian của mạng sau swap là [J, J_v] (các cột của a_v bằng 0 tại B_v = 0, (g)).
- **(i.5)** Từ (i.1). ∎

Kiểm:
- T7a: 3.4e-14 (T7a-NEG: dùng đích −r thì lệch ≥ 0.99);
- T7b: 4.5e-11 (T7b-NEG, chạy riêng với μ_rel = 0.3: s = J_vᵀr lệch tối thiểu 2.5e-3, trung vị 1.95; dung sai phép kiểm dương là 1e-8);
- T7c: 3.8e-12; T7d: 3.6e-11; T7e: tỉ số ≤ 0.9989; T7f: v̂ = argmax ℓ(v) (đúng quy tắc của bước 6), hệ số bảo đảm μ/(c_H + μ) giữ trong mọi trial (tỉ số chuẩn hóa tối thiểu 4.98 ≥ 1); T7f-WIT: argmax ℓ không phải argmax gain ở 32/60 trial, nên điều được khẳng định là hệ số, không phải tính tối ưu;
- T7g: |cos| trung vị giữa phần bù xiên và phần bù Euclid là 0.763 khi H ≠ I; T7h: 1.4e-10 khi H = I, μ → 0;
- T7i: hàm giữ nguyên (0.0); T7j: K′ = K + J_vJ_vᵀ (1.6e-16) (T7j-WIT: bỏ một đơn vị sống làm đổi hàm, 0.705); T7l: (d̂)_u = 0 chính xác. Lưu ý: T7l dùng H = I và đích ngẫu nhiên, nên chỉ là kiểm cấu trúc (cột bằng 0 ⇒ thành phần bằng 0), không phải kiểm trên đúng d̂ của VNA;
- T7k: phần dư bậc ba, tỉ số 7.93 và 7.96 mỗi lần ‖f_u‖ giảm một nửa (T7k-NEG: dấu cũ của `unified` Hệ quả 3c bị loại, lệch ≥ 2.0);
- Slot sống, nhiều slot, ReLU, BN: T12, T13, T16, T17.
- **Empirical cho việc chọn v:**
  - Toy (`gaps_out` G4n, ứng viên chuẩn hóa S_x): regret trung bình 1.8% với cotangent r_⊥, 57% với e, 61% với ngẫu nhiên.
  - Toy hai task, 40 seed (`repair_swap`): khi có slot chết, chọn theo r_⊥ thắng không-swap 72% trên CE lớp cũ. Nhưng CE trung bình của nó **bằng** cách chọn theo e (0.167); lợi thế chỉ nằm ở tỉ lệ thắng.

### T8. Quỹ đạo nhiều bước (cận Grönwall rời rạc)

**Định lý 8.** Giả thiết A7 (tham chiếu **tự trị** trong không gian output: một xấp xỉ kernel đóng băng của mô hình grow, không phải một mạng grow thật). Gọi x_m = f(θ_m) sau m bước vĩ mô độ dài τ của thuật toán, và quỹ đạo tham chiếu y_{m+1} = y_m + τU(y_m), y₀ = x₀. Đặt δ_m := ‖(x_{m+1} − x_m)/τ − U(x_m)‖, là defect thực hiện được, đo được lúc chạy. Khi đó

‖x_m − y_m‖ ≤ Σ_{j<m}(1 + τL_U)^{m−1−j}·τδ_j.

*Chứng minh.* Đặt e_m = ‖x_m − y_m‖. Ta có e_{m+1} ≤ (1 + τL_U)e_m + τδ_m. Quy nạp. ∎

δ_m gồm ba phần: (i) defect chiếu λ‖Jd − u‖, chính là đại lượng VNA cực tiểu; (ii) độ trôi kernel; (iii) sai số rời rạc.

**Cận chỉ dùng phần (i) là sai.** Trên MLP tanh không lazy, defect thực hiện được trung bình 0.73, trong khi phần chiếu chỉ 0.02; cận chỉ theo phần chiếu bị vi phạm với tỉ số 2.7 (T8a-NEG).

→ Chỉ khẳng định **tối ưu chiếu cho từng bước**. Khẳng định "quỹ đạo gần mô hình grow" chỉ được đưa ra khi δ_m đo được là nhỏ.

Kiểm: T8a (tỉ số ≤ 1.0000), T8a-NEG (tỉ số 2.7; defect thực hiện trung bình 0.728 so với phần chiếu 0.020). Phép kiểm T8a chỉ kiểm bất đẳng thức quy nạp (chứng minh hiển nhiên); giá trị của nó nằm ở T8a-NEG.

### T9. Solve không chính xác

**Định lý 9.** Giả thiết A8.

1. **(a)** ‖x̂ − x‖ ≤ ρ‖b‖/μ. Warm start không ảnh hưởng, **miễn là** ρ đo so với ‖b‖.
2. **(b) Chứng chỉ chính xác trên vector đã áp dụng.** ĥ = g_Oᵀd̂, ĝ = gᵀd̂ và leak(d̂_N) = ‖ΛCd̂_N‖² được tính trực tiếp, không qua solve:
   - g_O là một backward;
   - Cd̂ là một JVP;
   - Λ là khối theo mẫu.

   Vậy với quy tắc chặt (τ = 0), T5(d–f) đúng cho bước thực sự áp dụng bất kể sai số của solver. Với quy tắc có dung sai của bước 4 (v3), T5(d–f) chỉ đúng **sai khác tối đa λτ_g (cho L) và ελτ_h (cho KD bậc một)**, với τ từ T9(e); hai số hạng này → 0 khi ρ₁, ρ₂ → 0.
3. **(c) Kiểm cos.** Nếu cos(d̂, d\*) ≥ c thì |g_Oᵀd̂/‖d̂‖ − g_Oᵀd\*/‖d\*‖| ≤ ‖g_O‖√(2(1 − c)). Với c = 0.99, độ suy giảm ≤ 0.141‖g_O‖ trên mỗi đơn vị chuẩn.
4. **(d) Chuỗi d_rep (hai lần giải).** ‖d̂_rep − d_rep‖ ≤ (‖JᵀHK_EHJ‖ρ₁‖g_O‖/μ + ρ₂‖b̂₂‖)/μ.
5. **(e) Quy tắc bỏ qua có dung sai (v3).** Xét chuỗi hai lần giải của d\* ở một β cố định: x̂₁ ≈ A_β⁻¹b₁ với b₁ = Jᵀe (residual tương đối ρ₁), ẑ = HJx̂₁, û = K_Eẑ, b₂ = JᵀHû (tính chính xác từ û), d̂ ≈ A_β⁻¹b₂ (residual ρ₂). (Ở β = 0, đây đúng là d\*(0) = GJᵀHK_Ez; với β > 0 và phân rã d_rep + d_N^β, áp dụng cho từng chuỗi rồi cộng.) Đặt
   Δ̂ := ‖JᵀHK_E‖‖HJ‖ρ₁‖b₁‖/μ² + ρ₂‖b₂‖/μ,  τ_h := ‖g_O‖Δ̂,  τ_g := ‖g‖Δ̂.
   Khi đó:
   - ‖d̂ − d\*‖ ≤ Δ̂, nên |ĥ − h\*| ≤ τ_h và |ĝ − g\*| ≤ τ_g, với h\* = g_Oᵀd\*, g\* = gᵀd\*.
   - **Không bỏ nhầm:** nếu h\* ≥ 0 (bảo đảm bởi T3e cho β có chứng chỉ, với solve chính xác) và g\* ≥ 0 (chỉ bảo đảm ở β = 0; với β > 0 thì g\* < 0 có thể xảy ra, T5(g), và khi đó bước bị bỏ qua là đúng, không phải bỏ nhầm), quy tắc "bỏ qua khi ĥ < −τ_h hoặc ĝ < −τ_g" không bao giờ bỏ qua. Đặc biệt khi V_O = 0 (h\* = 0 chính xác), quy tắc chặt "ĥ < 0" bỏ nhầm khoảng một nửa số bước (T18b-WIT: 15/30), còn quy tắc có dung sai không bỏ nhầm (T18c: 0/30).
   - **Bảo đảm khi áp dụng:** bước được áp dụng có g_Oᵀd̂ ≥ −τ_h, nên mức tăng KD bậc một do hiệu chỉnh không quá ελτ_h; tương tự cho L với τ_g. Cả hai → 0 khi ρ₁, ρ₂ → 0. Với solve chính xác, τ = 0 và quy tắc trở về quy tắc chặt; trong trường hợp đó vẫn nên dùng một dung sai cỡ sai số làm tròn (T11a đo h(0) chuẩn hóa 1.6e-14 khi V_O = 0).

*Chứng minh.*
- **(a)** x̂ − x = A⁻¹(Ax̂ − b), và ‖A⁻¹‖ ≤ μ⁻¹.
- **(b)** Theo định nghĩa.
- **(c)** Với vector đơn vị, ‖u₁ − u₂‖² = 2 − 2cos; cộng Cauchy–Schwarz.
- **(d)** (a) hai lần, cộng bất đẳng thức tam giác.
- **(e)** Theo (a): ‖x̂₁ − x₁‖ ≤ ρ₁‖b₁‖/μ (vì A_β ⪰ μI). Do đó ‖ẑ − z‖ ≤ ‖HJ‖ρ₁‖b₁‖/μ. Gọi d̃ := A_β⁻¹b₂ (nghiệm chính xác của lần giải thứ hai với vế phải đã tính). Thì ‖d̃ − d\*‖ = ‖A_β⁻¹JᵀHK_E(ẑ − z)‖ ≤ ‖JᵀHK_E‖‖ẑ − z‖/μ, và ‖d̂ − d̃‖ ≤ ρ₂‖b₂‖/μ theo (a). Cộng lại được Δ̂. Hai hệ quả còn lại là Cauchy–Schwarz và định nghĩa của quy tắc. ∎

Kiểm:
- T9a: tỉ số ≤ 0.727 (T9a-NEG: dừng theo residual ban đầu, khởi đầu xa, thì vi phạm cận gấp 3.2e5 lần);
- T9b: không vi phạm (vi phạm lớn nhất −7.7e-4 < 0);
- (d) kiểm trong `gaps_out` G6c: tỉ số ≤ 1.5e-3, đúng nhưng lỏng;
- (e): T18a (0/60 vi phạm của ‖d̂ − d\*‖ ≤ Δ̂ và |ĥ − h\*| ≤ τ_h), T18a-NEG (bỏ hệ số 1/μ thì vi phạm 26/60), T18b-WIT, T18c.

### T10. Phạm vi áp dụng

**Định lý 10.**

1. **(a) BN train mode.** Các đồng nhất và chứng chỉ của T1–T5, T7(a–c, e–i), T8, T9, T11 và T12 chỉ dùng J, H, W của loss trên cửa sổ, nên đúng **chính xác** cho BN train mode (J dày giữa các mẫu, H vẫn khối chéo), miễn là A1 thỏa. **Ngoại lệ:** tính tối ưu của bước chọn trong T6 và T7(d) (ý "tiny_V") cần A4′; với BN theo sau neuron ảo, A4′ có dạng riêng (thống kê đã trừ trung bình, σ(a) tự chuẩn hóa) và được chứng minh ở T14.
   - Đại lượng tính ở eval mode (running stats) dự báo **sai** thay đổi của loss train-mode.
2. **(b)** Loại tham số BN khỏi S: mọi định lý vẫn đúng với J_S, và range(K_S) ⊆ range(K).
3. **(c) A4 (tuyến tính toàn cục) không đúng khi BN đứng sau kênh ảo.** BN train mode làm đặc trưng bất biến theo tỉ lệ: φ(2a) = φ(a). Thay bằng A4′ (vùng mặt nạ cố định), dưới đó bài toán chọn là bài toán Rayleigh suy rộng với thống kê đã trừ trung bình (T14). Tối ưu toàn cục (qua các vùng mặt nạ) vẫn là giả thiết. Các chứng chỉ (T3, T5, T9), công thức gain chính xác (T7b) và việc bảo toàn hàm khi chèn với B = 0 (T7f) vẫn chính xác.
4. **(d) Momentum.**
   - Cộng ngoài buffer: T5(f) đúng, không phụ thuộc ν.
   - Ghi d̂ vào buffer (đưa d̂ vào như gradient, hoặc cộng vào buffer): d̂ bị phát lại với tổng trọng số 1/(1−ν) (bằng 10 khi ν = 0.9) tại các điểm nó không được tính, nên dấu của nó không còn được chứng chỉ. Referee 2 kiểm với `torch.optim.SGD` thật: đưa vào như gradient cho 10.000, cộng vào buffer sau khi dùng cho 9.000 = ν/(1−ν).
   - **Swap:** trạng thái optimizer của slot được đặt về 0 (A9). Khi đó, với momentum thường (nesterov = False, dampening = 0, như PyCIL), bước đầu tiên của B_v đúng bằng −lr·∇_{B_v}L, và của a_v đúng bằng −lr(∇_{a_v}L + ω a_v) với ∇_{a_v}L = 0 tại B_v = 0 (T13b). Với nesterov thì hệ số là (1 + ν), với dampening δ (buffer bằng 0, không phải None) thì là (1 − δ) (T13b2); cả hai vẫn là bội dương của gradient, không có số hạng momentum cũ. Không đặt lại thì momentum cũ đẩy B_v và a_v đi mà không có gradient (T13b-NEG). Với Adam: đặt hai moment của các phần tử đó về 0; vì bộ đếm bước dùng chung cho cả tensor, bước đầu tiên là −lr·D·g với D đường chéo dương, nên vẫn đi theo hướng giảm ở bậc một, chỉ khác biên độ.
5. **(e) Weight decay.** Các bảo đảm áp cho L và L_KD trên dữ liệu. Đối với mục tiêu có regularization, hiệu chỉnh thêm một số hạng −ελω θᵀd̂; cần log số hạng này.
6. **(f) Augmentation.** Chỉ có bảo đảm cho loss thực nghiệm trên đúng các mẫu đã augment trong cửa sổ.

*Chứng minh.*
- (a), (b): các định lý chỉ dùng tính chất đại số của (J, H, W) tại θ.
- (c): BN train mode chuẩn hóa ((Xa − mean)/std); thay a bằng 2a không đổi giá trị.
- (d): T5(f); số hạng phát lại là Σ_k ν^k.
- (e), (f): theo định nghĩa. ∎

Kiểm:
- T10a: 8.4e-14; T10b: sai phân bậc một 2.0e-2 → 2.0e-3 (T10b-NEG: dùng eval mode lệch 4.87 lần);
- T10c: 3.6e-15; T10d: |φ(2a) − φ(a)| = 4.7e-5, |φ(2a) − 2φ(a)| = 1.98;
- T10e: tỉ số 0.9871 → 0.99872;
- T10f (dùng `torch.optim.SGD` thật, momentum 0.9): cộng ngoài buffer dịch chuyển đúng lr·d (tỉ số 1.000000); T10f-NEG: đưa d qua buffer thì bị phát lại 10.0000 lần.

### T11. Các trường hợp suy biến của quy tắc β

**Định lý 11.** Giả thiết A2, A3. Quy tắc β ở bước 3 (các nhánh a–c) cho một β mà, với solve chính xác, h(β) ≥ 0 và gᵀd\*(β) được kiểm ở bước 4. Cụ thể:
1. **(a)** e_O = 0 (kể cả c_old = 1): β = 0, h ≡ 0 (T3(g)(iv)).
2. **(b)** V_O = 0, e_O ≠ 0: β = 0, h(0) = 0 (T3(g)(iii)).
3. **(c)** Ngược lại: nếu leak(0) ≤ ε\* thì β = 0; nếu không, bisection trên log β trong [β_suf·10^{−k_lo}, β_suf] trả về một đầu mút trên β đã kiểm leak(β) ≤ ε\*, nên h(β) ≥ 0 theo T3(e). Khoảng này luôn hợp lệ vì leak(β_suf) ≤ ε\* (T3(c, e)); không cần đầu mút 0 trên thang log.

*Chứng minh.* (a), (b) là T3(g)(iii, iv). (c): T3(e) nói mọi β với leak(β) ≤ ε\* đều có chứng chỉ; bisection chỉ trả về điểm đã được kiểm; tính đơn điệu của leak (T3c) bảo đảm khoảng [β_cert, ∞) chứa β_suf. ∎

Kiểm: T11a (h(0) chuẩn hóa 1.6e-14 khi V_O = 0), T11b (gᵀd = V, lệch 1.0e-13), T11c-WIT (60/60 bài V_O = 0 có leak(0) > 0, nên quy tắc v1 vứt bỏ các bước an toàn), T11d (e_O = 0: g_O = 0, χ² = 0, nhánh 'eO=0'), T11e (c_old = 1: W = 0, e_O = 0), T11g (bisection log 12 bước: 0 kết quả sai trên 120 bài).

### T12. Slot sống: các đại lượng chính xác và cận L₃

Nội dung là T7(i) và L10. Kiểm (v3: các phép kiểm L₃ có độ mạnh, tức bắt được hằng số sai):
- T12a: với hằng số chặt c₃ = 0.2722, cấu hình hai điểm đạt **1.0000** cận cho CE và cho KD ở T = 1, 2, 4, dò ngẫu nhiên (kể cả softmax bão hòa) tối đa 0.886 cận. Phép kiểm là hai phía: đúng (mọi tỉ số ≤ 1) **và** chặt (cấu hình đạt ≥ 0.999);
- T12a-NEG: cùng phép kiểm hai phía loại hằng số đột biến (chuẩn hóa T⁻¹ cho KD: hỏng tính chặt; hằng số ×0.1: hỏng tính đúng); T12a-WIT: hằng số 0.708 của v2 đúng nhưng lỏng (tỉ số 0.384);
- T12a2: CE + KD, 6 mẫu, tỉ số tối đa 0.576 so với c₃(1 + T⁻³)/N; hai phía: cấu hình thẳng hàng (N = 1, T = 1) đạt 1.0000; T12a2-NEG: chuẩn hóa max(1, T⁻³) bị vi phạm (2.000);
- T12b: phần dư Taylor của j_u không vượt (L₃/6)‖f_u‖³ với L₃ = c₃(1 + T⁻³)/N, kiểm tính đúng ở T = 1, 2, 4 (tỉ số tối đa 1.0000, 0.370, 0.584); ở T = 1 trên cấu hình mà softmax của CE và KD trùng nhau, tỉ số đạt 1.0000 (cận đạt); T12b-NEG: bỏ số hạng KD khỏi L₃ thì bị vi phạm (tỉ số 2.000); T12b-NEG2: khẳng định "j_u đúng bằng phần bậc hai" sai khi f_u lớn;
- T12c [structural]: trên MLP ReLU, swap một slot sống gần chết (3 mẫu kích hoạt) làm đổi loss và KD đúng bằng j_u (sai lệch 0.0); đúng theo cách dựng f_u bằng forward;
- T12d (thay đồng nhất hiển nhiên của v2): trên mạng ReLU thật có đơn vị chết, hiệu mô hình q\* tính từ chính Jacobian của mạng sau swap bằng −gain dạng Schur dựng chỉ từ φ(a_v): sai lệch 1.8e-14; T12d-NEG: dùng s = J_vᵀe (không lấy phần bù) thì lệch ≥ 1.53;
- T12e-WIT: một đơn vị chết trên cửa sổ có thể kích hoạt ngoài cửa sổ (0 trên cửa sổ, 5/50 ngoài).

### T13. GIR nhiều slot, trạng thái optimizer, ReLU

**Định lý 13.** Giả thiết A2, A3. Với 𝒥 bất kỳ và một khối cột mới V (của một neuron v), đặt S := VᵀH_⊥(𝒥)V + μI và s := Vᵀr_⊥(𝒥).
1. **(a) Cộng dồn.** Với các slot điền lần lượt v₁, …, v_k: gain({v₁, …, v_k} | J) = Σⱼ gain(vⱼ | [J, J_{v₁}, …, J_{v_{j−1}}]).
2. **(b) Cập nhật hạng q (deflation chính xác).**
   r_⊥([𝒥, V]) = r_⊥(𝒥) − H_⊥(𝒥)V S⁻¹s,  H_⊥([𝒥, V]) = H_⊥(𝒥) − H_⊥(𝒥)V S⁻¹VᵀH_⊥(𝒥).
   Vì vậy số hạng thứ j trong (a) là ½sⱼᵀSⱼ⁻¹sⱼ, tính với phần bù đã cập nhật.
3. **(c) Quy tắc rẻ có cận.** Với mọi tập slot 𝒮: gain(𝒮 | J) ≥ max_{v∈𝒮} gain(v | J). Vì vậy nếu không cập nhật phần bù (để tiết kiệm q + 1 lần giải mỗi slot), lợi chung vẫn ít nhất bằng lợi của slot tốt nhất.
4. **(d) Trạng thái optimizer.** Với SGD momentum thường (momentum ν, weight decay ω, nesterov = False, dampening = 0), sau khi đặt trạng thái của slot bằng 0: bước đầu của B_v là −lr·∇_{B_v}L và của a_v là −lr·ω·a_v (vì ∇_{a_v}L = 0 tại B_v = 0). Nếu không đặt lại, có thêm số hạng −lr·ν·m_cũ, tức một dịch chuyển không có gradient tương ứng. **Phạm vi:** với nesterov = True, bước đầu là −lr(1 + ν)(∇ + ωθ); với dampening δ và buffer bằng 0 (không phải None), là −lr(1 − δ)(∇ + ωθ). Cả hai là bội dương của gradient (vẫn đi xuống ở bậc một), chỉ khác biên độ.

*Chứng minh.*
- **(a)** q\*_r là hàm của span; mỗi số hạng là hiệu của hai giá trị q\* liên tiếp, nên tổng là telescoping: q\*(J) − q\*([J, J_{v₁}, …, J_{v_k}]).
- **(b)** Với 𝒥′ = [𝒥, V], nghịch đảo khối của 𝒥′ᵀH𝒥′ + μI cho H𝒥′G(𝒥′)𝒥′ᵀH = H𝒥G(𝒥)𝒥ᵀH + H_⊥V S⁻¹VᵀH_⊥ (phần bù Schur của khối 𝒥). Lấy H trừ đi cho công thức H_⊥. Với r ∈ range(H): r_⊥(𝒥) = r − H𝒥G𝒥ᵀ(HH⁺r) = H_⊥(𝒥)H⁺r; nhân công thức H_⊥ với H⁺r và dùng VᵀH_⊥H⁺r = Vᵀr_⊥ = s.
- **(c)** Thêm cột không làm q\* tăng (min trên tập lớn hơn), nên q\*([J, 𝒮]) ≤ q\*([J, v]) với mọi v ∈ 𝒮.
- **(d)** Quy tắc cập nhật của `torch.optim.SGD`: m ← νm + (1 − δ)(∇ + ωθ), rồi θ ← θ − lr·m (hoặc θ ← θ − lr·((∇ + ωθ) + νm) khi nesterov); với m = 0 trước bước ta được đúng các biểu thức nêu. ∎

Kiểm:
- T13a-WIT (18/40: chết trước bước 5, sống sau), T13a (kiểm lại rồi mới chèn: hàm đổi 0.0), T13a-NEG (chèn không kiểm lại: hàm đổi 0.108);
- T13b (`torch.optim.SGD` thật, momentum 0.9, weight decay 5e-4: sai lệch 0.0 và 4.9e-17, ∇_{a_v} = 0 chính xác), T13b-NEG (không đặt lại: lệch 4.4e-2 và 9.7e-3), T13b2 (nesterov: hệ số 1.9; dampening 0.5: hệ số 0.5; sai lệch 3.5e-18 và 0.0), T13b2-NEG (công thức momentum thường sai cho nesterov: lệch 1.3e-2);
- T13c (telescoping, sai lệch 0.0, [structural]), T13d (cập nhật hạng q của r_⊥ và H_⊥, 5.0e-14), T13d-NEG (dùng r_⊥ cũ cho slot thứ hai: gain sai ≥ 2.2e-2);
- T13e (quy tắc rẻ: 0 vi phạm / 40), T13e-WIT (greedy có deflation thắng "lấy 2 cái tốt nhất riêng lẻ" ở 31/40, hòa 9/40, không bao giờ thua);
- T13f (công thức Schur đúng với đặc trưng ReLU cho mọi a: 9.0e-13), T13f-NEG (dự báo tuyến tính kiểu A4 sai với ReLU: tối thiểu 2.3e-3, trung vị 1.36);
- T13g (cận dưới ℓ(v) không bao giờ vượt gain chính xác: 0 vi phạm), T13h-WIT (chọn dấu/bias theo ℓ cho gain chính xác ≥ cách cố định (+a₀, b = 0) ở 40/40 bài).

### T14. Chọn hướng khi có BN, và ridge của S_x

**Định lý 14.** Neuron ảo hoặc neuron mới là một kênh mới, BN train mode theo sau (γ = 1, β_BN = 0), có thể có ReLU sau BN. Cho s ∈ ℝ^{N×q} (residual xếp theo mẫu, ví dụ r_⊥) và mặt nạ D.
1. **(a)** Với mọi a (X̃a ≠ 0): ‖Σᵢ φᵢ(a)sᵢ‖² = N · aᵀX̃ᵀDssᵀDX̃a / aᵀX̃ᵀX̃a, với D là mặt nạ thực tế tại a. Tức là một thương Rayleigh suy rộng với thống kê **đã trừ trung bình**, và bất biến theo tỉ lệ của a (chuẩn hóa tự động).
2. **(b)** Không có ReLU (D = I): giá trị lớn nhất theo a là N·λ_max(X̃ᵀssᵀX̃, X̃ᵀX̃), đạt tại eigvec suy rộng đầu. Với r neuron đòi hỏi S₀-trực giao, tổng lớn nhất là N lần tổng r trị riêng đầu (Ky Fan).
3. **(c)** Có ReLU: trong một vùng mặt nạ R_D, giá trị không vượt N·λ_max(X̃ᵀDssᵀDX̃, X̃ᵀX̃), và đạt cận này khi eigvec tương ứng nằm trong R_D.
4. **(d)** Dùng thống kê *không* trừ trung bình (A4 như v1 viết) thì sai khi x̄ ≠ 0.

**Phạm vi (eps của BN).** (a) chính xác khi eps_BN → 0. Với eps_BN > 0, φ bị nhân thêm σ/√(σ² + eps_BN), nên sai số tương đối của đồng nhất là eps_BN/(σ² + eps_BN) < eps_BN/σ² (T17d: 4.0e-6 ở eps = 1e-5, đúng bằng eps/σ²; 4.4e-15 ở eps = 1e-14). Hướng tối ưu không đổi (hệ số dương chung).

*Chứng minh.* (a) BN train mode cho φ = D·(X̃a)/σ(a), σ(a)² = aᵀX̃ᵀX̃a/N (trung bình X̄a = x̄ᵀa tuyến tính theo a, nên X̃a là phần đã trừ trung bình). Thay vào ‖Σᵢφᵢsᵢ‖² = φᵀssᵀφ. (b) Rayleigh–Ritz cho bút chì (X̃ᵀssᵀX̃, X̃ᵀX̃) trên range(X̃ᵀ) (L9 cho trường hợp suy biến); Ky Fan cho tổng. (c) Trên R_D, D cố định; áp (b) với X̃ thay bằng DX̃ ở tử số. (d) Phản ví dụ số. ∎

Kiểm: T14a (đồng nhất Rayleigh, có và không có mặt nạ ReLU: 2.0e-14), T14b (không a ngẫu nhiên nào vượt eigvec đầu), T14b-NEG (thống kê không trừ trung bình sai ≥ 1.7e-4, so với dung sai 1e-10), T14c (L9: phần ngoài range 1.2e-7, khoảng cách trị riêng 1.9e-8 ở ρ = 1e-8), T14c-WIT (không ridge thì Cholesky thất bại khi d_a > N).

### T15. Kiểm ở mức thuật toán (không phải định lý mới)

Trên MLP tanh thật với CE + KD của PyCIL, chạy đúng bước 3–4 (β có chứng chỉ, quy tắc bỏ qua), rồi đi một bước θ − ε(g + λd̂):
- T15a: 24 lần áp dụng, 1 lần bỏ qua; L giảm ở mọi lần áp dụng, và dự báo bậc một −εgᵀ(g + λd̂) đúng tới 1.1e-4 (tương đối) ở ε = 1e-4;
- T15b: riêng bước hiệu chỉnh θ − ελd̂ không làm L_KD tăng ở lần nào khi ĥ ≥ 0 (0/24).

### T16. Swap slot sống: điểm khai triển, dự báo dưới A10, và độ tin cậy

**Định lý 16 (điểm khai triển).** Giả thiết A2. Với mọi f, f_u, Δ ∈ ℝⁿ và ℓ ∈ {L, L_KD}:
1. **(a)** |ℓ(f − f_u + Δ) − [ℓ(f − f_u) + ∇ℓ(f − f_u)ᵀΔ + ½Δᵀ∇²ℓ(f − f_u)Δ]| ≤ (L₃/6)‖Δ‖³.
2. **(b)** Nếu thay ∇²ℓ(f − f_u) bằng ∇²ℓ(f) (khai triển Hessian ở đầu ra *trước* swap), sai số chỉ còn bị chặn bởi (L₃/6)‖Δ‖³ + ½L₃‖f_u‖‖Δ‖², tức có thêm một số hạng **bậc hai** theo Δ, tỉ lệ với ‖f_u‖.
3. **(c) Hệ quả.** Trong dự báo ΔL_pred, ΔKD_pred (§0), mọi đại lượng sau swap phải lấy tại f − f_u. Với lựa chọn đó, dưới A10 và trong chế độ tuyến tính, sai số của dự báo là (L₃/6)(‖Δ‖³ + ‖Δ′‖³), cộng sai số tuyến tính hóa của Jacobian dọc bước hồi phục (không kiểm được; thuộc giả thiết hồi phục A10, không phải A6).

*Chứng minh.* (a) Taylor bậc ba tại f − f_u với phần dư Lagrange và L10. (b) Hiệu hai mô hình là ½Δᵀ(∇²ℓ(f − f_u) − ∇²ℓ(f))Δ; Hessian L₃-Lipschitz theo L10 cho |·| ≤ ½L₃‖f_u‖‖Δ‖²; cộng (a). (c) Áp (a) cho hai trạng thái và lấy hiệu như trong T7(i.3). ∎

Kiểm: T16a (tỉ số với cận (a) tối đa 0.225; với cận (b) tối đa 0.062), T16a-NEG (dùng H(f) thì vi phạm cận bậc ba tới 11.1 lần).

**Kết quả thực nghiệm (empirical), quyết định mặc định.** Trên MLP ReLU thật (giao thức của referee 3: slot sống u thay bằng v ngẫu nhiên, B_v = 0; mọi tham số):

| Điều kiện | Khớp dấu ΔL_pred với thực tế | Khớp dấu ΔKD_pred | Trong các cặp được nhận, cả L và KD thực sự giảm |
|---|---|---|---|
| μ_rel = 10, so với bước GN thực hiện | 194/198 (tương quan 0.97) | 193/198 (0.99) | 82/86 |
| μ_rel = 1, bước GN | 131/198 (0.44) | 131/198 (0.65) | 31/49 |
| μ_rel = 1e-2 (damping dùng thật), bước GN | 87/198 (−0.17) | 101/198 (0.07) | 1/29 |
| μ_rel = 10, so với 200 bước GD | 131/198 (0.33) | 116/198 (0.18) | 31/86 |

(`live_slot_198_out.txt`; T16b–d chạy bản 80 cặp với cùng kết luận: T16b: 0.97/0.99 khớp, tương quan 0.97/1.00 ở μ_rel = 10; T16c-WIT: 0.36/0.59 ở μ_rel = 1e-2; T16d-WIT: 0.66/0.64 so với GD.) Quy tắc Net_KD của v2 lệch dấu so với dự báo nhất quán A10 ở 43/80 cặp (T16e-WIT).

**Quyết định:** dự báo chỉ đáng tin khi bước hồi phục nhỏ (damping lớn, gần tuyến tính), không đáng tin ở damping dùng thật và không mô tả hồi phục thật. Vì vậy swap slot sống được hạ xuống **measured, not certified** và **TẮT mặc định** (LIVE_SWAP = off). Khi bật (measured), điều duy nhất được chứng nhận là chứng chỉ tức thời T7(i.5) (T16e: 0/37 vi phạm).

### T17. Slot có BN theo sau

**Định lý 17.** Giả thiết A1, A2, A12, BN train mode, kênh u của slot có affine (γ_u, β_u).
1. **(a) Bảo toàn hàm.** Với B_v = 0, việc đặt a_v, γ_v, β_v và running stats tùy ý cho kênh u không làm đổi hàm trên cửa sổ (chính xác).
2. **(b) Cần đặt lại γ, β.** Nếu kênh chết vì β_u (ví dụ β_u ≪ 0), giữ nguyên γ_u, β_u thì neuron mới vẫn chết (φ ≡ 0), nên ứng viên bị loại và không có swap. Đặt γ_v = σ, β_v theo quy tắc dấu/bias.
3. **(c) Bias trước BN bị triệt tiêu** ở train mode; quy tắc bias phải tác động lên β_BN: β_v = −median(σx̂) làm khoảng một nửa cửa sổ kích hoạt.
4. **(d) Công thức gain.** T7(b) đúng với J_v = φ ⊗ I tính từ đặc trưng BN thực tế φ = ReLU(γ_v x̂ + β_v); hiệu mô hình từ Jacobian của mạng sau chèn bằng −gain dạng Schur.
5. **(e) Running stats.** Ở train mode chúng không ảnh hưởng đầu ra; đặt running_mean_u = 0, running_var_u = 1 để thống kê eval của kênh mới không mang giá trị của kênh cũ (chúng hội tụ lại sau vài chục forward với momentum BN mặc định). Đây là quy ước cài đặt, không ảnh hưởng tới (a)–(d).

*Chứng minh.* (a) Đầu ra của kênh u chỉ đi qua B_u (A12), mà B_v = 0; BN theo kênh nên các kênh khác không đổi; ở train mode running stats không vào đầu ra. (b) φ = ReLU(γ x̂ + β) ≡ 0 khi γx̂ᵢ + β ≤ 0 với mọi i. (c) x̂ = (Xa + b − mean(Xa + b))/std không phụ thuộc b. Với β_v = −median(σx̂), số mẫu có σx̂ᵢ > median là ⌊N/2⌋ (nếu không trùng giá trị). (d) T7(b) đúng cho mọi đặc trưng (không cần A4′); cột của a_v, γ_v, β_v bằng 0 tại B_v = 0. (e) Định nghĩa của BN train mode. ∎

Kiểm: T17a (sai lệch 0.0 cho mọi ứng viên), T17a-WIT (không đặt lại: tổng φ = 0; đặt lại: 6.61), T17b (8/16 mẫu kích hoạt), T17b-WIT (bias trước BN: sai lệch 5.6e-17), T17c (8.0e-16), T17d (phạm vi eps của T14).

### T18. Quy tắc bỏ qua có dung sai (kiểm của T9(e))

Nội dung là T9(e). Kiểm trên 60 bài (một nửa có V_O = 0, dựng như T11a), CG với tol 1e-6:
- T18a: ‖d̂ − d\*‖ ≤ Δ̂ và |ĥ − h\*| ≤ τ_h ở mọi bài (0/60 vi phạm);
- T18a-NEG: bỏ hệ số 1/μ khỏi Δ̂ thì vi phạm 26/60;
- T18b-WIT: khi V_O = 0, quy tắc chặt ĥ < 0 bỏ nhầm 15/30 bước có chứng chỉ;
- T18c: quy tắc có dung sai bỏ nhầm 0/30.

## 5. Về "một nguyên lý duy nhất Ψ" (đã rút lại)

Claim cũ trong `unified` §1, rằng VNA, β và GIR là ba số hạng của một Lagrangian Ψ duy nhất, được rút lại. Lý do:
- phần liên tục tối ưu **độ trung thành chiếu**, không phải mức giảm loss;
- β bị dùng lại cho các đại lượng khác thang đo.

Các khẳng định thay thế, đã chứng minh:
1. **Cùng một dạng hàm chiếu E(𝒥, t)** (T7a). VNA tìm nghiệm của E trên span hiện tại với đích u (một vận tốc). GIR mở rộng span để giảm nhiều nhất E với đích −H⁺r (một độ dời Newton). Đây là **hai tiêu chí liên kết có cùng cấu trúc**, không phải một hàm mục tiêu chung: hai đích có ý nghĩa vật lý khác nhau, nên không có chung thang đo, và cùng một μ tác động khác nhau.
2. **β là nhân tử KKT của ngân sách không tham số ε\*** (T3f), và chỉ tác động lên phần liên tục. ε\* là ngân sách *đủ* theo Cauchy–Schwarz, có tính bảo thủ.
3. **β không bao giờ được chuyển sang swap.** Với slot chết, bước nhảy hàm và rò rỉ bằng 0 chính xác (T7f). Với slot sống (chỉ khi bật), bước nhảy trên KD được đo chính xác (j_u^{KD}, T7(i)) và đưa thẳng vào chứng chỉ tức thời và dự báo ΔKD_pred, cũng không qua β.
4. **Các liên hệ ngoài ngành là interpretation**, không phải định lý:
   - nguyên lý ràng buộc tối thiểu của Gauss: T3f là một bài toán khoảng cách nhỏ nhất có ràng buộc trong metric (H, μ);
   - điều khiển ưu tiên tác vụ: T2(b, d) là phép chiếu null-space của ưu tiên 2 vào ưu tiên 1, với ràng buộc ΛCx = 0;
   - OMP: chỉ chính xác khi H = I, μ → 0 (T7e);
   - không có khẳng định nào về tương đương với DER hay "growth với hàm cũ đóng băng" (xem §7, mục 7). Phát biểu hình thức duy nhất về chủ đề này là T2(d): ở giới hạn cứng, vận tốc học-mới nằm trong range(K_NS).

## 6. Bảng trạng thái các khẳng định

| ID | Khẳng định | Trạng thái | Kiểm (tên phép kiểm) |
|---|---|---|---|
| L5 | e, ẽ_O ∈ range(H); e_O ∈ range(W) | proven (A2) | T0a, T0b, T0b-WIT |
| L6 | ‖e_O‖²_{W⁺} = (1/N)Σχ² | proven | T0c 2.8e-15, T0c-NEG |
| L7 | Thừa số chính xác Λ của W | proven | T0f |
| L8 | Hessian KD dùng p_s của student | proven | T0d, T0e, T0e-NEG |
| L9 | S_x có ridge: eigvec trong range(X̃ᵀ), trị riêng hội tụ | proven | T14c, T14c-WIT |
| L10 | L₃ ≤ c₃(1 + T⁻³)/N, c₃ = 2√2/(6√3) ≈ 0.2722, chặt cho CE và cho KD | proven | T12a, T12a-NEG, T12a-WIT, T12a2 |
| T1a,b | Nghiệm duy nhất; d\* = d_rep + d_N^β; Φ(d_rep + x) = Φ_N(x) + const | proven | T1a 3.7e-12, T1a-NEG |
| T1c,d | Dạng output-space và z không cần range | proven | T1b, T1c, T1c-NEG |
| T2a | β = 0 cho VNA; gᵀd = V; Q ⪰ 0 | proven | T2a–c |
| T2b,c | Giới hạn cứng ΛCx = 0 (không phải Cx = 0); tốc độ O(1/β); C hạng tùy ý; suy biến khi Ĉ hạng cột đầy | proven | T2d–g, T2f-NEG, T11f |
| T2d | Jd_N^∞ ∈ range(K_NS), phần bù Schur | proven | T2h, T2i |
| T3a | g_Oᵀd_rep = V_O ≥ 0 | proven | T3a |
| T3b,c | Cận Cauchy–Schwarz; leak không tăng, ≤ ‖u_N‖²/β | proven | T3b, T3b-NEG, T3c, T3d |
| T3d | ΔL_KD = −εg_Oᵀd + ½ε²[…] (dấu đã sửa) | proven (phần dư: A6) | T3j; hệ số bậc hai: referee 2 |
| T3e | leak ≤ ε\* ⇒ h ≥ 0; β_cert ≤ β_suf | proven | T3e (0/6900), T3e-WIT, T3f |
| T3f | β_cert = nhân tử KKT của ngân sách ε\* (ngân sách *đủ*, bảo thủ) | proven | T3i 1.0e-7, T3i-WIT |
| T3g(i,ii) | 0 ≤ h(β_cert) ≤ 2V_O; h(∞) = V_O | proven | T3g, T3h |
| T3g(iii) | V_O = 0 ⇒ h(0) = 0, β = 0 có chứng chỉ | proven | T11a, T11b, T11c-WIT |
| T3g(iv) | e_O = 0 ⇒ h ≡ 0; ε\* = 0/0 được xử lý | proven | T11d, T11e |
| — | h đơn điệu | **sai** | T3d-WIT (62/300) |
| T3i | Phạt về đích 0 làm mất phần sửa | proven | `check_rep` 3c (referee xác nhận) |
| T4a,b | Φ_N(d_N^β) không giảm; ≤ β(leak₀ − leak_β) | proven | T4a, T4b |
| — | fit_β đơn điệu | **sai** | T4c-WIT, T4d |
| T5a,b | Cận chuẩn; cận dưới gᵀd\* | proven | T5a, T5b |
| T5c | γ ≠ 1 phá bảo đảm giảm loss; không có núm γ | proven | T5e (tìm kiếm đối kháng), T5e-NEG |
| T5d,e | Có quy tắc bỏ qua: giảm loss bậc một với mọi λ; bậc hai với ε < 2/(L_s(1+ρ)²) | proven (A6) | T5d (bổ đề), T5d-NEG, T15a |
| T5f | So với bước baseline (momentum); cải thiện chặt cần ĝ, ĥ > 0 | proven (A6) | T10e |
| — | Giảm loss vô điều kiện khi β > 0 | **sai** | T5c-WIT (8/200) |
| T6a–d | Lợi = tr(AᵀM_SA); tối ưu theo eigvec suy rộng; tiny_V tối ưu khi β = 0; cận dưới tối ưu | proven (A4′, trong một vùng mặt nạ) | T6a, T6a-NEG, T6b–d, T6d-WIT |
| T6e | Quy tắc xen kẽ cải thiện lợi | empirical | `gaps_out` G1f: +22% trung vị |
| T7a | Cùng *dạng* hàm chiếu E cho VNA và swap (không cùng thang đo) | proven (cấu trúc) | T7a, T7a-NEG |
| T7b | Schur theo r_⊥; đúng với mọi a_v, mọi phi tuyến | proven | T7b, T7b-NEG, T13f |
| T7c,d | Tốc độ ban đầu; cận hai phía; chọn theo argmax ℓ(v) trên 𝒱 đạt hệ số μ/(c_H+μ) (không phải tối ưu) | proven | T7c–e, T7f, T7f-WIT, T13g |
| T7d (tiny_V) | Hướng tốt nhất là tiny_V(r_⊥) / eigvec có trừ trung bình | proven (A4′) | T7f, T14a, T14b, T14b-NEG |
| T7d (dấu/bias ReLU) | Chọn theo ℓ trên tập ứng viên ± và bias | proven trên 𝒱; toàn cục: A4-glob | T13h-WIT |
| T7e | r_⊥ là phần bù xiên; OMP chỉ chính xác khi H = I | proven | T7g, T7h |
| T7f | Slot chết: bảo toàn hàm chính xác, giá 0; phải kiểm lại A5 sau bước 5 | proven (A5) | T7i, T7j-WIT, T13a, T13a-NEG, T13a-WIT |
| T7g | Sau swap K′ = K + J_vJ_vᵀ (không giao hoán); d̂ không đụng slot chết | proven | T7j, T7l (cấu trúc) |
| T7h | Quyết định swap slot chết không cần chân trời; với r = ẽ_O (CIL), "gain = giảm KD về sau" không suy ra từ A10 | proven; ghi chú phạm vi | hệ quả của T7b, T7f; T12d, T12d-NEG |
| T7i.1,2 | Slot sống: j_u đo chính xác; cận Taylor với L₃ chặt | proven (j_u: measured) | T7k, T7k-NEG, T12b, T12b-NEG, T12b-NEG2, T12c [structural] |
| T7i.3 | ΔL_pred, ΔKD_pred dọc bước hồi phục trên L, đại lượng sau swap tại f − f_u | proven (dạng mô hình, T16a); diễn giải: A10 | T16a, T16a-NEG |
| T7i.5 | j_u^L ≤ 0 và j_u^KD ≤ 0 ⇒ L, KD trên cửa sổ không tăng lúc chèn | proven | T16e [structural] |
| T7i.6 | Dự báo slot sống đáng tin ở damping dùng thật | **sai** (empirical) | T16c-WIT; `live_slot_198_out.txt` |
| — | Swap slot sống "có chứng chỉ trên cửa sổ" / quy tắc Net_KD < 0 của v2 | **rút lại** | T16e-WIT; §7 mục 30, 31 |
| T7j | GIR không có slot chết trong CIL; với cấu hình mặc định GIR không hoạt động trong CIL | **empirical** (số đo) + hệ quả của mặc định | số đo dormant = 0 trên ResNet32 |
| A10 | Hồi phục trên L (một bước GN có damping, tuyến tính) là mô hình đúng để diễn giải gain và dự báo | assumption, interpretive | `repair_swap`: thắng 72%, n = 40, toy; T16b (đúng ở μ_rel = 10), T16d-WIT (không khớp hồi phục thật) |
| T8 | Cận Grönwall theo defect thực hiện được, so với tham chiếu tự trị | proven (A7) | T8a, T8a-NEG |
| — | "Quỹ đạo ≈ mô hình grow" | **không claim** | 0.728 so với 0.020 |
| T9a–d | Sai số CG; chứng chỉ chính xác trên d̂; kiểm cos | proven (A8) | T9a, T9a-NEG, T9b |
| T9e | Quy tắc bỏ qua có dung sai τ = ‖·‖Δ̂: không bỏ nhầm; khi áp dụng, KD tăng bậc một ≤ ελτ_h | proven (A8) | T18a, T18a-NEG, T18b-WIT, T18c |
| T10a,b | BN train mode (trừ phần tối ưu của T6, T7d); loại tham số BN | proven (A1) | T10a–c, T10b-NEG |
| T10c | A4 toàn cục sai khi có BN; thay bằng A4′ | proven (phản ví dụ) | T10d |
| T10d | Cộng ngoài buffer đúng; trong buffer phát lại 1/(1−ν); đặt lại trạng thái khi swap | proven | T10e, T10f, T10f-NEG, T13b, T13b-NEG |
| T11 | Quy tắc β, các nhánh suy biến, bisection log | proven | T11a–g |
| T13a–c | Cộng dồn nhiều slot; cập nhật hạng q; quy tắc rẻ có cận | proven | T13c, T13d, T13d-NEG, T13e, T13e-WIT |
| T13d | Đặt lại trạng thái optimizer khi swap (momentum thường; nesterov/dampening: bội dương) | proven | T13b, T13b-NEG, T13b2, T13b2-NEG |
| T14 | BN: Rayleigh suy rộng với thống kê đã trừ trung bình (chính xác khi eps_BN → 0; sai số < eps/σ²) | proven (A4′) | T14a, T14b, T14b-NEG, T17d |
| T16a | Điểm khai triển f − f_u (bậc ba) so với f (thêm bậc hai) | proven | T16a, T16a-NEG |
| T17 | Slot có BN: bảo toàn hàm, cần đặt lại γ/β, bias lên β_BN, gain Schur | proven (A12) | T17a, T17a-WIT, T17b, T17b-WIT, T17c |
| — | Một Lagrangian duy nhất Ψ | **rút lại** (interpretation) | §5 |
| — | Tương đương DER / "growth với hàm cũ đóng băng" | **bỏ** | §7 mục 7 |

## 7. Các khẳng định đã bị thay thế (không dùng nữa)

1. **Núm γ.** "γ = 1 là thành phần sửa mới", "đăng ký γ ∈ {0, 1}". γ = 1 chính là VNA hiện tại, và γ ≠ 1 phá bảo đảm giảm loss (T5c).
2. **Dạng η, ‖Cd + ηe_old‖².** Sai dấu và sai metric. Hệ số đúng của tương đương Taylor là 2β/ε², không phải 2β. Đã thay bằng đích Cd_rep (T1).
3. **Phạt về đích 0 β‖Cd‖²_W, hay metric H + βH_KD, làm "đề xuất chính"** (`math.md` §1, §6). Dạng này làm mất phần sửa (T3i).
4. **`math.md` Prop 2: giới hạn Cd = 0.** Sai; đúng là ΛC(d − d_rep) = 0 (T2c).
5. **`math.md` Prop 3: số hạng bậc một +εe_OᵀCd.** Dấu đúng là −ε (T3d).
6. **`math.md` §5: Hessian KD dùng xác suất teacher.** Phải dùng student (L8).
7. **`math.md` Prop 7 và bản thu hẹp ở v1: "VNA-NS = chiếu của growth kiểu DER" / "chiếu một bước của growth với hàm cũ đóng băng, trừ phần sửa".** **Bỏ hẳn** ở v2: không có định lý nào phát biểu hay chứng minh điều này (referee 2, mục 47). Phát biểu hình thức duy nhất còn lại là T2(d).
8. **`math.md` Prop 8(ii), dạng (1+β).** Chỉ xấp xỉ, vì H của iCaRL không tách khối KD/CE.
9. **`math.md` z_β với vế phải JᵀHe.** Sai đơn vị; đã thay bằng y_β = HJG_βg (T6).
10. **`math_rep` 4d, "fit tăng đơn điệu".** Sai; Φ_N mới đơn điệu (T4). Referee 2 tìm được cả phản ví dụ *trong mô hình* (B ⪯ F) ở 8/150 bài.
11. **`math_rep` 5(i), "Jd ∈ range(K)".** Hiển nhiên đúng, không có nội dung; đã bỏ.
12. **Các arm β ∈ {0, 3, ∞}.** Thay bằng β có chứng chỉ (một phương pháp). β = 0 chỉ còn là ablation.
13. **`unified` §1, "nguyên lý duy nhất Ψ" như một định lý.** Rút lại; xem §5.
14. **`unified` Định lý 1′, h đơn điệu và bisection trên h.** Sai (T3h). Thay bằng chứng chỉ theo leak (T3e).
15. **`unified` Hệ quả 3a/3b: chọn neuron swap theo toàn bộ e.** Bị dữ liệu bác bỏ; cotangent đúng là r_⊥ (T7).
16. **`unified` Hệ quả 3c: dấu của giá swap.** Đã sửa thành −eᵀf_u + ½f_uᵀHf_u (T7i).
17. **`repair_swap`: "e_O − z_O ≈ r⊥ chính xác"; OMP "chính xác".** e_O − z_O chính là đại lượng đúng; OMP chỉ chính xác khi H = I (T7e).
18. **`proofs_final` T4g: "K không đổi, swap giao hoán với VNA".** Sai (T7g).
19. **`proofs_final` bước 4: "nhân đôi β tới β_suf khi ĥ < 0".** Thay bằng bỏ qua lần áp dụng.
20. **Dùng β_suf làm mặc định.** Quá bảo thủ (trung vị gấp 29 lần β_cert). Chỉ còn là đầu mút trên của khoảng bisection.
21. **(v1) "Slot sống không có chứng chỉ vì L₃ không kiểm được".** Sai: L₃ trong không gian output có dạng đóng (L10), và j_u tính chính xác bằng một forward. v2 cho phép slot sống với quy tắc Net; v3 thay quy tắc đó và tắt mặc định (mục 30–32). Chỉ L_s, L_K (không gian tham số) là không kiểm được.
22. **Các phép kiểm số rỗng** (`unified` C2/C4/C5, phép kiểm FD tổng loss trong `check_rep`, code GPM không dùng trong `check_ns`; ở v1: T5e dương và T10f). Thay bằng `final_check_v2.py`, nay là `final_check_v3.py`.
23. **(v1) "V_O = 0 ⇒ bỏ qua lần áp dụng" và ngưỡng τ_V.** Sai (T3(g)(iii)): β = 0 có chứng chỉ khi V_O = 0.
24. **(v1) Bisection log trên [0, β_suf].** Thang log không có đầu mút 0; thay bằng [β_suf·10^{−k_lo}, β_suf] (T11).
25. **(v1) T10(a) "T1–T9 đúng chính xác dưới BN".** Thu hẹp: phần tối ưu của T6 và T7(d) cần A4′ (T10a, T14).
26. **(v1) "Cùng đơn vị vận tốc² theo H" ở T7a/§5.** Bỏ; hai đích có thang đo khác nhau.
27. **(v1) "ε\* bằng đúng lượng phần sửa bù lại" ở T3f.** Thay bằng "ngân sách đủ theo Cauchy–Schwarz, bảo thủ".
28. **(v1) Đếm chi phí "3 + n_b (+1)".** Thiếu lần giải cho leak(0) và β_cert(A₀) khi làm mới A; đã sửa (§2).
29. **(v1) T7(j) gọi là "hệ quả".** Đó là số đo; đã đổi nhãn thành empirical.
30. **(v2) Quy tắc slot sống "swap khi Net(u, v; e, L) < 0 và Net(u, v; ẽ_O, L_KD) < 0", đọc dưới A10 như mức giảm KD về sau.** Sai về logic: Net_KD mô hình hồi phục chỉ trên KD, còn A10 là hồi phục trên L; lệch dấu ở 104/198 cặp (referee 3) và 43/80 (T16e-WIT). Thay bằng ΔL_pred, ΔKD_pred (§0, T7(i.3)). Điểm khai triển của q\* (H tại f hay tại f − f_u) trước đây không được nêu; nay là f − f_u (T16a).
31. **(v2) "Slot sống có chứng chỉ trên cửa sổ".** Nói quá: j_u được đo chính xác nhưng có thể dương (18/29 cặp được quy tắc v2 nhận có j_u > 0). Chứng chỉ thật duy nhất là j_u^L ≤ 0 và j_u^KD ≤ 0 (T7(i.5)), và nó chỉ nói về lúc chèn.
32. **(v2) Slot sống được phép mặc định.** Dự báo của nó không đáng tin ở damping dùng thật (T16c-WIT); nay TẮT mặc định.
33. **(v2) L₃ ≤ 0.708(1 + T⁻³)/N.** Đúng nhưng lỏng 2.6 lần; thay bằng c₃ = 0.2722 chặt (L10). Phép kiểm T12a/T12b của v2 không bắt được hằng số sai (referee 3); đã thay bằng phép kiểm hai phía.
34. **(v2) Bước 4 với quy tắc chặt ĥ < 0 khi dùng CG.** Bỏ nhầm khoảng một nửa số bước khi V_O = 0 (T18b-WIT); thay bằng dung sai τ từ T9(e).
35. **(v2) Bước 6a không nói gì về BN của slot.** Thiếu: phải đặt lại γ, β (và running stats), và quy tắc bias phải tác động lên β_BN (T17).
36. **(v2) T7f kiểm argmax ‖s‖ trong khi quy tắc là argmax ℓ; T12d là đồng nhất hiển nhiên (net_ = −gain theo cách dựng).** Đã thay (T7f, T12d).
37. **(v2) Đếm chi phí: thiếu lần giải có thể cho d_N(β_suf); "1 lần giải mỗi cặp" cho slot sống thiếu Jacobian/H mới và vế phải thứ hai.** Đã sửa (§2).

## 8. Giới hạn

1. **Cục bộ.** Mọi bảo đảm là bậc một (hoặc bậc hai với các hằng số A6 không kiểm được), cho từng lần áp dụng, trên loss của cửa sổ. Không có bảo đảm cho cả quỹ đạo, trừ khi defect thực hiện được nhỏ (T8, so với tham chiếu tự trị A7). Không có bảo đảm về tổng quát hóa hay accuracy trên test.
2. **Tính tối ưu của bước chọn chỉ trong một vùng mặt nạ (A4′).** Tối ưu toàn cục theo a (khi mặt nạ ReLU đổi) là giả thiết A4-glob, cho cả VNA lẫn GIR. Các chứng chỉ và công thức gain chính xác không bị ảnh hưởng.
3. **GIR trong CIL.** 0% đơn vị chết (số đo), và nhánh slot sống mặc định tắt vì dự báo của nó không đáng tin ở damping dùng thật (T16). **Với cấu hình mặc định, GIR không hoạt động trong CIL.** Bật LIVE_SWAP = measured là một thí nghiệm: chỉ có chứng chỉ tức thời T7(i.5); phải log ΔL_pred, ΔKD_pred và kết quả thực.
4. **Việc chọn v theo r_⊥ dựa trên A10**, và bằng chứng thực nghiệm chỉ có trên toy (40 seed): lợi thế nằm ở tỉ lệ thắng; CE lớp cũ trung bình hòa với cách chọn theo e.
5. **β_cert bảo thủ:** trung vị gấp 4.5 lần β nhỏ nhất trên lưới cho h ≥ 0 (`gaps_out` G2k). Bảo thủ thì mất một ít plasticity, theo T4b.
6. **Tỉ lệ bỏ qua** khoảng 4% trên toy (T5c-WIT: 8/200; T15a: 1/25). Chưa biết trên ResNet32; phải log.
7. **Chi phí:** tới 6 + n_b lần giải mỗi lần áp dụng trong CIL (9 + 2n_b khi làm mới A), so với 2 của VNA thuần (§2). Ảnh hưởng tới quy tắc chọn K.
8. **Toàn bộ kiểm số làm trên mô hình nhỏ, float64.** Ở float32 trên GPU phải kiểm lại các đồng nhất chính (T1, T3a, T3e) ở các điểm kiểm. Nếu p_s underflow, χ² có thể vô hạn; an toàn vẫn do chứng chỉ trực tiếp ở bước 4.
9. **Phần sửa chỉ hoạt động trong range(K).** Đặc trưng cũ đã mất hẳn (g_O = 0 theo các hướng đó) không sửa được ở bậc một (T3a, T2d). Trong CIL, cách duy nhất để thêm bậc tự do là nhánh slot sống, với các giới hạn ở mục 3.
10. **Swap slot sống không bảo toàn hàm**, kể cả trên cửa sổ; lượng thay đổi được đo chính xác, nhưng ngoài cửa sổ không có chứng chỉ.
11. **A10 không mô tả hồi phục thật.** Ngay ở chế độ tuyến tính, dự báo theo một bước GN khớp kém với 200 bước GD (T16d-WIT). Mọi diễn giải "lợi về sau" của GIR (cả slot chết) là interpretive; chỉ bảo toàn hàm và các chứng chỉ là chính xác.
12. **Dung sai τ_h, τ_g dùng chuẩn phổ** ‖HJ‖, ‖JᵀHK_E‖; lúc chạy chúng được ước lượng bằng lặp lũy thừa (cận dưới của chuẩn thật). Để giữ đúng bảo đảm, nhân ước lượng với một hệ số an toàn (ví dụ 1.1) hoặc chạy đủ vòng lặp.
13. **Slot hợp lệ (A12)** loại các kênh đi vào phép cộng residual; trong ResNet32 chỉ kênh trong của mỗi basic block là ứng viên.

## 9. Đối chiếu với referee 3

| Mục referee 3 | Xử lý ở v3 |
|---|---|
| #33 (sai: Net_KD dưới A10; H chưa nêu) | Thay bằng ΔL_pred, ΔKD_pred dọc x_L (dùng lại lần giải của bước 1 cho trạng thái trước swap); đại lượng sau swap tại f − f_u, có chứng minh (T16a, T7(i.3)); chạy lại thí nghiệm 198 cặp (`live_slot_198_out.txt`, T16b–d): chỉ đáng tin ở μ_rel = 10 ⇒ slot sống hạ xuống measured, TẮT mặc định; áp lưu ý tương tự cho slot chết với r = ẽ_O (T7(h)) |
| #34 (nói quá: "có chứng chỉ trên cửa sổ") | Bỏ; chứng chỉ tức thời j_u^L ≤ 0, j_u^KD ≤ 0 là điều kiện bắt buộc khi bật, và phát biểu đúng điều nó bảo đảm (T7(i.5), T16e) |
| #7, #46 (L₃ lỏng; T12a/T12b không bắt được hằng số sai) | c₃ = 0.2722 chặt với chứng minh đạt cận (L10); T12a hai phía + T12a-NEG (T⁻¹, ×0.1), T12a-WIT (0.708 lỏng); T12b có cấu hình gần chặt + T12b-NEG (bỏ số hạng KD) |
| #30, #31 (BN: γ, β; bias) | Bước 6a đặt lại γ, β, running stats; quy tắc dấu/bias lên β_BN (T17a–c) |
| #19 (V_O = 0 với CG) | Dung sai τ_h, τ_g từ chuỗi CG (T9(e), T18) |
| #44 (chi phí) | +1 lần giải cho d_N(β_suf); slot sống: tính lại tại θ′, 1 lần giải/cặp trên mạng đã sửa, không cần vế phải thứ hai |
| #41, #42 (phạm vi SGD; eps BN) | T13(d), T10(d), A9 (nesterov, dampening; T13b2); T14 phạm vi eps (T17d) |
| #29, #46 (T7f; T12d hiển nhiên; nhãn) | T7f kiểm argmax ℓ (+ T7f-WIT); T12d trên mạng thật (+ T12d-NEG); nhãn [structural] cho T12c, T13c, T16e |
| #43 (θ và θ′) | Quy ước điểm tính ở bước 6; tính lại sau mỗi swap sống |
| #45 (kênh residual) | A12 |

## 9b. Đối chiếu với referee 2 (lịch sử v2)

| Mục referee 2 | Xử lý ở v2 |
|---|---|
| #19, #40 (V_O = 0 ⇒ bỏ qua; τ_V) | Sửa: T3(g)(iii), T11a–c; bỏ τ_V; nhánh e_O = 0 (T3(g)(iv), T11d, T11e) |
| #34 (slot sống, L₃) | Sửa: L10, A6′, T7(i), T12; §7 mục 21 |
| #44 (GIR bước 6) | Sửa: kiểm lại A5 (T7f, T13a), đặt lại optimizer (A9, T13b), nhiều slot (T13a–c), dấu/bias ReLU (T7d, T13f–h); r_⊥ tính tại θ (T7h) |
| #30 (A4 với ReLU/BN) | Sửa: A4′, T6, T7d, T14 |
| #26 (S_x) | Sửa: định nghĩa §0, L9, T14c |
| #38 (T10a) | Sửa: T10(a) loại trừ phần tối ưu của T6, T7d |
| #45 (chi phí) | Sửa: bảng chi phí §2 |
| #17, #28, #46 (ngân sách, đơn vị) | Sửa: T3f, T7a, §5 |
| #36 (A7) | Sửa: A7 tự trị, T8 |
| #25 (T5f) | Sửa: cải thiện chặt cần > 0 |
| #11 (suy biến hard limit) | Sửa: T2(b), T11f |
| #41 (bisection, χ²) | Sửa: bước 3, T11g, §8.8 |
| #47 (DER) | Bỏ claim (§7 mục 7) |
| #35 (T7j) | Đổi nhãn empirical |
| #48 và §3 (số cũ, kiểm rỗng, NEG) | Mọi số lấy từ `final_check_v2_out.txt`; T5e và T10f viết lại (T10f dùng `torch.optim.SGD` thật); tách NEG (22, đột biến) và WIT (14, nhân chứng); T7b-NEG chạy riêng với biên độ rõ; sửa điều kiện chết trong T2e; thêm kiểm cho V_O = 0, e_O = 0, c_old = 1, S_x suy biến, p ≤ N(c_old − 1), A5 sau bước 5, swap với optimizer |
| #23 (T4c ngoài mô hình) | Ghi nhận phản ví dụ trong mô hình của referee 2 (§7 mục 10) |
| T5d, T7l, T3j, T8a (kiểm bổ đề / cấu trúc) | Giữ, nhưng ghi rõ phạm vi; thêm kiểm mức thuật toán T15 |
