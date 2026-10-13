# Implementation spec: Virtual-NTK Augmentation (VNA) with adaptive when / how-much

Repo: `Counterfactual-Projection` (this repository). Target backbone for this task: **CIFAR-ResNet18 (Gromo) on CIFAR-100**. Do not touch DeiT/VGG/DenseNet code paths.

## 0. Context (read first)

Earlier experiments applied a one-shot "TINY virtual expansion → project into original block" at a converged plateau. It does not work, for a provable reason: at a stationary point any update in range(Jᵀ) has zero first-order gain, and with saturated cross-entropy every train-based signal is bounded by the train loss (≈1e-3). Logs confirm gains ≈1e-6 and all "improvements" came from the rollback controller.

The new method moves the virtual expansion **into training** as an **auxiliary-space preconditioner**:

- The virtual neurons are never installed.
- Their function-space effect is transferred to the existing parameters at every VNA step.
- An adaptive controller decides **when** to activate or refresh, and **how much** to expand (rank and number of sites).

Reuse existing code wherever possible:

- `projection/functional.py`: `FunctionalProjector` (torch.func JVP/VJP, dual CG, `_preserve_functional_state`, `relative_gram` damping).
- `projection/cg.py`: PCG.
- `probe/gromo_adapter.py`: `TransactionalCandidateSource`, `CandidateExpansionProbe`, `_snapshot` / `_assert_unchanged`.
- `probe/counterfactual_tiny.py`: `CounterfactualTinyProbe` (rank beyond full width).
- `third_party/One-Shot-TAS-CCIL/dual_growth/adapters/tiny_adapter.py`: `propose_all`, `_virtual`, and `payload["tiny_eigenvalues"]`.
- `methods/e_projection.py`: `candidate_projection_parameter_names(..., "residual_path")`.
- `experiments/shared_protocol.py`: datasets/splits, `build_cifar_gromo_resnet`, `train_epoch`, `evaluate`, RNG and atomic-save helpers.
- `experiments/run_gromo_pilot.py`: `rebuild_sgd_after_growth` (needed for the real-growth reference arm).

## 0.1 What changes vs. the existing pipeline (read carefully; do not port old behaviour)

This is a **new method in new files**. Do not modify or refactor existing runners (`run_plateau_fork.py`, `run_deit_*`, `run_shared_comparison.py`, …). Old arms must keep working.

**REMOVED. Must not appear in any new code path:**

- Plateau detection, the stall window, and forking from θ_P / `plateau_checkpoint.pt`. Every arm trains **from random init** with the full recipe.
- The rollback / anchor controller: `retrigger_patience`, `controller_anchor_*`, `vanilla_rollback`, restoring best checkpoints, "scientific_escape", and "report_best" metrics.
- One-shot parameter jumps: `ProjectionResult.apply_(model, scale)`, the line search over `{0.0125, 0.025, 0.05}`, the 32-sample gate batch, and `correction_applied` gating.
- Momentum reset after a jump (`reset_projected_momentum`). The VNA correction is added to the **gradient**, so SGD momentum and weight decay treat it like any gradient.
- WHERE by raw observed E loss gain on 3×32 un-augmented train images (`select_by_expansion_gain`).
- Probe batches built from `eval_set` (un-augmented) train images.
- Validation-based checkpoint selection for reporting, and per-intervention checkpoints.

**KEPT (reuse as-is):**

- Matrix-free JVP/VJP machinery, CG/PCG, Hutchinson trace and diagonal estimates, `_preserve_functional_state`, the state-invariance helpers.
- The TINY / Gromo statistics and solve, used **only to obtain the incoming weights A and the eigenvalues λ**.
- Data splits, model builder, and the 200-epoch recipe.

**CHANGED:**

| Aspect | Old | New |
|---|---|---|
| When | Once, or every 10 epochs after a stall | Continuously during training, gated by the controller |
| What is applied | Δθ = J⁺Δf_E, as a parameter jump | Gradient correction λ·Jᵀ R K_E R K g, every `vna_every` steps |
| E's outgoing weights | TINY's B | Not used. B is a variable evaluated at 0; only A (incoming) and λ come from TINY |
| Sites / rank | Top-1 site by raw gain, rank 4 | Dörfler on transferable V_s, adaptive rank |
| Reporting | Max validation accuracy | Final model, official test set, once |

## 0.2 Reference implementations policy

- **Use the official / pinned upstream code for every external component.** Do not re-implement:

| Component | Source |
|---|---|
| TINY | Gromo `growingnet/gromo` @ `8d19107b61a9459a9021065a329b699adcb0f25b`: `compute_statistics`, `compute_optimal_updates`, extension layers. This is already used via `third_party/One-Shot-TAS-CCIL/dual_growth/adapters/tiny_adapter.py` |
| SWA | `torch.optim.swa_utils` |
| SAM, DBB, RepOpt, K-FAC, GradMax (later baselines) | The authors' official repositories. The agent must locate them, verify they are official, and pin the commit hash in `third_party/REFERENCES.md`. Record the license. |

- **Alignment check is mandatory before use.** Gromo / TINY was designed for growing, not for VNA. For each reused piece, write a short note in `third_party/REFERENCES.md` and add a test proving it matches this spec:
  1. **Over-expansion beyond full width.** TINY must be asked for `current_width + r` neurons (use `CounterfactualTinyProbe`). Otherwise Gromo returns no candidate for a full-width model.
  2. **Use A only.** The adapter's `_virtual` sets `optimal_delta_scaling = 0` and uses TINY's own B. VNA must instead treat B as a free variable at 0. Take A from the extension's incoming layer; ignore TINY's B and the existing-weight `optimal_delta`.
  3. **Loss scaling.** Gromo uses `CrossEntropyLoss(reduction="sum")`. λ are therefore sum-scaled. Only ratios (λ_k/λ_1, the noise-floor comparison, d_μ) may use them. Never mix λ with U/V, which are defined with mean CE.
  4. **Statistics batches** must use the train transform. Statistics collection can update BN buffers; the transactional wrapper already restores them, so keep it.
- If an upstream function behaves differently from this spec, **stop and report the discrepancy** instead of silently adapting either side.

## 1. Math (implement exactly)

**Notation.** For a batch of m samples and C classes, n = m·C:

| Symbol | Definition |
|---|---|
| f | Logits, flattened to length n |
| g | (softmax(f) − onehot(y)) / m, i.e. the gradient of mean CE w.r.t. logits, flattened |
| θ_b | Parameters of the selected block (residual-path scope) |
| J | ∂f/∂θ_b. Use only matrix-free products: `Jv` (JVP) and `Jᵀu` (VJP) |
| K | J Jᵀ, the block NTK on the batch |
| R | (K + μI)⁻¹, with μ = `damping_rel` · tr(K)/n (Hutchinson trace, as in `relative_gram`) |
| B | Outgoing weights of the r virtual neurons (incoming weights A come from TINY and are held fixed) |
| J_E | ∂f/∂B evaluated at B = 0, gate = 1 |
| K_E | J_E J_Eᵀ |

**VNA block update**, added to the SGD gradient of θ_b:

```
w  = K g          = J (Jᵀ g)
z  = R w          (CG on K + μI, warm-start from previous z, ≤ cg_iters)
s  = K_E z        = J_E (J_Eᵀ z)
q  = R s          (CG)
d  = Jᵀ q         (VJP)  →  grad_θb ← grad_θb + λ_vna · d
```

The full update is Δθ_b = −lr·(∇L + λ·Jᵀ R K_E R K g). The preconditioner P = I + λ Jᵀ R K_E R J is symmetric PSD, so the update is always a descent direction.

**Diagnostics** (cheap by-products):

| Quantity | Formula | Meaning |
|---|---|---|
| U | gᵀ K_E g = ‖J_Eᵀ g‖² | First-order gain the expansion offers |
| V | zᵀ s = ‖J_Eᵀ R K g‖² | First-order gain transferable to O |
| η | V / U | Transfer fraction |
| ρ | λ_vna · V / ‖Jᵀ g‖² | VNA gain relative to SGD on this block |

Per-neuron split: V_k = (j_kᵀ R K g)², where j_k is the column block of J_E for neuron k. Report V_k so ranks can be cut greedily.

**Implementing J_E products.**
- Inside the candidate's virtual context, the extension output equals gate · B·σ(A·x).
- Compute products w.r.t. the outgoing extension weight tensor (Gromo: `block.second_layer.extended_input_layer.weight`, plus bias if present) at value **zero**, with gate = 1, using `torch.func.functional_call` + `jvp` / `vjp` w.r.t. that tensor only.
- Model parameters must remain bit-identical: wrap in the existing snapshot/assert helpers.
- **Unit-test against finite differences.**

### 1.1 Implementation conventions (previously unspecified; follow exactly)

- **Normalization.** g uses mean CE over the VNA batch. The correction d is computed per site, multiplied by `lambda_vna`, and **added to `p.grad` of that site's parameters after `loss.backward()` and before `optimizer.step()`**. Do not rescale by LR. Gradient clipping, if any, applies after adding.
- **Parameter scope** of θ_b: `candidate_projection_parameter_names(model, site, "residual_path")`, i.e. the two convs plus their BN affine parameters. Shortcuts are excluded. Sites are distinct blocks, so scopes never overlap. Assert this.
- **BN mode.** All J, J_E, and K products run with `model.eval()` (running statistics), restoring modes afterwards. This is a known approximation, because SGD steps use batch statistics. Log it. Add an optional flag `--vna-bn train` that evaluates the functional forward with batch statistics and does **not** update running buffers (pass copied buffers to `functional_call`). Test that buffers are unchanged.
- **Precision.** Network JVP/VJP in fp32. CG vectors and dot products in fp64 (as in `FunctionalProjector`). Disable autocast/AMP inside VNA computations. Compute diagnostics CE in fp64.
- **μ and preconditioner.** Estimate tr(K)/n and the Jacobi diagonal with 8 Hutchinson probes **at each refresh** (cached, not every step), on a fixed refresh batch. Each VNA step uses `cg_iters` with warm start from the previous step's z and q. If CG is not converged, use the iterate and log it. If anything is non-finite, skip that VNA step, log it, and continue with plain SGD.
- **RNG isolation (critical for paired comparison).** The SGD training loader and its generator must be **identical across all arms**. VNA batches, statistics batches, check batches, Hutchinson probes and random-A draws use **separate** `torch.Generator`s seeded from `(seed, purpose)`. They must never consume the training loader or the global RNG. Test: arms `sgd` and `vna_adaptive` with `lambda_vna = 0` produce bitwise-identical CPU training trajectories.
- **Cost accounting.** Time VNA work (refresh, checks, per-step corrections) separately with CUDA synchronization. Report total and per-component seconds.

## 2. Controller: when and how much

Run a check every `check_every` epochs (default 1). Maintain the cache `E = {site: (A_site, rank_site)}` and `state ∈ {off, active, stopped}`.

**Refresh** (expensive; only when needed). Triggered when the cache is empty, when mean η over cached sites has dropped below `eta_drift` × its value at the last refresh, or after `refresh_max_epochs`. Steps:

1. **TINY statistics** on `stat_samples` augmented train images, using the train transform with the model in eval mode. Keep an EMA across refreshes with weight `stat_ema` (default 0.5) to stabilize, as in Wentges smoothing. Per site, obtain eigenvalues λ_1 ≥ λ_2 ≥ … from the payload.
2. **Noise floor (G1).** Recompute λ_1 on `n_boot` = 4 disjoint sub-batches. Set `noise = mean + 2·std` of those. A site is significant if λ_1(full) > noise.
3. **Rank per site:**
   r = min(`r_max`, ⌈d_μ⌉, max{k : λ_k/λ_1 > `tau_rel`}), where d_μ = Σ_k λ_k/(λ_k + noise).
   Then compute V_k on a held-in check batch (m = `vna_batch`) and keep the smallest prefix whose cumulative V reaches `rank_energy` (default 0.9) of total V.
4. **Site selection (Dörfler marking).** Compute V_s for every significant site. Choose the minimal set of sites, sorted by V_s, whose sum reaches `dorfler_theta` (default 0.5) of Σ V_s, capped at `max_sites`.
5. Store A and rank per selected site, and record λ, noise, d_μ, U_s, V_s, η_s.

**Activation gates**, evaluated at every check with the cached E:

| Gate | Condition |
|---|---|
| G1 | At least one selected site is significant (from the last refresh) |
| G2 | ρ > `tau_rho` (default 0.3), measured on the check batch |
| G3 (logged; enforced only if `use_g3`) | The EMA train-loss decrease per wall-second over the last epoch is < `g3_ratio` × its value after the last refresh |

state = active iff G1 ∧ G2 (∧ G3 when enabled).

**Accept test** (bundle-method serious vs null step).
- After the first `accept_steps` VNA steps following a (re)activation, measure the actual decrease of check-batch loss and compare with the predicted value lr·λ·V.
- If the ratio is < 0.1: set state = off until the next check, double `check_every` (cap 5), and increase `stat_ema` toward 0.8.

**Stop.** If no site is significant for `stop_patience` = 3 consecutive refreshes, set state = stopped. From then on the arm is plain SGD.

**While active:** run the VNA update every `vna_every` optimizer steps (default 4) on a separate VNA batch of `vna_batch` (default 128) augmented train samples. With multiple sites, sum the per-site corrections. Each site uses its own θ_b and its own K_E; K is the NTK of that site's block.

## 3. Default hyperparameters (put them in one dataclass and log it)

| Parameter | Default |
|---|---|
| `lambda_vna` | 1.0 |
| `damping_rel` | 1e-2 |
| `cg_iters` | 30 |
| `cg_tol` | 1e-2 |
| `vna_every` | 4 |
| `vna_batch` | 128 |
| `r_max` | 16 |
| `tau_rel` | 0.05 |
| `rank_energy` | 0.9 |
| `dorfler_theta` | 0.5 |
| `max_sites` | 3 |
| `stat_samples` | 1024 |
| `n_boot` | 4 |
| `stat_ema` | 0.5 |
| `tau_rho` | 0.3 |
| `check_every` | 1 epoch |
| `refresh_max_epochs` | 10 |
| `eta_drift` | 0.5 |
| `accept_steps` | 50 |
| `stop_patience` | 3 |
| `use_g3` | False |

## 4. Deliverables

### 4.1 `experiments/train_recipe_checkpoints.py`

- Plain ResNet18 recipe: SGD lr 0.1, momentum 0.9, WD 5e-4, MultiStep at 100/150 (γ 0.1), 200 epochs, batch 64. Splits come from `datasets_and_indices` (5000 validation, of which 2000 trigger and 3000 eval, plus 128 tuning).
- Save full-state checkpoints every 10 epochs (model, optimizer, scheduler, RNG, loader state).
- Keep only these checkpoints plus `latest`. **No `best` / plateau / rollback logic.**

### 4.2 `diagnostics/signal_curve.py` (Phase 0, no training)

For each saved checkpoint and every growing site, compute and write one row per (checkpoint, site) to `signal_curve.csv`, plus `summary.json`:

- λ spectrum (top 16), noise floor, d_μ.
- U_s, V_s, η_s, ρ_s (with r = `r_max`), and per-neuron V_k.
- The same V_s with **random A** of matched rank and norm (control).
- **Held-out check.**
  - Build the VNA direction Δθ_b on 128 train samples.
  - On 512 samples of the **trigger split**, report the cosine between JΔθ_b and −(K + K_E)g of those samples, and the directional derivative of the held-out mean CE along Δθ_b (in float64).
  - Report the same two numbers for the plain gradient direction and for the random-A direction.
- Fraction of ‖g‖² outside the top-q eigenspace of K (q = 20, via Lanczos or a randomized eigensolver on the matrix-free K).
- Optional flag `--grow-epochs N`: install the selected TINY neurons for real, train N epochs at the checkpoint's LR, and report the change in eval-split accuracy and loss versus plain SGD over the same N epochs from the same state.

### 4.3 `methods/vna.py`

- `VirtualExpansionCache`: refresh, rank, Dörfler and gate logic, with state_dict / load_state_dict.
- `vna_block_correction(model, site, A, batch, cfg) -> (grad_delta: dict[str, Tensor], diag: dict)`.
- Must not mutate model parameters, buffers, `.grad` or train/eval modes (reuse the invariance helpers).

### 4.4 `experiments/run_vna_training.py`

Full 200-epoch recipe from random init, `--arm` in:

| Arm | Description |
|---|---|
| `sgd` | Baseline |
| `vna_adaptive` | Full controller |
| `vna_fixed` | Always active, fixed `r_max`, top-1 site by TINY score, refresh every 10 epochs |
| `vna_random` | Same as adaptive, but A random with matched rank and norm |
| `oonly_gn` | Same schedule as `vna_fixed`, but the correction is −Jᵀ R g, scaled so its norm equals the VNA correction norm at the same step (i.e. no E) |
| `grow` | Real persistent growth: install the TINY neurons at the same epochs, sites and ranks as `vna_fixed`, and keep training them (use `rebuild_sgd_after_growth`). Reference ceiling only |

- Per-epoch log, as JSON lines tagged by arm:
  - train loss/accuracy, eval-split loss/accuracy, LR, wall-seconds;
  - VNA seconds, number of VNA steps, controller state, gate values (G1/G2/G3, ρ, η);
  - selected sites with ranks, λ_1, noise floor;
  - accept-test ratio and refresh events.
- At the end, evaluate **once on the official CIFAR-100 test set** with the final model. **No validation-based checkpoint selection or rollback anywhere.**
- Report parameter count. It must equal the baseline's for every arm except `grow`.

### 4.5 Kaggle notebooks (T4×2)

- `notebooks/kaggle_vna_signal_curve.ipynb`: runs 4.1 and then 4.2 for seed 0.
- `notebooks/kaggle_vna_seed{0,1,2}.ipynb`: 6 arms queued over 2 GPUs.
- **Pin the repo commit hash**, not `main`.
- Resume from attached outputs. Keep only `latest` per arm; archive the logs and JSON only. Disk usage must stay below 5 GB.

## 5. Tests (`tests/test_vna.py`, CPU, tiny Gromo ResNet or a small MLP)

1. J_E JVP/VJP match a finite difference of the expanded forward to rtol 1e-3.
2. Descent: gᵀ(J d) ≥ 0 for the correction, so the update decreases loss to first order. Check on 50 random batches.
3. With μ → 0 on a small full-rank model, J Δθ ≈ −(K + K_E) g restricted to the batch, with relative error < 1e-2.
4. V = Σ_k V_k; U ≥ 0; 0 ≤ η, within numerical tolerance.
5. Gate logic: a synthetic λ below the noise floor gives state ≠ active; three such refreshes give `stopped`.
6. Dörfler selection returns the minimal prefix reaching θ.
7. Invariance: model state, `.grad`, and modes are bit-identical before and after `vna_block_correction` and cache refresh.
8. Resume: training 2 epochs straight equals training 1 epoch, saving, resuming and training 1 more epoch (bitwise on CPU).
9. RNG isolation: `sgd` and `vna_adaptive` with `lambda_vna = 0` give a bitwise-identical trajectory (CPU, 2 epochs, tiny model).
10. Gromo alignment:
    - The A extracted from the counterfactual TINY candidate reproduces the candidate's own virtual forward when B is set to TINY's B. This proves A is read from the correct layer.
    - VNA ignores TINY's B: changing TINY's B does not change the VNA correction.
11. Scope: the correction touches only `residual_path` parameters of selected sites. All other `.grad` values are unchanged by VNA.
12. No forbidden code paths: a grep-style test asserts that the new modules do not import or call `apply_`, `reset_projected_momentum`, rollback/anchor helpers, or `plateau_checkpoint`.

## 6. Acceptance / go-no-go (report, do not tune toward it)

- **Phase 0 passes if**, at some epochs before 150:
  - V_s with TINY is at least 3× V_s with random A;
  - the held-out directional derivative of VNA is negative and at least 3× larger in magnitude than for random A;
  - ρ, U and V decay toward 0 near the end of training, as the theory predicts.
- **Phase 1 passes if** `vna_adaptive` beats `sgd`, `vna_random` and `oonly_gn` on test accuracy (mean over 3 seeds, paired by seed), with total wall-clock overhead ≤ 30%. Also report the fraction of the `grow` arm's gain over SGD that VNA recovers.

## 7. Do NOT

- Do not use rollback, best-of-validation selection, or "scientific escape" metrics.
- Do not compute signals on un-augmented memorized train images in eval mode only. Use the train transform.
- Do not change the training recipe between arms.
- Do not save per-intervention checkpoints.
