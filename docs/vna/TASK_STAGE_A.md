# Stage A: diagnostics on Phase 0 checkpoints (+ gate module)

Branch `vna-phase0`. `.git` is read-only: do NOT commit. Use `.venv/bin/python`. Keep every existing
`tests/test_vna*.py` passing unchanged.

Read first: `docs/vna/STAGE_AB_SPEC.md` (§Giai đoạn A, §Cổng G, §3 tests 1–5, 9, 11, 15),
`docs/vna/PROMPT_STAGE_AB.md` §3, `docs/vna/DESIGN_V2.md` §5–§7, `docs/vna/STAGE2_REPORT.md`.
The spec wins on numbers; this file wins on code organisation and on the decisions in §0.

## 0. Decisions already taken (do not re-decide)

1. **Do not modify `diagnostics/signal_curve.py` output.** Stage A lives in a new module
   `diagnostics/stage_a.py` (CLI: `python -m diagnostics.stage_a {null,mu,selection,joint,solver,summary,all}`).
   Every row/JSON it writes carries `schema_version="stage1d-v1"`, `code_commit`, `seed`, `device`,
   `wall_seconds`. Loading a file with another schema_version raises.
2. **Residual-driven selection.** Nulls must re-run selection on the null residual, so add a
   cotangent-driven TINY (`tiny_U`) closed form in `methods/vna_select.py`:
   `tiny_U_from_cotangent(model, site, inputs, cotangent, rank, cfg)` returning `(A, eigenvalues)`
   built by back-propagating the given output cotangent (no labels). Test: with the true mean-CE
   cotangent it reproduces `experiments.vna_support.tiny_candidate` eigenvalues (rtol 1e-5, float64
   tiny model) and the same A subspace (subspace_sim ≥ 1 − 1e-6). Real and null statistics in A1 both
   use this function, so they are on one scale. `tiny_V` already accepts `cotangents=`; use it likewise.
3. **One normalisation.** Every A (tiny_U, tiny_V, greedy_V, random_trace, random_frob, null A's) goes
   through `normalize_A` and then a single entry point `prepare_A(model, site, A, kind, ...)` that logs
   the normalisation constant. `random_frob`: random A rescaled so ‖A‖_F equals the normalised tiny_U
   A of the same site/rank (exact). `random_trace`: existing `trace_matched_random_A`.
4. **Joint K_E (A4).** New `methods/vna_joint.py::joint_correction(model, sites, A_by_site, batch, cfg,
   o_params=None)`: K over `o_params` (default: all trainable parameters of the model), K_E = Σ_site
   K_E^site (each with its own virtual B=0 at its site), one shared CG solve, same μ rule
   μ = μ_rel·tr(K)/n, same GGN metric H and 1/n normalisation as `vna_ggn.py`. Returns d, V, ρ, CG
   record. Matrix-free (JVP/VJP via torch.func over the full model); never materialise J.
   Required test (spec test 3): with one site and `o_params` = that block's parameters it equals
   `vna_block_correction` (V, ρ, d; relative error ≤ 1e-6, float64 tiny model). Report in joint.csv:
   ρ_joint (o = all params), Σρ_site, max ρ_site.
5. **Gate module now** (pure logic, used later by Stage B): `diagnostics/vna_gate.py` with module-level
   constants `RHO_MIN=0.3, RHO_FRAC=0.5, N1_FRAC=0.5, TINYV_FRAC=0.7, TRAIN_ACC_MAX=0.99, GATE_RANK=16`
   and `evaluate_gate(rows, setting_cfg) -> dict`. `setting_cfg` supplies a window predicate
   (which measurement points count) and `train_acc_applicable: bool`. Built-in configs:
   B1/B2 `epoch < 210`; B4 `epoch < 150`; B3 `task ≥ 2 and not end-of-task`; B5 `task in second half
   of the tasks run`, train-acc condition not applicable (recorded as `"n/a"`, gate uses the other
   three). Only rows with `cg_converged` count. Output keeps raw fractions next to PASS/FAIL.
   Test 9: synthetic rows give the expected PASS/FAIL; a test locks the constant values.

## 1. Measurements (all on Phase 0 recipe checkpoints, ResNet-18/CIFAR-100, batch 64)

Data/model/checkpoint loading: reuse what `signal_curve.compute` uses (`build_model`, `setup_data`,
`augmented_batch`, `dedicated_generator` purposes, `vna_recipe` checkpoint kind, 45k/5k split).
Defaults: 8 sites, stat batch 64×4, `cache_block_input=True`, r=16, μ_rel=1e-2 unless stated.
Each measurement uses its own named generator purpose; never touch global RNG.

- **A1 nulls** → `null.csv` cols `epoch, site, null_kind, stat, real, null_mean, null_max, p_value,
  significant`. Kinds N0 (shuffled labels, also report λ/‖g‖²), N1 (permute residual rows across
  samples, inputs fixed), N2 (per-sample N(0,I), project to zero-sum, rescale to ‖g_i‖). n=8 each,
  seed per replicate. For each null: select A from the null residual, then λ₁, V, ρ with the same
  μ and rank. Stats: λ₁ and ρ. p = (1+#{null ≥ real})/(1+n); significant = p ≤ 1/9.
  Epochs 20,40,…,200 × 8 sites. Helpers `null_residual(g, kind, generator)` and
  `empirical_p(real, nulls)` are unit-tested (spec tests 1, 2).
- **A2 μ sweep** → `mu_sweep.csv`: epochs 40,80,100,120,140,160; top-3 sites by V (same selection
  stream as signal_curve) + `joint`; μ_rel ∈ {1e-1,1e-2,1e-3,1e-4}; exact solver, cg_iters=1000,
  cg_tol=1e-6. Cols: U, V, η, ρ, realized_gain, held-out derivative, cg_iters_used, cg_converged,
  cg_rel_residual. Pre-registered rule (constant in code): if ρ(1e-3) ≥ 3·ρ(1e-2) at ≥ 50% of
  converged points → `mu_rel_default=1e-3`, else 1e-2. Written to summary.json.
- **A3 selection** → `selection.csv`: kinds tiny_U, tiny_V, greedy_V, random_trace, random_frob;
  rank {4,16,64}; epochs 40,80,120,140,160; top-3 sites; 3 independent stat-batch draws (mean ± std).
  V and ρ on a correction batch disjoint from the stat batch; held-out loss derivative (fixed 512 eval
  images); normalisation constant; `subspace_sim` (split-half A₁,A₂ → orthonormal row bases Q₁,Q₂,
  ‖Q₁ᵀQ₂‖_F²/r) plus its random expectation r/d_in. Test 5: subspace_sim(A,A)=1; two independent
  random A ≈ r/d_in.
- **A4 joint** → `joint.csv` at epochs 100,120,140,160, μ_rel ∈ {1e-2,1e-3}.
- **A5 bug** (V_TINY/V_random = 0 at epoch 100 in Phase 0). Investigate before coding. Leads: tiny
  path uses the gain guard while random uses `apply_gain_guard=False`; LR-boundary checkpoint state;
  nonfinite random; division. Fix at the root, add a regression test (spec test 11) that fails on the
  old code, document cause in `docs/vna/STAGE_A_REPORT.md`. If the fix changes any Phase 0 number,
  say exactly which.
- **A6 solver** → `solver.csv`: run `diagnostics.vna_solver_validation --mode recipe` at checkpoints
  40,100,160 with the training config (`experiments/vna_training_config.py`); add cos(d₁₆,d₁₂₈) and
  the disjoint-batch-16 cosine; cols K-FAC/exact cosine per site, overhead = n_site·cost/vna_every,
  ‖λd‖/‖g‖. Table only, no decision.
- **summary** → `summary.json`: mu_rel_default (+ the raw fractions), fraction of points with
  tiny_V > random_frob (per rank), fraction with significant_N1 (λ₁ and ρ), A5 status.

Invariance: every measurement entry point asserts parameters, buffers and train/eval mode are
bit-identical before/after (extend the existing non-destructive test to `stage_a`), and that the
parameter count is unchanged.

## 2. Lemma test (spec test 15)

`tests/test_vna_lemma.py`: random float64 J (n×p), J_E (n×q), μ → 0 (1e-12): the GN velocity of the
grown model projected back via J J⁺ equals the fixed model's GN velocity (rel err ≤ 1e-6); with μ>0
the gap is O(μ) (check it shrinks ~linearly over μ ∈ {1e-2,1e-3,1e-4}).

## 3. Resume, CLI, notebook

- Resumable per (measurement, epoch): one completion JSON per unit under the output dir; reruns skip
  completed units; identity check (schema, args, checkpoint hash) on resume. `--smoke` + `--fake-data`
  path runs all six measurements on CPU in < 2 min (add a test).
- `scripts/build_kaggle_vna_stage_a_notebook.py` → `notebooks/kaggle_vna_stage_a.ipynb`, modelled on
  `scripts/build_kaggle_vna_signal_curve_notebook.py`: `REPO_COMMIT` pin (`--commit`), input
  dataset path constant `PHASE0_DIR` (extracted `vna_phase0_seed0.tar.gz`, `recipe/epoch_*.pt`), runs
  `tests/test_vna*.py` first, GPU0: A1 then A3; GPU1: A2, A4, A6 (A4 before A2 because A2 needs
  `joint`); hourly archive + same disk cap as the current notebooks; last cell runs `summary`, prints
  the tables and plots ρ(μ), V by selection kind, null p-values; archives `outputs/stageA/`.

## 4. Report

Create `docs/vna/STAGE_A_REPORT.md` (implementation notes, A5 cause, how to read each CSV; results
section left as TODO). Run all `tests/test_vna*.py` and the new tests; report counts and runtime.
Final summary: list every file you created/modified. Remove your own /tmp outputs (disk is nearly full).
