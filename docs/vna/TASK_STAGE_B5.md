# Stage B5: loss-of-plasticity runner (measure only)

Prerequisite: TASK_STAGE_B_R2 done. Same rules (no commit, `.venv/bin/python`, `REQUIRE_UPSTREAM=1`
suite green). Read `docs/vna/STAGE_AB_SPEC.md` §B5, `PROMPT_STAGE_AB.md` §2–§4, §6–§7, and the upstream
code under `third_party/loss-of-plasticity/lop/` (`slowly_changing_regression/`, `permuted_mnist/`,
`imagenet/`, `algos/bp.py`, `algos/cbp.py`, `nets/`). Never edit upstream files; a patch only if
unavoidable, with reason, listed in SOURCES.md and your summary.

## 1. Hook strategy (thin, no loop rewrite)
All three upstream loops call `learner.learn(x=..., target=...)` once per step. The runner imports the
upstream experiment function (`expr`, `online_expr`, `repeat_expr`) and runs it **unchanged** inside a
context manager that wraps `learn` of the learner classes (`Backprop`, `ContinualBackprop`, and the
conv variants used by imagenet) at class level, restoring them on exit. The wrapper:
1. calls the original `learn` (training is exactly upstream);
2. pushes the (x, target) it just saw into `MeasurementBuffer` (declared m; regression/MNIST 64,
   ImageNet 100 = one upstream minibatch);
3. tracks the global step and the task index from the upstream schedule (regression: `flip_after`;
   MNIST: `change_after`; ImageNet: examples_per_epoch × num_showings) and, at measurement points,
   runs `MeasurementHook` **after** the step.
Upstream uses numpy/python global RNG (e.g. `np.random.permutation` per task): extend
`MeasurementHook` to save/restore `random`, `numpy` and torch CPU/CUDA RNG states. Upstream output files
go to a run dir under `outputs/stageB/b5/<benchmark>/<arm>/`; if an upstream function writes to a
relative path, run it with that run dir as cwd (no patch).
If any loop cannot be hooked this way, stop and propose an alternative (do not choose yourself).

## 2. Measurements (spec §B5)
- Tasks t ∈ {1,2,5,10,20,50,100,150,200} (only those ≤ tasks run); at the first step of the task and
  mid-task. Data = MeasurementBuffer only (never test data, never future tasks).
- Sites: every consecutive hidden pair (Linear→act→Linear; for the ImageNet CNN conv→act→conv and the
  conv→FC/FC→FC pairs), declared with `HookSite`. Loss: regression `MSE` with the upstream
  `loss_func` passed as `loss_fn`; classification `CE`.
- Ranks: regression net (5 hidden units) {1,2,4}, never above width; MNIST/ImageNet {4,16,64}.
  μ: `mu_rel_default` from Stage A summary if available (CLI arg), else 1e-2, plus 1e-2.
- Per point: U, V, η, ρ, ρ_joint (all sites, one solve), λ₁ with N1/N2 nulls (n=8, p rule),
  tiny_V vs random_frob, subspace_sim, cg_converged; selection default `tiny_V`, `tiny_U` for comparison.
- Extra logs: online accuracy / error per task, dormant-unit fraction (ReDo τ=0.1), effective rank of
  activations, weight magnitude per layer, K spectrum on the measurement batch (top-k eigenvalues and
  the fraction of residual energy in eigen-directions with λ < μ).
- GIR counterfactual (measure only): units selected by CBP utility (same formula/maturity as
  `algos/cbp*.py`, imported) and ReDo dormant τ=0.1; m = their count. `V_swap_tiny` (A from tiny_V,
  rescaled to the init row norm, placed at those units, B=0), `V_swap_random` (A from the upstream init
  distribution, same normalisation), ratio, and η before/after swap. Not in gate G.
- Output CSV rows carry `schema_version="stage1d-v1"`, benchmark, arm, task, step, code_commit, seed.

## 3. Arms, baseline gate, alignment
- Arms per benchmark: `bp` (plain backprop, upstream cfg) and `cbp` (upstream cfg). Measurement hook
  only on `bp`; `cbp` is only for the baseline gate. Use the upstream cfg JSON values; where the cfg
  lists a sweep (e.g. step_size list), use the value the upstream README/plots use for the paper
  figure; if that is ambiguous, stop and ask.
- Screening length: regression full sequence; MNIST first 200 tasks; ImageNet first 200 tasks.
- **Baseline gate** (`b5_baseline_gate.json`, computed before measurements are trusted): bp online
  accuracy (MNIST/ImageNet) declines from early to late tasks (mean of tasks 1–10 vs last 10, with the
  numbers), regression error rises for bp; cbp holds (late ≥ early − small tolerance; record numbers).
  FAIL → measurements are flagged invalid in gate.json and the notebook prints it loudly.
- **T-align test** per benchmark (CPU, fixed seed, tiny config: e.g. regression 2 flips × 50 steps,
  MNIST 2 tasks × 50 samples with a 784-wide subset or fake MNIST of the right shape, ImageNet 2 tasks
  × 2 batches with fake 32×32 data): run upstream with and without the hook; final params + optimizer
  state + CBP internal state bit-identical; also the upstream returned logs identical.
- **T-params**: parameter count unchanged at every point (asserted inside the hook).
- Spec test 14: data order/permutation stream matches upstream with fixed seed (first 3 tasks
  bit-for-bit, using fake data if real MNIST is absent locally).
- Gate G: `diagnostics/vna_gate.py` B5 config (second half of tasks run, train-acc n/a) → write
  `outputs/stageB/gate.json` entries per benchmark.

## 4. Data and notebook
- MNIST: upstream `load_mnist.py` downloads via torchvision; on Kaggle prefer an attached MNIST dataset
  if present, else download; produce the exact pickle upstream expects (run their script, no rewrite).
- ImageNet-32: upstream expects per-class `.npy`; detect a Kaggle dataset path (constant
  `IMAGENET32_DIR`); if absent, record "skipped: data unavailable" in gate.json and continue.
- Builder `scripts/build_kaggle_vna_stage_b_nb1.py` → `notebooks/kaggle_vna_stage_b_nb1.ipynb` (spec
  Notebook 1 without A and without B6): pinned `--commit`, `pip install -r requirements-upstream.txt`,
  `fetch_third_party.sh`, full tests with `REQUIRE_UPSTREAM=1`; GPU0: ImageNet bp → cbp; GPU1:
  regression bp → cbp, then MNIST bp → cbp. Smoke (2 tasks per config) prints projected GPU-hours,
  stop over `GPU_HOURS_BUDGET` unless `ALLOW_OVER_BUDGET`. Hourly archive + resume (per arm, task
  granularity; resume must restore RNG so the upstream stream continues bit-identically — test it on
  CPU), same disk cap. Last cell: baseline gate, gate G table, ρ vs task plots, GIR ratio table.
- CLI: `python -m experiments.stage_b5 {regression,mnist,imagenet} --arm {bp,cbp} [--smoke]
  [--fake-data] --out ...`.

## Report
Append "Stage B5" to STAGE_AB_IMPLEMENTATION.md (hook point per benchmark, sites, m, data rule, any
paper/code discrepancy found). Full suite with `REQUIRE_UPSTREAM=1`; counts. Changed files.
