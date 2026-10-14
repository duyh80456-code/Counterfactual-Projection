# Stage 2a task: training runner + Kaggle notebook (solver-agnostic, validation-first)

Read `docs/vna/DESIGN_V2.md`, `docs/vna/STAGE2_REPORT.md`, `docs/vna/SPEC.md` §1.1/§4.4/§7.
Branch `vna-phase0`, HEAD = Stage 2-core. `.git` read-only: do NOT commit. Use `.venv/bin/python`.
Do not change Phase 0 code paths (`diagnostics/signal_curve.py`, its notebook/builder, resume identity).

Key Stage 2-core findings that shape this task: K-FAC agrees poorly with exact on the CPU toy
(cos 0.01–0.23), and the NGD control is highly aligned with exact VNA (cos ≈0.85). Therefore the solver
is a CLI choice and the notebook validates solvers on a REAL checkpoint before training.

## 1. `experiments/run_vna_training.py`
- Recipe identical to `experiments/train_recipe_checkpoints.py` (same splits, SGD 0.1/0.9/5e-4,
  MultiStep 100/150, 200 epochs, batch 64, same throughput flags). Random init. Final model evaluated
  ONCE on the official CIFAR-100 test set. No rollback, no best-of-validation, no per-intervention
  checkpoints. Keep only `latest.pt` per arm (full state incl. VNA cache/controller/K-FAC factors/
  warm starts/all dedicated generators) — bitwise CPU resume.
- `--arm`: `sgd`; `sgd_cm` (equal wall-clock: `--epochs-scale` default 1.3 applied to epochs AND
  milestones); `vna` (with `--solver {exact,kfac,lite}` and `--selection {tiny_U,tiny_V,greedy_V}`);
  `vna_random` (trace-matched random A, same solver/schedule); `ngd` (same solver, norm-matched to
  what the VNA correction norm would be — define precisely: compute the VNA correction at the same
  step and site, use only its norm; log both).
- Correction applied every `--vna-every` steps (default 10) to `p.grad` after backward, before
  `optimizer.step()`; BN tangent projection on by default; skip + log if projected realized gain ≤ 0
  or nonfinite. Sites: `--sites` explicit list or `auto` = top-`max_sites` by V at refresh (Dörfler
  θ per SPEC); refresh A/factors/μ every `--refresh-epochs` (default 5) on dedicated batches. No G1/G2/
  accept-test gating in this stage (always active) — log the would-be gate statistics only.
- Paired RNG: SGD loader stream identical across arms (test: `vna` with lambda 0 is bitwise `sgd` on
  CPU for 2 epochs). VNA/selection/statistics/K-FAC MC labels/Hutchinson use dedicated generators.
- Per-epoch JSONL: train loss/acc, eval-split loss/acc, LR, epoch seconds, VNA seconds by component,
  #corrections applied/skipped, mean ρ/V/realized gain, selected sites+ranks, per-layer ‖W‖,
  peak memory. End: test acc/loss, params (must equal sgd), GPU-hours.
- Tests (CPU tiny model): resume bitwise; lambda-0 identity; param count unchanged; no forbidden
  code paths (grep test as in Stage 1); ngd norm-matching.

## 2. `diagnostics/vna_solver_validation.py` on real checkpoints
Extend (without breaking the toy mode) to load a Phase 0 recipe checkpoint (`epoch_NNN.pt` model-only
is fine) + CIFAR, pick sites (top-3 by tiny_V V), and report cos(kfac/lite/euclidean/ngd vs exact) per
parameter group across the μ sweep, realized gains, and per-correction cost in SGD-step units on GPU.
Exact must converge (report residuals).

## 3. Notebook `notebooks/kaggle_vna_training_seed{0,1}.ipynb` + builder
- Same Kaggle conventions as the Phase 0 builder (pinned REPO_COMMIT, github_token, Gromo pin, CIFAR
  discovery incl. official test, resume from /kaggle/input archives, hourly atomic archive, <5 GB).
- Cell A (optional, default ON if a Phase 0 checkpoint is attached): GPU solver validation on
  epochs {40, 100, 160}; print table; set `SOLVER` automatically = `kfac` if median whole-vector
  cos(kfac, exact) ≥ 0.7 across those rows, else `exact` if its measured cost ≤ 30 SGD units, else
  `lite`; print the decision. A constant `SOLVER_OVERRIDE` (default None) wins if set.
- Cell B: two-GPU queue for arms [sgd, vna(tiny_V), ngd, vna_random, vna(tiny_U), sgd_cm] in that
  priority order; each arm a fresh process; resume skips finished arms.
- Final: table (test acc, Δ vs sgd, cost columns), curves.

## 4. Report
`docs/vna/STAGE2_REPORT.md` new section "Stage 2a": files, run instructions, runtime estimate per arm
on T4 (per solver), disk, open questions. Run tests/test_vna*.py and the full suite; report counts.
