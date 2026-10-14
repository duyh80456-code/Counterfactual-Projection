# nb1 fixes from the first Kaggle run

No commit. Do NOT run pytest. Allowed: pyflakes, rebuilding notebooks, compiling generated cells, one
`python -m experiments.stage_b5 mnist --arm cbp --smoke --fake-data --device cpu` (< 2 min).

Observed on Kaggle:
```
Stage A failed: No Phase-0 checkpoints: /kaggle/input/vna-phase0-seed0/vna_phase0_seed0/recipe
GPU1 394.760 projected GPU-hours including smoke      # MNIST cbp
CPU 10.649 projected CPU-hours including smoke        # regression
```

## 1. Phase 0 checkpoint discovery (nb1 and the standalone Stage A notebook)
Replace the fixed `PHASE0_DIR/"recipe"` with discovery:
- (Real case: the input is the output of the Kaggle notebook `kaggle_vna_signal_curve`, mounted under `/kaggle/input/<notebook-slug>/vna_phase0_seed0/recipe/epoch_*.pt`, next to `vna_phase0_seed0.tar.gz`; prefer the extracted directory.)
- search `/kaggle/input` recursively (follow links) for directories containing `epoch_*.pt`; accept a
  directory only if one of its `epoch_*.pt` loads (CPU, `weights_only=False`) with `kind == "vna_recipe"`;
- if none, look for `*.tar.gz` / `*.tgz` / `*.zip` under `/kaggle/input` whose members include
  `recipe/epoch_*.pt`, extract safely (reject absolute/`..` paths) to `/kaggle/working/phase0/`, and
  search again;
- exactly one match → use it; several → prefer the one whose epochs cover 20..200 and print the choice;
  none → print the `/kaggle/input` tree (depth 4, with file sizes) and raise.
Print the chosen directory and the epoch list before Stage A starts.

## 2. Projection from steady-state step time
The B5 smoke runs 100 steps (2 tasks × 50), so startup time dominates and is multiplied by 12M steps.
- In `LearnHook`, record wall-clock per step for the smoke (monotonic, after CUDA sync on GPU every
  k steps, not every step) and compute the median step time over the **second** smoke task only.
  Raise MNIST smoke `change_after` to 500 (2 tasks × 500 steps; still minutes) so the estimate is stable.
- Projection = steady step time × full steps (+ measurement points × measured seconds per point for bp)
  + measured startup once. Print per arm: startup s, ms/step, full steps, measurement s/point, projected
  hours. Same fix for the regression CPU lane (100 data points is too few: use 2 flips × 500 for smoke).
- Lane totals as now; budget stop uses the corrected numbers.

Rebuild nb1 and the Stage A notebook. Final message: files touched.
