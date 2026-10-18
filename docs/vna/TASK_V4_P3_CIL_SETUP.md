# Phase 3: main CIL setup (§3) and metrics (§5) for iCaRL arms

Read `docs/vna/TASK_V4_COMMON.md`, prompt §3, §4 (Đợt 1 arms `icarl`, `icarl_full`), §5.
Reuse `experiments/stage_b3.py`, `experiments/competitors.py` (cil_recipe/cil_runtime),
`adapters/icarl_hooks.py`, `adapters/pilot_icarl.py`, resume/checkpoint utilities.

- Setup exactly as the §3 table: PyCIL iCaRL original config, B0 Inc5 (20 tasks), memory 2000
  `fixed_memory=False`, epochs 100 first task / 50 later with lr-milestones scaled proportionally
  from the original config (rounded), dataset root `/kaggle/input/datasets/nlnk1607/cifar100`.
- Class order: `superclass` (main) — each task = the 5 fine classes of one coarse class, the 20
  superclass order permuted by seed, the 5 classes inside permuted by seed; `random` (control) =
  PyCIL default seed order. Read `coarse_labels` from the CIFAR-100 python `train` pickle; if
  absent, use the official fine->coarse table embedded in code and verify against
  `meta['coarse_label_names']` when available. No local CIFAR here: tests use the embedded table.
- Backbones: PyCIL `resnet32` (BN) and `resnet32_nobn` = SkipInit (De & Smith 2020): remove BN,
  conv with bias, learnable scalar alpha init 0 multiplying each residual branch. lr halving rule
  for noBN only if task-1 loss is NaN — implement as a recorded config flag (`nobn_lr_scale`) set
  by the runner/notebook, applied to all noBN arms, logged in protocol.json. Log parameter count.
- Arms `icarl` and `icarl_full` (phase-2 algorithm, r = 16 fixed, §2.5 schedule).
- Metrics §5 for every arm after each task: CNN and NME top-1 total/old/new, learning accuracy,
  forgetting (per seen class, max past minus current), average incremental and last accuracy,
  §2.6 diagnostics via `diagnose()` on the fixed probe batch (256 test samples of seen classes,
  fixed seed); feature accuracy (Kim et al. CVPR 2023 linear probe on frozen features, train on
  all seen classes' train data, eval on test) at tasks 5, 10, 15, 20; GPU-hours. Files per run:
  summary.json, tasks.json, continual_eval.csv, diagnostics.json, interventions.json (own arms
  only), protocol.json, status.json. Final table helpers: mean ± std over seeds, paired delta vs
  icarl, seeds won; stability–plasticity figure (x final forgetting, y mean learning accuracy).
- `validate_result` / collectors updated for the new arms and files.

Tests: 7.5 (superclass partition: 20 tasks x 5 fine classes of one coarse label, union = 100,
order changes with seed), 7.6 (noBN ResNet-32 forward+backward finite on one batch, param count
logged), plus a 2-task x 1-epoch fake-data smoke of `icarl` and `icarl_full` (CPU, tiny) that
checks all output files and that λ=0 / method off is bitwise equal to `icarl`. Do not commit.
