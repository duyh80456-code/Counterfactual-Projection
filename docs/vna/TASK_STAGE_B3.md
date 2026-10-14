# Stage B3: class-incremental iCaRL (PyCIL) runner, measure only

Branch `vna-phase0` (HEAD d5e4e9e). No commit. `.venv/bin/python`. Upstream: `third_party/PyCIL`
(read `main.py`, `trainer.py`, `models/icarl.py`, `models/base.py`, `utils/data_manager.py`,
`utils/inc_net.py`, `convs/cifar_resnet.py`). Spec: `STAGE_AB_SPEC.md` §B3; reuse the B5 patterns
(`adapters/lop_hooks.py`, `experiments/stage_b5.py`, `diagnostics/lop_measure.py`) and the Stage B core.

**Testing policy (new):** write only a few tiny CPU tests for the new code (listed in §4) and run only
those. Do NOT run the full suite. Keep the code small and readable; the supervisor reviews by reading.

## 1. Decisions already taken (do not re-decide, do not stop on these)
- Protocol = `exps/icarl.json` + `models/icarl.py` constants at the pinned commit, unchanged. Single GPU:
  `device=["0"]` (no DataParallel). `num_workers` may be overridden at runtime by assigning the module
  constant (`models.icarl.num_workers`): 0 for CPU tests, 4 on Kaggle. No patch.
- Reproduction gate: PyCIL pipeline unchanged (full train set, its own test evaluation), compare the
  mean of the 10 per-task top-1 accuracies (the "average accuracy" PyCIL logs) to the published
  **64.42** (PyCIL paper, iCaRL CIFAR-100 B0 Inc10; consistent with `resources/cifar100.png`); pass if
  within ±1.5pp. Also log the last-task accuracy (published ≈ 49.5). Record both reference values and
  their source in the gate JSON.
- No 10% eval split (it would change the upstream data pipeline and break the reproduction). Per-task
  logs use what PyCIL computes (seen-class test accuracy per task, average accuracy) plus our forgetting
  computed from those numbers, and train loss/accuracy per epoch. Measurement data never comes from test.
  Note this deviation from spec §B3 in STAGE_AB_IMPLEMENTATION.md for the supervisor.

## 2. Hook (thin)
- Batches: replace `models.icarl.DataLoader` at runtime with a pass-through subclass whose iterator
  yields the upstream items unchanged and copies the last seen train batches into `MeasurementBuffer`
  (m = 128 = one upstream batch; stat/correction batches drawn from the buffer as in B5). It must not
  consume any RNG or change worker seeding.
- Measurement points: wrap the scheduler's `step` (called once per epoch by the upstream loop) to run
  `MeasurementHook` after epochs {1, 10, mid, last} of every task (task 1 included). Model is measured
  with eval-mode BN via the existing functional path; train/eval mode, params, buffers and all RNG
  restored. Restore the wrappers on exit.
- Loss: task 1 `CE`; tasks ≥ 2 `ICaRLPyCIL(T=2, known_classes=self._known_classes)`, with `loss_fn`
  being the upstream expression (CE over seen classes + imported `_KD_loss`), teacher = `self._old_network`
  logits (eval, no grad). Logits are `network(x)["logits"]`.
- Sites: every basic block of `convs/cifar_resnet.py` resnet32 (conv→BN→ReLU→conv) via `HookSite`;
  ranks {4, 8, 16}, never above the block width. The fc head grows per task: T-params asserts the
  parameter count is unchanged **within** a measurement, not across tasks.
- Rows also give ρ computed separately on exemplar samples (targets < known_classes) and new-task samples.
- Gate G config B3 (task ≥ 2, not end-of-task) from `vna_gate.py`; train-acc condition uses the final
  task's train accuracy.

## 3. Runner and notebook
- `python -m experiments.stage_b3 [--smoke] [--fake-data] --out outputs/stageB/b3`: runs PyCIL's own
  entry (`trainer.train(args)` or equivalent) inside the hook context. Smoke = 2 tasks × 2 epochs
  (assign the module epoch constants at runtime, smoke only). Resume at task granularity if PyCIL
  allows it cheaply; otherwise rely on the hourly archive and document that a crash restarts the run.
- Builder `scripts/build_kaggle_vna_stage_b_nb2.py` → `notebooks/kaggle_vna_stage_b_nb2.ipynb`
  (pinned `--commit`, same setup cells as nb1). For now GPU0: B3; leave a clearly marked placeholder
  cell for GPU1 (B2 then B4, added later). Smoke + projected GPU-hours + budget stop, hourly archive,
  final cell: reproduction gate, gate G, ρ vs task/epoch plot.

## 4. Tests (tiny, CPU, run only these)
1. T-align: 2 tasks × 2 classes-per-task on fake 32×32 data (tiny class count, 1 epoch, batch 8),
   hook on vs off → params, optimizer state and old-network params bit-identical.
2. Spec 12: class order with seed 1993 equals PyCIL's `DataManager` order (call it, compare); herding
   memory per class = 2000 / seen classes (check on the fake run's `_data_memory` sizes); loss equals
   imported `_KD_loss` + CE (already partly covered; extend if needed).
3. HookSite on resnet32: zero-B output unchanged and parameter count unchanged.

## Report
Append "Stage B3" to STAGE_AB_IMPLEMENTATION.md (hook points, decisions §1, deviations). List files.
