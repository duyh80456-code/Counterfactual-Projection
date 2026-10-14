# Stage B remaining: B0 + B2 (KD) + B1 (tiny CE) + B4 (ResNet-18 ×0.25) + notebooks 2/3

Branch `vna-phase0` (HEAD ba7ed58). No commit. `.venv/bin/python`,
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` for anything importing RepDistiller.
Spec: `STAGE_AB_SPEC.md` §B0, B-recipe, B1, B2, B4, "Đo trong B", gate. Reuse the B3/B5 patterns
(`adapters/icarl_hooks.py`, `diagnostics/icarl_measure.py`, `experiments/stage_b3.py`) and the Stage B
core. Upstream: `third_party/RepDistiller` (`train_student.py`, `train_teacher.py`, `helper/loops.py`,
`helper/util.py`, `dataset/cifar100.py`, `distiller_zoo/KD.py`, `models/`), `third_party/One-Shot-TAS-CCIL`
(`dual_growth/backbones/hybrid_resnet18.py`, `dual_growth/adapters`).

**Testing policy:** do NOT run pytest. Write the tests listed in §5 (tiny, CPU) but leave them for the
Kaggle test cell. Your only allowed checks: `pyflakes` on new files, `python -m <runner> --help`, and ONE
`--smoke --fake-data --device cpu` run per runner (must finish in < 2 min). The supervisor reviews by
reading, so keep the code small and explicit.

## 1. Decisions (final; do not stop on these)
- **Architectures (B0):** import them from `third_party/RepDistiller/models` (`model_dict`); do not
  copy into `models/cifar_repdistiller.py` (spec said copy; "import, don't copy" from the agent prompt
  wins — note this in the report). Same for the KD loss: `distiller_zoo.KD.DistillKL`.
- **Teachers:** URLs from RepDistiller `scripts/fetch_pretrained_teachers.sh`; SHA256 (verified
  2026-10-15):
  - `resnet32x4_vanilla/ckpt_epoch_240.pth` 22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f (59,554,438 B)
  - `wrn_40_2_vanilla/ckpt_epoch_240.pth` 41100a8e8478b326645eb9aab09c18fc8ed34c1ab4004d658cb652125a81ca18 (18,113,193 B)
  `scripts/fetch_teachers.py` downloads into `third_party/RepDistiller/save/models/<name>/`, verifies size
  and hash, stops on mismatch. Teacher gate (official test set, the only allowed use): resnet32x4 ≥ 79.0,
  wrn_40_2 ≥ 75.0 top-1, evaluated with RepDistiller's own `validate`; written to `teacher_gate.json`.
- **Training entry points unchanged:** B2 = `train_student.py --model_s <s> --path_t <t> --distill kd
  -r 0.1 -a 0.9 -b 0 --trial 0` (defaults already = CRD recipe: 240 ep, lr 0.05, decay 150/180/210,
  bs 64, wd 5e-4, kd_T 4). B1 = `train_teacher.py --model <s> --trial 0` (same defaults, plain CE).
  Run their `main()` in-process (set `sys.argv`), inside a working dir under `outputs/stageB/<setting>/`.
- **45k/5k split (spec):** at runtime replace the module-level `get_cifar100_dataloaders` used by the
  entry point with a wrapper that calls the original (same transforms, batch size, workers) and then
  (a) restricts the train set to the existing 45k indices (`experiments.vna_support.setup_data` split,
  seed 0) via `Subset`, keeping the upstream shuffle; (b) uses the 5k holdout (test transform) as the
  `val_loader` the loop evaluates each epoch. Evaluate the official test set **once** after the last
  epoch (final model, not "best"), logged as `final_test_acc`. Record the deviation (upstream trains on
  50k) and compare to published (resnet8x4 72.50 / wrn_16_2 ≈ 73.3 student CE; KD numbers from the CRD
  table) for reference only, flag if > 2pp off.
- **Hook:** wrap `adjust_learning_rate` (called once at the start of every epoch by both entry points)
  to measure the state after epoch e-1, at e-1 ∈ {20,40,…,240} ∪ {148,178,208}; the final epoch 240 is
  measured once after `main()` returns (same model). Measurement data = `MeasurementBuffer` filled by a
  pass-through train-loader iterator (m = 128, two upstream batches). B2 residual/metric:
  `KDRepDistiller(alpha=0.1, beta=0.9, T=4)` with `loss_fn` built from the imported `DistillKL` and
  `CrossEntropyLoss` and teacher logits from the loop's own teacher (eval, no grad). B1: `CE`.
- **Sites:** every basic block (conv→BN→ReLU→conv) of resnet8x4 via `HookSite`; WRN blocks are
  pre-activation: site = conv1 → (bn2, relu2) → conv2 in that order (read `models/wrn.py`). Ranks
  {4,16,64}, never above width. μ: `--mu-rel-default` plus 1e-2. Same row schema/keys as B3 so
  `vna_gate.py` works (B1/B2 window epoch < 210; train-acc from the loop's final train accuracy).
- **B4:** no external training repo — our Phase 0 recipe (`experiments/train_recipe_checkpoints.py`, 200
  ep, lr 0.1, decay 100/150, bs 64, 45k/5k) with the model built by the `dual_growth` backbone at width
  0.25 (import; read `hybrid_resnet18.py` / `growable_resnet18.py` to pick the class whose Gromo blocks
  our existing Gromo site path supports; T-params: parameter count = the `dual_growth` count). Measure
  offline from checkpoints (like Phase 0's watcher) at epochs 20,…,200 ∪ {98,148}: make the recipe also
  save those two epochs (training math unchanged). Gate window epoch < 150.

## 2. Runners
- `experiments/stage_b12.py {b1,b2} --student {resnet8x4,wrn_16_2} [--teacher ...] [--smoke]
  [--fake-data] [--device] [--out] [--data-dir] [--mu-rel-default]`; B2 pairs fixed:
  resnet32x4→resnet8x4, wrn_40_2→wrn_16_2. Writes measurements.csv, epochs.csv, gate.json,
  reproduction.json, summary.json (schema stage1d-v1, code_commit, seed, device).
- `experiments/stage_b4.py {train,measure} [--smoke] [--fake-data] ...`.
- Smoke = 2 epochs; fake data = tiny random CIFAR-shaped tensors (pattern of stage_b3 `runtime`).

## 3. Notebooks
- nb2 (`scripts/build_kaggle_vna_stage_b_nb2.py`): replace the GPU1 placeholder with B2
  resnet32x4→resnet8x4, then B4 train, then B4 measure. Teacher fetch + teacher gate before B2.
- nb3 (`scripts/build_kaggle_vna_stage_b_nb3.py` → `notebooks/kaggle_vna_stage_b_nb3.ipynb`): GPU0 B1
  resnet8x4 then B1 wrn_16_2; GPU1 B2 wrn_40_2→wrn_16_2. Same setup/smoke/budget/archive cells as nb2.
- Both: pinned `--commit`, test cell runs the new test files, final cell prints teacher gate,
  reproduction table, gate G per setting, ρ-vs-epoch plots. Merge per-setting gate.json into
  `outputs/stageB/gate.json`.

## 4. Report
Append "Stage B0/B1/B2/B4" to STAGE_AB_IMPLEMENTATION.md: hook points, decisions §1, deviations from
spec (import instead of copy, 45k split), published reference numbers with sources.

## 5. Tests to write (not run)
T-align for b1 and b2 (2 epochs, fake data, tiny batch, CPU: params + optimizer bit-identical hook on/off);
architecture param counts equal `model_dict` instantiation; WRN pre-activation site order zero-B
invariance; KD residual equals autograd of imported `DistillKL` + CE; teacher hash check with a fake
file; B4 param count equals `dual_growth` and zero-B invariance.
