# CIFAR pilot: short real-data trial of B1/B2/B3/B4 before full runs

Branch `vna-phase0` (HEAD 688b61d). No commit. Do NOT run pytest. Allowed checks: pyflakes on touched
files, `--help`, and one `--pilot --fake-data --device cpu` run per runner (< 2 min each).

Goal: on Kaggle (T4×2, real CIFAR-100), every CIFAR runner goes through the **real** data path, real
upstream training loop, one real measurement point and gate/summary writing in ~15–20 min total, so a
wiring failure costs minutes, not hours.

## 1. `--pilot` flag on `experiments/stage_b12.py`, `stage_b3.py`, `stage_b4.py`
- Real data (no fake), upstream training loop unchanged, but bounded:
  - at most `--pilot-batches` (default 20) train batches per epoch: wrap the train loader iterator
    (the same pass-through wrapper already used for the measurement buffer) to stop after N batches;
  - 2 epochs (B1/B2/B4), B3: 2 tasks × 2 epochs (reuse the smoke constants);
  - evaluation loaders likewise capped (val/test ≤ 5 batches).
- Exactly one measurement point at the end of epoch 2 (B3: end of epoch 2 of task 2, so the iCaRL KD
  path is exercised), using the real measurement class with reduced cost: ranks (4,), null repeats 2,
  `cg_iters` 200. Rows must still contain every gate key.
- B2 pilot: fetch + hash-verify the real teacher and run the teacher gate on ≤ 5 test batches (report
  only; the full gate stays in the full run).
- Outputs under `outputs/pilot/<setting>/`; gate status forced to `PILOT` (never PASS/FAIL); summary
  records seconds per epoch, seconds per measurement point, peak GPU memory, and a projected full-run
  GPU-hours estimate (epoch time × full epochs × batch ratio + measurement time × number of points).

## 2. Pilot notebook
`scripts/build_kaggle_vna_cifar_pilot.py` → `notebooks/kaggle_vna_cifar_pilot.ipynb`, reusing the
nb2/nb3 setup cells (pinned `--commit`, requirements, `fetch_third_party.sh`, teacher fetch, CIFAR-100
dataset discovery under `/kaggle/input`). GPU0: B3 pilot → B1 resnet8x4 pilot → B1 wrn_16_2 pilot;
GPU1: B2 resnet8x4 pilot → B2 wrn_16_2 pilot → B4 pilot (train then measure). Each job's stdout/stderr
to its own log; a failed job does not stop the other queue. Final cell: one table — setting, status
(OK / FAILED + last 20 log lines), epoch seconds, measurement seconds, peak memory, projected full
GPU-hours, gate keys present — and total projected GPU-hours for nb2 and nb3. No archive needed.

## 3. Report
Short "CIFAR pilot" section in STAGE_AB_IMPLEMENTATION.md. Final message: files touched, pilot CLI.
