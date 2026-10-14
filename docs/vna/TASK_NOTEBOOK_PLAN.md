# Notebook plan (supervisor, 2026-10-15, revised)

No commit. Do NOT run pytest. Allowed: pyflakes, building the notebooks, compiling every generated
code cell. Run order on Kaggle: nb1 first (it produces μ), then nb2 and nb3 (separate sessions).
Every process uses 2 dataloader workers. After the smoke, print projected GPU-hours **per lane**.

## nb1 (`scripts/build_kaggle_vna_stage_b_nb1.py`)
| lane | jobs |
|---|---|
| GPU0 | Stage A (a5 → null → selection → joint → mu → solver → summary, same CLI/resume/identity rules as the Stage A notebook; needs the Phase 0 checkpoint dataset) → MNIST **bp** with `mu_rel_default` from the Stage A summary just produced |
| GPU1 | MNIST **cbp** (rr 1e-4), starts immediately (not measured, needs no μ) |
| CPU process | regression: the five cbp arms start immediately; the bp arm starts after Stage A's summary exists and uses its `mu_rel_default` |
- Upstream regression is CPU-fixed: run it as a separate CPU subprocess (`--device cpu`, no
  `CUDA_VISIBLE_DEVICES`), concurrently with both GPU lanes; limit its torch threads so it does not starve
  the GPU lanes' loaders (e.g. `OMP_NUM_THREADS=2`).
- The bp jobs wait on Stage A's `summary.json` (poll the file; if Stage A fails, run bp with 1e-2 and
  record `mu_source="fallback: stage A failed"`). Both μ (default and 1e-2) are measured as now.
- **ImageNet-32 dropped**: removed from the notebook (runner code kept); gate.json records
  `"imagenet": "dropped (supervisor 2026-10-15)"`.
- MNIST via upstream `load_mnist.py` (internet on); fall back to an attached MNIST dataset if present.
- Last cell: Stage A summary, B5 baseline gate, gate G (B5 rows), per-lane timings; archive.
- Keep `kaggle_vna_stage_a.ipynb` as a standalone fallback.

## nb2 (`scripts/build_kaggle_vna_stage_b_nb2.py` / shared rest builder)
| lane | jobs |
|---|---|
| GPU0 | B3 (iCaRL) → B4 **measure** (waits until B4 training's final checkpoint exists) |
| GPU1 | B2 resnet32x4→resnet8x4 → B4 **train** |

## nb3 — unchanged lanes
GPU0: B1 resnet8x4 → B1 wrn_16_2. GPU1: B2 wrn_40_2→wrn_16_2.

## μ for nb2/nb3
Read `mu_rel_default` from an attached Stage A `summary.json` (search `/kaggle/input` for
`stageA/summary.json`, or any `summary.json` containing that key). If absent: fall back to 1e-2 with a
prominent printed warning and record `mu_source` in every run summary. Pass it to every runner.

## B3 train-accuracy condition
Not applicable (CIL residual persists across tasks): `diagnostics/vna_gate.py` B3 config sets
`train_acc_applicable=False`; gate records `"n/a"` plus the raw final train accuracy, like B5. Constants
unchanged; decided before any data.

Regenerate nb1/nb2/nb3 and the CIFAR pilot notebook if it shares setup cells. Short section in
STAGE_AB_IMPLEMENTATION.md; final message lists files touched.
