# Notebook plan (supervisor, 2026-10-15)

No commit. Do NOT run pytest. Allowed: pyflakes, building the notebooks, `python -m json.tool` /
compiling every generated code cell.

Run order: nb1 → nb2 → nb3.

## nb1 = Stage A (GPU0) ∥ B5 light (GPU1)
- `scripts/build_kaggle_vna_stage_b_nb1.py`: GPU0 runs the full Stage A sequence (a5 → null →
  selection → joint → mu → solver → summary, same CLI and resume/identity rules as the Stage A
  notebook; needs the Phase 0 checkpoint dataset). GPU1 runs B5 regression (bp, then the five cbp rates)
  → Permuted MNIST bp → cbp. **ImageNet-32 is dropped from this round**: remove it from the notebook
  (keep the runner code); gate.json records `"imagenet": "dropped (supervisor 2026-10-15)"`.
- MNIST: upstream `load_mnist.py` downloads it (Kaggle internet on); fall back to an attached MNIST
  dataset if present.
- B5 uses μ = 1e-2 plus `mu_rel_default` once Stage A's summary exists in the same session: start B5
  after Stage A's `mu` subcommand finishes if that is < 1 h into the session; otherwise run B5 with
  1e-2 and record that the Stage A value was unavailable (both μ are measured anyway).
- Budget/archives/smoke as now. Last cell: Stage A summary + B5 baseline gate + gate G (B5 rows).
- `kaggle_vna_stage_a.ipynb` stays as a standalone fallback.

## nb2 / nb3: μ and the B3 train-accuracy condition
- Both read `mu_rel_default` from an attached Stage A `summary.json` (search `/kaggle/input` for
  `stageA/summary.json` or `summary.json` with that key); if absent, **stop** with a clear message unless
  `ALLOW_DEFAULT_MU=True` (then 1e-2, recorded). Pass it to every runner; record its source.
- B3 gate: train-accuracy condition **not applicable** (CIL residual persists across tasks); gate records
  `"n/a"` plus the raw final train accuracy, exactly like B5. Change `diagnostics/vna_gate.py`'s B3
  config accordingly (constants unchanged; this is a window/applicability flag set before any data).
- Regenerate nb1/nb2/nb3 (and the pilot notebook if it shares the setup cells).

Report: short section in STAGE_AB_IMPLEMENTATION.md; final message lists files touched.
