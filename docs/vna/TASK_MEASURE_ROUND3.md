# Stage B measurement round 3 (supervisor review of 5152d1c)

Branch `vna-measure-round2`. `.git` read-only: do NOT commit. **Do not run pytest or any smoke.**
Allowed checks only: `pyflakes` on touched files, `python -m py_compile`, rebuilding nb1/nb2 and compiling
their code cells. Keep the algorithm and every number unchanged (identities only). Update the existing
tests that obviously reference changed APIs, but do not run them.

## 1. B3: exact solve for large n_out (Cholesky per μ, M on CPU)
B3 has n_out = 256 × seen classes (2,560 → 25,600). With OUTPUT_EXACT_MAX = 4096, tasks 2–10 fall back to
CG (130–190 s per solve) and hit the 2 h point timeout. Change `methods/vna_output.py`:
- `auto` → output_exact always for B3/B5 (keep `cg` selectable; drop the 4096 cut for the exact path).
  Factorization mode: `eigh` when n_out ≤ 4096 (keeps spectrum diagnostics), else **Cholesky per μ**.
- Large-n_out path: build M directly into a preallocated **CPU** float64 tensor (≈4.9 GB at 25,600),
  columns via chunked vmap on GPU then copied to CPU; **symmetrize in place** (e.g. upper←lower via
  `matrix.triu_(1)`-style in-place ops or a blocked loop), never `(M+M.T)/2`. For each μ: add μ to the
  diagonal in place, `torch.linalg.cholesky_ex` (check `info`; on failure mark the point's
  `cg_converged_real=False` and record `factor_error`), subtract μ back, keep only the factor needed, free
  it before the next μ. Avoid more than two n_out² buffers alive at once; print peak RSS.
- Solves with `torch.cholesky_solve` on CPU (move small vectors to CPU and back):
  z = R (s − μ (M+μ)⁻¹ s), s = R⁺e; d = Jᵀ R (M+μ)⁻¹ (R J_E w). Reuse the factor for real, eval, all 32
  nulls, groups and d-verification of that μ.
- B3 exemplar/new groups: Cholesky of the principal submatrix of M_eval for the group indices (no rebuild).
- Skip the spectrum diagnostics (K top eigenvalues, residual energy below μ) when n_out > 4096; record
  `spectrum: 'skipped (n_out > 4096)'`.
- Record per point: `solver`, `factor_mode` (eigh/cholesky), `n_out`, `M_build_seconds`,
  `factor_seconds` (per μ), `M_device`, peak RSS / peak CUDA memory.
- nb2 passes the solver explicitly (`--solver auto`) and its smoke runs task 1 AND task 2 measurement points
  and prints n_out, build/factor seconds, so the supervisor can extrapolate to task 10.

## 2. GIR swap selection on CBP arms (B5)
On CBP arms the shadow GnT does not see the learner's replacements (stale utility/age), and argsort ignores
`maturity_threshold` (upstream `lop/algos/gnt.py` ~l.142).
- CBP arm: use the learner's own GnT `bias_corrected_util` and `ages`; only units with
  `ages > maturity_threshold` are eligible; pick the m lowest-utility eligible units (m as now:
  max(1, ReDo-dormant count of the layer)); if fewer eligible than m, use those and record the count.
- BP arm: keep the shadow GnT as now. Record `cbp_utility_source: 'learner'|'shadow'` and `eligible_units`.

## 3. Independent exact-vs-CG check
At the first measurement point of every run, additionally solve the real z by CG (tol 1e-6, max 1000) and
record `exact_vs_cg_relative = ‖z_exact − z_cg‖/‖z_exact‖` (and CG iterations/converged) in that point's
rows and in summary.json. Do not exclude on it; just report. (V_consistency stays as is.)

## 4. Measurement cost on MNIST CBP
Print a separate projected measurement cost for MNIST CBP in the smoke. Default: MNIST **CBP** measures only
points inside the gate window (task > tasks_run/2); MNIST BP and all regression arms measure the full
schedule. Flag `--measure-window-only {auto,yes,no}` (auto = the rule above); record it in summary.json.

## 5. Peak memory
Smoke (Kaggle) must print peak CUDA memory of `tiny_V` per setting (`torch.cuda.max_memory_allocated`,
reset before each tiny_V call) and peak host RSS of the M build/factor.

## Report
Append "Round 3" to `docs/vna/MEASURE_ROUND2_REPORT.md`: changed files and what each change does. Rebuild
nb1/nb2 with `--commit` of current HEAD (supervisor repins after committing).
