# Stage B measurement round 2

Implemented on `vna-measure-round2`, HEAD `20af3dd`. No commit, no full training.
The three notebooks were rebuilt with the full current HEAD hash, as requested;
the supervisor must repin after committing the implementation.

## Numerical and protocol changes

- A1: JVP decompositions load at module import, before MNIST patches `torch.zeros`.
  The fresh-process test executes the native MNIST loop with an actual measurement
  callback and verifies both the live patch and its restoration.
- A2: TINY streams sample chunks (8) and position chunks (1024), accumulating
  S, N, and optional YᵀY/YᵀR/XᵀY. Sign scoring sums chunk gradients before
  squaring, preserving the original ReLU score. Dense-oracle comparisons cover
  linear and convolutional sites, with and without projection, at 1e-10 relative.
- A3: shared Stage-B JSON writes and atomic JSON writes use `default=str`.
- B: `--solver {auto,output_exact,cg}` defaults to auto, exact through 4096 outputs;
  `--output-exact-max` (alias `--OUTPUT_EXACT_MAX`) changes the threshold. A chunked
  vmap builds RJJᵀR. One float64 eigendecomposition per batch serves both μ values,
  real/null residuals and d verification. B3 subsets use principal submatrices,
  scaled for the subset's mean-loss normalization; each B5 swap clone is separate.
  Large forced matrices move to CPU when the estimated GPU workspace is too large.
  Range(H) violations set a flag and invalidate convergence without an assertion.
  CG does not build a dense spectrum; its spectrum diagnostic is explicitly marked
  unavailable. Solver choice/reason and factor timings appear in logs and rows.
- C: real/null V, e_norm2 and nu, N1/N2 nu p-values, and the nu gate are recorded.
  See [PREREGISTRATION_CHANGES.md](PREREGISTRATION_CHANGES.md). B5 requires ten
  distinct points in its gate window. The regression schedule adds task 51 to
  the listed tasks to reconcile the simultaneous requirement for 22 window points;
  the listed 55–100 grid alone provides 20. MNIST has 20 window points.
  B3 has ten midpoints plus reference endpoints at tasks 2, 5 and 10.
- D: each CBP criterion selects the lowest bias-corrected shadow utility, with
  m=max(1, ReDo dormant count), and records `cbp_selection=lowest_utility_m`.
- Logging: per-task B5, per-epoch B1–B4, no 10k-step messages; runner subprocesses
  receive TQDM_DISABLE=1. The ten-minute notebook heartbeat remains.
- Notebooks: nb1 reruns six measured regression arms and measured MNIST BP/CBP,
  using a round2 output namespace and resumable archives; it prints per-session
  estimates. nb2 only runs B3 (13 points). nb3 is deferred and launches nothing.

## Bounded efficiency smoke

Command: `.venv/bin/python -m scripts.smoke_measure_efficiency --solver output_exact --profile-real-tiny`.
Fake batch 32, one small site, three classes; no datasets, training, or downloads.
All displayed timings are CPU timings from this shared host. They cannot estimate
T4 GPU-hours or full architecture training costs. Build/factor sums include all
batches and swap clones; matrix memory is the largest individual M. RSS is the
cumulative process peak. The script also prints a separately calibrated cubic
forced-exact estimate and its assumed output dimension.

| Setting | n_out | Solver | M build s | Factor s | Point s | CG s/solve (control) | Largest M MiB | Projected measurement CPU h |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| B1 | 48 | output_exact | 0.1900 | 0.0025 | 10.721 | 0.2212 | 0.0176 | 0.0447 |
| B2 | 48 | output_exact | 0.1772 | 0.0014 | 6.469 | 0.1634 | 0.0176 | 0.0270 |
| B3 | 48 | output_exact | 0.1976 | 0.0023 | 7.223 | 0.3464 | 0.0176 | 0.0261 |
| B4 | 48 | output_exact | 0.2452 | 0.0016 | 6.637 | 0.2366 | 0.0176 | 0.0221 |
| B5 | 48 | output_exact | 0.2827 | 0.0049 | 12.226 | 0.2358 | 0.0176 | 0.1290 |

Peak CPU process RSS: 218.6 MiB.

CUDA is unavailable on this machine. Actual-batch tiny_V GPU peak memory for
resnet8x4, wrn_16_2 and PyCIL ResNet32, and GPU-hours for real runs, remain unmeasured.
The `--profile-real-tiny` option runs every site of these actual models with batch
256 when CUDA is available; this host records explicit unavailable results in
[tiny_memory.json](measure_round2_logs/smoke/tiny_memory.json).
For CIFAR n_out=25,600, M alone is 5,242,880,000 bytes (4.88 GiB); factorization
workspace and eigenvectors are additional. Auto selects CG at this size.

Raw [exact costs](measure_round2_logs/smoke/costs.json),
[CG control costs](measure_round2_logs/cg_smoke/costs.json),
[exact log](measure_round2_logs/smoke.log), [CG log](measure_round2_logs/cg_smoke.log).

## Damping history

The requested `git diff 90a18f7..60b7b5b` is included verbatim in
[damping_history.patch](measure_round2_logs/damping_history.patch).
Output normalization uses trace(F)/n_out so widening the parameter space does not
silently shrink damping; the common 1e-12 floor keeps saturated/zero-trace cases
strictly positive. This implementation preserves that choice and both μ values.

## Validation

Combined verification: **262 passed, 1 skipped**, no remaining failures or errors.

Full-suite command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TQDM_DISABLE=1 REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python .venv/bin/python -m pytest -ra --junitxml=docs/vna/measure_round2_logs/full_suite.xml`.

An early full-suite attempt was stopped after 20 passing checks when review found that the existing scalar-regression resume test compared new observational timings. Its filter was corrected and its targeted test passed. The subsequent complete full-suite run was not repeated.

That complete run reported 259 passed, 3 failed, 1 skipped. The failures were old fixture expectations that task 3 was unmeasured. Under the new dense second-half schedule, all tasks of the three-task fixture are measured. Training parameters, RNG, logs and resumed numerical rows already matched before this outdated schedule assertion. After updating that expectation, all six BP/CBP resume cases passed in a focused rerun. The combined result above replaces only these six cases; both original logs are retained.

New tests cover CE/KD/MSE z and d at CG tolerance 1e-12, V/rho/rho_joint and all eight N1/N2 null statistics at 1e-8 relative; dense-vs-streamed TINY at 1e-10; factor reuse and subset normalization; range(H) invalidation; native MNIST first-JVP under the live zeros patch; JSON devices; nu gates; B3/B5 schedules; and nonempty CBP swaps.

Separately, 5 production/streaming checks, 7 affected notebook/gate/logging/timeout checks, 1 CBP/schedule check and 1 scalar-regression resume check passed. Notebook cells compile and all three pins equal current HEAD. `git diff --check` is clean.

Skipped: `tests.test_vna_split3.test_render_comparison_curves` — optional diagnostics dependency unavailable.

[Full log](measure_round2_logs/full_suite.log), [resume rerun](measure_round2_logs/resume_schedule.log), [original JUnit](measure_round2_logs/full_suite.xml), [rerun JUnit](measure_round2_logs/resume_schedule.xml), [combined results with provenance](measure_round2_logs/verified_results.xml), [complete implementation diff](measure_round2_logs/implementation.patch).

## Changed files

```text
7	4	adapters/icarl_hooks.py
20	8	adapters/lop_hooks.py
9	3	adapters/repdistiller_hooks.py
27	15	diagnostics/icarl_measure.py
24	41	diagnostics/lop_measure.py
61	5	diagnostics/measure_efficiency.py
4	4	diagnostics/vna_gate.py
1	1	experiments/shared_protocol.py
1	1	experiments/stage_b12.py
2	2	experiments/stage_b3.py
1	3	experiments/stage_b4.py
5	10	experiments/stage_b5.py
15	2	experiments/vna_resume.py
12	4	methods/vna_joint.py
36	10	methods/vna_sites.py
14	13	notebooks/kaggle_vna_stage_b_nb1.ipynb
7	8	notebooks/kaggle_vna_stage_b_nb2.ipynb
4	393	notebooks/kaggle_vna_stage_b_nb3.ipynb
13	11	scripts/build_kaggle_vna_stage_b_nb1.py
11	21	scripts/build_kaggle_vna_stage_b_rest.py
68	7	scripts/smoke_measure_efficiency.py
5	5	tests/test_measure_efficiency.py
1	1	tests/test_vna_stage_a.py
15	7	tests/test_vna_stage_b5.py
```

New files: methods/vna_output.py, tests/test_measure_round2.py, PREREGISTRATION_CHANGES.md, this report, and measurement logs/diffs.
