**Diff theo file**

- `diagnostics/measure_efficiency.py`: shared solve cache, whole-network `solve_z`, VJP-only virtual energies, shared explicit null draws and consistency check.
- `diagnostics/icarl_measure.py`: stat/eval solves once per μ/residual; per-row d verification; joint/group energies from sums; both μ, trace, true buffer sizes, null IDs and separate convergence flags.
- `diagnostics/lop_measure.py`: disjoint stat/eval selection/scoring, shared nulls, two-solve d verification, GIR swaps cached per site/μ/criterion and one solve per clone.
- `methods/vna_joint.py`: optional supplied trace; per-solve instrumentation without changing CG or correction equations.
- `projection/cg.py`: optional iteration callback, including normalized RHS recursion.
- `experiments/vna_resume.py`: extend existing Status/Progress machinery with measurement heartbeat, soft deadline, faulthandler and solves.csv; CLI configuration.
- `diagnostics/vna_align.py`: run measurements inside that monitor; restore model/RNG after timeout and emit excluded rows.
- `diagnostics/vna_gate.py`: rho-based N1; point exclusion counts/fraction; INVALID above 10%.
- `adapters/icarl_hooks.py`: 512 buffer; timeout checkpoint; one smoke point.
- `adapters/repdistiller_hooks.py`: 512 buffer; one smoke point.
- `adapters/lop_hooks.py`: 256 buffer for production measurements; timeout checkpoint; one smoke point; require full buffer.
- `experiments/stage_b12.py`: runtime CLI/wiring, both-μ identity, fixed official μ=1e-2.
- `experiments/stage_b3.py`: runtime CLI/wiring, both-μ identity, fixed official μ=1e-2.
- `experiments/stage_b4.py`: runtime CLI/wiring, 512 buffer, both-μ identity, fixed official μ=1e-2.
- `experiments/stage_b5.py`: runtime CLI/wiring and gate at 1e-2.
- `scripts/build_kaggle_vna_stage_b_nb1.py`: print solve totals and mean solve time; preserve lane plan.
- `scripts/build_kaggle_vna_stage_b_rest.py`: rho gate at 1e-2, both-μ identity, solve totals and mean solve time; preserve NB2/NB3 lanes.
- `notebooks/kaggle_vna_stage_b_nb1.ipynb`: regenerated.
- `notebooks/kaggle_vna_stage_b_nb2.ipynb`: regenerated.
- `notebooks/kaggle_vna_stage_b_nb3.ipynb`: regenerated.
- `scripts/smoke_measure_efficiency.py`: fake-data CPU point for every setting, solve-count assertions and cost JSON.
- `tests/test_measure_efficiency.py`: tests 5.1–5.4, legacy modules loaded from git show into temporary modules, explicit null hooks, runtime/timeout test.
- `tests/test_vna_stage_a.py`: rho significance fixture and explicit INVALID expectation for excluded points.
- `tests/test_vna_stage_b3.py`: updated buffer bound.
- `tests/test_vna_stage_b5.py`: both-μ row expectations; retain small buffer only in replay fixtures.

No commit made. Stage B notebook pins remain at existing HEAD `566eafc3c1b6f81968f6ea6c277a4787c3f0ff66`; this commit does not contain the working-tree implementation. The supervisor must publish a commit and regenerate all three notebooks with that SHA before Kaggle. A new implementation commit cannot be pinned under the no-commit/read-only-.git constraint.

**Kết quả test 5.1–5.5**

5.1–5.4: numerical old/new comparison, joint identity, B5 isolation and gate checks pass. Additional LoP arm equivalence and heartbeat/timeout restoration tests pass. Legacy code is read using `git show 688b61d:<path>`, never checked out. Both μ, N1 and N2 are covered; relative tolerance is 1e-6.

Full suite: **245 passed, 0 failed, 1 skipped** (246 collected; four added seed cases), exit code 0. Run once after the affected-file iterations with `REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python .venv/bin/python -m pytest -q`. [Final full log](measure_efficiency_logs/full_suite_fixed.log). The earlier 232 passed / 9 failed / 1 skipped run is retained as [historical evidence](measure_efficiency_logs/full_suite.log); its eight reproducible failures are fixed below. The ninth was the temporary-disk reserve failure and now passes. Completed test directories were cleaned before and during verification, and remaining test artifacts were removed afterward.

| Failure | Cause | Fix |
|---|---|---|
| Stage A smoke marker count | Stale: resume now persists each site/rank/kind/μ unit, giving 14 markers. | Assert counts per measurement using the registered grids; retain resume/hash/schema checks. |
| Stage A notebook pin/order | Real preflight bug: builder stripped pytest; stale manual Phase-0 path assertion. | Restore the full suite with `check=True`; check automatic discovery and retain lane/pin/compile assertions. |
| Stage A notebook cost/A5 | Stale: session orchestration uses `stage_launch`, with cost before A5. | Exercise the current runner; verify budget rejection archives/stops before A5 and override permits A5. |
| WRN finite difference | Numerical: fixed float64 step crosses downstream ReLU kinks. | Shrink step until base/± ReLU masks agree; retain original FD and adjoint tolerances; test seeds 0 and 1. |
| B4 finite difference | Numerical: same ReLU boundary crossing. | Use the shared branch-aware FD check and explicit seeds 0 and 1; no tolerance relaxation. |
| BP MNIST resume buffer | Stale: next-task pre-training rank checkpoint retains the previous task's 64 samples until `learn` clears them. | Assert retained size; keep bitwise learner/RNG/log/measurement/result checks and task-purity checks. |
| CBP MNIST resume buffer | Stale: same rank-checkpoint timing. | Assert retained size; keep bitwise resume checks. |
| B5 notebook sources | Stale: NB1 now includes Stage A/CIFAR, drops ImageNet, waits for the new summary and archives every 15 minutes; real stripped-pytest preflight bug too. | Update source assertions with a reason; restore/check full-suite preflight in NB1–NB3 and compile all code cells. |

FD evidence: [reproducible audit script](measure_efficiency_logs/fd_numerical_audit.py), [step sweep](measure_efficiency_logs/fd_numerical_audit.log), [final FD regression run](measure_efficiency_logs/eight_fixes_fd.log). Both original tests already used float64. WRN seed 0/site 0 at `h=1e-5` crosses two ReLU masks and has relative L2 error `1.62e-3`; at `h=1e-6`, masks agree and error is `3.38e-10`. B4 seed 0/site 2 crosses one mask at `1e-5` (error `1.16e-2`), versus no crossings and `3.13e-10` at `1e-6`. WRN seed 1 needs `1e-7` for its first site. The audit's adjoint discrepancies are at machine precision (maximum about `1.6e-15`). Code review confirms HookSite captures the first-layer input, applies the declared new-coordinate activation, injects B at the second-layer output, and replays the native downstream graph; JEv/JET differentiate the same functional call. There is no observed site/adjoint error. The final test captures both module and functional ReLUs, chooses the step solely from activation masks (never derivative agreement), fails if it cannot find a smooth neighborhood, and retains `rtol=1e-5, atol=1e-7` for FD and `rtol=1e-8, atol=1e-10` for the adjoint.

The final corrected smoke/notebook cases pass in [focused verification](measure_efficiency_logs/eight_fixes_remaining.log); the other affected-file checks passed apart from intermediate assertions corrected by that rerun and the final FD rerun. Standalone Stage A and NB1–NB3 were regenerated with their existing pins and all code cells compile. No commit made.

The B5 resume expectation now checks task IDs 1 and 2 explicitly: the production schedule is `{1, 2, 5, 10, ...}`, so task 3 is intentionally unmeasured. ImageNet's replay callback uses a 100-example buffer, filled on step 1; its two-step fixture has coincident first/mid points. The full-buffer requirement does not skip an eligible ImageNet point. ImageNet and regression BP/CBP resume cases passed in the full run and targeted rerun; MNIST's stale empty-buffer assertion is now corrected as described above. [Targeted rerun](measure_efficiency_logs/final_targeted.log).

Historical baseline checks (before these fixes): [Stage A/notebooks](measure_efficiency_logs/baseline_stage_a_notebooks.log), [MNIST](measure_efficiency_logs/baseline_mnist.log), [WRN/B4, seed 0](measure_efficiency_logs/baseline_fd_seed0.log), [branch WRN/B4, seed 0](measure_efficiency_logs/branch_fd_seed0.log), [B4, seed 1](measure_efficiency_logs/baseline_b4_seed1.log). Seed 0 gives the same WRN failure and B4 pass on baseline and branch; seed 1 reproduces the B4 failure on baseline. MNIST baseline checks preload PyTorch's JVP decompositions to avoid compiling them while the existing fixture temporarily patches `torch.zeros`. Temporary test/checkpoint directories were removed after saving logs. No commit made.

All notebook code cells compile; `git diff --check` passes.

**Bảng chi phí smoke**

One CPU fake-data point per setting; eight nulls of each kind and both μ, CG tolerance 1e-6 / maximum 1000. The one-site fixture supports rank 4 for B1–B4 and ranks 1/2/4 for B5; larger configured ranks are skipped only because the fake layer is narrower. Production ranks are unchanged. Old solve counts are analytical counts of the baseline loops normalized to the same two μ, not timed baseline runs. New counts are observed in solves.csv. Times are CPU measurements, with the full suite running concurrently.

| Setting | CG before | CG after | Seconds/point (after) | Mean seconds/solve (after) |
|---|---:|---:|---:|---:|
| B1 | 114 | 72 | 8.434 | 0.0709 |
| B2 | 114 | 72 | 8.036 | 0.0840 |
| B3 | 122 | 76 | 6.201 | 0.0553 |
| B4 | 114 | 72 | 6.054 | 0.0575 |
| B5 | 378 | 84 | 6.923 | 0.0503 |

Counts per μ: B1/B2/B4 `55×site_rank_pairs + 2×ranks` → `34 + 2×site_rank_pairs`; B3 adds old `2×ranks×nonempty_groups` versus new `nonempty_groups`; B5 adds old `6×site_rank_pairs×nonempty_swap_criteria` versus new `2×sites×nonempty_swap_criteria`. Shared real stat/eval solves and the 32 null solves account for 34.

Logs: [smoke](measure_efficiency_logs/smoke.log), [cost JSON](measure_efficiency_logs/costs.json), [5.1–5.4](measure_efficiency_logs/tests_51_54.log), [safety and LoP oracle](measure_efficiency_logs/tests_safety_and_lop.log).

**Ước tính GPU-giờ cho từng notebook**

| Notebook | GPU-hours estimate |
|---|---|
| nb1 | Pending supervisor's T4 smoke; Stage A + measured MNIST BP on GPU0 and MNIST CBP on GPU1; regression on CPU. |
| nb2 | Pending supervisor's T4 smoke; B3/B4 measure on GPU0, B2 resnet8x4/B4 train on GPU1. |
| nb3 | Pending supervisor's T4 smoke; two B1 jobs on GPU0 and B2 WRN on GPU1. |

CPU fake-data timings cannot determine credible GPU-hours for full native models. The generated notebooks calculate the estimates from GPU smoke training time, measured point time and registered point/epoch counts; report total GPU-hours as the sum of the lane hours. No Kaggle smoke or full run was launched locally.
