# Task-1 live GIR pilot — 2026-10-26

## Historical comparison

Reviewed `d88abc4` and `6be654310c2e4123391fd7d8d31716d25c89c6fb` directly
with `git show`, including the training hook, proposal search and apply wrapper.

* d88abc4: task 1 reaches the engine. GIR defines dead by exactly zero function
  contribution, then tries up to four live units with smallest H-norm contribution.
  Removing a live slot must not increase CE (and KD if present). It evaluates the
  two highest-ell proposals on the actual candidate model, accepting positive
  Gamma = -jump + q_current - q_candidate. A dead slot uses positive ell as the
  certified lower bound without measuring Gamma. Task 1 has no KD.
* VNA `cg_z` sets `split_reason` and suppresses GIR in d88abc4. A later VNA
  `untrusted` rejection does **not** suppress GIR when the split is trusted.
  Thus the supplied observation of swaps during a skipped application is
  consistent with the latter branch; it does not establish that cg_z permits GIR.
* 6be6543: `Intervention.after_step` immediately returns for `task < 2`.
  This is the direct reason task-1 interventions disappeared. Its normal GIR
  checks dead activation on frozen/current networks and uses the frozen-Jacobian
  Schur gain with a numerical error bound. `--gir-live` globally extends that
  newer rule and therefore is not the requested historical task-1 experiment.
  Several early VNA returns also occur before its normal GIR call.

## Implemented amendment

Base is exactly 6be6543 on branch `codex/gir-live-task1`, separate from later
paper-policy amendments. `--gir-live-task1` enables task 1 and ports the old
dead/live slot and CE/Gamma acceptance rule to actual-candidate Gram solves.
Latest amendment 2026-10-27: task 1 permits normally certified VNA, followed
by historical live/dead GIR. The native Gram and CE residual are retained for GIR;
current and candidate q use their actual native Jacobians. All candidate trials
restore parameters, buffers, optimizer, gradients, modes, RNG and install ages.
Task 2 onward takes the original 6be6543 path. Default off has the original
engine state shape and is checked against the historical source, bitwise.
Original s0 had no accepted VNA step in task 1; these observed rejections do
not disable VNA in the new run. A separate policy identity/output tag rejects
checkpoints from the superseded GIR-only experiment.

Fixed K0=512, power10, existing geometric cadence and compute-budget guard;
the first native SGD timing initializes that guard. Intervention logs contain
task-1 `swaps_live`, `swaps_dead`, individual slot kinds, Gamma or the explicitly
identified dead-slot lower bound, and candidate factorization failures.
The native runner already records cumulative GPU-hours including resume time.
N=8 is the planned seed set 0..7; the supplied notebook runs only s0 and s1.

Preregistration was written before the focused CPU tests. Baselines and
competitors are not executed or edited by the notebook. It uses no overlay,
no new preflight and no frozen-K0 file because this experiment fixes K0 directly.

## Requested diff: 84f833d → 6be6543

Command: `git diff 84f833d 6be6543 -- methods/ experiments/`.
There are **no differences in methods/**: native VNA/GIR mathematics and solvers
are identical between these two revisions. Nine experiment files differ:

* 84f833d adds `grad_clip.py`, `clip_study.py` and runner clipping/observation
  hooks. With clipping disabled, those hooks are measuring controls; enabled
  clipping changes optimization and is absent in 6be6543. Failure handling,
  divergence epoch reporting and persisted diagnostics differ too.
* Competitor configuration selection differs materially: 84f833d fixes original
  defaults for CBP, ReDo, S&P and UPGD; 6be6543 has sweep grids and upstream UPGD
  grid extraction. Their result-validation/provenance requirements also differ.
  Existing competitor results must retain their own configuration/provenance.
* Archive free-space and exclusion support, source registration for continuation,
  and result grouping/gradient-clip provenance also differ. These are pipeline
  differences, not a different native VNA/GIR update.
* The S&P device-placement correction is already present in both revisions;
  it is not a native full-arm difference in this comparison.

## Validation and limits

Only focused task-1 GIR tests and the existing dead-Schur/exact-solver test files
are run. They cover dense Gamma equality, dead shortcut, CE veto, candidate
rollback, read-only isolation, unchanged default/task-2 events/model/optimizer/RNG,
resume identity and the enabled hook's budget/cadence behavior.
See the adjacent pytest log for the final result. No GPU training or timing was
run; task-1 actual-candidate Gram recomputation can be expensive. Historical
29-swap counts are motivation supplied by the user, not a newly verified result.

Final focused run: **51 passed**, 36 upstream Torch/Python deprecation warnings,
247.32 seconds. Command (original workspace `.venv/bin/python`, with this
worktree on PYTHONPATH):

```
REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=. \
  .venv/bin/python -m pytest tests/test_gir_live_task1.py \
  tests/test_dead_schur_policy.py tests/test_vna_exact.py \
  -o addopts='' -q -p no:cacheprovider
```

The final comparison-signature assertion was separately rechecked with
`test_flag_resume_identity_and_default_task1_hook`: **1 passed**. Every code
cell in the generated notebook compiles; no notebook cell was executed.
The pre-amendment notebook `vna4_M1_gir_live_task1_s01.ipynb` pinned source
`c6a81f3eed937a1f88a1b391d940ae58bb3096c0` (a direct child of 6be6543).
GPU0 runs full s0; GPU1 runs full s1. Required data: CIFAR-100. The existing
GitHub Secret is still required to clone the private repository. No Gram input
or source overlay is needed for the registered fixed-K0 variant.

## 2026-10-27 GIR-only amendment

The notebook is regenerated with the new GIR-only source pin and output tag
`vna4_m1_gir_only_task1_s01`. Task 1 never calls the VNA/LM path; native
Gram and CE residual remain available to GIR. Damping is initialized for GIR
without a VNA trust-ratio search. Task 2 onward remains the original full path.
Old live-task1 checkpoints are rejected by the explicit task-1 policy identity.

Focused validation: 11 task-1 GIR tests pass, including both swap/no-swap cases,
read-only restoration, forbidden VNA/LM calls, damping initialization, legacy
checkpoint rejection and bitwise comparison of default/task-2 behavior with
6be6543. Actual output is in `GIR_TASK1_ONLY_PYTEST.log`. The 51-test result
above is historical validation of the preceding variant. No GPU run was made.

### Native float32 proposal repair — 2026-10-27

Real s0/s1 task-1 runs stopped at the first GIR power10 proposal with a
float64-Hessian/float32-JEv contraction error. This was a code exception,
not nonfinite-loss divergence. The local tests had previously mocked the
proposal search or used float64 native models and missed this path.

An unmocked proposal regression reproduces both live/dead float32 failures
before the fix; the corresponding float64 cases pass. Converting the metric
input to the Hessian blocks' dtype/device fixes the shared task-1 metric
callback without changing its definition. Latest focused rerun: **15 passed**,
36 upstream deprecation warnings, 6.99 seconds. The refreshed actual log is
`GIR_TASK1_ONLY_PYTEST.log`. Both s0/s1 and s2/s3 notebooks are repinned to
the repaired source. No GPU training was performed locally.

## Latest: normally certified task-1 VNA — 2026-10-27

The GIR-only amendment is superseded. The task-1 VNA attempt uses the unchanged
normal VNA certificate; historical live/dead GIR runs afterward, including when
VNA was skipped/rejected. Its swaps do not overwrite the separate VNA decision.
Each task-1 event logs vna_status=ACCEPTED/REJECTED/SKIPPED, vna_applied,
vna_skip_reason, vna_rho and s. The console prints the same decision and GIR
counts; the final notebook report lists each task-1 VNA outcome. Failed LM
attempts retain the last rho even when there is no accepted rho_tr.

The user reports original s0 it1 skipped for cg_z and it1025 rejected as
untrusted at rho=-23.9. These are historical observations, not forced outcomes
for the new Gram run. Task 2 onward and default-off behavior remain unchanged.
New identity: d88abc4_T28_vna_certified_gram; output tags:
vna4_m1_vna_certified_task1_s01/s23. Do not resume GIR-only checkpoints.

Focused validation: **40 passed**, 36 upstream deprecation warnings, 11.50s,
in test_gir_live_task1.py and test_dead_schur_policy.py. Covers accepted VNA,
cg_z skip, untrusted rejection at rho=-23.9, exact-loss rejection and failed-LM
rho logging, with/without GIR swaps, read-only isolation, native float32
proposal generation, and bitwise equality with original 6be6543 for task 2 and
flag-off. Actual output: GIR_TASK1_CERTIFIED_PYTEST.log. No GPU run performed.
