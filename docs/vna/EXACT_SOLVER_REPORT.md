# T29–T31 implementation report

Status: local implementation and section-8 numerical checks complete. T4 measurements and A–D outcomes are pending; this report does not authorize M1.

## Source and checkpoint

- Branch `vna-v4-exact`, base `d88abc496eb1fe21d1b6df66d90b47cf93d7ded1`; publication by commit/push authorized by the user on 2026-10-20.
- Read the supplied exact-solver prompt, reference Python, PDF and LaTeX source. Updated the existing theory files, rather than keeping competing source trees. Rendered and inspected the new PDF section.
- Read `Downloads/vna4_main.tar.gz` (86,904,798 bytes). Full-arm `latest.pt` is `completed=1, mid=False, epoch=100`, full/noBN/superclass/seed0, lr scale .5, source d88abc4. It is the requested task-1-end checkpoint; no task-1 retraining is needed for these diagnostics.
- `latest.previous.pt` is an epoch-100 mid-task checkpoint before task-end processing. The diagnostic runner deliberately selects `latest.pt`.

## Changes by file

| Files | Change |
|---|---|
| `methods/vna_exact.py` | Native-dtype batched VJP/JVP output Gram, float64 block-metric eigendecomposition/Cholesky, signed Woodbury replacement and sequential commit. No jitter, PSD projection or fallback. |
| `methods/vna_v3.py` | All engine/diagnose CG replaced; inner LM up to 40 trials; finite-positive rho gate; one mu_star; frozen model for GIR proposals, slot columns and contributions; real sequential loss/KD checks. Application snapshots include gradients, optimizer, buffers, modes and all RNG streams. Roll back any nonfinite parameter/buffer/optimizer outcome. Log norms, beta leaks, Gram/LM/Gamma costs and swap diagnostics. |
| `methods/vna_ggn.py` | Make native gradient pullback lazy. Exact output computations avoid an unnecessary parameter gradient; legacy callers still obtain the same gradient on access and the same dedicated trace probes. |
| `adapters/icarl_hooks.py`, `experiments/main_cil.py` | Guard loss before SGD; record FAILED_DIVERGED with task/epoch/step; retain safe initial/epoch checkpoint; pass current-task mu_star to diagnosis. Print explicit failure and checkpoint path. Training recipe is unchanged. |
| `adapters/vna_v3_hooks.py`, `experiments/pilot_v3.py` | Pass mu_diag; remove obsolete CG/warm configuration from pilot protocol/schedule output. |
| `diagnostics/exact_solver.py` | One-T4 old/new profiling and five-epoch task-2 A–D replay. Select/validate the archived checkpoint; keep B/C at d88abc4, carry checkpoint mu into D, suppress optimizer resets only for C. Per-step SGD JSONL, GGN power-10 every 10 steps and after intervention, first nonfinite tensor, explicit diagnostic identity, safe epoch resume and exit-75 propagation. |
| `scripts/build_kaggle_vna_parallel.py`, `notebooks/vna4_*.ipynb` | New diagnostic notebook; shared immutable implementation commit pin (source ZIP only for uncommitted overlays); one GPU used. M1/M1s0 stop before preflight/training pending review. Other existing presets and archive/runtime wiring remain. |
| `scripts/collect_parallel.py` | Validate current B/mu_diag/norm_w2 instead of reach/unreach. |
| `tests/test_vna_exact.py` | Section-8 float64 checks plus focused state/rollback/divergence/resume checks. Updated affected older test callers to the output-space representation; did not run historical smoke tests. |
| `docs/vna/` | Supplied prompt, PDF, LaTeX/reference script, preregistration amendment and this report. |

Deleted from the engine: parameter CG import/solve, warm cache and warm resets, tol/iteration tuning, cg_z rejection, per-candidate relinearization on a modified network, between-application-only LM, reach/unreach output fields. `projection/cg.py` remains because other production methods still import it.

## Local checks actually executed

Command environment: `.venv/bin/python`, `REQUIRE_UPSTREAM=1`, `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

- `pytest -q tests/test_vna_exact.py`: **13 passed**.
- T29: MLP/conv, each ±BN, H_old and beta 0/.5/5; x, z, Delta, xFx, B, d_N and protected residual match dense solves within 1e-8 relative tolerance.
- T29(e): dead installation with zero outgoing weights preserves logits and reproduces the Gram addition within 1e-10.
- T31: signed Woodbury Gamma and sequential updated solves match dense references; two accepted dead swaps preserve frozen theta and update the residual. Adjacent MLP sites remove the current shared columns rather than subtracting an original column twice.
- T30: mu0=1e-10 obtains finite positive rho; forced first trial NaN is rejected, with bitwise trial-state restoration.
- NaN gate: all-NaN trial losses exhaust LM and never run GIR. Engine has no CG import.
- Read-only application, nested nonfinite optimizer-state rollback, divergence location and one-native-step resume checks pass. Resume comparison excludes wall-clock observations; algorithm tensors, RNG and state match bitwise.
- Supplied `check_t29_31.py`: 200 cases, **0** violations for T29a/b/c/d, T31 and R1. All 50 LM cases terminate, median/max **7/7 retries**.
- Python/notebook syntax and source-pin/digest checks are recorded during packaging. No full pytest or notebook smoke run.

## GPU measurements: not run locally

Local host is an Intel Mac without a T4. No `profile_d88abc4.json` or `profile_exact.json` is fabricated.

Run `notebooks/vna4_exact_diagnostics.ipynb` with:

1. The existing `vna4_main.tar.gz` containing the task-1 checkpoint.
2. Official CIFAR-100 input and the existing GitHub secret used by notebook bootstrap.

Published notebooks load the implementation from their immutable Git commit and set `SOURCE_DIGEST=None`; no source ZIP is required. A ZIP is only needed for notebooks explicitly built with an uncommitted source overlay.

Use `MODE='profile'` first. It runs B at d88abc4 and D at the pinned implementation commit, on GPU0, sequentially. Reports are under `outputs/exact_diagnostics/profile/{B,D}/`; they include CG iteration/residual/cost, native parameter and vector dtypes, float32/64 JVP/VJP timings, Gram metadata, LM trajectories, per-candidate Gamma costs and CUDA peak allocation. Then use `MODE='nan'` for A/B/C/D, each for five epochs of task 2 from the same checkpoint. Exit 75 and epoch checkpoint resume remain available; no result is a main comparison score.

| Branch | Task-2 experiment | First nonfinite tensor/step | GGN >40 before failure | Outcome |
|---|---|---|---|---|
| A | Baseline continuation | pending T4 | pending | not run |
| B | Old single application at first task-2 step | pending T4 | pending | not run |
| C | B with optimizer state retained across swaps | pending T4 | pending | not run |
| D | T29–T31 single application | pending T4 | pending | not run |

The archive itself confirms the previously observed B-like event at step 2001: rho=NaN, 26 swaps, 2878.92 seconds; its summary says DIVERGED after one completed task. This is historical evidence, not the requested new per-step A–D experiment.

## Budget, capacity and discrepancies

- New K0, application count per task and 20-task 1.0x budget **cannot be validated before actual T4 costs**. Do not reuse the old CG schedule or infer authoritative K0 from the local tiny-model tests. The existing geometric schedule/calibration function is retained for subsequent authorized calibration.
- Keep top-2 proposals and four live slots. Per-candidate `gir_trials` records allow a later cost comparison; no limits were lifted.
- A Gram at m>=12800 exceeding 600 seconds raises an explicit resource stop. CUDA OOM is reported by the diagnostic runner; no fallback is selected.
- Existing 256-sample diagnosis means m=25600 at task 20, with about 5.24 GB for G alone. The engine window has 128 samples and m=12800 (about 1.31 GB). The prompt's capacity estimate covers the latter. Keep probe size/validation unchanged pending actual T4 measurements; factors and temporaries require additional memory.
- The supplied source `s9c_exact.tex` is printed as PDF section 11, although the prompt calls it section 9c. T29–T31/R1 formulas agree.
- LM real trial loss needs a pullback J^T y at every trial; the final VNA step also needs one. All solves remain in output space. This reconciles the prompt's “only final VJP” wording with its trial evaluation requirement.
- Archived mu is 6.9074e-8 at task 1 and **1.38148e-7** at task 2. D carries the actual checkpoint mu; the prompt's approximate 6.9e-8 describes task 1.
- No old baseline has been reused. Epoch-loss identity through two tasks has **not** been checked on T4. Additional failure/checkpoint observations do not change SGD math, but this is not evidence of the required old/new epoch-loss identity.
- No remote Kaggle session was controlled from this local workspace. The new M1 notebooks are held; this does not establish that an already-running historical session has stopped.

Skipped: full suite, notebook smoke runs, T4 profiling/A–D, full 20-task training and baseline reuse verification. Publication does not change these pending checks.


## 2026-10-21: profile-B bootstrap failure repaired

The supplied Kaggle log fails before any profile application: historical checkout reports missing SHA1 blobs, followed by an import of current `methods.vna_v3` without `conjugate_gradient`. Detached HEAD is expected for immutable pins; the missing objects and deleted tracked files are the failure. No usable new T4 profile or A–D outcome was produced by that attempt.

- Replace the local shared clone with a direct authenticated checkout of the full historical SHA. The common clone helper now fetches the requested SHA (including a SHA different from REPO_COMMIT) and keeps authentication available during partial-clone checkout.
- Automatically rebuild a historical source cache when it is shared, dirty, incomplete or at the wrong commit; preserve diagnostic outputs and the supplied task-1 archive. Reuse a clean independent historical checkout.
- Check the source paths for main_cil, vna_v3, icarl_hooks, vna_ggn and vna_resume before CUDA work; abort clearly on mixed source imports. Store these paths in the resulting report.
- `tests/test_exact_notebook_bootstrap.py`: **3 passed**, covering authenticated historical SHA checkout, repair/reuse of a disposable local shared checkout, and rejection of a current solver imported for the old checkout. No training or notebook smoke run.

Use the updated diagnostic notebook and run from the first cell. It rebuilds the broken historical source directory itself; keep the same vna4_main.tar.gz and CIFAR-100 inputs. Actual T4 profiling remains pending.
