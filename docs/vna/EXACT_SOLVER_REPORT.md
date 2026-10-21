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
| `diagnostics/exact_solver.py` | Independent one-T4 subprocesses for old/new profiling and five-epoch task-2 A–D replay. Select/validate the archived checkpoint; keep B/C at d88abc4, carry checkpoint mu into D, suppress optimizer resets only for C. Per-step SGD JSONL, GGN power-10 every 10 steps and after intervention, first nonfinite tensor, explicit diagnostic identity, safe epoch resume and exit-75 propagation. |
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

1. The existing task-1 checkpoint, loose or inside a TAR/ZIP output archive; its filename need not be `vna4_main.tar.gz`.
2. Official CIFAR-100 input and the existing GitHub secret used by notebook bootstrap.

Published notebooks load the implementation from their immutable Git commit and set `SOURCE_DIGEST=None`; no source ZIP is required. A ZIP is only needed for notebooks explicitly built with an uncommitted source overlay.

Use `MODE='profile'` first. With two GPUs it runs B at d88abc4 on GPU0 and D at the pinned implementation commit on GPU1 concurrently. With one GPU it runs B then D on GPU0. Reports are under `outputs/exact_diagnostics/profile/{B,D}/`; they include CG iteration/residual/cost, native parameter and vector dtypes, float32/64 JVP/VJP timings, Gram metadata, LM trajectories, per-candidate Gamma costs and CUDA peak allocation. Then use `MODE='nan'` for A/B/C/D, each for five epochs of task 2 from the same checkpoint. Exit 75 and epoch checkpoint resume remain available; no result is a main comparison score.

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


## 2026-10-21: automatic input checkpoint discovery

The previous notebook searched only the literal filename vna4_main.tar.gz and required exactly one hit; its error did not distinguish no hits from duplicate attachments. The notebook now scans `/kaggle/input` for loose checkpoints and TAR/ZIP members, follows mounted symlinks with cycle detection, validates the original checkpoint metadata, deduplicates identical bytes and prints the chosen source plus SHA256. Different valid checkpoint states cause an explicit ambiguity error; no match reports the scanned inventory/rejection reasons. The selected raw bytes are cached once under `outputs/exact_diagnostics/task1_source.pt` and passed explicitly to every branch.

- `tests/test_exact_notebook_bootstrap.py`: **6 passed**, including renamed archives, loose checkpoints, ZIP copies, mounted directory symlinks/cycles, duplicate input copies, incompatible metadata and distinct compatible sources. RNG state is unchanged by discovery.
- Ran discovery on the actual supplied Downloads/vna4_main.tar.gz: **5 checkpoint members scanned, 1 unique compatible task-1-end checkpoint selected**. SHA256: `76c9a041b71c994483ef1ab942c476f84dd9d75de7b0a03bc15fbb0584af2058`.
- T4 profiling/training was not run for this input-discovery fix. M1 remains held.


## 2026-10-21: diagnostic work split across two GPUs

At the user's request, the notebook now detects available GPUs and uses up to two subprocess lanes. Profile mode maps B to GPU0 and D to GPU1 concurrently; NaN mode runs A then C on GPU0 and B then D on GPU1. One-GPU sessions retain the original serial order. Each child sees one physical GPU as logical cuda:0 through the existing gpu_env/launch_logged helpers. The historical checkout and task-1 checkpoint selection finish before either lane starts.

Output/checkpoint/report paths remain separate per branch. Future exceptions propagate; archives are saved in finally. A false launch result (including exit 75) stops the remaining jobs on its lane, while the existing shared pause/deadline machinery gates new launches. Completed branch reuse and exact resume validation remain in the child driver. Existing running sessions keep their original notebook code; the new scheduling applies when using this updated notebook.

Reports now include execution.parallel_lanes and execution.cuda_visible_devices. Concurrent measurements share host CPU and I/O, so interpret them in this recorded timing context rather than labeling them as isolated serial measurements. No authoritative K0 has been recalibrated. M1 remains held.

Focused local checks: tests/test_exact_gpu_lanes.py covers two-GPU overlap in profile/NaN modes, one-GPU serial fallback, paused-lane stopping and exception/archive propagation using subprocess stand-ins. No T4 training/profiling or notebook smoke run was performed.


## 2026-10-21: memory bound for derivative chunks after actual T4 OOM

The supplied full traceback confirms that profile_D fails in methods/vna_exact.py:gram at vmap(products.JT)(E), during the first 64-vector VJP block. The allocation request is 512 MiB; PyTorch allocated 12.77 GiB, reserved-but-unallocated 1.56 GiB, and total process usage is 14.47 GiB on a 14.56-GiB T4. Both lanes started independently (B on physical GPU0, D on physical GPU1); D's local error says GPU0 because each child sees its assigned physical GPU as logical cuda:0. The trace does not show an allocation failure for the final Gram or a Cholesky factor.

At task 2 the N=128, C=10 output Gram is 1280 squared float64 entries, 12.5 MiB. The large intermediate batch of VJP activations is the identified failure point. Change the existing chunk-size default from 64 to 4 for both the Gram and GIR Jacobian-column helpers. This limits simultaneous derivative vectors by a factor of 16; it does not claim total VRAM use falls by exactly 16. Record gram_chunk in completed Gram metadata. Keep N, every output/Jacobian column, float64 Gram/factors, exact formulas, solver selection and the two-GPU lane separation unchanged.

This is the memory-dependent chunk selection allowed by prompt section 4.1, not a new approximation or fallback. No automatic OOM retry, CPU offload or allocator setting is introduced. The m>=12800 timing/resource stop is retained. Detach the scalar SGD loss for logging to remove the unrelated requires-grad conversion warning without changing training.

Focused local checks: all 14 tests in tests/test_vna_exact.py passed. The file now includes a public-vmap interception check that default batches never exceed four vectors, while Gram and Jacobian columns equal direct dense products. Existing T29–T31, NaN, rollback and resume checks remain in that file. Actual T4 peak VRAM and runtime with chunk=4 are still unmeasured; no successful T4 profile or feasible final-size Gram is claimed.
