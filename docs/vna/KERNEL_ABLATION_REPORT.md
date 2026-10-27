# P1 fixed-checkpoint kernel ablation — implementation report

Preregistration: KERNEL_ABLATION_PREREGISTRATION.md. Input findings:
KERNEL_ABLATION_INPUT_AUDIT.md. No experimental GPU results are asserted.

Implemented in diagnostics/kernel_ablation.py; production methods/adapters
are unchanged. The dedicated engine substitutes only the projection kernel:
tiny_V K_E, native Gram K, or per-site trace-matched random K_E. The native
kernel remains defined if tiny_V has no positive proposal; its virtual
scaffolding is unused by the projection. All arms retain the original rank-16
proposal machinery and normal 6be6543 LM/certificate/Gram trust region.
Actual parameter-update norms and any radius exceedance are observations,
never a new acceptance filter. Undefined early-exit quantities remain null.

The native P1 loop restores the original end-task1/start-task2 checkpoint and
executes its first task2 SGD step. The complete resulting it2001 state/window
is saved and compared bitwise across arms. Checkpoint task1 epoch traces and
the original it2001 pre-intervention losses must match. Full-arm sources need
a separate baseline checkpoint for the state comparison. Missing checkpoints
trigger historical baseline-task1 reconstruction with exact 100-epoch trace
validation; task20 checkpoints are never used as task1 states. The explicit
s1 missing-record exception below uses exact anchor verification instead.

Eight GIR-off attempts run on copied model/optimizer/engine state with a
fixed intervention window, held-out official-test losses, native power10 GGN
and full seen-test CNN accuracy. Measurement state/RNG isolation is asserted.
Copied-chain progress is saved per attempt. The native training loop then
receives only attempt1's state, with its engine counters/cadence and followup
tracker, and continues through task5 using that arm's kernel with ordinary GIR.
Teacher, exemplars, optimizer, scheduler, gradients, requires_grad and RNG are
retained. Native epoch checkpoints and exit75 remain the continuation mechanism.
Reused prefix elapsed time is recorded separately from the new run's duration.

s1 first replays the original full arm to task2 epoch43. Its audit reports the
first differing intervention fields and first epoch loss/LR mismatch, plus
actual deterministic/CuDNN/TF32/CUBLAS/library/device settings. No historical
GPU flags are fabricated. conclusion_allowed is exactly not differences;
original_flags='unrecorded' is recorded beside current flags, with no veto
from missing historical flags. Failed replay still gates interpretation.
If s1 primary protocol/epochs cannot be found, replay is skipped explicitly:
the same it2001 L/KD must equal 2.853116586477115 / 1.2374861653019964 and
interpretation_scope=no_replay_audit is recorded. No original task1 trace
equality is asserted when its epochs are absent. A missing protocol is derived
from the baseline setup and s0 frozen schedule, with provenance recorded.

Notebook: vna5_kernel_ablation_s01.ipynb; GPU0=s0, GPU1=s1 (replay first),
each lane vna -> kernel_self -> vna_rand. Seeds 2–7 are not accepted by this
runner. The notebook uses a real immutable source commit and no overlay.

Focused local validation: **38 passed**, 36 upstream deprecation warnings,
7.11 seconds, tests/test_kernel_ablation.py and tests/test_vna_kaggle_restore.py.
Includes original-6be fixed
system bitwise comparison on float32/64; native-kernel substitution; per-site
random trace/rank; complete first-step restoration; observation isolation;
source/resume rejection; discovery excluding own continuation outputs; and
algorithmic replay comparisons and terminal-result identity/horizon validation.
Native checkpoints also record runtime/GPU flags; a changed resume configuration
is rejected before its next SGD update. Actual log: KERNEL_ABLATION_PYTEST.log.

The generated notebook cells are syntax-checked without execution. Full native
checkpoint reconstruction/replay/continuation and T4 timings are not tested.
The local Downloads inventory and vna4_main.tar.gz contain no primary s1
protocol/epochs at the requested path. Recursive Kaggle input discovery checks
attached copies before selecting the registered no_replay_audit exception.
Neither exception nor anchor matching establishes s1's divergence cause.

## Review disposition — 2026-10-27

| Finding | Disposition |
|---|---|
| P0 missing inputs | Historical baseline task1 recovery is now authorized with exact 100-epoch validation; s1 missing original records uses an explicit anchor-only scope, never a claimed replay match. |
| P0 legacy Gram trust radius | Actual VNA displacement is measured before GIR in both chain and continuation; violations/missing nonzero-step audits block interpretation. The reference update is unchanged. Diagnostic timing is excluded from the algorithm budget. |
| P1 rank matching | No rank-matched claim. Native K is untruncated, not empirically proved full-rank. r=16 is incoming directions/site, not the global rank of K_E. |
| P1 legacy engine load | Strict kernel tags remain mandatory for ablation resumes. Explicit initial legacy imports retain solver/adapter state and add only the selected kernel tag; incompatible arm/policy/solver/K0 still refuse. |
| P1 random comparator | One seeded trajectory/seed, not repeated-random evidence. random_superiority_claim_allowed=false; no extra draws/held-out seeds are added as experiments. |
| P2 diagnostic meaning | Pre-VNA V/V_O, native versus virtual spectral operators, and post-VNA versus post-GIR stages are explicit. Legacy virtual lambda_E/ratio are labelled unsuitable for native K. Power10 ratios are descriptive; bound_certified=false. |

The intended definitions are V=z.T K_projection z and V_O=z_O.T K_projection z_O
before the update. Virtual lambda_E estimates lambda_max(J_E.T H J_E);
native-K spectral diagnostics estimate lambda_max(J.T H J), not the virtual
scaffolding eigenvalue. Both use the logit Hessian of the same window loss.
An approximate eigenvalue is not a certified upper bound. None of these
metadata corrections modifies the native acceptance filter.

New focused checks cover actual norm before GIR (even when a later swap has a
large displacement), audit-time exclusion from budget control, read-only apply
without persistent scratch state, legacy import preserving mu/nu/A, strict
cross-kernel resumes, and operator-correct diagnostic ratios. Adopted chain
step1 is counted once in the combined audit. No GPU violation is claimed.

Recovery checks cover exact 100-epoch comparison of loss/accuracy/LR, missing
checkpoint discovery, s1 fallback discovery and anchor validation, historical
commit selection for both seeds, no export on mismatch, provisional s1 trace
labelling and propagation of child exit75. Historical training is mocked in
these CPU tests; no claim of real T4 trace reproduction is made.

Restore fix: reference outputs from different attached datasets/archives are
kept in separate source directories, including attachments under the common
Kaggle datasets/ prefix. Historical flag_rates.json files no longer collide
in the active output tree. Only the exact source-pin/run-tag prefix is restored
into STAGE_B for resume; conflicting bytes for that selected run still refuse.
Reference copies live outside the output archive. Default restore callers keep
the existing strict merge and loose-file-over-own-stale-archive behavior.
Focused tests cover separate loose/archive sources, unread unrelated prefix
files, strict resume conflicts, stale tar precedence, unsafe archive members
and notebook wiring. No experimental acceptance or kernel logic changes.
