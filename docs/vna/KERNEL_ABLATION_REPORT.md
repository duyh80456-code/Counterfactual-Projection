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
a separate baseline checkpoint for the state comparison. It never trains a
replacement baseline or accepts task20 checkpoints as task1 states.

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
GPU flags are fabricated. Missing flags or failed replay keep s1 summaries
marked interpretation_allowed=false, even if exploratory computation finishes.

Notebook: vna5_kernel_ablation_s01.ipynb; GPU0=s0, GPU1=s1 (replay first),
each lane vna -> kernel_self -> vna_rand. Seeds 2–7 are not accepted by this
runner. The notebook uses a real immutable source commit and no overlay.

Focused local validation: **16 passed**, 36 upstream deprecation warnings,
6.53 seconds, tests/test_kernel_ablation.py only. Includes original-6be fixed
system bitwise comparison on float32/64; native-kernel substitution; per-site
random trace/rank; complete first-step restoration; observation isolation;
source/resume rejection; discovery excluding own continuation outputs; and
algorithmic replay comparisons. Actual log: KERNEL_ABLATION_PYTEST.log.

The generated notebook cells are syntax-checked without execution. Full native
checkpoint replay/continuation and T4 timings are not tested: supplied inputs
lack valid matching task1 baseline checkpoints and primary s1 protocol/epochs.
These missing artifacts are an explicit execution blocker, not permission to
infer s1's divergence cause or substitute the old live-GIR task1 checkpoint.
