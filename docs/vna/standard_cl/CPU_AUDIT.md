# Focused source and CPU audit — 2026-10-28

Scope: the review of b68a936, not a new algorithm or a dataset/GPU launch.
Only the three focused standard-CL test files are executed. Actual final
pytest output is saved in CPU_AUDIT_PYTEST.log: **48 passed**, 36 upstream
deprecation warnings, **10.54 seconds**, no skips. No full pytest suite,
CIFAR-100 training, real MNIST training, ResNet32 execution or GPU profiling
is part of this review. Research conclusions remain INCONCLUSIVE.

## Actual bugs and fixes

The initial run produced **14 failed, 19 passed**. Every failure entered
`ConstraintSolver.solve`: production and numerical tests pass a Tensor
with shape (2,p), but `torch.stack(gradients)` only accepts a sequence of
tensors. The shared solver now accepts either a two-row Tensor or a
sequence, validates finite aligned inputs and retains the same QP.

The dual sign is correct: with Lagrangian
`(d-d0).T A (d-d0)/2 - lambda.T C d`, stationarity gives
`d=d0+A^-1 C.T lambda`, and active dual sets solve `W lambda=-C d0`.
The numerical tests compare against a SciPy constrained reference, checking
primal/dual feasibility, stationarity and complementarity, including zero,
opposed and rank-deficient gradients. This is not just a shape check.

State review also found that state_dict alone omits nonpersistent buffers
and does not encode freeze flags. State now explicitly snapshots/restores
all registered buffers, their persistence registration, requires_grad,
parameter/buffer shapes/dtypes, modes, gradients, optimizer and global RNG.
CPU tests cover mutated nonpersistent buffers/freeze flags/dtype, rejected
physical candidates and momentum. A GPU/device roundtrip is not tested.

H3 previously executed the chosen callback a second time at commit. A
stateful callback/external random generator could therefore commit a
different, unvalidated candidate. Commit now restores the exact winning
model/optimizer/RNG state recorded during the trial. A regression uses a
callback that would worsen loss if called twice. This repairs transaction
integrity; it does not add a candidate generator or change H1.

ReplayLoss also clones/detaches supplied teacher logits, so a caller accidentally
passing an autograd graph cannot backpropagate into the teacher. The runner
already supplied detached logits; this hardens the public loss contract
without changing its evaluated objective or default native trajectory. A caller's
later in-place change of the supplied tensor cannot change the frozen window.

## Comparison protocol

`compare_local` verifies the same model/optimizer/buffers/gradients/modes/
freeze flags/RNG and training window/teacher for each arm. Model/window
fingerprints are recorded. Damping is computed once at that fixed checkpoint
and passed explicitly to every correction; random draw index does not
change that damping. This is a fixed-state diagnostic: separate training
trajectories are not claimed to remain at the same checkpoint after updates.

Tiny_V, native K and random belong to the kernel group. GN is separately
labelled as a candidate control, not confused with native-kernel VNA.
All use the same constraint setting, initial LR and acceptance procedure.
Diagnostic optional `update_norm` caps the physical displacement in every
trial. Accepted step norms may differ due to LR caps, rounding and line
search; actual norms are **not** falsely labelled matched. Per-unit-norm
directional gain is also available. Without a common norm budget, the
kernel-attribution eligibility flag is false.

Optional `compute_seconds` supplies the same wall allowance per correction.
Boundary checks cannot preempt a single PyTorch operation. Overshoot is
recorded; even a candidate already applied is rolled back/censored on
overrun, with no method-rejection interpretation. A deterministic test
exercises this late rollback, plus timeout-context cleanup. This is not a
proof of a hard real-time limit or measured equal compute on T4.

All rows expose end-to-end `seconds_total`. Correction `seconds` includes
its snapshots, kernel construction, solves, finite acceptance and diagnostic
measurements. Native reference `update_seconds` separately measures its
forward/backward/optimizer path, while its total includes all diagnostics.
Shared preparation/damping time is reported once as common metadata;
duplicated metadata is not charged repeatedly as per-arm work.

## Native optimizer and BN: separate references

* `native_optimizer`: real training forward, real BN running-statistic
  updates, native backward/optimizer, no correction acceptance filter.
* `native_optimizer_frozen_buffers`: training batch statistics with private
  functional buffer copies, native parameter step, no acceptance filter.
  This is **not** frozen-BN evaluation using running statistics.

The tests verify positive real BN-buffer drift versus zero private-buffer
drift, with identical parameter-step norms at the same initial state. Both
references are explicitly marked ineligible for virtual-kernel attribution
and not matched to the correction protocol. They may be reported as native
references, never as evidence for the isolated virtual-kernel contribution.

## Source conclusions and test limits

| Module | Audited outcome |
|---|---|
| cl_projection | Correct dual/primal sign; real Tensor interface bug fixed; degenerate KKT/SciPy toy reference tested. |
| cl_replay | Incoming/reference parameter gradients checked against direct backprop on the same BN forward in float32/64; shared metric and physical acceptance tested. Default H1 objective/damping/cadence/optimizer unchanged. |
| cl_state | BN, nonpersistent buffers, freeze flags, modes, gradients, momentum, RNG and dtype rollback covered on CPU. |
| standard_cl | Source retains PyCIL head/herding/NME and configured CE+KD/optimizer schedule; no full PyCIL/DataManager/rehearsal/loader-resume run is claimed. One-step ER native-loss/SGD identity has a CPU check. |
| cl_h2 | The same feature-penalty callable enters split and final operators; nonzero penalty and cotangent recomputation tested. Not LPR. |
| cl_h3 | Executes and commits provided in-place candidates, with identity and measured checks. Tested as an API; still no autonomous candidate generator/full plasticity benchmark. |
| standard_domain | Fixed ten-class head and same per-domain transform for train/test; domain identity does not select a head. Protocol/eval source inspected and transform/head unit checks run, but real-data loop not executed. |
| mathematical tests | Direct systems, generalized noncommuting metrics, additive split, teacher-precision Hessian, true residual, KKT and Schur/finite differences; not just finite/shape assertions. |

BN end-to-end toy test performs one native SGD step (including real BN and
momentum updates), then runs local corrections with private state. It checks
acceptance, norm budgets, common damping, full restoration and native-BN
separation. Toy BN conditioning is deliberately explicit: relative damping
0.5, float32 CG tolerance 1e-5 / float64 1e-8. These are **test fixtures**,
not changes to the registered CIFAR defaults (relative damping0.01 and
CG tolerance1e-6). Passing them does not establish default ResNet32 stability.

The three focused files also include the earlier direct systems and linear
resume/unified tests. Passing CPU checks permits further validation; it is
not approval of GPU readiness, statistical superiority or the real-data
pilot. No pilot is started automatically by this audit.
