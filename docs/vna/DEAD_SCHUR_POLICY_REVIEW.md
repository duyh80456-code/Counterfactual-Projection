# Dead-slot Schur policy review (2026-10-22)

## What changes

The current source uses `policy_version=schur_gamma_bound_rho25_v2` for both `gram`
and `cg_ir`. This is an algorithmic revision, not a speed-only patch:

1. Native GIR defaults to dead slots. `--gir-live` enables live-slot trials as an
   ablation, in `experiments.main_cil`, `experiments.pilot_v3` and the exact-solver
   diagnostic. Existing top-2 candidate and live-ablation top-4 limits remain.
2. Dead slots use direct Schur gain and a residual-error gate, not a difference
   between two minimized quadratics. Live ablation retains the q-difference and now gates it by the sum of both q error bounds.
3. LM accepts a finite ratio only when `rho >= 0.25`. All later solves use that
   accepted `mu_star`. The other damping/step/window rules remain in place.
4. Main CIL `Intervention.after_step` does nothing before task 2, including during
   preflight. K0 is now calibrated on the three task-2 points. This condition is
   scoped to CIL, not the LoP regression/MNIST first task.
5. Main CIL records before/after GGN spectral estimates, dead-mask comparisons,
   and the next 20 native SGD losses. These observations add no stability gate.

Nocedal–Wright Algorithm 4.1 shrinks the trust-region radius below ratio 1/4,
while its acceptance parameter eta can be smaller. Requiring acceptance at 1/4
is the user's requested stricter policy, not a universal requirement of the
trust-region method. See [the primary text, Algorithm 4.1](https://people.vts.su.ac.rs/~papzoli/Optimalizacio/Irodalom/Nocedal_Wright_Numerical%20Optimization.pdf).

The claim about 65 harmful/useless live swaps is supplied by the user; this
change does not independently verify that archived run.

## Direct gain and its residual bound

All quantities below belong to one frozen, damped GN problem; they do not
promise improved future nonlinear SGD loss. Let M be the PSD metric and

```
A0 = J^T M J + mu I
C  = J^T M V
D  = V^T M V + mu I
S  = D - C^T A0^-1 C >= mu I
s* = V^T (e - M J x0*)
gain* = 0.5 s*^T S^-1 s*
```

Replacing a truly dead slot removes zero Jacobian columns. The outgoing
coordinates of the augmented-system solution equal `t* = S^-1 s*`.
Thus the code computes `gain = 0.5 s^T t` from the base residual and those
coordinates. It reuses the augmented solve; it does not form S explicitly or
solve once for each Schur column.

For computed base and augmented solutions, let r0 and ra be their parameter-
system residuals. Put

```
a = sqrt(trace(V^T M V) / mu) * ||r0||
b = ||ra|| / mu
error_bound = 0.5 * (a ||t|| + (||s|| + a) b)
```

Then `||s-s*|| <= a`, `||t-t*|| <= b`, so
`|gain-gain*| <= error_bound`. The first inequality uses
`C^T A0^-2 C <= C^T A0^-1 C / mu <= V^T M V / mu`;
the second uses the augmented matrix's lower eigenvalue bound mu.
The final scalar inequality follows by adding and subtracting `s*^T t`.
A swap requires finite `gain > max(0, error_bound)` and the existing actual
loss/KD window checks. A failed CG-IR convergence test still rejects the solve.

CG-IR measures both parameter residuals with its actual double model/operator.
Gram derives upper bounds from output residuals using
`||J^T r|| <= sqrt(||G||_2) ||r||`; the maximum absolute row sum bounds
`||G||_2`, and the augmented bound adds `||V||_F^2`.
This certifies linear-solve error for the represented frozen system. It does
not certify rounding in Jacobian/Gram/metric assembly or nonlinear loss changes.
The formal theory lemma can be incorporated separately by the user.

A slot must be dead on the frozen native window, the actual double target
window for CG-IR, and the current post-VNA window. Deadness is checked again
before installation because upstream corrections can revive a slot. Probe
256 disagreement is logged only; it does not disqualify a window-dead slot.

## Logs and resume

- `interventions.json`: `stability_before/after`, `lambda_max_ggn_before/after`,
  `stability_seconds`, `wall_seconds`; each trial carries `slot_kind`,
  `gain_method`, `gain`, `gain_accepted`, `linear_solve_error_bound` and the
  residual/norm terms used in the bound.
- The spectral values are **10-step power estimates**, not exact eigenvalues or
  upper bounds. Before/after use the same dedicated generator and active loss
  window. They neither advance training RNG nor mutate BN state.
- Each stability mask record lists window/probe dead indices, intersection,
  window-only/probe-only indices and disagreement fraction. The probe is the
  same seeded test-data selection used by task-end diagnostics (256 real
  samples; tiny fake-data tests may have fewer).
- `stability.json` and native checkpoint intervention state contain separate
  followups for effective applications. The first 20 subsequent losses are
  sampled before native SGD, once per training step. Records survive checkpoint
  resume and can overlap. A new task closes incomplete records as `TASK_ENDED`;
  nonfinite losses are logged with a finite flag, without relaxing divergence
  checks. Artifacts are persisted with the existing main-run persistence flow.
- Diagnostic profiles retain immediate post-application `final_L_KD`, then run
  20 native SGD steps before `PROFILE_COMPLETED`. `post_application_sgd` holds
  those losses; a failure or deadline retains an explicit partial status.
- Stability measurement is accounted separately from correction compute used
  by K0/budget; ordinary SGD timing subtracts the whole intervention wall time.
  All work still consumes real wall-clock/deadline time.

Cross-policy checkpoint and Main schedule reuse are rejected. A baseline task-1
checkpoint is conceptually reusable because task 1 now has no intervention;
this does not add an importer or authorize relabelling arbitrary historical
full-run checkpoints. A task-1 comparison test checks native parameters, RNG and
epoch losses bitwise on the tiny upstream-backed fixture.

Historical `d88abc4` full-task1 inputs remain useful as common *diagnostic theta*;
they are not evidence of new-policy task1/baseline identity. New-policy replays
start fresh damping/install-age state rather than inherit task1 interventions.
Strict profile comparison refuses a policy mismatch. The archival Gram test
now compares fixed-system direction math to 8c56d02, since whole-application
identity is intentionally broken by this requested policy change.

The notebooks are regenerated with one updated source pin. The dedicated Gram
M1s0 notebook gates training on a completed current-policy Gram profile and a
strict fixed-system check against 8c56d02 (historical OutputSolve and direction
source). It does not compare changed LM/GIR decisions against the old policy.
Standalone solver profiles compare same-policy Gram/power10 and CG-IR/power10.
See GRAM_M1_IR_NOTEBOOK.md for lanes, inputs, cache/resume and remaining holds.

## Live-slot ablation error gate (follow-up)

Both slot kinds require finite positive gain strictly above their corresponding
linear-solve error bound. For live Gamma, the bound is the sum of the base and
swapped q bounds. For `q = -0.5 b^T A^-1 b`, its solve error is bounded by
`||b|| ||r|| / (2 mu)`. CG-IR already records this from the double residual.
Gram uses `||b|| <= sqrt(g) ||e||` and `||r|| <= sqrt(g) ||r_out||`,
where `g` is the Gram spectral-norm upper bound. For the swapped operator,
`g <= base.norm_bound() + ||V||_F^2`; dropping columns cannot increase the
PSD Gram. This bound still excludes jump/assembly rounding as described above.
Logs include `base_q_error_bound` and `swapped_q_error_bound`.

The policy identity is bumped to v2 so prior live-ablation runs cannot resume
with a new acceptance rule. Main preflight still finishes after exactly three
task-2 points; its integration test checks empty task-1 timing and positive K0.
Tests deliberately perturb base/augmented solutions and check actual Schur
gain error <= bound, plus live Gram Gamma error <= summed bound. Both backends
have a test rejecting a positive live Gamma below its bound.
