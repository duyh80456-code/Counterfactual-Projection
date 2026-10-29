# Specification and theorem-to-code mapping

Conventions follow the supplied master prompt and
`VNA_CL_Proof_and_Research_Protocol.tex` (27 October 2026).
All vectors use **descent** sign: the physical update is `theta -= s*d`.

Let `A = J.T M J + mu R + P`, with M and P PSD, R SPD,
`mu > 0`. The caller must supply a valid lower bound for R. The operator
does not claim to verify arbitrary matrix-free PSD assumptions automatically.
For H1, M is the output Hessian of the actual CE+KD mean loss, R=I and P=0.
For H2, P=lambda*Jphi.T*Jphi with lambda>=0. With a fixed PSD kernel Q,

```
x = solve(A, g)
z = M J x
d = solve(A, J.T M Q z)
```

Both solves use the same A. Thus g.T*d=z.T*Q*z in exact arithmetic.
`ProjectionOperator.project` implements this generalized form. In production
H1, `ReplayCorrection.propose` shares the total-gradient cotangent, chooses
the virtual/native/random kernel, and solves the final projected system.
The incoming/replay gradients sum to the same total gradient mathematically;
their split alone is not claimed as novelty.

| Requirement | Code / written check |
|---|---|
| T1 unique damped projection | `ProjectionOperator.solve/project`; direct dense toy system |
| T2 descent identity | generalized metric test with noncommuting M/R and a feature penalty |
| T3 native tangent limitation | direction is a real parameter vector; only J*d is realized |
| T4 native-kernel comparison | H0 `native` uses J*J.T; scalar counterexample test |
| T5 no universal virtual advantage | explicit better/worse PSD-kernel examples |
| T6 GN optimality limitation | separate `gn` candidate; no claim of beating its identical surrogate optimum |
| H1 convex projection | `ConstraintSolver`, four active sets, SciPy SLSQP reference tests |
| H2 recompute cotangent | same penalty enters both split and final A; no stale z reuse |
| H3 added-column GN gain | `schur_gain`, compared with two direct minima |
| H3 VNA derivative | `vna_column_derivative`, finite differences with noncommuting PSD matrices |
| KD Hessian consistency | mean CE + student old-softmax covariance / T²; autograd Hessian test |
| KD/global classification distinction | old-logit shift test with changed global winner |
| Physical finite update | line search checks rounded real parameters, not a Gram norm |

## Loss split

For a window of n samples and incoming/replay mask, ER uses
`L_N = sum(incoming CE)/n`, `L_R = sum(replay CE)/n`.
iCaRL-style adds old-logit KD over the **whole window** to L_R.
No T² rescaling is added: the PyCIL KD loss uses T=2 and its Hessian
consequently contains 1/T². `ReplayLoss.total` calls the original loss
definition directly to preserve the native CE+KD evaluation order.
The alternative KD-only reference protects all-CE and KD separately.
All replay CE uses the full currently visible class head, not old-only logits.

## Two-halfspace QP

For candidate d0, minimize `(d-d0).T A (d-d0)/2`, subject to C*d>=0,
where C contains incoming/reference gradients. With lambda>=0,
`d=d0+solve(A,C.T)*lambda`. Normalize nonzero constraint rows for numerical
conditioning; undo that normalization in the reported multipliers.
Enumerate no active constraint, either singleton, or both. Rank-deficient
2x2 dual blocks use a pseudoinverse. Check primal and dual feasibility,
stationarity and complementarity and select the lowest primal objective.
Failure to find a numerically certified active set gives UNCERTIFIED.
Zero/opposed gradients and zero directions are valid cases.

For a candidate satisfying every halfspace, the exact metric projection is
`d0` for any SPD A. The shortcut requires a positive floating-point roundoff
margin for every nonzero normalized row, or an exactly zero candidate.
Boundary cases use the ordinary KKT path. Rows are scaled by their maximum
absolute entry before L2 normalization to avoid norm overflow/underflow.
The shortcut retains the operator's finite-probe audit and deadline boundary.
This changes neither the mathematical projection nor the finite physical
acceptance checks. A right-hand side exactly equal to zero is returned as an
exact zero solution without Jacobi or PCG, also after the operator audit.
Neither shortcut weakens the nonzero-RHS residual certificate.

There is **no trust ball** hidden in this QP. The dyadic finite line search
is separate: initial s is the declared physical LR, at most 12 trials by
default, each halves s. Constrained arms require both actual window losses
to be nonincreasing; VNA-only requires their sum. No test metric enters
acceptance. Candidate trials and accepted physical displacement are logged.

The finite check applies at floating-point tolerance to the computed
first-order gains and by direct <= comparisons to finite losses. It is not
an exact-arithmetic proof, global Lipschitz certificate or accuracy guarantee.
CG uses the true residual, logs `||b-A*x||/mu` (or `/mu*r_min`) as a conditional
solution error bound and refuses uncertified solves. The bound assumes the
declared PSD/SPD properties and lower bound hold. It does not certify all
autodiff/JVP rounding errors. Float32 therefore has its own written tests.

## BN and state

`training_products` uses training batch statistics with private cloned BN
buffers. Physical acceptance and diagnostics use cloned buffers and isolated
RNG. `State` snapshots parameters, buffers, modes, gradients, optimizer and
RNG. Rejection restores them; acceptance keeps the physical parameter delta.
Intervention leaves momentum untouched. Unified mode commits the simulated
optimizer state only when the whole candidate is accepted; it does not
pretend the momentum buffer equals a projected gradient. Eval-mode losses
and functional drift are measured separately on the realized parameters.
Dropout correction currently refuses explicitly because a fixed stochastic
linearization has not been implemented. The target BN ResNet32 has no dropout.


## H2 penalty and constrained interpretation

H2 uses A_lambda = J^T H J + lambda J_phi^T J_phi + mu I in **both**
solves. For a specified convex drift constraint ||J_phi d||^2 <= epsilon^2,
a KKT multiplier can yield such a penalty at its constrained optimum, with
complementarity and feasibility additionally verified. A fixed configured
lambda is only the penalized method; no epsilon or KKT multiplier is inferred
from it. Noncommuting matrix tests compare the complete matrix-free solution
and identity g^T d = z_lambda^T K_E z_lambda to direct dense solves. Feature
JVPs use a frozen linearization, and actual finite drift is logged separately.

## H3 fixed-capacity objective

For each **actual post-swap** network c, recompute g_c, J_c and H_c at the
training window and use Q_c = L_c - 0.5 g_c^T (J_c^T H_c J_c + mu I)^-1 g_c.
The identity and all finite candidates share mu. Minimize Q_c only among swaps
passing componentwise actual loss nonincrease. This accounts for removal and
insertion by recomputation; it is not a nonnegative added-column gain theorem.
The finite candidate GN objective can change nonmonotonically with its measured
VNA diagnostic z_c^T K_E,c z_c. Choosing one objective does not prove improvement
of the other. At most one validated slot replacement is installed per event.

H3 evaluates `Qhat_c = L_c - .5*g_c.T*xhat_c` using an inexact PCG solution.
If `A_c >= m_c I` and `r_c = g_c - A_c*xhat_c`, then the exact surrogate lies
within `Qhat_c +/- .5*||g_c||*||r_c||/m_c`. Candidate and identity intervals
are logged; a winner is marked `surrogate_wins_identity_certified=True` only
when its upper bound is strictly below the identity's lower bound. This is a
*conditional exact-surrogate ordering* for fixed actual networks, not a
guarantee of future-task plasticity or a change to the swap objective.
