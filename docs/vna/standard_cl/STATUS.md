# Delivery and evidence status — 2026-10-28

Original delivery: code only, no execution. Follow-up review: focused CPU
tests authorized and executed; no dataset training or GPU run. See
[CPU audit](CPU_AUDIT.md) and its actual pytest log.

| Deliverable | Status |
|---|---|
| Repository/theory/old-code audit | Read-only audit documented |
| Shared generalized matrix-free projection | Focused CPU toy checks passed |
| Two-halfspace QP and residual/KKT checks | Tensor interface fixed; CPU/SciPy reference checks passed |
| BN/optimizer/RNG transactions | CPU rollback tests include momentum, freeze flags and nonpersistent buffers |
| ER with herding / iCaRL-style H1 pilot runner | Implemented, NOT RUN |
| Intervention and unified modes | CPU toy/resume/rejection scope tested |
| H0 fixed-state local controls | Shared damping/window/norm-budget checks; native BN references separated |
| H2 consistent feature penalty | Same penalty in both operators verified on CPU toy |
| H3 derivative/Schur/finite-candidate search | Small-system and in-place finite-candidate API checks passed; no generator/benchmark claim |
| H4 fixed-head Domain-IL runner | Implemented, NOT RUN |
| New mathematical/integration tests | Three focused files executed; see actual log |
| Numerical verification results | CPU toy verification only; not a full-model certificate |
| CIFAR/MNIST benchmark metrics | NOT RUN |
| GPU memory/runtime profiling | NOT RUN; only instrumentation provided |
| Strong compatible CL benchmark comparison | Pending; no adapted baseline/result claimed |
| New H3 full plasticity benchmark | Pending research gate, not scheduled |
| General virtual GroupNorm / dropout support | Explicitly unsupported |

All H0–H4 research conclusions are **INCONCLUSIVE**. Source formulas are
conditional algebraic statements, distinct from the bounded scope of executed
tests. Old OOM and incomplete replay are not classified as negative
method results. No test set selects hyperparameters or acceptance.

## Research questions, by branch

* H1 targets incoming/reference gradient interference. Native replay already
  mixes old data (and iCaRL-style KD). The virtual candidate changes the
  proposed direction; the QP changes its stability/plasticity trade-off.
  Only local directional/window checks apply. Whether it beats native
  constrained/GN/random or a cheaper strong baseline is unknown.
* H2 targets feature drift left unresolved by loss constraints. It changes
  the metric in both solves and is separate from LPR. Necessity, benefits
  and extra solver/feature cost are unmeasured.
* H3 targets tangent-capacity/plasticity loss. Schur gain for adding columns
  is not a live-swap gain and not a VNA gain. Finite swap search accounts
  for the removed slot and measured jump but has no whole-history guarantee.
  A suitable new benchmark integration awaits evidence; existing baseline
  hooks are preserved rather than silently redefined.
* H4 tests a fixed label space under domain change, using ordinary ER
  reservoir replay and a fixed classifier. Parameter-window certificates
  do not protect BN running-statistic effects on eval outputs. Generality,
  forgetting and cost-benefit remain unmeasured.

## Next validation sequence

1. The focused CPU files have run, including the numerical QP reference.
   Preserve these checks as regressions; they do not validate a full PyCIL run.
2. Profile a real small BN intervention; record resource failures as such.
3. Run the paired three-task H1 pilot and compatible native/GN/random
   controls, keeping the held-out seeds untouched.
4. Only with those gates passed, consider full tasks/strong comparisons
   and evidence-based independent H2/H4/H3 experiments.

No CIFAR/MNIST-training, ResNet32 or GPU phase gate is represented as passed.

The original static-only review parsed/compiled Python syntax trees and
the JSON config parsed without importing the new modules or executing their
code. This is not a test, mathematical verification or runtime validation.
