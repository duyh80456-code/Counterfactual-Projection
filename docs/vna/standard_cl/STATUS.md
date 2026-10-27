# Delivery and evidence status — 2026-10-28

User instruction: implement code, **do not run**, push authorized.

| Deliverable | Status |
|---|---|
| Repository/theory/old-code audit | Read-only audit documented |
| Shared generalized matrix-free projection | Implemented, NOT RUN |
| Two-halfspace QP and residual/KKT checks | Implemented, NOT RUN |
| BN/optimizer/RNG transactions | Implemented, NOT RUN |
| ER with herding / iCaRL-style H1 pilot runner | Implemented, NOT RUN |
| Intervention and unified modes | Implemented, NOT RUN |
| H0 fixed-state local controls | Implemented API, NOT RUN |
| H2 consistent feature penalty | Independent API, NOT RUN |
| H3 derivative/Schur/finite-candidate search | Independent API, NOT RUN |
| H4 fixed-head Domain-IL runner | Implemented, NOT RUN |
| New mathematical/integration tests | Written, NOT RUN |
| Numerical verification results | NOT RUN, no invented residuals or pass count |
| CIFAR/MNIST benchmark metrics | NOT RUN |
| GPU memory/runtime profiling | NOT RUN; only instrumentation provided |
| Strong compatible CL benchmark comparison | Pending; no adapted baseline/result claimed |
| New H3 full plasticity benchmark | Pending research gate, not scheduled |
| General virtual GroupNorm / dropout support | Explicitly unsupported |

All H0–H4 research conclusions are **INCONCLUSIVE**. Source formulas are
conditional algebraic statements, not evidence that these new implementations
passed tests. Old OOM and incomplete replay are not classified as negative
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

## Next validation sequence (not executed)

1. Run only the new focused mathematical/integration test files. Resolve
   float32 failures, missing reference-QP dependencies and resume mismatches.
2. Profile a real small BN intervention; record resource failures as such.
3. Run the paired three-task H1 pilot and compatible native/GN/random
   controls, keeping the held-out seeds untouched.
4. Only with those gates passed, consider full tasks/strong comparisons
   and evidence-based independent H2/H4/H3 experiments.

No phase gate is represented as passed in this report.

Static-only review: Python sources were parsed/compiled as syntax trees and
the JSON config parsed without importing the new modules or executing their
code. This is not a test, mathematical verification or runtime validation.
