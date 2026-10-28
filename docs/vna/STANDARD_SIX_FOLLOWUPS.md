# Six audit followups — 2026-10-29

1. GIR candidate failure is local; only identity/background failure retries the
   whole search. Both early bad/later good and failed-search rollback are tested.
2. Shared PCG adds true-residual iterative refinement at fixed damping, retaining
   strict V2 certificates and reporting total iterations, sampled Rayleigh values,
   preconditioner quality and refinement residuals. Synthetic CPU ResNet32 BN is
   checked at1e-6. This does not establish success on unavailable GPU checkpoints.
3. Train/eval numerical gates share a dtype-based8eps+1e-12 allowance; logs expose
   each gate's thresholds/changes and all failure names. Improvement is still a
   separate significance classification. Double-precision micro-increases above
   that allowance remain rejected.
4. H2 reports geometry/GGN/ridge directional quadratic contributions and ratios;
   no unvalidated automatic geometry-weight increase.
5. Unmocked tiny_V H3/H4 correction tests require actual nonzero updates. H3 also
   requires a real generated-candidate GIR swap; candidate-local failure and
   rollback tests cover failure transactions. These are CPU synthetic fixtures.
6. H4 is described as prior-domain constrained VNA using domain labels at training
   only. The existing mixed-reference ablation isolates that contribution. No
   architectural domain adaptation or empirical superiority is claimed.

All main and ablation notebooks are repinned together into new output tags. Do
not resume earlier policy checkpoints into this revision. No real-data training,
GPU checkpoint replay or multi-seed acceptance/accuracy claim is made here.

CPU evidence:
- Before variance-shrinkage was added, the ResNet32 BN synthetic-window smoke
  with locked mu=10 was ACCEPTED at1e-6;
  one solve improved measured true residual6.69965e-6 ->4.70503e-6 through
  fixed-mu refinement, relative residual8.42598e-7. This does not show that the
  automatic damping schedule succeeds on a real CIFAR checkpoint.
- Broad affected tests:126 passed,4 Avalanche skips,1 obsolete assertion that
  required aborting the entire GIR search. That expectation was replaced with
  candidate-local failure plus preservation of the prior winner.
- Corrected focused safety/finalize tests:30 passed,1 Avalanche skip in23.42s.
  Groups overlap; do not sum them as independent trials. This includes both
  runner tests without mocking correction engines. H3 runner uses a small
  synthetic backbone/data with the actual upstream Backprop and real VNA/GIR;
  a separate generated-candidate test requires a real committed GIR swap.
  H4 uses its actual fake-data DomainNetwork path and verifies nonzero tiny_V
  corrections, gate success and previous-domain references.

No full repository pytest. Native Avalanche and GPU checkpoint audits remain
unverified here; CPU integration is not an accuracy/forgetting benchmark.

Large-system Jacobi now uses probe-variance shrinkage, retaining the exact ridge
and testing a noisy rank-one system against the dense solution. Zero-variance
diagonal probes remain exact. Raw clipping, shrinkage and standard-error fields
make estimator quality observable; no universal convergence improvement is
claimed. The preconditioner never changes the operator being certified.

Final release checks after variance shrinkage and exact-zero-curvature handling:
- STANDARD_SIX_RELEASE_TESTS.log:82 passed,1 Avalanche skip in72.75s; includes
  the strict1e-6 ResNet32 BN certificate check, both real-engine runner fixtures,
  generated GIR swap, candidate-local failures, numerical gates and geometry
  contribution checks. No CG/residual certificate was relaxed.
- STANDARD_SIX_RESUME_TESTS.log:12 passed in86.92s; H1/H2 task resume, H4 domain
  resume, H3 candidate deadline/resume and five native ImageNet arm wiring checks,
  plus H2 geometry-weight-zero bitwise identity and unified-update wiring.

Python3.12 CPU, REQUIRE_UPSTREAM=1, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python,
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1. Earlier broad/overlapping logs above are
historical intermediate checks; the final clean subsets verify the corrected
assertion and final preconditioner. No full repository pytest or GPU benchmark.
