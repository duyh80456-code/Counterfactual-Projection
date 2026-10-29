# V2 review implementation — 2026-10-29

Implemented directly from the supplied report; its referenced patch/source ZIP
were not attached. Report SHA256: 4b35af1329ae6e16373e45f521b0e5c1c9b9a2099f036a98a01ee41f10f77593

Optional physical_update_norm and cap/matched policy propagate through H1
correction config and H4 JSON/CLI. Explicit step update_norm overrides config.
Default cap arithmetic is retained. Matched trial0 uses budget/||d|| times
(1-16*eps_dtype); later trials halve it. Both nominal and measured norms are logged.
Measured norm must still satisfy the budget: roundoff may reject trial0 even
with a nominal match. Accepted norm and compute are not matched by this protocol.

Paired h1/h4_physical_matched_v1/v2.json differ only in candidate strategy.
The absolute norm0.01 is an unvalidated example, not a tuned benchmark choice.
H3 logs conditional surrogate/recovery separation and eligibility/uncertainty
per candidate plus counts. Eligibility uses the original strict comparisons.
H2 calibration/deployment drift-guard consistency is preserved.

Tests run with Python3.10/Torch2.2.2, actual Avalanche0.4 and pinned upstreams,
REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1, pytest -q -rs -o addopts='' on:
- tests/test_vna_v2_review.py
- tests/test_vna_algorithm_v2.py
- tests/test_vna_correctness_integration.py
- tests/test_standard_cl_audit.py
- tests/test_standard_safety_revision.py
- tests/test_cl_solver_acceptance.py
- tests/test_standard_benchmark_protocols.py

H4 fake-data CPU smoke completed2 domains using the new V2 paired config.
This smoke verifies config/runner propagation; actual nonzero Pareto correction
is checked separately on a fixed tiny checkpoint in the regression tests.
Existing eight notebook plans keep their original configs when repinned.

No full pytest, original-data GPU benchmark, matched-compute claim or
accuracy/plasticity improvement claim.

Final regression result: 86 passed, 32 warnings in 117.97s (0:01:57)
See VNA_V2_REVIEW_TESTS.log and VNA_V2_REVIEW_H4_SMOKE.log.
