# Matched LR acceptance fix — 2026-10-29

Report SHA256: cbc7f9279979065ead78ed85f340d1cd8c803fb32544dfe59d437b39c9c1a1ee
Compared supplied recovered ZIP with live source and applied only the relevant
line-search change plus its two regressions. Existing H3 schema, error messages,
upstream dependencies and repository-specific fixes are preserved.

Red-before test reproduced REJECTED in matched although cap accepted the tiny-LR
correction. Matched now retains its trial0 and uses
min(s0/2**k, lr_limited_step/2**(k-1)) thereafter. The cap schedule and maximum
trial count are unchanged. LR reference diagnostics appear only in matched.
With one trial no LR fallback is tested. No PCG or acceptance thresholds changed.

Tests use Python3.10/Torch2.2.2 CPU and real pinned upstream dependencies,
REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONPATH=.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1, pytest -q -rs -o addopts='' on:
- tests/test_vna_v2_review.py
- tests/test_vna_algorithm_v2.py
- tests/test_standard_cl_audit.py
- tests/test_standard_safety_revision.py
- tests/test_cl_solver_acceptance.py
Final two regressions also verify the matched-only reference log field.
See VNA_MATCHED_LR_FIX_TESTS.log for full output and red-before reproduction.

No full pytest, GPU run, real-checkpoint solver claim or dataset benchmark.

Affected regression result: 64 passed, 2 warnings in 42.84s
Final focused regressions: 2 passed, 6 deselected in 2.82s
