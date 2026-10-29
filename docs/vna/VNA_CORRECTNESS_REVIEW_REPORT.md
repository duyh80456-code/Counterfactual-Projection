# Projection correctness review — 2026-10-29

Applied supplied patch; added boundary-KKT and invalid-RHS regressions and
updated mathematical documentation. No residual thresholds or physical gates
changed. Patch SHA256: b6196f99f9fd01b4b713783a3573f57bfa95b33295738abca7421d0326175408

Python3.10 / Torch2.2.2 CPU, REQUIRE_UPSTREAM=1,
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python, PYTHONPATH=., OMP/MKL threads=1.
- test_vna_correctness_integration.py, test_standard_cl_math.py,
  test_standard_cl_audit.py, test_standard_safety_revision.py: 68 passed.
- Additional boundary/nonfinite regressions: 2 passed.
- test_standard_cl_completion.py::test_nearly_parallel_halfspaces: 2 passed.
Total: 72 passed, no skips. See VNA_CORRECTNESS_REVIEW_TESTS.log.

No full pytest, GPU benchmark or real-data training was run.
