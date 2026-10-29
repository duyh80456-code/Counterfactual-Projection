# VNA/GIR correctness integration — 2026-10-29

Applied the supplied patch without algorithmic additions.
Patch SHA256: 729bf9b2975999b3b65500ace36596a83b10421c69e4ad9826987945338f2063

- Audited zero-RHS solve avoids Jacobi/PCG and records an exact certificate.
- Audited feasible metric projections avoid constraint inverse solves.
- H3 records surrogate error intervals and conditional winner certification.
- Nonzero solve thresholds, candidate objective and physical gates stay unchanged.

Affected CPU tests use Python3.10, Torch2.2.2 and real upstream dependencies.
Command: REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/tmp/vna_h2_final_env/bin/python -m pytest
tests/test_vna_correctness_integration.py tests/test_standard_safety_revision.py
tests/test_standard_cl_math.py tests/test_standard_cl_audit.py
tests/test_standard_cl_completion.py -q -rs -o addopts=''

Result: 94 passed, no skips, in 175.35 seconds.
See VNA_CORRECTNESS_INTEGRATION_TESTS.log for full output.
No real-data training or GPU benchmark was run.
