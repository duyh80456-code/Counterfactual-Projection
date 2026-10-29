# Algorithm V2 integration — 2026-10-29

Patch SHA256: 521211d4cf74127146d901913598b1865c07ba3819d0dcb93f8bf7b2098d1a50
All 11 patch targets matched the supplied ZIP immediately after application.
The ZIP was used for comparison, not wholesale replacement of the repository.

Additional integration fix: pass feature_drift_budget into H2 held-out coefficient
calibration as well as applied corrections. A focused regression verifies this.
Original benchmark configs and algorithm defaults remain unchanged.

CPU environment: Python3.10, Torch2.2.2, actual Avalanche0.4 and pinned upstreams.
Command: REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/tmp/vna_h2_final_env/bin/python -m pytest
tests/test_vna_algorithm_v2.py tests/test_vna_correctness_integration.py
tests/test_standard_cl_math.py tests/test_standard_safety_revision.py
tests/test_standard_cl_audit.py tests/test_standard_validation_phase.py
tests/test_standard_benchmark_protocols.py -q -rs -o addopts=''

Result: 98 passed, no skips, 32 dependency warnings, in 147.47 seconds.
See VNA_ALGORITHM_V2_TESTS.log for exact results. Tests include dense Pareto
oracles, feature-gate rollback, actual GIR selection and native runner resume.
No full pytest, GPU benchmark or accuracy claim. Existing notebooks retain
original configs; V2 is selected explicitly through the four new configs.
