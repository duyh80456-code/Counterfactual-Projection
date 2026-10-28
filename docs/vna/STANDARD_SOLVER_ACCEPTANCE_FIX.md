# Shared solver/acceptance patch — 2026-10-28

Applied `/Users/pxokq/Downloads/vna_solver_acceptance.patch` without conflicts to
the current repository. Existing CUDA startup, Agg backend and recursive input
scan fixes are retained. Numerical behavior and retry diagnostics follow the
supplied patch; no algorithm-specific baseline changes.

Validation (Python3.12 CPU, torch2.2.2):

```
REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/vna_mpl_312 \
.venv/bin/python -m pytest tests/test_cl_solver_acceptance.py \
 tests/test_standard_cl_math.py tests/test_standard_cl_audit.py \
 tests/test_standard_cl_pipeline.py tests/test_standard_cl_completion.py \
 -q -o addopts=''
```

82 passed, 36 warnings in86.21s. Includes ill-conditioned SPD, deterministic
Jacobi probes, damping retry/lock, rollback, zero update rejection, BN ResNet32,
resume and H2 zero-geometry physical-update identity. No full pytest run.

The prior temporary Python3.10/Avalanche environment is no longer present, so
native Avalanche callback/resume tests were not repeated in that environment.
No actual-data GPU pilot or acceptance-rate/accuracy claim. H1/H2/H4 notebooks
receive a new source commit and fresh output tag; H3 notebook is unchanged.
Old running notebooks remain on their historical source. Do not overwrite old
outputs or resume old-engine state into this new correction configuration.
