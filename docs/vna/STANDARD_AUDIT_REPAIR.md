# H1–H4 audit repair — 2026-10-29

Implementation details and prospective protocol changes are recorded in the
2026-10-29 audit entry of PREREGISTRATION_CHANGES.md. No historical run is reclassified.

Main notebooks: H1/H2/H4 retain paired baseline/full lanes. H3 now queues CBP
then BP on GPU0, full then VNA-only then GIR-only on GPU1. All five H3 jobs use
100tasks ×50epochs and the same native optimizer, data order and resets. Main
cadence remains4000. Exact T4 timings are unmeasured; five arms can require
several11-hour sessions. Do not silently cut the horizon.

Four additional `standard_cl_H*_ablations_s0.ipynb` notebooks keep ablations
separate: no-PCG/no-retry solver (certificates unchanged), train-only gate,
H2 no-geometry and ER, H3 cadence1000/8000, H4 mixed-domain reference and solver/
gate controls. Main notebook tags differ from ablation tags; methods are never
silently replaced. Additional seeds are a later empirical phase, not fabricated
CPU evidence. No full pytest or GPU training is required to build these artifacts.

`acceptance_by_task.json` and `.csv` include both GIR and VNA streams. They
separate attempted/certified/candidate/effective/quality stages; neutral is not
improvement. `accuracy_matrix.json` and existing task records retain accuracy,
forgetting/BWT and old/new-task information. H3 task records include AUC and
early learning in addition to raw epoch curves.

Seeded operator probes test actual matrix-free products, not global SPD. The
relative error bound still assumes a valid lower bound. The hard inference gate
uses the training window and is not held-out generalization. No real GPU
checkpoint acceptance, multi-seed effect or accuracy improvement is claimed.
Native Avalanche tests require its isolated Python environment; unavailable
local dependencies must be reported as skipped. The upstream H3 Google Drive
source can still fail on quota/permissions; attach compatible original data in
that case. This patch cannot repair remote dataset permissions.

Validation evidence (CPU, Python3.12, REQUIRE_UPSTREAM=1,
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python, OMP/MKL threads1):

- STANDARD_AUDIT_CPU_TESTS.log:117 passed,4 skipped in235.86s across affected
  solver, math, audit, pipeline, completion, benchmark, finalize/notebook files.
  All4 skipped tests require unavailable Avalanche; no full repository pytest.
- STANDARD_AUDIT_REPORT_TESTS.log:12 passed in5.53s, overlapping focused tests
  plus the zero-intervention-task report and actual collector export.
- STANDARD_AUDIT_H3_TESTS.log:5 passed in7.20s, native synthetic ImageNet wiring
  and bitwise interruption/resume for CBP/BP/VNA/GIR/full. Correction calls are
  stubbed in this wiring test; separate math/transaction tests exercise them.

These groups overlap and must not be added as independent experiments. No
accuracy, acceptance-rate or wall-time conclusion on T4 follows from CPU tests.
