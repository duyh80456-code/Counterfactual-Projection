# Solver/acceptance V2 — 2026-10-29

Integrated the supplied V2 full patch incrementally on top of V1, retaining
current notebook CUDA, Matplotlib and dataset-discovery fixes. Added full
per-solve diagnostics to every damping attempt and separate acceptance counts
to collector JSON and comparison CSV. Historical accepted events without detail
are explicitly unclassified, never assumed improved.

Default solution-error ratio remains0.1, normalized by approximate solution norm
and mu*regularizer_floor. Its exact-solution-relative implication is at most1/9;
this is not a10% exact-solution certificate or a bound on the combined KKT
update. The configured improvement threshold is mixed absolute/relative,
1e-6*max(1,abs(before)); it is a window-loss diagnostic, not test improvement.

Validation uses only affected solver/math/audit/pipeline/completion tests plus
collector export and notebook checks. Additional focused coverage verifies
nonunit regularizer lower bounds, zero RHS and acceptance counting. Existing
near-parallel-halfspace, BN ResNet32, rollback/resume and H2 zero-penalty checks
remain in the affected test set.

H1/H2/H4 use fresh source-specific output tags. Do not resume V1 engine states
into V2; old in-flight notebooks keep their pinned policy. H3 notebook unchanged.
No GPU training or checkpoint acceptance/accuracy audit is performed here.

CPU results:88 passed (36 deprecation warnings) in138.17s across
`test_cl_solver_acceptance`, `test_standard_cl_math`, `test_standard_cl_audit`,
`test_standard_cl_pipeline`, `test_standard_cl_completion`; collector export
1 passed (11 deselected) in46.04s. Python3.12/.venv; REQUIRE_UPSTREAM=1,
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python, OMP/MKL threads1.
