# Phase0 interrupted — 2026-11-01

Real source81a7d04, unchanged registered data/init/pool seeds and hyperparameters.
SCR measured anchor0 then failed Cholesky at anchor52625. The implementation
incorrectly required Cholesky before the mandatory exact-dead C1 branch. That
ordering is corrected and T2 passes2 cases in4.52s. The exact failed Gram was
not retained, so attributing that specific failure solely to all-zero features
is not independently verified. A new-source SCR measurement is still required.

PM-small reached approximately378912/1200000 updates before a checkpoint file
open failure; the latest valid saved native checkpoint is step370000. Filesystem
free space at diagnosis was67MiB, later109MiB; a small write subsequently worked.
Do not claim the precise I/O cause established from the PyTorch message alone.
The existing checkpoints/data are preserved; no seed or run horizon was changed.

Partial JSON/CSV and plots are in outputs/ett_phase0_partial_report; its ZIP
contains all raw JSONL/CSV/logs and plots. No complete Phase0 gate can be inferred.
No Phase1/2/3 or full pytest/smoke was run. Core verification: T1–T10 initially
20 passed6.29s, dtype remap T8/T10 4 passed5.22s, C1 ordering T2 2 passed4.52s.

Phase0 is INCOMPLETE, not a negative result or a benchmark success.
