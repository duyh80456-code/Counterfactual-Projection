# Fresh-process CUDA statistics startup fix — 2026-10-28

Observed Kaggle H1 failure: both icarl_s0 and icarl_replay_vna_s0 raised
RuntimeError: Invalid device argument at reset_peak_memory_stats, before any
training update. The separate preflight had initialized its own CUDA context;
a new training subprocess did not inherit that initialization.

Shared reset_cuda_peak_memory_stats now initializes CUDA, selects the requested
logical device, then resets allocator statistics. CPU is a no-op. H1/H2/H3/H4
runners all use it at their original statistics-reset point. No optimizer,
loss, seed, dataset, training budget or algorithm settings change.

Each notebook additionally probes fresh subprocesses with CUDA_VISIBLE_DEVICES=0
and1. Both use logical cuda:0 inside their masked process, matching the real
training commands. Both probes must finish successfully before either training
lane is launched. Failure still reaches the archive/finalization path.

Focused verification, overlapping groups (no full suite):
- Repository Python: CUDA init-before-reset tests, CPU no-op, notebook compilation,
  tiny CIL two-task/resume, MNIST resume, native ImageNet fixture/resume:
  10 passed,36 warnings,26.44s.
- Added fresh-child probe failure gate plus startup/notebook checks:
  5 passed,14 warnings,3.40s. Checks that GPU1 probe failure launches neither arm
  and still archives output; CUDA APIs are mocked, no GPU pass is inferred.
- Actual Python3.10/Avalanche LPR and Geometry online resume:
  2 passed,12 warnings,18.82s.

No actual CUDA/T4 execution or benchmark training was performed locally.
Use rebuilt notebooks with the new source pin; existing dataset attachments are
unchanged. The attached failing H1 log shows no SGD step to recover.
