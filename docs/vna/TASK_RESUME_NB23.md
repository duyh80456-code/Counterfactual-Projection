# Full resume + observability for nb2/nb3 (B1, B2, B3, B4)

Branch `vna-phase0`. No commit. Do NOT run pytest. Allowed checks: pyflakes, building notebooks, and per
runner ONE fake-data CPU round trip (< 3 min): run N epochs/tasks straight vs. stop after k + resume,
compare final params/optimizer/log rows bit-for-bit (print the comparison). Measurement semantics must
NOT change (no change to ranks, nulls, CG settings, μ) — only checkpointing, logging, deadlines.

Context from Kaggle: training runs fine (B1 resnet8x4 ~20 s/epoch, B2 ~21–34 s/epoch, teachers 79.42 /
75.61), but each measurement point takes hours and prints nothing, and B1/B2/B3 cannot resume.

## 1. Resume must restore the complete state
General rule: a resumed run continues **bit-identically** to an uninterrupted one (CPU). Checkpoint =
everything that influences the future: model(s), optimizer (incl. momentum), scheduler/LR state, epoch /
task counters, best-acc and any other upstream loop locals that matter, python/numpy/torch CPU/CUDA RNG,
our hook state (measurement rows so far, epochs log, MeasurementBuffer contents, step counter,
measurement generator states, persisted CSV/JSON), plus a protocol/identity block (setting, student,
teacher hash, code commit, μ candidates, upstream commit) checked on resume (mismatch → refuse).
Write atomically (tmp + rename), keep latest and previous, load with `weights_only=False`.

- **B1/B2 (RepDistiller `main()`, unchanged upstream):** checkpoint at the end of every 10th epoch and
  after every measured epoch. Resume by prefix replay, like B5: on restart, for epochs before the
  checkpoint the hook substitutes the module-level `train`/`validate` used by the entry point with
  functions returning the saved per-epoch values (no compute), and before the first live epoch restores
  model_s, optimizer, RNG and the relevant `main()` locals (e.g. `best_acc`) via the loop frame (as
  `adapters/lop_hooks.py` does). `adjust_learning_rate` stays upstream (LR is a pure function of epoch).
  Skip the upstream per-epoch `save_freq` files when replaying. The final epoch-240 measurement and the
  final official-test evaluation are part of the checkpointed progress (don't redo if done).
- **B3 (PyCIL `trainer.train`, unchanged upstream):** checkpoint after each task's `after_task()`:
  the learner's full state (`_network` and `_old_network` objects incl. grown fc, `_known_classes`,
  `_total_classes`, `_cur_task`, exemplar memory `_data_memory`/`_targets_memory`, `_class_means`, any
  other attributes iCaRL reads later), the trainer-side curves (cnn/nme accuracy lists), RNG, hook state.
  Resume: for completed tasks skip `incremental_train`/`eval_task`/`after_task` (return saved eval results
  so trainer curves are identical), then restore the learner state before the first live task. Also
  checkpoint every 25 epochs inside a task (model, optimizer, scheduler, epoch, RNG, hook state) and resume
  mid-task by replaying the epoch loop prefix (skip training batches for replayed epochs) — if PyCIL's
  loop makes mid-task replay impossible without patching, keep task granularity and say so.
- **B4:** training resume exists — make the notebook always pass `--resume` when a checkpoint exists;
  `measure` skips epochs already in `measurements.csv` (identity-checked).

## 2. Deadline: always end the session cleanly
Notebook sets `VNA_DEADLINE` = session start + 11.5 h (constant `SESSION_HOURS = 11.5`). Runners check it
at every epoch boundary and before every measurement point: if remaining time < (last epoch time + last
measurement-point time + 10 min margin), write a checkpoint, record status `PAUSED` in summary, and exit
with code 75. The notebook treats 75 as paused (not failure), stops launching new jobs whose turn has not
come, does a final archive, and prints "PAUSED — rerun with this version's output attached". On rerun,
`restore_archives` brings back checkpoints from `/kaggle/input` and each runner resumes automatically;
finished jobs (summary with final status) are skipped.

## 3. Observability (print to stdout, flush, prefixed with lane + job)
- one line per epoch: setting/student, epoch, train loss/acc, eval loss/acc, lr, seconds;
- measurement progress: start/end of each point; inside, one line per (μ, rank, site) with seconds, CG
  iterations and converged flag, and one line per null batch (kind, done/8, seconds);
- a heartbeat thread in the notebook every 10 min: elapsed, time to deadline, per lane current job and
  its last log line, `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader`;
- after the first real measurement point of each job: projected total = training projection + measured
  point time × remaining points; print it (no automatic stop on this; the deadline handles the session).

Rebuild nb2/nb3 (and the pilot notebook if it shares cells). Short section in STAGE_AB_IMPLEMENTATION.md.
Final message: files touched and the printed results of the round-trip checks.
