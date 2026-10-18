# Main CIL runner (phase 3)

`experiments/main_cil.py` reuses pinned PyCIL iCaRL training, the existing
`ICaRLHook` optimizer interception and native epoch/task checkpoints. It does
not modify `methods/vna_v3.py` or `adapters/vna_v3_hooks.py`.

Merge Phase 2 before training: the runner requires its public
`methods.vna_v3.diagnose(model, batch)` function. `icarl_full` uses the engine's
protected `vna_gir` arm through `apply`, `state_dict`, and `load_state_dict`.
Task-local geometric cadence is checkpointed alongside the engine. Damping
state is retained across tasks; task warm starts and cached directions reset.
Real full runs consume the matching backbone authority schedule with `K0`.

```bash
REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
.venv/bin/python -m experiments.main_cil --arm icarl \
  --backbone resnet32_nobn --class-order superclass --seed 0 \
  --out outputs/main/icarl__nobn__superclass__s0

REQUIRE_UPSTREAM=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
.venv/bin/python -m experiments.main_cil --arm icarl_full \
  --backbone resnet32_nobn --class-order superclass --seed 0 \
  --schedule outputs/_schedules/icarl_noBN.json \
  --out outputs/main/icarl_full__nobn__superclass__s0
```

The default input root is `/kaggle/input/datasets/nlnk1607/cifar100`; use
`--data-dir` to override it. Repeat for seeds 1/2, BN (`--backbone resnet32`),
and the noBN random control (`--class-order random`). The random control
reuses the noBN schedule. Resume by repeating the identical command with the
same output directory; deadline pauses propagate exit 75. A completed run is
returned without training again. Existing checkpoints with different setup
or code identity are rejected.

If task-1 loss is NaN on noBN, the summary records
`nobn_lr_halving_required=true`, and the run/status become `DIVERGED`.
Restart every noBN arm with `--nobn-lr-scale 0.5` in fresh output directories;
the flag is part of the strict comparison and resume protocol. It is never
changed independently for one arm during training.

Each run writes summary, task and continual-evaluation metrics, fixed-probe
diagnostics, protocol and status; full arms also write interventions.
Parameter counts are in task rows, protocol and summary. Feature probes run
on all seen training examples at tasks 5/10/15/20, evaluating all seen test
examples. Their frozen-feature, zero-initialized LBFGS recipe is recorded in
protocol.json and PREREGISTRATION_CHANGES.md.

```bash
.venv/bin/python -m scripts.collect_parallel outputs --run-tag main \
  --json-out main_results.json --stability-figure stability.svg
```

The collector rejects synthetic/preflight/incomplete runs and missing v4
artifacts. Tables report mean ± sample standard deviation, seed-paired deltas
against matching iCaRL, seeds won, and GPU-hour ratios. The SVG includes
standard-deviation error bars on both axes. Helpers are also callable from
notebooks as `experiments.main_cil_results.table(records)` and
`stability_plasticity(records, path)`.

Local checks use embedded CIFAR labels, one float64 noBN batch, a tiny
linear-probe fixture and tiny two-task iCaRL training. They call the real
Phase-2 diagnostics and engine, check coefficient-zero bitwise identity and
exact deadline replay, and build a six-measurement authority schedule.

Authority preflight uses the same runner with `--arm icarl_full
--preflight-only --schedule-output outputs/_schedules/icarl_noBN.json`.
BN uses `--backbone resnet32` and its own schedule. Generated main notebooks
perform this before their full jobs; M3 imports M1's schedule when available.
Competitors share this module's training settings, measurements and table.
