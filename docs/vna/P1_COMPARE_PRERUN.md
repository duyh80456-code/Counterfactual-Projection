# P1 full versus baseline: four configurations

## Baseline protocol shape repair (2026-10-26)

The native producer persists `protocol.json` as identity plus observed
`parameter_count` and `mu0`, while `summary.identity` contains identity only.
Baseline validation now compares exactly that producer contract: all identity
fields must match, observed parameter count must equal the last task's positive
integer model size, and baseline `mu0` must be null. No other extra/missing field
is accepted. Historical baseline files are never edited. 37 focused tests pass,
including tampered identity/model-size/mu0 rejection, and an NB1 CPU fake-data
smoke confirms the actual producer's shape for both baseline and full. Reports:
`p1_compare_logs/pytest_nb1_baseline_protocol.xml` and
`p1_compare_logs/pytest_nb1_protocol_producer.xml`. No GPU run was launched.

## Baseline restore repair (2026-10-26)

Kaggle reported both expected NB1 baseline outer summary paths missing. This
alone does not establish whether the original NB1 Output is attached. The
shared restore now follows mount directory symlinks, prunes .git and detects
cycles while retaining conflict/exclusion rules. A completed native
`run/summary.json` may authorize reuse when only its outer copy is absent;
protocol, dataset, config, measurements, tasks and source checks still apply.
No baseline file is created or rewritten to simulate completion. Error messages
distinguish missing summaries, incomplete runs and unregistered source digests.
The timing reader uses the same native summary fallback. The scientific policy
and allowed historical baseline digests are unchanged. 56 affected CPU tests
passed, including symlink/archive restore, exclusions and immutable baseline
validation. Original NB1 Output containing both baselines is still required.

## NB1 post-log amendment (2026-10-26)

The new `icarl_full` policy is `schur_t32_sharpness_trust_v3`; the amendment
and proposal/candidate rules are registered in PREREGISTRATION_CHANGES.md.
Skipped VNA (`no_positive_directions`, `chol_fail`, `negative_certificate`)
does not prevent independent T32 dead-slot trials. Actual CE and KD must both
be finite and non-increasing. Accepted swaps still get post-intervention
follow-up logging even when the VNA part of the event remains marked skipped.
The VNA sharpness gate runs before GIR, with threshold
`max(lambda_before, 2/max_optimizer_lr)` and bitwise rejection rollback.
Log the candidate curvature separately from final post-GIR curvature. The
trust-region calculation materializes the actual parameter direction and
checks its scaled norm against Delta, including very large unconstrained scales.

NB1 now launches only `icarl_full` s0/s1 in
`outputs/p1_compare_nb1_logfix_v3/`, with a fresh locked preflight. It restores
only original baseline folders from the attached old NB1 Output into
`outputs/p1_compare_nb1/`, validates both completed results before launching
either lane, and writes separate `baseline_reuse__s{seed}.json` manifests.
Original baseline files/digests remain unchanged. Old full models, failed
runs, old preflight and old schedules are not imported into the new namespace.
Attach CIFAR-100, the matching new source dataset, and the original NB1 Output;
use the newly generated NB1 notebook. Other notebooks are not submitted here.

CPU verification: the affected suite has 118 tests, 116 passed and two CUDA
tests skipped; its final report is
`p1_compare_logs/pytest_nb1_revision_final.xml`. The additional focused report
`p1_compare_logs/pytest_nb1_revision.xml` covers T32 on BN/noBN, actual KD
rejection, sharpness/rollback, large scales, exact resume and zero coefficient,
baseline validation and preservation, and post-swap follow-up. These tests use
synthetic data and do not establish real CIFAR-100 accuracy or T4 performance.

GPU rerun is not submitted: no Kaggle CLI/credentials are configured locally,
and the Chrome computer-use tool reports that permissions are not granted.
No baseline training or GPU computation has been performed by this revision.

## T4 failure review (2026-10-26)

The supplied NB4 log fails with CUDA OOM during Gram VJP at task 2, not
numerical divergence. Gram now retries the same columns with derivative vmap
width halved after CUDA OOM, down to one; a failure at width one still raises.
Loss, samples, output Gram shape, solver and intervention schedule are unchanged.
The successful widths remain in the existing Gram timing record. This reduces
derivative activation memory but does not guarantee that the dense Gram fits T4.

Snapshot hard-linking retries a transient missing source up to three times;
persistent missing files still fail and retain the prior archive. No file is
silently omitted. Focused CPU verification: 26 tests passed in
`tests/test_vna_exact.py` and `tests/test_vna_archive_race.py`, including simulated
CUDA OOM with dense Gram equality and a simulated atomic replacement race.

Actual NB4 calibration in the supplied log estimates 22.16–79.60 hours per
lane (3–8 sessions of 11 hours). This supersedes illustrative estimates below
for NB4; other notebooks still need their own actual calibration. No T4
validation of the memory retry has been performed here. A rebuilt overlay has
a new source digest: existing outputs must retain their old provenance; the
strict resume identity checks have not been relaxed or migrated.

Registered in PREREGISTRATION_CHANGES.md before the CPU runs. No GPU jobs have
been launched here. Source is supplied as a SHA-256 checked ZIP overlay on
the current immutable commit because AGENTS.md prohibits commits.

| Notebook | Backbone | Increment | Tasks | Seeds | LR |
|---|---|---:|---:|---|---|
| NB1 | native ResNet32 noBN/SkipInit | 20 | 5 | 0, 1 | 0.05 |
| NB2 | native ResNet32 noBN/SkipInit | 10 | 10 | 0, 1 | 0.05 |
| NB3 | native ResNet32 BN | 20 | 5 | 0, 1 | read C-Flat BN protocol |
| NB4 | native ResNet32 BN | 10 | 10 | 0, 1 | same source as NB3 |

Every lane uses its own seed for baseline and full; the pair's seed-0 authority
preflight is serialized under thread and file locks. K0 is printed and stored
under a backbone/increment-specific path. Native epoch resume and exit 75
remain enabled. A FAILED_DIVERGED result is terminal and its lane proceeds;
there is no clipping, seed substitution, accuracy gate, or best-epoch selection.

The first/later epoch counts are 100/50, milestones [30,60,85]/[24,35], memory
2000, no warmup. Each task consists of 4 or 2 complete adjacent superclasses
in that seed's permutation. All 100 fine classes occur exactly once.
The engine is unchanged: Gram/power10, schur_gamma_bound_rho25_v2, schur_dead.

## Measurements and outputs

`experiments.p1_compare` wraps the existing main_cil entry point. Per-task
CNN/NME total/old/new accuracies and incremental means retain the shared
runner's implementation. Feature accuracy is measured at the final task.
`run/` contains the native protocol, epoch/task logs, diagnostics, raw
interventions, safe checkpoints and divergence coordinates. Additional files:

- `measurement_protocol.json`: immutable measurement identity and cadence;
- `stability_gap/measurements.json`: Tracker from task 1 through the last task,
  r=0..10, every 5 updates to 200, then epoch starts and task end;
- `ggn_power10.json`: the exact current training minibatch used by VNA,
  normally 128 examples drawn by the native shuffled new-data-plus-exemplar
  loader; no separate subset or quota. Loss, ten power iterations, dtype and
  purpose seed reuse the engine's `_stability` implementation. r=0 precedes
  SGD, r>0 follows SGD and precedes intervention, so a coincident engine event
  has the same `lambda_max_ggn_before`. Each row logs global iteration, seed,
  actual sample count and old-example count. Every row asserts bitwise equality of
  the real model, buffers, gradients, optimizer, modes, RNG and intervention
  state before/after the copy measurement;
- `intervention_log.json`: full raw events plus accepted, h and swap fields;
- `summary.json`: native completion/divergence status and final metrics.

Task-1 old-class accuracy is null because no old classes exist. Points beyond
a task's actual update count remain missing. Measurements never add a training
step. Tracker and GGN logs survive a deadline resume and suppress duplicates.

Inc20 exposed PyCIL's fixed 10-class reporting bins: its final accuracy matrix
would have 10 bins but only 5 task columns. The shared runtime now requests
20-class bins for Inc20 evaluation; training and prediction-level scientific
metrics are unchanged. Fake memory scales with increment to avoid zero
exemplars; the real memory budget stays 2000.

## CPU verification

70 affected tests passed in 83.86 seconds, including both arms on all four
configurations with `--fake-data`, superclass coverage, T29e zero slot Jacobian
and function preservation on BN/noBN, copy-GGN bitwise isolation, and exact
model/optimizer/RNG resume. Log: `p1_compare_logs/pytest_affected.xml`.
Fake runs use a two-channel residual fixture, two one-epoch tasks, memory
max(20, 4*increment), and four-example GGN/intervention windows. They establish
wiring and invariants; real runs use native ResNet32, all tasks and window 128.
The BN protocol used in CPU tests is explicitly a fixture, not the historical
C-Flat BN evidence. No full test suite was requested or run.

Review checks: 19 tests passed in 112.67 seconds (`p1_compare_logs/pytest_review.xml`).
They include equality against an actual engine application with CE+KD on BN
and noBN, missing BN evidence, class-scaled cost/session arithmetic, real-run
digest requirements and source_digest on every generated protocol. Earlier
70-test results above describe the initial implementation.
The four-configuration smoke was additionally rerun with a direct comparison
between coincident `ggn_power10.json` rows and actual `interventions.json`
events: 4 passed in 106.99 seconds (`p1_compare_logs/pytest_engine_parity.xml`).

## Timing and the 11-hour boundary

The old 3x-baseline proxy and the 2.75/2.95-hour figures are withdrawn. They
did not account for increasing output-space system size. No real T4 timings
or K0 exist yet for these four pairs, so their measured duration/session count
is currently **PENDING** for both seed lanes.

Before launching each full arm, `time_estimate.json` records separate seed-0
and seed-1 baseline/preflight/full components, a row for every intervention
task, and `sessions_11h_range`. With C classes, updates per later task are
estimated as ceil((500*increment+2000)/128)*50. Applications occur at
1, 1+K0, 1+K0+2*K0, ... while inside that task; extra budget-guard doubling
can reduce the count. For each measured preflight point at C0 classes,
application cost includes `vna_seconds + stability_seconds` and is scaled
by (C/C0)^2 and (C/C0)^3. The larger measured point estimate is used per task.
These are scenarios for Gram construction/factorization, not proven bounds.

Total per lane = measured instrumented baseline + shared preflight reservation
+ baseline-sized full training/measurement cost + sum of per-task intervention
costs. The measured baseline already includes Tracker, GGN and final metrics;
those are not omitted or added twice. Preflight is computed once; charging it
to both lanes reserves compute/wait time. Full measurement cost can differ
from baseline, and this assumption is explicit. Notebook duration is the
maximum lane total, and sessions = ceil(duration / 11 hours).

For a concrete **hypothetical planning scenario only**, assume K0=512, each
of the three task-2 preflight points takes 60 seconds plus 5 seconds stability,
shared preflight takes 5 minutes, and baseline (including all measurements)
takes 1 hour in every lane. No number below is a measured T4 result:

| Notebook | Lane / seed | Baseline h | Preflight h | Full h range | Total h range | 11h sessions |
|---|---|---:|---:|---:|---:|---|
| NB1 noBN Inc20 | GPU0/s0 | 1 | .083 | 1.98–3.02 | 3.06–4.11 | 1 |
| NB1 noBN Inc20 | GPU1/s1 | 1 | .083 | 1.98–3.02 | 3.06–4.11 | 1 |
| NB2 noBN Inc10 | GPU0/s0 | 1 | .083 | 6.20–21.48 | 7.28–22.56 | 1–3 |
| NB2 noBN Inc10 | GPU1/s1 | 1 | .083 | 6.20–21.48 | 7.28–22.56 | 1–3 |
| NB3 BN Inc20 | GPU0/s0 | 1 | .083 | 1.98–3.02 | 3.06–4.11 | 1 |
| NB3 BN Inc20 | GPU1/s1 | 1 | .083 | 1.98–3.02 | 3.06–4.11 | 1 |
| NB4 BN Inc10 | GPU0/s0 | 1 | .083 | 6.20–21.48 | 7.28–22.56 | 1–3 |
| NB4 BN Inc10 | GPU1/s1 | 1 | .083 | 6.20–21.48 | 7.28–22.56 | 1–3 |

BN and noBN appear equal only because the scenario inputs are deliberately
equal; they require independent real measurements. **Inc10 can need three
sessions even in this illustrative scenario.** Each notebook warns above 11h,
retains all tasks and resumes; it does not silently shorten the experiment.
With missing baseline/preflight evidence the JSON reports PENDING and no
numeric session claim. Real per-lane estimates replace these examples after
baseline and preflight, before the corresponding full arm begins.

The actual C-Flat BN protocol.json is absent locally, so a numeric BN LR and
verified NB3/NB4 timing cannot be asserted. Both notebooks reject absent or
conflicting LR evidence, print the source path, SHA-256 and both LRs before
launch, and propagate those values into run and K0 authority identities.
There is no default fallback. All new `protocol.json`/measurement protocol
files carry the ZIP SHA-256 as `source_digest`, inherited by child processes
from the verified overlay. A real run without that digest raises before
creating artifacts. Historical imported C-Flat evidence retains its own identity.

## Generate and run

```sh
.venv/bin/python -m scripts.build_kaggle_p1_compare \
  --commit ccf60af0909afa8853357c660f8760c98cc1b828 \
  --source-bundle /private/tmp/p1_compare/vna_v3_source.zip
```

Use `notebooks/vna5_p1_compare_NB1.ipynb` through `NB4.ipynb`. Attach the
generated source ZIP, CIFAR-100 and that notebook's prior output archive for
resume; NB3/NB4 also need the same historical C-Flat BN protocol.json.
The notebook checks the source digest before importing the runner. Local CPU
checks are complete; GPU results and real timing/K0 remain to be measured.

### Kaggle automatic ZIP extraction

Kaggle may expose the uploaded ZIP as folders instead of preserving the ZIP
file. The rebuilt NB1–NB4 accept either layout. The raw ZIP path verifies its
SHA-256; the extracted path verifies all 117 member hashes embedded in the
notebook against one source root, then copies only those verified files.
Missing or modified files stop the bootstrap. `source_manifest.json` records
the verification mode and member hashes; the original ZIP `source_digest`
continues into every run protocol.

The existing ZIP is unchanged (SHA-256
`cb902d3e3a4f475672398d24f228fd9307f8e8d9098642ea4b8fb89473bfd9f8`).
Replace the notebook with its rebuilt NB1/NB2/NB3/NB4 counterpart and rerun
the first code cell; the already attached matching extracted dataset can stay.
Five bootstrap/notebook tests passed in 3.50 seconds, covering ZIP, extracted,
modified and missing-member inputs (`p1_compare_logs/pytest_source_overlay.xml`).
