# Counterfactual Projection

Look into a larger model without becoming one.

This repository implements the block-local pilot for:

1. asking TINY/Gromo for a temporary rank-r over-expansion beyond a full-width
   ResNet-18;
2. measuring `delta_logits = (f_E(epsilon) - f) / epsilon` without installing
   or training the expansion;
3. solving `(J J^T + mu I) u = delta_logits`, `delta_theta = J^T u` with
   matrix-free JVP/VJP products and conjugate gradient;
4. applying the correction to the original block with unchanged deploy size.

The primary expansion source is the real TINY/Gromo transaction
`candidate.virtual_direction(gate)`. The expanded hidden channels exist only
inside that context. The old convolution-gradient SVD is retained under the
explicit name `GradientLowRankControlProbe`; it is not evidence for the main
hypothesis.

## Quick start

```python
from methods import EProjection

method = EProjection()
step = method.discover_candidate(
    model, counterfactual_tiny_candidate, projection_batch, gate=0.05)
print(step.projection.relative_residual)
print(step.projection.cosine_alignment)
step.projection.apply_(model, scale=0.05)
```

The core invariant is checked on every probe: every parameter and buffer before
the probe must be bitwise identical afterward. Existing gradients and
train/eval modes are also restored.

`StructuralAuxiliarySpace` is currently only a one-dimensional scalar-gate
diagnostic with user-supplied curvature. It is not presented as the full
rank-r auxiliary-space method. Likewise, `ERepOpt` remains a gradient-SVD
control because structural TINY candidates do not yet expose the rank-r
coordinates needed by that optimizer.

## Test

```bash
python3 -m pytest
```

## Kaggle T4 x2 unified three-seed experiment

The current comparison uses three separately generated, protocol-identical
notebooks:

- [`notebooks/kaggle_unified_seed0_end_to_end_t4x2.ipynb`](notebooks/kaggle_unified_seed0_end_to_end_t4x2.ipynb)
- [`notebooks/kaggle_unified_seed1_end_to_end_t4x2.ipynb`](notebooks/kaggle_unified_seed1_end_to_end_t4x2.ipynb)
- [`notebooks/kaggle_unified_seed2_end_to_end_t4x2.ipynb`](notebooks/kaggle_unified_seed2_end_to_end_t4x2.ipynb)

Each starts a randomly initialized CIFAR-ResNet18 at epoch 0. No theta150 or
theta300 input is accepted. Base training uses a metric-independent CIFAR
recipe: SGD for 200 recipe epochs, LR 0.1 with MultiStep drops at epochs 100 and
150 (`gamma=0.1`), momentum 0.9, and weight decay 5e-4. Validation never
controls LR. Every raw validation best is a complete resumable checkpoint.
After the base recipe completes, stall is confirmed only when 100 epochs have
elapsed since the latest strict raw validation best at or after epoch 200.
Theta_P is that raw validation-best checkpoint; any later strict best moves
theta_P and restarts the 100-epoch target. The already observed Vanilla
trajectory then continues for 50 more epochs, yielding a 150-epoch post-fork
baseline without restarting. Theta_P is loaded into recurrent E-driven O,
scaled Bypass 70/30 plus compact train3, and single-intervention O-only; every
completed comparator receives a 150-SGD-epoch post-fork budget. Only E-driven
O rolls back and retriggers after its configured number of epochs without a
new strict validation best. O-only applies one supervised projection at
theta_P and then follows one uninterrupted SGD trajectory.
The 2,000-sample trigger split is diagnostic only.

All configuration, splits, schedules, GPU placement, checkpoint fields, and
metrics are identical across notebooks; only the seed and output directory
differ. GPU0 runs E-driven O only. GPU1 runs Bypass then fresh O-only. Every
method process starts from the same checkpoint hash. Per-epoch histories plus latest, best,
fork, and intervention checkpoints are retained for later plots and resume.
When a prior output is attached under `/kaggle/input`, the notebooks select the
furthest compatible progress checkpoint, pair it with the corresponding raw
validation-best checkpoint, print the minimum remaining epochs if no new best
appears, and resume rather than restarting. If the exact observed-best weights
are unavailable, the notebook selects the closest compatible full checkpoint
at or before that epoch and replays forward with model, optimizer, scheduler,
RNG, and loader state intact. Metrics alone are never used to reconstruct
weights. Every newly observed post-recipe raw validation best is also retained
as an immutable full-state checkpoint.

## Legacy shared-checkpoint runs

The earlier theta300 workflow remains available for reproducing pilot runs:

For the simplest Kaggle workflow, use the single-file
[`notebooks/kaggle_plateau_end_to_end_t4x2.ipynb`](notebooks/kaggle_plateau_end_to_end_t4x2.ipynb).
It accepts theta300 plus CIFAR-100, saves every exact Vanilla trigger-set best,
and waits for 100 epochs without a significant +0.1 pp trigger improvement
before launching the method jobs
from that meaningful-best checkpoint. The following 100-epoch stall window is
the matched Vanilla control. If epoch 500 is
reached without a stall, attach its output and rerun with a larger `MAX_EPOCH`;
both the latest and best Phase-1 states resume.

The same workflow is also available as two explicit notebooks:

1. [`notebooks/kaggle_vanilla_to_plateau.ipynb`](notebooks/kaggle_vanilla_to_plateau.ipynb)
   trains only Vanilla from shared theta300 while preserving the checkpoint LR.
   The held-out pool is split into 2,000 trigger/selection and 3,000 report-only
   evaluation samples. Every exact trigger improvement is retained separately
   in `checkpoint_exact_best.pt` for diagnostics. A +0.1 pp trigger improvement
   updates `checkpoint_best.pt` (theta_P) and resets the stall clock. After 100
   epochs without another significant improvement, the run creates
   `plateau_checkpoint.pt` from theta_P rather than from the later potentially
   degraded model. Epoch 500 remains a review point; latest progress and both
   checkpoint streams are retained for resuming and diagnosis.
2. [`notebooks/kaggle_plateau_fork_t4x2.ipynb`](notebooks/kaggle_plateau_fork_t4x2.ipynb)
   launches supervised O-only, scaled matched-horizon Bypass, and E→O, each for
   a 150-SGD-epoch budget. E→O runs recurrent best-checkpoint trials: if no new
   best appears within its patience, trainable state rolls back to the arm's
   best checkpoint, the stochastic stream remains advanced, and a fresh
   intervention starts. O-only performs one supervised projection at theta_P
   and never rolls back or retriggers.
   The matched Phase-1 window is Vanilla. All methods inherit the same model,
   optimizer, scheduler, momentum,
   RNG, training order, and held-out split. Every trained method saves latest
   and exact-trigger-best checkpoints and reports fork/best/final evaluation
   accuracy, loss, deltas, epoch to best, wall time, peak memory/parameters,
   and expanded time. GPU0 runs E→O only. GPU1 runs Bypass with a
   70-epoch opt1 and at most 30-epoch opt2, then starts O-only. Every method is
   a fresh process from the same hashed theta_best checkpoint. The scaled Bypass
   penalty switches from ×1 to ×2 at opt2 epoch 15; this is explicitly a
   matched-horizon scaling, not an exact reproduction of the native schedule.
   An uncontracted Bypass run
   is retained diagnostically but not presented as a completed comparator.

The architecture-transfer experiment is provided for
[seed 0](notebooks/kaggle_resnet34_seed0_end_to_end_t4x2.ipynb),
[seed 1](notebooks/kaggle_resnet34_seed1_end_to_end_t4x2.ipynb), and
[seed 2](notebooks/kaggle_resnet34_seed2_end_to_end_t4x2.ipynb).
Each notebook trains a full-width Gromo
CIFAR-ResNet34 from random initialization with the same metric-independent
200-epoch base recipe, then requires 150 consecutive epochs without a new
strict raw validation best before accepting theta_P. That exact 150-epoch
trajectory is Vanilla. From the same theta_P hash, GPU0 runs recurrent
E-driven O and GPU1 runs single-intervention O-only for 150 SGD epochs each;
Bypass is not part of this experiment. Only E-driven O retains the raw-best
rollback controller, while E-driven O scans all 16 ResNet34 BasicBlocks.

The same architecture-transfer protocol is also provided for native-Gromo
CIFAR-VGG16-BN at
[seed 0](notebooks/kaggle_vgg16_seed0_end_to_end_t4x2.ipynb),
[seed 1](notebooks/kaggle_vgg16_seed1_end_to_end_t4x2.ipynb), and
[seed 2](notebooks/kaggle_vgg16_seed2_end_to_end_t4x2.ipynb). The VGG adapter
exposes all twelve adjacent-convolution interfaces: eight intra-stage links
use native Gromo TINY, while four interfaces crossing MaxPool use a
bridge-aware closed-form solver. It gathers source activations passed through
the actual MaxPool and destination pre-activation gradients from full-model
backprop, selects source channels by gradient correlation, and fits the
incoming extension with damped least squares; it does not train the auxiliary
branch with Adam. Boundary statistics use summed cross-entropy, matching
native Gromo TINY and making sufficient statistics invariant to minibatch
partitioning. These four sites are operator-aware candidates, not native Gromo
TINY. Their source bases copy selected existing post-activation channels, so
they do not claim to discover novel source features in the R18-TINY sense. All
twelve use the same local functional projection. VGG WHERE scans all twelve
sites using the same observed CE loss gain, selection batch, and gate; per-site
loss gain and `||delta_f_E||` are retained in the intervention history.
DenseNet-121 is evaluated separately with its architecture-native
block-boundary operator.
The VGG16 sensitivity configuration uses controller-anchor patience 15,
line-search scales `{0.025, 0.05, 0.1, 0.2}`, and 64 projection
samples. Reporting tracks strict accuracy best, while recurrent controller
rollback tracks accuracy and then lower loss on an exact accuracy tie. Anchor
checkpoints include model, optimizer, scheduler, RNG, and loader state. Both projected arms
preserve the optimizer, scheduler, and LR inherited from theta_P; no LR is
changed after intervention. O-only still performs one initial projection and
does not roll back or retrigger. Its outputs use a separate `v2` directory and
reject old arm checkpoints with mismatched intervention settings.
When a completed VGG16 seed-1 `plateau_checkpoint.pt` already exists, use
[`notebooks/kaggle_vgg16_seed1_methods_from_plateau_t4x2.ipynb`](notebooks/kaggle_vgg16_seed1_methods_from_plateau_t4x2.ipynb)
to skip Vanilla entirely and launch both projected arms directly from the
verified byte-identical theta_P.

For the exact-original-schedule plateau experiment from shared theta-300,
use [`notebooks/kaggle_plateau_eo_t4x2.ipynb`](notebooks/kaggle_plateau_eo_t4x2.ipynb).
It restores the exact model, optimizer, scheduler, SGD momentum, RNG, data
indices, and loader state for both arms. No LR, `initial_lr`, `T_max`, or
scheduler state is changed. The run stops at the horizon encoded in the source
scheduler rather than stepping beyond it. The original training indices remain
unchanged; the pre-existing 5,000-example held-out validation pool is split
into 2,000 trigger and 3,000 evaluation examples. E selects WHERE from observed
structural expansion loss gain across all eight blocks; projection runs only
for that winner and is applied only when a separate gate batch accepts a
line-search scale. The official test set is not used. A theta300→500 schedule
would be a separate matched extended-training experiment, not this exact
original-training baseline.

To run only the current main algorithm from an existing theta-300 warm-up,
use [`notebooks/kaggle_adaptive_e_driven_o.ipynb`](notebooks/kaggle_adaptive_e_driven_o.ipynb).
It requires only CIFAR-100 plus the matching `shared_seed1_epoch300.pt/json`,
does not run any comparison arm or warm-up, and writes a complete resumable
adaptive-arm checkpoint after every epoch.

Use
[`notebooks/kaggle_counterfactual_projection_t4x2.ipynb`](notebooks/kaggle_counterfactual_projection_t4x2.ipynb).
Enable **T4 x2**, attach CIFAR-100 with a `cifar-100-python` directory, and add
the private-repository token as the Kaggle Secret `github_token`. The notebook
also requires the completed `shared_seed1_epoch300.pt/json` pair as input; it
does not accept theta-150 or silently rebuild the warm-up.

The comparison uses one randomly initialized full-width CIFAR-ResNet18 at
32x32. Vanilla training produces the sole fork checkpoint
`warmup/shared_seed1_epoch300.pt`, containing the model, SGD, cosine scheduler,
epoch, exact train/validation/tuning indices, loader-generator state, and all
RNG states. Its SHA-256 is checked by every arm.

The shared epoch-300 checkpoint retains its full optimizer, RNG, split, and
loader state. At the fork, every arm preserves the checkpoint LR and starts the
same 60-epoch cosine segment that reaches zero at epoch 360. This is explicitly
a post-fork/rebased LR schedule and is not equivalent to training from
initialization with one `CosineAnnealingLR(T_max=360)`.

All four trajectories therefore have the same 360-epoch budget:

```text
theta_300
  +-- vanilla_continue: 60 epochs
  +-- ours_e_driven_o: 60 epochs (WHEN-WHERE-HOW adaptive selection)
  +-- bypass: opt1 40, opt2 <= 20, train3 after early contraction
  +-- o_projection_only: 60 epochs (projection-only supervised control)
```

Wave 1 runs Ours on GPU 0 and Bypass on GPU 1. Wave 2 runs Vanilla on GPU 0 and
O/projection-only on GPU 1. Projection-only uses no virtual expansion: its
functional target is the negative summed-cross-entropy logit gradient
`one_hot(y) - softmax(f(x))`, fitted through the same residual-path projector
and application gate as Ours. It is a supervised projector control, not merely
“Ours minus E”; a win would show that the supervised functional target is
strong, not by itself prove that structural E is useless. Each arm atomically
saves `checkpoint_latest.pt` after
every epoch, including model, optimizer, scheduler, complete history, exact
split indices, RNG, phase, and loader-generator state. Burn-in likewise writes
a full rolling `shared_seed1_progress.pt`; a new Kaggle session can resume by
attaching the previous output as an input dataset. The official CIFAR-100 test
set is never constructed.

At each Ours intervention, `--site auto` evaluates rank-4 TINY proposals at all
eight growing residual blocks using exactly the same statistics batches. Raw
`proposal_score` is logged but does not filter the main search. On a separate
16-sample selection batch, every candidate produces a structural functional
direction and a cheap 25-CG residual-path projection. The main selector maximizes the
projected loss utility `mean((one_hot(y) - p) * J delta_theta)` rather than the
raw TINY score. Full projection runs only when the best finite projected
utility is positive; otherwise that epoch performs normal SGD only.
Projectability rho, residual, and cosine remain solver diagnostics rather than
scientific thresholds. Per-epoch diagnostics preserve all raw scores, expansion
utilities, projectabilities, projected utilities, the WHEN decision, and the
selected site. `--site-selection-mode tiny_score_argmax` retains the old TINY
argmax ablation; `--site-selection-mode fast_topk_projectability` retains the
top-k plus rho-threshold runtime approximation; and a concrete
`--site stages.2.blocks.0` retains fixed-site E-to-O. Full O uses epsilon 0.05,
residual-path scope, 200 CG iterations, and the existing held-out application
gate.

Bypass follows the relaxed residual-network form of Algorithm 1: every ReLU in
the residual stages is embedded as `ReLU(x) + D x` with `D=0`; opt1 trains task
loss in the extended space; opt2 adds `gamma(t) * sum(||D||)` and projects
activations whose contraction norm reaches epsilon; the final projection drops
the remaining D coordinates and train3 continues in the original ResNet. The
arm is a matched-60-epoch-budget relaxed Bypass control, not the full long-opt1
ResNet reproduction from the paper. It uses 40 opt1 epochs and at most 20 opt2
epochs. If contraction succeeds early, projection occurs and train3 consumes
the remainder. If contraction has not succeeded at epoch 360, the expanded
checkpoint is preserved with `bypass_completed=false`; it is never forcibly
projected. Such a run is marked `accuracy_comparison_eligible=false` and is
reported as budget-exhausted diagnostic output, not as a completed Bypass
accuracy comparator. The shared
SGD/cosine optimizer is pilot scaling, not the paper's original long-run Adam
hyperparameter schedule.

The comparison reports both final and best validation accuracy. Primary deltas
are anchored to the validation metrics measured directly at the shared fork:
`validation_accuracy_delta = final_accuracy - accuracy(theta_300)` and
`validation_loss_delta = loss(theta_300) - final_loss`. Best-accuracy and
best-loss deltas are reported separately so transient improvements are not
hidden by the final epoch.

## DenseNet-121 architecture transfer

The DenseNet experiment is isolated from the residual/VGG structural
operators. Its four E sites are the four DenseBlock boundaries. A temporary
rank-r branch reads the complete block representation and contributes only to
the next transition (or to the classifier for the final block). Gate zero is
therefore exactly function preserving. WHERE compares observed loss gain for
all four boundary branches on shared batches; HOW projects only the winning
direction into the selected DenseBlock plus its direct consumer. The temporary
branch is then discarded, so the deployed network remains the original
CIFAR-DenseNet121.

Three matched notebooks are provided at
`notebooks/kaggle_densenet121_seed{0,1,2}_end_to_end_t4x2.ipynb`. They run the
same raw-validation-best/stall protocol as the existing architecture-transfer
experiments and compare Vanilla, O-only, and E-driven O. No DenseNet-specific
code path is selected by the ResNet18, ResNet34, or VGG16 notebooks.

## Reference implementation

The local, ignored checkout at `third_party/One-Shot-TAS-CCIL` points to
`duyh80456-code/One-Shot-TAS-CCIL`, branch `ccil-residual-capacity`. Ideas
reused at the interface level are transactional virtual directions, explicit
RepOpt gradient handlers, and invariant-focused tests. No Gromo source is
copied into this project.

### CPU projection diagnostics (phase A)

Install the `diagnostics` extra, then analyze E-to-O result files:

```bash
python scripts/analyze_projection_diagnostic.py /path/to/run_outputs \
  --patterns '**/result.json' --output /path/to/analysis --bootstrap 2000
```

For JSON history or JSONL console records, supply matching `--patterns`.
The script prints the observed intervention keys and configured field mapping
before extracting metrics. Optional `--config config.json` overrides fields
and missing run metadata; for example:

```json
{
  "mapping": {
    "r_heldout": ["heldout_relative_residual"],
    "cos_heldout": ["heldout_cosine_alignment"],
    "realized_gain": ["actual_loss_improvement"]
  },
  "metadata_overrides": {
    "resnet18_seed0/ours_e_driven_o/result.json": {
      "backbone": "R18", "seed": 0, "method": "ours_e_driven_o"
    }
  }
}
```

Outputs include intervention/site/backbone CSVs, selected-site frequency and
residual/cosine histograms, Spearman estimates with percentile bootstrap
intervals (by backbone and by backbone/seed, for all and applied trials), and
an inventory/mapping JSON. Missing diagnostics remain missing. Residuals refer
only to the selected site; WHERE scores for other sites are not residuals.
`realized_gain` is the immediate gate-batch loss reduction, not accuracy gain
or persistent-growth gain. Current E-to-O logs evaluate `heldout_*` on the gate
batch also used to select scale, so this analysis does not establish independent
held-out performance or capacity need. Correlations are descriptive, especially
with small samples; the event bootstrap does not remove dependence among repeated
interventions in a run. Independent site probes, random-direction controls and
persistent-growth labels require phase B and model/GPU access.

### Independent site and persistent-growth diagnostics (phase B)

These runners require the original `plateau_checkpoint.pt`, CIFAR-100, the
reference repository exposing `dual_growth`, and Gromo. Invoke them from the
repository checkout:

```bash
python -m diagnostics.site_projection_probe plateau_checkpoint.pt \
  --data-root /path/to/cifar --reference-root /path/to/reference \
  --architecture vgg16 --device cuda:0 --output site_probes.json

python -m diagnostics.persistent_growth_probe plateau_checkpoint.pt \
  --data-root /path/to/cifar --reference-root /path/to/reference \
  --architecture vgg16 --device cuda:0 --horizon 20 --output growth_labels

python -m diagnostics.summarize_capacity \
  --projection site_probes.json --growth growth_labels/summary.json \
  --output capacity_analysis
```

Repeat for R18/R34/VGG forks and all seeds. Site projection measures every
proposed site; statistics, WHERE, fit (64), and independent held-out (256)
samples have disjoint indices. The random control fits an independent Gaussian
logit direction, matched to the true direction norm on each batch. Its held-out
noise is independent of fit noise; this baseline measures transfer to unstructured
noise and does not model a structured random expansion.

Growth defaults to top-3 and bottom-2 sites ranked on WHERE, or accepts explicit
`--sites`. Each site's Vanilla and growth arms start from the same fork weights,
optimizer momentum, scheduler, RNG and loader state. The inherited LR is used.
The extension remains registered and active throughout the horizon; its parameters
join the existing SGD group. Output scale uses the candidate's Gromo scaling
convention and is chosen by bounded CE line search (grid followed by local
refinement), not an analytic or unconstrained native Gromo optimum. This diagnostic
retains the transaction for training rather than materializing a compact widened
architecture. Growth checkpoints include active extension tensors and require the
recorded candidate/site/scale to reconstruct that architecture before loading.

`PG_gain` compares best validation accuracy (including epoch zero); loss at that
same best state, minimum loss, final loss and parameter counts are also recorded.
The capacity summary joins by fork hash, backbone, seed and site, reports
Spearman within each seed/horizon, compares true/random residuals, and aggregates
seed estimates without pooling sites across seeds. A positive correlation would
be evidence for the capacity hypothesis under this short-horizon diagnostic;
absence of correlation leaves residual as a projection diagnostic only.

For Kaggle Run All, import
[`notebooks/kaggle_projection_capacity_diagnostics_t4x2.ipynb`](notebooks/kaggle_projection_capacity_diagnostics_t4x2.ipynb).
Attach CIFAR-100, original Phase-1 fork outputs and optional E-to-O JSON outputs.
Enable Internet, GPU and the `github_token` Secret used by the existing notebooks.
The default run filter is VGG16 seed 1 and ResNet18/ResNet34 seeds 1, 2, 3.
It runs whichever matching forks are attached, prints their inventory, and saves
an archive containing phase A/B results and within/across-seed summaries.
Discovery processes archived checkpoints one at a time, discards unwanted or
duplicate states, and reserves 256 MiB of free disk before materializing each
checkpoint. Its checkpoint cache is outside the results folder and is excluded
from the downloadable results archive. After updating this notebook, start a
fresh Kaggle session to avoid retaining caches produced by older versions.
For another session, reattach the full diagnostic notebook output: exact matching
fork/config/code manifests allow completed runs and sites to be reused. Incomplete
site trials restart from theta_P. Seven forks at the defaults cost about 1,400
SGD epochs and may exceed a single Kaggle session.

### CPU-only follow-up on existing logs

For Kaggle Run All, import
[`notebooks/kaggle_projection_log_analysis_cpu.ipynb`](notebooks/kaggle_projection_log_analysis_cpu.ipynb).
Select Accelerator **None**, enable Internet and `github_token`, and attach
previous E-to-O outputs exposing JSON/history files. No checkpoint or CIFAR
input is needed. Edit its CONFIG cell to label unknown runs and rerun the
analysis cells. Logs stored only inside archives must first be exposed as JSON
input files. This notebook runs phase A only.

No model, checkpoint, CIFAR data or GPU is needed for this command:

```bash
python scripts/analyze_projection_diagnostic.py /kaggle/input \
  --patterns '**/result.json' '**/history.json' '**/history.jsonl' \
  --config configs/projection_log_analysis.example.json \
  --horizons 1 5 15 --output /kaggle/working/projection_log_analysis
```

Edit the example configuration's directory patterns to match attached runs.
Directory rules fill missing/unknown labels; JSON run/config metadata wins over
rules, and exact `metadata_overrides` (paths relative to the input root) wins
over both. Set `run_id` explicitly when JSON/history for one run live in separate
directories. Otherwise the containing directory identifies the run. A shared
fork hash does not merge distinct runs. Unresolved labels appear in
`unknown_metadata.csv`.

The script prints the available keys, configured mapping and resolved sources.
`metric_sources.csv` separates functional fit, held-out linear projection and
actual update metrics. Fit metrics can come from the **selected** CG attempt's
functional residual/cosine; CG solver residual is never treated as functional r.
`cosine_by_site_boundary.csv` reports counts and Q25/median/Q75 for each run/site
and for VGG boundary/non-boundary groups (`boundary_to_N` site names).
`run_spearman.csv` reports per-run correlations for fit/held-out/actual cosine
against immediate `realized_gain`, exact accuracy at +1/+5/+15 epochs, and its
change from the logged intervention-epoch accuracy. Each row includes paired
quartiles, sample count and a bootstrap interval when estimable.

Missing epochs are not interpolated; conflicting history entries are rejected.
Windows with another logged intervention are excluded from horizon correlations
and marked in `interventions.csv`. Accuracy units remain those of the input;
the baseline's timing may be before or after the update. These observations are
descriptive and are not a matched Vanilla comparison or persistent-growth label.
`spearman.csv` retains the earlier pooled descriptive analysis; use
`run_spearman.csv` for the follow-up within each run.

## DeiT-Tiny-CIFAR: first one-shot architecture experiment

Import [`notebooks/kaggle_deit_tiny_seed1_one_shot.ipynb`](notebooks/kaggle_deit_tiny_seed1_one_shot.ipynb)
on Kaggle. Attach **CIFAR-100**, enable Internet, a CUDA GPU and `github_token`,
then Run All. No pretrained model or CNN plateau checkpoint is used. Optional
prior expanded DeiT output files allow epoch resume; different matching forks
are rejected rather than selected by filename. This notebook uses one GPU
sequentially and may require multiple sessions for the baseline.

The local backbone keeps [official DeiT-Tiny geometry](https://github.com/facebookresearch/deit/blob/main/models.py):
embed192, depth12, heads3, MLP ratio4, QKV biases and LayerNorm eps1e-6. The CIFAR
adaptation changes image size to32 and patch size to4: 64 patch tokens plus CLS,
no distillation/pretraining/Dropout/DropPath. Attention uses explicit matmul and
softmax so the existing torch.func JVP/VJP projector does not require fused
attention forward derivatives. This is a random-init CIFAR adaptation, not the
full official ImageNet training recipe.

- **E:** exactly twelve homogeneous sites, `blocks.0.mlp` through `blocks.11.mlp`.
  Temporary Gromo LinearGrowingModule wrappers collect real GELU MLP activations
  and downstream gradients and invoke the existing native covariance/TINY
  solve. Its sufficient statistics flatten tokens and normalize by image count,
  following pinned Gromo's Linear implementation. No random branch training,
  attention expansion, boundary expansion or auxiliary optimizer is used.
  Requested rank8 is fixed; Gromo's thresholds can reduce effective rank, which
  is logged separately. At gate0 the candidate returns the unchanged base output.
- **WHERE/WHAT:** reuse the CNN raw mean E-gain selector, with three shared
  32-image WHERE batches and epsilon=.05. Delta norms and gain/norm are
  diagnostics only. Statistics256, projection64, gate32 are disjoint deterministic
  subsets of the training pool; their actual indices are stored in the record.
- **HOW:** reuse FunctionalProjector's dual solve with CG200, damping retries
  `[.001,.01,.1,1,10]`, and scales `[.025,.05,.1,.2]`. Only the selected original
  MLP's fc1/fc2 weight/bias enter the projection. The native TINY optimal existing
  weight update is used only in proposal statistics, never committed directly.
  Only tensors actually changed after floating-point rounding have Adam moments zeroed, including
  AMSGrad's max moment if present; Adam step counters, other state, LR and the
  scheduler are retained. `adam_moments_reset_parameters` lists their names;
  `momentum_states_reset` is the integer count. `gate_*` and `actual_gate_*`
  metrics describe the scale-selection batch, with
  `evaluation_role=gate_batch_used_for_scale_selection`. No independent held-out
  diagnostic batch is used. `actual_loss_improvement` is gate loss reduction
  after selecting scale on that same batch, so it may be optimistic. Use the
  validation immediately after projection and at epochs 1–5 to
  assess generalization; do not interpret gate metrics as that evidence.
- **Fork arms:** Vanilla, fixed-last-MLP supervised O-only (`one_hot-softmax`)
  and one-shot E-to-O all start from the same SHA-verified historical validation-best theta_P.
  Vanilla reuses the uninterrupted K-epoch plateau-confirmation window already
  observed in Phase 1, including its original metrics and full terminal state.
  Only O-only and E-to-O restore model/optimizer/scheduler/RNG/loader state, apply
  one correction and train new AdamW arms for K epochs. O-only has no rollback;
  E-to-O uses the rollback controller below. Neither arm retriggers E. Epoch0 after-correction checkpoints ensure resume does not reapply
  the intervention; all candidate tensors/hooks disappear before saving.

The configurable **initial baseline recipe** is AdamW LR5e-4, WD.05, betas
(.9,.999), with no decay on biases/normalization/CLS/position embeddings; 5 warmup
epochs then a fixed global cosine through epoch400 with minimum LR ratio.01.
Batch128, seed1, `stall_patience=150` with no minimum-epoch gate, max baseline epoch300,
`post_fork_epochs=150`. The CIFAR split remains 5000 held out (2000 reserved and unused, 3000
validation for historical-best checkpoint and plateau selection) plus128 tuning
excluded from training. Accuracy ties, even with lower loss, do not move theta_P.
After plateau detection, the full historical-best checkpoint is loaded to export
theta_P; the detection-epoch checkpoint is never the fork. All arms use the same
SHA256-verified starting point and compare the same fixed K-epoch window. Validation never selects site, scale
or damping. E-to-O does use validation for its rollback decisions; both arms
still run the full K epochs without early stopping.
`report_best_accuracy/loss/epoch` describe epochs 1 through K of each arm; epoch0
validation immediately after projection is reported separately. `scientific_escape`
is strict post-fork best accuracy > `historical_best_accuracy`, and
`delta_vs_historical_best` may be negative. DeiT protocol v5 rejects older forks. No official test set
is loaded. These defaults are declared experiment choices, not tuned results;
no plateau within the cap produces a status report and no fork. Phase 1 stall
and Phase 2 horizon are distinct: best at epoch23 and 150 epochs without a
strict improvement confirm plateau at epoch173. Vanilla reuses epochs24–173;
O-only and E-to-O reload epoch23 and each train 150 new epochs. Phase 1 writes
`vanilla_reference.pt` with the terminal model/optimizer/scheduler/RNG/loader state
and a hash binding it to theta_P. Its CPU export writes Vanilla `result.json` and
`checkpoint_latest.pt` with `trajectory_source=phase1_plateau_window` and
`additional_training_epochs=0`. Vanilla escape is false by construction for this
plateau window. A missing reference or mismatched fork/window raises an error;
Vanilla is never retrained as a fallback. Reuse requires the post-fork horizon to
equal stall patience. `algorithm_patience=10` applies only to E-to-O: the anchor starts at theta_P
before projection and updates on higher accuracy or an exact accuracy tie with
lower loss. The accuracy stall resets only on strictly higher accuracy; a
lower-loss tie updates the full anchor but still counts as one epoch without
accuracy improvement. After 10 such epochs, restore model, AdamW and scheduler
from the anchor, reset the accuracy stall to zero, and continue with the current
RNG and loader stream. No new E query/projection occurs. Scheduler/optimizer state
(including LR and Adam step) return to the anchor; the elapsed K-epoch horizon
is never rewound. `checkpoint_best.pt` stores the E anchor; latest checkpoints
also embed it and the counter/events for standalone resume. Scientific reporting
remains strict accuracy only. History logs validation before rollback separately
from the actual post-controller state metrics used for final-checkpoint reporting.
The E controller configuration is part of its resume identity, so old E arms
without rollback cannot silently resume this algorithm. Phase 1, O-only and the
observed Vanilla reference remain compatible with the same protocol-v5 forks. `--horizon` remains an
alias for `--post-fork-epochs`.

Standalone commands after installing pinned Gromo (Python >=3.10):

```bash
python -m experiments.train_deit_plateau --data-root /path/to/cifar \
  --output runs/deit/vanilla --seed 1
python -m experiments.run_deit_fork --data-root /path/to/cifar \
  --plateau-checkpoint runs/deit/vanilla/plateau_checkpoint.pt \
  --plateau-checkpoint-hash YOUR_SHA256 --method ours_e_driven_o \
  --output runs/deit/ours_e_driven_o --post-fork-epochs 150
```

Run the fork command for `o_projection_only` with the same checkpoint/hash/horizon.
For `vanilla_continue`, use the same fork/hash and
`--vanilla-reference runs/deit/vanilla/vanilla_reference.pt` (the default is beside the fork). This
exports the observed Vanilla arm on CPU; no CIFAR input or GPU is needed. `--resume` accepts each arm's `checkpoint_latest.pt`;
recipe, backbone, protocol, fork, method and CP configuration must match. Baseline
resume additionally needs its matching `checkpoint_best.pt` in the output folder.
When reattaching a completed Phase 1 fork on Kaggle, include `vanilla_reference.pt`
or the full Phase 1 `checkpoint_latest.pt`. The latter regenerates the reference
on CPU after verifying the prefix, contiguous plateau window, split, recipe and
historical best. A fork alone is insufficient to recover the terminal state.
Matching protocol-v5 Phase 1 checkpoints can be reused without retraining. Protocol-v4 checkpoints are rejected by the updated runner.

Logs include validation before and immediately after correction, the first five
ordinary AdamW epochs, all site gains/norms, WHERE winners/stability, projection
fit/gate/actual metrics, CG attempts/damping and Adam reset tensor names. Strict
report accuracy and loss refer to the same observed state. The CPU log analyzer
recognizes the new `DeiT` backbone label.

The required small-model native TINY tests cover function preservation, equivalence
to explicit MLP widening, finite-difference stability, exact site scope, no
auxiliary persistence, selected-site/batch replay, Adam moments, and bitwise
main-loop resume. The Kaggle notebook gates training on them. They do not
establish full-width CUDA runtime, convergence or historical-best escape;
those require the actual CIFAR experiment.

### DeiT algorithm ablations A0–A6 (protocol v5)

Run [`notebooks/kaggle_deit_tiny_seed1_all_arms.ipynb`](notebooks/kaggle_deit_tiny_seed1_all_arms.ipynb)
on Kaggle with CIFAR-100, Internet, CUDA, and the `github_token` Secret. No fork
checkpoint is needed for a fresh run. Optional expanded outputs from the exact
same v5 recipe support Phase 1 and per-arm resume; attach the raw reference as
well as arm checkpoints. Old v4 forks are rejected. This notebook is separate
from the three-arm one-shot notebook.

The shared experimental budgets match the current ResNet18 unified notebook:
batch64, requested rank4, projection32, WHERE3x32, statistics256, gate32, CG200,
scales `{.0125,.025,.05}`, algorithm patience10, and post-fork horizon150.
DeiT retains AdamW (explicit LR5e-4, WD.05 with bias/norm/token/position excluded),
five warmup epochs and a 300-epoch cosine recipe, then LR floor5e-6. This is a
starting CIFAR recipe, not a tuned ImageNet recipe or an accuracy guarantee.
The default baseline cap is epoch800 and can be increased with a new declared
recipe. No intervention runs if a complete plateau/control window is unavailable.

Plateau tracking starts at epoch0; there is no minimum epoch or requirement to
finish the 300-epoch cosine schedule. Each strict validation accuracy best saves
the historical-best checkpoint and resets stall. After150 consecutive epochs
without beating that best, reload it as theta_P and reuse the already-observed
150 Vanilla epochs as A0. No extra50-epoch continuation and no Vanilla retraining
are needed. For example, best epoch23 with no improvement through173 forks from23,
not173. Each intervention arm trains150 new epochs from that same checkpoint.
The cosine schedule continues at its inherited position, with no restart.
This plateau window length/no-recipe gate differs from the current Res18 unified
notebook; the projection budgets and DeiT-specific optimizer setup remain as
listed above. The 3000 evaluation samples select plateau/rollback and report
results; the 2000 reserved samples and official test do not select checkpoints.

| Arm | Intervention |
| --- | --- |
| A0 `vanilla_continue` | Reuse the observed150-epoch Vanilla continuation |
| A1 `o_projection_only` | Supervised projection at the fixed last MLP |
| A2 `e_driven_o_raw` | Native TINY E direction; raw mean-gain WHERE |
| A3 `e_driven_o_normalized` | Mean gain / (mean delta-logit norm +1e-8) WHERE |
| A4a `random_control_parameter` | Gaussian deltas, matching each of A2's four tensor norms and A2's applied scale |
| A4b `random_control_logit` | Gaussian logit target matching A2's norm; same projector, its own gate line search |
| A5 `persistent_growth` | Commit the same native width proposal; gate gamma grid `{0,.025,.05,.1,.2}` |
| A6 `opt_e` | Fixed native A/a; B-only mean-CE Gauss–Newton with new disjoint opt_fit/opt_val |

A4/A5/A6 use A2's raw-selected site. A3 selects its own site, and A1 remains
fixed-site: A2 vs A1 therefore changes both site and direction. Raw and
normalized WHERE scores/ranks are recorded for every method arm, and A0 carries the same table explicitly as offline-only diagnostics. A4a skips if A2 has no
applied scale. A4b has its own scale search: it tests the projector+gate pipeline,
not fixed-step direction quality. It has no meaningful independent held-out
random field; its gate random-target alignment is explicitly a null diagnostic.

A2/A3/A4/A5/successful-A6 have the same anchor rollback: a higher accuracy or an
exact accuracy tie with lower loss updates the full anchor and resets stall.
Ten failures restore model/AdamW/scheduler and keep the consumed RNG/data stream.
There is one intervention only, no recurrent E query; this ablation differs from
Res18's recurrent E arm. A0/A1 have no rollback; a failed A6 trains like Vanilla
without rollback. Comparisons to A0/A1 include the controller effect.

A5 is a capacity comparator, **not a guaranteed upper bound**. It concatenates
native A/a and gamma*B inside the MLP. Parameter growth is `rank*(2*dim+1)`:
1540 for effective rank4 at dim192, or3080 for rank8. Effective rank is logged.
Adam moments in old slices and the tensor step counter are preserved, new
moment slices start at zero, optimizer parameter-group references are replaced,
and the scheduler is unchanged. Resume reconstructs the widened shape before
loading model and optimizer. The fallback rollback anchor has the widened
zero-output geometry, not the original narrower tensor shapes.

A6 adds disjoint64-sample opt_fit/opt_val partitions **after** all old probe
partitions, preserving their indices. It starts B=0, holds A/a fixed, and uses
matrix-free JVP/VJP softmax Gauss–Newton with at most5 inner steps. Damping
multipliers `{.001,.01,.1,1,10}` multiply a seeded Hutchinson mean-diagonal
estimate (floor1e-8); CG uses at most80 iterations, tolerance1e-4; backtracking
uses `{1,.5,.25}` and opt_val improvement tolerance1e-6. The projection grid is
`{.1,.25,.5,1}`, deliberately different from A2, and is part of resume identity.
Predicted gain uses the undamped mean-CE quadratic model at the accepted step.
No accepted step means `opt_e_failed`, no jump and no rollback. opt_val is an
internal selection set, not an unbiased held-out diagnostic. The gate batch
also selects scales, so its loss gains do not demonstrate generalization.

The runner writes `suite_protocol.json`, `raw_intervention_reference.pt`,
`arm_status.json`, `summary.json`, and full per-arm latest/anchor checkpoints,
history, immediate validation, and epochs1–5 validation. Arms run in separate
sequential processes; Opt-E is last. Failure of one arm does not skip later
arms, and the notebook archives resumable state even on an error.

```bash
python -m experiments.run_deit_all_arms \
  --data-root /path/to/cifar-parent --output /path/to/output --seed 1
```

Tests in `tests/test_deit_ablation.py` and
`tests/test_deit_ablation_integration.py` cover normalized selection, disjoint
partitions, random norms/scope, zero-gate equivalence, moment migration, widened
resume, GN symmetry/PSD/toy descent/no auxiliary residue, epoch302 lower-loss
anchor rollback, all seven method main loops and bitwise checkpoint resume,
recipe-complete plateau, and a longer reused Vanilla window. These CPU tests
use small models and native Gromo; full-width CUDA cost/accuracy is not measured.

DeiT console logging follows the tagged JSON style of ResNet18. Phase 1 prints
`deit_vanilla` with train/validation loss and accuracy, LR used and next LR,
paired report-best accuracy/loss/epoch, strict improvement, epochs since best,
patience and plateau status. Method rows are tagged with their arm name and
include report/anchor metrics, separate stalls, rollback flag/count, intervention
count, delta versus historical best and scientific escape. Disabled controllers
use `null` anchor/stall fields rather than invented metrics. Separate events
announce run/arm start, intervention diagnostics (WHERE/site/scale, projection,
Adam resets, persistent growth or Opt-E), rollback target, plateau confirmation
and completion/failure. No trigger accuracy is computed or introduced.
Every output folder saves the same console events to `console.jsonl`. Epoch wall
time and peak allocated GPU bytes are console observations; they do not enter
training/controller state or deterministic history. CPU tests report GPU bytes0.
`learning_rates` denotes the LR actually used to train that epoch;
`next_learning_rates` reflects the scheduler and any subsequent rollback.
