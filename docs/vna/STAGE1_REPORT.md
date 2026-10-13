# Stage 1 report — VNA core and Phase 0 diagnostics

Date: 2026-10-13. Stage 1 implementation only; controller, training arms and notebooks are deferred by TASK_STAGE1. No existing runner or test was changed. No push was performed.

## Environment and validation

- Reused `.venv`: Python 3.12.7, CPU torch 2.2.2, torchvision 0.17.2; pytest and numpy import successfully. CUDA and MPS are unavailable.
- Installed `gromo-0.0.1.dist-info/direct_url.json` records requested revision AND resolved commit `8d19107b61a9459a9021065a329b699adcb0f25b` from `https://github.com/growingnet/gromo`. The pin matches. Gromo's declared dependencies are installed. Existing notebook `notebooks/kaggle_plateau_end_to_end_t4x2.ipynb:41,67-83` uses the same detached revision and installs the repository and Gromo package with dependencies; no replacement TINY implementation was added.
- Initial baseline, before edits: `.venv/bin/python -m pytest -q`: **221 passed, 11 failed, 5 skipped**: the 236-test progress stream includes four test skips, plus the collection-time skip of `tests/test_projection_diagnostic.py` for missing pandas (repository addopts already includes `-q`, suppressing the usual count summary). Ten failures were archive/checkpoint discovery refusing the available ~114–122 MB disk space against a 256 MB reserve. One was a stale source assertion in `tests/test_plateau_fork.py::test_plateau_fork_has_recurrent_e_and_single_shot_o_control`.
- Existing suite rerun without the new file: `.venv/bin/python -m pytest -o addopts='' -q --ignore=tests/test_vna.py`: **1 failed, 231 passed, 5 skipped, 36 warnings in 85.99s (0:01:25)**.
- Final full suite: `.venv/bin/python -m pytest -o addopts='' -q`: **1 failed, 240 passed, 5 skipped, 36 warnings in 495.97s (0:08:15)**.
- New tests: **9 passed**, covering all requested Stage 1 items (1/2/3/4/7/10/11/12), including 50 random batches, a genuinely full-rank six-output Jacobian, finite differences, the λ-scaled gain identity, copied-buffer train BN, dedicated RNG, scope disjointness, native candidate parity and B independence. Checkpoint resume also compares model, optimizer momentum, scheduler, RNG and loader generator bitwise; signal-curve smoke runs without CIFAR.
- Ten environmental failures ceased when free disk space rose to ~1.5 GB during the session; no unrelated code was fixed, no disk guard was bypassed and no failure was intentionally reproduced by filling the disk. The pre-existing source assertion remains unchanged. See `stage1_test_results.json` for failure IDs.
- CPU smoke commands:

```bash
.venv/bin/python -m experiments.train_recipe_checkpoints --fake-data --max-epochs 1 --limit-batches 1 --output /tmp/vna-recipe
.venv/bin/python -m diagnostics.signal_curve --fake-data --smoke --checkpoint-dir /tmp/vna-recipe --output /tmp/vna-signal
.venv/bin/python -m diagnostics.signal_curve --fake-data --checkpoint-dir /tmp/vna-recipe --output /tmp/vna-default-signal
```

No local `data/cifar-100-python` dataset was found, so the conditional CIFAR sanity run was skipped. These observations use a native Gromo tiny ResNet (one block, width 3, 3 classes, 8×8 images), not a trained full CIFAR-ResNet18.

## Gromo alignment audit (pinned source evidence)

In this section `gromo/...` denotes `.venv/lib/python3.12/site-packages/gromo/...`; adapter paths denote `third_party/One-Shot-TAS-CCIL/dual_growth/adapters/tiny_adapter.py`.

1. **Exact A/B and extended forward.** A is `block.first_layer.extended_output_layer.weight` (and its bias if present), mapping block input into r virtual channels. B is `block.second_layer.extended_input_layer.weight`, shape `(out_channels,r,kH,kW)`; incoming virtual neurons are its dimension **1**, not dimension 0. Adapter `_virtual` installs those layers, sets the first scaling to 1, the second to gate, and delta scaling to 0 (`tiny_adapter.py:296-328`). Native growing forward adds B's convolution before the second post-layer function (`gromo/modules/growing_module.py:1450-1490`). The growing BN normalizes ordinary channels but returns virtual channels unchanged (`gromo/modules/growing_normalisation.py:270-296`); ReLU is then applied separately (`gromo/modules/growing_module.py:1393-1403`). Default dropout rate is 0 (`gromo/containers/resnet.py:92,439-441`). For the required **non-preactivation** model, writing `*` for convolution and `D` for the shortcut, the block output is exactly:

   `h = ReLU(BN1(W1*x)); h_E = ReLU(A*x); block_E(x) = D(x) + BN2(W2*h + gate*(B*h_E))`.

   Base convs and native extension convs are bias-free for this ResNet (`gromo/containers/resnet.py:428-438`). There is no virtual BN and no extra BN affine factor on A. BN2 is on the original output channels **after** the extension sum; upstream stage activations and all subsequent layers remain part of f. The block itself returns the residual sum (`gromo/containers/growing_block.py:260-269`). VNA invokes this actual upstream forward with B=0 and gate=1; it does not substitute a manually approximated feature map. Tests reproduce the candidate's own logits exactly when B is restored to native B.

2. **Statistics normalization and empirical scaling.** `compute_statistics` requires sum reduction CE, initializes statistics, forces eval mode, backpropagates the sum loss, and updates each statistic (`gromo/utils/training_utils.py:363-387`; adapter call `tiny_adapter.py:466-472`). Each `TensorStatistic` divides its accumulated tensor by its **sample count** (`gromo/utils/tensor_statistic.py:95-108,122-140`). Conv S and M sum spatial positions but count only the batch dimension (`gromo/modules/conv2d_growing_module.py:1017-1025,1056-1061`); S_growth likewise sums masked unfolded dimensions and counts samples (`:2064-2083`). Consequently the loss convention avoids the extra 1/batch factor that mean CE would inject into M. It does **not** make the spectrum scale linearly with batch size. For the same seeded native model:

| Data | λ1 |
|---|---:|
| First 256 samples | 0.00160904927179 |
| All 1024 independent samples | 0.00178834202234 |
| Same 256 samples repeated four times (1024 total) | 0.00160904892255 |

   Replication ratio is 0.999999783, rather than 4. Different independent samples change estimation noise. Values are also recorded in `stage1_scaling.json`. Statistics are streamed in batches of at most 64 to bound forward/unfold memory while retaining the native global sample normalization.

3. **Payload meaning and truncation.** Native TINY constructs `P=S^(-1/2)N`, SVDs it and returns retained singular values, not squared singular values (`gromo/utils/tools.py:272-280,302-339`). N includes the existing-weight projection; `gromo/modules/conv2d_growing_module.py:2143-2165` defines it from normalized M/cross covariance and delta. The solve reads S/N and returns those singular values (`gromo/modules/growing_module.py:2408-2455`). Native thresholding keeps `s >= min(statistical_threshold,s.max())` then applies `maximum_added_neurons` (`gromo/utils/tools.py:317-328`). The adapter computes effective rank and slices again (`tiny_adapter.py:489-502,520-522`). Thus `payload["tiny_eigenvalues"]` is the **post-threshold, post-rank-cap** spectrum, and effective-rank truncation does affect it. Phase 0 requests rank 16, retaining however many upstream can supply. For measurements the adapter uses numerical threshold 1e-6 and statistical threshold 0 so small singular values remain observable; no native solver is reimplemented. The tiny model supplies only 3 modes, so measured rank is 3, not 16.

4. **BN mutation and transactional restoration.** At this pin statistics collection explicitly forces eval (`gromo/utils/training_utils.py:375-377`), so tracked standard BN running mean/variance/counters are not updated. It does change modes and `.grad`. The generic warning that statistics *can* change BN buffers is conservative, not actual behavior of this pinned path. `TransactionalCandidateSource` saves all state_dict buffers, gradients and modes, reloads them and asserts equality (`probe/gromo_adapter.py:89-107`, snapshot/assert `:18-30`). Counterfactual target width is separately restored (`probe/counterfactual_tiny.py:63-84`). Tests enter from train mode and verify bitwise restoration. VNA's optional train-BN products pass fresh copied buffers on every functional call, so batch statistics are used without changes to registered running buffers.

## DISCREPANCIES and implementation decisions

- SPEC §0.2's statement "λ are therefore sum-scaled" is misleading if interpreted as proportional to sample/batch count. Upstream normalizes S/M per sample. Native singular values are retained unchanged; mean-CE g and U/V are computed separately. Noise comparisons are logged in native units without multiplying or dividing λ by sample count.
- The term "eigenvalues" in the payload names **singular values**, after upstream threshold and rank truncation. The split-half score uses a squared Rayleigh quotient and takes its square root to return to these units. Specifically, use half-1's leading left singular vector u of `S1^(-1/2)N1`, set `a=S1^(-1/2)u`, and report `sqrt(||aᵀN2||²/(aᵀS2a))`; this is explicitly a cross-validated estimate, not a new full-stat eigenvalue. Each half estimates its own native existing-weight residualization.
- No BN discrepancy was found on virtual channels. BN2's action on the output sum is included through the full native forward.
- `grad_delta` contains **λ_vna*d**, ready to add to existing SGD gradients; calling `add_vna_grad_` with its default scale 1 avoids multiplying λ twice. Diagnostics V exclude λ; ρ and the tested identity `gᵀJ grad_delta = λV` include λ.
- Exact symmetry/descent identities concern converged linear solves. Truncated CG is logged and its iterate retained as specified; it does not provide an exact identity or guaranteed PSD correction for arbitrary unconverged solves. This caveat matters for the measured default q solve below.
- Outgoing bias is included in J_E if present. Its shared energy is distributed equally across V_k to maintain `sum(V_k)=V`; native CIFAR ResNet extensions have no bias.

## Phase 0 supervisor measurements (CPU surrogate, one checkpoint/site)

Artifact: `stage1_cpu_signal_curve.csv` and `stage1_cpu_summary.json`. Checkpoint is seed-0 tiny recipe epoch 1 after one SGD batch. This is an integration/cost sanity result, not Phase 0 acceptance on CIFAR-100. Non-smoke defaults were used: stat_samples=1024, four disjoint 256 sub-batches, vna_batch=128, cg_iters=30, cg_tol=.01, λ_vna=1; held-out uses 512 trigger samples. Fake trigger set has 128 unique samples, so it cycles to 512; real trigger split has 2000 and does not repeat. Real statistics/VNA inputs come from the augmented TRAIN dataset; held-out comes from the eval-transform trigger subset.

| Site | Full λ1 | Sub-batch mean ± population std | Split-half λ1 | Shuffled-label λ1 |
|---|---:|---:|---:|---:|
| stages.0.blocks.0 | 0.0043363236 | 0.0046514082 ± 0.00070868186 | 0.0038330953 | 0.0037506851 |

Noise floor mean+2std = 0.0060687719; all four requested estimates are reported, and none is chosen to redefine the controller. ρ = 4.3501816e-05; **0/1 (0%)** rows exceed 0.3. Tiny V = 1.7173061e-07, random-A V = 1.5489428e-07, ratio 1.109. Held-out derivatives: VNA -1.8056683e-07, random A -1.6918125e-07, norm-matched plain block gradient -6.1000023e-07. This sample does not meet the 3× TINY-versus-random criteria. A single early checkpoint cannot establish decay near epoch 200.

| Cost (same CPU/model) | Seconds | Network JVP / VJP | Extension JVP / VJP | Ratio to one SGD step | Overhead at every 4 steps: 1 / 3 sites |
|---|---:|---:|---:|---:|---:|
| SGD, batch 64 | 0.00341353 | — | — | 1 | — |
| VNA cold, batch 128 | 0.72486 | 46 / 48 | 1 / 3 | 212.3 | 5308.73% / 15926.18% |
| VNA warm, batch 128 | 0.586595 | 34 / 36 | 1 / 3 | 171.8 | 4296.11% / 12888.32% |

SGD seconds are the median of three separately timed single steps after one untimed setup step; raw trials are in CSV. Each restores the exact checkpoint state. Cold VNA includes 8 Hutchinson probes; warm reuses the same μ/Jacobi cache and previous z/q. They are measured on the **same batch**, so warm is a favorable lower bound, not a changing-batch training prediction. Timing includes state snapshots/restoration and CUDA synchronization when applicable. The 3-site number extrapolates sequential independent corrections; it is not an actual three-site measurement on this one-site model. Full ResNet18/GPU cost remains unmeasured locally. Cold z converged; cold and warm q exhausted 30 iterations (relative residuals 0.15723 and 0.13381). No thresholds were tuned to improve these observations.

The top-q complement uses a matrix-free randomized eigensolver (q=20, oversampling 8, two power passes; removes numerical null modes), not a dense K construction. U/V and CE diagnostics use fp64 reductions, while production network products are fp32 with autocast disabled. Damping/warm caches are returned by the core so a later controller can refresh them without recomputing every step.

## Files and intentionally deferred items

Added implementation/tests: `methods/vna.py`, `experiments/vna_support.py`, `experiments/train_recipe_checkpoints.py`, `diagnostics/signal_curve.py`, `tests/test_vna.py`. Added documentation/evidence: this report, `stage1_scaling.json`, `stage1_cpu_signal_curve.csv`, `stage1_cpu_summary.json`, `stage1_test_results.json`. The task/spec inputs `docs/vna/TASK_STAGE1.md` and `docs/vna/SPEC.md` are included in the isolated local commits for reproducibility. Updated only the allowed existing file `third_party/REFERENCES.md`. Existing `.DS_Store` edits and unrelated `docs/vna/TASK_STAGE1B.md` are left outside these commits.

- `--grow-epochs` is a clearly labeled TODO and fails explicitly if nonzero. A scientifically paired persistent-growth reference requires full optimizer state migration, identical transformed loader streams/RNG, two independent resumable trajectories and evaluation controls; integrating and verifying those is beyond the task's approximately 150-line optional budget. No persistent growth is installed by this diagnostic.
- Controller/cache refresh policies, G1/G2/G3, accept test, stop, training arms and notebooks are not implemented, as explicitly required by Stage 1 scope. The core supports damping and warm-start caches only.
- Full 200-epoch CIFAR training, every-site GPU measurements and Phase 0 scientific acceptance require CIFAR data and accelerator execution; CPU/fake data do not substantiate those conclusions. All required outputs for those runs are implemented.

Open questions for Stage 2 (no controller choices made): Does G1 remain meaningful with native sample-normalized singular values? Which cross-validation treatment of the nuisance existing-weight projection should be standardized? Is the nonconverged default q solve acceptable or must tolerances/preconditioning change? How different is η/ρ under batch-stat BN versus eval BN? What `g3_ratio` default is intended (not listed in SPEC §3)? Should future rank attribution include a separate shared-bias component outside V_k?

## Exact GPU commands

Assume the pinned Gromo package, torchvision/CUDA torch and the reference adapter checkout are installed as in the existing notebooks; CIFAR-100 is extracted under `/kaggle/input/cifar100`. Run from the repository root:

```bash
.venv/bin/python -m experiments.train_recipe_checkpoints --device cuda:0 --data-root /kaggle/input/cifar100 --output outputs/vna_recipe_seed0 --seed 0 --workers 4
# Resume exactly the same recipe if interrupted:
.venv/bin/python -m experiments.train_recipe_checkpoints --device cuda:0 --data-root /kaggle/input/cifar100 --output outputs/vna_recipe_seed0 --seed 0 --workers 4 --resume outputs/vna_recipe_seed0/latest.pt
.venv/bin/python -m diagnostics.signal_curve --device cuda:0 --data-root /kaggle/input/cifar100 --checkpoint-dir outputs/vna_recipe_seed0 --output outputs/vna_signal_seed0 --seed 0 --vna-bn eval
# Optional BN-approximation comparison (copied running buffers):
.venv/bin/python -m diagnostics.signal_curve --device cuda:0 --data-root /kaggle/input/cifar100 --checkpoint-dir outputs/vna_recipe_seed0 --output outputs/vna_signal_seed0_train_bn --seed 0 --vna-bn train
```

Checkpoints retain only multiples of 10 plus latest. Scheduler is MultiStepLR(100,150,γ=.1), 200 epochs; SGD lr=.1, momentum=.9, WD=5e-4, default batch 64. Training/evaluation/splits/atomic saves/RNG reuse shared protocol helpers. The diagnostic deduplicates latest against the same epoch's periodic checkpoint.

## Local commit delivery and sandbox limitation

The original checkout is on `vna-phase0`, but writing `.git/index.lock` was rejected with `Operation not permitted`: `.git` is explicitly read-only in this session's filesystem profile, and escalation is disabled. Therefore its branch pointer cannot be updated here. No approval was requested and no protected metadata was modified.

The authorized files are instead committed in the isolated local clone `/tmp/vna-stage1-committed` on **branch `vna-phase0`**, in three small logical commits. This clone has no pushes. `docs/vna/STAGE1_COMMITS.bundle` exports the branch's Stage 1 commits for local import; the bundle itself is a delivery artifact outside its own commit history. Once Git writes are permitted, import locally from the original checkout with:

```bash
git fetch docs/vna/STAGE1_COMMITS.bundle vna-phase0
git merge --ff-only FETCH_HEAD
```

The current worktree already contains identical authorized file contents; because they are untracked/modified relative to the original branch, archive those working copies before merging if Git reports an overwrite conflict. Do not discard unrelated changes. The clone is the immediately reviewable committed result; the original-checkout commit requirement remains limited by the sandbox.

## Stage 1B — supervisor fixes and resumable Kaggle Phase 0

Date: 2026-10-13. Changes are left in the working tree. No commit, clone, bundle,
push, or Git metadata write was performed for Stage 1B. The preceding Stage 1
commit-delivery section is historical: Stage 1 is now at HEAD
`454f3bd61000e65d72bb86f3a768061fb11b5f79`. The obsolete
`docs/vna/STAGE1_COMMITS.bundle` was absent; the existing `stage1_*.json/csv`
evidence files were retained. No new generated measurement files were added to
the repository; CPU timing scratch evidence is `/tmp/stage1b_cpu_benchmark.json`.

### Supervisor fixes

- `_Products.JT` and `JET` lazily build one pullback each per product object
  (fixed parameter set/batch/virtual context), then reuse it. `forward_calls`
  counts actual functional forwards plus the optional prefix evaluation;
  `vjp_calls` and `extension_vjp_calls` count pullback applications, including
  applications of an already-built pullback. Existing JVP counters remain.
  Tests compare both eval- and copied-buffer train-BN pullbacks bitwise against
  the old rebuild-on-every-call path, and verify subsequent applications perform
  no new forward.
- `realized_gain = gᵀJ grad_delta` uses one additional fp32 network JVP and an
  fp64 dot product. It measures the returned, rounded, lambda-scaled parameter
  correction. A finite gain <=0 sets `nonpositive_gain`, `skipped`, and
  `skip_reason=nonpositive_realized_gain`; the returned correction is empty.
  Nonfinite gains also return no correction. Predicted V/rho, solver residuals,
  and warm/damping caches remain logged. Tests force a negative q iterate and
  prove it is rejected. Converged dense-oracle/descent tests still pass.
- This guard addresses the unsafe sign of an unconverged correction; it does
  not assert convergence, repair the default CG tolerance, or imply that the
  predicted lambda*V equals the realized gain for truncated solves.

### Kaggle inputs

Read and reused the conventions in
`scripts/build_kaggle_resnet18_seed1_raw_vs_vanilla_rollback_notebook.py`,
`scripts/build_kaggle_deit_seed5_raw_vs_vanilla_rollback_notebook.py`,
`scripts/build_kaggle_unified_seed_notebooks.py`, and
`experiments/kaggle_checkpoint_discovery.py`. The new builder reuses the unified
notebook's bootstrap/setup cells; the older experiment-specific selection and
training logic is not incorporated.

- Attach one CIFAR-100 Python-format dataset. Discovery walks `/kaggle/input`
  with `os.walk(..., followlinks=True)` and requires a directory named
  `cifar-100-python` containing **train, meta, and official test**. The notebook
  prints the verified test path and passes its parent as `--data-root`.
  Multiple matches are sorted and the first is used, matching unified discovery.
  This is a path/content-presence check; no local CIFAR data exists to validate
  the dataset bytes or perform the full scientific run.
- Enable **Internet**, **T4×2**, and Kaggle secret **github_token** with repository
  access. The existing temporary askpass script supplies the token from a child
  environment, never from a URL or printed command. It is removed in `finally`.
- Clone `https://github.com/duyh80456-code/Counterfactual-Projection.git` into
  `/kaggle/working/counterfactual-projection` with no checkout, fetch
  `REPO_COMMIT` using the same authenticated environment, detach at that full
  hash, and verify HEAD. The repository is never selected by a branch tip.
- Reuse the existing reference checkout from
  `https://github.com/duyh80456-code/One-Shot-TAS-CCIL.git`, branch
  `ccil-residual-capacity`, under `/kaggle/working/One-Shot-TAS-CCIL`.
  Gromo is cloned under `/kaggle/working/gromo`, fetched and detached at
  `8d19107b61a9459a9021065a329b699adcb0f25b`, and installed editable with its
  declared dependencies. The repo is installed with test/vision/diagnostics
  extras. Runtime PYTHONPATH follows the existing repo/Gromo-src/reference order.
- Before accelerator work, a fresh subprocess with `CUDA_VISIBLE_DEVICES=""`
  must pass `python -m pytest -q tests/test_vna.py`. Two GPUs are then required.
- Optional previous outputs may be attached as a directory, tar/tar.gz/tgz, zip,
  or Kaggle-expanded torch-save directory. The existing `discover_checkpoints`
  helper searches by `kind=vna_recipe` and matching recipe/throughput protocol,
  recovers full latest state plus all periodic model snapshots, and releases
  temporary extraction storage. Diagnostic completion JSON is restored from
  directories or archive members without extracting arbitrary paths. Conflicting
  completed outputs are rejected. Model hashes and algorithm/config identity
  prevent silently reusing diagnostic rows from different math or trajectories.
- Outputs live under `/kaggle/working/vna_phase0_seed0/{recipe,signal}`.
  `recipe/latest.pt` retains fp32 model, optimizer momentum, scheduler, RNG,
  training-generator state, splits, history, protocol, and AMP scaler state.
  `epoch_010.pt` through `epoch_200.pt` retain fp32 model/epoch/protocol only.
  `signal/completed/epoch_NNN.json` commits all rows for one checkpoint atomically;
  CSV and summary are rebuildable from these completion files.
- `experiments/vna_kaggle.py` creates an atomic sibling
  `/kaggle/working/vna_phase0_seed0.tar.gz` hourly and in the execution cell's
  `finally`, then again after final plots. It hard-links immutable checkpoint/
  JSON inodes into a temporary snapshot so replacement of latest during archive
  creation cannot mix checkpoint bytes. Logs are copied; locks and `.tmp` files
  are omitted. A hard kill preserves the last archive and working files; attach
  both available outputs when resuming. The interrupted epoch and unfinished
  checkpoint transaction may be repeated; committed checkpoints are not recomputed.

### Notebook instructions and execution

Generate with `.venv/bin/python scripts/build_kaggle_vna_signal_curve_notebook.py`
(or `--commit FULL_40_CHARACTER_HASH`). The delivered notebook has the required
**current HEAD default** above. Because this task explicitly forbids committing,
that Stage 1 hash does not yet contain Stage 1B. **After the supervisor commits
these changes, regenerate with that new committed/published hash before Kaggle
execution.** A preflight error explains a missing Stage 1B revision. No local
uncommitted code is silently injected into the pinned checkout.

Attach the inputs above and run all cells. Seed 0, 200 epochs, batch 64, the
shared splits, SGD/momentum/WD, and milestones 100/150 remain fixed. GPU0 trains;
GPU1 watches atomically saved periodic checkpoints. At training completion the
watcher finishes at most its current checkpoint, then remaining checkpoints are
split by `(epoch // 10) % 2` across both GPUs, sharing the completion directory.
This explicitly avoids putting all even-numbered epochs on one GPU. Drain workers
never write the merged CSV concurrently; one merge follows their successful exit.
A crashed claim is recoverable on the next invocation. Duplicate latest/periodic
models at the same epoch are skipped, not processed twice.

Final output displays per-site U/V/eta/rho, TINY/random V ratio, held-out VNA,
random and norm-matched LR-boost derivatives, realized gains, cold/warm SGD cost
ratios, native/full/sub-batch/split-half/shuffled noise controls, and rho>0.3
frequency. It saves plots and automatically evaluates the simultaneous pre-150
3x criteria. For the qualitative "decay toward 0" criterion, the declared,
untuned operational rule is: mean U, V and rho at epochs >=180 must each be <=10%
of the pre-150 mean, per site. Missing end checkpoints or synthetic/smoke data
produce **not assessed**, never a scientific go result. This operational rule
is an explicit interpretation, pending supervisor standardization.

### Throughput flags, correctness, and evidence

All CLI throughput flags default **off**. The notebook enables only
`CACHE_BLOCK_INPUT=True`, supported by exact-product tests; optional
`GPU_DATA`, `CHANNELS_LAST`, `CUDNN_BENCHMARK`, and `AMP` are false.

- `--gpu-data`: keeps CIFAR train uint8 resident on the selected device. Each
  gathered batch uses zero-pad-4, uniform 32x32 crops, horizontal flip p=.5,
  float conversion and the unchanged CIFAR normalization. Only the supplied
  CPU training generator draws shuffle/crop/flip decisions; VNA/statistics/cost
  batches use their separate purpose generators. No global RNG is consumed.
  CPU-oracle tests verify bitwise paired-arm batches, global-RNG preservation,
  dedicated statistics batches, and continuation after restoring generator state.
  The distribution matches the original transform, but its draw ordering is not
  the legacy PIL loader's bitwise stream. Use the same flag in every paired arm;
  changing it is a resume protocol mismatch. Resident uint8 costs ~154 MB/GPU.
- `--channels-last`: changes model and SGD input layout; no float precision
  change. `--cudnn-benchmark`: enables cuDNN algorithm benchmarking. Both remain
  optional because local CUDA timing/determinism could not be tested. Existing
  shared loaders already supply pin_memory and non_blocking copies; the new SGD
  helper retains those transfers. Resident training needs no per-batch image
  host-to-device copy beyond the small random offsets/index vectors.
- `--amp`: CUDA fp16 autocast and GradScaler for plain SGD only, with saved scaler
  state. It must match every arm. **OFF** for Phase 0: no CUDA is available to
  demonstrate the required >=1.5x speedup. VNA and block product contexts disable
  enclosing autocast explicitly; tests verify fp32/bitwise results even inside
  CPU bfloat16 autocast. No fp16 diagnostic checkpoints are written.
- `--cache-block-input`: exact native eval-BN Gromo prefix input cached once per
  product object, then functional products execute only the selected block and
  all subsequent native blocks/activations/post-net. Train-BN uses the full path.
  Tests prove bitwise logits, J/J_E JVP/VJP equality both at the first block and
  a later block after earlier stages. Diagnostic batches are fixed across sites/
  epochs; logits and mean-CE g are cached once per checkpoint for VNA and heldout
  batches. A shared-logits/g equivalence test verifies identical correction and
  realized gain with one fewer forward.

CPU measurements used Python 3.12.7/torch 2.2.2, one thread, native tiny Gromo,
128 VNA samples, rank 3, default 30 CG iterations and eight Hutchinson probes.
Baseline code was read from HEAD into a temporary Python module, not cloned.
Each correction timing is the median of three runs after a setup call. Epoch
measurements use the same initialized tiny model and 16 prebuilt batch-64 batches,
three timed epochs after setup, with identical SGD; augmentation/I/O is excluded.

| CPU surrogate | Stage 1 | Stage 1B | Interpretation |
|---|---:|---:|---|
| Cold correction, full path | 0.66751 s | 0.50162 s | 1.33x; includes new gain JVP |
| Cold correction, cached prefix | 0.66751 s | 0.44711 s | 1.49x on this tiny model |
| 16-batch plain SGD epoch | 0.06837 s | 0.06742 s | essentially unchanged default SGD |

Old correction executes 46 JVP and 48 VJP applications, each VJP rebuilding its
forward. New full path executes 47 JVP and 48 VJP applications with only two
pullback-building forwards; total forward count is 51 including extension
products/logits. Suffix path has 52 calls including its one cached prefix, with
all later product forwards shorter. This count is not a claim about full-model
GPU speed. No CUDA/MPS is available and no local CIFAR exists.

T4×2 planning estimates (unmeasured, deliberately broad): training at roughly
10–40 s/epoch would take 0.6–2.2 h for 200 epochs; diagnostics at 1–10 min per
checkpoint/site across 20 checkpoints ×8 sites would consume 2.7–26.7 GPU-hours,
or approximately 1.3–13.3 h on two GPUs after allowing both to drain. Overlap
reduces elapsed time, but a full run may exceed Kaggle's ~12-hour session; resume
is required rather than assumed unnecessary. Native TINY covariance solves,
512-sample heldout products, and the matrix-free top-q eigensolver may dominate,
so these are capacity-planning assumptions, not predicted/verified benchmarks.
Actual train-only epoch seconds are saved in history; every row includes refresh,
cold/warm, SGD-step timings and ratios for revising estimates from the first
real checkpoints. CPU augmentation on four vCPUs motivates the optional resident
path; fp32/batch-64 is kept until accelerator evidence justifies other toggles.

Disk: full ResNet18 has 11,220,132 parameters, and state_dict tensor storage is
44,980,656 bytes. Twenty fp32 model-only snapshots total ~0.900 GB; latest model
plus SGD momentum is ~0.090 GB, giving ~0.99 GB raw outputs plus small JSON/logs.
Output + final archive should be ~2 GB even without compression savings. During
atomic archive replacement, raw + old + new archives is conservatively ~3 GB,
plus checkouts and an old hard-linked latest inode. Restore extraction storage is
removed before execution. Archive creation checks a conservative 3x raw-byte
budget against 5 GB and prints raw/archive/free bytes; training prints checkpoint
bytes each epoch. Periodic momentum retention would nearly double snapshot
storage, so the Stage 1B model-only exception is used from the outset.

### Stage 1B validation and open questions

Supervisor decision: Phase 0 trains one SGD trajectory with no paired-arm bitwise requirement, so Kaggle defaults GPU_DATA=True, CUDNN_BENCHMARK=True, CHANNELS_LAST=True for T4 speed; AMP=False and CACHE_BLOCK_INPUT=True remain unchanged, with relevant flags propagated to training/diagnostics and enforced in resume identities.

Focused checks cover bitwise old/new pullbacks, forced negative gain rejection,
outer-autocast isolation, exact first/later-block suffixes, shared logits/g,
generator-only device augmentation on CPU, atomic completion idempotence,
watch/shard/merge, tar resume of model/latest/diagnostics, and generated notebook
code-cell compilation/pinning/CPU gate/test-file discovery.

Open questions (recorded without asking): what numerical criterion should define
"decay toward 0" for the official Phase-0 decision? What are actual T4 epoch,
TINY refresh and per-site diagnostic times? Do channels_last/cuDNN benchmarking
preserve the intended accelerator reproducibility, and does identical-arm AMP
exceed 1.5x speedup? Is the existing reference branch checkout sufficient, or
should its helper code also get a separate immutable pin? Stage 2 controller/
real-growth questions from Stage 1 remain deferred. Full 200-epoch CIFAR/T4
scientific acceptance and accelerator flag performance cannot be established
in this CPU-only environment.

Final full-suite command (after the final shared-logits cache change):
`.venv/bin/python -m pytest -o addopts='' -q` → **248 passed, 1 failed,
5 skipped, 36 warnings in 112.83 s**. The one failure is the same pre-existing
`tests/test_plateau_fork.py::test_plateau_fork_has_recurrent_e_and_single_shot_o_control`
source assertion at line 252 documented in Stage 1. The unrelated runner and
its test were left unchanged. All **17 VNA tests pass** (nine Stage 1 cases plus
eight Stage 1B cases counting parameterization).

An additional real CLI integration started a CPU watcher before a concurrent
10-epoch, one-batch-per-epoch fake-data recipe. The watcher processed the arriving
atomic epoch-10 checkpoint with `--cache-block-input`; the periodic checkpoint
had only fp32 model tensors and no optimizer. Re-running diagnostics, including
the duplicate latest, preserved the completion marker's nanosecond mtime and
merged exactly one row. The smoke result is synthetic and not Phase-0 acceptance.
Notebook generation was reproduced byte-for-structure from current HEAD; every
code cell compiled. `compileall` for all implementation/builder files and
`git diff --check` passed. HEAD is unchanged; the pre-existing `.DS_Store`
modification is retained and outside this task.

## Stage 1C — controls, VNA-lite, and stride-20 rank diagnostics

Date: 2026-10-13. All changes remain in the working tree; no commit, push,
clone, bundle, or Git metadata write was made. The pre-existing `.DS_Store`
modification and untracked `TASK_STAGE1C.md` were preserved. No unrelated runner
or shared helper was modified.

### Changes and declared analysis rules

- Full VNA accepts `apply_gain_guard=True`; random-A controls use `False`,
  retaining finite negative-gain directions rather than replacing them with an
  empty mapping. Full/lite/random/random-lite realized gains, guard settings,
  skip flags/reasons, and nonfinite flags are logged. Nonfinite corrections are
  still skipped. Held-out empty directions produce null derivatives/cosines and
  explicit validity/empty flags. Every criterion requires its actual finite,
  nonempty directions; predicted V cannot rescue a skipped direction. Rank
  matching is checked, and each criterion has eligible/passed/excluded counts.
- G1's primary comparison is `lambda_full` from all `stat_samples` against
  three independently seeded shuffled-label solves on exactly those same
  inputs/sample counts. The shuffled values, mean, maximum, and sample counts
  are logged. `significant_vs_shuffled` uses **full lambda > shuffled maximum**;
  `significant_split_half` uses **cross-validated lambda > shuffled maximum**.
  These conservative, declared comparisons are reported per checkpoint/site.
  The sub-batch mean+2std remains an extra logged diagnostic; it no longer drives
  the primary significance flag or d_mu. No threshold was tuned.
- `vna_lite_correction` shares full VNA's native `_Products`, state/RNG/mode/
  gradient restoration, precision, and Hutchinson trace. It replaces both R
  products by I/sqrt(c), c=(trace_K/n)^2, and never invokes CG. It reports U,
  V/V_lite, eta, rho/rho_lite, c, per-neuron energies, realized gain, actual
  product/forward counts, and synchronized wall time. Cached lite has two
  network JVPs (including the gain measurement), two network VJPs, one extension
  JVP, and three extension VJPs (including diagnostic energies). Cold lite adds
  the eight Hutchinson products; diagnostic lite per-step cost reuses the full
  correction's same trace cache. Full VNA now also reuses its already computed
  J^Tg when forming Kg, removing one redundant pullback application.
- All J^Tg, J, and J_E products use the configured `--vna-bn` mode. **Stage 2
  may reuse the SGD batch's train-mode gradient only with `bn_mode="train"`
  and copied buffers, or must recompute J^Tg in eval mode.** A logit gradient
  (`shared_g`) is distinct from a parameter gradient; the diagnostics compute
  shared logits/g in the matching BN mode and never silently mix modes.
- Each checkpoint selects up to three sites by default-rank full V on an
  independent augmented selection batch. Selection TINY statistics use their
  own `rank-selection-statistics` generator and stat_samples; selection products
  use a distinct `rank-selection` generator and vna_batch. Neither uses the
  measurement statistics or measurement VNA batch. Selection V/time and chosen
  sites are logged. Nonfinite selection scores are omitted rather than ranked.
- All sites retain a default-rank row. Selected sites additionally request
  ranks 4, 16, 64, and current hidden width, deduplicating identical requests.
  Native counterfactual Gromo receives width+r and supplies its available rank;
  both requested/effective ranks, width, and spectrum count/min/sum are logged.
  Each sweep row includes full/lite/random/random-lite U/V/eta/rho, gains,
  timings, held-out derivatives/cosines, and independently norm-matched full
  and lite LR-boost comparisons (including scales/realized training gains).
  Rank shortfalls are recorded and do not abort. The tiny CPU model naturally
  returns only rank 3 for requests 4/16/64, demonstrating that path.
- `--diagnostic-stride 20` filters compute, watcher, and final drain; training
  still retains ten-epoch checkpoints. Sharding divides by diagnostic stride
  before assigning GPU shards, so both GPUs receive work at stride 20. Explicit
  partial/smoke diagnostics require a stride override when epochs are off-grid.
- Every row measures SGD at **batch 64**, including smoke, with three restored
  single-step trials after setup and median cost. Full cold/warm and cached
  lite ratios use that same model/device baseline; all CUDA timings synchronize.
- `summary.json` and the printed compact table report each requested rank's
  three full and three lite early criteria, joint rows, exclusion counts,
  cost ratios, and per-site decay. Early means epoch <150; late means >=180;
  the existing <=10%-of-early-mean rule applies separately to U/V/rho for each
  requested rank/site/variant. Decay exclusions are counted too. Synthetic or
  incomplete coverage remains unassessed. Defaults and sweeps are kept separate
  to avoid double-counting the default rank. A go requires all three criteria
  together on an early row and the declared decay rule; full and lite decisions
  are available separately in the per-rank table.
- Resume identity now includes `schema_version="stage1c-v1"`, stride, flags,
  and the existing algorithm source digest. Run preflights **all** old markers,
  even epochs excluded by the new stride; merge also rejects incompatible data.
  Training checkpoints remain resumable. Use a fresh diagnostic output directory
  for Stage 1C when attaching pre-Stage-1C diagnostics.
- The notebook was regenerated through its builder at current HEAD (supervisor
  must re-pin after committing Stage 1C). Its common diagnostic command carries
  stride 20 into watcher and both drain workers, and final cells show new
  full/lite columns, significance, and the per-rank acceptance/cost/exclusion
  table. Plots group by site and requested rank instead of joining rank sweeps.

Files touched: `methods/vna.py`, `diagnostics/signal_curve.py`,
`scripts/build_kaggle_vna_signal_curve_notebook.py`,
`notebooks/kaggle_vna_signal_curve.ipynb`, `tests/test_vna.py`, and this report.
**Shared helpers touched: none.** Existing training and unrelated runners are
unchanged.

### Validation and CPU timing

New tests cover forced-negative unguarded controls and exclusion of empty/NaN
comparisons; dense VNA-lite equality to the full chain with R=I/sqrt(c), descent
identity to 2e-5 relative tolerance, and centered double-forward finite differences
at 1e-3 relative tolerance in eval and train BN; state/gradient/mode/RNG invariance;
no CG calls; standalone/cached Hutchinson consistency and product counts;
independent selection statistics/products; same-count three-shuffle significance;
requested/effective rank shortfalls; stride-20 watcher and two-shard drain;
pre-Stage-1C identity rejection; and per-rank decay, missing-direction exclusions,
and epoch-150/180 boundaries. Existing recipe resume and notebook checks remain.

Focused command `.venv/bin/python -m pytest -o addopts='' -q tests/test_vna.py`
passed **24 tests**, with 36 upstream deprecation warnings, in **31.06 s**.
The notebook is structurally identical to `build(current_HEAD)` and every code
cell compiles; its current pin is `018f79415cb64491b5b304bfb1b8c4c2658a86fb`.

CPU CLI smoke (native one-block Gromo, hidden width 3, three classes, fake 8x8
inputs; stat_samples=16, selection/VNA batch=2, heldout=4, CG cap=10,
`--cache-block-input`, fp32, no AMP) produced one default row plus four sweep
rows and a per-rank table. All rows were finite; requests 4/16/64 returned rank
3. This is an integration observation, not evidence for CIFAR/T4 acceptance.
SGD was genuinely batch 64; full warm and lite reused the cold trace cache,
and full warm also reused z/q on the identical batch (a favorable bound).

| CPU row | Full cold seconds / SGD ratio | Full warm seconds / SGD ratio | Cached lite seconds / SGD ratio |
|---|---:|---:|---:|
| Default requested/effective rank 2 | 0.14617 / 45.24 | 0.03048 / 9.43 | 0.02063 / 6.39 |
| Mean of four sweep requests (all effective rank 3) | 0.14032 / 43.43 | 0.03329 / 10.30 | 0.02313 / 7.16 |

Median batch-64 SGD step: **0.003231 s**. These are single-run CPU timings with
other test work possible during execution, not stable GPU speed predictions.
The synthetic checkpoint met neither three-way joint early criterion set;
summary correctly remains `not assessed: synthetic/partial run` because no
late training trend exists. Scratch commands/evidence (outside the repository):

```bash
.venv/bin/python -m experiments.train_recipe_checkpoints --fake-data --max-epochs 1 --limit-batches 1 --output /tmp/stage1c-recipe
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m diagnostics.signal_curve --fake-data --smoke --diagnostic-stride 1 --cache-block-input --checkpoint-dir /tmp/stage1c-recipe --output /tmp/stage1c-smoke-final
```

Scratch CSV/summary/markers are under `/tmp/stage1c-smoke-final`; focused logs
are `/tmp/stage1c_vna_tests_final.log`. A first full-suite attempt was interrupted
at 92 passed/1 skipped after 135.12 s during checkpoint serialization. The next
full run completed with **248 passed, 8 failed, 5 skipped** in **122.48 s**:
seven failures were the unchanged 256 MiB checkpoint-discovery free-space guard
(including the VNA archive round trip), and one was the historical plateau source
assertion. Free space had fallen to ~212 MiB because those two full-suite scratch
runs together retained ~1.5 GiB. Only those newly generated scratch directories
were removed; logs were retained, no guard was bypassed, and the full suite was
rerun with OMP/MKL threads limited to one.

### Updated T4 planning estimate and open questions

The stride-20 plan has ten checkpoint times (20..200), eight default sites per
checkpoint, and up to 3x4 additional rank rows: **80 default + up to 120 sweep =
200 measurement rows**, plus **80 independent selection-site evaluations**.
Width 64 coincides with request 64 at some sites and reduces actual row count.
Three null shuffles increase refresh work; nulls/sub-batches/split-half estimates
are reused across requested ranks. The expensive top-q check remains only on
80 default rows. Selection uses 1,024 statistics samples but only 128 product
samples; it does not construct full-stat-sample NTKs or run CG on 1,024 samples.

Using Stage 1B's deliberately broad, **unmeasured** 1–10 min/site-diagnostic
budget and conservatively charging each measurement or selection evaluation a
whole such unit gives **280–2,800 GPU-min = 4.7–46.7 GPU-hours**, approximately
**2.3–23.3 h with both T4s draining**, plus the existing 0.6–2.2 h training budget
(10–40 s/epoch). A more favorable planning scenario charges sweep rows half and
selection rows one quarter of a default diagnostic, yielding 160 units,
**2.7–26.7 GPU-hours / 1.3–13.3 h dual-GPU drain**. These are capacity ranges,
not hardware benchmarks. Training overlaps the watcher; the two-GPU drain and
strict checkpoint transactions support interruption/resume if the session ends.
Use actual selection/refresh/rank/full/lite/heldout row costs from the first T4
checkpoints to revise this estimate. Local CUDA/MPS and CIFAR data are unavailable.

Open questions recorded without asking: actual T4 cold/warm/lite and high-rank
memory costs; whether split-half significance should eventually compare against
split-half shuffled Rayleigh scores instead of the declared full-sample null
maximum; how to assess per-site decay if top-three membership changes so a site
has no valid early or late sweep measurements (currently left unassessed, never
imputed); whether the existing 10% decay rule is the intended scientific rule;
and whether default truncated full CG needs a later accuracy change. No
controller parameters, convergence tolerances, recipe, or acceptance thresholds
were tuned in Stage 1C. Stage 2 gradient reuse must observe the BN restriction
above. The supervisor still needs to commit and regenerate the notebook pin;
this task deliberately leaves all changes uncommitted.

Final validation after all implementation changes:

- `.venv/bin/python -m pytest -o addopts='' -q tests/test_vna.py`:
  **24 passed, 36 warnings in 19.28 s**. Log:
  `/tmp/stage1c_vna_tests_delivery.log`.
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -o addopts='' -q`:
  **253 passed, 3 failed, 5 skipped, 36 warnings in 122.65 s**. All 24 VNA
  cases passed in this full run. Log: `/tmp/stage1c_full_tests_delivery.log`.
  The two environmental failures were
  `tests/test_deit_resume.py::test_archive_resume_selects_latest_and_preserves_original_bytes`
  (199,933,952 bytes free versus the 256 MiB reserve) and
  `tests/test_kaggle_checkpoint_discovery.py::test_rebuilds_kaggle_expanded_torch_archive`
  (262,459,392 bytes free versus that reserve). Space temporarily dipped during
  large checkpoint writes despite ~1.9 GiB available at the start. After deleting
  only this run's completed scratch directory, these two exact tests passed:
  **2 passed in 0.21 s**, `/tmp/stage1c_disk_rechecks.log`. No guard or unrelated
  code was changed. The remaining failure is the previously documented
  `tests/test_plateau_fork.py::test_plateau_fork_has_recurrent_e_and_single_shot_o_control`
  source assertion at line 252; its runner/test remain untouched.
- The intervening clean full rerun had **248 passed, 8 failed, 5 skipped in
  130.31 s** (seven disk-reserve failures and that same source assertion).
  A focused rerun during that temporary disk shortage had 23 passed/1 archive
  failure; the final focused run above recovered all 24 without code changes.
  Historical failed-run logs remain in `/tmp/stage1c_full_tests_final.log`,
  `/tmp/stage1c_full_tests_clean.log`, and `/tmp/stage1c_vna_tests_verified.log`.
- Final-source CLI smoke under `/tmp/stage1c-delivery-signal` was verified against
  the **current** diagnostic identity: five finite rows, four requested-rank
  summary entries, one checkpoint/site significance entry, SGD batch 64,
  explicit missing late decay (`n/a`), and synthetic/partial acceptance.
  The report's timing table above uses the separately retained earlier smoke
  timing sample; the final smoke has the same math/schema and its own timings.
- Regenerated notebook equals `build(current_HEAD)`; every code cell compiles.
  `py_compile` for all four implementation/test/builder files and
  `git diff --check` pass. HEAD remains
  `018f79415cb64491b5b304bfb1b8c4c2658a86fb`; no commit was made.
