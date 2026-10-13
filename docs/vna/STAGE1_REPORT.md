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
