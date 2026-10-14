# Stage A/B implementation

## Stage B core

This implements `TASK_STAGE_B_CORE.md` shared infrastructure only. It adds no
setting-specific runner, applies no VNA direction to baseline training, and
makes no commit. Stage A continues using `GGNConfig(damping_norm='outputs')`.
Its schemas, thresholds, damping rule and numerical defaults are unchanged.

### Loss interface and normalization

`methods/vna_losses.py` defines `LossSpec`, `CE`, `MSE`,
`KDRepDistiller(alpha=.1, beta=.9, T=4)` and
`ICaRLPyCIL(T=2, known_classes=...)`. For real measurements, supply
`ctx={'targets': targets, 'loss_fn': upstream_callable, ...}` where
`upstream_callable(z, ctx)` calls the original upstream loss object. Teacher
logits and old-class count are passed in the same context. Residuals are
obtained by autograd on this callable, evaluated in float64 on a detached
output copy. No baseline gradients accumulate.

Both residual rows and metric actions contain **one batch factor 1/n**:

| Loss | Residual | H action |
|---|---|---|
| CE | `(p-one_hot(y))/n` | `(diag(p)-p p.T)v/n` |
| MSE, batchmean | `2(z-y)/n` | `2v/n` |
| MSE, element mean | `2(z-y)/(n*d)` | `2v/(n*d)` |
| RepDistiller KD | `.1*g_CE + .9*T*(p_student_T-p_teacher_T)/n` | `[.1*Cov(p)+.9*Cov(p_student_T)]v/n` |
| PyCIL iCaRL | `g_CE + (q_student_T-q_teacher_T)/(T*n)` on old classes | `[Cov(p)+Cov(q_student_T)/T² on old block]v/n` |

First-task iCaRL has CE only. PyCIL's KD is cross entropy with soft teacher
probabilities, with no T²; its scalar differs from KL by the teacher entropy.
MSE uses the supplied callable's actual quadratic Hessian action when present,
so an upstream element-mean reduction cannot silently receive a batchmean H.
Built-in defaults and copied reference formulas are for tests and compatibility;
real integrations pass their original loss through `loss_fn`.

`GGNProducts`, exact/lite/NGD solvers, filtered cotangents, joint correction,
Euclidean block/lite corrections, selection and residual-oriented diagnostic
helpers accept the generic loss. `tiny_V` also accepts one context per batch.
GGN H is the true output-loss Hessian, never empirical Fisher. Default/explicit
CE with no context preserves the original float64 softmax-minus-onehot rounding
after autograd evaluation; supplied upstream callables use their autograd
residual directly. A bitwise test locks the legacy CE residual, H and gradient.

### Declaring a site

```python
site = HookSite(model, 'block.conv1', 'block.conv2', 'conv',
                between=('block.bn2', 'block.relu2'))
# MLP: HookSite(model, 'fc1', 'fc2', 'linear', between=('relu',))
```

`between` lists native module paths in actual execution order. For WRN this
means conv1 -> BN -> ReLU -> conv2, after the block's initial BN/ReLU. The
original normalization executes normally. On new virtual coordinates norms
are identity, matching pinned Gromo's extension convention: new units do not
inherit old-channel affine parameters or running statistics. Incoming A uses
the first layer's geometry; external B uses the second layer's geometry and
starts at zero, including its bias if present. B's output is added by a
forward hook to the native second-layer output, before its downstream norm,
activation and residual addition. Grouped convolutions are rejected explicitly.

O defaults to the first/second layers and trainable between modules. Native
Gromo declarations reuse its full residual-path scope, including both norms
and excluding shortcuts. `o_params` can declare an explicit scope.
`HookProducts` supplies the existing J/J_E, adjoint, kernel and correction
contracts. Hooks exist only during a forward call; virtual tensors are never
registered on the original model. Native Gromo cached suffixes remain available;
generic branching graphs replay their full native forward for exact suffix
propagation. All contexts restore modes/buffers and assert unchanged parameter
count and state, including exceptional exits.

Native Gromo HookSites reuse its TINY statistics/solve. Generic linear/conv
sites form input-patch covariance and output-cotangent cross statistics, with
optional projection off existing second-layer features, then use a whitened
SVD. All selected maps pass the same normalization/sign-selection boundaries.
`tiny_U_from_cotangent`, `tiny_V`, `normalize_A`, block products/correction and
single/multiple-site joint correction work with HookSites. Gromo remains the
Phase 0/Stage A resolver.

### Alignment and measurement data

`diagnostics/vna_align.py::MeasurementHook(measure, seed=..., purpose=...)`
accepts `(model, batch_the_loop_already_has, loss_spec, ctx)`, calling
`measure(..., generator=dedicated_stream)`. It saves/restores original parameter
and buffer registrations/objects/values, gradients, module graph, each module's
train/eval mode, Python/NumPy and global CPU/CUDA RNG state, even on exceptions.
It asserts the parameter count at each measurement. Measurement code uses the
provided dedicated stream; it never receives or iterates a dataloader.

`assert_bitwise_alignment(run_fn, steps, seed)` calls the user's unchanged
CPU loop twice with `with_hook=False/True`, identical seeds and deterministic
algorithms. `run_fn` returns `(model.state_dict(), optimizer.state_dict())`.
The helper compares tensor bytes (including optimizer state), and restores
caller RNG/determinism settings. A toy Adam/dropout/BN loop passes; a raw hook
that draws from global RNG fails. `MeasurementBuffer(m=64)` clones/detaches
only batches already supplied by the loop and retains their last m samples.

### Upstream sources

Pins, URLs, declared licenses, intended imports and fetch status are recorded
in `third_party/SOURCES.md`. `scripts/fetch_third_party.sh` is strict and
idempotent: clone, obtain/check the exact commit, detached checkout, then check
and apply each sorted patch (already-applied patches are recognized). All three
new clones are gitignored. **No upstream code was modified and no patches were
created.** The fetch attempt failed at GitHub DNS resolution; upstream licenses,
imports, entrypoint help/one-step smoke and actual imported loss equality remain
pending. Tests skip explicitly per absent clone. Once present, tests verify
pins/licenses, patch application on a clean disposable clone, imports and
native entrypoint help or network one-step smoke. PINN was not cloned:
**blocked: no license, asked supervisor**, as required by the task.

### Verification

Final command: `.venv/bin/python -m pytest -rA`.
**174 passed, 11 skipped, 0 failures; 185 collected; 126.22 seconds.**
Ten skips are upstream checks blocked by absent clones; the remaining skip is
the existing optional Matplotlib diagnostics test. All existing VNA/Stage A
cases pass. `git diff --check` and `bash -n scripts/fetch_third_party.sh` pass.
No checkpoint measurements or setting-specific training results are claimed.

Coverage includes float64 analytic gradients and
finite-difference Hessians for all four losses; MSE reduction plumbing; all
eight Phase 0 ResNet-18 blocks' K, K_E, V, rho and d equivalence (1e-8 relative);
MLP/CNN and WRN pre-activation adjoints/finite differences; LoP 5-unit regression
and 3x2000 MLP shapes; B=0 output, modes, buffers, module graph, parameter-count
and RNG invariance; multi-site kernel addition and loss-generic joint equality;
positive/negative bitwise alignment and exception restoration; and the complete
existing test suite. Native Phase 0 equivalence uses four shared CG iterations
to bound CPU cost; it compares the same computed direction, without asserting
solver convergence as a training/measurement result.

### Created/modified files

Created:

- `methods/vna_losses.py`
- `methods/vna_sites.py`
- `diagnostics/vna_align.py`
- `scripts/fetch_third_party.sh`
- `third_party/SOURCES.md`
- `tests/test_vna_stage_b_losses.py`
- `tests/test_vna_stage_b_sites.py`
- `tests/test_vna_stage_b_align.py`
- `tests/test_vna_stage_b_upstream.py`
- `docs/vna/STAGE_AB_IMPLEMENTATION.md`

Modified:

- `.gitignore`
- `methods/vna.py`
- `methods/vna_ggn.py`
- `methods/vna_joint.py`
- `methods/vna_select.py`
- `diagnostics/signal_curve.py`
- `diagnostics/stage_a.py`

The pre-existing `.DS_Store` change was left untouched. Temporary test logs are
removed after verification; test scratch artifacts use pytest-managed temp
paths, and upstream subprocess fixtures use automatic temporary-directory cleanup.
