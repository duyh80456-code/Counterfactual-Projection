# Stage B core: loss-generic GGN, hook-based sites, alignment harness, upstream sources

Branch `vna-phase0`. `.git` read-only: do NOT commit. `.venv/bin/python`. All existing
`tests/test_vna*.py` (incl. Stage A) must pass with unchanged numbers.
Read: `docs/vna/PROMPT_STAGE_AB.md` (§2–§3, §6–§7), `docs/vna/STAGE_AB_SPEC.md` (B0, B2, B3, B5, tests),
`docs/vna/TASK_STAGE_A.md` §0.

This task builds only the shared infrastructure. No setting-specific runners yet.

## 1. Loss interface (`methods/vna_losses.py`)

`LossSpec` objects with:
- `residual(z, ctx)`: g = ∂L/∂z by **autograd on the given upstream loss callable** (`ctx` carries
  targets, teacher logits, old-class count…). Returns the per-sample rows with the same 1/n convention
  as `vna_ggn.py` (document it in the docstring).
- `metric_apply(z, v, ctx)`: H·v per sample, the true GGN/Fisher of L w.r.t. z (never empirical Fisher).
- `name`, `output_dim`.

Implementations and their H (per sample, scaled consistently with `residual`):
- `CE`: diag(p) − ppᵀ.
- `MSE`: 2·I for per-sample squared error averaged over the batch (verify against the actual
  reduction of the upstream loss passed in; the FD test is authoritative).
- `KDRepDistiller(alpha=0.1, beta=0.9, T=4)`: L = α·CE(z,y) + β·T²·KL(softmax(z_t/T) ‖ softmax(z/T));
  H = α(diag(p) − ppᵀ) + β(diag(p_T) − p_T p_Tᵀ) with p_T = softmax(**z_student**/T).
- `ICaRLPyCIL(T=2, known_classes)`: CE over seen classes + PyCIL `_KD_loss` on `z[:, :known]`
  (no T²); H = (diag(p) − ppᵀ) + (1/T²)(diag(q_T) − q_T q_Tᵀ) on the old-class block,
  q_T = softmax(z[:, :known]/T). First task: CE only.
Test (T-loss, spec tests 7 and 12-H): for each spec, autograd g equals the analytic formula and
`metric_apply` equals the finite-difference Hessian of L w.r.t. z (float64, ≤ 1e-5), on random data.
For KD/iCaRL write the reference loss in the test by copying the upstream formula verbatim (with a
source comment); once `third_party/` is present (§4), add a test that imports the upstream loss and
checks equality.

Refactor `methods/vna_ggn.py::GGNProducts` (and every caller that hardcodes `_mean_ce_logit_grad`) to
take a `LossSpec`, default `CE`, so that CE results are bit-identical to before (existing tests prove it).

## 2. Hook-based generic site (`methods/vna_sites.py`)

`HookSite(model, first: str, second: str, kind)` declares a site as (first layer → [norm] →
nonlinearity → second layer) by module path, with `kind ∈ {"linear", "conv"}` and an explicit
`between` description (which norm/activation sit between). It must give `vna_block_correction`,
`block_products`, `tiny_V`, `tiny_U_from_cotangent`, `normalize_A`, `joint_correction` everything they
currently get from the Gromo `candidate_projection_block` (virtual A layer, virtual B=0 layer whose
output is added to the second layer's output via a forward hook, O-parameter scope, suffix forward).
Never install anything permanently: hooks are context managers; on exit the module graph, params,
buffers, mode and parameter count are identical.
Required tests:
- On the Phase 0 ResNet-18 (fake data, float64), a `HookSite` declared on each basic block gives the
  same K·v, K_E·v, V, ρ, d (≤ 1e-8 rel) as the existing Gromo path. This proves equivalence.
- MLP (Linear→ReLU→Linear, also the LoP `deep_ffnn`/`fix_ltu_net` shapes) and small CNN: adjoint /
  finite-difference test (pattern `test_extension_finite_difference_adjoint_and_native_alignment`);
  output unchanged with B=0; buffers unchanged.
- Pre-activation (WRN): site conv1 → (BN, ReLU) → conv2 in the correct order.
Keep the Gromo path working (Phase 0/Stage A use it); `HookSite` is an alternative resolver.

## 3. Alignment harness (`diagnostics/vna_align.py` + tests)

- `MeasurementHook`: called with (model, batch_the_loop_already_has, loss_spec, ctx) at declared
  measurement points. It saves/restores params, buffers, train/eval mode, global CPU/CUDA RNG state,
  uses only `dedicated_generator` streams, never pulls from the upstream dataloader, asserts the
  parameter count is unchanged (T-params).
- `assert_bitwise_alignment(run_fn, steps, seed)`: runs a user-provided short training loop on CPU
  (deterministic algorithms on) twice, with and without the hook, and compares final state_dicts and
  optimizer states bit-for-bit. Unit-test it with a toy loop, plus a negative test where a hook that
  draws from global RNG makes it fail.
- Measurement data rule (spec §3.5) as a helper: `MeasurementBuffer` sliding window of the last m
  samples the loop has already seen (m declared in config, default 64); never test/future data.

## 4. Upstream sources (`third_party/`)

Clone at a pinned commit, do not vendor-edit:
| name | URL | commit | license |
|---|---|---|---|
| loss-of-plasticity | github.com/shibhansh/loss-of-plasticity | a6b79580d85f3025bdb601566d3627c5f489f13b | MIT |
| PyCIL | github.com/LAMDA-CL/PyCIL (moved from G-U-N/PyCIL) | f3509b8ca3f20660ce4aa13f19d5283de81b4b35 | MIT |
| RepDistiller | github.com/HobbitLong/RepDistiller | b84f547c5db6a35318d4671d7d5c4de74c822403 | BSD-2-Clause |
- `scripts/fetch_third_party.sh`: idempotent clone + checkout of the pinned commit + `git apply` of
  every `third_party/patches/<name>/*.patch`; fails loudly on any error. Clones are gitignored;
  `third_party/SOURCES.md` (URL, commit, license, what we use, patches with one-line reasons) is tracked.
- Do not modify upstream code in this task. If something can only be done by patching, write the patch
  file + reason and list it in your final summary for review.
- Test (T-upstream, skipped with a clear reason if the clone is absent): patches apply cleanly; each
  upstream entry point we will use imports, and `--help` (or a 1-step smoke) runs.
- The PINN repo (pratikrathore8/opt_for_pinns) has no LICENSE file: do NOT clone it; note it in
  SOURCES.md as "blocked: no license, asked supervisor".

## Report

Append "Stage B core" to a new `docs/vna/STAGE_AB_IMPLEMENTATION.md` (what each module does, the 1/n
convention per loss, how a site is declared). Run all tests; report counts. List every created/modified
file. Remove your own /tmp outputs.
