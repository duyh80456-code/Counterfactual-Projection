# Stage 1 task: VNA core math + Phase 0 diagnostics (no controller, no training arms)

Full spec: `docs/vna/SPEC.md`. Read it entirely first. This stage implements ONLY the parts listed
below. Do NOT implement `experiments/run_vna_training.py`, the controller gates (G1/G2/G3, accept test,
stop), or any notebook in this stage. Do not modify existing runners or existing tests.

Work on the current git branch `vna-phase0`. Commit locally in small logical commits. Do NOT push.

## Step 0 — environment

- System `/usr/bin/python3` is 3.9 and has no Gromo. Create `.venv` (already git-ignored) with
  `~/.pyenv/versions/3.12.7/bin/python -m venv .venv`, then install CPU torch + torchvision, pytest,
  numpy, and Gromo pinned at `git+https://github.com/growingnet/gromo@8d19107b61a9459a9021065a329b699adcb0f25b`
  (check how existing notebooks install Gromo and mirror that exactly).
- Run the existing suite with `.venv/bin/python -m pytest -q` and record pass/fail counts
  (before your changes) in `docs/vna/STAGE1_REPORT.md`. Do not fix unrelated pre-existing failures.

## Step 1 — Gromo alignment audit (SPEC §0.2) — STOP-AND-REPORT

Read the pinned Gromo source and `third_party/One-Shot-TAS-CCIL/dual_growth/adapters/tiny_adapter.py`,
`probe/counterfactual_tiny.py`, `probe/gromo_adapter.py`. Answer each with file:line evidence in
`docs/vna/STAGE1_REPORT.md` and a short note in `third_party/REFERENCES.md`:

1. For a CIFAR-ResNet18 Gromo BasicBlock, exactly which tensor is A (incoming) and which is B
   (outgoing) of the virtual neurons; what non-linearity / BN (if any) is applied to the r virtual
   hidden channels in the extended forward. Write the exact extended-forward formula.
2. How Gromo normalizes its statistics (S, M tensors): divided by number of samples? by spatial
   positions? sum-reduction CE? Consequently: does `tiny_eigenvalues` scale with batch size or not?
   Verify empirically: compute λ_1 for the same model on 256 vs 1024 samples and report.
3. What `payload["tiny_eigenvalues"]` contains exactly (singular values? squared? before/after
   thresholding?) and whether `effective_rank` truncation affects it.
4. Whether statistics collection mutates BN running buffers and whether the transactional wrapper
   restores them.

If anything contradicts SPEC §0.2/§1 (e.g. there is BN on virtual channels so "B·σ(A·x)" is not the
actual form), DO NOT adapt silently: record it under "DISCREPANCIES" in the report, implement the
form that matches the actual Gromo extended forward, and state precisely what differs.

## Step 2 — `methods/vna.py` core math (SPEC §1, §1.1)

Implement (no controller logic yet):
- `VNAConfig` dataclass with all SPEC §3 defaults (controller fields included but unused).
- A function to extract A (and rank) from a counterfactual TINY candidate, and to build a random-A
  control of matched rank and Frobenius norm using a dedicated `torch.Generator`.
- J_E products w.r.t. the outgoing extension weight (+bias) at value 0, gate = 1, via
  `torch.func.functional_call` + `jvp`/`vjp`; J/Jᵀ products w.r.t. the `residual_path` scope from
  `candidate_projection_parameter_names`. Reuse `_preserve_functional_state` and the snapshot/assert
  helpers. eval-mode BN by default; `bn_mode="train"` option that uses batch statistics with COPIED
  buffers so running buffers are never updated.
- `vna_block_correction(model, site, A, batch, cfg, warm_start=None) -> (grad_delta, diag)`:
  the exact w,z,s,q,d chain of SPEC §1, μ = damping_rel·tr(K)/n with 8 Hutchinson probes (dedicated
  generator; allow passing a cached μ/Jacobi diagonal), PCG with fp64 vectors, warm start, cg_iters,
  cg_tol. diag must include U, V, η, ρ, per-neuron V_k, ‖Jᵀg‖², CG iterations/residuals/converged for
  both solves, non-finite flag, wall time, and counts of JVP and VJP calls (network + extension).
  Non-finite → return empty grad_delta and flag, never raise mid-training.
- No AMP inside; fp32 network products, fp64 CG/dots; diagnostics CE in fp64.

## Step 3 — tests `tests/test_vna.py` (SPEC §5 items 1, 2, 3, 4, 7, 10, 11, 12 only)

Use a tiny Gromo ResNet (small width, few classes, 8×8 or 16×16 inputs) so the suite is fast on CPU.
Item 3 needs a model/batch where K is full rank (n ≤ number of block params). Additionally test:
- `bn_mode="train"` leaves BN running buffers bit-identical.
- the exact identity gᵀ(J d) == λ·V (not just ≥ 0) to tight tolerance.
- A dedicated-generator test: calling `vna_block_correction` (and random-A construction) does not
  change `torch.get_rng_state()` nor any passed training DataLoader generator state.

## Step 4 — `experiments/train_recipe_checkpoints.py` (SPEC §4.1)

As specified. Reuse `experiments/shared_protocol.py` helpers (splits, model builder, train/eval,
RNG, atomic save). Save full-state checkpoints every 10 epochs + `latest`; resumable. Add a
`--max-epochs`/`--limit-batches` style smoke option so it can be smoke-tested on CPU with fake data
(add a fast test).

## Step 5 — `diagnostics/signal_curve.py` (SPEC §4.2) + supervisor additions

Everything in SPEC §4.2 (the `--grow-epochs` option may be left as a clearly-marked TODO if it would
require more than ~150 lines; say so in the report). All probe/statistics/VNA batches use the TRAIN
transform with dedicated generators; held-out uses the trigger split.

Supervisor additions (required; they decide Stage 2 design):
a. **Cost**: per site, wall time and JVP/VJP counts of one `vna_block_correction` at defaults
   (vna_batch 128, cg_iters 30, cold and warm start), versus one SGD step at batch 64 on the same
   device. Report the ratio and the implied overhead at vna_every=4 for 1 and 3 sites.
b. **Noise-floor bias**: per site, λ_1 on the full stat batch (1024), mean/std of λ_1 on 4 disjoint
   sub-batches (256), a split-half cross-validated λ_1 (top eigvector from half 1, Rayleigh quotient
   with half-2 statistics), and a shuffled-label null λ_1. Report all four; do not choose one.
c. **ρ distribution**: ρ_s at λ_vna = 1 for every site and checkpoint (already in the CSV) and a
   summary of how often ρ > 0.3.
d. **LR-boost confound**: for each (checkpoint, site), the held-out directional derivative of the
   plain block gradient scaled to the same norm as the VNA correction d (compare with VNA's).

Smoke-test the script end-to-end on CPU with a tiny model + fake data (a test), so it runs without
CIFAR. If CIFAR-100 is present locally use it only for a quick sanity run, otherwise skip.

## Deliverable / reporting

`docs/vna/STAGE1_REPORT.md` containing: env + test counts before/after; alignment audit answers with
file:line evidence; DISCREPANCIES; list of files added; any spec item not done and why; the exact
commands to run 4.1 and 4.2 on a GPU. Run the full suite at the end and report counts (new tests must
all pass; pre-existing failures must be unchanged).

Constraints: new files only (plus `third_party/REFERENCES.md` and `docs/vna/`). Match existing code
style. No rollback/anchor/plateau/apply_/momentum-reset code paths. Don't push.
