# Baseline provenance and comparison scope — 2026-10-28

* ER/iCaRL: existing pinned PyCIL `f3509b8ca3f20660ce4aa13f19d5283de81b4b35`.
  Native head expansion, herding, NME and iCaRL CE+KD reduction are retained in
  the pooled objective. The explicitly selected separate-mean objective is a
  separate experiment, not an upstream reproduction.
* A-GEM: incoming CE gradient is projected onto the replay-gradient halfspace
  only on negative inner product; zero reference gradient is safe. The update
  law was checked against [Mammoth A-GEM](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/models/agem.py).
* ER-ACE: incoming logits mask present and not-yet-seen classes, replay CE stays
  unmasked; the two gradients are summed. Checked against [Mammoth ER-ACE](https://github.com/aimagelab/mammoth/blob/e75a491c69fd729edeb01431afb753d9157d9a81/models/er_ace.py).
  This adapter uses the declared expanding linear PyCIL head, corresponding
  to disabling Mammoth's optional custom cosine classifier. Neither adapter
  is advertised as an official Mammoth experiment reproduction.

A-GEM/ER-ACE adapters train on incoming-only loaders and separately sampled old
herded exemplars. The dedicated replay RNG state is checkpointed as a tensor.
Task-boundary herding differs from original online memory policies. Incoming
then replay forwards each update native BN buffers; this differs from H1's
single pooled-forward decomposition. Native momentum and weight decay follow
any gradient projection. Accordingly the adaptation names/protocols remain
visible and their performance cannot isolate virtual-kernel benefits. H1/H2
native-kernel controls supply the matched kernel comparison instead.

Plasticity uses the existing pinned LoP commit
`a6b79580d85f3025bdb601566d3627c5f489f13b`. BP/CBP invoke upstream `learn`;
ReDo invokes upstream `GnTredo.gen_and_test` on supervised activation history
after BP's SGD, with default threshold0.01/reset_period1000. The smoke test
sets reset_period1 only in the isolated reset unit test. S&P invokes upstream
Backprop perturbation with its cfg's weight decay. No upstream files change.
Effective algorithm state (including CBP utilities/ages and ReDo period counter)
is saved alongside optimizer and global RNG. Registered default settings are
not claimed to be best-tuned baselines. Native LoP protocols run on CPU.

## Authentic LPR: BLOCKED comparison

The [official LPR repository](https://github.com/plai-group/LPR), remote HEAD
`76c44f8f2481f4e23881b2463eb58aaf94affabb`, specifies a separate Python3.10
Hydra/Avalanche-based experiment stack. Its source entry point is
`experiments/main.py`, enabled with `strategy.lpr=true` and its own benchmark,
model and replay configuration. The present environment has no Hydra,
OmegaConf, Avalanche or wandb installed and no pinned LPR checkout. An authentic
compatible adapter is not integrated or tested. No simplified H2 implementation
is renamed LPR. This comparison remains incomplete; do not merge its future
results until backbone, stream, sampler and optimizer differences are audited.
