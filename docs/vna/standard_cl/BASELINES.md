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

## LPR official callback adapter — added for seed-zero notebooks

Official source pin: `76c44f8f2481f4e23881b2463eb58aaf94affabb`.
`methods/cl_lpr.py` verifies both Git HEAD and the official file SHA256 before
loading its definitions. Only the unused Avalanche base import and disabled
wandb import are adapted; algorithm callback bodies remain unchanged. PyCIL's
identical F.linear classifier is exposed as an nn.Linear view sharing the same
Parameters. Tests verify head output/RNG identity, the source preconditioner,
two-task training, preconditioner/optimizer/RNG resume and paired collection.

The earlier full-stack dependency block is superseded for this callback adapter;
Hydra/Avalanche training-stack reproduction remains outside scope. The baseline
runs under our shared ER/herding/expanding-head protocol. See
[seed-zero notebook registration](../STANDARD_COMPARE_S0.md) for options,
limitations, inputs and executable notebooks. No published result is a gate.

### Superseding main notebook setup — 2026-10-28

The offline PyCIL adapter above remains a separate experiment. Main H2 now uses
original online LPR with actual Avalanche0.4 replay/class balancing, SlimResNet18,
minibatch10×3 repeats and5% validation, with README omega0=4,beta=2,every_iter30.
Main H3 uses H3-small Continual ImageNet32 (100 binary tasks,50epochs), keeping
native CBP parameters and the separate5000×250 config. Main H4 is20 rotated MNIST domains.
All four main notebooks use two GPUs. See the current registration, including
compatibility overrides, supported H3 sites and absence of benchmark evidence.
