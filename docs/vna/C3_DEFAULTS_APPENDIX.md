# Appendix: C3 original-source defaults without selection

The same table is included in the LaTeX appendix
`theory/a2_c3_defaults.tex`, included by `vna_gir_theory.tex`.

## Protocol

C1 is retired. C3 runs CBP, ReDo, S&P and UPGD on all seeds {0,1,2}:
12 full 20-task noBN/superclass iCaRL runs, two isolated T4 lanes. There is
no hyperparameter search and no dependency on C1 or best_config.json.

User-confirmed learning rate: use the common iCaRL/PyCIL default 0.1 multiplied
by 0.5, giving **0.05 for every method** in both initial and later tasks, with
the registered milestones/decays. This is not half of each method's original
RL/Adam/UPGD learning rate. The shared iCaRL model, memory, loss, training
horizon and evaluation protocol remain the controlled benchmark adaptation.
Original papers use different tasks and have multiple settings; the specific
source setting adopted here is identified below, rather than claiming a
universal default across all experiments in each paper.

## Frozen values and authority

| Arm | Frozen method parameters | Original source setting |
|---|---|---|
| `icarl_cbp` | `replacement_rate=1e-5`, `maturity_threshold=1000`, `util_type=contribution`, `decay_rate=0.99` | Authors' incremental CIFAR configuration and experiment constructor |
| `icarl_redo` | `tau=0.025`, `reset_steps=1000`, `score_batch_size=64` | Original paper Appendix A, Table 1, DQN **default setting** |
| `icarl_snp` | `shrink=0.4`, `sigma=0.1` (upstream name `perturb`) | Authors' warm-start `run.py` argument defaults |
| `icarl_upgd` | `beta_utility=0.999`, `sigma=0.001`, `weight_decay=0.001` | Authors' README recommended UPGD constructor |

- **CBP**, pinned `a6b79580d85f3025bdb601566d3627c5f489f13b`:
  [incremental CIFAR config](https://github.com/shibhansh/loss-of-plasticity/blob/a6b79580d85f3025bdb601566d3627c5f489f13b/lop/incremental_cifar/cfg/continual_backpropagation.json),
  [experiment constructor](https://github.com/shibhansh/loss-of-plasticity/blob/a6b79580d85f3025bdb601566d3627c5f489f13b/lop/incremental_cifar/incremental_cifar_experiment.py).
  These replace the adapter's old maturity100/weight-utility behavior. Use the
  native ResGnT contribution utility; replacement decisions are per SGD update.
- **ReDo**: [Sokar et al., ICML 2023, Appendix A Table 1](https://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf).
  The table distinguishes tau0.025 for the default DQN setting and tau0.1 for
  other settings; C3 fixes the former, without selecting between them. It also
  lists recycling period1000 and score batch64. This is the published DQN
  setting, not the upstream BaseRecycler constructor's SAC-like period200000.
  In this supervised port, the period counts actual optimizer updates. Use up
  to64 examples from the current training minibatch for the activation score.
  Keep the existing NCHW recycling implementation (Xavier incoming weights,
  zero outgoing weights, reset affected optimizer state); persist both update
  and last-recycle counters with the learner across tasks and native resumes.
  The existing score uses pre-update activations of the current minibatch.
- **S&P**, pinned `896176b2cf89f259bcde8e241a8987e6019741d0`:
  [run.py](https://github.com/JordanAsh/warm_start/blob/896176b2cf89f259bcde8e241a8987e6019741d0/run.py),
  [README example](https://github.com/JordanAsh/warm_start/blob/896176b2cf89f259bcde8e241a8987e6019741d0/README.md).
  sigma scales a fresh native initialization, not unit Gaussian noise. Preserve
  this adapter's per-task intervention timing and the expanded-head device fix;
  this is an iCaRL adaptation, not a reproduction of the original Adam run.
- **UPGD**, pinned `b75e90ad4b09c28971ac9dbb902a8fd86709b28c`:
  [authors' recommended defaults](https://github.com/mohmdelsayed/upgd/blob/b75e90ad4b09c28971ac9dbb902a8fd86709b28c/README.md),
  [actual native implementation](https://github.com/mohmdelsayed/upgd/blob/b75e90ad4b09c28971ac9dbb902a8fd86709b28c/core/optim/weight_upgd/first_order.py).
  Explicitly pass the README's recommended utility/noise/decay values because
  the native FirstOrderGlobalUPGD signature instead defaults to beta0/sigma1.
  Keep that pinned native class, including its update formula; the README
  snippet has a different factor in its update and is not copied into the
  adapter. The common lr override is0.05; SGD momentum is not added to UPGD.

All four runs write the exact dictionary above in protocol.method_config and
summary.identity, with parameter_policy=`original_defaults_no_search_v1` and
pinned implementation provenance. Old C1-selected results are excluded from
this protocol, even when some individual numbers happen to match. Restarting
an old tuned checkpoint with new defaults would change its experiment; native
resume requires the complete new identity to agree instead of relabeling it.

## Inputs and aggregation

C3 needs only CIFAR-100. Attach a previous output from this exact C3 version to
resume its native task-boundary checkpoints. No source ZIP is required.
Aggregate C3 directly with M1/C2 outputs using scripts.collect_cil_main; do not
supply C1 winners as C3 seed0. C4 uses the same method dictionaries on BN and
also no longer needs C1 selections; it remains deferred.

Before constructing any method state, the hook moves the network to the native
training device. This also puts CBP utility/age state on the GPU where updates
run, and retains the S&P expanded-head device correction. Optimizers are created
after this move.

No real GPU training or new runtime estimate was produced for this change.
