# Pinned upstream sources

| Directory | URL | Pinned commit | License | What we use |
|---|---|---|---|---|
| loss-of-plasticity | https://github.com/shibhansh/loss-of-plasticity | a6b79580d85f3025bdb601566d3627c5f489f13b | MIT | Regression, online MNIST and continual ImageNet networks and original training entry points |
| PyCIL | https://github.com/LAMDA-CL/PyCIL (moved from G-U-N/PyCIL) | f3509b8ca3f20660ce4aa13f19d5283de81b4b35 | MIT | models/icarl.py `_KD_loss`, iCaRL and original main.py |
| upgd | https://github.com/mohmdelsayed/upgd | b75e90ad4b09c28971ac9dbb902a8fd86709b28c | MIT | UPGD optimizer for the continual-learning competitor |
| C-Flat | https://github.com/WanNaa/C-Flat | f57a557523136834bdc28a4902cde9842859b2a8 | no LICENSE file | C-Flat iCaRL comparator |
| warm_start | https://github.com/JordanAsh/warm_start | 896176b2cf89f259bcde8e241a8987e6019741d0 | no LICENSE file | Ash–Adams Shrink & Perturb comparator |
| dopamine | https://github.com/google/dopamine | 5873f5494ee0c2d7c016d0ab2ad530354fec59d0 | Apache-2.0 | ReDo dormant score and reset implementation |
| FIRE | https://github.com/isaac7778/FIRE | 3f73d7848c42dbd42e7770cb3aa61b4fd3ffe3e1 | no LICENSE file | vision/interventions/fire.py:fire (round-3 competitor, called unmodified) |
| SWR | https://github.com/JFernando4/collas_2025_swr_paper | 6637c351e5e34d3c7f7f9fe71d26e2c70bc6a2ac | MIT | src/swr_functions/selective_weight_reinitialization.py (round-3 competitor); init table ported to resnet32 |
| MRFA | https://github.com/bwnzheng/MRFA_ICML2024 | 2435c35d7d923e79ca5f564739b4f69963fce117 | no LICENSE file | PyCIL fork with iCaRL-MRFA (round 3; needs a dedicated runner, not wired) |

Run `bash scripts/fetch_third_party.sh`. Clones are ignored; upstream is never
vendor-edited. Patches: **none** in Stage B core. Any future patches go under
`third_party/patches/<name>/`, with a reason recorded here.

## V4 competitors (2026-10-19)

| Directory | URL | Pinned commit | License | What we use |
|---|---|---|---|---|
| upgd | https://github.com/mohmdelsayed/upgd | b75e90ad4b09c28971ac9dbb902a8fd86709b28c | MIT | core/optim/weight_upgd/first_order.py:FirstOrderGlobalUPGD; grids from experiments/label_permuted_{cifar10,emnist}.py:19-20 |
| C-Flat | https://github.com/WanNaa/C-Flat | f57a557523136834bdc28a4902cde9842859b2a8 | No LICENSE file | Original models/icarl.py and optims/c_flat.py; original config/CLI defaults; README.md:20 links resources/cflat_results.png (Table 1: iCaRL+C-Flat, B0 Inc5 = 59.13%) |
| warm_start | https://github.com/JordanAsh/warm_start | 896176b2cf89f259bcde8e241a8987e6019741d0 | No LICENSE file | run.py:126-136 shrink_perturb, ported to the native iCaRL initialization; no source vendoring |
| dopamine | https://github.com/google/dopamine | 5873f5494ee0c2d7c016d0ab2ad530354fec59d0 | Apache-2.0 | dopamine/labs/redo/weight_recyclers.py:estimate_neuron_score, NeuronRecycler recycle/masks, weight_reinit_random/zero, reset_momentum; NCHW port in adapters/v4_competitors.py |

Round 3: FIRE, SWR and MRFA are pinned above (fetched 2026-10-19). FIRE and SWR are wired
(`icarl_fire`, `icarl_swr`); MRFA is a PyCIL fork whose `models/` and `utils/` packages clash
with PyCIL's, so it still needs a dedicated runner. Forget-Forgetting (arXiv:2502.07274) has no
public code and no locally available full equations; its runner entry fails clearly, since
implementing guessed reset/consolidation formulas is prohibited.