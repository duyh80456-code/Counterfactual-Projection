# Pinned upstream sources

| Directory | URL | Pinned commit | License | What we use |
|---|---|---|---|---|
| loss-of-plasticity | https://github.com/shibhansh/loss-of-plasticity | a6b79580d85f3025bdb601566d3627c5f489f13b | MIT | Regression, online MNIST and continual ImageNet networks and original training entry points |
| PyCIL | https://github.com/LAMDA-CL/PyCIL (moved from G-U-N/PyCIL) | f3509b8ca3f20660ce4aa13f19d5283de81b4b35 | MIT | models/icarl.py `_KD_loss`, iCaRL and original main.py |
| upgd | https://github.com/mohmdelsayed/upgd | b75e90ad4b09c28971ac9dbb902a8fd86709b28c | MIT | UPGD optimizer for the continual-learning competitor |
| C-Flat | https://github.com/WanNaa/C-Flat | f57a557523136834bdc28a4902cde9842859b2a8 | no LICENSE file | C-Flat iCaRL comparator |
| warm_start | https://github.com/JordanAsh/warm_start | 896176b2cf89f259bcde8e241a8987e6019741d0 | no LICENSE file | Ash–Adams Shrink & Perturb comparator |
| dopamine | https://github.com/google/dopamine | 5873f5494ee0c2d7c016d0ab2ad530354fec59d0 | Apache-2.0 | ReDo dormant score and reset implementation |

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

Absent Round 3 code: SWR https://github.com/JFernando4/collas_2025_swr_paper,
MRFA https://github.com/bwnzheng/MRFA_ICML2024. Set VNA_FETCH_ROUND3=1 and
VNA_SWR_PIN / VNA_MRFA_PIN to audited 40-character commits to fetch; no moving
branch is pinned automatically. FIRE project https://isaac7778.github.io/fire/
has no locally verified repository URL. Forget-Forgetting (arXiv:2502.07274)
has no locally available full equations or supplementary code. Their runner
entries fail clearly; implementing guessed reset/consolidation formulas is prohibited.