# Stage B upstream sources

| Directory | URL | Pinned commit | License | What we use |
|---|---|---|---|---|
| loss-of-plasticity | https://github.com/shibhansh/loss-of-plasticity | a6b79580d85f3025bdb601566d3627c5f489f13b | MIT | Regression, online MNIST and continual ImageNet networks and original training entry points |
| PyCIL | https://github.com/LAMDA-CL/PyCIL (moved from G-U-N/PyCIL) | f3509b8ca3f20660ce4aa13f19d5283de81b4b35 | MIT | models/icarl.py `_KD_loss`, iCaRL and original main.py |
| RepDistiller | https://github.com/HobbitLong/RepDistiller | b84f547c5db6a35318d4671d7d5c4de74c822403 | BSD-2-Clause | distiller_zoo/KD.py `DistillKL`, CIFAR ResNet/WRN and train_student.py |
| upgd | https://github.com/mohmdelsayed/upgd | b75e90ad4b09c28971ac9dbb902a8fd86709b28c | MIT | UPGD optimizer for the continual-learning competitor |
| C-Flat | https://github.com/WanNaa/C-Flat | f57a557523136834bdc28a4902cde9842859b2a8 | no LICENSE file | C-Flat iCaRL comparator |
| warm_start | https://github.com/JordanAsh/warm_start | 896176b2cf89f259bcde8e241a8987e6019741d0 | no LICENSE file | Ash–Adams Shrink & Perturb comparator |
| dopamine | https://github.com/google/dopamine | 5873f5494ee0c2d7c016d0ab2ad530354fec59d0 | Apache-2.0 | ReDo dormant score and reset implementation |

Run `bash scripts/fetch_third_party.sh`. Clones are ignored; upstream is never
vendor-edited. Patches: **none** in Stage B core. Any future patches go under
`third_party/patches/<name>/`, with a reason recorded here.

Fetched and verified on 2026-10-15: each clone is at the pinned commit and its
LICENSE file reads MIT (loss-of-plasticity, PyCIL) and BSD 2-Clause (RepDistiller).
Install `requirements-upstream.txt` (torch, torchvision, numpy, scipy, Pillow,
tqdm, tensorboard_logger, protobuf, six). Pinned entrypoint imports were read
from the real sources. RepDistiller requires
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing its entrypoints;
future runners must set it too. Tests set it in subprocess environments.
Upstream tests fail on absent clones when `REQUIRE_UPSTREAM=1` (notebook default);
without that variable an absent clone is an explicit skip.

PINN https://github.com/pratikrathore8/opt_for_pinns:
**blocked: no license, asked supervisor**. Do not clone.
B6 dropped from this round (supervisor).
