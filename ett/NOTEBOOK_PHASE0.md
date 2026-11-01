# ETT Phase0 Kaggle notebook

Open notebooks/ett_phase0_scr_pmnist_s0.ipynb. Select CPU (accelerator None) and
enable Internet. If the source GitHub repo is private, add Kaggle Secret
github_token. It is passed via temporary askpass, never embedded in URLs/logs.

Inputs: none required. Optional MNIST raw IDX files (or gzip) can be attached;
all input subfolders are scanned, or DATA_DIR can point to an explicit tree.
Absent input is downloaded by torchvision. SCR is generated from the pinned
upstream target/data protocol. No CIFAR/ImageNet/Gram/old baseline/source ZIP.

Fresh BP: SCR1e6 and PM20×60000 updates, h1/10/100,20 anchors each. Scientific
settings match registered configs; only local native-reuse paths are removed.
The fresh evenly spaced anchor grid differs from the previous reused-anchor grid.
Upstream is pinned to a6b79580d85f3025bdb601566d3627c5f489f13b and recorded in
protocol.json. This is PM-small/SCR Phase0, not a full Nature-score reproduction.

Run All. No pytest or smoke cell. Two CPU processes (SCR/PM), each one BLAS thread;
the four branches at each anchor execute serially to isolate global RNG. Every
branch uses the same data/RNG and training budget. Runtime packages are isolated
Python3.10, Torch2.2.2+cpu, torchvision0.17.2+cpu, NumPy1.26.4, matplotlib3.8.4.
No GPU is required or used. Exact Kaggle installation/runtime has not been run.

Logs stream live; heartbeat/status every30 seconds; archive every5 minutes and
at completion.11-hour deadline gives exit75 with exact native/case resume; no
horizon truncation. For resume, attach this notebook Output and keep source/run tag.
RESUME_FROM optionally selects the exact archive or extracted run folder if several
snapshots exist. Only source/tag/runtime-matching snapshots are restored; invalid
old measurements cannot be used. The archive contains results/configs/checkpoints,
not source virtualenv or raw datasets. Data is regenerated/downloaded deterministically.

Final output: Phase0 gate table (primary net versus native and CBP), secondary
net_post, JSONL/CSV/plots ZIP and full resumable archive. Always stop after Phase0.
The earlier local run did not pass; no Phase1 training is automatically scheduled.

Pinned source:d903da6be00016021ca596a2f5ad31c77fdd09c3.

## CPU installer correction
uv prioritizes extra-index-url above index-url. The CPU index is now the extra
(high-priority) index; PyPI is the default fallback. The earlier inverted flags
made uv see PyPI torch first and reject the +cpu pin. Keep the same CPU versions
and default first-index policy; do not switch to CUDA wheels or unsafe-best-match.
Official index semantics: https://docs.astral.sh/uv/concepts/indexes/

The corrected flags were resolved with uv0.13.6 for Linux x86_64/Python3.10,
without installing Torch or running models/tests. Full Kaggle execution remains
unverified. See phase0/reports/CPU_SETUP_RESOLVER.log. The failed setup leaves no
ready marker, so rerunning the corrected setup cell safely retries installation.
