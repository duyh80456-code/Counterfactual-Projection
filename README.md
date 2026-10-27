# Virtual-NTK Augmentation (VNA)

The separate [standard continual-learning implementation](docs/vna/standard_cl/README.md)
provides independent H1 replay-aware, H2 feature-geometry, H3 plasticity and
H4 Domain-IL runners. CPU numerical and synthetic pipeline checks include
checkpoint/resume and a genuine PyCIL ResNet32 BN forward/backward/correction.
See the [completion report](docs/vna/STANDARD_CL_COMPLETION_REPORT.md) for
executed checks and limits. Real benchmark results, GPU compatibility and
all superiority claims remain **UNVERIFIED**.
Historical runners and pinned notebooks below remain unchanged.

VNA uses T26–T28 trust-region corrections and protected GIR swaps without
expanding the trained model. Main entrypoints are `experiments.main_cil`,
`experiments.v4_competitors`, `experiments.v4_b5_width`, and
`experiments.pilot_v3`. Native PyCIL/LoP runners live in
`experiments/runners/`; shared atomic writers live in `experiments/io_utils.py`.

Install the project and pinned upstream dependencies:

```bash
.venv/bin/python -m pip install -e '.[test,vision,diagnostics]' -r requirements-upstream.txt
bash scripts/fetch_third_party.sh
```

Pins and source limitations are in [third_party/SOURCES.md](third_party/SOURCES.md).
The [main-run specification](docs/vna/VNA_agent_prompt_main_run.md),
[pre-run report](docs/vna/V4_PRERUN_REPORT.md),
[cleanup inventory](docs/vna/CLEANUP.md), and
[historical pilot summary](docs/vna/pilot_v3_summary.md) document the current setup.

Generate M1–M3, C1–C5 and B5w with one shared immutable commit:

```bash
.venv/bin/python scripts/build_kaggle_vna_parallel.py --commit FULL_HASH --variants
```

The nine `notebooks/vna4_*.ipynb` files currently use `PLACEHOLDER` for supervisor
repinning. M1–M3/C1–C5 use CIFAR-100 and T4×2; B5w uses CPU workers. Kaggle
requires Internet and the `github_token` secret. Attach previous output archives
for exact checkpoint resume. Synthetic local checks do not establish GPU or
real-data results. Older in-flight Kaggle runs retain their independently pinned
sources; their local builders and notebooks have been removed.
