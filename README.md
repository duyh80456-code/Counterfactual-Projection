# Virtual-NTK Augmentation (VNA)

The separate [standard continual-learning implementation](docs/vna/standard_cl/README.md)
adds the H1 replay-aware numerical core and BN pilot runner, with isolated
H2/H3/H4 research paths. This delivery is **code only, not executed or validated**;
see its audit, mathematical specification, protocols and evidence status.
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
