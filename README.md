# Virtual-NTK Augmentation (VNA)

This repository studies whether virtual neurons provide useful functional directions
without expanding the trained model. The ResNet-18 / CIFAR-100 Phase 0 result is a
negative example (no-go), not evidence of an accuracy improvement. Historical
measurements and the research specifications are retained in [docs/vna](docs/vna).

Phase 0 retains the plain SGD checkpoint recipe and signal-curve diagnostics in
`experiments/train_recipe_checkpoints.py` and `diagnostics/signal_curve.py`.
[The Phase 0 notebook](notebooks/kaggle_vna_signal_curve.ipynb) runs training and
checkpoint diagnostics on separate GPUs with resumable outputs.

Stage A diagnoses existing Phase 0 checkpoints: null controls, damping sweeps,
virtual-neuron selection, joint block corrections, the A5 numerical investigation,
and solver checks. Use [the Stage A notebook](notebooks/kaggle_vna_stage_a.ipynb)
and consult [the Stage A report](docs/vna/STAGE_A_REPORT.md). Stage B context
screening and its preregistered gate are described in
[the Stage A/B specification](docs/vna/STAGE_AB_SPEC.md); `diagnostics/vna_gate.py`
implements the gate logic.

The retained training engine (`experiments/run_vna_training.py` and
`experiments/vna_training_*.py`) supports SGD, NGD, VNA and matched controls,
resumable checkpoints, provenance and cross-part comparisons. VNA methods include
GGN solvers, K-FAC and joint corrections. Stage 2a training is deferred while
Stage A/B diagnostics proceed; the six generated training notebooks are no longer
checked in. Their builder remains available for generating reviewed experiments.

The failed E-driven-O pipeline is archived at tag `archive/e-driven-o-v1` and
branch `archive/old-pipeline`, both at commit `781af55`.

Install the project and run the full retained test suite:

```bash
.venv/bin/python -m pip install -e '.[test,vision,diagnostics]'
.venv/bin/python -m pytest -q
```

Real TINY/Gromo runs additionally use the upstream dependencies documented in
[third_party/REFERENCES.md](third_party/REFERENCES.md). CPU tests and synthetic
smoke runs do not establish real-checkpoint or GPU results.

Regenerate the retained notebooks with a full immutable commit hash:

```bash
.venv/bin/python scripts/build_kaggle_vna_signal_curve_notebook.py --commit FULL_HASH
.venv/bin/python scripts/build_kaggle_vna_stage_a_notebook.py --commit FULL_HASH
```

Generate the six training notebooks outside the tracked notebook directory:

```bash
.venv/bin/python scripts/build_kaggle_vna_training_notebooks.py --commit FULL_HASH --output-dir /tmp/vna-training-notebooks
```

Kaggle notebooks require CIFAR-100, T4×2, Internet and the `github_token` secret.
Stage A also requires extracted Phase 0 recipe checkpoints. Review the notebook
configuration and pinned revision before running; retain full output archives to
resume a session.
