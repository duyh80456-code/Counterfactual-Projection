# Cleanup: remove the failed E-driven-O pipeline (one commit)

Branch `vna-phase0`. `.git` read-only: do NOT commit (the supervisor commits). `.venv/bin/python`.
Archive already exists: tag `archive/e-driven-o-v1` and branch `archive/old-pipeline` at 781af55.
Stage A files (diagnostics/stage_a.py, vna_gate.py, methods/vna_joint.py, tests/test_vna_stage_a.py,
tests/test_vna_lemma.py, notebooks/kaggle_vna_stage_a.ipynb, its builder) are KEPT and must still pass.

## 1. Move before delete
- Move `candidate_projection_block` and `candidate_projection_parameter_names` (and only the private
  helpers they need) from `methods/e_projection.py` into `methods/vna.py`, verbatim. Update every
  surviving import. Then delete `methods/e_projection.py`.
- `methods/__init__.py`: drop `EProjection`, `ERepOpt`, `RepOptGradientHandler` exports.
- `probe/__init__.py`: drop exports from `pretrained_gromo` and `virtual_expansion`; keep
  `gromo_adapter` and `counterfactual_tiny` (verify they do not import the deleted modules; if they
  do, move the needed symbol rather than keep the old module, and say so).
- `experiments/shared_protocol.py`: remove the `vgg16` / `densenet121` builders, their registry and
  label entries, and anything else that only serves them. Keep the ResNet-18 paths byte-identical.

## 2. Delete
- `adapters/` (whole package, all deit_* files + `__init__.py`), `baselines/` (whole),
  `models/` (whole: only `deit_tiny_cifar.py`), `configs/projection_log_analysis.example.json`.
- `experiments/`: `deit_*.py`, `run_deit_*.py`, `train_deit_plateau.py`, `run_plateau_*.py`,
  `plateau_protocol.py`, `run_vanilla_to_plateau.py`, `run_unified_vanilla_to_stall.py`,
  `run_shared_comparison.py`, `run_gromo_pilot.py`, `run_cifar_pilot.py`.
- `methods/e_repopt.py`, `methods/vanilla.py`, `methods/e_projection.py` (after §1).
- `diagnostics/`: `deit_*.py`, `persistent_growth_probe.py`, `site_projection_probe.py`,
  `summarize_capacity.py`, `protocol.py` (only used by the deleted probes).
- `probe/virtual_expansion.py`, `probe/pretrained_gromo.py`, `probe/vgg_tiny_adapter.py`,
  `probe/densenet_auxiliary.py`.
- `projection/`: `auxiliary_space.py`, `transfer_operator.py`, `exact_reference.py`,
  `margin_diagnostics.py`, `metrics.py` (keep `cg.py`, `functional.py`, fix `projection/__init__.py`).
- `scripts/analyze_projection_diagnostic.py` and every `scripts/build_kaggle_*.py` EXCEPT
  `build_kaggle_vna_signal_curve_notebook.py`, `build_kaggle_vna_training_notebooks.py`,
  `build_kaggle_vna_stage_a_notebook.py`.
- `notebooks/`: every `kaggle_*.ipynb` EXCEPT `kaggle_vna_signal_curve.ipynb` and
  `kaggle_vna_stage_a.ipynb`. This includes the six `kaggle_vna_seed{0,1}_part{1,2,3}.ipynb`
  (ResNet-18 Stage 2a). Keep the training engine (`run_vna_training.py`, `vna_training_*.py`,
  `vna_kfac.py`, `vna_solver_validation.py`, `vna_provenance.py`) and its builder; any VNA test that
  reads the deleted notebook files must instead build the notebook into `tmp_path` and check that.
- `tests/`: `test_deit_*`, `test_plateau_*`, `test_*_notebook.py` (except VNA ones),
  `test_vgg16_protocol.py`, `test_densenet121_protocol.py`, `test_resnet34_protocol.py`,
  `test_end_to_end.py`, `test_shared_comparison.py`, `test_structural_e.py`,
  `test_research_projection_v2.py`, `test_pretrained_gromo.py`, `test_projection_diagnostic.py`,
  `test_pilot_protocol.py`, `test_convergence_*.py`, `test_four_arm_notebook.py`,
  `test_unified_seed_notebooks.py`, `test_adaptive_notebook.py`, `test_methods.py`, `test_probe.py`,
  `test_gromo_integration.py` (all three exercise only EProjection/ERepOpt/VirtualExpansionProbe).

Do NOT delete: `experiments/train_recipe_checkpoints.py`, `diagnostics/signal_curve.py`,
`experiments/vna_*.py`, `experiments/shared_protocol.py`, `experiments/kaggle_checkpoint_discovery.py`,
`probe/gromo_adapter.py`, `probe/counterfactual_tiny.py`, `projection/cg.py`,
`projection/functional.py`, `training/`, `third_party/`, `tests/conftest.py`, `tests/test_cg.py`,
`tests/test_functional.py`, `tests/test_kaggle_checkpoint_discovery.py`, all `tests/test_vna*.py`.
If any of these imports something on the delete list, stop and report instead of deleting.

## 3. Verify
- After deleting, grep the whole repo (excluding third_party) for every deleted module name; there
  must be no remaining import or `-m` / path reference (docs under `docs/` may mention them historically).
- Run the full test suite (`pytest -q`); report counts. All VNA/Stage A tests pass.
- Update `README.md`: describe the repo as the VNA project (Phase 0 negative example, Stage A/B
  diagnostics, training engine), mention the archive tag/branch for the old pipeline, and remove
  instructions for deleted scripts.
- Final summary: list of deleted files (count), moved symbols, edited files.
