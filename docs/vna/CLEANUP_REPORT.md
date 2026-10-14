# VNA cleanup report

Implemented `TASK_CLEANUP.md`, including the authorised §4 dependency moves.
No commit or Git metadata write was made. The pre-existing `.DS_Store` and
`docs/vna/TASK_CLEANUP.md` changes were retained.

## Moved definitions

All definitions were moved verbatim:

- `candidate_projection_block`, `candidate_projection_parameter_names`, and
  their sole private helper `_module_path`: `methods/e_projection.py` → `methods/vna.py`.
- `cosine_alignment`, `fitted_norm_ratio`, and `relative_residual`:
  `projection/metrics.py` → `projection/functional.py`.
- `ProbeSignal` (with its `dataclass` import): `probe/virtual_expansion.py` →
  `probe/gromo_adapter.py`. This was the retained probe dependency on a deleted module;
  `probe/counterfactual_tiny.py` needs no move or edit.

## Other edits

Removed obsolete package exports, VGG16/DenseNet121 builders and registry/label
entries, five pilot-helper tests, and the three DeiT fixtures. Retained every
projection-only functional test. No surviving test uses the removed fixtures.
ResNet-18 paths in `shared_protocol.py` remain byte-identical.

The six-notebook VNA test now builds and reads notebooks in `tmp_path`.
The Phase 0 builder's setup cells are self-contained because its former input
notebook was deleted. All eight generated VNA notebook outputs were compared
against the original builders and are exactly equal for the same immutable pin.
Both retained notebooks, the Stage A builder, the training notebook builder,
training engine, and protected Stage A files are unchanged.

README now describes VNA, the Phase 0 negative example, Stage A/B diagnostics,
the retained training engine, and the old pipeline's archive tag/branch.
Setuptools package discovery no longer names deleted packages.

## Verification

`.venv/bin/python -m pytest -q` completed successfully: **143 passed, 1 skipped,
0 failures** (144 collected). All 23 Stage A cases and the lemma test passed.
The sole skip is the existing optional Matplotlib plotting test in
`tests/test_vna_split3.py` because the plotting dependency is unavailable.
Full test output: `/tmp/vna_cleanup_pytest.log`.

The repository reference scan includes hidden text files and excludes Git metadata,
the virtual environment, third-party sources, and historical mentions under `docs/`.
It found no surviving references to deleted module paths or dotted module names.
`git diff --check` passed. The 167-entry deletion manifest matches Git's tracked
file deletions exactly.

## Edited files

- `README.md`
- `experiments/shared_protocol.py`
- `methods/__init__.py`
- `methods/vna.py`
- `probe/__init__.py`
- `probe/gromo_adapter.py`
- `projection/__init__.py`
- `projection/functional.py`
- `pyproject.toml`
- `scripts/build_kaggle_vna_signal_curve_notebook.py`
- `tests/conftest.py`
- `tests/test_functional.py`
- `tests/test_vna_split3.py`
- `docs/vna/CLEANUP_REPORT.md`

## Deleted files (167)

### adapters (10)

- `adapters/__init__.py`
- `adapters/deit_ablation.py`
- `adapters/deit_bypass.py`
- `adapters/deit_cp_adapter.py`
- `adapters/deit_expansion_generators.py`
- `adapters/deit_mlp_growth.py`
- `adapters/deit_opt_e.py`
- `adapters/deit_persistent_growth.py`
- `adapters/deit_projection_aware.py`
- `adapters/deit_random_control.py`

### baselines (11)

- `baselines/__init__.py`
- `baselines/bypass/__init__.py`
- `baselines/bypass/extension.py`
- `baselines/bypass/optimizer.py`
- `baselines/bypass/projection.py`
- `baselines/official_common.py`
- `baselines/run_bypass.py`
- `baselines/run_expandnets.py`
- `baselines/run_repan.py`
- `baselines/run_repoptimizer.py`
- `baselines/structural.py`

### configs (1)

- `configs/projection_log_analysis.example.json`

### diagnostics (7)

- `diagnostics/deit_expansion_proposals.py`
- `diagnostics/deit_expansion_suite.py`
- `diagnostics/deit_projection_v2.py`
- `diagnostics/persistent_growth_probe.py`
- `diagnostics/protocol.py`
- `diagnostics/site_projection_probe.py`
- `diagnostics/summarize_capacity.py`

### experiments (26)

- `experiments/deit_ablation_runner.py`
- `experiments/deit_bypass_comparison.py`
- `experiments/deit_bypass_resume.py`
- `experiments/deit_e_rollback.py`
- `experiments/deit_logging.py`
- `experiments/deit_projection_resume.py`
- `experiments/deit_protocol.py`
- `experiments/deit_research_resume.py`
- `experiments/deit_resume.py`
- `experiments/deit_vanilla_reference.py`
- `experiments/plateau_protocol.py`
- `experiments/run_cifar_pilot.py`
- `experiments/run_deit_all_arms.py`
- `experiments/run_deit_bypass.py`
- `experiments/run_deit_expansion_diagnostics.py`
- `experiments/run_deit_fork.py`
- `experiments/run_deit_projection_aware.py`
- `experiments/run_deit_projection_aware_suite.py`
- `experiments/run_deit_research_v2.py`
- `experiments/run_gromo_pilot.py`
- `experiments/run_plateau_comparison.py`
- `experiments/run_plateau_fork.py`
- `experiments/run_shared_comparison.py`
- `experiments/run_unified_vanilla_to_stall.py`
- `experiments/run_vanilla_to_plateau.py`
- `experiments/train_deit_plateau.py`

### methods (3)

- `methods/e_projection.py`
- `methods/e_repopt.py`
- `methods/vanilla.py`

### models (2)

- `models/__init__.py`
- `models/deit_tiny_cifar.py`

### notebooks (38)

- `notebooks/kaggle_adaptive_e_driven_o.ipynb`
- `notebooks/kaggle_counterfactual_projection_t4x2.ipynb`
- `notebooks/kaggle_deit_tiny_seed0_rollback_comparison.ipynb`
- `notebooks/kaggle_deit_tiny_seed1_all_arms.ipynb`
- `notebooks/kaggle_deit_tiny_seed1_one_shot.ipynb`
- `notebooks/kaggle_deit_tiny_seed2_expansion_diagnostics.ipynb`
- `notebooks/kaggle_deit_tiny_seed2_projection_aware.ipynb`
- `notebooks/kaggle_deit_tiny_seed2_rollback_comparison.ipynb`
- `notebooks/kaggle_deit_tiny_seed3_raw_vs_bypass.ipynb`
- `notebooks/kaggle_deit_tiny_seed4_raw_vs_bypass.ipynb`
- `notebooks/kaggle_deit_tiny_seed5_raw_vs_bypass.ipynb`
- `notebooks/kaggle_deit_tiny_seed5_raw_vs_vanilla_rollback.ipynb`
- `notebooks/kaggle_densenet121_seed0_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_densenet121_seed1_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_densenet121_seed2_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_plateau_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_plateau_eo_t4x2.ipynb`
- `notebooks/kaggle_plateau_fork_t4x2.ipynb`
- `notebooks/kaggle_projection_capacity_diagnostics_t4x2.ipynb`
- `notebooks/kaggle_projection_log_analysis_cpu.ipynb`
- `notebooks/kaggle_resnet18_seed1_raw_vs_vanilla_rollback.ipynb`
- `notebooks/kaggle_resnet34_seed0_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_resnet34_seed1_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_resnet34_seed2_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_unified_seed0_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_unified_seed1_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_unified_seed2_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_vanilla_to_plateau.ipynb`
- `notebooks/kaggle_vgg16_seed0_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_vgg16_seed1_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_vgg16_seed1_methods_from_plateau_t4x2.ipynb`
- `notebooks/kaggle_vgg16_seed2_end_to_end_t4x2.ipynb`
- `notebooks/kaggle_vna_seed0_part1.ipynb`
- `notebooks/kaggle_vna_seed0_part2.ipynb`
- `notebooks/kaggle_vna_seed0_part3.ipynb`
- `notebooks/kaggle_vna_seed1_part1.ipynb`
- `notebooks/kaggle_vna_seed1_part2.ipynb`
- `notebooks/kaggle_vna_seed1_part3.ipynb`

### probe (4)

- `probe/densenet_auxiliary.py`
- `probe/pretrained_gromo.py`
- `probe/vgg_tiny_adapter.py`
- `probe/virtual_expansion.py`

### projection (5)

- `projection/auxiliary_space.py`
- `projection/exact_reference.py`
- `projection/margin_diagnostics.py`
- `projection/metrics.py`
- `projection/transfer_operator.py`

### scripts (22)

- `scripts/analyze_projection_diagnostic.py`
- `scripts/build_kaggle_adaptive_notebook.py`
- `scripts/build_kaggle_convergence_notebook.py`
- `scripts/build_kaggle_deit_all_arms_notebook.py`
- `scripts/build_kaggle_deit_bypass_notebooks.py`
- `scripts/build_kaggle_deit_expansion_diagnostics_notebook.py`
- `scripts/build_kaggle_deit_notebook.py`
- `scripts/build_kaggle_deit_rollback_notebooks.py`
- `scripts/build_kaggle_deit_seed2_projection_aware_notebook.py`
- `scripts/build_kaggle_deit_seed5_raw_vs_vanilla_rollback_notebook.py`
- `scripts/build_kaggle_densenet121_notebooks.py`
- `scripts/build_kaggle_end_to_end_plateau_notebook.py`
- `scripts/build_kaggle_notebook.py`
- `scripts/build_kaggle_plateau_fork_notebook.py`
- `scripts/build_kaggle_plateau_notebook.py`
- `scripts/build_kaggle_projection_diagnostics_notebook.py`
- `scripts/build_kaggle_projection_log_analysis_notebook.py`
- `scripts/build_kaggle_resnet18_seed1_raw_vs_vanilla_rollback_notebook.py`
- `scripts/build_kaggle_resnet34_notebook.py`
- `scripts/build_kaggle_unified_seed_notebooks.py`
- `scripts/build_kaggle_vgg16_notebooks.py`
- `scripts/build_kaggle_vgg16_seed1_methods_notebook.py`

### tests (38)

- `tests/test_adaptive_notebook.py`
- `tests/test_convergence_notebook.py`
- `tests/test_convergence_runner.py`
- `tests/test_deit_ablation.py`
- `tests/test_deit_ablation_integration.py`
- `tests/test_deit_bypass.py`
- `tests/test_deit_e_rollback.py`
- `tests/test_deit_expansion_diagnostics.py`
- `tests/test_deit_function_preserving.py`
- `tests/test_deit_logging.py`
- `tests/test_deit_no_aux_persistence.py`
- `tests/test_deit_parallel.py`
- `tests/test_deit_projection.py`
- `tests/test_deit_projection_aware.py`
- `tests/test_deit_protocol.py`
- `tests/test_deit_resume.py`
- `tests/test_deit_sites.py`
- `tests/test_deit_vanilla_reference.py`
- `tests/test_densenet121_protocol.py`
- `tests/test_end_to_end.py`
- `tests/test_end_to_end_plateau_notebook.py`
- `tests/test_four_arm_notebook.py`
- `tests/test_gromo_integration.py`
- `tests/test_methods.py`
- `tests/test_pilot_protocol.py`
- `tests/test_plateau_fork.py`
- `tests/test_plateau_fork_notebook.py`
- `tests/test_plateau_notebook.py`
- `tests/test_plateau_protocol.py`
- `tests/test_pretrained_gromo.py`
- `tests/test_probe.py`
- `tests/test_projection_diagnostic.py`
- `tests/test_research_projection_v2.py`
- `tests/test_resnet34_protocol.py`
- `tests/test_shared_comparison.py`
- `tests/test_structural_e.py`
- `tests/test_unified_seed_notebooks.py`
- `tests/test_vgg16_protocol.py`

