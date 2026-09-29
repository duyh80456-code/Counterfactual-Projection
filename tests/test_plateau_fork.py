from pathlib import Path


def test_plateau_fork_has_four_arms_and_one_immediate_intervention():
    source = Path("experiments/run_plateau_fork.py").read_text()
    assert '"o_projection_only")' in source
    assert 'source.get("kind") != "plateau_fork_checkpoint"' in source
    assert "optimizer.load_state_dict(source[\"optimizer\"])" in source
    assert "scheduler.load_state_dict(source[\"scheduler\"])" in source
    assert "train_indices = list(source[\"train_indices\"])" in source
    assert '"structural_E_interventions": (' in source
    assert "run_intervention(" in source
    assert "run_o_only_intervention(" in source
    assert "supervised_functional_descent_direction" in source
    assert '"supervised_O_interventions"' in source
    assert "pre_probe_rng = rng_state()" in source
    assert "restore_rng(pre_probe_rng)" in source
    assert "embed_relaxed_bypass" in source
    assert "transition_from_opt2_" in source
    assert '"time_spent_expanded_seconds"' in source
    assert '"peak_train_params"' in source
    assert '"epochs_to_best"' in source
    assert '"plateau_checkpoint_hash": fork_hash' in source
    assert '"train_indices": train_indices' in source
    assert '"trigger_indices": trigger_indices' in source
    assert '"evaluation_indices": evaluation_indices' in source
