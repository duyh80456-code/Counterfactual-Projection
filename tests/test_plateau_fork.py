from pathlib import Path


def test_plateau_fork_has_recurrent_projected_arms_and_best_checkpoints():
    source = Path("experiments/run_plateau_fork.py").read_text()
    assert '"o_projection_only")' in source
    assert 'source.get("kind") != "plateau_fork_checkpoint"' in source
    assert "optimizer.load_state_dict(source[\"optimizer\"])" in source
    assert "scheduler.load_state_dict(source[\"scheduler\"])" in source
    assert "train_indices = list(source[\"train_indices\"])" in source
    assert '"mode": "recurrent_best_rollback"' in source
    assert '"patience": args.retrigger_patience' in source
    assert "run_intervention(" in source
    assert "run_o_only_intervention(" in source
    assert "supervised_functional_descent_direction" in source
    assert "ten_sgd_epochs_without_new_best" in source
    assert "live_rng = rng_state()" in source
    assert "live_loader_state = train_loader.generator.get_state().clone()" in source
    assert 'best_state["model"]' in source
    assert "perform_intervention(" in source
    assert '"intervention_count"' in source
    assert '"rollback_count"' in source
    assert "pre_probe_rng = rng_state()" in source
    assert "restore_rng(pre_probe_rng)" in source
    assert "embed_relaxed_bypass" in source
    assert "transition_from_opt2_" in source
    assert '"time_spent_expanded_seconds"' in source
    assert '"peak_train_params"' in source
    assert '"epochs_to_best"' in source
    assert 'best_checkpoint = output / "checkpoint_best.pt"' in source
    assert '"plateau_fork_arm_best"' in source
    assert 'default=100' in source
    assert '"--opt1-epochs", type=int, default=70' in source
    assert '"--max-opt2-epochs", type=int, default=30' in source
    assert '"--gamma-increase-opt2-epoch", type=int, default=15' in source
    assert '"--gamma-post-increase-multiplier", type=float, default=10.0' in source
    assert '"--significant-improvement", type=float, default=1e-3' in source
    assert '"plateau_checkpoint_hash": fork_hash' in source
    assert '"train_indices": train_indices' in source
    assert '"trigger_indices": trigger_indices' in source
    assert '"evaluation_indices": evaluation_indices' in source
    assert '"theta_best_hash": fork_hash' in source
    assert '"opt1_epochs": opt1_done if args.method == "bypass" else None' in source
    assert 'exact_improved = trigger["accuracy"] > exact_best_trigger_accuracy' in source
    assert 'significant_improved = (' in source
    assert 'validation["accuracy"] > best_accuracy' in source
    assert "epochs_since_best" not in source
