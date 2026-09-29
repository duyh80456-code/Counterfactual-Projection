from pathlib import Path


def test_convergence_search_preserves_training_pool_and_saves_full_plateau():
    source = Path("experiments/run_vanilla_to_plateau.py").read_text()
    assert 'train_indices = list(source["train_indices"])' in source
    assert "validation_pool[:args.trigger_samples]" in source
    assert "validation_pool[args.trigger_samples:]" in source
    assert "ConstantCheckpointScheduler" in source
    assert "required-plateau-windows" in source
    assert 'default=20' in source
    assert 'default=2' in source
    assert 'payload["kind"] = "plateau_fork_checkpoint"' in source
    assert '"optimizer": optimizer.state_dict()' in source
    assert '"scheduler": scheduler.state_dict()' in source
    assert '"rng": rng_state()' in source
    assert '"train_loader_generator_state"' in source
    assert '"review_horizon_reached_no_plateau"' in source
