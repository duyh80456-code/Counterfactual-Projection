from pathlib import Path


def test_convergence_search_preserves_training_pool_and_saves_full_plateau():
    source = Path("experiments/run_vanilla_to_plateau.py").read_text()
    assert 'train_indices = list(source["train_indices"])' in source
    assert "validation_pool[:args.trigger_samples]" in source
    assert "validation_pool[args.trigger_samples:]" in source
    assert "ConstantCheckpointScheduler" in source
    assert "BestCheckpointStallDetector" in source
    assert '"--no-new-best-patience"' in source
    assert 'default=100' in source
    assert '"--best-min-gain"' in source
    assert 'default=1e-3' in source
    assert 'best["kind"] = "plateau_fork_checkpoint"' in source
    assert '"kind": "vanilla_best_checkpoint"' in source
    assert 'best_path = output / "checkpoint_best.pt"' in source
    assert 'detector.update(epoch, trigger["accuracy"])' in source
    assert 'detector.update(START_EPOCH, trigger["accuracy"])' in source
    assert '"evaluation_role": "report-only (3,000 held-out samples)"' in source
    assert 'if selection["improved"]:' in source
    assert "finalize_best_stall" in source
    assert '"vanilla_control_history"' in source
    assert '"vanilla_control"' in source
    assert "len(control) != detector.patience" in source
    assert '"optimizer": optimizer.state_dict()' in source
    assert '"scheduler": scheduler.state_dict()' in source
    assert '"rng": rng_state()' in source
    assert '"train_loader_generator_state"' in source
    assert '"review_horizon_reached_no_plateau"' in source
