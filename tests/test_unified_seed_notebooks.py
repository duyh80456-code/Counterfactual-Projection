import json
from pathlib import Path


def notebook_source(seed):
    notebook = json.loads(Path(
        f"notebooks/kaggle_unified_seed{seed}_end_to_end_t4x2.ipynb"
    ).read_text())
    return notebook, "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"])


def test_three_notebooks_differ_only_by_declared_seed_and_output_name():
    normalized = []
    for seed in (0, 1, 2):
        notebook, source = notebook_source(seed)
        assert f"SEED = {seed}" in source
        assert f'"--seed", "{seed}"' in source
        assert f"unified_seed{seed}_end_to_end_v5" in source
        assert "experiments.run_unified_vanilla_to_stall" in source
        assert '"--stall-patience", "100"' in source
        assert '"unified_vanilla_progress"' in source
        assert '"vanilla_validation_best_checkpoint"' in source
        assert '"minimum_additional_epochs_if_no_new_best"' in source
        assert "compatible_training_lineage" in source
        assert "is_post200_raw_best_snapshot" in source
        assert "Phase-1 checkpoint inventory:" in source
        assert "local_discovered" in source
        assert "input_discovered" in source
        assert '"replay_missing_best_epoch"' in source
        assert "must observe a new strict raw validation best before fork" in source
        assert 'PHASE1_MAX_EPOCH = max(' in source
        assert "Cannot replay attached legacy run" in source
        assert "old_best_epoch" in source
        assert '"--recipe-epochs", "200"' in source
        assert '"--lr-milestones", "100,150"' in source
        assert '"--lr-gamma", "0.1"' in source
        assert '"--lr-reduction-patience"' not in source
        assert '"--decay-epochs"' not in source
        assert "100 epochs have elapsed since the" in source
        assert "raw validation-best checkpoint" in source
        assert '"--post-fork-epochs", "150"' in source
        assert '"--retrigger-patience", "20"' in source
        assert 'ours_job = launch(0, "ours_e_driven_o")' in source
        assert 'bypass_job = launch(1, "bypass")' in source
        assert 'o_only_job = launch(1, "o_projection_only")' in source
        assert 'results["vanilla"] = VANILLA_CONTROL' in source
        assert '"theta_P_validation_accuracy"' in source
        assert '"meaningful_best_validation_accuracy"' in source
        assert '"--gamma-post-increase-multiplier", "2.0"' in source
        assert "shared_seed1_epoch300" not in source
        assert "run_vanilla_to_plateau" not in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"seed{seed}.ipynb", "exec")
        normalized.append(
            source.replace(f"SEED = {seed}", "SEED = <seed>")
                  .replace(f'"--seed", "{seed}"', '"--seed", "<seed>"')
                  .replace(f"unified_seed{seed}_end_to_end_v5",
                           "unified_seed<seed>_end_to_end_v5")
                  .replace(f"seed {seed}", "seed <seed>")
                  .replace(f"seed={seed}", "seed=<seed>"))
    assert normalized[0] == normalized[1] == normalized[2]


def test_unified_runner_has_one_schedule_and_complete_resume_state():
    source = Path("experiments/run_unified_vanilla_to_stall.py").read_text()
    assert "StandardMultiStepScheduler" in source
    assert '"scheduler_restart_count": 0' in source
    assert ('"schedule_id": '
            '"cifar-resnet18-sgd-multistep-200-v5-post200-val-best"') in source
    assert ('"stall_gate": "100 epochs after raw validation best at epoch '
            '>=200"') in source
    assert ('"theta_P_scope": "raw validation best at or after recipe '
            'epoch 200"') in source
    assert "optimizer.load_state_dict(saved[\"optimizer\"])" in source
    assert "scheduler.load_state_dict(saved[\"scheduler\"])" in source
    assert 'restore_rng(saved["rng"])' in source
    assert 'saved["train_loader_generator_state"]' in source
    assert "detector.update(epoch, evaluation[\"accuracy\"])" in source
    assert "scheduler.step(trigger[\"accuracy\"])" not in source
    assert source.count("scheduler.step()") == 1
    assert "detector.arm_stall" in source
    assert source.count('"learning_rates": training_lrs') == 1
    assert source.count('"next_learning_rates":') == 2
    assert "min_epoch=args.recipe_epochs" in source
    assert '"history": history' in source
    assert 'selection["improved"]' in source
    assert '"kind": "plateau_fork_checkpoint"' not in source
    assert 'checkpoint_validation_best_epoch{epoch:04d}.pt' in source
    assert 'replay_missing_best_epoch = saved.get(' in source
    assert 'selection["improved"]' in source
    assert "raw validation stall reached but theta_P weights are" in source
