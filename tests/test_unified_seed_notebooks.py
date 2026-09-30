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
        assert f"unified_seed{seed}_end_to_end_v2" in source
        assert "experiments.run_unified_vanilla_to_stall" in source
        assert '"--stall-patience", "100"' in source
        assert '"--best-min-gain", "0.0"' in source
        assert '"--significant-min-gain", "0.001"' in source
        assert '"--recipe-epochs", "200"' in source
        assert '"--lr-milestones", "100,150"' in source
        assert '"--lr-gamma", "0.1"' in source
        assert '"--lr-reduction-patience"' not in source
        assert '"--decay-epochs"' not in source
        assert "Only after the base recipe completes" in source
        assert "exact-best search is rebased at the" in source
        assert '"--post-fork-epochs", "100"' in source
        assert '("ours_e_driven_o", "vanilla")' in source
        assert '("bypass", "o_projection_only")' in source
        assert 'results["vanilla"] = VANILLA_CONTROL' not in source
        assert '"--gamma-post-increase-multiplier", "2.0"' in source
        assert "shared_seed1_epoch300" not in source
        assert "run_vanilla_to_plateau" not in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"seed{seed}.ipynb", "exec")
        normalized.append(
            source.replace(f"SEED = {seed}", "SEED = <seed>")
                  .replace(f'"--seed", "{seed}"', '"--seed", "<seed>"')
                  .replace(f"unified_seed{seed}_end_to_end_v2",
                           "unified_seed<seed>_end_to_end_v2")
                  .replace(f"seed {seed}", "seed <seed>")
                  .replace(f"seed={seed}", "seed=<seed>"))
    assert normalized[0] == normalized[1] == normalized[2]


def test_unified_runner_has_one_schedule_and_complete_resume_state():
    source = Path("experiments/run_unified_vanilla_to_stall.py").read_text()
    assert "StandardMultiStepScheduler" in source
    assert '"scheduler_restart_count": 0' in source
    assert ('"schedule_id": '
            '"cifar-resnet18-sgd-multistep-200-v2-post-arm-best"') in source
    assert '"stall_gate": "base backbone recipe complete"' in source
    assert '"theta_P_scope": "exact trigger best at or after stall arm"' in source
    assert "optimizer.load_state_dict(saved[\"optimizer\"])" in source
    assert "scheduler.load_state_dict(saved[\"scheduler\"])" in source
    assert 'restore_rng(saved["rng"])' in source
    assert 'saved["train_loader_generator_state"]' in source
    assert "detector.update(epoch, trigger[\"accuracy\"])" in source
    assert "scheduler.step(trigger[\"accuracy\"])" not in source
    assert source.count("scheduler.step()") == 1
    assert "detector.arm_stall(epoch, trigger[\"accuracy\"])" in source
    assert source.count('"learning_rates": training_lrs') == 1
    assert source.count('"next_learning_rates":') == 2
    assert "detector.update(epoch, evaluation" not in source
    assert '"history": history' in source
    assert 'selection["improved"]' in source
    assert '"kind": "plateau_fork_checkpoint"' not in source
