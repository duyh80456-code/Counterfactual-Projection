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
    for seed in (1, 2, 3):
        notebook, source = notebook_source(seed)
        assert f"SEED = {seed}" in source
        assert f'"--seed", "{seed}"' in source
        assert f"unified_seed{seed}_end_to_end_v1" in source
        assert "experiments.run_unified_vanilla_to_stall" in source
        assert '"--stall-patience", "100"' in source
        assert '"--best-min-gain", "0.0"' in source
        assert '"--significant-min-gain", "0.001"' in source
        assert '"--lr-reduction-patience", "20"' in source
        assert '"--lr-reduction-factor", "0.2"' in source
        assert '"--min-lr", "0.002"' in source
        assert '"--decay-epochs"' not in source
        assert '"--post-fork-epochs", "100"' in source
        assert 'ours_job = launch(0, "ours_e_driven_o")' in source
        assert 'bypass_job = launch(1, "bypass")' in source
        assert 'o_only_job = launch(1, "o_projection_only")' in source
        assert "shared_seed1_epoch300" not in source
        assert "run_vanilla_to_plateau" not in source
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"seed{seed}.ipynb", "exec")
        normalized.append(
            source.replace(f"SEED = {seed}", "SEED = <seed>")
                  .replace(f'"--seed", "{seed}"', '"--seed", "<seed>"')
                  .replace(f"unified_seed{seed}_end_to_end_v1",
                           "unified_seed<seed>_end_to_end_v1")
                  .replace(f"seed {seed}", "seed <seed>")
                  .replace(f"seed={seed}", "seed=<seed>"))
    assert normalized[0] == normalized[1] == normalized[2]


def test_unified_runner_has_one_schedule_and_complete_resume_state():
    source = Path("experiments/run_unified_vanilla_to_stall.py").read_text()
    assert "SignificantPlateauScheduler" in source
    assert '"scheduler_restart_count": 0' in source
    assert '"schedule_id": "event-driven-significant-plateau-v1"' in source
    assert "optimizer.load_state_dict(saved[\"optimizer\"])" in source
    assert "scheduler.load_state_dict(saved[\"scheduler\"])" in source
    assert 'restore_rng(saved["rng"])' in source
    assert 'saved["train_loader_generator_state"]' in source
    assert "detector.update(epoch, trigger[\"accuracy\"])" in source
    assert "scheduler.step(trigger[\"accuracy\"])" in source
    assert "detector.update(epoch, evaluation" not in source
    assert '"history": history' in source
    assert 'selection["improved"]' in source
    assert '"kind": "plateau_fork_checkpoint"' not in source
