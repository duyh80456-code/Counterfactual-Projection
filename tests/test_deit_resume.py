from dataclasses import replace
import tarfile

import pytest
import torch

from adapters.deit_ablation import res18_cp_config
from experiments.deit_protocol import DeitRecipe, protocol, canonical_model_config
from experiments.deit_resume import prepare_resume, FULL_STATE
from experiments.shared_protocol import sha256_file


def phase1_state(recipe, kind, epoch, history):
    return {**{key: {} for key in FULL_STATE}, 'kind': kind,
        'protocol': protocol(recipe, canonical_model_config()), 'epoch': epoch,
        'history': history, 'historical_best_epoch': 1}


def test_archive_resume_selects_latest_and_preserves_original_bytes(tmp_path):
    recipe, cp = DeitRecipe(), res18_cp_config()
    input_dir, output = tmp_path / 'input', tmp_path / 'output'
    input_dir.mkdir()
    history = [{'epoch': 1, 'validation_accuracy': .5, 'validation_loss': 1.}]
    best = phase1_state(recipe, 'deit_vanilla_best', 1, history)
    latest = phase1_state(recipe, 'deit_vanilla_latest', 3, history + [{'epoch': 2}, {'epoch': 3}])
    source = tmp_path / 'source'; source.mkdir()
    torch.save(best, source / 'best.pt'); torch.save(latest, source / 'latest.pt')
    with tarfile.open(input_dir / 'output.tar.gz', 'w:gz') as archive:
        archive.add(source, arcname='run')
    plan = prepare_resume([input_dir], output, recipe, cp, horizon=150)
    assert plan['phase1'] == 'resume' and plan['phase1_epoch'] == 3
    assert sha256_file(output / 'vanilla_stall/checkpoint_latest.pt') == sha256_file(source / 'latest.pt')
    # A stale attachment cannot overwrite a newer local checkpoint.
    newer = {**latest, 'epoch': 4, 'history': latest['history'] + [{'epoch': 4}]}
    torch.save(newer, output / 'vanilla_stall/checkpoint_latest.pt')
    plan = prepare_resume([input_dir], output, recipe, cp, horizon=150)
    assert plan['phase1_epoch'] == 4


def test_missing_full_best_blocks_phase1_resume(tmp_path):
    recipe = DeitRecipe()
    torch.save(phase1_state(recipe, 'deit_vanilla_latest', 1, []), tmp_path / 'latest.pt')
    with pytest.raises(FileNotFoundError, match='matching full'):
        prepare_resume([tmp_path], tmp_path / 'out', recipe, res18_cp_config(), horizon=150)


def test_protocol_or_incomplete_state_is_rejected(tmp_path):
    recipe = DeitRecipe()
    wrong = phase1_state(replace(recipe, seed=2), 'deit_vanilla_latest', 1, [])
    incomplete = phase1_state(recipe, 'deit_vanilla_latest', 1, [])
    del incomplete['optimizer']
    torch.save(wrong, tmp_path / 'wrong.pt'); torch.save(incomplete, tmp_path / 'incomplete.pt')
    plan = prepare_resume([tmp_path], tmp_path / 'out', recipe, res18_cp_config(), horizon=150)
    assert plan['phase1'] == 'start'
    assert len(plan['rejected']) == 2


def test_arm_import_and_completed_skip_with_bound_references(tmp_path, monkeypatch):
    import experiments.deit_resume as resume
    from experiments.deit_ablation_runner import arm_identity
    recipe, cp = DeitRecipe(), res18_cp_config()
    source, output = tmp_path / 'input', tmp_path / 'out'; source.mkdir()
    row = {'epoch': 1, 'post_fork_epoch': 0, 'validation_accuracy': .5, 'validation_loss': 1.}
    fork = phase1_state(recipe, 'deit_plateau_fork', 1, [row])
    fork.update(historical_best_accuracy=.5, historical_best_loss=1., vanilla_history=[])
    torch.save(fork, source / 'fork.pt'); fork_hash = sha256_file(source / 'fork.pt')
    vanilla = {**fork, 'kind': 'deit_vanilla_reference', 'theta_best_hash': fork_hash}
    torch.save(vanilla, source / 'vanilla.pt')
    raw = {'kind': 'deit_raw_intervention_reference', 'suite_version': 1, 'seed': recipe.seed,
           'cp_config': __import__('dataclasses').asdict(cp), 'fork_hash': fork_hash}
    torch.save(raw, source / 'raw.pt')
    identity = arm_identity(cp, fork_hash, 'o_projection_only', 2, 10, 5)
    identity['raw_reference_hash'] = sha256_file(source / 'raw.pt')
    state = {**fork, 'kind': 'deit_fork_arm_latest', 'method': 'o_projection_only',
        'run_identity': identity, 'epoch': 3, 'completed_epochs': 2,
        'history': [row, {**row, 'epoch': 2}, {**row, 'epoch': 3}],
        'interventions': [{'validation_1_to_5_epochs_after': []}],
        'validation_before': {'accuracy': .5, 'loss': 1.},
        'validation_immediately_after_projection': {'accuracy': .5, 'loss': 1.},
        'report_best_accuracy': .5, 'report_best_loss': 1., 'report_best_epoch': 2}
    torch.save(state, source / 'arm.pt')
    monkeypatch.setattr(resume, 'checked_source', lambda path, kinds: (torch.load(path, weights_only=False), recipe))
    plan = prepare_resume([source], output, recipe, cp, horizon=2, arms=['o_projection_only'])
    assert plan['arms']['o_projection_only']['action'] == 'completed'
    assert (output / 'arms/o_projection_only/result.json').exists()
    # Mid-run checkpoint resumes without repeating the intervention.
    state.update(completed_epochs=1, epoch=2, history=state['history'][:2])
    torch.save(state, source / 'arm.pt')
    plan = prepare_resume([source], tmp_path / 'mid', recipe, cp, horizon=2, arms=['o_projection_only'])
    assert plan['arms']['o_projection_only'] == {'action': 'resume', 'completed_epochs': 1}
    with pytest.raises(ValueError, match='incompatible horizon'):
        prepare_resume([source], tmp_path / 'wrong', recipe, cp, horizon=3, arms=['o_projection_only'])
    (source / 'raw.pt').unlink()
    with pytest.raises(FileNotFoundError, match='original raw'):
        prepare_resume([source], tmp_path / 'missing', recipe, cp, horizon=2, arms=['o_projection_only'])
