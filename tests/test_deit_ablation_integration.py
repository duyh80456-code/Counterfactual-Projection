"""Native TINY + real optimizer main loops on small models, including resume."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from adapters.deit_ablation import build_raw_reference, res18_cp_config
from adapters.deit_persistent_growth import restore_growth_geometry
from experiments.deit_protocol import (DeitRecipe, build_optimizer_scheduler, save_state, protocol,
    evaluate_without_rng, canonical_model_config)
from experiments.shared_protocol import restore_rng, sha256_file
from models import DeiTTinyCifar


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_state_equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize('method', ['o_projection_only', 'e_driven_o_raw', 'e_driven_o_normalized',
                                   'random_control_parameter', 'random_control_logit', 'persistent_growth', 'opt_e'])
def test_all_arm_main_loop_and_resume(deit_small, deit_batches, tmp_path, monkeypatch, method):
    import experiments.deit_ablation_runner as runner
    model = deit_small
    model.eval()
    recipe = replace(DeitRecipe(), workers=0, batch_size=8)
    optimizer, scheduler = build_optimizer_scheduler(model, recipe)
    batch = deit_batches[0]
    dataset = TensorDataset(*batch)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(1))
    evaluation = DataLoader(dataset, batch_size=8)
    baseline = evaluate_without_rng(model, evaluation, torch.device('cpu'))
    fork_path = tmp_path / 'fork.pt'
    save_state(fork_path, model=model, optimizer=optimizer, scheduler=scheduler, loader=loader,
        epoch=23, history=[{'epoch': 23, 'validation_accuracy': baseline['accuracy'], 'validation_loss': baseline['loss']}],
        train_indices=list(range(8)), evaluation_indices=[0, 1], source_tuning_indices=[], trigger_indices=[],
        run_protocol=protocol(recipe, model), kind='deit_plateau_fork', historical_best_accuracy=baseline['accuracy'],
        historical_best_loss=baseline['loss'], historical_best_epoch=23)
    fork = torch.load(fork_path, weights_only=False)
    config = replace(res18_cp_config(), rank=2, statistics_samples=16, where_batches=2,
        projection_samples=8, gate_samples=8, opt_fit_samples=8, opt_val_samples=8,
        cg_iterations=8, cg_preconditioner_probes=0, o_only_site='blocks.1.mlp')
    batches = {'statistics': deit_batches[:2], 'where_batches': deit_batches[2:],
               'projection_batch': deit_batches[2], 'gate_batch': deit_batches[3],
               'opt_fit': deit_batches[0], 'opt_val': deit_batches[1]}
    # Dataset partition disjointness is covered independently; tiny tensors exercise numerical paths.
    indices = {'fixture': True}
    raw = build_raw_reference(model, batches, config, sha256_file(fork_path), seed=1, indices=indices)
    raw_path = tmp_path / 'raw.pt'
    torch.save(raw, raw_path)
    def context(_root, _recipe, device, source):
        current = DeiTTinyCifar(**model.config).double()
        if source.get('persistent_growth'):
            restore_growth_geometry(current, source['model'])
        opt, sched = build_optimizer_scheduler(current, recipe)
        current.load_state_dict(source['model'])
        opt.load_state_dict(source['optimizer']); sched.load_state_dict(source['scheduler'])
        gen = torch.Generator().set_state(source['train_loader_generator_state'])
        train = DataLoader(dataset, batch_size=8, shuffle=True, generator=gen)
        restore_rng(source['rng'])
        return current, opt, sched, train, evaluation, dataset, list(range(8)), [0, 1], [], []
    monkeypatch.setattr(runner, 'load_training_context', context)
    monkeypatch.setattr(runner, 'materialize_probe_batches', lambda *_a, **_k: (batches, indices))
    monkeypatch.setattr(runner, 'checked_source', lambda path, kinds: (torch.load(path, weights_only=False), recipe))
    from adapters.deit_opt_e import OptEConfig
    monkeypatch.setattr(runner, 'OptEConfig', lambda **kw: OptEConfig(cg_iterations=8, diagonal_probes=2, **kw))
    output = tmp_path / method
    args = SimpleNamespace(method=method, opt_inner_steps=1, post_fork_epochs=3, algorithm_patience=2,
        vanilla_reference=None, plateau_checkpoint=fork_path, resume=None, raw_reference=raw_path,
        data_root='fixture', device='cpu', output=output)
    midpoint = tmp_path / 'midpoint.pt'
    original_save = runner.save_state
    def capture(path, **kwargs):
        original_save(path, **kwargs)
        if kwargs['completed_epochs'] == 1:
            torch.save(torch.load(path, weights_only=False), midpoint)
    monkeypatch.setattr(runner, 'save_state', capture)
    runner.run_arm(args, config, fork, recipe, sha256_file(fork_path))
    final = torch.load(output / 'checkpoint_latest.pt', weights_only=False)
    args.resume = midpoint
    runner.run_arm(args, config, fork, recipe, sha256_file(fork_path))
    resumed = torch.load(output / 'checkpoint_latest.pt', weights_only=False)
    for key in ('model', 'optimizer', 'scheduler', 'train_loader_generator_state', 'history', 'interventions', 'e_controller'):
        assert_state_equal(final[key], resumed[key])
    result = json.loads((output / 'result.json').read_text())
    assert result['intervention_count'] == 1 and result['retrigger'] is False
    assert result['scientific_escape'] == (max(row['validation_accuracy'] for row in result['history'][1:]) > baseline['accuracy'])
    record = result['interventions'][0]
    assert 'site_evaluations' in record
    assert all('where_score_normalized' in row for row in record['site_evaluations'].values())
    if method == 'persistent_growth':
        assert final['persistent_growth']['param_count'] > 0
    else:
        assert final['model'].keys() == fork['model'].keys()
        assert all(final['model'][name].shape == fork['model'][name].shape for name in final['model'])
    if method == 'random_control_parameter':
        assert record['selected_scale'] == raw['record']['selected_scale']
    if method == 'opt_e':
        assert record['opt_e_protocol']['scales'] == [.1, .25, .5, 1.]


def test_recipe_complete_plateau_and_reused_longer_vanilla_window(tmp_path, monkeypatch):
    import experiments.train_deit_plateau as baseline
    from experiments.deit_vanilla_reference import export_reused_vanilla_arm
    recipe = replace(DeitRecipe(), workers=0, batch_size=2, schedule_epochs=2, warmup_epochs=1,
        stall_start_epoch=2, stall_patience=2, reference_epochs=4, max_epoch=10)
    model = torch.nn.Linear(2, 2)
    model.config = canonical_model_config()
    optimizer, scheduler = build_optimizer_scheduler(model, recipe)
    data = TensorDataset(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]))
    loader = DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(1))
    context = (model, optimizer, scheduler, loader, loader, data, [0, 1], [2, 3], [], [])
    monkeypatch.setattr(baseline, 'load_training_context', lambda *_a: context)
    metrics = iter([.9, .8, .4, .6, .5, .5, .8, .7])
    monkeypatch.setattr(baseline, 'evaluate_without_rng', lambda *_a: {'accuracy': next(metrics), 'loss': 2.})
    output = tmp_path / 'phase1'
    flags = [argument for key, value in asdict(recipe).items() for argument in ('--' + key.replace('_', '-'), str(value))]
    monkeypatch.setattr(sys, 'argv', ['train', '--data-root', 'fixture', '--output', str(output), '--device', 'cpu', *flags])
    baseline.main()
    fork = torch.load(output / 'plateau_checkpoint.pt', weights_only=False)
    reference = torch.load(output / 'vanilla_reference.pt', weights_only=False)
    assert fork['epoch'] == 3 and fork['stall_detected_epoch'] == 5
    assert len(fork['stall_history']) == 2 and len(fork['vanilla_history']) == 4
    assert fork['historical_best_accuracy'] == .6
    assert reference['epoch'] == 7
    result = export_reused_vanilla_arm(fork, reference, tmp_path / 'vanilla',
                                      {'fork_hash': sha256_file(output / 'plateau_checkpoint.pt'), 'post_fork_epochs': 4})
    assert result['scientific_escape'] and result['report_best_accuracy'] == .8
    assert result['additional_training_epochs'] == 0
    # A0 suite exports on CPU and carries WHERE scores strictly for offline reading.
    from experiments.deit_ablation_runner import run_arm
    config = res18_cp_config()
    raw_path = tmp_path / 'raw.pt'
    table = {'blocks.0.mlp': {'where_score_raw': 1., 'where_score_normalized': .5}}
    torch.save({'kind': 'deit_raw_intervention_reference', 'suite_version': 1,
                'fork_hash': sha256_file(output / 'plateau_checkpoint.pt'), 'cp_config': asdict(config),
                'record': {'site_evaluations': table}}, raw_path)
    args = SimpleNamespace(method='vanilla_continue', opt_inner_steps=1, post_fork_epochs=4,
        algorithm_patience=10, vanilla_reference=output / 'vanilla_reference.pt',
        plateau_checkpoint=output / 'plateau_checkpoint.pt', resume=None, raw_reference=raw_path,
        output=tmp_path / 'a0')
    def forbidden(*args, **kwargs):
        raise AssertionError('Vanilla export must not use CUDA or train')
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    exported = run_arm(args, config, fork, recipe, sha256_file(output / 'plateau_checkpoint.pt'))
    assert exported['additional_training_epochs'] == 0 and exported['site_evaluations'] == table
    assert exported['where_role'] == 'offline_only_did_not_select_vanilla'
