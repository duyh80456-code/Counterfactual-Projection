"""WHERE, nulls, width growth, GN, and complete tiny-model arm/resume tests."""
import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from adapters.deit_cp_adapter import CPConfig, probe_indices
from adapters.deit_ablation import build_raw_reference, execute_intervention, apply_delta, res18_cp_config
from adapters.deit_random_control import random_parameter_delta
from adapters.deit_persistent_growth import commit_growth, restore_growth_geometry
from adapters.deit_opt_e import gn_system, optimize_B, optimize_candidate, OptEConfig
from adapters.deit_mlp_growth import DeitMLPGrowthAdapter
from experiments.deit_protocol import DeitRecipe, build_optimizer_scheduler


def test_where_normalized_changes_argmax(monkeypatch):
    import experiments.run_plateau_comparison as selector
    candidates = [SimpleNamespace(module_name='a', proposal_score=0.), SimpleNamespace(module_name='b', proposal_score=0.)]
    monkeypatch.setattr(selector, 'propose_structural_candidates', lambda *_a, **_k: candidates)
    monkeypatch.setattr(selector, 'CandidateExpansionProbe', lambda: lambda _model, *, candidate, **_kw:
        SimpleNamespace(observed_loss_gain=2. if candidate.module_name == 'a' else 1.,
                        delta_logits=torch.tensor([100. if candidate.module_name == 'a' else 1.])))
    model = torch.nn.Linear(1, 1)
    config = CPConfig()
    raw, table = selector.select_by_expansion_gain(model, [], [None, None], config, torch.device('cpu'))
    normalized, normalized_table = selector.select_by_expansion_gain(model, [], [None, None], config, torch.device('cpu'), normalized=True)
    assert raw.module_name == 'a' and normalized.module_name == 'b'
    assert normalized_table['site_evaluations']['a']['rank_raw'] == 1
    assert normalized_table['site_evaluations']['b']['rank_normalized'] == 1
    assert table['site_evaluations']['b']['where_score_normalized'] == pytest.approx(1. / (1 + 1e-8))


def test_opt_batches_append_without_changing_original_partition():
    config = res18_cp_config()
    ids = list(range(1000))
    selected = probe_indices(ids, 1, config)
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(911731 + 10000)).tolist()
    original_sizes = [256, 32, 32, 32, 32, 32]
    original = [selected['statistics'], *selected['where'], selected['projection'], selected['gate']]
    cursor = 0
    for batch, size in zip(original, original_sizes):
        assert batch == order[cursor:cursor + size]
        cursor += size
    partitions = original + [selected['opt_fit'], selected['opt_val']]
    flattened = sum(partitions, [])
    assert len(flattened) == len(set(flattened))
    assert selected['opt_fit'] == order[cursor:cursor + config.opt_fit_samples]


def test_random_per_tensor_norm_scope_and_inplace(deit_small):
    site = 'blocks.0.mlp'
    names = DeitMLPGrowthAdapter.original_mlp_parameters(deit_small, site)
    reference = {name: torch.randn_like(dict(deit_small.named_parameters())[name]) for name in names}
    reference[names[-1]].zero_()
    before = {name: p.detach().clone() for name, p in deit_small.named_parameters()}
    identities = dict(deit_small.named_parameters())
    rng_before = torch.get_rng_state().clone()
    delta, log = random_parameter_delta(deit_small, reference, site, 123)
    for name in names:
        assert float(delta[name].norm()) == pytest.approx(float(reference[name].norm()), abs=1e-6)
    assert torch.equal(torch.get_rng_state(), rng_before)
    optimizer = torch.optim.AdamW(deit_small.parameters())
    apply_delta(deit_small, optimizer, delta, .05)
    for name, p in deit_small.named_parameters():
        assert p is identities[name]
        if name not in names:
            assert torch.equal(p, before[name])
    with pytest.raises(ValueError):
        random_parameter_delta(deit_small, {names[0]: reference[names[0]]}, site, 123)


@pytest.mark.parametrize('gamma', [0., .2])
def test_persistent_exact_expansion_and_moment_slices(deit_small, deit_batches, deit_native_candidate, gamma):
    model, candidate = deit_small, deit_native_candidate
    recipe = DeitRecipe()
    optimizer, scheduler = build_optimizer_scheduler(model, recipe)
    batch = deit_batches[2]
    torch.nn.functional.cross_entropy(model(batch[0]), batch[1]).backward()
    optimizer.step()
    # Candidate remains a valid detached proposal after this unrelated optimizer step.
    old = dict(model.named_parameters())
    states = {name: copy.deepcopy(optimizer.state[p]) for name, p in old.items()}
    groups = {name: next(g['weight_decay'] for g in optimizer.param_groups if any(q is p for q in g['params'])) for name, p in old.items()}
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    baseline = model(batch[0]).detach()
    with candidate.virtual_direction(gamma):
        virtual = model(batch[0]).detach()
    info = commit_growth(model, optimizer, candidate, gamma)
    actual = model(batch[0]).detach()
    assert torch.allclose(actual, virtual, atol=1e-6, rtol=1e-6)
    if gamma == 0:
        assert torch.allclose(actual, baseline, atol=1e-6, rtol=1e-6)
    assert info['param_count'] == candidate.A.shape[0] * (2 * model.config['embed_dim'] + 1)
    for name, p in model.named_parameters():
        assert next(g['weight_decay'] for g in optimizer.param_groups if any(q is p for q in g['params'])) == groups[name]
        if p is old[name]:
            continue
        assert old[name] not in optimizer.state
        assert torch.equal(optimizer.state[p]['step'], states[name]['step'])
        axis = 1 if name.endswith('fc2.weight') else 0
        slices = [slice(None)] * p.ndim
        slices[axis] = slice(0, old[name].shape[axis])
        new_slices = list(slices); new_slices[axis] = slice(old[name].shape[axis], None)
        for key in ('exp_avg', 'exp_avg_sq'):
            assert torch.equal(optimizer.state[p][key][tuple(slices)], states[name][key])
            assert torch.count_nonzero(optimizer.state[p][key][tuple(new_slices)]) == 0
    assert scheduler.state_dict() == scheduler_before
    optimizer.zero_grad()
    torch.nn.functional.cross_entropy(model(batch[0]), batch[1]).backward()
    optimizer.step()  # Growth remains trainable with migrated AdamW state.
    from models import DeiTTinyCifar
    restored = DeiTTinyCifar(**model.config).double()
    restore_growth_geometry(restored, model.state_dict())
    new_optimizer, _ = build_optimizer_scheduler(restored, recipe)
    restored.load_state_dict(model.state_dict())
    new_optimizer.load_state_dict(optimizer.state_dict())
    assert torch.equal(restored(batch[0]), model(batch[0]))


def test_gn_symmetric_psd_and_toy_loss_decrease():
    x = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]], dtype=torch.float64)
    y = torch.tensor([0, 1, 1, 0])
    B = torch.zeros(2, 2, dtype=torch.float64)
    logits = lambda value: x @ value.T
    rhs, matvec = gn_system(logits, B, y)
    u, v = torch.randn_like(B), torch.randn_like(B)
    assert float((u * matvec(v)).sum()) == pytest.approx(float((v * matvec(u)).sum()), abs=1e-10)
    assert float((u * matvec(u)).sum()) >= -1e-10
    optimized, record = optimize_B(logits, logits, B, y, y, config=OptEConfig(inner_steps=1))
    assert record['status'] == 'ok'
    assert record['inner_val_losses'][-1] < record['inner_val_losses'][0]
    assert record['newton_decrement_predicted'][0] > 0
    assert torch.count_nonzero(B) == 0


def test_opt_e_preserves_function_at_zero_and_leaves_no_aux(deit_small, deit_batches, deit_native_candidate):
    from dataclasses import replace
    candidate = deit_native_candidate
    original = {name: p.detach().clone() for name, p in deit_small.named_parameters()}
    identities = dict(deit_small.named_parameters())
    zero = replace(candidate, B=torch.zeros_like(candidate.B))
    baseline = deit_small(deit_batches[2][0]).detach()
    with zero.virtual_direction(1.):
        assert torch.equal(deit_small(deit_batches[2][0]), baseline)
    optimized, record = optimize_candidate(deit_small, candidate, deit_batches[2], deit_batches[3],
                                          config=OptEConfig(inner_steps=1, cg_iterations=8, diagonal_probes=2))
    assert record['B_dimension'] == candidate.B.numel()
    for name, p in deit_small.named_parameters():
        assert p is identities[name] and torch.equal(p, original[name])
    assert all(not block.mlp._forward_hooks for block in deit_small.blocks)


def test_failed_opt_e_is_noop():
    B = torch.zeros(2, 2, dtype=torch.float64)
    logits = lambda value: torch.zeros(4, 2, dtype=torch.float64) + value.sum() * 0
    y = torch.tensor([0, 1, 0, 1])
    result, record = optimize_B(logits, logits, B, y, y)
    assert record['status'] == 'opt_e_failed' and not record['steps']
    assert torch.equal(result, B)


def test_suite_anchor_counter_resets_on_same_accuracy_lower_loss(deit_small):
    from experiments.deit_e_rollback import EAccuracyRollback
    optimizer, scheduler = build_optimizer_scheduler(deit_small, DeitRecipe())
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(4))
    controller = EAccuracyRollback(deit_small, optimizer, scheduler, loader, {'accuracy': .5, 'loss': 2.}, 243,
                                   stall_on_anchor=True)
    controller.accuracy_stall_counter = 9
    result = controller.observe(deit_small, optimizer, scheduler, loader, {'accuracy': .5, 'loss': 1.9}, 302, 59)
    assert result['controller_anchor_improved'] and result['accuracy_stall_counter'] == 0
    for offset in range(1, 11):
        result = controller.observe(deit_small, optimizer, scheduler, loader, {'accuracy': .4, 'loss': 2.}, 302 + offset, 59 + offset)
    assert result['rollback_applied'] and result['rollback_anchor_epoch'] == 302


def test_job_failure_does_not_skip_later_arms(tmp_path):
    from experiments.run_deit_all_arms import run_jobs
    calls = []
    def invoke(command, check):
        calls.append(command[0])
        if command[0] == 'bad':
            raise RuntimeError('intentional')
    statuses = run_jobs({'a': ['bad'], 'b': ['good']}, tmp_path / 'status.json', runner=invoke)
    assert calls == ['bad', 'good']
    assert statuses['a']['status'] == 'failed' and statuses['b']['status'] == 'completed'
