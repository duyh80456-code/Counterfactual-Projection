import copy
import random
import numpy as np
import pytest
import torch
from torch import nn
from diagnostics.vna_align import MeasurementHook, MeasurementBuffer, assert_bitwise_alignment
from methods.vna_losses import CE
from methods.vna_sites import HookSite
from methods.vna_ggn import vna_exact_ggn, GGNConfig


def loop(steps, seed, with_hook, *, bad=False):
    model = nn.Sequential(nn.Linear(4, 5), nn.BatchNorm1d(5), nn.ReLU(), nn.Dropout(.2), nn.Linear(5, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    site = HookSite(model, '0', '4', 'linear', between=('1', '2', '3'))
    def measure(model, batch, spec, ctx, generator):
        A = {'weight': torch.randn(2, 4, generator=generator), 'bias': torch.randn(2, generator=generator)}
        return vna_exact_ggn(model, site, A, batch, GGNConfig(hutchinson_probes=1, cg_iters=20),
                             loss_spec=spec, ctx=ctx)
    hook = MeasurementHook(measure, seed=seed)
    buffer = MeasurementBuffer(m=8)
    for step in range(steps):
        batch = (torch.randn(4, 4), torch.randint(3, (4,)))
        optimizer.zero_grad()
        nn.functional.cross_entropy(model(batch[0]), batch[1]).backward()
        buffer.update(batch)
        if with_hook:
            if bad: torch.randn(1)
            else: hook(model, buffer.batch(), CE(), {})
        optimizer.step()
    return model.state_dict(), optimizer.state_dict()


def test_bitwise_training_alignment_and_negative_global_rng():
    assert assert_bitwise_alignment(loop, steps=5, seed=19)
    with pytest.raises(AssertionError, match='state'):
        assert_bitwise_alignment(lambda **kw: loop(**kw, bad=True), steps=5, seed=19)


@pytest.mark.parametrize('raises', [False, True])
def test_measurement_restores_values_objects_grads_modes_and_all_rng(raises):
    model = nn.Sequential(nn.Linear(4, 5), nn.BatchNorm1d(5), nn.ReLU(), nn.Linear(5, 3))
    model.train(); model[1].eval()
    for p in model.parameters(): p.grad = torch.ones_like(p)
    values = copy.deepcopy(model.state_dict())
    params, grads = list(model.parameters()), [p.grad for p in model.parameters()]
    buffers, modules = list(model.buffers()), list(model.modules())
    modes = [m.training for m in model.modules()]
    rng, py, npstate = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    def measure(model, batch, spec, ctx, generator):
        model.eval()
        with torch.no_grad():
            for p in model.parameters(): p.add_(torch.randn(p.shape, generator=generator))
            for b in model.buffers(): b.add_(1)
        for p in model.parameters(): p.grad = None
        torch.randn(4); random.random(); np.random.rand()
        if torch.cuda.is_available(): torch.randn(4, device='cuda')
        if raises: raise RuntimeError('injected')
        return 'measured'
    hook = MeasurementHook(measure)
    if raises:
        with pytest.raises(RuntimeError, match='injected'): hook(model, None, CE(), {})
    else:
        assert hook(model, None, CE(), {}) == 'measured'
    assert list(model.parameters()) == params and list(model.buffers()) == buffers
    assert list(model.modules()) == modules and modes == [m.training for m in modules]
    assert all(p.grad is g and torch.equal(g, torch.ones_like(g)) for p, g in zip(params, grads))
    assert all(torch.equal(v, values[n]) for n, v in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state()) and random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], npstate[1])
    assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all() if cuda else []))


def test_parameter_count_violation_restored_and_detected():
    model = nn.Linear(2, 3)
    original = list(model.parameters())
    def measure(model, *args, **kwargs): model.extra = nn.Linear(3, 3)
    with pytest.raises(AssertionError, match='parameter count'):
        MeasurementHook(measure)(model, None, CE(), {})
    assert list(model.parameters()) == original and not hasattr(model, 'extra')


def test_measurement_buffer_only_last_seen_detached_samples():
    buffer = MeasurementBuffer(3)
    x = torch.arange(10.).reshape(5, 2).requires_grad_()
    y = torch.arange(5)
    buffer.update((x[:2], y[:2])); buffer.update((x[2:], y[2:]))
    saved = buffer.batch()
    assert len(buffer) == 3 and torch.equal(saved[0], x[2:]) and torch.equal(saved[1], y[2:])
    assert not saved[0].requires_grad
    with torch.no_grad(): x.zero_()
    assert torch.equal(buffer.batch()[0], saved[0])
    assert MeasurementBuffer().m == 64
    with pytest.raises(ValueError): MeasurementBuffer(0)
    with pytest.raises(ValueError): MeasurementBuffer().batch()
