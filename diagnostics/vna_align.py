"""Measurement isolation and hook-on/off baseline alignment on CPU."""
from collections import deque
from contextlib import contextmanager
import copy
import random
import numpy as np
import torch
from methods.vna import dedicated_generator


@contextmanager
def _global_state():
    python, numpy = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        random.setstate(python); np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


class MeasurementHook:
    """Call measure(model, existing_batch, loss_spec, ctx, generator=stream).

    The callback uses only its dedicated stream. Any incidental global draws
    are isolated too. Original parameter/buffer objects, values, gradients,
    registrations and per-module modes are restored even after an exception.
    No dataloader is accepted or iterated.
    """
    def __init__(self, measure, *, seed=20261014, purpose='measurement'):
        self.measure, self.seed, self.purpose = measure, seed, purpose
        self.generators = {}

    def __call__(self, model, batch_the_loop_already_has, loss_spec, ctx):
        modules = list(model.modules())
        registrations = [(m, dict(m._parameters), dict(m._buffers), dict(m._modules)) for m in modules]
        modes = {m: m.training for m in modules}
        tensors = [(t, t.detach().clone()) for t in (*model.parameters(), *model.buffers())]
        grads = [(p, p.grad, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()]
        count = sum(p.numel() for p in model.parameters())
        device = next(model.parameters()).device
        key = str(device)
        if key not in self.generators:
            self.generators[key] = dedicated_generator(self.seed, self.purpose, device)
        changed_count = False
        with _global_state():
            try:
                return self.measure(model, batch_the_loop_already_has, loss_spec, ctx,
                                    generator=self.generators[key])
            finally:
                changed_count = count != sum(p.numel() for p in model.parameters())
                for module, params, buffers, children in registrations:
                    module._parameters.clear(); module._parameters.update(params)
                    module._buffers.clear(); module._buffers.update(buffers)
                    module._modules.clear(); module._modules.update(children)
                with torch.no_grad():
                    for tensor, value in tensors:
                        if not torch.equal(tensor, value):
                            tensor.copy_(value)
                    for parameter, grad, value in grads:
                        parameter.grad = grad
                        if grad is not None and not torch.equal(grad, value):
                            grad.copy_(value)
                for module, mode in modes.items():
                    module.training = mode
                assert not changed_count, 'measurement changed parameter count'
                assert count == sum(p.numel() for p in model.parameters()), 'parameter count changed'


def _assert_bitwise(left, right, path='state'):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape, path
        assert left.tobytes() == right.tobytes(), path
    elif isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.device.type == right.device.type == 'cpu', path
        assert left.dtype == right.dtype and left.shape == right.shape, path
        assert torch.equal(left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8)), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_bitwise(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            _assert_bitwise(a, b, f'{path}[{i}]')
    else:
        assert type(left) is type(right) and left == right, path


def assert_bitwise_alignment(run_fn, steps, seed):
    """run_fn(steps=, seed=, with_hook=) -> (model state, optimizer state).

    run_fn owns the real loop and callback; this helper never rewrites it.
    CPU deterministic algorithms and identical Python/NumPy/Torch seeds are
    enforced for both runs. State dictionaries must contain CPU tensors.
    A raw callback that consumes global RNG is caught by the comparison.
    """
    states = []
    with _global_state():
        torch.use_deterministic_algorithms(True)
        for enabled in (False, True):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            state = run_fn(steps=steps, seed=seed, with_hook=enabled)
            if not isinstance(state, tuple) or len(state) != 2:
                raise ValueError('run_fn must return (model state_dict, optimizer state_dict)')
            states.append(copy.deepcopy(state))
    _assert_bitwise(*states)
    return True


class MeasurementBuffer:
    """Sliding window of the last m samples supplied by the training loop.

    m is declared by the caller (default 64). update() accepts only a batch
    already seen by the loop; it has no access to test/future data or loaders.
    Samples are detached and cloned so subsequent augmentation cannot edit it.
    """
    def __init__(self, m=64):
        if not isinstance(m, int) or m < 1:
            raise ValueError('m must be a positive integer')
        self.m = m
        self.samples = deque(maxlen=m)

    def update(self, batch):
        if len(batch) != 2 or len(batch[0]) != len(batch[1]):
            raise ValueError('expected an aligned (inputs, targets) batch')
        for x, y in zip(*batch):
            self.samples.append((x.detach().clone(), y.detach().clone()))

    def batch(self):
        if not self.samples:
            raise ValueError('measurement buffer is empty')
        return tuple(torch.stack(items) for items in zip(*self.samples))

    def __len__(self):
        return len(self.samples)
