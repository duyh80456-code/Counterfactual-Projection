"""Stage 1 data and pinned Gromo utilities, shared by the two new CLIs."""
from contextlib import contextmanager
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "third_party/One-Shot-TAS-CCIL"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))

from experiments.shared_protocol import build_cifar_gromo_resnet, datasets_and_indices
from methods.vna import dedicated_generator


@contextmanager
def isolated_rng(generator):
    """Let transforms/upstream code use a private stream, restoring all globals."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.set_rng_state(generator.get_state())
        seed = int(torch.randint(2**31, (), generator=generator))
        random.seed(seed)
        np.random.seed(seed)
        try:
            yield
        finally:
            generator.set_state(torch.get_rng_state())
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def tiny_model(device="cpu", classes=3):
    from dual_growth.adapters import GromoResNet18
    from gromo.containers.resnet import init_full_resnet_structure
    class TinyResNet(GromoResNet18):
        def __init__(self):
            nn.Module.__init__(self)
            self.start_width = 1.0
            self.core = init_full_resnet_structure(
                input_shape=(3, 8, 8), out_features=classes, reduction_factor=1.,
                number_of_blocks_per_stage=1, inplanes=3, nb_stages=1,
                small_inputs=True, use_preactivation=False, device=device)
            self._extended_forward = False
            self._parameter_migrations = []
    return TinyResNet().to(device)


def setup_data(data_root, fake=False, seed=0):
    if not fake:
        train, evaluation, indices, validation, tuning = datasets_and_indices(data_root)
        return train, evaluation, indices, validation[:2000], validation[2000:], tuning
    generator = dedicated_generator(seed, "fake-data")
    dataset = TensorDataset(torch.randn(1280, 3, 8, 8, generator=generator),
                            torch.randint(3, (1280,), generator=generator))
    return dataset, dataset, list(range(1024)), list(range(1024, 1152)), list(range(1152, 1264)), list(range(1264, 1280))


def build_model(device, fake=False):
    return tiny_model(device) if fake else build_cifar_gromo_resnet("resnet18", device)


def augmented_batch(dataset, indices, count, generator, device):
    # Materialize under an isolated stream, including torchvision transforms.
    order = torch.randperm(len(indices), generator=generator)
    chosen = [indices[int(order[i % len(order)])] for i in range(count)]
    with isolated_rng(generator):
        pairs = [dataset[i] for i in chosen]
    return (torch.stack([x for x, _ in pairs]).to(device),
            torch.tensor([int(y) for _, y in pairs], device=device))


def tiny_candidate(model, site, batch, rank, seed=0, capture=None):
    from dual_growth.adapters import TinyAdapter
    from dual_growth.controller import GrowthBudget
    from probe.counterfactual_tiny import CounterfactualTinyProbe
    adapter = TinyAdapter(quantum_params=10**9, numerical_threshold=1e-6, statistical_threshold=0.)
    # Capture the exact native normalized S/N before the upstream solve resets
    # statistics. This hook only reads tensors; no external algorithm is replaced.
    hook = None
    if capture is not None:
        second = model.block(site).second_layer
        original = second._compute_optimal_added_parameters
        def hooked(*args, **kwargs):
            capture["S"] = second.tensor_s_growth().detach().double().clone()
            capture["N"] = second.tensor_n.detach().double().clone()
            return original(*args, **kwargs)
        second._compute_optimal_added_parameters = hooked
        hook = original
    try:
        with isolated_rng(dedicated_generator(seed, "tiny-statistics")):
            return CounterfactualTinyProbe(rank, site).propose(
                adapter, model, list(zip(batch[0].split(64), batch[1].split(64))), GrowthBudget(10**9))
    finally:
        if hook is not None:
            second._compute_optimal_added_parameters = hook
