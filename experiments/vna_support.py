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
    if isinstance(dataset, DeviceCIFAR):
        return dataset.batch(chosen, generator)
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


class DeviceCIFAR:
    """Resident uint8 CIFAR; all crop/flip choices use the supplied CPU stream.

    CPU is supported as an oracle/fallback. Transform distribution matches the
    reference train transform; the legacy PIL stream is not bitwise equivalent.
    Every paired arm must use the same path and restored generator state.
    """
    def __init__(self, dataset, device):
        self.data = torch.as_tensor(np.asarray(dataset.data), device=device).permute(0, 3, 1, 2).contiguous()
        self.targets = torch.as_tensor(dataset.targets, device=device, dtype=torch.long)
        self.mean = torch.tensor((.5071, .4867, .4408), device=device)[None, :, None, None]
        self.std = torch.tensor((.2675, .2565, .2761), device=device)[None, :, None, None]

    def batch(self, indices, generator):
        import torch.nn.functional as F
        index = torch.as_tensor(indices, device=self.data.device, dtype=torch.long)
        images = F.pad(self.data[index], (4, 4, 4, 4))
        count = len(index)
        offsets = torch.randint(9, (count, 2), generator=generator).to(self.data.device)
        flip = (torch.rand(count, generator=generator) < .5).to(self.data.device)
        axis = torch.arange(32, device=self.data.device)
        yy = offsets[:, 0, None, None] + axis[None, :, None]
        xx = offsets[:, 1, None, None] + axis[None, None, :]
        images = images.permute(0, 2, 3, 1)[torch.arange(count, device=self.data.device)[:, None, None], yy, xx].permute(0, 3, 1, 2)
        images = torch.where(flip[:, None, None, None], images.flip(-1), images).float().div_(255)
        return (images - self.mean) / self.std, self.targets[index]


class DeviceTrainLoader:
    def __init__(self, dataset, indices, batch_size, seed, generator_state=None):
        self.dataset, self.indices, self.batch_size = dataset, indices, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        if generator_state is not None:
            self.generator.set_state(generator_state.cpu())

    def __iter__(self):
        order = torch.randperm(len(self.indices), generator=self.generator).tolist()
        for start in range(0, len(order), self.batch_size):
            yield self.dataset.batch([self.indices[i] for i in order[start:start + self.batch_size]], self.generator)


def fast_train_epoch(model, loader, optimizer, device, *, amp=False, scaler=None, channels_last=False):
    """Plain SGD only; VNA math is never entered under this autocast scope."""
    model.train()
    loss_sum = correct = count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.float(), targets)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.detach()) * targets.numel()
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return {"loss": loss_sum/count, "task_loss": loss_sum/count, "accuracy": correct/count}
