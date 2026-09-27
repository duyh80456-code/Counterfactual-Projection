"""Vanilla fine-tuning step used by the pilot baseline."""

import torch.nn.functional as F
from torch import Tensor, nn


def vanilla_step(model: nn.Module, optimizer, batch: tuple[Tensor, Tensor],
                 loss_fn=F.cross_entropy) -> float:
    inputs, targets = batch
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach())

