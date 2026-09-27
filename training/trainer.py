"""Small research trainer that keeps method comparisons explicit."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


class Trainer:
    def __init__(self, model: nn.Module, optimizer, device: str | torch.device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)

    def train_epoch(self, loader) -> EpochMetrics:
        self.model.train()
        total_loss = total_correct = total = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            total_correct += int((logits.argmax(1) == targets).sum())
            total += targets.numel()
        return EpochMetrics(total_loss / total, total_correct / total)

