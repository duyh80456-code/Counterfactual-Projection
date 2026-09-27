"""Metrics for the counterfactual-projection hypothesis."""

from torch import Tensor
import torch


def projection_ratio(fitted_delta: Tensor, target_delta: Tensor) -> float:
    denominator = torch.linalg.vector_norm(target_delta).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(fitted_delta) / denominator)


def relative_residual(fitted_delta: Tensor, target_delta: Tensor) -> float:
    denominator = torch.linalg.vector_norm(target_delta).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(target_delta - fitted_delta) / denominator)


def growth_benefit_recovery(projected_gain: float, growth_gain: float) -> float:
    if growth_gain == 0:
        raise ZeroDivisionError("growth gain must be non-zero")
    return float(projected_gain / growth_gain)

