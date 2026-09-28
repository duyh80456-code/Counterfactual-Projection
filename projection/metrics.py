"""Metrics for the counterfactual-projection hypothesis."""

from torch import Tensor
import torch


def fitted_norm_ratio(fitted_delta: Tensor, target_delta: Tensor) -> float:
    """Norm ratio only; with damping this is not an orthogonal projection ratio."""
    denominator = torch.linalg.vector_norm(target_delta).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(fitted_delta) / denominator)


# Backward-compatible name for old result files. New reports must use
# ``fitted_norm_ratio`` because damping destroys exact projection semantics.
projection_ratio = fitted_norm_ratio


def relative_residual(fitted_delta: Tensor, target_delta: Tensor) -> float:
    denominator = torch.linalg.vector_norm(target_delta).clamp_min(1e-12)
    return float(torch.linalg.vector_norm(target_delta - fitted_delta) / denominator)


def cosine_alignment(fitted_delta: Tensor, target_delta: Tensor) -> float:
    fitted = fitted_delta.reshape(-1)
    target = target_delta.reshape(-1)
    denominator = (torch.linalg.vector_norm(fitted) *
                   torch.linalg.vector_norm(target)).clamp_min(1e-12)
    return float(torch.dot(fitted, target) / denominator)


def growth_benefit_recovery(projected_gain: float, growth_gain: float) -> float:
    if growth_gain == 0:
        raise ZeroDivisionError("growth gain must be non-zero")
    return float(projected_gain / growth_gain)
