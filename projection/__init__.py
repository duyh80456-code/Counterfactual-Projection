from .cg import CGResult, conjugate_gradient
from .functional import FunctionalEvaluation, FunctionalProjector, ProjectionResult
from .metrics import (
    cosine_alignment, fitted_norm_ratio, growth_benefit_recovery,
    projection_ratio,
    relative_residual)
from .auxiliary_space import StructuralAuxiliarySpace
from .transfer_operator import StructuralExpansionTransfer

__all__ = [
    "CGResult", "conjugate_gradient", "FunctionalEvaluation",
    "FunctionalProjector", "ProjectionResult",
    "cosine_alignment", "fitted_norm_ratio", "growth_benefit_recovery",
    "projection_ratio",
    "relative_residual", "StructuralAuxiliarySpace",
    "StructuralExpansionTransfer",
]
