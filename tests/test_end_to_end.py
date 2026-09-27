import torch

from methods import EProjection
from probe import VirtualExpansionProbe
from projection import FunctionalProjector
from tests.test_probe import TinyConv


def test_e_projection_keeps_deploy_parameter_count():
    torch.manual_seed(6)
    model = TinyConv().eval()
    batch = (torch.randn(3, 3, 8, 8), torch.tensor([0, 1, 2]))
    before_count = sum(parameter.numel() for parameter in model.parameters())
    method = EProjection(
        VirtualExpansionProbe(scale=0.005),
        FunctionalProjector(damping=1e-4, max_iter=20, tolerance=1e-5))
    step = method.discover(model, batch, block="conv", rank=2)
    before = model(batch[0]).detach()
    step.projection.apply_(model)
    actual = model(batch[0]).detach() - before

    assert sum(parameter.numel() for parameter in model.parameters()) == before_count
    assert step.projection.relative_residual < 0.25
    assert torch.allclose(actual, step.projection.fitted_delta,
                          atol=2e-4, rtol=0.1)

