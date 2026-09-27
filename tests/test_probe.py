import torch
from torch import nn

from probe import VirtualExpansionProbe


class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.head = nn.Linear(4, 3)

    def forward(self, inputs):
        features = torch.relu(self.bn(self.conv(inputs))).mean((2, 3))
        return self.head(features)


def test_probe_is_nonzero_and_transactional():
    torch.manual_seed(4)
    model = TinyConv().train()
    inputs = torch.randn(5, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 1, 0])
    original = {name: value.clone() for name, value in model.state_dict().items()}
    stale_grad = torch.randn_like(model.head.bias)
    model.head.bias.grad = stale_grad.clone()

    signal = VirtualExpansionProbe(scale=0.02)(
        model, block="conv", rank=2, batch=(inputs, labels))

    assert signal.rank == 2
    assert signal.delta_logits.norm() > 0
    assert signal.delta_feature.shape == (5, 4, 8, 8)
    assert model.training and model.bn.training
    assert torch.equal(model.head.bias.grad, stale_grad)
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name]), name

