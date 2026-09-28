from torch import nn

from probe.pretrained_gromo import _first


def test_first_accepts_direct_or_nested_module():
    direct = nn.BatchNorm2d(4)
    assert _first(direct, nn.BatchNorm2d) is direct

    nested_batch_norm = nn.BatchNorm2d(4)
    nested = nn.Sequential(nn.ReLU(), nested_batch_norm)
    assert _first(nested, nn.BatchNorm2d) is nested_batch_norm
