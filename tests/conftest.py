"""Small deterministic DeiT fixtures; existing CNN tests use their own fixtures."""
import os

import pytest
import torch


@pytest.fixture
def deit_small():
    from models import DeiTTinyCifar
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = DeiTTinyCifar(num_classes=3, image_size=8, patch_size=4,
                          embed_dim=12, depth=2, num_heads=3, mlp_ratio=2).double()
    yield model
    torch.set_num_threads(previous)


@pytest.fixture
def deit_batches():
    generator = torch.Generator().manual_seed(812)
    return [(torch.randn(8, 3, 8, 8, dtype=torch.float64, generator=generator),
             torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])) for _ in range(4)]


@pytest.fixture
def deit_native_candidate(deit_small, deit_batches):
    try:
        from gromo.modules.linear_growing_module import LinearGrowingModule  # noqa: F401
    except ImportError as error:
        if os.environ.get("REQUIRE_DEIT_INTEGRATION") == "1":
            pytest.fail(f"native Gromo Linear TINY is required: {error}")
        pytest.skip(f"native Gromo not installed: {error}")
    from adapters import DeitMLPGrowthAdapter
    return DeitMLPGrowthAdapter().propose_auxiliary_growth(
        model=deit_small, site="blocks.0.mlp", batches=deit_batches[:2], rank=2)
