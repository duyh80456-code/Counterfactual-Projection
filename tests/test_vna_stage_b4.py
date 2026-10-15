"""Quarter-width imported Gromo counts and zero-B/adjoint checks; Kaggle only."""
import pytest
import torch
from experiments.stage_b4 import build_model, sites_for
from tests.test_vna_stage_b12 import check_site


@pytest.mark.parametrize('seed', [0, 1])
def test_b4_parameter_count_and_sites(seed):
    torch.manual_seed(seed)
    from dual_growth.adapters import GromoResNet18
    old = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        model = build_model().double()
        native = GromoResNet18(num_classes=100, start_width=.25, device='cpu', use_preactivation=False)
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in native.parameters())
        assert [int(r.module.hidden_neurons) for r in model.growing_blocks()] == [16, 16, 32, 32, 64, 64, 128, 128]
        sites = sites_for(model); assert len(sites) == 8
        for site in sites:
            check_site(model, site, torch.randn(1, 3, 8, 8, dtype=torch.float64))
    finally: torch.set_num_threads(old)
