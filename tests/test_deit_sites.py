import pytest
import torch
from models import DeiTTinyCifar
from adapters import DeitMLPGrowthAdapter


def test_twelve_homogeneous_mlp_sites_and_geometry():
    model = DeiTTinyCifar()
    sites = DeitMLPGrowthAdapter.enumerate_sites(model)
    assert sites == tuple(f"blocks.{index}.mlp" for index in range(12))
    assert model.num_patches == 64
    assert model.pos_embed.shape == (1, 65, 192)
    assert not hasattr(model, "dist_token")
    assert not any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    for site in sites:
        mlp = dict(model.named_modules())[site]
        assert (mlp.fc1.in_features, mlp.fc1.out_features, mlp.fc2.out_features) == (192, 768, 192)
        assert len(DeitMLPGrowthAdapter.original_mlp_parameters(model, site)) == 4
    assert not any("attn" in site or "boundary" in site for site in sites)
    with pytest.raises(KeyError):
        DeitMLPGrowthAdapter.resolve_site(model, "blocks.0.attn")


def test_default_how_configuration_and_probe_batch_replay():
    from adapters.deit_cp_adapter import CPConfig, probe_indices
    config = CPConfig()
    assert (config.rank, config.projection_samples, config.cg_iterations) == (8, 64, 200)
    assert config.scales == (.025, .05, .1, .2)
    assert config.probe_epsilon == .05
    first = probe_indices(list(range(1000)), 1, config)
    assert first == probe_indices(list(range(1000)), 1, config)
    partitions = [first["statistics"], *first["where"], first["projection"], first["gate"]]
    assert sum(map(len, partitions)) == len(set(sum(partitions, [])))
    assert first != probe_indices(list(range(1000)), 2, config)
