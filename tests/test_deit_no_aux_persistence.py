import copy
import os
import pytest
import torch
from adapters import DeitMLPGrowthAdapter
from probe import CandidateExpansionProbe


def test_probe_preserves_state_gradients_modes_and_parameter_identity(deit_small, deit_batches):
    if os.environ.get("REQUIRE_DEIT_INTEGRATION") == "1":
        from gromo.modules.linear_growing_module import LinearGrowingModule  # noqa: F401
    else:
        pytest.importorskip("gromo")
    before = copy.deepcopy(deit_small.state_dict())
    parameters = dict(deit_small.named_parameters())
    deit_small.train(); deit_small.blocks[0].norm1.eval()
    modes = {name: module.training for name, module in deit_small.named_modules()}
    for p in deit_small.parameters():
        p.grad = torch.ones_like(p)
    rng = torch.get_rng_state().clone()
    candidate = DeitMLPGrowthAdapter().propose_auxiliary_growth(
        model=deit_small, site="blocks.0.mlp", batches=deit_batches[:2], rank=2)
    assert torch.equal(torch.get_rng_state(), rng)
    CandidateExpansionProbe()(deit_small, candidate=candidate, batch=deit_batches[2], gate=.05)
    assert set(deit_small.state_dict()) == set(before)
    assert all(torch.equal(value, before[name]) for name, value in deit_small.state_dict().items())
    assert all(dict(deit_small.named_parameters())[name] is p for name, p in parameters.items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in deit_small.parameters())
    assert {name: module.training for name, module in deit_small.named_modules()} == modes
    assert "forward" not in deit_small.blocks[0].mlp.__dict__
    assert not deit_small.blocks[0].mlp._forward_hooks


def test_save_resume_reproduces_selected_site_and_probe_batches(deit_small, deit_batches, deit_native_candidate, tmp_path):
    from types import SimpleNamespace
    from adapters.deit_cp_adapter import CPConfig, probe_indices
    from experiments.run_plateau_comparison import select_by_expansion_gain
    from experiments.shared_protocol import rng_state, restore_rng
    from models import DeiTTinyCifar
    config = CPConfig(rank=2)
    torch.save({"model": deit_small.state_dict(), "rng": rng_state(), "config": deit_small.config,
                "probe_indices": probe_indices(list(range(1000)), 1, config)}, tmp_path / "fork.pt")
    first, record = select_by_expansion_gain(deit_small, deit_batches[:2], deit_batches[2:], config, torch.device("cpu"))
    state = torch.load(tmp_path / "fork.pt", weights_only=False)
    restored = DeiTTinyCifar(**state["config"]).double()
    restored.load_state_dict(state["model"]); restore_rng(state["rng"])
    second, resumed = select_by_expansion_gain(restored, deit_batches[:2], deit_batches[2:], config, torch.device("cpu"))
    assert first.module_name == second.module_name
    assert record["site_evaluations"] == resumed["site_evaluations"]
    assert state["probe_indices"] == probe_indices(list(range(1000)), 1, config)
