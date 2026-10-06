import copy
import torch

from adapters import DeitMLPGrowthAdapter
from adapters.deit_cp_adapter import reset_adam_moments
from probe import CandidateExpansionProbe
from projection import FunctionalProjector
from methods.e_projection import candidate_projection_parameter_names


def test_projection_only_updates_original_selected_mlp(deit_small, deit_batches, deit_native_candidate):
    before = {name: value.clone() for name, value in deit_small.state_dict().items()}
    signal = CandidateExpansionProbe()(deit_small, candidate=deit_native_candidate,
                                       batch=deit_batches[2], gate=.05)
    scope = DeitMLPGrowthAdapter.original_mlp_parameters(deit_small, "blocks.0.mlp")
    assert candidate_projection_parameter_names(deit_small, deit_native_candidate) == scope
    projection = FunctionalProjector(max_iter=8, preconditioner_probes=0).project(
        deit_small, deit_batches[2][0], signal.delta_logits, block="blocks.0.mlp", parameter_names=scope)
    assert torch.isfinite(projection.fitted_delta).all()
    assert set(projection.parameter_delta) == set(scope)
    projection.apply_(deit_small, .05)
    changed = {name for name, value in deit_small.state_dict().items() if not torch.equal(value, before[name])}
    assert changed and changed.issubset(scope)
    assert not deit_small.blocks[0].mlp._forward_hooks


def test_adam_moments_reset_only_changed_tensors(deit_small, deit_batches):
    optimizer = torch.optim.AdamW(deit_small.parameters(), lr=.001, amsgrad=True)
    loss = torch.nn.functional.cross_entropy(deit_small(deit_batches[0][0]), deit_batches[0][1])
    loss.backward(); optimizer.step()
    parameters = dict(deit_small.named_parameters())
    before = copy.deepcopy(optimizer.state_dict())
    target = "blocks.0.mlp.fc1.weight"
    zero = "blocks.0.mlp.fc1.bias"
    reset = reset_adam_moments(optimizer, deit_small, {
        target: torch.ones_like(parameters[target]), zero: torch.zeros_like(parameters[zero])})
    assert reset == [target]
    for index, (name, parameter) in enumerate(parameters.items()):
        state = optimizer.state[parameter]
        old = before["state"][index]
        assert torch.equal(state["step"], old["step"])
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name == target:
                assert torch.count_nonzero(state[key]) == 0
            else:
                assert torch.equal(state[key], old[key])
    assert optimizer.param_groups[0]["lr"] == .001
