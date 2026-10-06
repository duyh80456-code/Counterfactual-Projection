import torch
from probe import CandidateExpansionProbe
from projection.metrics import cosine_alignment


def test_gamma_zero_and_nonzero_direction(deit_small, deit_batches, deit_native_candidate):
    deit_small.eval()
    inputs = deit_batches[2][0]
    with torch.no_grad():
        base = deit_small(inputs)
        with deit_native_candidate.virtual_direction(0):
            expanded = deit_small(inputs)
    assert torch.equal(base, expanded)
    assert (expanded - base).abs().max() < 1e-6
    signal = CandidateExpansionProbe()(deit_small, candidate=deit_native_candidate,
                                       batch=deit_batches[2], gate=.05)
    assert signal.is_structural_expansion
    assert signal.delta_logits.norm() > 0


def test_finite_difference_stability(deit_small, deit_batches, deit_native_candidate):
    probe = CandidateExpansionProbe()
    first = probe(deit_small, candidate=deit_native_candidate, batch=deit_batches[2], gate=.05)
    second = probe(deit_small, candidate=deit_native_candidate, batch=deit_batches[2], gate=.025)
    assert cosine_alignment(first.delta_logits, second.delta_logits) > .99


def test_matches_explicitly_widened_mlp(deit_small, deit_batches, deit_native_candidate):
    import copy
    from torch import nn
    mlp = deit_small.blocks[0].mlp
    clone = copy.deepcopy(deit_small)
    grown = clone.blocks[0].mlp
    candidate = deit_native_candidate
    with torch.no_grad():
        incoming = torch.cat((mlp.fc1.weight, candidate.A))
        bias = torch.cat((mlp.fc1.bias, candidate.a))
        outgoing = torch.cat((mlp.fc2.weight, .05 * candidate.B), dim=1)
    grown.fc1 = nn.Linear(incoming.shape[1], incoming.shape[0], dtype=incoming.dtype)
    grown.fc2 = nn.Linear(outgoing.shape[1], outgoing.shape[0], dtype=outgoing.dtype)
    with torch.no_grad():
        grown.fc1.weight.copy_(incoming); grown.fc1.bias.copy_(bias)
        grown.fc2.weight.copy_(outgoing); grown.fc2.bias.copy_(mlp.fc2.bias)
    deit_small.eval(); clone.eval()
    with torch.no_grad(), candidate.virtual_direction(.05):
        assert torch.allclose(deit_small(deit_batches[2][0]), clone(deit_batches[2][0]), atol=1e-12, rtol=1e-10)
