"""B-only softmax Gauss-Newton in the temporary MLP extension space."""
from __future__ import annotations

from dataclasses import dataclass, replace
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp
from projection.cg import conjugate_gradient
from adapters.deit_mlp_growth import DeitMLPGrowthAdapter


@dataclass(frozen=True)
class OptEConfig:
    inner_steps: int = 5
    cg_iterations: int = 80
    cg_tolerance: float = 1e-4
    tolerance: float = 1e-6
    damping_multipliers: tuple = (.001, .01, .1, 1., 10.)
    backtracks: tuple = (1., .5, .25)
    diagonal_probes: int = 8
    diagonal_floor: float = 1e-8
    scales: tuple = (.1, .25, .5, 1.)

    def __post_init__(self):
        if not (1 <= self.inner_steps <= 5 and self.cg_iterations > 0 and self.diagonal_probes > 0 and
                self.tolerance >= 0 and self.cg_tolerance > 0 and self.diagonal_floor > 0 and
                self.damping_multipliers and all(v > 0 for v in self.damping_multipliers) and
                self.backtracks and all(0 < v <= 1 for v in self.backtracks) and
                self.scales and all(v > 0 for v in self.scales)):
            raise ValueError('invalid Opt-E configuration')


def gn_system(logits_fn, point, labels):
    """Mean-CE RHS and matrix-free J^T F J; no dense sample Jacobian."""
    logits, pullback = vjp(logits_fn, point)
    probabilities = logits.softmax(-1).detach()
    target = F.one_hot(labels, logits.shape[-1]).to(logits)
    rhs = pullback((target - probabilities) / labels.numel())[0].detach()

    def matvec(direction):
        _, image = jvp(logits_fn, (point,), (direction,))
        fisher_image = probabilities * (image - (probabilities * image).sum(-1, keepdim=True))
        return pullback(fisher_image / labels.numel())[0].detach()
    return rhs, matvec


def optimize_B(logits_fit, logits_val, initial_B, fit_labels, val_labels, *, config=OptEConfig(), seed=0):
    B = initial_B.detach().clone()
    generator = torch.Generator(device=B.device).manual_seed(seed)
    initial_loss = float(F.cross_entropy(logits_val(B), val_labels))
    losses, steps, attempts = [initial_loss], [], []
    for inner in range(config.inner_steps):
        rhs, curvature = gn_system(logits_fit, B, fit_labels)
        estimates = []
        for _ in range(config.diagonal_probes):
            signs = torch.randint(0, 2, B.shape, device=B.device, generator=generator).to(B) * 2 - 1
            estimates.append(float((signs * curvature(signs)).mean()))
        diagonal_mean = max(config.diagonal_floor, sum(estimates) / len(estimates))
        accepted = False
        for multiplier in config.damping_multipliers:
            damping = multiplier * diagonal_mean
            def flat_matvec(value):
                direction = value.reshape_as(B)
                return (curvature(direction) + damping * direction).reshape(-1)
            cg = conjugate_gradient(flat_matvec, rhs.reshape(-1), max_iter=config.cg_iterations,
                                    tolerance=config.cg_tolerance)
            step = cg.solution.reshape_as(B)
            attempt = {'inner': inner, 'lambda': damping, 'multiplier': multiplier,
                       'mean_diagonal_estimate': diagonal_mean, 'cg_converged': cg.converged,
                       'cg_relative_residual': cg.relative_residual, 'accepted': False}
            attempts.append(attempt)
            if not torch.isfinite(step).all() or not torch.isfinite(torch.tensor(cg.relative_residual)):
                continue
            for tau in config.backtracks:
                trial = (B + tau * step).detach()
                value = float(F.cross_entropy(logits_val(trial), val_labels))
                if torch.isfinite(torch.tensor(value)) and losses[-1] - value > config.tolerance:
                    # Undamped quadratic model for the actually accepted displacement.
                    displacement = tau * step
                    predicted = float((rhs * displacement).sum() - .5 * (displacement * curvature(displacement)).sum())
                    B = trial
                    losses.append(value)
                    attempt['accepted'] = True
                    steps.append({'inner': inner, 'lambda': damping, 'tau': tau,
                                  'newton_decrement_predicted': predicted})
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            break
    return B, {'status': 'ok' if steps else 'opt_e_failed', 'inner_val_losses': losses,
               'steps': steps, 'lambda': [row['lambda'] for row in steps], 'attempts': attempts,
               'newton_decrement_predicted': [row['newton_decrement_predicted'] for row in steps],
               'prediction_model': 'undamped_mean_CE_quadratic_at_accepted_displacement',
               'B_dimension': B.numel(), 'val_role': 'inner_step_selection_not_unbiased_heldout',
               'loss_reduction': 'mean', 'seed': seed}


def optimize_candidate(model, candidate, opt_fit, opt_val, *, config=OptEConfig(), seed=0):
    """Functional MLP replacement: no hooks, auxiliary Parameters or model mutation."""
    mlp = DeitMLPGrowthAdapter.resolve_site(model, candidate.module_name)
    site = candidate.module_name
    parameters = {name: p.detach() for name, p in model.named_parameters()}
    buffers = dict(model.named_buffers())
    original_w1, original_b1, original_w2 = mlp.fc1.weight.detach(), mlp.fc1.bias.detach(), mlp.fc2.weight.detach()
    w1 = torch.cat((original_w1, candidate.A), 0)
    b1 = torch.cat((original_b1, candidate.a), 0)
    def logits(batch):
        def evaluate(B):
            changed = {**parameters, f'{site}.fc1.weight': w1, f'{site}.fc1.bias': b1,
                       f'{site}.fc2.weight': torch.cat((original_w2, B), 1)}
            return functional_call(model, (changed, buffers), (batch[0],))
        return evaluate
    B, record = optimize_B(logits(opt_fit), logits(opt_val), torch.zeros_like(candidate.B),
                            opt_fit[1], opt_val[1], config=config, seed=seed)
    return replace(candidate, B=B), record
