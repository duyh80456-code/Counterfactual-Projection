"""Per-sample sufficient margin bounds; diagnostics, not generalization guarantees."""
import torch


def margins(logits,labels):
    other=logits.clone();other.scatter_(1,labels[:,None],float('-inf'))
    return logits.gather(1,labels[:,None]).squeeze(1)-other.max(-1).values


def margin_transfer(base,expanded,linear_delta,actual,labels):
    if not (base.shape==expanded.shape==linear_delta.shape==actual.shape):raise ValueError('logit shapes differ')
    me=margins(expanded,labels);mb=margins(base,labels);ma=margins(actual,labels)
    fit=(expanded-base-linear_delta).abs().amax(-1)
    nonlinear=(actual-base-linear_delta).abs().amax(-1)
    direct=(expanded-actual).abs().amax(-1)
    e_correct=expanded.argmax(-1)==labels;base_correct=base.argmax(-1)==labels;actual_correct=actual.argmax(-1)==labels
    sufficient=e_correct & (me>2*(fit+nonlinear))
    return {'e_margin_gain':float((me-mb).mean()),'margin_gain_after_projection':float((ma-mb).mean()),
        'corrected_samples':int((~base_correct & actual_correct).sum()),
        'damaged_samples':int((base_correct & ~actual_correct).sum()),
        'sufficient_bound_samples':int(sufficient.sum()),
        'bound_violations':int((sufficient & ~actual_correct).sum()),
        'fit_error_linf_mean':float(fit.mean()),'nonlinear_error_linf_mean':float(nonlinear.mean()),
        'direct_transfer_error_linf_mean':float(direct.mean()),
        'sample_e_margin':me.detach().cpu().tolist(),'sample_fit_error_linf':fit.detach().cpu().tolist(),
        'sample_nonlinear_error_linf':nonlinear.detach().cpu().tolist(),
        'bound_role':'sufficient_on_this_batch_not_a_generalization_guarantee'}
