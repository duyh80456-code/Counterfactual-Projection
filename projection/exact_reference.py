"""Dense small-problem SVD reference; never instantiate a full DeiT Jacobian."""
import torch


def svd_reference(jacobian, target, damping, *, max_elements=2_000_000):
    if jacobian.ndim!=2 or jacobian.numel()>max_elements or damping<0:
        raise ValueError('SVD reference is restricted to small matrices and nonnegative damping')
    J=jacobian.double();d=target.reshape(-1).double().to(J)
    if d.numel()!=J.shape[0]:raise ValueError('target/Jacobian dimension mismatch')
    U,s,Vh=torch.linalg.svd(J,full_matrices=False)
    threshold=torch.finfo(J.dtype).eps*max(J.shape)*(float(s.max()) if s.numel() else 0.)
    keep=s>threshold
    coefficients=U.T@d
    inverse=torch.where(keep,s.reciprocal(),torch.zeros_like(s))
    least_squares=Vh.T@(inverse*coefficients)
    ridge_factors=(s/(s.square()+damping)) if damping>0 else inverse
    ridge=Vh.T@(ridge_factors*coefficients)
    floor=d-J@least_squares;shrink=J@(least_squares-ridge)
    norm=d.norm().clamp_min(1e-12)
    return {'least_squares_delta':least_squares,'ridge_delta':ridge,
        'minimum_residual':float(floor.norm()/norm),'ridge_residual':float((d-J@ridge).norm()/norm),
        'regularization_error':float(shrink.norm()/norm),'numerical_rank':int(keep.sum()),
        'singular_values':s,'gram_trace_mean':float(J.square().sum()/J.shape[0])}


def solver_error(jacobian,target,approximate_delta,reference):
    J=jacobian.double();d=target.reshape(-1).double().to(J);delta=approximate_delta.reshape(-1).double().to(J)
    return {'solver_functional_error':float((J@(delta-reference['ridge_delta'])).norm()/d.norm().clamp_min(1e-12)),
        'solver_parameter_error':float((delta-reference['ridge_delta']).norm()),
        'approximate_residual':float((d-J@delta).norm()/d.norm().clamp_min(1e-12))}
