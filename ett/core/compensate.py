"""C3a/C3c: compensation in the supplied (possibly regularized) Gram metric."""
import torch


def columns(units,r):return [u*r+j for u in units for j in range(r)]


def compensate(G,W,kept):
    idx=torch.as_tensor(kept,device=G.device)
    dropped=[j for j in range(len(G)) if j not in kept]
    if not dropped or not bool(W[dropped].any()):return W[idx].clone()
    # C3a: this form is bitwise identity when dropped outgoing weights are zero.
    return W[idx]+torch.linalg.solve(G[idx][:,idx],G[idx][:,dropped]@W[dropped])


def g_exact(G,W,dropped,inverse=None):
    if not dropped:return 0.
    if not bool(W[dropped].any()):return 0.
    M=torch.cholesky_inverse(torch.linalg.cholesky(G)) if inverse is None else inverse
    return float((W[dropped]*torch.linalg.solve(M[dropped][:,dropped],W[dropped])).sum())


def data_residual(G,W,kept,compensated):
    delta=-W.clone();delta[kept]+=compensated
    return float((delta*(G@delta)).sum())
