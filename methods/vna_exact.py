"""T29 output-space solves and T31 signed Woodbury updates (float64)."""
import time
import torch


def gram(products, chunk=64, vjp_chunk=16, jvp_chunk=16):
    start = time.perf_counter()
    m = products._call({}).numel()
    ref = products.parameter_refs[0][1]
    G = torch.empty((m, m), dtype=torch.float64, device=ref.device)
    for lo in range(0, m, chunk):
        ids = torch.arange(lo, min(lo+chunk, m), device=ref.device)
        E = torch.nn.functional.one_hot(ids, m).to(ref)
        # Bound backward activation memory separately from the stored vector block.
        tangents = torch.vmap(products.JT, chunk_size=vjp_chunk)(E)
        G[:, lo:lo+len(ids)] = torch.vmap(products.Jv, chunk_size=jvp_chunk)(tangents).double().T
        del tangents
        if m >= 12800 and G.is_cuda: torch.cuda.synchronize(G.device)
        if m >= 12800 and time.perf_counter()-start > 600:
            raise RuntimeError(f'GRAM_LIMIT: m={m} gram_seconds={time.perf_counter()-start}; review required')
    G = (G+G.T)*.5
    if G.is_cuda: torch.cuda.synchronize(G.device)
    elapsed = time.perf_counter()-start
    if m >= 12800 and elapsed > 600:
        raise RuntimeError(f'GRAM_LIMIT: m={m} gram_seconds={elapsed}; review required')
    return G, dict(gram_seconds=time.perf_counter()-start, m=m, gram_dtype=str(G.dtype),
                   parameter_dtype=str(ref.dtype), product_vector_dtype=str(ref.dtype), gram_chunk=chunk,
                   gram_vjp_chunk=vjp_chunk, gram_jvp_chunk=jvp_chunk)


def columns(action, indices, size, ref, chunk=16):
    result = []
    for lo in range(0, len(indices), chunk):
        ids = indices[lo:lo+chunk]
        E = torch.nn.functional.one_hot(ids, size).to(ref)
        result.append(torch.vmap(action)(E).double().T)
    return torch.cat(result, 1) if result else torch.empty((0, 0), device=ref.device, dtype=torch.float64)


class OutputSolve:
    def __init__(self, G, blocks, mu):
        self.G, self.mu = G, float(mu)
        blocks = blocks.double()
        self.blocks = blocks
        n, c, _ = blocks.shape
        eig, Q = torch.linalg.eigh(blocks)
        threshold = torch.finfo(torch.float64).eps*c*eig.abs().amax(-1, keepdim=True)
        keep = eig > threshold
        # Block factors omit the softmax null directions without projecting the Gram.
        self.L = torch.zeros((n*c, int(keep.sum())), dtype=G.dtype, device=G.device)
        offset = 0
        for i in range(n):
            k = int(keep[i].sum())
            self.L[i*c:(i+1)*c, offset:offset+k] = Q[i, :, keep[i]]*eig[i, keep[i]].sqrt()
            offset += k
        K = self.L.T @ G @ self.L
        K = (K+K.T)*.5
        K.diagonal().add_(self.mu)
        self.chol, info = torch.linalg.cholesky_ex(K)
        if bool(info.any()) or not bool(torch.isfinite(self.chol).all()):
            raise torch.linalg.LinAlgError('chol_fail')
        self.inverse = None
        self.solves, self._q_key = 0, None

    def metric(self, v):
        shape = v.shape
        return torch.einsum('nij,nj->ni', self.blocks, v.to(self.blocks).reshape(self.blocks.shape[:2])).reshape(shape)

    def A_solve(self, rhs):
        return self.inverse @ rhs if self.inverse is not None else torch.cholesky_solve(rhs, self.chol)

    def solve(self, c):
        self.solves += 1
        return (c-self.L @ self.A_solve((self.L.T @ (self.G @ c))[:, None])[:, 0])/self.mu

    def q(self, c):
        key = (id(c), c._version, self.G._version)
        if self._q_key != key:
            self._q_value = -.5*float(c @ (self.G @ self.solve(c)))
            self._q_key, self._q_c = key, c
        return self._q_value

    def swapped(self, remove, add):
        return WoodburySwap(self, remove, add)


class WoodburySwap:
    def __init__(self, base, remove, add):
        self.base, self.remove, self.add = base, remove, add
        W = torch.cat((remove, add), 1)
        self.U = base.L.T @ W
        self.AU = base.A_solve(self.U)
        signs = torch.cat((-torch.ones(remove.shape[1], device=W.device, dtype=W.dtype),
                            torch.ones(add.shape[1], device=W.device, dtype=W.dtype)))
        self.small = torch.diag(signs)+self.U.T @ self.AU

    def Gv(self, v):
        return self.base.G@v-self.remove@(self.remove.T@v)+self.add@(self.add.T@v)

    def A_solve(self, rhs):
        original = self.base.A_solve(rhs)
        return original-self.AU @ torch.linalg.solve(self.small, self.U.T @ original)

    def solve(self, c):
        self.base.solves += 1
        return (c-self.base.L @ self.A_solve((self.base.L.T @ self.Gv(c))[:, None])[:, 0])/self.base.mu

    def gamma(self, e, fu):
        eprime = e-self.base.metric(fu)
        jump = -float(e@fu)+.5*float(fu@self.base.metric(fu))
        qprime = -.5*float(eprime @ self.Gv(self.solve(eprime)))
        return -jump+self.base.q(e)-qprime

    def commit(self):
        base = self.base
        if base.inverse is None:
            base.inverse = torch.cholesky_inverse(base.chol)
            base.chol = None
        base.inverse -= self.AU @ torch.linalg.solve(self.small, self.AU.T)
        base.inverse = (base.inverse+base.inverse.T)*.5
        base.G.addmm_(self.remove, self.remove.T, beta=1., alpha=-1.)
        base.G.addmm_(self.add, self.add.T)
