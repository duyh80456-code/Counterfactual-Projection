"""Reusable float64 output-space GGN factorization."""
import time
import tempfile
import weakref
import torch

OUTPUT_EXACT_MAX = 4096


def choose_solver(requested, n_out, maximum=OUTPUT_EXACT_MAX):
    if requested != 'auto':
        return requested, 'explicit CLI choice'
    return 'output_exact', 'auto uses exact output-space solve'


class OutputFactor:
    r"""Let A=RJ, F=AᵀA and M=AAᵀ, with R=H^(1/2).

    (AᵀA+μI)Aᵀ=Aᵀ(AAᵀ+μI), hence inverse(F+μI)JᵀR
    =JᵀR inverse(M+μI). For e in range(H), s=R⁺e and
    z=HJ inverse(F+μI)Jᵀe=RM inverse(M+μI)s.
    Likewise d=JᵀR inverse(M+μI)(R J_E w). R⁺ uses the
    numerical positive eigenspace, including singular CE/KD metrics.
    One eigendecomposition serves all residuals, sites and both μ values.
    """
    _large_factors = weakref.WeakSet()

    def __init__(self, op, *, chunk_size=16, matrix=None, factor_mode=None):
        def sync():
            if op.logits.is_cuda: torch.cuda.synchronize(op.logits.device)
        sync()
        start = time.perf_counter()
        self.shape = op.logits.shape
        n, c = self.shape
        eye = torch.eye(c, device=op.logits.device, dtype=torch.float64)
        blocks = torch.stack([op.loss_spec.metric_apply(op.logits, eye[j].expand(n, c), op.ctx)
                              for j in range(c)], -1)
        vals, q = torch.linalg.eigh((blocks+blocks.transpose(-1, -2))/2)
        positive = vals > vals.abs().amax(-1, keepdim=True)*c*torch.finfo(vals.dtype).eps
        root = torch.where(positive, vals.clamp_min(0).sqrt(), 0.)
        self.root = (q*root.unsqueeze(-2)) @ q.transpose(-1, -2)
        self.pinv = (q*torch.where(positive, root.clamp_min(1e-300).reciprocal(), 0.).unsqueeze(-2)) @ q.transpose(-1, -2)
        self.projector = (q*positive.double().unsqueeze(-2)) @ q.transpose(-1, -2)
        if factor_mode == 'cholesky' and matrix is not None:
            matrix = matrix.cpu()
        size = n*c
        self.size = size
        self._matrix = None
        self._parked = None
        if size > OUTPUT_EXACT_MAX:
            self._park_others()
            self._large_factors.add(self)
        if matrix is None:
            storage = torch.device('cpu') if size > OUTPUT_EXACT_MAX else op.logits.device
            if storage.type == 'cuda' and size*size*8*6 > torch.cuda.mem_get_info(storage)[0]*.6:
                storage = torch.device('cpu')
            matrix = torch.empty((size, size), device=storage, dtype=torch.float64)
            # Chunked vmap avoids storing J (n_out by n_parameters).
            for offset in range(0, size, chunk_size):
                from experiments.vna_resume import measurement_update
                measurement_update(solve='M:build', column=offset)
                ids = torch.arange(offset, min(size, offset+chunk_size), device=op.logits.device)
                coordinates = torch.nn.functional.one_hot(ids, size).double()
                columns = torch.vmap(lambda v: self.apply_root(op.p.K(self.apply_root(v))))(coordinates)
                matrix[:, offset:offset+len(ids)] = columns.T.to(matrix)
        # Average corresponding blocks without allocating another square matrix.
        block_size = 512
        for i in range(0, size, block_size):
            for j in range(i, size, block_size):
                a = matrix[i:i+block_size, j:j+block_size]
                b = matrix[j:j+block_size, i:i+block_size]
                average = (a + b.T) * .5
                a.copy_(average)
                b.copy_(average.T)
        self._matrix = matrix
        self.matrix_device = str(matrix.device)
        self.factor_mode = factor_mode or ('eigh' if size <= OUTPUT_EXACT_MAX else 'cholesky')
        self.factor_error = None
        self.active_mu = None
        self.cholesky = None
        self.factor_times = {}
        sync()
        self.build_seconds = time.perf_counter()-start
        start = time.perf_counter()
        if self.factor_mode == 'eigh':
            self.eigen, self.basis = torch.linalg.eigh(self.matrix)
        sync()
        self.factor_seconds = time.perf_counter()-start
        print(f'M build n_out={size} seconds={self.build_seconds} peak RSS={self.peak_rss()}', flush=True)
        self.range_ok = True
        self.range_error = 0.

    def _park_others(self):
        # Preserve inactive matrices/factors on disk only when host RAM cannot
        # also hold this matrix and its factor (2 * n^2 * 8 bytes + 2 GB margin).
        try:
            import psutil
            if psutil.virtual_memory().available > 2*self.size*self.size*8 + 2*1024**3:
                return
        except ImportError:
            pass
        for other in list(self._large_factors):
            if other is not self and other._matrix is not None:
                print(f'output factor park n_out={other.size} (host RAM low)', flush=True)
                other._parked = tempfile.TemporaryFile()
                torch.save((other._matrix, other.cholesky), other._parked)
                other._matrix = None
                other.cholesky = None

    def _activate(self):
        if self.size > OUTPUT_EXACT_MAX:
            self._park_others()
        if self._parked is not None:
            self._parked.seek(0)
            self._matrix, self.cholesky = torch.load(self._parked, weights_only=True)
            self._parked.close()
            self._parked = None

    @property
    def matrix(self):
        self._activate()
        return self._matrix

    def release(self):
        if self._parked is not None:
            self._activate()
        self.cholesky = None
        self.active_mu = None

    def prepare(self, mu):
        self._activate()
        if self.factor_mode == 'eigh' or self.active_mu == mu:
            return
        self.release()
        self.active_mu = mu
        self.factor_error = None
        start = time.perf_counter()
        diagonal = self.matrix.diagonal()
        diagonal.add_(mu)
        try:
            factor, info = torch.linalg.cholesky_ex(self.matrix)
            if int(info):
                self.factor_error = f'cholesky_ex info={int(info)}'
                del factor
            else:
                self.cholesky = factor
        except RuntimeError as error:
            self.factor_error = str(error)
        finally:
            diagonal.sub_(mu)
        self.factor_seconds = time.perf_counter()-start
        self.factor_times[mu] = self.factor_seconds
        print(f'output factor mu={mu} seconds={self.factor_seconds} peak RSS={self.peak_rss()}', flush=True)

    @staticmethod
    def peak_rss():
        import resource
        import sys
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)

    def apply_root(self, v):
        return torch.einsum('nij,nj->ni', self.root, v.double().reshape(self.shape)).flatten()

    def inverse(self, rhs, mu):
        self.prepare(mu)
        if self.factor_mode == 'cholesky':
            if self.factor_error:
                return torch.full_like(rhs, float('nan')).to(self.root)
            return torch.cholesky_solve(rhs.flatten().to(self.matrix).unsqueeze(1), self.cholesky).flatten().to(self.root)
        return (self.basis @ ((self.basis.T @ rhs.flatten().to(self.basis))/(self.eigen.clamp_min(0)+mu))).to(self.root)

    def z(self, residual, mu):
        e = residual.double().reshape(self.shape)
        projected = torch.einsum('nij,nj->ni', self.projector, e)
        error = float((e-projected).norm()/e.norm().clamp_min(1e-300))
        self.range_error = max(self.range_error, error)
        self.range_ok &= error <= 1e-10
        s = torch.einsum('nij,nj->ni', self.pinv, e).flatten()
        if self.factor_mode == 'cholesky':
            return self.apply_root(s-mu*self.inverse(s, mu)).reshape_as(residual)
        coefficients = self.basis.T @ s.to(self.basis)
        value = self.basis @ (self.eigen.clamp_min(0)/(self.eigen.clamp_min(0)+mu)*coefficients)
        return self.apply_root(value.to(self.root)).reshape_as(residual)

    def d(self, products, w, mu):
        return products.JT(self.apply_root(self.inverse(self.apply_root(products.JEv(w)), mu))).double()

    def metadata(self):
        return dict(solver='output_exact', n_out=self.size, M_build_seconds=self.build_seconds,
                    factor_mode=self.factor_mode, factor_error=self.factor_error,
                    peak_rss_bytes=self.peak_rss(), peak_cuda_bytes=torch.cuda.max_memory_allocated(self.root.device) if self.root.is_cuda else 0,
                    factor_seconds=self.factor_seconds, M_device=self.matrix_device, M_memory_bytes=self.size*self.size*8,
                    min_eig=float(self.eigen.min()) if self.factor_mode == 'eigh' else None,
                    max_eig=float(self.eigen.max()) if self.factor_mode == 'eigh' else None,
                    residual_in_range_H=self.range_ok, residual_range_relative_error=self.range_error)

    def spectrum(self, residual, mu, top_k=20):
        if self.size > OUTPUT_EXACT_MAX:
            return {'spectrum': 'skipped (n_out > 4096)'}
        if self.factor_mode == 'cholesky':
            return {'K_spectrum_method': 'unavailable_cholesky_group'}
        import json
        energy = (self.basis.T @ residual.flatten().to(self.basis)).square()
        return dict(K_top_eigenvalues=json.dumps(self.eigen.flip(0)[:top_k].clamp_min(0).tolist()),
                    residual_energy_lambda_lt_mu=float(energy[self.eigen < mu].sum()/energy.sum()) if energy.sum() else 0.,
                    K_spectrum_method='exact_output_GGN')
