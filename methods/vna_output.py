"""Reusable float64 output-space GGN factorization."""
import time
import torch

OUTPUT_EXACT_MAX = 4096


def choose_solver(requested, n_out, maximum=OUTPUT_EXACT_MAX):
    if requested != 'auto':
        return requested, 'explicit CLI choice'
    return ('output_exact', f'n_out <= OUTPUT_EXACT_MAX={maximum}') if n_out <= maximum else (
        'cg', f'n_out > OUTPUT_EXACT_MAX={maximum}')


class OutputFactor:
    r"""Let A=RJ, F=AᵀA and M=AAᵀ, with R=H^(1/2).

    (AᵀA+μI)Aᵀ=Aᵀ(AAᵀ+μI), hence inverse(F+μI)JᵀR
    =JᵀR inverse(M+μI). For e in range(H), s=R⁺e and
    z=HJ inverse(F+μI)Jᵀe=RM inverse(M+μI)s.
    Likewise d=JᵀR inverse(M+μI)(R J_E w). R⁺ uses the
    numerical positive eigenspace, including singular CE/KD metrics.
    One eigendecomposition serves all residuals, sites and both μ values.
    """
    def __init__(self, op, *, chunk_size=16, matrix=None):
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
        size = n*c
        if matrix is None:
            storage = op.logits.device
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
        self.matrix = (matrix+matrix.T)/2
        sync()
        self.build_seconds = time.perf_counter()-start
        start = time.perf_counter()
        self.eigen, self.basis = torch.linalg.eigh(self.matrix)
        sync()
        self.factor_seconds = time.perf_counter()-start
        self.range_ok = True
        self.range_error = 0.

    def apply_root(self, v):
        return torch.einsum('nij,nj->ni', self.root, v.double().reshape(self.shape)).flatten()

    def inverse(self, rhs, mu):
        return (self.basis @ ((self.basis.T @ rhs.flatten().to(self.basis))/(self.eigen.clamp_min(0)+mu))).to(self.root)

    def z(self, residual, mu):
        e = residual.double().reshape(self.shape)
        projected = torch.einsum('nij,nj->ni', self.projector, e)
        error = float((e-projected).norm()/e.norm().clamp_min(1e-300))
        self.range_error = max(self.range_error, error)
        self.range_ok &= error <= 1e-10
        s = torch.einsum('nij,nj->ni', self.pinv, e).flatten()
        coefficients = self.basis.T @ s.to(self.basis)
        value = self.basis @ (self.eigen.clamp_min(0)/(self.eigen.clamp_min(0)+mu)*coefficients)
        return self.apply_root(value.to(self.root)).reshape_as(residual)

    def d(self, products, w, mu):
        return products.JT(self.apply_root(self.inverse(self.apply_root(products.JEv(w)), mu))).double()

    def metadata(self):
        return dict(solver='output_exact', n_out=self.matrix.shape[0], M_build_seconds=self.build_seconds,
                    factor_seconds=self.factor_seconds, M_device=str(self.matrix.device), M_memory_bytes=self.matrix.numel()*8,
                    min_eig=float(self.eigen.min()), max_eig=float(self.eigen.max()),
                    residual_in_range_H=self.range_ok, residual_range_relative_error=self.range_error)

    def spectrum(self, residual, mu, top_k=20):
        import json
        energy = (self.basis.T @ residual.flatten().to(self.basis)).square()
        return dict(K_top_eigenvalues=json.dumps(self.eigen.flip(0)[:top_k].clamp_min(0).tolist()),
                    residual_energy_lambda_lt_mu=float(energy[self.eigen < mu].sum()/energy.sum()) if energy.sum() else 0.,
                    K_spectrum_method='exact_output_GGN')
