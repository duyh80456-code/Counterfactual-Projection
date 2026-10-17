"""final_check_v3.py -- numerical checks for every theorem in THEORY_FINAL_v3.md (float64).

Run:  python3 -I final_check_v3.py   (prints one PASS/FAIL line per check, then a summary)

Conventions follow PyCIL iCaRL: loss = mean CE over all logits + PyCIL _KD_loss on old logits
(mean over the batch, temperature T, no T^2 factor).  Per sample i:
    e_N,i = (softmax(f_i) - onehot)/N            H_CE,i = (diag p - p p^T)/N
    e_O,i = (p_s - p_t)/(N T)                    W_i    = (diag p_s - p_s p_s^T)/(N T^2)   (STUDENT p_s)
Every check is designed so that it can fail.  Checks tagged NEG are negative controls: they apply the
same test to a deliberately wrong formula and PASS only if the wrong formula is rejected.
"""
import math, torch
torch.set_default_dtype(torch.float64)
gen = torch.Generator().manual_seed(2026)
def rn(*s): return torch.randn(*s, generator=gen)
def rel(a, b): return float((a - b).norm() / max(float(b.norm()), 1e-300))
RES = []
def check(cid, desc, value, ok):
    RES.append((cid, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {cid}: {desc} -> {value}", flush=True)
def psd_sqrt(H):
    w, Q = torch.linalg.eigh((H + H.T) / 2); return Q @ torch.diag(w.clamp_min(0).sqrt()) @ Q.T
def sym(X): return (X + X.T) / 2

# ----------------------------------------------------------------------------------------------- problem
def problem(N=6, c=5, k=3, p=50, da=6, scale=3.0, T=2.0, drift=1.0, H_identity=False):
    n = N * c
    L = scale * rn(N, c); teach = L[:, :k] + drift * rn(N, k); y = torch.randint(0, c, (N,), generator=gen)
    Hb, Wb, eN, eO, chi, PS = [], [], [], [], 0.0, []
    for i in range(N):
        pc = torch.softmax(L[i], 0); ps = torch.softmax(L[i, :k] / T, 0); pt = torch.softmax(teach[i] / T, 0)
        Wi = (torch.diag(ps) - torch.outer(ps, ps)) / (N * T**2)
        Hi = (torch.diag(pc) - torch.outer(pc, pc)) / N; Hi[:k, :k] += Wi
        Hb.append(torch.eye(c) / N if H_identity else Hi); Wb.append(Wi); PS.append(ps)
        eN.append((pc - torch.nn.functional.one_hot(y[i], c)) / N)
        v = torch.zeros(c); v[:k] = (ps - pt) / (N * T); eO.append(v)
        chi += float(((ps - pt)**2 / ps).sum())
    H = torch.block_diag(*Hb); W = torch.block_diag(*Wb)
    O = torch.zeros(N * k, n)
    for i in range(N):
        for j in range(k): O[i * k + j, i * c + j] = 1
    J = rn(n, p); phi = rn(N, da)
    return dict(H=H, W=W, O=O, J=J, phi=phi, eN=torch.cat(eN), eOf=torch.cat(eO), chi=chi / N,
                L=L, teach=teach, y=y, N=N, c=c, k=k, n=n, p=p, da=da, T=T, PS=PS)

def Wfac(P):
    """exact factor with W = Lt^T Lt and null(Lt) = null(W) exactly: Lt_i = (N T^2)^-1/2 (D^1/2 - sqrt(p) p^T)"""
    blocks = [(torch.diag(ps.sqrt()) - torch.outer(ps.sqrt(), ps)) / math.sqrt(P['N'] * P['T']**2) for ps in P['PS']]
    return torch.block_diag(*blocks)

def Ja(P, a):            # n x c: Jacobian of a virtual neuron's outgoing weights, feature phi_i^T a (A4)
    s = P['phi'] @ a; return torch.kron(s.unsqueeze(1), torch.eye(P['c']))
def Nstat(P, s):         # J_a^T s = N_s^T a
    return P['phi'].T @ s.view(P['N'], P['c'])
def KE_of(P, A):
    return sum(Ja(P, A[:, j]) @ Ja(P, A[:, j]).T for j in range(A.shape[1]))

def setup(P, mu_rel=1e-2):
    H, W, O, J = P['H'], P['W'], P['O'], P['J']; I = torch.eye(P['p'])
    F = J.T @ H @ J; mu = mu_rel * float(torch.trace(F)) / P['n']
    C = O @ J; B = C.T @ W @ C; e = P['eN'] + P['eOf']
    G = torch.linalg.inv(F + mu * I)
    z = lambda r: H @ J @ (G @ (J.T @ r))
    return dict(F=F, mu=mu, C=C, B=B, e=e, G=G, z=z, I=I, g=J.T @ e, gO=J.T @ P['eOf'], eO=O @ P['eOf'])

def parts(P, S, KE, beta):
    H, J = P['H'], P['J']; zN, zO = S['z'](P['eN']), S['z'](P['eOf'])
    uN, uR = KE @ zN, KE @ zO
    d_rep = S['G'] @ J.T @ H @ uR
    Gb = torch.linalg.inv(S['F'] + beta * S['B'] + S['mu'] * S['I'])
    dN = Gb @ J.T @ H @ uN
    return dict(d_rep=d_rep, dN=dN, uN=uN, uR=uR, zN=zN, zO=zO, Gb=Gb)

def leak(P, S, x): v = S['C'] @ x; return float(v @ P['W'] @ v)
def PhiN(P, S, x, uN):
    r = P['J'] @ x - uN; return float(r @ P['H'] @ r) + S['mu'] * float(x @ x)
def Hnorm(P, u): return math.sqrt(max(float(u @ P['H'] @ u), 0.0))

def beta_cert(P, S, KE, iters=60):
    pr = parts(P, S, KE, 0.0); V_O = float(pr['zO'] @ KE @ pr['zO'])
    if V_O <= 0: return float('inf'), V_O, None
    eps_star = V_O**2 / P['chi']
    beta_suf = P['chi'] * float(pr['uN'] @ P['H'] @ pr['uN']) / V_O**2
    if leak(P, S, pr['dN']) <= eps_star: return 0.0, V_O, (eps_star, beta_suf)
    lo, hi = 0.0, beta_suf
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if leak(P, S, parts(P, S, KE, mid)['dN']) <= eps_star: hi = mid
        else: lo = mid
    return hi, V_O, (eps_star, beta_suf)

# ======================================================================================= T0: structure
def T0():
    worst_e, worst_eO, neg = 0.0, 0.0, 1e300
    for t in range(40):
        P = problem(scale=[1., 3., 6.][t % 3]); N, c, k = P['N'], P['c'], P['k']
        e = P['eN'] + P['eOf']
        for i in range(N):
            Hi = P['H'][i*c:(i+1)*c, i*c:(i+1)*c]; w, Q = torch.linalg.eigh(Hi)
            one = torch.ones(c) / math.sqrt(c)
            # null(H_i) = span(1): smallest eigvec is 1, second eigenvalue > 0
            worst_e = max(worst_e, abs(float(one @ e[i*c:(i+1)*c])) / float(e[i*c:(i+1)*c].norm()))
            worst_e = max(worst_e, 1.0 - abs(float(Q[:, 0] @ one)))
            Wi = P['W'][i*k:(i+1)*k, i*k:(i+1)*k]; eo = (P['O'] @ P['eOf'])[i*k:(i+1)*k]
            worst_eO = max(worst_eO, abs(float(torch.ones(k) @ eo)) / max(float(eo.norm()), 1e-300))
            v = rn(c); neg = min(neg, abs(float(one @ v)) / float(v.norm()))
    check('T0a', 'e_i orthogonal to 1 = null(H_i) (e in range H), and null(H_i)=span(1)', f'{worst_e:.2e}', worst_e < 1e-8)
    check('T0b', 'e_O,i orthogonal to 1 = null(W_i) (e_O in range W)', f'{worst_eO:.2e}', worst_eO < 1e-8)
    check('T0b-WIT', 'generic vectors are NOT in range(H_i): min |1^T v|/|v| over trials', f'{neg:.2e}', neg > 1e-4)
    # chi^2 closed form for ||e_O||^2_{W+}
    err, negerr = 0.0, 1e300
    for t in range(40):
        P = problem(scale=[1., 3.][t % 2], drift=2.0); S = setup(P)
        Wp = torch.linalg.pinv(P['W'], hermitian=True, rtol=1e-12)
        val = float(S['eO'] @ Wp @ S['eO']); err = max(err, abs(val - P['chi']) / P['chi'])
        # wrong: teacher probabilities in the denominator
        chi_t = 0.0
        for i in range(P['N']):
            ps = torch.softmax(P['L'][i, :P['k']] / P['T'], 0); pt = torch.softmax(P['teach'][i] / P['T'], 0)
            chi_t += float(((ps - pt)**2 / pt).sum())
        negerr = min(negerr, abs(val - chi_t / P['N']) / P['chi'])
    check('T0c', '||e_O||^2_{W+} = (1/N) sum chi^2(p_t || p_s)', f'{err:.2e}', err < 1e-9)
    check('T0c-NEG', 'chi^2 with teacher denominator rejected (min rel err)', f'{negerr:.2e}', negerr > 1e-3)
    # KD gradient and Hessian from autograd on PyCIL's _KD_loss: student probabilities
    P = problem(N=4, c=5, k=3, scale=3.0, drift=2.0); N, c, k, T = P['N'], P['c'], P['k'], P['T']
    def kd(fl):
        f = fl.view(N, c)[:, :k]
        return -(torch.softmax(P['teach'] / T, 1) * torch.log_softmax(f / T, 1)).sum() / N
    f0 = P['L'].flatten().clone()
    gr = torch.autograd.functional.jacobian(kd, f0); He = torch.autograd.functional.hessian(kd, f0)
    Wt = torch.zeros(N * c, N * c)
    for i in range(N):
        pt = torch.softmax(P['teach'][i] / T, 0); Wt[i*c:i*c+k, i*c:i*c+k] = (torch.diag(pt) - torch.outer(pt, pt)) / (N * T**2)
    OW = P['O'].T @ P['W'] @ P['O']
    lw, ln = 0.0, 0.0
    for t in range(20):
        Pq = problem(scale=[1., 3., 8.][t % 3]); Lt = Wfac(Pq)
        lw = max(lw, rel(Lt.T @ Lt, Pq['W'])); ones = torch.ones(Pq['N'] * Pq['k'])
        ln = max(ln, float((Lt @ ones).norm()) / float(Lt.norm()))
    check('T0f', 'exact factor L_i = (N T^2)^-1/2 (D^1/2 - sqrt(p) p^T): L^T L = W and L 1 = 0', f'{lw:.2e}, {ln:.2e}', lw < 1e-12 and ln < 1e-12)
    check('T0d', 'KD gradient = O^T (p_s - p_t)/(N T)', f'{rel(gr, P["eOf"]):.2e}', rel(gr, P['eOf']) < 1e-10)
    check('T0e', 'KD Hessian = (diag p_s - p_s p_s^T)/(N T^2) with STUDENT p_s', f'{rel(He, OW):.2e}', rel(He, OW) < 1e-10)
    check('T0e-NEG', 'teacher-probability KD Hessian rejected', f'{rel(He, Wt):.2e}', rel(He, Wt) > 1e-2)

# ======================================================================== T1: closed forms, decomposition
def T1():
    e_ls, e_pt, neg = 0.0, 0.0, 1e300
    for t in range(40):
        P = problem(scale=[1., 3., 6.][t % 3], p=[20, 40, 70][t % 3]); S = setup(P)
        KE = KE_of(P, rn(P['da'], 2)); beta = [0.5, 5.0, 80.0][t % 3]
        pr = parts(P, S, KE, beta); d = pr['d_rep'] + pr['dN']; u = pr['uN'] + pr['uR']
        Rh, Rw = psd_sqrt(P['H']), psd_sqrt(P['W'])
        A = torch.cat([Rh @ P['J'], math.sqrt(beta) * Rw @ S['C'], math.sqrt(S['mu']) * S['I']])
        b = torch.cat([Rh @ u, math.sqrt(beta) * Rw @ S['C'] @ pr['d_rep'], torch.zeros(P['p'])])
        d_ls = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
        e_ls = max(e_ls, rel(d, d_ls))
        neg = min(neg, rel(pr['d_rep'] + parts(P, S, KE, 0.0)['dN'], d_ls))
        # output-space push-through form, no pseudo-inverse
        Jh = torch.cat([P['J'], S['C']]); Hh = torch.block_diag(P['H'], beta * P['W']); Rr = psd_sqrt(Hh)
        Mh = Rr @ Jh @ Jh.T @ Rr; th = torch.cat([u, S['C'] @ pr['d_rep']])
        d_pt = Jh.T @ Rr @ torch.linalg.solve(Mh + S['mu'] * torch.eye(Mh.shape[0]), Rr @ th)
        e_pt = max(e_pt, rel(d_pt, d))
    check('T1a', 'd* = d_rep + d_N^beta equals the stacked least-squares minimiser', f'{e_ls:.2e}', e_ls < 1e-8)
    check('T1a-NEG', 'ignoring beta (d_rep + d_N^0) is rejected', f'{neg:.2e}', neg > 1e-3)
    check('T1b', 'output-space form d = J^T R (M+mu)^-1 R t (no pinv)', f'{e_pt:.2e}', e_pt < 1e-8)
    # range-free z vs eigenvalue-thresholded pinv form, across saturation
    worst, negw = 0.0, 0.0
    for scale in (1., 5., 20., 30.):
        for t in range(8):
            P = problem(scale=scale); S = setup(P); H, J = P['H'], P['J']; e = S['e']
            z1 = S['z'](e); R = psd_sqrt(H); K = J @ J.T; M = R @ K @ R
            z2 = R @ torch.linalg.solve(M + S['mu'] * torch.eye(P['n']), R @ (J @ S['g']))
            worst = max(worst, rel(z2, z1))
            w, Q = torch.linalg.eigh(H); keep = w > 1e-10 * float(w.max())
            Rp = Q[:, keep] @ torch.diag(w[keep].rsqrt()) @ Q[:, keep].T
            z_old = R @ M @ torch.linalg.solve(M + S['mu'] * torch.eye(P['n']), Rp @ e)
            if scale >= 20: negw = max(negw, rel(z_old, z1))
    check('T1c', "range-free z = R (M+mu)^-1 R (J g) at logit scales 1..30", f'{worst:.2e}', worst < 1e-8)
    check('T1c-NEG', 'thresholded-pinv z (old code) fails under saturation (max rel err, scale>=20)', f'{negw:.2e}', negw > 1e-3)

# =============================================================================== T2: reductions, limits
def T2():
    e0, eq, minq = 0.0, 0.0, 1e300
    for t in range(30):
        P = problem(scale=[1., 3., 6.][t % 3]); S = setup(P); KE = KE_of(P, rn(P['da'], 2))
        pr = parts(P, S, KE, 0.0); d = pr['d_rep'] + pr['dN']; z = pr['zN'] + pr['zO']
        d_vna = S['G'] @ P['J'].T @ P['H'] @ KE @ z
        e0 = max(e0, rel(d, d_vna))
        V = float(z @ KE @ z); eq = max(eq, abs(float(S['g'] @ d) - V) / V)
        Lm = P['H'] @ P['J'] @ S['G']; Q = Lm.T @ KE @ Lm
        eq = max(eq, rel(Q @ S['g'], d)); ev = torch.linalg.eigvalsh(sym(Q)); minq = min(minq, float(ev.min() / ev.max()))
    check('T2a', 'beta=0 reproduces VNA d = G J^T H K_E z', f'{e0:.2e}', e0 < 1e-10)
    check('T2b', 'g^T d = V and d = Q g with Q = L^T K_E L', f'{eq:.2e}', eq < 1e-10)
    check('T2c', 'Q is PSD (min eig / max eig)', f'{minq:.2e}', minq > -1e-12)
    # beta -> infinity: constraint W^1/2 C x = 0 (W singular), rate O(1/beta); C possibly rank deficient
    worst_bd, worst_rate, near_inf, near_euc, cres, n_rate = 0.0, 0.0, 0.0, 1e300, 0.0, 0
    for t in range(24):
        P = problem(scale=[1., 3.][t % 2], p=[60, 12][t % 2]); S = setup(P)   # p=12 < N k = 18: C rank-deficient
        KE = KE_of(P, rn(P['da'], 2)); pr = parts(P, S, KE, 0.0)
        A = S['F'] + S['mu'] * S['I']; b = P['J'].T @ P['H'] @ pr['uN']
        Ch = Wfac(P) @ S['C']; Ai = torch.linalg.inv(A)
        d_inf = Ai @ b - Ai @ Ch.T @ torch.linalg.pinv(Ch @ Ai @ Ch.T, hermitian=True, rtol=1e-10) @ Ch @ Ai @ b
        Ce = S['C']
        d_euc = Ai @ b - Ai @ Ce.T @ torch.linalg.pinv(Ce @ Ai @ Ce.T, hermitian=True, rtol=1e-10) @ Ce @ Ai @ b
        Ah = psd_sqrt(A); Aih = torch.linalg.inv(Ah)
        U_, sv, Vt = torch.linalg.svd(Ch @ Aih, full_matrices=False); keep = sv > 1e-8 * float(sv.max())
        smin = float(sv[keep].min()); r = Vt[keep].T @ (Vt[keep] @ (Aih @ b))
        prev = None; sc0 = float(parts(P, S, KE, 0.0)['dN'].norm())
        for beta in (1e2, 1e3, 1e4, 1e5, 1e6):
            db = parts(P, S, KE, beta)['dN']
            dist = math.sqrt(float((db - d_inf) @ A @ (db - d_inf)))
            worst_bd = max(worst_bd, dist / (float(r.norm()) / (1 + beta * smin**2)))
            if prev is not None and dist > 1e-14 * sc0 and beta / 10 * smin**2 > 30:
                worst_rate = max(worst_rate, abs(prev / dist - 10.0) / 10.0); n_rate += 1
            prev = dist
        sc = float(parts(P, S, KE, 0.0)['dN'].norm())
        cres = max(cres, float((Ch @ d_inf).norm()) / max(float((Ch @ parts(P, S, KE, 0.0)['dN']).norm()), 1e-300))
        if P['p'] == 60:                       # C full row rank, W singular: the two constraints differ
            db = parts(P, S, KE, 1e7)['dN']
            near_inf = max(near_inf, float((db - d_inf).norm()) / sc); near_euc = min(near_euc, float((db - d_euc).norm()) / sc)
    check('T2d', '||d_beta - d_inf||_A <= ||r||/(1+beta sigma_min^2)', f'max ratio {worst_bd:.3f}', worst_bd <= 1 + 1e-6)
    check('T2e', 'asymptotic O(1/beta) rate (successive ratio ~10 once beta sigma^2 >> 1)', f'max |ratio-10|/10 = {worst_rate:.2e} over {n_rate} pairs', worst_rate < 0.05 and n_rate > 0)
    schur, inrange = 0.0, 0.0
    for t in range(20):
        P = problem(scale=[1., 3.][t % 2], p=60); S = setup(P); KE = KE_of(P, rn(P['da'], 2)); J = P['J']
        Ch = Wfac(P) @ S['C']
        Uc, sc, Vc = torch.linalg.svd(Ch, full_matrices=False); kk = sc > 1e-7 * float(sc.max())
        Pn = torch.eye(P['p']) - Vc[kk].T @ Vc[kk]                      # projector onto null(C_hat), column side
        lam_, Q_ = torch.linalg.eigh(Ch @ Ch.T); kq = lam_ > 1e-14 * float(lam_.max())   # row side, independent route
        CCp = Q_[:, kq] @ torch.diag(1 / lam_[kq]) @ Q_[:, kq].T
        KNS = J @ Pn @ J.T; alt = J @ J.T - (J @ Ch.T) @ CCp @ (Ch @ J.T)
        schur = max(schur, rel(KNS, alt))
        A = S['F'] + S['mu'] * S['I']; Ai = torch.linalg.inv(A); b = J.T @ P['H'] @ parts(P, S, KE, 0.0)['uN']
        M_ = Ch @ Ai @ Ch.T; lm, Qm = torch.linalg.eigh(M_); km = lm > 1e-14 * float(lm.max())
        dinf = Ai @ b - Ai @ Ch.T @ (Qm[:, km] @ torch.diag(1 / lm[km]) @ Qm[:, km].T) @ Ch @ Ai @ b
        v = J @ dinf; Ur, w, _ = torch.linalg.svd(J @ Pn, full_matrices=False); Ur = Ur[:, w > 1e-9 * float(w.max())]
        inrange = max(inrange, rel(Ur @ (Ur.T @ v), v))
    check('T2h', 'hard-limit kernel K_NS = J P J^T = K - J C^T(C C^T)^+ C J^T with C = W^1/2 C (Schur complement)', f'{schur:.2e}', schur < 1e-8)
    check('T2i', 'J d_N^inf lies in range(K_NS) (nested projection, hard limit)', f'{inrange:.2e}', inrange < 1e-8)
    check('T2f', 'beta=1e7 solution matches KKT with W^1/2 C x = 0 (C full row rank), error / ||d_N^0||', f'{near_inf:.2e}', near_inf < 1e-4)
    check('T2g', 'pinv-KKT limit satisfies W^1/2 C d_inf = 0 also for rank-deficient C (p < N k)', f'{cres:.2e}', cres < 1e-6)
    check('T2f-NEG', 'Euclidean constraint C x = 0 (math.md Prop 2) is NOT the limit (min dist / ||d_N^0||)', f'{near_euc:.2e}', near_euc > 1e-2)

# ======================================================================= T3: repair, leak, certificate
def T3():
    e_rep, csr, wrong_viol, mono_viol, h_nonmono, bnd = 0.0, 0.0, 0, 0, 0, 0.0
    cert_viol, cert_n, vna_bad, n_inst, suf_ok, slack, kkt_err, kkt_neg = 0, 0, 0, 0, 0, 0.0, 0.0, 1e300
    inf_err = 0.0
    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None
    betas = [0.0] + [10**x for x in torch.linspace(-3, 7, 41).tolist()]
    for t in range(300):
        P = problem(scale=[1., 3., 6.][t % 3], drift=float(torch.rand(1, generator=gen) * 4 + .1), p=40 if t % 10 else 12)
        S = setup(P); KE = KE_of(P, rn(P['da'], 2)); pr0 = parts(P, S, KE, 0.0)
        V_O = float(pr0['zO'] @ KE @ pr0['zO'])
        e_rep = max(e_rep, abs(float(S['gO'] @ pr0['d_rep']) - V_O) / max(V_O, 1e-300))
        nW = math.sqrt(P['chi']); nWw = math.sqrt(float(S['eO'] @ P['W'] @ S['eO']))
        Ls, hs = [], []
        for b in betas:
            dN = parts(P, S, KE, b)['dN']; lk = leak(P, S, dN); Ls.append(lk)
            hs.append(V_O + float(S['eO'] @ S['C'] @ dN))
            lhs = abs(float(S['eO'] @ S['C'] @ dN))
            if lk > 0:
                csr = max(csr, lhs / (nW * math.sqrt(lk)))
                wrong_viol += int(lhs > nWw * math.sqrt(lk) * (1 + 1e-9))
            if b > 0: bnd = max(bnd, lk / (float(pr0['uN'] @ P['H'] @ pr0['uN']) / b))
        mono_viol += int(any(Ls[i + 1] > Ls[i] * (1 + 1e-9) + 1e-300 for i in range(len(Ls) - 1)))
        h_nonmono += int(any(hs[i + 1] < hs[i] - 1e-12 * abs(V_O) for i in range(len(hs) - 1)) and any(hs[i + 1] > hs[i] + 1e-12 * abs(V_O) for i in range(len(hs) - 1)))
        if V_O <= 0: continue
        n_inst += 1; eps_star = V_O**2 / P['chi']
        vna_bad += int(hs[0] < 0)
        for lk, h in zip(Ls, hs):
            if lk <= eps_star: cert_n += 1; cert_viol += int(h < -1e-12 * V_O)
        bc, _, (es, bsuf) = beta_cert(P, S, KE)
        suf_ok += int(bc <= bsuf * (1 + 1e-9))
        if bc > 0:
            hc = V_O + float(S['eO'] @ S['C'] @ parts(P, S, KE, bc)['dN']); slack = max(slack, hc / V_O)
        # hard limit: e_O^T C d_N^inf = 0 so h(inf) = V_O
        A = S['F'] + S['mu'] * S['I']; bb = P['J'].T @ P['H'] @ pr0['uN']; Ch = Wfac(P) @ S['C']; Ai = torch.linalg.inv(A)
        dinf = Ai @ bb - Ai @ Ch.T @ torch.linalg.pinv(Ch @ Ai @ Ch.T, hermitian=True, rtol=1e-10) @ Ch @ Ai @ bb
        inf_err = max(inf_err, abs(float(S['eO'] @ S['C'] @ dinf)) / V_O)
        # KKT: constrained minimiser of Phi_N s.t. leak <= eps* equals d_N^{beta_cert}
        if minimize is not None and P['p'] == 12 and bc > 0:
            uN = pr0['uN']
            fun = lambda x: PhiN(P, S, torch.tensor(x), uN)
            jac = lambda x: (2 * (P['J'].T @ P['H'] @ (P['J'] @ torch.tensor(x) - uN)) + 2 * S['mu'] * torch.tensor(x)).numpy()
            con = {'type': 'ineq', 'fun': lambda x: (es - leak(P, S, torch.tensor(x))) / es,
                   'jac': lambda x: (-2 * (S['B'] @ torch.tensor(x)) / es).numpy()}
            x0 = pr0['dN'].numpy() * 0.0
            res = minimize(fun, x0, jac=jac, constraints=[con], method='SLSQP', options=dict(ftol=1e-15, maxiter=2000))
            xs = torch.tensor(res.x); kkt_err = max(kkt_err, rel(xs, parts(P, S, KE, bc)['dN']))
            kkt_neg = min(kkt_neg, leak(P, S, pr0['dN']) / es)
    check('T3a', 'g_O^T d_rep = V_O (repair is always in the descent direction of KD)', f'{e_rep:.2e}', e_rep < 1e-9)
    check('T3b', '|e_O^T C d| <= ||e_O||_{W+} sqrt(leak)', f'max ratio {csr:.3f}', csr <= 1 + 1e-9)
    check('T3b-NEG', 'same bound with ||e_O||_W (wrong metric) is violated in some instances', f'{wrong_viol} violations', wrong_viol > 0)
    check('T3c', 'leak(beta) nonincreasing on grid', f'{mono_viol} instances violate', mono_viol == 0)
    check('T3d', 'leak(beta) <= ||u_N||_H^2 / beta', f'max ratio {bnd:.3f}', bnd <= 1 + 1e-9)
    check('T3d-WIT', 'h(beta) is non-monotone in some instances (so bisection on h is invalid)', f'{h_nonmono}/300', h_nonmono > 0)
    check('T3e', 'certificate: leak(beta) <= eps* = V_O^2/||e_O||^2_{W+}  =>  h(beta) >= 0', f'{cert_viol} violations / {cert_n} grid points', cert_viol == 0 and cert_n > 0)
    check('T3e-WIT', 'certificate is not vacuous: pure VNA (beta=0) raises KD to first order', f'{vna_bad}/{n_inst}', vna_bad > 0)
    check('T3f', 'beta_cert <= beta_suf', f'{suf_ok}/{n_inst}', suf_ok == n_inst)
    check('T3g', 'h(beta_cert) <= 2 V_O when beta_cert > 0', f'max {slack:.3f}', slack <= 2 + 1e-9)
    check('T3h', 'hard limit: e_O^T C d_N^inf = 0, i.e. h(inf) = V_O', f'{inf_err:.2e}', inf_err < 1e-7)
    if minimize is not None:
        check('T3i', 'beta_cert is the KKT multiplier: SLSQP constrained minimiser = d_N^{beta_cert}', f'{kkt_err:.2e}', kkt_err < 1e-4)
        check('T3i-WIT', 'unconstrained d_N^0 is infeasible for the budget (min leak0/eps*)', f'{kkt_neg:.2f}', kkt_neg > 1.0)
    # second-order KD change along theta - eps d on a real MLP: first-order term is -eps g_O^T d
    torch.manual_seed(4)
    from torch.func import jacrev, functional_call
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 5))
    X = torch.randn(7, 4); teach = torch.randn(7, 3); Tt = 2.0
    names = [k for k, _ in net.named_parameters()]; shapes = [v.shape for v in net.parameters()]
    th = torch.cat([v.detach().flatten() for v in net.parameters()])
    def unf(v):
        o, i = {}, 0
        for k_, s in zip(names, shapes):
            m = math.prod(s); o[k_] = v[i:i + m].view(s); i += m
        return o
    f = lambda v: functional_call(net, unf(v), (X,))
    kd = lambda v: -(torch.softmax(teach / Tt, 1) * torch.log_softmax(f(v)[:, :3] / Tt, 1)).sum() / 7
    thr = th.clone().requires_grad_(); gO = torch.autograd.grad(kd(thr), thr)[0]
    d = torch.randn(len(th)); errs = []
    for eps in (1e-3, 1e-4):
        ch = float(kd(th - eps * d) - kd(th)); errs.append(abs(ch + eps * float(gO @ d)) / abs(eps * float(gO @ d)))
    check('T3j', 'Delta L_KD(theta - eps d) = -eps g_O^T d + O(eps^2) (sign as in THEORY_FINAL; math.md Prop 3 had +)', f'rel err {errs[0]:.1e} -> {errs[1]:.1e}', errs[1] < errs[0] / 5 and errs[1] < 1e-2)

# ============================================================================ T4: plasticity cost of beta
def T4():
    viol, bnd = 0, 0.0
    for t in range(200):
        P = problem(scale=[1., 3., 6.][t % 3], drift=2.0, p=[12, 40][t % 2]); S = setup(P); KE = KE_of(P, rn(P['da'], 2))
        uN = parts(P, S, KE, 0.0)['uN']; d0 = parts(P, S, KE, 0.0)['dN']; l0 = leak(P, S, d0); p0 = PhiN(P, S, d0, uN)
        prev = p0
        for b in [10**x for x in torch.linspace(-3, 6, 31).tolist()]:
            db = parts(P, S, KE, b)['dN']; ph = PhiN(P, S, db, uN)
            viol += int(ph < prev - 1e-12 * abs(prev)); prev = ph
            bnd = max(bnd, (ph - p0) / max(b * (l0 - leak(P, S, db)), 1e-300) if b * l0 > 0 else 0.0)
    check('T4a', 'Phi_N(d_N^beta) = fit + mu||d||^2 is nondecreasing in beta', f'{viol} violations', viol == 0)
    check('T4b', 'Phi_N(d_beta) - Phi_N(d_0) <= beta (leak_0 - leak_beta)', f'max ratio {bnd:.3f}', bnd <= 1 + 1e-9)
    # referee counterexample: fit alone (without mu||d||^2) can DECREASE (math_rep 4d claim false)
    F = torch.tensor([[2.077, .563], [.563, .265]]); B = torch.tensor([[12.70, -.027], [-.027, 1e-4]]); b = torch.tensor([.278, .320]); mu = 1.0
    fit = lambda be: (lambda d: float(d @ F @ d - 2 * b @ d))(torch.linalg.solve(F + be * B + mu * torch.eye(2), b))
    phi = lambda be: (lambda d: float(d @ F @ d - 2 * b @ d + mu * d @ d))(torch.linalg.solve(F + be * B + mu * torch.eye(2), b))
    fs = [fit(x) for x in (0.0, 0.05, 0.2)]; ps = [phi(x) for x in (0.0, 0.05, 0.2)]
    check('T4c-WIT', 'fit alone decreases on the explicit counterexample (claim "fit monotone" rejected)', [round(x, 5) for x in fs], fs[0] > fs[1] > fs[2])
    check('T4d', 'on the same counterexample fit + mu||d||^2 is nondecreasing', [round(x, 5) for x in ps], ps[0] <= ps[1] <= ps[2])

# ================================================================================== T5: descent, no gamma
def T5():
    na, lb, skip, n = 0.0, 0.0, 0, 0
    for t in range(200):
        P = problem(scale=[1., 3., 6.][t % 3], drift=float(torch.rand(1, generator=gen) * 4 + .1), p=40); S = setup(P)
        KE = KE_of(P, rn(P['da'], 2)); g = S['g']; mu = S['mu']; gG = math.sqrt(float(g @ S['G'] @ g))
        l0 = leak(P, S, parts(P, S, KE, 0.0)['dN'])
        for beta in (0.0, 1.0, 30.0, 1e3):
            pr = parts(P, S, KE, beta)
            na = max(na, float(pr['d_rep'].norm()) * math.sqrt(mu) / max(Hnorm(P, pr['uR']), 1e-300),
                     float(pr['dN'].norm()) * math.sqrt(mu) / max(Hnorm(P, pr['uN']), 1e-300))
            z = pr['zN'] + pr['zO']; V = float(z @ KE @ z); gd = float(g @ (pr['d_rep'] + pr['dN']))
            low = V - gG * min(math.sqrt(beta * l0), 2 * Hnorm(P, pr['uN']))
            lb = max(lb, (low - gd) / max(abs(V), 1e-300))
        bc, V_O, _ = beta_cert(P, S, KE)
        if math.isfinite(bc):
            pr = parts(P, S, KE, bc); n += 1; skip += int(float(g @ (pr['d_rep'] + pr['dN'])) < 0)
    check('T5a', '||d_rep|| <= ||u_R||_H/sqrt(mu), ||d_N^beta|| <= ||u_N||_H/sqrt(mu) for all beta', f'max ratio {na:.3f}', na <= 1 + 1e-9)
    check('T5b', 'g^T d* >= V - ||g||_G min(sqrt(beta leak_0), 2||u_N||_H)', f'max violation {lb:.1e}', lb <= 1e-10)
    check('T5c-WIT', 'g^T d* < 0 occurs at beta_cert (descent is NOT unconditional -> skip rule needed)', f'{skip}/{n}', skip > 0)
    # second-order descent with the skip rule (g^T d >= 0, any rho = lam||d||/||g||) on quadratics with known L_s
    worst, negw = 1e300, 1e300
    for t in range(300):
        n_, p_ = 30, 20; X = rn(n_, p_); yv = rn(n_); th = rn(p_); Ls = float(torch.linalg.eigvalsh(X.T @ X).max())
        Lf = lambda th: 0.5 * float(((X @ th - yv)**2).sum()); g = X.T @ (X @ th - yv)
        d = rn(p_); d = d if float(g @ d) >= 0 else -d
        rho = [0.3, 1.0, 3.0][t % 3]; lam = rho * float(g.norm()) / float(d.norm())
        eps = 0.999 * 2 / (Ls * (1 + rho)**2)
        dec = Lf(th) - Lf(th - eps * (g + lam * d)); lbd = eps * float(g @ g) * (1 - 0.5 * Ls * eps * (1 + rho)**2)
        worst = min(worst, dec - lbd)
        dn = -g / float(g.norm()) * float(d.norm()) * 1.0       # g^T d < 0 (violates the skip rule)
        lam2 = 3.0 * float(g.norm()) / float(dn.norm()); eps2 = 0.999 * 2 / (Ls * 16)
        negw = min(negw, Lf(th) - Lf(th - eps2 * (g + lam2 * dn)))
    check('T5d', 'with g^T d >= 0: L decreases by >= eps||g||^2(1 - L_s eps (1+rho)^2/2) for eps < 2/(L_s(1+rho)^2)', f'min slack {worst:.3e}', worst >= -1e-9)
    check('T5d-NEG', 'without the skip rule (g^T d < 0, rho=3) the same step can increase L', f'min decrease {negw:.3e}', negw < 0)
    # gamma uniqueness (T5c): adversarial search over virtual output directions v (K_E = v v^T), with d computed
    # through the actual solve d_gamma = solve(F+mu, J^T H K_E (z_N + gamma z_O)) and g^T d evaluated on that vector.
    # The search minimises g^T d_gamma / ||g||^2 by gradient descent on v (it CAN find negative values if they exist).
    res = {}
    for gam in (1.0, 0.0, 2.0, 5.0):
        mn = 1e300
        for t in range(12):
            P = problem(scale=[1., 3.][t % 2], drift=2.0); S = setup(P); H, J = P['H'], P['J']
            zN, zO = S['z'](P['eN']), S['z'](P['eOf']); Aop = S['F'] + S['mu'] * S['I']; g = S['g']
            for rep in range(3):
                v = rn(P['n']).requires_grad_()
                opt = torch.optim.Adam([v], lr=0.05)
                for it in range(150):
                    vn = v / v.norm()
                    KEv = torch.outer(vn, vn)
                    d = torch.linalg.solve(Aop, J.T @ H @ KEv @ (zN + gam * zO))
                    val = (g @ d) / (g @ g)
                    opt.zero_grad(); val.backward(); opt.step()
                    mn = min(mn, float(val.detach()))
        res[gam] = mn
    check('T5e', 'gamma = 1: adversarial search over K_E = v v^T (solve-based d) never finds g^T d < 0', f'min {res[1.0]:.2e}', res[1.0] >= -1e-12)
    check('T5e-NEG', 'gamma in {0,2,5}: the same search finds g^T d < 0 (wrong residual weighting rejected)', {k: f'{v:.2e}' for k, v in res.items() if k != 1.0}, all(res[k] < -1e-6 for k in (0.0, 2.0, 5.0)))

# ============================================================================== T6: selection of A
def T6():
    ga, rb, bd, b0, neg = 0.0, -1e300, 0.0, 0.0, 1e300
    sub = []
    for t in range(60):
        P = problem(scale=[1., 3., 8.][t % 3], drift=[0.3, 1., 3.][t % 3]); S = setup(P); r = 3
        Sx = P['phi'].T @ P['phi']; Li = torch.linalg.inv(torch.linalg.cholesky(Sx))
        for beta in (0.0, 3.0, 50.0):
            zN, zO = S['z'](P['eN']), S['z'](P['eOf']); z = zN + zO
            Gb = torch.linalg.inv(S['F'] + beta * S['B'] + S['mu'] * S['I']); yb = P['H'] @ P['J'] @ Gb @ S['g']
            Nz, NzO, NzN, Ny = Nstat(P, z), Nstat(P, zO), Nstat(P, zN), Nstat(P, yb)
            MS = sym(Nz @ NzO.T) + sym(Ny @ NzN.T)
            A = rn(P['da'], r); KE = KE_of(P, A); pr = parts(P, S, KE, beta)
            gain = float(S['g'] @ (pr['d_rep'] + pr['dN']))
            ga = max(ga, abs(gain - float(torch.trace(A.T @ MS @ A))) / max(abs(gain), 1e-300))
            if beta > 0: neg = min(neg, abs(gain - float(torch.trace(A.T @ Nz @ Nz.T @ A))) / max(abs(gain), 1e-300))
            w, U = torch.linalg.eigh(Li @ MS @ Li.T); keep = [j for j in range(len(w) - 1, -1, -1) if w[j] > 0][:r]
            Aopt = Li.T @ U[:, keep]; gopt = float(torch.trace(Aopt.T @ MS @ Aopt)) if keep else 0.0
            for _ in range(100):
                Q, _r = torch.linalg.qr(rn(P['da'], r)); Ar = Li.T @ Q; rb = max(rb, float(torch.trace(Ar.T @ MS @ Ar)) - gopt)
            wq, Uq = torch.linalg.eigh(Li @ Nz @ Nz.T @ Li.T); At = Li.T @ Uq[:, -r:]; gt = float(torch.trace(At.T @ MS @ At))
            D = sym(Nstat(P, yb - z) @ NzN.T); ev = torch.linalg.eigvalsh(Li @ D @ Li.T).abs().sort(descending=True).values[:r]
            bnd = 2 * float(ev.sum())
            if bnd > 1e-12 * abs(gopt): bd = max(bd, (gopt - gt - 1e-10 * abs(gopt)) / bnd)
            if beta == 0: b0 = max(b0, abs(gopt - gt) / max(abs(gopt), 1e-300))
            else: sub.append((gopt - gt) / max(abs(gopt), 1e-300))
    check('T6a', 'g^T d* = tr(A^T M_S A), M_S = sym(N_z N_zO^T) + sym(N_y N_zN^T)', f'{ga:.2e}', ga < 1e-9)
    check('T6a-NEG', 'using the tiny_V matrix N_z N_z^T at beta>0 is rejected', f'min rel err {neg:.2e}', neg > 1e-4)
    check('T6b', 'top positive generalised eigvecs of (M_S, S_x) are optimal (random S_x-orthonormal A never better)', f'max excess {rb:.2e}', rb <= 1e-9)
    check('T6c', 'beta = 0: tiny_V(z) is exactly optimal', f'{b0:.2e}', b0 < 1e-9)
    check('T6d', 'suboptimality of tiny_V(z) <= 2 sum top-r |eig(S^-1/2 Delta S^-1/2)|', f'max ratio {bd:.3f}', bd <= 1 + 1e-9)
    check('T6d-WIT', 'tiny_V(z) is strictly suboptimal for some beta>0 instance', f'max rel subopt {max(sub):.3f}', max(sub) > 1e-3)

# =================================================================================== T7: swap / GIR
def qstar(Jt, H, r, mu):
    A = Jt.T @ H @ Jt + mu * torch.eye(Jt.shape[1]); b = Jt.T @ r; return -0.5 * float(b @ torch.linalg.solve(A, b))
def Efun(Jt, H, t, mu):
    A = Jt.T @ H @ Jt + mu * torch.eye(Jt.shape[1]); x = torch.linalg.solve(A, Jt.T @ H @ t)
    res = Jt @ x - t; return float(res @ H @ res) + mu * float(x @ x)
def T7():
    eE, eS, eS_neg, eperp, eg, b2, eEneg = 0.0, 0.0, 1e300, 0.0, 0.0, 0.0, 1e300
    fac, cos_obl, cos_I, nonopt, ntr = 1e300, [], [], 0, 0
    for t in range(60):
        P = problem(scale=[1., 3., 6.][t % 3], drift=2.0, p=[12, 25, 40][t % 3]); S = setup(P)
        H, J, mu = P['H'], P['J'], S['mu']; r = P['eOf'] if t % 2 else S['e']
        c_ = P['c']; Hpr = torch.zeros(P['n'])
        for i in range(P['N']):
            sl = slice(i * c_, (i + 1) * c_); Hpr[sl] = torch.linalg.solve(H[sl, sl] + torch.ones(c_, c_), r[sl])
        tr_ = -Hpr
        rp = r - S['z'](r); K = J @ J.T
        eperp = max(eperp, rel(rp, mu * torch.linalg.solve(H @ K + mu * torch.eye(P['n']), r)))
        Hperp = H - H @ J @ S['G'] @ J.T @ H; Hn = float(torch.linalg.eigvalsh(H).max())
        Sx = P['phi'].T @ P['phi']; Li = torch.linalg.inv(torch.linalg.cholesky(Sx))
        gains, crit, lmaxs = [], [], []
        for _ in range(20):
            a = Li.T @ rn(P['da']); a = a / math.sqrt(float(a @ Sx @ a)); Jv = Ja(P, a); Jt = torch.cat([J, Jv], 1)
            gain = qstar(J, H, r, mu) - qstar(Jt, H, r, mu)
            if t % 3 == 0:
                gE = 0.5 * (Efun(J, H, tr_, mu) - Efun(Jt, H, tr_, mu)); eE = max(eE, abs(gain - gE) / max(gain, 1e-300))
                gW = 0.5 * (Efun(J, H, -r, mu) - Efun(Jt, H, -r, mu)); eEneg = min(eEneg, abs(gain - gW) / max(gain, 1e-300))
            s = Jv.T @ rp; Sv = Jv.T @ Hperp @ Jv + mu * torch.eye(P['c'])
            pred = 0.5 * float(s @ torch.linalg.solve(Sv, s)); eS = max(eS, abs(gain - pred) / max(gain, 1e-300))
            sw = Jv.T @ r; eS_neg = min(eS_neg, abs(gain - 0.5 * float(sw @ torch.linalg.solve(Sv, sw))) / max(gain, 1e-300))
            xs = -S['G'] @ J.T @ r; eg = max(eg, rel(Jv.T @ (r + H @ J @ xs), s))
            b2 = max(b2, gain / (0.5 * float(s @ s) / mu), (0.5 * float(s @ s) / (float(torch.linalg.eigvalsh(Jv.T @ H @ Jv).max()) + mu)) / gain)
            lmax_v = float(torch.linalg.eigvalsh(Jv.T @ H @ Jv).max())
            gains.append(gain); crit.append(0.5 * float(s @ s) / (lmax_v + mu)); lmaxs.append(lmax_v)
        j = max(range(len(crit)), key=lambda i: crit[i])            # v_hat = argmax ell(v)  (THEORY T7(d))
        cH = max(lmaxs)                                             # c_H = max_v lambda_max(J_v^T H J_v) over the candidate set
        fac = min(fac, (gains[j] / max(gains)) / (mu / (cH + mu)))
        nonopt += int(gains[j] < max(gains) * (1 - 1e-12)); ntr += 1
        # oblique vs Euclidean complement
        Pj = J @ torch.linalg.pinv(J); re = r - Pj @ r
        if float(re.norm()) > 1e-12 * float(r.norm()):
            S2 = setup(P, mu_rel=1e-8); ro = r - S2['z'](r); cos_obl.append(abs(float(ro @ re)) / (float(ro.norm()) * float(re.norm())))
    check('T7a', 'shared functional: gain(v|J) = (E_J(t_r) - E_[J,Jv](t_r))/2, t_r = -H^+ r (logit scale 1)', f'{eE:.2e}', eE < 1e-7)
    check('T7a-NEG', 'same identity with target -r (no H^+) is rejected', f'min rel err {eEneg:.2e}', eEneg > 1e-3)
    check('T7b', 'Schur form gain = s^T S_v^-1 s / 2 with s = J_v^T r_perp, r_perp = r - z(r)', f'{eS:.2e}', eS < 1e-8)
    # T7b-NEG is run on its own, with a damping where r and r_perp differ materially (mu_rel = 0.3) and a
    # clear margin (the wrong formula must be off by > 5% in EVERY trial).
    negl = []
    for t in range(40):
        P = problem(scale=[1., 3., 6.][t % 3], drift=2.0, p=[12, 25][t % 2]); S = setup(P, mu_rel=0.3)
        H, J, mu = P['H'], P['J'], S['mu']; r = P['eOf'] if t % 2 else S['e']
        Hperp = H - H @ J @ S['G'] @ J.T @ H; Sx = P['phi'].T @ P['phi']; Li = torch.linalg.inv(torch.linalg.cholesky(Sx))
        for _ in range(10):
            a = Li.T @ rn(P['da']); a = a / math.sqrt(float(a @ Sx @ a)); Jv = Ja(P, a); Jt = torch.cat([J, Jv], 1)
            gain = qstar(J, H, r, mu) - qstar(Jt, H, r, mu); Sv = Jv.T @ Hperp @ Jv + mu * torch.eye(P['c'])
            sw = Jv.T @ r; negl.append(abs(gain - 0.5 * float(sw @ torch.linalg.solve(Sv, sw))) / max(gain, 1e-300))
    negl.sort()
    check('T7b-NEG', 's = J_v^T r (no complement) rejected: min err >= 1e3 x the positive tolerance, median > 1e-2', f'min {negl[0]:.2e}, median {negl[len(negl)//2]:.2e} (positive check tol 1e-8)', negl[0] > 1e-5 and negl[len(negl)//2] > 1e-2)
    check('T7c', 'r_perp = r - z(r) = mu (HK+mu)^-1 r', f'{eperp:.2e}', eperp < 1e-9)
    check('T7d', 'gradient wrt new outgoing weights at the relaxed live optimum = s', f'{eg:.2e}', eg < 1e-8)
    check('T7e', 'two-sided bound |s|^2/(2(lmax+mu)) <= gain <= |s|^2/(2 mu)', f'max ratio {b2:.4f}', b2 <= 1 + 1e-9)
    check('T7f', 'v_hat = argmax ell(v) = |s|^2/(2(lmax(J_v^T H J_v)+mu)) attains factor mu/(c_H+mu) of the best candidate', f'min normalised ratio {fac:.3f}', fac >= 1 - 1e-9)
    check('T7f-WIT', 'argmax ell is not always the exact argmax of gain (so the factor, not optimality, is the claim)', f'{nonopt}/{ntr} trials', nonopt > 0)
    cos_obl.sort()
    check('T7g', 'practical r_perp is the H-weighted (oblique) complement, not the Euclidean one (median |cos|, H != I, mu->0)', f'{cos_obl[len(cos_obl)//2]:.3f}', cos_obl[len(cos_obl) // 2] < 0.99)
    # OMP correspondence holds exactly when H = I and mu -> 0
    worst = 0.0
    for t in range(20):
        P = problem(H_identity=True, p=12); S = setup(P, mu_rel=1e-10); r = S['e']; J = P['J']
        ro = r - S['z'](r); re = r - J @ torch.linalg.pinv(J) @ r
        worst = max(worst, rel(ro, re))
    check('T7h', 'H = I, mu -> 0: r_perp equals the Euclidean residual (OMP atom selection)', f'{worst:.2e}', worst < 1e-6)
    # dead slot on a real ReLU MLP: exact function preservation; K' = K + J_v J_v^T (tangent space grows)
    from torch.func import jacrev, functional_call
    torch.manual_seed(9)
    net = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3)); X = torch.randn(10, 4)
    with torch.no_grad(): net[0].weight[2] = torch.randn(4) * 0.1; net[0].bias[2] = -100.0   # unit 2 dead on window
    names = [k for k, _ in net.named_parameters()]; shapes = [v.shape for v in net.parameters()]
    def flat(): return torch.cat([v.detach().flatten() for v in net.parameters()])
    def unf(v):
        o, i = {}, 0
        for k_, s in zip(names, shapes):
            m = math.prod(s); o[k_] = v[i:i + m].view(s); i += m
        return o
    fJ = lambda v: functional_call(net, unf(v), (X,)).flatten()
    th0 = flat(); f0 = fJ(th0); J0 = jacrev(fJ)(th0); K0 = J0 @ J0.T
    with torch.no_grad():
        net[2].weight[:, 2] = 0.0; net[0].weight[2] = torch.randn(4) * 2; net[0].bias[2] = 0.5   # install v, B_v = 0
    th1 = flat(); f1 = fJ(th1); J1 = jacrev(fJ)(th1)
    hv = torch.relu(X @ net[0].weight[2] + net[0].bias[2]); Jv = torch.kron(hv.unsqueeze(1), torch.eye(3))
    torch.manual_seed(3)
    W1, W2 = torch.randn(7, 4), torch.randn(5, 7); X2 = torch.randn(12, 4); yv = torch.randint(0, 5, (12,))
    Lf = lambda f: torch.nn.functional.cross_entropy(f.view(12, 5), yv, reduction='sum')
    errs, errs_neg = [], []
    for tt in (1e-1, 5e-2, 2.5e-2):
        W2s = W2.clone(); W2s[:, 2] *= tt; f0_ = (torch.tanh(X2 @ W1.T) @ W2s.T).flatten()
        fu = (torch.tanh(X2 @ W1.T)[:, 2:3] @ W2s[:, 2:3].T).flatten()
        e_ = torch.autograd.functional.jacobian(Lf, f0_); H_ = torch.autograd.functional.hessian(Lf, f0_)
        true = float(Lf(f0_ - fu) - Lf(f0_))
        errs.append(abs(true - float(-e_ @ fu + 0.5 * fu @ H_ @ fu))); errs_neg.append(abs(true - float(e_ @ fu - 0.5 * fu @ H_ @ fu)) / abs(true))
    check('T7k', 'jump cost of removing unit u = -e^T f_u + 1/2 f_u^T H f_u + O(|f_u|^3) (remainder ratio ~8 per halving)', f'{errs[0] / errs[1]:.2f}, {errs[1] / errs[2]:.2f}', 6 < errs[0] / errs[1] < 10 and 6 < errs[1] / errs[2] < 10)
    check('T7k-NEG', 'unified Cor 3c sign (+e^T f_u - 1/2 f^T H f) is rejected', f'min rel err {min(errs_neg):.2f}', min(errs_neg) > 0.5)
    Hm = torch.eye(J0.shape[0]); mu_ = 1e-2 * float(torch.trace(J0.T @ J0)) / J0.shape[0]
    d_dead = torch.linalg.solve(J0.T @ Hm @ J0 + mu_ * torch.eye(J0.shape[1]), J0.T @ Hm @ torch.randn(J0.shape[0]))
    u_idx = [2 * 4 + j for j in range(4)] + [24 + 2] + [30 + 6 * r_ + 2 for r_ in range(3)]   # W1[2,:], b1[2], W2[:,2]
    check('T7l', 'the VNA step computed before the swap does not touch the dead slot ((d)_u = 0)', f'{float(d_dead[u_idx].abs().max()):.1e}', float(d_dead[u_idx].abs().max()) < 1e-14)
    check('T7i', 'dead-slot insertion with B_v = 0 preserves the window function exactly', f'{float((f1 - f0).abs().max()):.1e}', float((f1 - f0).abs().max()) == 0.0)
    check('T7j', "after insertion K' = K + J_v J_v^T (the new unit adds a tangent direction; a_v columns are 0)", f'{rel(J1 @ J1.T, K0 + Jv @ Jv.T):.2e}', rel(J1 @ J1.T, K0 + Jv @ Jv.T) < 1e-12)
    with torch.no_grad():
        net[0].bias[2] = 0.5; net[2].weight[:, 2] = torch.randn(3)
    f_live = fJ(flat())
    with torch.no_grad(): net[2].weight[:, 1] = 0.0
    check('T7j-WIT', 'removing a LIVE unit changes the function (jump cost != 0)', f'{float((fJ(flat()) - f_live).abs().max()):.2e}', float((fJ(flat()) - f_live).abs().max()) > 1e-3)

# ======================================================================= T8: discrete Gronwall (trajectory)
def T8():
    from torch.func import jacrev, functional_call
    torch.manual_seed(5)
    net = torch.nn.Sequential(torch.nn.Linear(3, 6), torch.nn.Tanh(), torch.nn.Linear(6, 2))
    X = torch.randn(10, 3); Y = torch.randn(10, 2)
    names = [k for k, _ in net.named_parameters()]; shapes = [v.shape for v in net.parameters()]
    th = torch.cat([v.detach().flatten() for v in net.parameters()])
    def unf(v):
        o, i = {}, 0
        for k_, s in zip(names, shapes):
            m = math.prod(s); o[k_] = v[i:i + m].view(s); i += m
        return o
    f = lambda v: functional_call(net, unf(v), (X,)).flatten()
    Phi = torch.tanh(X @ torch.randn(3, 4)); JE = torch.kron(Phi, torch.eye(2)); KE = JE @ JE.T
    lam, eps, mu = 0.5, 0.02, 1e-3
    J0 = jacrev(f)(th); K0 = J0 @ J0.T; Lip = float(torch.linalg.matrix_norm(K0 + lam * KE, 2))
    U = lambda y: -(K0 + lam * KE) @ (y - Y.flatten())
    x = f(th); y = x.clone(); ratio, bound, nolip = 0.0, 0.0, 0.0; acc = 0.0; mean_d, mean_p = 0.0, 0.0
    for step in range(100):
        J = jacrev(f)(th); e = f(th) - Y.flatten()
        G = torch.linalg.inv(J.T @ J + mu * torch.eye(len(th))); z = J @ G @ J.T @ e; d = G @ J.T @ KE @ z
        thn = th - eps * (J.T @ e + lam * d); xn = f(thn); delta = float(((xn - x) / eps - U(x)).norm())
        dproj = lam * float((J @ d - KE @ z).norm()); mean_d += delta / 100; mean_p += dproj / 100
        y = y + eps * U(y); th, x = thn, xn
        bound = (1 + eps * Lip) * bound + eps * delta; acc = (1 + eps * Lip) * acc + eps * dproj
        ratio = max(ratio, float((x - y).norm()) / bound); nolip = max(nolip, float((x - y).norm()) / acc)
    check('T8a', 'discrete Gronwall: ||x_m - y_m|| <= sum (1+tau L_U)^(m-1-j) tau delta_j', f'max ratio {ratio:.4f}', ratio <= 1 + 1e-9)
    check('T8a-NEG', 'the same bound with only the projection defect lambda||Jd-u|| (not the realised defect) is violated', f'max ratio {nolip:.1f}; mean realised {mean_d:.3f} vs projection part {mean_p:.3f}', nolip > 1.0)

# ======================================================================================= T9: inexact CG
def cg(A, b, x0, tol, maxit, ref='b'):
    x = x0.clone(); r = b - A @ x; pv = r.clone(); rs = float(r @ r); nb = float(b.norm()); nr0 = math.sqrt(rs)
    for _ in range(maxit):
        if math.sqrt(rs) <= tol * (nb if ref == 'b' else nr0): break
        Ap = A @ pv; al = rs / float(pv @ Ap); x = x + al * pv; r = r - al * Ap
        rs2 = float(r @ r); pv = r + rs2 / rs * pv; rs = rs2
    return x, float((b - A @ x).norm()) / nb
def T9():
    a, cosv, neg = 0.0, -1e300, 0.0
    for t in range(100):
        P = problem(scale=[1., 3., 6.][t % 3], drift=2.0, p=60); S = setup(P); H, J, mu = P['H'], P['J'], S['mu']
        KE = KE_of(P, rn(P['da'], 2)); A0 = S['F'] + mu * S['I']
        for tol in (1e-1, 1e-2, 1e-3):
            b = S['gO']; xe = torch.linalg.solve(A0, b); xh, rho = cg(A0, b, torch.zeros_like(b), tol, 50)
            a = max(a, float((xh - xe).norm()) * mu / max(rho * float(b.norm()), 1e-300))
            b2 = J.T @ H @ KE @ (H @ J @ xh); dh, _ = cg(A0, b2, torch.zeros_like(b2), tol, 50)
            de = torch.linalg.solve(A0, J.T @ H @ KE @ (H @ J @ xe))
            c = float(dh @ de) / (float(dh.norm()) * float(de.norm()))
            lhs = float(S['gO'] @ dh) / float(dh.norm()); rhs = float(S['gO'] @ de) / float(de.norm()) - float(S['gO'].norm()) * math.sqrt(max(2 * (1 - c), 0))
            cosv = max(cosv, rhs - lhs)
            # stopping relative to the INITIAL residual with a far start: tol*||b||/mu bound breaks
            x0 = 1e3 * rn(len(b)); xw, _ = cg(A0, b, x0, tol, 3, ref='r0')
            neg = max(neg, float((xw - xe).norm()) / (tol * float(b.norm()) / mu))
    check('T9a', '||x_hat - x|| <= rho ||b|| / mu (rho = final residual relative to ||b||)', f'max ratio {a:.3f}', a <= 1 + 1e-9)
    check('T9b', 'cos(d_hat, d*) >= c  =>  normalised certificate degrades by <= ||g_O|| sqrt(2(1-c))', f'max violation {cosv:.2e}', cosv <= 1e-12)
    check('T9a-NEG', 'stopping rule relative to the initial residual (far warm start) breaks the tol||b||/mu bound', f'max ratio {neg:.1e}', neg > 1.0)

# ================================================================================ T10: scope (BN, momentum)
def T10():
    from torch.func import jacrev, functional_call
    torch.manual_seed(11)
    net = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.BatchNorm1d(6), torch.nn.ReLU(), torch.nn.Linear(6, 3))
    X = torch.randn(9, 4); yv = torch.randint(0, 3, (9,))
    with torch.no_grad():
        net.train(); [net(torch.randn(9, 4) * 3 + 2) for _ in range(5)]          # stale running stats
    names = [k for k, _ in net.named_parameters()]; shapes = [v.shape for v in net.parameters()]
    th = torch.cat([v.detach().flatten() for v in net.parameters()])
    def unf(v):
        o, i = {}, 0
        for k_, s in zip(names, shapes):
            m = math.prod(s); o[k_] = v[i:i + m].view(s); i += m
        return o
    buffers = {k: v.clone() for k, v in net.named_buffers()}
    def f_mode(v, train):
        net.train(train); out = functional_call(net, (unf(v), {k: b.clone() for k, b in buffers.items()}), (X,)).flatten(); net.train(True); return out
    ftr = lambda v: f_mode(v, True); fev = lambda v: f_mode(v, False)
    Lf = lambda fl: torch.nn.functional.cross_entropy(fl.view(9, 3), yv, reduction='sum')
    J = jacrev(ftr)(th); e = torch.autograd.functional.jacobian(Lf, ftr(th)); H = torch.autograd.functional.hessian(Lf, ftr(th))
    mu = 1e-2 * float(torch.trace(J.T @ H @ J)) / J.shape[0]; G = torch.linalg.inv(J.T @ H @ J + mu * torch.eye(len(th)))
    JE = torch.randn(J.shape[0], 3); KE = JE @ JE.T; z = H @ J @ G @ J.T @ e; d = G @ J.T @ H @ KE @ z; g = J.T @ e; V = float(z @ KE @ z)
    check('T10a', 'BN train mode: g^T d = V', f'{abs(float(g @ d) - V) / V:.2e}', abs(float(g @ d) - V) / V < 1e-10)
    fd = [abs((float(Lf(ftr(th - s * d)) - Lf(ftr(th))) + s * float(g @ d)) / (s * float(g @ d))) for s in (1e-3, 1e-4)]
    check('T10b', 'BN train mode: first-order FD of the train-mode loss matches -eps g^T d', f'{fd[0]:.1e} -> {fd[1]:.1e}', fd[1] < fd[0] / 5 and fd[1] < 1e-2)
    Jev = jacrev(fev)(th); eev = torch.autograd.functional.jacobian(Lf, fev(th)); Hev = torch.autograd.functional.hessian(Lf, fev(th))
    Gev = torch.linalg.inv(Jev.T @ Hev @ Jev + mu * torch.eye(len(th))); zev = Hev @ Jev @ Gev @ Jev.T @ eev; dev = Gev @ Jev.T @ Hev @ KE @ zev
    pred = -1e-4 * float((Jev.T @ eev) @ dev); true = float(Lf(ftr(th - 1e-4 * dev)) - Lf(ftr(th)))
    check('T10b-NEG', 'eval-mode (running-stats) quantities mispredict the train-mode loss change', f'rel err {abs(true - pred) / abs(pred):.2f}', abs(true - pred) / abs(pred) > 0.05)
    bn_idx = list(range(24 + 6, 24 + 6 + 12)); keep = [i for i in range(len(th)) if i not in bn_idx]
    Js = J[:, keep]; Gs = torch.linalg.inv(Js.T @ H @ Js + mu * torch.eye(len(keep))); zs = H @ Js @ Gs @ Js.T @ e; ds = Gs @ Js.T @ H @ KE @ zs
    check('T10c', 'BN parameters excluded: identity g_S^T d_S = V_S still exact', f'{abs(float((Js.T @ e) @ ds) - float(zs @ KE @ zs)) / float(zs @ KE @ zs):.2e}',
          abs(float((Js.T @ e) @ ds) - float(zs @ KE @ zs)) / float(zs @ KE @ zs) < 1e-10)
    # assumption A4 fails when BN follows the virtual pre-activation: feature is scale invariant in a
    Xb = torch.randn(9, 4); a = torch.randn(4)
    bnf = lambda a: torch.relu((Xb @ a - (Xb @ a).mean()) / ((Xb @ a).var(unbiased=False) + 1e-5).sqrt())
    check('T10d', 'BN after the virtual neuron: phi(2a) = phi(a) (A4 linearity fails -> stated as assumption)', f'|phi(2a)-phi(a)| = {float((bnf(2 * a) - bnf(a)).abs().max()):.1e}, |phi(2a)-2phi(a)| = {float((bnf(2 * a) - 2 * bnf(a)).abs().max()):.2f}',
          float((bnf(2 * a) - bnf(a)).abs().max()) < 1e-4 and float((bnf(2 * a) - 2 * bnf(a)).abs().max()) > 1e-1)
    # momentum: correction outside the buffer has relative first-order effect -eps lam g^T d, independent of m
    m = torch.randn(len(th)); rr = []
    for eps in (1e-3, 1e-4):
        base = th - eps * m; rr.append(float(Lf(ftr(base - eps * d)) - Lf(ftr(base))) / (-eps * float(g @ d)))
    check('T10e', 'outside-buffer correction: [L(th - eps m - eps d) - L(th - eps m)] / (-eps g^T d) -> 1', [round(x, 5) for x in rr], abs(rr[1] - 1) < abs(rr[0] - 1) + 1e-12 and abs(rr[1] - 1) < 1e-2)
    # T10(d) with the REAL torch.optim.SGD (momentum 0.9, no dampening, no weight decay, zero later gradients):
    #   outside the buffer: theta <- theta - lr*lam*d once, the optimiser never sees d  -> displacement ratio 1;
    #   inside the buffer (d fed as a gradient at step 0)                              -> ratio 1/(1-nu) = 10.
    def displacement(mode, steps=400, nu=0.9, lr=0.1):
        torch.manual_seed(7); lin = torch.nn.Linear(5, 1, bias=False); w0 = lin.weight.detach().clone()
        opt = torch.optim.SGD(lin.parameters(), lr=lr, momentum=nu); dvec = torch.randn_like(lin.weight)
        for s in range(steps):
            opt.zero_grad(); lin.weight.grad = torch.zeros_like(lin.weight)
            if s == 0 and mode == 'inside': lin.weight.grad = dvec.clone()
            opt.step()
            if s == 0 and mode == 'outside':
                with torch.no_grad(): lin.weight -= lr * dvec
        return float(((w0 - lin.weight.detach()) * dvec).sum() / (lr * (dvec * dvec).sum()))
    r_out, r_in = displacement('outside'), displacement('inside')
    check('T10f', 'real torch SGD(momentum=.9): outside-buffer correction displaces exactly lr*d (ratio 1)', f'{r_out:.6f}', abs(r_out - 1.0) < 1e-10)
    check('T10f-NEG', 'real torch SGD: the same d fed through the buffer is replayed ~1/(1-nu)=10 times (ratio 1 rejected)', f'{r_in:.4f}', abs(r_in - 10.0) < 1e-3 and abs(r_in - 1.0) > 1.0)

# ====================================================== T11: degenerate cases of the certificate (V_O=0, e_O=0, ...)
def beta_rule(P, S, KE, n_b=40, k_lo=8):
    """Algorithm step 3 exactly as in THEORY_FINAL_v2 (returns beta, tag)."""
    pr = parts(P, S, KE, 0.0); V_O = float(pr['zO'] @ KE @ pr['zO']); chi = P['chi']
    if chi <= 0.0 or float(S['eO'].norm()) == 0.0: return 0.0, 'eO=0'          # KD constant to first order
    if V_O <= 0.0:                                                               # T3(g)(iii'): h(0) = 0 exactly
        return 0.0, 'VO=0'
    eps = V_O**2 / chi
    if leak(P, S, pr['dN']) <= eps: return 0.0, 'leak0<=eps'
    bsuf = chi * float(pr['uN'] @ P['H'] @ pr['uN']) / V_O**2
    lo, hi = math.log(bsuf) - k_lo * math.log(10.0), math.log(bsuf)
    if leak(P, S, parts(P, S, KE, math.exp(lo))['dN']) <= eps: return math.exp(lo), 'lo'
    for _ in range(n_b):
        mid = 0.5 * (lo + hi)
        if leak(P, S, parts(P, S, KE, math.exp(mid))['dN']) <= eps: hi = mid
        else: lo = mid
    return math.exp(hi), 'bisect'

def T11():
    # (a) V_O = 0: virtual directions with J_E^T z_O = 0  =>  d_rep = 0, h(0) = z_O^T K_E z = 0 exactly; beta = 0 certified
    hmax, gmin, old_skip, n = 0.0, 1e300, 0, 0
    for t in range(60):
        P = problem(scale=[1., 3., 6.][t % 3], drift=2.0); S = setup(P)
        zO = S['z'](P['eOf']); NzO = Nstat(P, zO)                              # da x c, da = 6 > c = 5
        U_, s_, Vh = torch.linalg.svd(NzO.T); rk = int((s_ > 1e-12 * float(s_.max())).sum())
        nullb = Vh[rk:].T; A = nullb @ rn(nullb.shape[1], 1)
        KE = KE_of(P, A); pr = parts(P, S, KE, 0.0); d = pr['d_rep'] + pr['dN']
        V_O = float(zO @ KE @ zO); z = pr['zN'] + pr['zO']; V = float(z @ KE @ z)
        h0 = float(S['gO'] @ d); hmax = max(hmax, abs(h0) / max(float(S['gO'].norm()) * float(d.norm()), 1e-300))
        gmin = min(gmin, (float(S['g'] @ d) - V) / max(V, 1e-300))
        b, tag = beta_rule(P, S, KE); n += 1
        old_skip += int(leak(P, S, pr['dN']) > 0.0 and V_O <= 1e-14 * V)     # rule of v1 (eps* = 0) would have skipped
    check('T11a', 'V_O = 0: h(0) = g_O^T d*(0) = 0 (normalised), so beta = 0 is certified', f'{hmax:.2e}', hmax < 1e-10)
    check('T11b', 'V_O = 0: g^T d*(0) = V >= 0 (step passes the descent certificate)', f'min rel dev {gmin:.2e}', abs(gmin) < 1e-9)
    check('T11c-WIT', 'V_O = 0 with leak(0) > 0 occurs, so the v1 skip rule discarded certified steps', f'{old_skip}/{n}', old_skip > 0)
    # (b) e_O = 0 (teacher = student on old logits): chi^2 = 0, eps* = 0/0; g_O = 0 so h == 0 for every d
    worst = 0.0; tags = set()
    for t in range(20):
        P = problem(scale=[1., 3.][t % 2], drift=0.0); S = setup(P); KE = KE_of(P, rn(P['da'], 2))
        worst = max(worst, float(S['gO'].norm()) / float(S['g'].norm()), P['chi'])
        tags.add(beta_rule(P, S, KE)[1])
    check('T11d', 'e_O = 0: g_O = 0 and chi^2 = 0; the rule returns beta = 0 with tag eO=0 (no 0/0)', f'max {worst:.1e}, tags {sorted(tags)}', worst < 1e-14 and tags == {'eO=0'})
    # (c) c_old = 1: W == 0 and e_O == 0 identically
    P = problem(k=1); S = setup(P)
    check('T11e', 'c_old = 1: W = 0 and e_O = 0 (KD is constant), handled by the eO=0 branch', f'|W| {float(P["W"].abs().max()):.1e}, |e_O| {float(S["eO"].abs().max()):.1e}',
          float(P['W'].abs().max()) < 1e-15 and float(S['eO'].abs().max()) < 1e-15)
    # (d) p <= N (c_old - 1) and C_hat of full column rank: null(C_hat) = {0}, so d_N^inf = 0 (hard limit freezes new learning)
    ratios = []
    for t in range(10):
        P = problem(N=6, c=5, k=3, p=10); S = setup(P); KE = KE_of(P, rn(P['da'], 2))
        Ch = Wfac(P) @ S['C']; rk = int(torch.linalg.matrix_rank(Ch))
        d0 = parts(P, S, KE, 0.0)['dN']; d8 = parts(P, S, KE, 1e10)['dN']
        ratios.append((rk, float(d8.norm()) / float(d0.norm())))
    check('T11f', 'p <= N(c_old-1), rank C_hat = p: ||d_N^beta||/||d_N^0|| -> 0 (beta = 1e10)', f'ranks {sorted(set(r for r, _ in ratios))}, max ratio {max(x for _, x in ratios):.1e}',
          all(r == 10 for r, _ in ratios) and max(x for _, x in ratios) < 1e-5)
    # (e) log-scale bisection bracket [beta_suf 1e-8, beta_suf]: returned beta always certified (leak <= eps*), compare with h
    bad, cnt = 0, 0
    for t in range(120):
        P = problem(scale=[1., 3., 6.][t % 3], drift=float(torch.rand(1, generator=gen) * 4 + .1)); S = setup(P); KE = KE_of(P, rn(P['da'], 2))
        b, tag = beta_rule(P, S, KE, n_b=12)
        if tag in ('bisect', 'lo', 'leak0<=eps'):
            pr = parts(P, S, KE, b); V_O = float(pr['zO'] @ KE @ pr['zO']); cnt += 1
            bad += int(leak(P, S, pr['dN']) > V_O**2 / P['chi'] * (1 + 1e-12) or float(S['gO'] @ (pr['d_rep'] + pr['dN'])) < -1e-12 * V_O)
    check('T11g', 'log-bracket bisection (12 steps) always returns a certified beta: leak <= eps* and h >= 0', f'{bad} bad / {cnt}', bad == 0 and cnt > 50)

# ====================================================== T12: live-slot certification (exact window jump, analytic L3)
def _pycil_loss(Nn, c, k, T, y, tlog):
    def L(f):
        F = f.view(Nn, c); ce = torch.nn.functional.cross_entropy(F, y)
        kd = -(torch.softmax(tlog / T, 1) * torch.log_softmax(F[:, :k] / T, 1)).sum() / Nn
        return ce + kd
    def LKD(f):
        F = f.view(Nn, c); return -(torch.softmax(tlog / T, 1) * torch.log_softmax(F[:, :k] / T, 1)).sum() / Nn
    return L, LKD

L3C = 2 * math.sqrt(2) / (6 * math.sqrt(3))        # tight constant of L10 (v3): sup |E(X-m)^3| / ||h||^3 = 0.2722

def _third(Lf, f, h):
    """exact D^3 L(f)[h,h,h] by nested autograd"""
    f = f.clone().requires_grad_(True)
    g = torch.autograd.grad(Lf(f), f, create_graph=True)[0]
    g2 = torch.autograd.grad(g @ h, f, create_graph=True)[0]
    return float(torch.autograd.grad(g2 @ h, f)[0] @ h)

def _two_point(T, kind, c=5, k=3):
    """near-worst configuration of L10: two classes carry the mass q* = (3-sqrt3)/6 and 1-q*, h = (1,-1,0..)/sqrt2.
    kind='CE' -> on all logits (KD part removed by using c_old=1 not needed: we evaluate CE alone);
    kind='KD' -> on the old logits, f_old/T = (log q, log(1-q), -40)."""
    q = (3 - math.sqrt(3)) / 6
    f = torch.full((c,), -40.0); h = torch.zeros(c)
    if kind == 'CE':
        f[0], f[1] = math.log(q), math.log(1 - q)
    else:
        f[:k] = T * torch.tensor([math.log(q), math.log(1 - q)] + [-40.0] * (k - 2))
    h[0], h[1] = 1 / math.sqrt(2), -1 / math.sqrt(2)
    return f, h

def _l3_test(norm_fn, ratios_by_family, constructed_by_family):
    """two-sided test of a candidate L3 constant: every observed ratio <= 1 (validity) AND the near-worst
    construction reaches >= 0.999 (tightness).  norm_fn(family) returns the candidate bound / ||h||^3."""
    ok_up = all(r / norm_fn(fam) <= 1 + 1e-6 for fam, rs in ratios_by_family.items() for r in rs)
    ok_tight = all(constructed_by_family[fam] / norm_fn(fam) >= 0.999 for fam in constructed_by_family)
    return ok_up and ok_tight, ok_up, ok_tight

def T12():
    torch.manual_seed(21)
    # ---------------- (a) L10 with the tight constant 2sqrt2/(6sqrt3), validity AND tightness, per family
    ratios, constructed = {}, {}
    for T in (1.0, 2.0, 4.0):
        y1 = torch.tensor([4]); tl1 = torch.randn(1, 3)
        Lce = lambda f: torch.nn.functional.cross_entropy(f.view(1, 5), y1)
        Lkd = lambda f, T=T: -(torch.softmax(tl1 / T, 1) * torch.log_softmax(f.view(1, 5)[:, :3] / T, 1)).sum()
        for fam, Lf in (('CE', Lce), (f'KD,T={T}', Lkd)):
            if fam == 'CE' and T != 1.0: continue
            rs = []
            for t in range(150):                                                  # random (incl. saturated) probes
                f = torch.randn(5) * [0.3, 2., 6., 15.][t % 4]; h = torch.randn(5)
                if fam.startswith('KD'): h[3:] = 0.0
                rs.append(abs(_third(Lf, f, h)) / float(h.norm())**3)
            f, h = _two_point(T, 'CE' if fam == 'CE' else 'KD')
            con = abs(_third(Lf, f, h)) / float(h.norm())**3
            rs.append(con); ratios[fam] = rs; constructed[fam] = con
    nrm = lambda fam: L3C * (1.0 if fam == 'CE' else float(fam.split('=')[1])**-3)
    ok, up, ti = _l3_test(nrm, ratios, constructed)
    check('T12a', 'L10 tight: |D^3 l[h,h,h]| <= 2sqrt2/(6sqrt3) * (1 for CE, T^-3 for KD) * ||h||^3, and the two-point construction attains it (>= 0.999)',
          f"constructed/bound: {', '.join(f'{k_} {constructed[k_] / nrm(k_):.4f}' for k_ in constructed)}; max random/bound {max(r / nrm(k_) for k_, rs in ratios.items() for r in rs[:-1]):.4f}", ok)
    nrm_m1 = lambda fam: L3C * (1.0 if fam == 'CE' else float(fam.split('=')[1])**-1)          # mutant: T^-1 normaliser
    nrm_m2 = lambda fam: 0.1 * nrm(fam)                                                           # mutant: constant x 0.1
    ok1, _, ti1 = _l3_test(nrm_m1, ratios, constructed); ok2, up2, _ = _l3_test(nrm_m2, ratios, constructed)
    check('T12a-NEG', 'mutated L3 constants rejected by the same two-sided test: KD normaliser T^-1 (fails tightness), constant x0.1 (fails validity)',
          f'T^-1: tight={ti1}; x0.1: valid={up2}', (not ok1) and (not ok2))
    nrm_old = lambda fam: (0.708 / L3C) * nrm(fam)
    _, upo, tio = _l3_test(nrm_old, ratios, constructed)
    check('T12a-WIT', 'the v2 constant 0.708 is valid but 2.6x loose (passes validity, fails tightness)', f'valid={upo}, tight={tio}, ratio {L3C / 0.708:.3f}', upo and not tio)
    # combined loss, several samples: |D^3 L[h,h,h]| <= L3C (1+T^-3)/N ||h||^3
    Nn, c, k, T = 6, 5, 3, 2.0
    y = torch.randint(0, c, (Nn,)); tlog = torch.randn(Nn, k) * 2; L, LKD = _pycil_loss(Nn, c, k, T, y, tlog)
    worst = 0.0
    for t in range(150):
        f = torch.randn(Nn * c) * [1., 5., 20.][t % 3]; h = torch.zeros(Nn * c)
        if t % 2: h[:c] = torch.randn(c)
        else: h = torch.randn(Nn * c)
        worst = max(worst, abs(_third(L, f, h)) / (L3C * (1 + T**-3) / Nn * float(h.norm())**3))
    # two-sided: tightness of the combined bound on the aligned N=1, T=1 configuration. D^3 of the soft-target CE does
    # not depend on the teacher (the target enters linearly), so at T=1 the CE and KD third derivatives add up exactly.
    yA = torch.tensor([4]); tlA = torch.randn(1, 3)
    fA, hA = _two_point(1.0, 'KD')
    LA = lambda f_: torch.nn.functional.cross_entropy(f_.view(1, 5), yA) + (-(torch.softmax(tlA, 1) * torch.log_softmax(f_.view(1, 5)[:, :3], 1)).sum())
    d3A = abs(_third(LA, fA, hA)) / float(hA.norm())**3
    tightA = d3A / (L3C * 2.0)
    check('T12a2', 'combined CE + KD, N samples: |D^3L[h,h,h]| <= 0.2722 (1+T^-3)/N ||h||^3 (validity) and attained >= 0.95 on the aligned N=1, T=1 configuration (tightness)',
          f'max ratio {worst:.3f}, aligned {tightA:.4f}', worst <= 1 + 1e-6 and tightA >= 0.95)
    mutA = d3A / (L3C * max(1.0, 1.0**-3))                          # mutant: max(1, T^-3) instead of the sum (1 + T^-3)
    check('T12a2-NEG', 'mutant combined normaliser max(1, T^-3) (no additivity of CE and KD) is violated on the aligned configuration',
          f'ratio under mutant {mutA:.3f}', mutA > 1.5)
    # ---------------- (b) jump remainder with the tight L3; power: near-tight configuration at T = 1 (CE and KD aligned)
    T1 = 1.0; y1 = torch.tensor([4]); tl1 = torch.randn(1, 3)
    L1 = lambda f: torch.nn.functional.cross_entropy(f.view(1, 5), y1) + (-(torch.softmax(tl1 / T1, 1) * torch.log_softmax(f.view(1, 5)[:, :3] / T1, 1)).sum())
    L3_1 = L3C * (1 + T1**-3) / 1
    def rem_ratio(Lf, f, fu, L3):
        fr = f.clone().requires_grad_(True); g = torch.autograd.grad(Lf(fr), fr)[0]; Hh = torch.autograd.functional.hessian(Lf, f)
        R3 = float(Lf(f - fu) - Lf(f) - (-(g @ fu) + 0.5 * fu @ Hh @ fu)); return abs(R3) / (L3 / 6 * float(fu.norm())**3), R3
    f, h = _two_point(1.0, 'KD')                                    # at T=1, CE and KD softmaxes coincide on the two classes
    best_tight, worst_b = 0.0, 0.0
    for sgn in (1.0, -1.0):
        for dl in (3e-2, 1e-2):
            r_, _ = rem_ratio(L1, f, sgn * dl * h, L3_1); best_tight = max(best_tight, r_); worst_b = max(worst_b, r_)
    for t in range(150):
        f_ = torch.randn(5) * [1., 3., 8.][t % 3]; fu = torch.randn(5) * [0.05, 0.5, 2., 5.][t % 4]
        worst_b = max(worst_b, rem_ratio(L1, f_, fu, L3_1)[0])
    worst_T = {}
    for Tt in (1.0, 2.0, 4.0):
        L3_T = L3C * (1 + Tt**-3) / 1; wT = 0.0
        for t in range(150):
            ytt = torch.tensor([t % 5]); tlt = torch.randn(1, 3) * [1., 3.][t % 2]
            LT = lambda f_, ytt=ytt, tlt=tlt, Tt=Tt: torch.nn.functional.cross_entropy(f_.view(1, 5), ytt) + (-(torch.softmax(tlt / Tt, 1) * torch.log_softmax(f_.view(1, 5)[:, :3] / Tt, 1)).sum())
            f_ = torch.randn(5) * [1., 3., 8.][t % 3]; fu = torch.randn(5) * [0.05, 0.5, 2., 5.][t % 4]
            wT = max(wT, rem_ratio(LT, f_, fu, L3_T)[0])
        worst_T[Tt] = wT
    check('T12b', "jump remainder |j_u - (-e^T f_u + f_u^T H f_u/2)| <= L3/6 ||f_u||^3 with L3 = 0.2722(1+T^-3)/N, validity at T = 1, 2, 4; near-tight construction (T = 1) >= 0.95",
          f"max ratio T=1 {max(worst_b, worst_T[1.0]):.4f}, T=2 {worst_T[2.0]:.4f}, T=4 {worst_T[4.0]:.4f}; constructed {best_tight:.4f}",
          max(worst_b, *worst_T.values()) <= 1 + 1e-6 and best_tight >= 0.95)
    L3_drop = L3C / 1                                                # mutant: KD term of L3 dropped
    neg_r = max(rem_ratio(L1, f, sgn * 1e-2 * h, L3_drop)[0] for sgn in (1.0, -1.0))
    check('T12b-NEG', 'mutant L3 without the KD term (0.2722/N) is violated on the aligned configuration', f'max ratio {neg_r:.3f}', neg_r > 1.5)
    quad_only = 0.0
    for t in range(100):
        f_ = torch.randn(5) * 3; fu = torch.randn(5) * 5.0
        _, R3 = rem_ratio(L1, f_, fu, L3_1); quad_only = max(quad_only, abs(R3) / max(abs(float(L1(f_ - fu) - L1(f_))), 1e-300))
    check('T12b-NEG2', 'claim "jump is exactly quadratic" (R3 = 0) is rejected for large f_u (max rel err)', f'{quad_only:.2f}', quad_only > 0.05)
    # ---------------- (c) exact jump on a real ReLU MLP (structural: holds by construction of f_u as a forward difference)
    Nn, c, k, T = 8, 5, 3, 2.0
    y = torch.randint(0, c, (Nn,)); tlog = torch.randn(Nn, k) * 2; L, LKD = _pycil_loss(Nn, c, k, T, y, tlog)
    torch.manual_seed(21)
    net = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, c)); X = torch.randn(Nn, 4)
    with torch.no_grad():
        net[0].weight[2] = torch.randn(4) * 0.3; pre = (X @ net[0].weight[2]).sort().values
        net[0].bias[2] = -float(pre[-4]); net[2].weight[:, 2] *= 0.05                                  # 3 active samples: near-dead live unit
    f0 = net(X).flatten().detach(); hu = torch.relu(X @ net[0].weight[2] + net[0].bias[2]).detach()
    fu = (hu.unsqueeze(1) * net[2].weight[:, 2].detach().unsqueeze(0)).flatten(); nact = int((hu > 0).sum())
    ju, juK = float(L(f0 - fu) - L(f0)), float(LKD(f0 - fu) - LKD(f0))
    with torch.no_grad():
        net[2].weight[:, 2] = 0.0; net[0].weight[2] = torch.randn(4); net[0].bias[2] = 0.3
    f1 = net(X).flatten().detach()
    check('T12c', '[structural] live-slot swap: window loss change equals the forward-pass jump j_u exactly (also for KD)', f'{nact} active samples, |dL - j_u| = {abs(float(L(f1) - L(f0)) - ju):.1e}, |dKD - j_u^KD| = {abs(float(LKD(f1) - LKD(f0)) - juK):.1e}',
          nact > 0 and abs(float(L(f1) - L(f0)) - ju) < 1e-14 and abs(float(LKD(f1) - LKD(f0)) - juK) < 1e-14)
    # ---------------- (d) replaces the v2 tautology: on a REAL network with a dead unit, the network Jacobian after the swap
    #     equals [J (dead columns 0), J_v] with J_v = phi(a_v) (x) I, so the model change computed from the network's own
    #     Jacobians equals the Schur gain built from phi alone.  Can fail if insertion touched other columns or the
    #     Schur formula were wrong.
    from torch.func import jacrev, functional_call
    worst_d, worst_dneg = 0.0, 1e300
    for t in range(10):
        torch.manual_seed(100 + t)
        net = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, c)); X = torch.randn(Nn, 4)
        with torch.no_grad(): net[0].bias[2] = -100.0
        names = [k_ for k_, _ in net.named_parameters()]; shapes = [v.shape for v in net.parameters()]
        def unf(v):
            o, i = {}, 0
            for k_, s_ in zip(names, shapes):
                m = math.prod(s_); o[k_] = v[i:i + m].view(s_); i += m
            return o
        fJ = lambda v: functional_call(net, unf(v), (X,)).flatten()
        th = torch.cat([v.detach().flatten() for v in net.parameters()]); f_ = fJ(th); J0 = jacrev(fJ)(th)
        e_ = torch.autograd.functional.jacobian(L, f_); H_ = torch.autograd.functional.hessian(L, f_)
        mu = 1e-1 * float(torch.trace(J0.T @ H_ @ J0)) / J0.shape[0]
        with torch.no_grad(): net[2].weight[:, 2] = 0.0; net[0].weight[2] = torch.randn(4); net[0].bias[2] = 0.2
        th1 = torch.cat([v.detach().flatten() for v in net.parameters()]); J1 = jacrev(fJ)(th1)
        ph = torch.relu(X @ net[0].weight[2] + net[0].bias[2]).detach(); Jv = torch.kron(ph.unsqueeze(1), torch.eye(c))
        model = qstar(J1, H_, e_, mu) - qstar(J0, H_, e_, mu)                   # Net with j_u = 0, r' = r, from network Jacobians
        G0 = torch.linalg.inv(J0.T @ H_ @ J0 + mu * torch.eye(J0.shape[1])); Hp = H_ - H_ @ J0 @ G0 @ J0.T @ H_
        rp = e_ - H_ @ J0 @ G0 @ J0.T @ e_; s_ = Jv.T @ rp; Sv = Jv.T @ Hp @ Jv + mu * torch.eye(c)
        schur = 0.5 * float(s_ @ torch.linalg.solve(Sv, s_)); sw = Jv.T @ e_
        worst_d = max(worst_d, abs(model + schur) / schur)
        worst_dneg = min(worst_dneg, abs(model + 0.5 * float(sw @ torch.linalg.solve(Sv, sw))) / schur)
    check('T12d', 'real ReLU net, dead slot: model change from the network Jacobians after the swap = -Schur gain from phi(a_v) alone', f'{worst_d:.1e}', worst_d < 1e-8)
    check('T12d-NEG', 'same comparison with s = J_v^T e (no complement) is rejected', f'min rel err {worst_dneg:.2e}', worst_dneg > 1e-4)
    Xoff = torch.randn(50, 4) * 3
    with torch.no_grad():
        net2 = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, c)); X = torch.randn(Nn, 4)
        net2[0].bias[2] = -2.5; f_on = (X @ net2[0].weight[2] + net2[0].bias[2] > 0).sum(); f_off = (Xoff @ net2[0].weight[2] + net2[0].bias[2] > 0).sum()
    check('T12e-WIT', 'a unit dead on the window can be active off the window (preservation is a window statement)', f'active on window {int(f_on)}, off window {int(f_off)}/50', int(f_on) == 0 and int(f_off) > 0)

# ====================================================== T13: GIR step 6 (recheck, optimiser state, multi-slot, ReLU)
def T13():
    # (a) A5 recheck after the correction step
    torch.manual_seed(5); revived = 0; trials = 0; viol_rule = 0.0; viol_norule = 0.0
    for t in range(40):
        net = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3)); X = torch.randn(10, 4)
        with torch.no_grad():
            pre = X @ net[0].weight[2]; net[0].bias[2] = -float(pre.max()) - 1e-3                         # dead by a hair
            th = torch.cat([p.flatten() for p in net.parameters()]); step = torch.randn_like(th) * 0.05
            i = 0
            for p in net.parameters(): m = p.numel(); p -= step[i:i + m].view(p.shape); i += m             # correction step
        alive = int(((X @ net[0].weight[2] + net[0].bias[2]) > 0).sum()); trials += 1; revived += int(alive > 0)
        f_before = net(X).detach()
        def install(n_):
            with torch.no_grad(): n_[2].weight[:, 2] = 0.0; n_[0].weight[2] = torch.randn(4); n_[0].bias[2] = 0.2
        import copy
        n_rule = copy.deepcopy(net)
        if alive == 0: install(n_rule)                                                                    # rule: insert only if still dead
        viol_rule = max(viol_rule, float((n_rule(X) - f_before).abs().max()))
        n_no = copy.deepcopy(net); install(n_no); viol_norule = max(viol_norule, float((n_no(X) - f_before).abs().max()))
    check('T13a-WIT', 'a unit dead before step 5 can be alive after it', f'{revived}/{trials}', revived > 0)
    check('T13a', 'recheck A5 after step 5, insert only if still dead: window function unchanged', f'{viol_rule:.1e}', viol_rule == 0.0)
    check('T13a-NEG', 'inserting without the recheck changes the window function', f'{viol_norule:.2e}', viol_norule > 1e-6)
    # (b) optimiser state of the swapped slot must be reset (real torch SGD, momentum .9, weight decay)
    torch.manual_seed(1); X = torch.randn(16, 4); yv = torch.randint(0, 3, (16,))
    def run(reset):
        torch.manual_seed(2)
        net = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3))
        opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        for _ in range(5):
            opt.zero_grad(); torch.nn.functional.cross_entropy(net(X), yv).backward(); opt.step()
        with torch.no_grad():
            net[0].weight[2] = torch.randn(4); net[0].bias[2] = 0.3; net[2].weight[:, 2] = 0.0             # swap slot 2 (B_v = 0)
        if reset:
            for p, idx in ((net[0].weight, (2,)), (net[0].bias, (2,)), (net[2].weight, (slice(None), 2))):
                opt.state[p]['momentum_buffer'][idx] = 0.0
        a_before = net[0].weight[2].detach().clone(); B_before = net[2].weight[:, 2].detach().clone()
        opt.zero_grad(); torch.nn.functional.cross_entropy(net(X), yv).backward()
        gB = net[2].weight.grad[:, 2].clone(); ga_ = net[0].weight.grad[2].clone(); opt.step()
        dB = net[2].weight[:, 2].detach() - B_before; da_ = net[0].weight[2].detach() - a_before
        return float((dB + 0.1 * gB).abs().max()), float((da_ + 0.1 * (ga_ + 5e-4 * a_before)).abs().max()), float(ga_.abs().max())
    eb, ea, ga0 = run(True); eb2, ea2, _ = run(False)
    check('T13b', 'reset optimiser state: first update of B_v is exactly -lr grad, of a_v exactly -lr(grad + wd a_v) with grad_a = 0', f'|dB+lr g| {eb:.1e}, |da+lr(g+wd a)| {ea:.1e}, |grad_a| {ga0:.1e}', eb < 1e-15 and ea < 1e-15 and ga0 == 0.0)
    check('T13b-NEG', 'without the reset, stale momentum moves B_v / a_v beyond the gradient step', f'{eb2:.2e}, {ea2:.2e}', max(eb2, ea2) > 1e-6)
    # scope of (d): nesterov / dampening variants of torch SGD (zeroed buffer): first step = -c lr grad with c = 1+nu or 1-damp
    def first_dB(**kw):
        torch.manual_seed(2)
        net = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 3))
        opt = torch.optim.SGD(net.parameters(), lr=0.1, **kw)
        for _ in range(5):
            opt.zero_grad(); torch.nn.functional.cross_entropy(net(X), yv).backward(); opt.step()
        with torch.no_grad(): net[0].weight[2] = torch.randn(4); net[0].bias[2] = 0.3; net[2].weight[:, 2] = 0.0
        for p_, idx in ((net[0].weight, (2,)), (net[0].bias, (2,)), (net[2].weight, (slice(None), 2))):
            opt.state[p_]['momentum_buffer'][idx] = 0.0
        B0 = net[2].weight[:, 2].detach().clone(); opt.zero_grad(); torch.nn.functional.cross_entropy(net(X), yv).backward()
        gB = net[2].weight.grad[:, 2].clone(); opt.step(); return net[2].weight[:, 2].detach() - B0, gB
    dBn, gBn = first_dB(momentum=0.9, weight_decay=5e-4, nesterov=True); dBd, gBd = first_dB(momentum=0.9, weight_decay=5e-4, dampening=0.5)
    en = float((dBn + 0.1 * 1.9 * gBn).abs().max()); ed = float((dBd + 0.1 * 0.5 * gBd).abs().max())
    check('T13b2', 'scope of T13(d): nesterov -> first step -(1+nu) lr grad; dampening delta -> -(1-delta) lr grad (positive multiple, no stale term)', f'{en:.1e}, {ed:.1e}', en < 1e-15 and ed < 1e-15)
    check('T13b2-NEG', 'the plain-momentum formula -lr grad is wrong for nesterov (claimed scope matters)', f'{float((dBn + 0.1 * gBn).abs().max()):.2e}', float((dBn + 0.1 * gBn).abs().max()) > 1e-6)
    # (c) multi-slot: exact deflation  r_perp' = r_perp - H_perp J_v1 S_v1^-1 s_v1,  gain({v1,v2}) = gain(v1|J) + gain(v2|[J,J_v1])
    e_tel, e_upd, neg_stale, wins, ties, n, cheap_viol = 0.0, 0.0, 1e300, 0, 0, 0, 0
    for t in range(40):
        P = problem(scale=[1., 3.][t % 2], drift=2.0, p=20); S = setup(P, mu_rel=0.1); H, J, mu = P['H'], P['J'], S['mu']
        r = P['eOf'] if t % 2 else S['e']; c_ = P['c']
        G = S['G']; Hperp = H - H @ J @ G @ J.T @ H; rp = r - S['z'](r)
        cands = [Ja(P, rn(P['da'])) for _ in range(12)]
        def gain_of(Jb, Jv): return qstar(Jb, H, r, mu) - qstar(torch.cat([Jb, Jv], 1), H, r, mu)
        Jv1 = cands[0]; Jv2 = cands[1]
        s1 = Jv1.T @ rp; S1 = Jv1.T @ Hperp @ Jv1 + mu * torch.eye(c_)
        rp2 = rp - Hperp @ Jv1 @ torch.linalg.solve(S1, s1)
        J1 = torch.cat([J, Jv1], 1); G1 = torch.linalg.inv(J1.T @ H @ J1 + mu * torch.eye(J1.shape[1]))
        rp2_true = r - H @ J1 @ G1 @ J1.T @ r; e_upd = max(e_upd, rel(rp2, rp2_true))
        Hp_upd = Hperp - Hperp @ Jv1 @ torch.linalg.solve(S1, Jv1.T @ Hperp); Hp_true = H - H @ J1 @ G1 @ J1.T @ H
        e_upd = max(e_upd, rel(Hp_upd, Hp_true))
        tot = qstar(J, H, r, mu) - qstar(torch.cat([J, Jv1, Jv2], 1), H, r, mu)
        e_tel = max(e_tel, abs(tot - gain_of(J, Jv1) - gain_of(J1, Jv2)) / tot)
        Hperp1 = H - H @ J1 @ G1 @ J1.T @ H; S2 = Jv2.T @ Hperp1 @ Jv2 + mu * torch.eye(c_)
        s2_stale = Jv2.T @ rp; neg_stale = min(neg_stale, abs(gain_of(J1, Jv2) - 0.5 * float(s2_stale @ torch.linalg.solve(S2, s2_stale))) / gain_of(J1, Jv2))
        # greedy with deflation vs naive "top-2 individual gains"
        ind = [gain_of(J, Jc) for Jc in cands]; o = sorted(range(12), key=lambda i: -ind[i]); naive = gain_of(J, cands[o[0]]) + gain_of(torch.cat([J, cands[o[0]]], 1), cands[o[1]])
        J1g = torch.cat([J, cands[o[0]]], 1); second = max((gain_of(J1g, cands[i]), i) for i in range(12) if i != o[0])
        greedy = gain_of(J, cands[o[0]]) + second[0]; n += 1
        joint3 = qstar(J, H, r, mu) - qstar(torch.cat([J, cands[o[2]], cands[o[5]], cands[o[7]]], 1), H, r, mu)
        cheap_viol += int(joint3 < max(ind[o[2]], ind[o[5]], ind[o[7]]) * (1 - 1e-9))
        wins += int(greedy > naive * (1 + 1e-9)); ties += int(abs(greedy - naive) <= 1e-9 * naive)
    check('T13c', 'telescoping: gain({v1,v2}|J) = gain(v1|J) + gain(v2|[J,J_v1])', f'{e_tel:.1e}', e_tel < 1e-9)
    check('T13d', "rank-q deflation: r_perp' = r_perp - H_perp J_v1 S_v1^-1 s_v1 and H_perp' = H_perp - H_perp J_v1 S_v1^-1 J_v1^T H_perp equal the recomputed ones", f'{e_upd:.1e}', e_upd < 1e-9)
    check('T13d-NEG', 'using the stale r_perp for the 2nd slot misstates its gain (min rel err)', f'{neg_stale:.2e}', neg_stale > 1e-2)
    check('T13e', 'cheap multi-slot rule: joint gain of any set >= max single gain (adding columns never raises q*)', f'{cheap_viol} violations / {n}', cheap_viol == 0)
    check('T13e-WIT', 'deflated greedy beats naive top-2-by-individual-gain in some instances (never worse by construction)', f'{wins} wins, {ties} ties / {n}', wins > 0 and wins + ties == n)
    # (f) ReLU features: the Schur gain formula is exact for ANY a (J_v = phi(a) (x) I needs no A4);
    #     the A4 (linear) prediction is wrong; sign/bias selection by certified lower bound
    e_relu, neg_lin, lb_viol, better, cnt = 0.0, [], 0, 0, 0
    for t in range(40):
        P = problem(scale=[1., 3.][t % 2], drift=2.0, p=20); S = setup(P, mu_rel=0.1); H, J, mu = P['H'], P['J'], S['mu']
        r = P['eOf'] if t % 2 else S['e']; c_ = P['c']; Xf = P['phi']; G = S['G']
        Hperp = H - H @ J @ G @ J.T @ H; rp = r - S['z'](r)
        def Jrelu(a, b): ph = torch.relu(Xf @ a + b); return torch.kron(ph.unsqueeze(1), torch.eye(c_)), ph
        def exact_gain(Jv): return qstar(J, H, r, mu) - qstar(torch.cat([J, Jv], 1), H, r, mu)
        def lower(Jv): s = Jv.T @ rp; return 0.5 * float(s @ s) / (float(torch.linalg.eigvalsh(Jv.T @ H @ Jv).max()) + mu)
        Nr = Nstat(P, rp); Sx = Xf.T @ Xf / P['N'] + 1e-8 * torch.eye(P['da'])
        w, U = torch.linalg.eigh(torch.linalg.solve(torch.linalg.cholesky(Sx), torch.linalg.solve(torch.linalg.cholesky(Sx), Nr @ Nr.T).T))
        a0 = torch.linalg.solve(torch.linalg.cholesky(Sx).T, U[:, -1])
        for a_ in (a0, -a0, rn(P['da'])):
            Jv, ph = Jrelu(a_, 0.1)
            if float(ph.abs().sum()) == 0: continue
            s = Jv.T @ rp; Sv = Jv.T @ Hperp @ Jv + mu * torch.eye(c_)
            g_ex = exact_gain(Jv); e_relu = max(e_relu, abs(g_ex - 0.5 * float(s @ torch.linalg.solve(Sv, s))) / g_ex)
            glin = 0.5 * float(Nr.T @ a_ @ torch.linalg.solve(Sv, Nr.T @ a_))          # A4 (linear-feature) prediction
            neg_lin.append(abs(g_ex - glin) / g_ex)
            lb_viol += int(lower(Jv) > g_ex * (1 + 1e-9))
        # selection: candidates {+a0, -a0} x biases, choose by certified lower bound; compare with fixed (+a0, b=0)
        best = None
        for sgn in (1.0, -1.0):
            proj = Xf @ (sgn * a0)
            for b in (0.0, -float(proj.median())):
                Jv, ph = Jrelu(sgn * a0, b)
                if float(ph.sum()) == 0: continue
                lb = lower(Jv)
                if best is None or lb > best[0]: best = (lb, Jv)
        Jfix, phf = Jrelu(a0, 0.0)
        if best is not None and float(phf.sum()) > 0:
            cnt += 1; better += int(exact_gain(best[1]) >= exact_gain(Jfix) * (1 - 1e-9))
    check('T13f', 'ReLU features: Schur gain 1/2 s^T S_v^-1 s with s = J_v^T r_perp is exact for any a (no A4 needed)', f'{e_relu:.1e}', e_relu < 1e-9)
    neg_lin.sort()
    check('T13f-NEG', 'linear-feature (A4) gain prediction rejected for ReLU slots: min err >= 1e3 x tol, median > 5e-2', f'min {neg_lin[0]:.2e}, median {neg_lin[len(neg_lin)//2]:.2e}', neg_lin[0] > 1e-6 and neg_lin[len(neg_lin)//2] > 5e-2)
    check('T13g', 'certified lower bound 1/2|s|^2/(lmax(J_v^T H J_v)+mu) never exceeds the exact gain', f'{lb_viol} violations', lb_viol == 0)
    check('T13h-WIT', 'sign/bias chosen by the certified lower bound vs fixed (+a0, b=0): exact gain >= in most trials', f'{better}/{cnt}', cnt > 0 and better >= 0.5 * cnt)

# ====================================================== T14: BN (frozen mask) selection and ridged S_x
def T14():
    # (a) BN after the virtual pre-activation (no ReLU): phi(a) = Xc a / sigma(a), scale invariant.
    #     ||J_v^T s||^2 = N * a^T Xc^T S S^T Xc a / a^T Xc^T Xc a  exactly (generalised Rayleigh quotient) -> top eigvec optimal
    e_id, exc, neg = 0.0, -1e300, 1e300
    for t in range(30):
        Nn, da, c_ = 9, 5, 4; X = rn(Nn, da) + 1.5 * rn(1, da); Xc = X - X.mean(0); s = rn(Nn, c_)
        def gainBN(a, mask=None):
            pre = Xc @ a; ph = pre / pre.pow(2).mean().sqrt()
            if mask is not None: ph = ph * mask
            return float((ph.unsqueeze(1) * s).sum(0).pow(2).sum())
        Mn = Xc.T @ s @ s.T @ Xc; Sc = Xc.T @ Xc / Nn + 1e-12 * torch.eye(da)
        Lc = torch.linalg.cholesky(Sc); Li = torch.linalg.inv(Lc); w, U = torch.linalg.eigh(Li @ Mn @ Li.T); aopt = Li.T @ U[:, -1]
        for _ in range(50):
            a = rn(da); rq = float(a @ Mn @ a) / float(a @ (Xc.T @ Xc / Nn) @ a)
            e_id = max(e_id, abs(gainBN(a) - rq) / rq); exc = max(exc, gainBN(a) - gainBN(aopt))
            Mu = X.T @ s @ s.T @ X; rq_u = float(a @ Mu @ a) / float(a @ (X.T @ X / Nn) @ a)
            neg = min(neg, abs(gainBN(a) - rq_u) / rq)
        # ReLU after BN with the REALISED mask: identity holds with the masked statistics
        a = rn(da); pre = Xc @ a; mask = (pre > 0).double(); Xm = mask.unsqueeze(1) * Xc
        rqm = float(a @ (Xm.T @ s @ s.T @ Xm) @ a) / float(a @ (Xc.T @ Xc / Nn) @ a)
        e_id = max(e_id, abs(gainBN(a, mask) - rqm) / max(rqm, 1e-300))
    check('T14a', 'BN (train mode, fresh channel): gain = N * generalised Rayleigh quotient with CENTRED stats (also masked ReLU)', f'{e_id:.1e}', e_id < 1e-10)
    check('T14b', 'BN: the top generalised eigvec of (Xc^T S S^T Xc, Xc^T Xc) is optimal (no random a exceeds it)', f'max excess {exc:.1e}', exc <= 1e-9)
    check('T14b-NEG', 'uncentred statistics (A4 as written, ignoring BN mean subtraction) mispredict the gain', f'min rel err {neg:.2e}', neg > 1e-4)
    # (c) ridged S_x when d_a > N: positive generalised eigvecs lie in range(X^T); eigenvalues -> restricted problem as rho -> 0
    out_rng, conv = 0.0, []
    for t in range(20):
        Nn, da = 6, 15; X = rn(Nn, da); Q = rn(Nn, Nn); Q = Q @ Q.T; M = X.T @ Q @ X; S0 = X.T @ X / Nn
        Px = X.T @ torch.linalg.pinv(X.T)
        U_, s_, Vh = torch.linalg.svd(X, full_matrices=False); B_ = Vh.T                                    # basis of range(X^T)
        lam_r = float(torch.linalg.eigvals(torch.linalg.solve(B_.T @ S0 @ B_, B_.T @ M @ B_)).real.max())
        prev = None
        for rho in (1e-2, 1e-4, 1e-6, 1e-8):
            Sx = S0 + rho * float(torch.trace(S0)) / da * torch.eye(da); L_ = torch.linalg.cholesky(Sx); Li = torch.linalg.inv(L_)
            w, U = torch.linalg.eigh(Li @ M @ Li.T); a = Li.T @ U[:, -1]
            out_rng = max(out_rng, float(((torch.eye(da) - Px) @ a).norm()) / float(a.norm()))
            prev = abs(float(w[-1]) - lam_r) / lam_r
        conv.append(prev)
    check('T14c', 'ridged S_x (d_a > N): top eigvec in range(X^T) (exact in theory; tol = cond*eps); eigenvalue -> range-restricted value', f'out-of-range {out_rng:.1e}, rel gap at rho=1e-8: {max(conv):.1e}', out_rng < 1e-5 and max(conv) < 1e-6)
    try:
        torch.linalg.cholesky(X.T @ X / Nn); chol_ok = True
    except Exception:
        chol_ok = False
    check('T14c-WIT', 'without the ridge S_x is singular when d_a > N (Cholesky fails)', f'cholesky succeeded={chol_ok}', not chol_ok)

# ====================================================== T15: algorithm-level descent on a real network (CE + KD)
def T15():
    from torch.func import jacrev, functional_call
    torch.manual_seed(13); Nn, c, k, T = 10, 5, 3, 2.0
    applied, skipped, worst, kd_bad, n_kd = 0, 0, 0.0, 0, 0
    for t in range(25):
        net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, c))
        X = torch.randn(Nn, 4); y = torch.randint(0, c, (Nn,)); tlog = torch.randn(Nn, k) * 2
        L, LKD = _pycil_loss(Nn, c, k, T, y, tlog)
        names = [k_ for k_, _ in net.named_parameters()]; shapes = [v.shape for v in net.parameters()]
        th = torch.cat([v.detach().flatten() for v in net.parameters()])
        def unf(v):
            o, i = {}, 0
            for k_, s_ in zip(names, shapes):
                m = math.prod(s_); o[k_] = v[i:i + m].view(s_); i += m
            return o
        f = lambda v: functional_call(net, unf(v), (X,)).flatten()
        J = jacrev(f)(th); f0 = f(th)
        e = torch.autograd.functional.jacobian(L, f0); H = torch.autograd.functional.hessian(L, f0)
        eK = torch.autograd.functional.jacobian(LKD, f0); HK = torch.autograd.functional.hessian(LKD, f0)
        mu = 1e-2 * float(torch.trace(J.T @ H @ J)) / J.shape[0]; I = torch.eye(len(th)); G = torch.linalg.inv(J.T @ H @ J + mu * I)
        zf = lambda r: H @ J @ G @ J.T @ r; z, zO = zf(e), zf(eK); zN = z - zO
        JE = torch.randn(J.shape[0], 3); KE = JE @ JE.T
        uN, uR = KE @ zN, KE @ zO; d_rep = G @ J.T @ H @ uR; B = J.T @ HK @ J
        V_O = float(zO @ KE @ zO); eKp = torch.linalg.pinv(HK, hermitian=True, rtol=1e-10)
        chi = float(eK @ eKp @ eK); eps_ = V_O**2 / chi
        dN0 = G @ J.T @ H @ uN; beta = 0.0
        if float(dN0 @ B @ dN0) > eps_:
            lo, hi = 0.0, chi * float(uN @ H @ uN) / V_O**2
            for _ in range(50):
                mid = 0.5 * (lo + hi); dm = torch.linalg.solve(J.T @ H @ J + mid * B + mu * I, J.T @ H @ uN)
                if float(dm @ B @ dm) <= eps_: hi = mid
                else: lo = mid
            beta = hi
        dN = torch.linalg.solve(J.T @ H @ J + beta * B + mu * I, J.T @ H @ uN); dh = d_rep + dN
        g = J.T @ e; gO = J.T @ eK; gh, hh = float(g @ dh), float(gO @ dh)
        if gh < 0 or hh < 0: skipped += 1; continue
        applied += 1; lam = 1.0
        p = g + lam * dh
        for eps in (1e-3, 1e-4):
            dL = float(L(f(th - eps * p)) - L(f0)); pred = -eps * float(g @ p)
            if eps == 1e-4: worst = max(worst, abs(dL - pred) / abs(pred))
            if eps == 1e-4 and dL >= 0: worst = max(worst, 1e9)
        dK = float(LKD(f(th - 1e-4 * lam * dh)) - LKD(f0)); n_kd += 1; kd_bad += int(dK > 1e-4 * 1e-6 * abs(hh) and hh > 0)
    check('T15a', 'real MLP, CE+KD, certified beta + skip rule: applied steps decrease L, first-order prediction within 1%', f'{applied} applied, {skipped} skipped, max rel err {worst:.1e}', applied > 0 and worst < 1e-2)
    check('T15b', 'real MLP: the correction alone (theta - eps lam d_hat) does not increase L_KD when h_hat >= 0 (eps=1e-4)', f'{kd_bad} increases / {n_kd}', kd_bad == 0)


# ====================================================== T16: live-slot rule v3 (prediction under L-recovery; H at f - f_u)
def T16():
    from torch.func import jacrev, functional_call
    import copy
    # (a) PROVEN: the post-swap model must be expanded at f - f_u.  Model at the current output has third-order error
    #     <= L3/6 |D|^3; expanding H at f (pre-swap point) adds a second-order term <= 1/2 L3 |f_u| |D|^2.
    Nn, c, k, T = 4, 5, 3, 2.0
    y = torch.randint(0, c, (Nn,), generator=gen); tlog = rn(Nn, k) * 2; L, LKD = _pycil_loss(Nn, c, k, T, y, tlog)
    L3 = L3C * (1 + T**-3) / Nn; wc, wd, wneg = 0.0, 0.0, 0.0
    for t in range(120):
        f = rn(Nn * c) * 3; fu = rn(Nn * c) * [1., 3.][t % 2]; D = rn(Nn * c) * [0.02, 0.1][t % 2]
        g1 = f - fu; fr = g1.clone().requires_grad_(True); e1 = torch.autograd.grad(L(fr), fr)[0]
        H1 = torch.autograd.functional.hessian(L, g1); H0 = torch.autograd.functional.hessian(L, f)
        ex = float(L(g1 + D) - L(g1)); cur = float(e1 @ D + 0.5 * D @ H1 @ D); wrg = float(e1 @ D + 0.5 * D @ H0 @ D)
        nD, nf = float(D.norm()), float(fu.norm())
        wc = max(wc, abs(ex - cur) / (L3 / 6 * nD**3))
        wd = max(wd, abs(ex - wrg) / (L3 / 6 * nD**3 + 0.5 * L3 * nf * nD**2))
        wneg = max(wneg, abs(ex - wrg) / (L3 / 6 * nD**3))
    check('T16a', 'post-swap model expanded at f - f_u: error <= L3/6 |D|^3; expanded with H(f): error <= L3/6 |D|^3 + L3 |f_u| |D|^2 / 2', f'max ratios {wc:.3f}, {wd:.3f}', wc <= 1 + 1e-6 and wd <= 1 + 1e-6)
    check('T16a-NEG', 'H taken at the pre-swap output f violates the third-order bound (it carries an O(|f_u||D|^2) term)', f'max ratio {wneg:.1f}', wneg > 2.0)
    # (b-e) real ReLU MLP, all parameters, referee-3 protocol (live slot u replaced by random v, B_v = 0)
    def lossf(N, c, k, T, y, tl, which):
        def Lx(f):
            F = f.view(N, c); kd = -(torch.softmax(tl / T, 1) * torch.log_softmax(F[:, :k] / T, 1)).sum() / N
            return kd if which == 'KD' else torch.nn.functional.cross_entropy(F, y) + kd
        return Lx
    def mk(net, X):
        names = [n_ for n_, _ in net.named_parameters()]; shapes = [p_.shape for p_ in net.parameters()]
        def fn(v):
            o, i = {}, 0
            for n_, s_ in zip(names, shapes):
                m = math.prod(s_); o[n_] = v[i:i + m].view(s_); i += m
            return functional_call(net, o, (X,)).flatten()
        th = torch.cat([p_.detach().flatten() for p_ in net.parameters()]); return fn, th
    def gh(Lf, f):
        fr = f.clone().requires_grad_(True); g = torch.autograd.grad(Lf(fr), fr)[0]; return g, torch.autograd.functional.hessian(Lf, f)
    def model(J, e, H, eK, HK, mu):          # recovery step on L (A10) and the loss / KD models along it
        x = -torch.linalg.solve(J.T @ H @ J + mu * torch.eye(J.shape[1]), J.T @ e); dv = J @ x
        return float(e @ dv + 0.5 * dv @ H @ dv), float(eK @ dv + 0.5 * dv @ HK @ dv), x
    def gd(fn, th, Lf, LKf, n=200, lr=0.05):
        th = th.clone().requires_grad_(True)
        for _ in range(n):
            g = torch.autograd.grad(Lf(fn(th)), th)[0]; th = (th - lr * g).detach().requires_grad_(True)
        f = fn(th).detach(); return float(Lf(f)), float(LKf(f))
    torch.manual_seed(3); N, din, hid = 12, 5, 8
    rows = {1e-2: [], 10.0: []}; imm_bad, imm_n, oldmis = 0, 0, 0
    for trial in range(10):
        X = torch.randn(N, din); y = torch.randint(0, c, (N,)); tl = torch.randn(N, k) * 2
        Lf = lossf(N, c, k, T, y, tl, 'L'); LKf = lossf(N, c, k, T, y, tl, 'KD')
        net = torch.nn.Sequential(torch.nn.Linear(din, hid), torch.nn.ReLU(), torch.nn.Linear(hid, c))
        fn0, th0 = mk(net, X); f0 = fn0(th0).detach(); J0 = jacrev(fn0)(th0).detach()
        e0, H0 = gh(Lf, f0); eK0, HK0 = gh(LKf, f0); gL0, gK0 = gd(fn0, th0, Lf, LKf)
        trF = float(torch.trace(J0.T @ H0 @ J0)) / J0.shape[0]
        for u in range(hid):
            net2 = copy.deepcopy(net)
            with torch.no_grad(): net2[2].weight[:, u] = 0.0; net2[0].weight[u] = torch.randn(din); net2[0].bias[u] = 0.1
            fn1, th1 = mk(net2, X); f1 = fn1(th1).detach()
            if float((f0 - f1).abs().max()) == 0: continue
            J1 = jacrev(fn1)(th1).detach(); e1, H1 = gh(Lf, f1); eK1, HK1 = gh(LKf, f1)
            ju, juK = float(Lf(f1) - Lf(f0)), float(LKf(f1) - LKf(f0)); gL1, gK1 = gd(fn1, th1, Lf, LKf)
            if ju <= 0 and juK <= 0:                       # immediate exact certificate (window, at insertion)
                imm_n += 1; imm_bad += int(float(Lf(f1)) > float(Lf(f0)) or float(LKf(f1)) > float(LKf(f0)))
            for mr in rows:
                mu = mr * trF
                mL0, mK0, x0 = model(J0, e0, H0, eK0, HK0, mu); mL1, mK1, x1 = model(J1, e1, H1, eK1, HK1, mu)
                rL = float(Lf(fn1(th1 + x1))) - float(Lf(fn0(th0 + x0))); rK = float(LKf(fn1(th1 + x1))) - float(LKf(fn0(th0 + x0)))
                pL, pK = ju + mL1 - mL0, juK + mK1 - mK0
                if mr == 1e-2:                              # v2 rule's KD quantity (KD-only recovery) vs v3 prediction
                    def q(J, H, r): b = J.T @ r; return -0.5 * float(b @ torch.linalg.solve(J.T @ H @ J + mu * torch.eye(J.shape[1]), b))
                    oldmis += int((juK + q(J1, H1, eK1) - q(J0, H0, eK0) < 0) != (pK < 0))
                rows[mr].append(dict(pL=pL, pK=pK, rL=rL, rK=rK, gL=gL1 - gL0, gK=gK1 - gK0))
    def stat(rs, a, b):
        n = len(rs); ag = sum((r_[a] < 0) == (r_[b] < 0) for r_ in rs) / n
        xa = torch.tensor([r_[a] for r_ in rs]); xb = torch.tensor([r_[b] for r_ in rs]); return ag, float(torch.corrcoef(torch.stack([xa, xb]))[0, 1])
    sL, sK = stat(rows[10.0], 'pL', 'rL'), stat(rows[10.0], 'pK', 'rK')
    check('T16b', '[empirical] near-linear regime (mu_rel = 10): predicted dL, dKD (L-recovery, H at f - f_u) match the realised GN step in sign >= 90%, corr >= 0.9',
          f'n={len(rows[10.0])}, L: agree {sL[0]:.2f} corr {sL[1]:.2f}; KD: agree {sK[0]:.2f} corr {sK[1]:.2f}', min(sL[0], sK[0]) >= 0.9 and min(sL[1], sK[1]) >= 0.9)
    pL_, pK_ = stat(rows[1e-2], 'pL', 'rL'), stat(rows[1e-2], 'pK', 'rK')
    check('T16c-WIT', '[empirical] production damping (mu_rel = 1e-2): sign agreement near chance -> live-slot prediction unreliable', f'L: agree {pL_[0]:.2f} corr {pL_[1]:.2f}; KD: agree {pK_[0]:.2f} corr {pK_[1]:.2f}', max(pL_[0], pK_[0]) < 0.7)
    gL_, gK_ = stat(rows[10.0], 'pL', 'gL'), stat(rows[10.0], 'pK', 'gK')
    check('T16d-WIT', '[empirical] against real recovery (200 GD steps) the prediction is weak even at mu_rel = 10 (A10 is interpretive)', f'L: agree {gL_[0]:.2f} corr {gL_[1]:.2f}; KD: agree {gK_[0]:.2f} corr {gK_[1]:.2f}', min(gL_[0], gK_[0]) < 0.8)
    check('T16e', '[structural] immediate exact certificate: swaps with j_u <= 0 and j_u^KD <= 0 never raise window L or KD at insertion', f'{imm_bad} violations / {imm_n}', imm_bad == 0 and imm_n > 0)
    check('T16e-WIT', 'the v2 KD quantity (KD-only recovery) disagrees in sign with the v3 prediction (L-recovery) in many pairs', f'{oldmis}/{len(rows[1e-2])}', oldmis > 0)

# ====================================================== T17: BN slot insertion (reset gamma/beta, bias rule on beta_BN)
def T17():
    from torch.func import jacrev, functional_call
    import copy
    torch.manual_seed(5); Nn, c = 16, 3; X = torch.randn(Nn, 4); yv = torch.randint(0, c, (Nn,))
    Lf = lambda f: torch.nn.functional.cross_entropy(f.view(Nn, c), yv)
    net = torch.nn.Sequential(torch.nn.Linear(4, 6, bias=False), torch.nn.BatchNorm1d(6), torch.nn.ReLU(), torch.nn.Linear(6, c)); net.train()
    with torch.no_grad(): net[1].bias[2] = -50.0                       # channel 2 dead on the window because of beta_BN
    f0 = net(X).flatten().detach()
    dead = int((net[1](net[0](X))[:, 2] > 0).sum())
    a_v = torch.randn(4)
    def insert(n_, gamma, beta):
        with torch.no_grad():
            n_[3].weight[:, 2] = 0.0; n_[0].weight[2] = a_v
            if gamma is not None: n_[1].weight[2] = gamma; n_[1].bias[2] = beta; n_[1].running_mean[2] = 0.0; n_[1].running_var[2] = 1.0
    xhat = (X @ a_v - (X @ a_v).mean()) / ((X @ a_v).var(unbiased=False) + net[1].eps).sqrt()
    worst_f, phis, half = 0.0, {}, None
    for sgn in (1.0, -1.0):
        for b in (0.0, -float((sgn * xhat).median())):
            n_ = copy.deepcopy(net); insert(n_, sgn, b)                 # gamma = sigma (sign), beta_BN = bias rule
            worst_f = max(worst_f, float((n_(X).flatten().detach() - f0).abs().max()))
            ph = torch.relu(n_[1](n_[0](X))[:, 2]).detach(); phis[(sgn, b)] = ph
            if b != 0.0 and sgn == 1.0: half = int((ph > 0).sum())
    n_keep = copy.deepcopy(net); insert(n_keep, None, None)
    ph_keep = torch.relu(n_keep[1](n_keep[0](X))[:, 2]).detach()
    check('T17a', 'BN slot: insertion with B_v = 0 preserves the window function exactly for every (gamma, beta_BN) candidate', f'dead before: {dead == 0}, max |df| {worst_f:.1e}', dead == 0 and worst_f == 0.0)
    check('T17a-WIT', 'without resetting the slot BN (gamma, beta_BN) the inserted unit stays dead (phi = 0): the candidate is dropped', f'sum phi keep {float(ph_keep.sum()):.1e}, reset {float(phis[(1.0, 0.0)].sum()):.2f}', float(ph_keep.abs().sum()) == 0 and float(phis[(1.0, 0.0)].sum()) > 0)
    check('T17b', 'bias rule acts on beta_BN: beta = -median(sigma x_hat) makes about half the window active', f'{half}/{Nn} active', half is not None and abs(half - Nn / 2) <= 1)
    lin_b = torch.nn.Linear(4, 1); bn = torch.nn.BatchNorm1d(1); bn.train()
    with torch.no_grad(): lin_b.bias[:] = 0.0
    o1 = bn(lin_b(X)).detach()
    with torch.no_grad(): lin_b.bias[:] = -float((X @ lin_b.weight[0]).median())
    o2 = bn(lin_b(X)).detach()
    check('T17b-WIT', 'a pre-BN bias is cancelled by train-mode BN (so a bias rule on the linear layer does nothing)', f'{float((o1 - o2).abs().max()):.1e}', float((o1 - o2).abs().max()) < 1e-12)
    # Schur gain exact with BN features: model change from the network Jacobians = Schur gain from the realised phi
    names = [k_ for k_, _ in net.named_parameters()]; shapes = [v.shape for v in net.parameters()]
    def fJ_of(n_):
        n_ = copy.deepcopy(n_); n_[1].track_running_stats = False                                     # batch-stat BN, no buffer mutation
        n_[1].running_mean = None; n_[1].running_var = None; n_[1].num_batches_tracked = None
        def fJ(v):
            o, i = {}, 0
            for k_, s_ in zip(names, shapes):
                m = math.prod(s_); o[k_] = v[i:i + m].view(s_); i += m
            return functional_call(n_, o, (X,)).flatten()
        return fJ, torch.cat([v.detach().flatten() for v in n_.parameters()])
    fJ0, th0 = fJ_of(net); J0 = jacrev(fJ0)(th0).detach(); e_ = torch.autograd.functional.jacobian(Lf, f0); H_ = torch.autograd.functional.hessian(Lf, f0)
    mu = 1e-1 * float(torch.trace(J0.T @ H_ @ J0)) / J0.shape[0]
    n1 = copy.deepcopy(net); insert(n1, 1.0, -float(xhat.median())); fJ1, th1 = fJ_of(n1); J1 = jacrev(fJ1)(th1).detach()
    ph = torch.relu(n1[1](n1[0](X))[:, 2]).detach(); Jv = torch.kron(ph.unsqueeze(1), torch.eye(c))
    model = qstar(J1, H_, e_, mu) - qstar(J0, H_, e_, mu)
    G0 = torch.linalg.inv(J0.T @ H_ @ J0 + mu * torch.eye(J0.shape[1])); Hp = H_ - H_ @ J0 @ G0 @ J0.T @ H_
    rp = e_ - H_ @ J0 @ G0 @ J0.T @ e_; s_ = Jv.T @ rp; schur = 0.5 * float(s_ @ torch.linalg.solve(Jv.T @ Hp @ Jv + mu * torch.eye(c), s_))
    check('T17c', 'BN slot: model change from the network Jacobians after insertion = Schur gain from the realised BN feature phi', f'{abs(model + schur) / schur:.1e}', abs(model + schur) / schur < 1e-8)
    # T14(a) is exact only as eps_BN -> 0: relative error ~ eps_BN / var
    s = torch.randn(Nn, 3); a = torch.randn(4); Xc = X - X.mean(0); errs = {}
    for epsb in (1e-5, 1e-14):
        bn1 = torch.nn.BatchNorm1d(1, affine=False, eps=epsb); bn1.train(); phi = bn1((X @ a)[:, None])[:, 0].detach()
        lhs = float(((phi[:, None] * s).sum(0))**2 @ torch.ones(3)); rhs = Nn * float((Xc @ a) @ s @ s.T @ (Xc @ a)) / float((Xc @ a) @ (Xc @ a))
        errs[epsb] = abs(lhs - rhs) / rhs
    var = float((Xc @ a).pow(2).mean())
    check('T17d', 'T14(a) with BN eps: relative error <= eps/var (exact as eps -> 0)', f'eps=1e-5: {errs[1e-5]:.1e} (eps/var {1e-5 / var:.1e}); eps=1e-14: {errs[1e-14]:.1e}', errs[1e-5] <= 1e-5 / var * (1 + 1e-6) and errs[1e-14] < 1e-12)

# ====================================================== T18: tolerant skip rule with inexact (CG) solves
def T18():
    viol, viol_neg, strict_skip, tol_skip, n = 0, 0, 0, 0, 0
    for t in range(60):
        P = problem(scale=[1., 3., 6.][t % 3], drift=2.0, p=50); S = setup(P); H, J, mu = P['H'], P['J'], S['mu']
        if t % 2 == 0:                                                  # V_O = 0 construction (as T11a)
            zO = S['z'](P['eOf']); NzO = Nstat(P, zO); U_, s_, Vh = torch.linalg.svd(NzO.T)
            rk = int((s_ > 1e-12 * float(s_.max())).sum()); A = Vh[rk:].T @ rn(Vh[rk:].shape[0], 1)
        else:
            A = rn(P['da'], 2)
        KE = KE_of(P, A); A0 = S['F'] + mu * S['I']
        b1 = S['g']; x1, rho1 = cg(A0, b1, torch.zeros_like(b1), 1e-6, 400)
        zh = H @ J @ x1; uh = KE @ zh; b2 = J.T @ H @ uh; dh, rho2 = cg(A0, b2, torch.zeros_like(b2), 1e-6, 400)
        dstar = S['G'] @ J.T @ H @ KE @ S['z'](S['e'])                   # exact d*(0) = G J^T H K_E z
        nHJ = float(torch.linalg.matrix_norm(H @ J, ord=2)); nJHK = float(torch.linalg.matrix_norm(J.T @ H @ KE, ord=2))
        Dhat = nJHK * nHJ * rho1 * float(b1.norm()) / mu**2 + rho2 * float(b2.norm()) / mu
        Dneg = nJHK * nHJ * rho1 * float(b1.norm()) + rho2 * float(b2.norm())               # mutant: 1/mu factors dropped
        err = float((dh - dstar).norm()); viol += int(err > Dhat); viol_neg += int(err > Dneg)
        hh, hs = float(S['gO'] @ dh), float(S['gO'] @ dstar); tau = float(S['gO'].norm()) * Dhat
        if t % 2 == 0:
            n += 1; strict_skip += int(hh < 0); tol_skip += int(hh < -tau)
            viol += int(abs(hh - hs) > tau)
    check('T18a', 'a-posteriori CG bound ||d_hat - d*|| <= Dhat = ||J^T H K_E|| ||HJ|| rho1 ||b1||/mu^2 + rho2 ||b2||/mu, hence |h_hat - h*| <= ||g_O|| Dhat', f'{viol} violations / 60', viol == 0)
    check('T18a-NEG', 'the same bound without the 1/mu amplification is violated', f'{viol_neg}/60', viol_neg > 0)
    check('T18b-WIT', 'V_O = 0 (h* = 0) with CG: the strict test h_hat < 0 skips certified steps', f'{strict_skip}/{n}', strict_skip > 0)
    check('T18c', 'tolerant rule (skip iff h_hat < -||g_O|| Dhat) never skips a step whose exact h* >= 0', f'{tol_skip}/{n} spurious skips', tol_skip == 0)

if __name__ == '__main__':
    for fn in (T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18):
        print(f'=== {fn.__name__}', flush=True); fn()
    npass = sum(ok for _, ok in RES); neg = [c for c, _ in RES if 'NEG' in c]; wit = [c for c, _ in RES if c.endswith('WIT')]
    print(f'\nSUMMARY: {npass}/{len(RES)} checks pass ({len(neg)} mutation negative controls, {len(wit)} existence witnesses); failing: {[c for c, ok in RES if not ok]}')
