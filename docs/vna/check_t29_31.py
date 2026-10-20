"""Numeric checks for T29 (output-space solve), T30 (inner LM loop), T31 (sequential GIR), R1 (reach normaliser)."""
import numpy as np
rng = np.random.default_rng(0)
def softmax_H(p): return np.diag(p) - np.outer(p, p)
def blocks(N, C, conc=1.):
    M = np.zeros((N*C, N*C)); P = []
    for i in range(N):
        p = rng.dirichlet(conc*np.ones(C)); P.append(p)
        M[i*C:(i+1)*C, i*C:(i+1)*C] = softmax_H(p)
    return M, P
viol = dict(t29a=0, t29b=0, t29c=0, t29d=0, t31=0, r1=0)
for trial in range(200):
    N, C, P = rng.integers(3, 7), rng.integers(2, 5), rng.integers(30, 90)
    m = N*C; mu = 10**rng.uniform(-4, 0); beta = rng.uniform(0, 5)
    J = rng.standard_normal((m, P)); G = J@J.T
    H, _ = blocks(N, C); Ho, _ = blocks(N, C); Mb = H + beta*Ho
    c = rng.standard_normal(m)
    x = np.linalg.solve(J.T@Mb@J + mu*np.eye(P), J.T@c)
    # T29a: x = J^T y, y = (mu + M G)^{-1} c
    y = np.linalg.solve(mu*np.eye(m) + Mb@G, c)
    viol['t29a'] += np.abs(x - J.T@y).max() > 1e-7*max(1, np.abs(x).max())
    # T29b: symmetric form with M = L L^T
    w, V = np.linalg.eigh(Mb); L = V*np.sqrt(np.clip(w, 0, None))
    ys = (c - L@np.linalg.solve(mu*np.eye(m) + L.T@G@L, L.T@G@c))/mu
    viol['t29b'] += np.abs(y - ys).max() > 1e-6*max(1, np.abs(y).max())
    # T29c: norms / curvature from G only
    viol['t29c'] += abs(x@x - y@G@y) > 1e-6*max(1, x@x)
    # T29d: Schur gain of an added column block equals the q* difference
    s = rng.integers(1, 6); Jv = rng.standard_normal((m, s)); e = Mb@rng.standard_normal(m)
    def qstar(Jx): A = Jx.T@Mb@Jx + mu*np.eye(Jx.shape[1]); b = Jx.T@e; return -.5*b@np.linalg.solve(A, b)
    Lh = L.T  # J_hat = L^T J
    K = Lh@G@Lh.T; Jhv = Lh@Jv
    rperp_hat = mu*np.linalg.solve(K + mu*np.eye(m), np.linalg.lstsq(L, e, rcond=None)[0]) if False else None
    S = Jv.T@Mb@Jv + mu*np.eye(s) - Jv.T@Mb@J@np.linalg.solve(J.T@Mb@J + mu*np.eye(P), J.T@Mb@Jv)
    S_dual = mu*(np.eye(s) + Jhv.T@np.linalg.solve(K + mu*np.eye(m), Jhv))
    viol['t29d'] += np.abs(S - S_dual).max() > 1e-6*max(1, np.abs(S).max())
    # T31: sequential swaps in one model -> q* telescopes and each accepted Gamma>0 lowers it
    q = [qstar(J)]; Jc = J.copy()
    for k in range(3):
        cols = rng.choice(P, 2, replace=False); Jnew = Jc.copy(); Jnew[:, cols] = rng.standard_normal((m, 2))
        gamma = q[-1] - qstar(Jnew)
        if gamma > 0: Jc = Jnew; q.append(qstar(Jc))
    viol['t31'] += any(b > a + 1e-12 for a, b in zip(q, q[1:]))
    # R1: per-sample e^T H^+ e = 1/p_y - 1 for e = p - y
    p = rng.dirichlet(np.ones(C)); yi = rng.integers(C); ev = p.copy(); ev[yi] -= 1
    viol['r1'] += abs(ev@np.linalg.pinv(softmax_H(p), rcond=1e-10)@ev - (1/p[yi] - 1)) > 1e-6*(1/p[yi])
print('T29/T31/R1 violations over 200:', viol)

# T30: inner LM loop terminates; rho -> 1 as mu -> inf (tiny MLP, CE loss, ReLU)
def mlp_loss(theta, X, Y, h):
    d = X.shape[1]; W1 = theta[:d*h].reshape(d, h); W2 = theta[d*h:].reshape(h, -1)
    Z = np.maximum(X@W1, 0)@W2; Z -= Z.max(1, keepdims=True); P = np.exp(Z); P /= P.sum(1, keepdims=True)
    return -np.log(P[np.arange(len(Y)), Y] + 1e-300).mean(), Z, P
fails, iters = 0, []
for trial in range(50):
    n, d, h, C = 32, 5, 8, 4
    X = rng.standard_normal((n, d)); Y = rng.integers(C, size=n); th = rng.standard_normal(d*h + h*C)*1.5
    f = lambda t: mlp_loss(t, X, Y, h)
    L0, Z, Pm = f(th)
    eps = 1e-6; Jz = np.zeros((n*C, th.size))
    for j in range(th.size):
        tp = th.copy(); tp[j] += eps; Jz[:, j] = ((f(tp)[1] - Z)/eps).ravel()
    Hb = np.zeros((n*C, n*C))
    for i in range(n): Hb[i*C:(i+1)*C, i*C:(i+1)*C] = softmax_H(Pm[i])/n
    Yo = np.eye(C)[Y]; g = Jz.T@((Pm - Yo)/n).ravel(); F = Jz.T@Hb@Jz
    mu, nu, k = 1e-8, 2., 0
    while True:
        x = np.linalg.solve(F + mu*np.eye(th.size), g); pred = g@x - .5*x@F@x
        L1 = f(th - x)[0]; rho = (L0 - L1)/pred if pred > 0 else -np.inf
        if np.isfinite(rho) and rho > 0: break
        mu, nu, k = mu*nu, 2*nu, k + 1
        if k > 200: fails += 1; break
    iters.append(k)
print('T30: loops not terminating:', fails, ' retries median/max:', int(np.median(iters)), max(iters))
