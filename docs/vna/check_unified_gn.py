"""Unified projection: GN step x_GN, K-part x_K, virtual part d (float64).

Part 1 (math, U1-U8): random problems with softmax CE(+KD) output Hessians.
Part 2 (toy): toy A (CIL-like) and toy B (plasticity) from check_continuous.py,
arms sgd, sgd_kself (x_K = K_E->K control), sgd_vna (d), sgd_gn (x_GN), sgd_gn_vna (x_GN + d),
plus line-search variants sgd_vna_ls, sgd_gn_ls.
Usage: python3 -I check_unified_gn.py [math|toy s0 s1]
"""
import sys, math, json, os
import torch
torch.set_default_dtype(torch.float64)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES = []


def rec(name, ok, msg):
    RES.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")


def rel(a, b):
    return float((a - b).norm() / max(float(b.norm()), 1e-300))


def cos(a, b):
    return float(a @ b / max(float(a.norm() * b.norm()), 1e-300))


def problem(gen, N=6, c=4, p=30, old=2, T=2.0, rE=3, scale=2.0, lowrank=None):
    f = scale * torch.randn(N, c, generator=gen)
    y = torch.randint(c, (N,), generator=gen)
    teacher = scale * torch.randn(N, c, generator=gen)
    P = torch.softmax(f, 1)
    e = (P - torch.nn.functional.one_hot(y, c)) / N
    Hb = []
    for i in range(N):
        Hi = (torch.diag(P[i]) - torch.outer(P[i], P[i])) / N
        if old > 0:
            ps = torch.softmax(f[i, :old] / T, 0); pt = torch.softmax(teacher[i, :old] / T, 0)
            e[i, :old] += (ps - pt) / (N * T)
            Hi[:old, :old] += (torch.diag(ps) - torch.outer(ps, ps)) / (N * T * T)
        Hb.append(Hi)
    H = torch.block_diag(*Hb)
    n = N * c
    if lowrank:
        J = torch.randn(n, lowrank, generator=gen) @ torch.randn(lowrank, p, generator=gen) / math.sqrt(p)
    else:
        J = torch.randn(n, p, generator=gen) / math.sqrt(p)
    JE = torch.randn(n, rE, generator=gen)
    KE = JE @ JE.T
    e = e.reshape(-1)
    F = J.T @ H @ J
    mu = 1e-2 * float(F.trace()) / n
    G = torch.linalg.inv(F + mu * torch.eye(p))
    return dict(J=J, H=H, e=e, KE=KE, F=F, G=G, mu=mu, n=n, p=p)


def parts(P):
    J, H, e, KE, F, G, mu = (P[k] for k in ('J', 'H', 'e', 'KE', 'F', 'G', 'mu'))
    K = J @ J.T
    g = J.T @ e
    xGN = G @ g
    z = H @ J @ xGN
    d = G @ J.T @ H @ (KE @ z)
    xK = G @ J.T @ H @ (K @ z)
    return K, g, xGN, z, d, xK


def math_checks(trials=200):
    gen = torch.Generator().manual_seed(2026)
    # U1 projection closed form
    worst = 0.; neg = []
    for t in range(40):
        P = problem(gen)
        J, H, G, mu, n, p = P['J'], P['H'], P['G'], P['mu'], P['n'], P['p']
        tgt = torch.randn(n, generator=gen)
        x = G @ J.T @ H @ tgt
        L, Q = torch.linalg.eigh(H); R = Q @ torch.diag(L.clamp_min(0).sqrt()) @ Q.T
        A = torch.cat([R @ J, math.sqrt(mu) * torch.eye(p)]); b = torch.cat([R @ tgt, torch.zeros(p)])
        xl = torch.linalg.lstsq(A, b.unsqueeze(1), driver='gelsd').solution[:, 0]
        worst = max(worst, rel(x, xl)); neg.append(rel(G @ J.T @ tgt, xl))
    rec('U1', worst < 1e-8, f'argmin ||Jx-t||_H^2+mu||x||^2 = G J^T H t, max rel err {worst:.1e}')
    rec('U1-NEG', min(neg) > 1e-2, f'G J^T t (no H) rejected, min rel err {min(neg):.2e}')
    # U2 x_K identities, U3 linearity, U4 split identities, U5 gains
    e2 = e3 = e4a = e4b = e4c = 0.; negU2 = []; negU4 = []; gK = []; gV = []; gEneg = 0; cosKGN = []
    for t in range(trials):
        P = problem(gen, lowrank=(None if t % 2 else 8))
        J, H, e, KE, F, G, mu = (P[k] for k in ('J', 'H', 'e', 'KE', 'F', 'G', 'mu'))
        K, g, xGN, z, d, xK = parts(P)
        FG = F @ G
        e2 = max(e2, rel(xK, FG @ FG @ g), rel(xK, G @ F @ F @ G @ g))
        # NGD arm of the code base: d with K_E -> K
        e2 = max(e2, rel(xK, G @ J.T @ H @ (J @ J.T) @ (H @ J @ G @ J.T @ e)))
        negU2.append(rel(xGN, xK)); cosKGN.append(cos(xGN, xK))
        e3 = max(e3, rel(G @ J.T @ H @ ((K + KE) @ z), xK + d))
        r_perp = e - z
        M = torch.linalg.inv(H @ K + mu * torch.eye(P['n']))
        e4a = max(e4a, rel(r_perp, mu * M @ e))
        e4b = max(e4b, rel(K @ e - K @ z, mu * J @ xGN))
        e4c = max(e4c, rel(z, H @ K @ M @ e))
        negU4.append(rel(K @ e - K @ z, mu * K @ e))
        gK.append(float(g @ xK) - float((FG @ g).norm() ** 2)); gV.append(float(g @ d) - float(((KE @ z) @ z)))
        if float(g @ xK) < -1e-14 or float(g @ d) < -1e-14:
            gK.append(1e9)
        dE = G @ J.T @ H @ (KE @ e)
        gEneg += float(g @ dE) < 0
    rec('U2', e2 < 1e-8, f'x_K = G J^T H K z = (FG)^2 g = G F^2 G g = code NGD arm (d with K_E->K), max rel {e2:.1e}')
    rec('U2-WIT', min(negU2) > 1e-3, f'x_K != x_GN = G g: rel diff min {min(negU2):.2e}, median cos {sorted(cosKGN)[len(cosKGN)//2]:.3f}')
    rec('U3', e3 < 1e-8, f'P((K+K_E) z) = x_K + d, max rel {e3:.1e}')
    rec('U4', max(e4a, e4b, e4c) < 1e-7, f'e = z + r_perp with r_perp = mu(HK+mu)^-1 e, z = HK(HK+mu)^-1 e, Ke - Kz = mu J x_GN; max rel {max(e4a, e4b, e4c):.1e}')
    rec('U4-NEG', min(negU4) > 1e-2, f'wrong identity Ke-Kz = mu K e rejected, min rel {min(negU4):.2e}')
    rec('U5', max(abs(v) for v in gK) < 1e-10 and max(abs(v) for v in gV) < 1e-10,
        f'g^T x_K = ||FGg||^2 >= 0 and g^T d = z^T K_E z = V >= 0 ({trials} problems)')
    rec('U5-WIT', gEneg > 0, f'with cotangent e instead of z, g^T G J^T H K_E e < 0 in {gEneg}/{trials} problems (z is what makes the virtual part a PSD preconditioner)')
    # U6 grown damped-GN dynamics: first-order theta-correction
    errs = []; negs = []; cs = []
    for t in range(60):
        P = problem(gen)
        J, H, e, KE, F, G, mu = (P[k] for k in ('J', 'H', 'e', 'KE', 'F', 'G', 'mu'))
        K, g, xGN, z, d, xK = parts(P)
        n = P['n']
        step = lambda s: -J.T @ torch.linalg.solve(H @ (K + s * KE) + mu * torch.eye(n), e)
        h = 1e-5 * mu / max(float(torch.linalg.matrix_norm(H @ KE, 2)), 1e-300)
        fd = (step(h) - step(-h)) / (2 * h)
        pred = G @ J.T @ H @ (KE @ (e - z)) / mu
        errs.append(rel(fd, pred)); negs.append(rel(fd, d)); cs.append(abs(cos(pred, d)))
        assert rel(step(0.), -xGN) < 1e-8
    rec('U6', max(errs) < 1e-5, f'd/dt of grown damped-GN theta-step at K_E=0 equals G J^T H K_E r_perp / mu, max rel {max(errs):.1e}')
    rec('U6-NEG', min(negs) > 1e-2, f'"virtual part of grown GN = VNA d" rejected: min rel {min(negs):.2e}, median |cos(pred,d)| {sorted(cs)[len(cs)//2]:.3f}')
    # U7 z-flow K-part vs gradient flow: ||Ke-Kz|| / ||Ke|| -> 0 as mu -> 0
    P = problem(torch.Generator().manual_seed(7))
    J, H, e, F = P['J'], P['H'], P['e'], P['F']
    K = J @ J.T; ratios = []
    for mr in (1e0, 1e-1, 1e-2, 1e-3, 1e-4):
        mu = mr * float(F.trace()) / P['n']
        G = torch.linalg.inv(F + mu * torch.eye(P['p']))
        z = H @ J @ G @ (J.T @ e)
        ratios.append(float((K @ e - K @ z).norm() / (K @ e).norm()))
    rec('U7', all(ratios[i + 1] < ratios[i] for i in range(len(ratios) - 1)) and ratios[-1] < 1e-2,
        'K-part of z-flow -> gradient flow as mu -> 0: ||Ke-Kz||/||Ke|| = ' + ', '.join(f'{r:.1e}' for r in ratios))
    # U8 GIR consistency: enlarging span by J_v lowers GN-model value by 1/2 s^T S^-1 s, s = J_v^T r_perp
    errs = []; negs = []
    for t in range(60):
        P = problem(gen)
        J, H, e, G, mu = P['J'], P['H'], P['e'], P['G'], P['mu']
        Jv = torch.randn(P['n'], 3, generator=gen)
        def qstar(Jm):
            Gm = torch.linalg.inv(Jm.T @ H @ Jm + mu * torch.eye(Jm.shape[1]))
            v = Jm.T @ e
            return -0.5 * float(v @ Gm @ v)
        Jp = torch.cat([J, Jv], 1)
        z = H @ J @ G @ (J.T @ e); rp = e - z
        Hp = H - H @ J @ G @ J.T @ H
        S = Jv.T @ Hp @ Jv + mu * torch.eye(3)
        s = Jv.T @ rp
        gain = 0.5 * float(s @ torch.linalg.solve(S, s))
        true = qstar(J) - qstar(Jp)
        errs.append(abs(gain - true) / max(abs(true), 1e-300))
        s2 = Jv.T @ z
        negs.append(abs(0.5 * float(s2 @ torch.linalg.solve(S, s2)) - true) / max(abs(true), 1e-300))
    rec('U8', max(errs) < 1e-8, f'swap gain = q*(J) - q*([J,J_v]) = 1/2 s^T S^-1 s with s = J_v^T r_perp, max rel {max(errs):.1e}')
    rec('U8-NEG', min(negs) > 1e-2, f'same formula with z in place of r_perp rejected, min rel {min(negs):.2e}')
    # U9 lambda*: maximiser of GN-model decrease along -d
    errs = []
    for t in range(30):
        P = problem(gen)
        K, g, xGN, z, d, xK = parts(P)
        Fm = P['F'] + P['mu'] * torch.eye(P['p'])
        ls = float(g @ d) / float(d @ Fm @ d)
        grid = torch.linspace(0, 3 * ls, 3001)
        vals = grid * float(g @ d) - 0.5 * grid ** 2 * float(d @ Fm @ d)
        errs.append(abs(float(grid[vals.argmax()]) - ls) / ls)
    rec('U9', max(errs) < 2e-3, f'lambda* = g^T d / d^T(F+mu)d maximises the GN-model decrease (grid check, max rel {max(errs):.1e})')


# ---------------- toy ----------------
def toy(s0, s1):
    import check_continuous as cc
    T = cc.T

    def corr_all(m, x, y, old, teacher):
        f = m(x).detach()
        e, eO, H = cc.loss_parts(f, y, old, teacher)
        J = cc.jac(m, x); n, p = J.shape
        F = J.T @ H @ J
        mu = max(1e-2 * float(F.trace()) / n, 1e-12)
        G = torch.linalg.inv(F + mu * torch.eye(p))
        ev, evO = e.reshape(-1) + eO.reshape(-1), eO.reshape(-1)
        g = J.T @ ev; gO = J.T @ evO
        xGN = G @ g
        z = H @ J @ xGN
        KE = cc.virtual_KE(m, x, z.reshape(f.shape), 4)
        d = G @ J.T @ H @ (KE @ z)
        xK = G @ J.T @ H @ (J @ (J.T @ z))
        return dict(g=g, gO=gO, d=d, xK=xK, xGN=xGN, gnvna=xGN + d, Fm=F + mu * torch.eye(p))

    vec_of = {'sgd_kself': 'xK', 'sgd_vna': 'd', 'sgd_gn': 'xGN', 'sgd_gn_vna': 'gnvna', 'sgd_vna_ls': 'd', 'sgd_gn_ls': 'xGN'}

    def apply(arm, C, old, lr):
        v = C[vec_of[arm]]
        if v.norm() == 0: return None
        if float(C['g'] @ v) < 0 or (old > 0 and float(C['gO'] @ v) < -1e-12): return None
        lam = float(C['g'].norm() / v.norm())
        if arm.endswith('_ls'):
            lstar = float(C['g'] @ v) / float(v @ C['Fm'] @ v) / lr
            lam = min(lam, lstar)
        return lr * lam * v

    arms = ['sgd', 'sgd_kself', 'sgd_vna', 'sgd_gn', 'sgd_gn_vna', 'sgd_vna_ls', 'sgd_gn_ls']

    def toyA(seed, steps=300, lr=0.1, K=5):
        g = torch.Generator().manual_seed(1000 + seed)
        din, Cn = 20, 10
        means = 0.55 * torch.randn(Cn, din, generator=g)
        def sample(nn):
            y = torch.arange(Cn).repeat_interleave(nn)
            return means[y] + torch.randn(len(y), din, generator=g), y
        xtr, ytr = sample(150); xte, yte = sample(150)
        t1 = ytr < 5; t2 = ytr >= 5
        base = cc.make_model(din, 16, Cn, seed)
        opt = torch.optim.SGD(base.parameters(), lr=lr)
        X1, Y1 = xtr[t1], ytr[t1]
        for s in range(steps):
            idx = torch.randint(len(X1), (64,), generator=g)
            opt.zero_grad(); torch.nn.functional.cross_entropy(base(X1[idx]), Y1[idx]).backward(); opt.step()
        te_old = yte < 5; te_new = yte >= 5
        a1 = cc.acc(base, xte[te_old], yte[te_old])
        ex = torch.cat([torch.nonzero(ytr == k)[:6, 0] for k in range(5)])
        Xe, Ye = xtr[ex], ytr[ex]
        tm = cc.make_model(din, 16, Cn, seed); tm.load_state_dict(base.state_dict())
        X2, Y2 = xtr[t2], ytr[t2]
        out = {}
        for arm in arms:
            m = cc.make_model(din, 16, Cn, seed); m.load_state_dict(base.state_dict())
            gg = torch.Generator().manual_seed(77 + seed)
            skips = 0; speed = None
            for s in range(steps):
                i2 = torch.randint(len(X2), (48,), generator=gg); ie = torch.randint(len(Xe), (16,), generator=gg)
                x = torch.cat([X2[i2], Xe[ie]]); y = torch.cat([Y2[i2], Ye[ie]])
                with torch.no_grad(): tl = tm(x)
                m.zero_grad(); L = cc.total_loss(m, x, y, 5, tl); L.backward()
                gvec = torch.cat([p.grad.reshape(-1) for p in m.parameters()])
                th = cc.flat(m) - lr * gvec
                if s % K == 0 and arm != 'sgd':
                    st = apply(arm, corr_all(m, x, y, 5, tl), 5, lr)
                    if st is None: skips += 1
                    else: th = th - st
                cc.setflat(m, th)
                if s == 49:
                    with torch.no_grad(): speed = float(torch.nn.functional.cross_entropy(m(X2), Y2))
            a_old = cc.acc(m, xte[te_old], yte[te_old]); a_new = cc.acc(m, xte[te_new], yte[te_new])
            out[arm] = dict(forget=a1 - a_old, old=a_old, new=a_new, avg=cc.acc(m, xte, yte), speed_loss50=speed, skips=skips)
        return out

    def toyB(seed, tasks=6, steps=200, lr=0.1, K=5):
        g = torch.Generator().manual_seed(5000 + seed)
        din, Cn, Nn = 10, 4, 256
        x = torch.randn(Nn, din, generator=g)
        m0 = cc.make_model(din, 6, Cn, seed)
        labels = []
        for t in range(tasks):
            tmm = cc.make_model(din, 32, Cn, 9000 + 97 * seed + t)
            with torch.no_grad(): labels.append((3 * tmm(x)).argmax(1))
        out = {}
        for arm in arms:
            m = cc.make_model(din, 6, Cn, seed); m.load_state_dict(m0.state_dict())
            gg = torch.Generator().manual_seed(99 + seed)
            finals, speeds = [], []
            for t in range(tasks):
                y = labels[t]
                for s in range(steps):
                    idx = torch.randint(Nn, (64,), generator=gg)
                    m.zero_grad(); L = torch.nn.functional.cross_entropy(m(x[idx]), y[idx]); L.backward()
                    gvec = torch.cat([p.grad.reshape(-1) for p in m.parameters()])
                    th = cc.flat(m) - lr * gvec
                    if s % K == 0 and arm != 'sgd':
                        st = apply(arm, corr_all(m, x[idx], y[idx], 0, None), 0, lr)
                        if st is not None: th = th - st
                    cc.setflat(m, th)
                    if s == 29:
                        with torch.no_grad(): speeds.append(float(torch.nn.functional.cross_entropy(m(x), y)))
                with torch.no_grad(): finals.append(float(torch.nn.functional.cross_entropy(m(x), y)))
            out[arm] = dict(final_last3=sum(finals[-3:]) / 3, speed_last3=sum(speeds[-3:]) / 3)
        return out

    R = []
    for s in range(s0, s1):
        R.append(dict(seed=s, A=toyA(s), B=toyB(s)))
        print('seed', s, 'done', flush=True)
    with open(f'unified_gn_toy_{s0}_{s1}.json', 'w') as fh:
        json.dump(R, fh)


def summarize(files):
    R = []
    for fn in files:
        R += json.load(open(fn))
    n = len(R)
    arms = list(R[0]['A'].keys())
    print(f'Toy A (CIL-like), {n} seeds: mean (avg acc, forget, new, speed_loss50, skips)')
    for a in arms:
        mm = lambda k: sum(r['A'][a][k] for r in R) / n
        print(f"  {a:12s} avg {mm('avg'):.4f} forget {mm('forget'):+.4f} new {mm('new'):.4f} speed50 {mm('speed_loss50'):.4f} skips {mm('skips'):.1f}")
    def pair(A, B, key, lower=False, which='A'):
        dif = [r[which][A][key] - r[which][B][key] for r in R]
        w = sum((x < 0) if lower else (x > 0) for x in dif)
        return f"{A} vs {B} [{key}]: {sum(dif)/n:+.4f}, wins {w}/{n}"
    for A, B in [('sgd_vna', 'sgd'), ('sgd_vna', 'sgd_kself'), ('sgd_vna', 'sgd_gn'), ('sgd_gn', 'sgd'),
                 ('sgd_gn_vna', 'sgd_gn'), ('sgd_gn_vna', 'sgd_vna'), ('sgd_vna_ls', 'sgd_gn_ls'), ('sgd_vna_ls', 'sgd')]:
        print('  ' + pair(A, B, 'avg'))
    print('  ' + pair('sgd_vna', 'sgd_gn', 'speed_loss50', lower=True))
    print(f'Toy B (plasticity), {n} seeds: mean final_last3 (lower better)')
    for a in arms:
        print(f"  {a:12s} final {sum(r['B'][a]['final_last3'] for r in R)/n:.4f} speed {sum(r['B'][a]['speed_last3'] for r in R)/n:.4f}")
    for A, B in [('sgd_vna', 'sgd'), ('sgd_vna', 'sgd_kself'), ('sgd_vna', 'sgd_gn'), ('sgd_gn', 'sgd'),
                 ('sgd_gn_vna', 'sgd_gn'), ('sgd_vna_ls', 'sgd_gn_ls')]:
        print('  ' + pair(A, B, 'final_last3', lower=True, which='B'))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'math'
    if mode == 'math':
        math_checks()
        print(f"\nSUMMARY: {sum(ok for _, ok in RES)}/{len(RES)} checks pass; failing: {[n for n, ok in RES if not ok]}")
    elif mode == 'toy':
        toy(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == 'sum':
        summarize(sys.argv[2:])
