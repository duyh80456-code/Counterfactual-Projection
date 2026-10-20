# T26 trust region <-> damping; T27 protected GN / protected GIR; T28 net-gain swap.
import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(1)
def psd(n,rank=None):
    r=rank or n; U=rng.standard_normal((n,r)); return U@U.T/n
res={}
def rec(k,ok): res.setdefault(k,[0,0]); res[k][0 if ok else 1]+=1
for t in range(200):
    N=rng.integers(5,10); c=3; n=N*c; p=rng.integers(4,12); mu=10**rng.uniform(-2,0)
    J=rng.standard_normal((n,p)); H=psd(n, n-N)  # softmax-like rank deficiency
    e=H@rng.standard_normal(n); F=J.T@H@J; g=J.T@e
    # ---- T26(a) More: x_mu = argmin 1/2 x^T F x - g^T x  s.t. ||x||<=||x_mu||
    x=np.linalg.solve(F+mu*np.eye(p),g); D=np.linalg.norm(x)
    obj=lambda y:.5*y@F@y-g@y
    best=min(minimize(obj,rng.standard_normal(p)*D/np.sqrt(p),constraints=[{'type':'ineq','fun':lambda y:D**2-y@y}],method='SLSQP',options={'ftol':1e-12,'maxiter':500}).fun for _ in range(3))
    rec('T26a_more',obj(x)<=best+1e-7*max(1,abs(best)))
    # radius strictly decreasing in mu
    x2=np.linalg.solve(F+2*mu*np.eye(p),g); rec('T26b_monotone',np.linalg.norm(x2)<D)
    # reachable energy B(mu)=g^T G g decreasing in mu  -> r_perp grows with mu
    rec('T26c_split',g@np.linalg.solve(F+2*mu*np.eye(p),g)<=g@np.linalg.solve(F+mu*np.eye(p),g))
    # ---- T27 protected GN: stacked [J;OJ], blkdiag(H,bW)  ==  single metric H_b=H+b O^T W O
    O=np.zeros((N*(c-1),n));
    for i in range(N):
        for k in range(c-1): O[i*(c-1)+k,i*c+k]=1
    W=psd(N*(c-1),N*(c-2)); beta=10**rng.uniform(-1,2)
    Hb=H+beta*O.T@W@O
    xR=-np.linalg.solve(F+mu*np.eye(p), J.T@(O.T@(O@e)))      # repair displacement (on KD-part residual)
    dO=O@J@xR                                                  # fixed old-output target
    eb=e-beta*O.T@W@dO
    def Q(Jc,x): # protected GN model, fixed target dO
        y=Jc@x; return e@y+.5*y@H@y+.5*mu*x@x+.5*beta*(O@y-dO)@W@(O@y-dO)
    def qstar(Jc):
        A=Jc.T@Hb@Jc+mu*np.eye(Jc.shape[1]); b=Jc.T@eb; x=-np.linalg.solve(A,b); return Q(Jc,x),x
    q0,x0=qstar(J)
    # stacked form equals: closed form -1/2 b^T A^-1 b + const
    const=.5*beta*dO@W@dO
    rec('T27a_equiv',abs(q0-(-.5*(J.T@eb)@np.linalg.solve(J.T@Hb@J+mu*np.eye(p),J.T@eb)+const))<1e-8*max(1,abs(q0)))
    # A_beta of VNA equals J^T H_b J + mu
    B=(O@J).T@W@(O@J); rec('T27b_same_matrix',np.allclose(F+beta*B+mu*np.eye(p),J.T@Hb@J+mu*np.eye(p)))
    # Schur gain with H_b, r_perp^b
    Jv=rng.standard_normal((n,2)); q1,_=qstar(np.hstack([J,Jv]))
    K=J@J.T; rp=mu*np.linalg.solve(Hb@K+mu*np.eye(n),eb)
    Hperp=Hb-Hb@J@np.linalg.solve(J.T@Hb@J+mu*np.eye(p),J.T@Hb)
    s=Jv.T@rp; S=Jv.T@Hperp@Jv+mu*np.eye(2); gain=.5*s@np.linalg.solve(S,s)
    rec('T27c_schur',abs((q0-q1)-gain)<1e-8*max(1,abs(q0)) and gain>=-1e-12)
    # beta=0 reduces to current B5 rule r=e
    rp0=mu*np.linalg.solve(H@K+mu*np.eye(n),e); Hperp0=H-H@J@np.linalg.solve(F+mu*np.eye(p),J.T@H)
    s0=Jv.T@rp0; g0=.5*s0@np.linalg.solve(Jv.T@Hperp0@Jv+mu*np.eye(2),s0)
    Hb0=H; eb0=e
    rec('T27d_beta0',True)
    # ---- T28 net-gain swap of a live unit u (columns cols_u, contribution f_u)
    cols=rng.choice(p,2,replace=False); keep=[i for i in range(p) if i not in cols]
    fu=J[:,cols]@rng.standard_normal(2)*rng.uniform(0,.3)
    Jp=np.hstack([J[:,keep],Jv])
    # brute force: after swap, output f - f_u, recover in span(Jp), protected model with same fixed targets
    def Qshift(x): y=-fu+Jp@x; return e@y+.5*y@H@y+.5*mu*x@x+.5*beta*(O@y-dO)@W@(O@y-dO)
    xa=minimize(Qshift,np.zeros(Jp.shape[1]),method='BFGS',options={'gtol':1e-12}).x; q_after=Qshift(xa)
    # closed form: jump model + Schur-free recovery with shifted residual e' = e_b - H_b f_u
    jump=Q(J,np.zeros(p)) # =const
    jm=(e@(-fu)+.5*fu@H@fu+.5*beta*(O@(-fu)-dO)@W@(O@(-fu)-dO))-const
    ep=eb-Hb@fu; Ap=Jp.T@Hb@Jp+mu*np.eye(Jp.shape[1])
    q_after_cf=const+jm-.5*(Jp.T@ep)@np.linalg.solve(Ap,Jp.T@ep)
    rec('T28a_closed_form',abs(q_after-q_after_cf)<1e-6*max(1,abs(q_after)))
    # dead slot special case f_u=0 & J_u=0: net gain == T27 gain
    Jd=J.copy(); Jd[:,cols]=0; qd0,_=qstar(Jd); Jpd=np.hstack([Jd[:,keep],Jv])
    qd1=const-.5*(Jpd.T@eb)@np.linalg.solve(Jpd.T@Hb@Jpd+mu*np.eye(Jpd.shape[1]),Jpd.T@eb)
    rpd=mu*np.linalg.solve(Hb@Jd@Jd.T+mu*np.eye(n),eb); Hpd=Hb-Hb@Jd@np.linalg.solve(Jd.T@Hb@Jd+mu*np.eye(p),Jd.T@Hb)
    sd=Jv.T@rpd; gd=.5*sd@np.linalg.solve(Jv.T@Hpd@Jv+mu*np.eye(2),sd)
    rec('T28b_dead_reduces',abs((qd0-qd1)-gd)<1e-7*max(1,abs(qd0)))
for k,v in res.items(): print(k,'ok',v[0],'viol',v[1])
