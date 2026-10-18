# T25: VNA gain bounded by live reachability; monotone in K; GIR raises bound by 2*gain.
import numpy as np
rng=np.random.default_rng(0); ok={'a':0,'b':0,'c':0,'d':0}; viol={'a':0,'b':0,'c':0,'d':0}; T=300
tight=[]
for t in range(T):
    N=rng.integers(4,12); p=rng.integers(3,15); q=rng.integers(2,8); mu=10**rng.uniform(-3,0)
    J=rng.standard_normal((N,p)); Rh=rng.standard_normal((N,N)); H=Rh@Rh.T/N
    if rng.random()<.3: # rank-deficient H (softmax-like)
        U=rng.standard_normal((N,N-1)); H=U@U.T/N
    w=np.linalg.eigh(H); R=w[1]@np.diag(np.sqrt(np.clip(w[0],0,None)))@w[1].T
    e=H@rng.standard_normal(N)              # e in range(H)
    JE=rng.standard_normal((N,q)); KE=JE@JE.T
    def parts(J):
        F=J.T@H@J; G=np.linalg.inv(F+mu*np.eye(J.shape[1])); g=J.T@e
        z=H@J@G@g; V=z@KE@z; B=g@G@g; return V,B,z
    V,B,z=parts(J); lamE=np.linalg.eigvalsh(R@KE@R).max()
    # (a) V <= lamE * B
    (ok if V<=lamE*B*(1+1e-9)+1e-12 else viol)['a']+=1; tight.append(V/(lamE*B+1e-300))
    # (b) unit death: drop random columns -> B nonincreasing
    keep=rng.random(p)>.4; keep[0]=True
    _,Bd,_=parts(J[:,keep]); (ok if Bd<=B*(1+1e-9)+1e-12 else viol)['b']+=1
    # (c) adding column Jv raises B by exactly 2*gain (Schur, r=e)
    Jv=rng.standard_normal((N,rng.integers(1,3)))
    _,B2,_=parts(np.hstack([J,Jv]))
    M=R@J@J.T@R; Hperp=mu*R@np.linalg.inv(M+mu*np.eye(N))@R
    rperp=mu*np.linalg.solve(H@J@J.T+mu*np.eye(N),e)
    s=Jv.T@rperp; S=Jv.T@Hperp@Jv+mu*np.eye(Jv.shape[1]); gain=.5*s@np.linalg.solve(S,s)
    (ok if abs((B2-B)-2*gain)<=1e-8*max(1,B) else viol)['c']+=1
    # (d) NEG: bound with Euclidean ||g||^2/mu instead is not the reachability; check V can be >0 while B small -> V small
    (ok if V<=lamE*B+1e-12 else viol)['d']+=1
print('ok',ok,'viol',viol,'tightness median %.3f max %.3f'%(np.median(tight),max(tight)))
# WIT: bound not monotone for V itself (V can increase when a unit dies)
cnt=0
for t in range(300):
    N,p,q,mu=8,6,3,.1; J=rng.standard_normal((N,p)); H=np.eye(N); e=rng.standard_normal(N); JE=rng.standard_normal((N,q)); KE=JE@JE.T
    f=lambda J:(lambda G,g:(H@J@G@g)@KE@(H@J@G@g))(np.linalg.inv(J.T@J+mu*np.eye(J.shape[1])),J.T@e)
    if f(J[:,1:])>f(J): cnt+=1
print('WIT: V itself increases after a unit dies in',cnt,'/300')
