"""C3c–C3e: score candidate versus backward downdates, exact finite-set costs."""
from itertools import combinations
import torch
from ett.core.compensate import columns,g_exact


def backward(G,W,groups,candidates,k):
    M=torch.cholesky_inverse(torch.linalg.cholesky(G));current=W.clone()
    alive=list(range(len(G)));removed=[];cand=list(candidates)
    for _ in range(k):
        pos={c:j for j,c in enumerate(alive)}
        def cost(unit):
            I=[pos[c] for c in groups[unit]]
            return float((current[I]*torch.linalg.solve(M[I][:,I],current[I])).sum())
        unit=min(cand,key=lambda u:(cost(u),u));I=[pos[c] for c in groups[unit]]
        R=[j for j in range(len(alive)) if j not in I]
        solved=torch.linalg.solve(M[I][:,I],torch.cat((current[I],M[I][:,R]),1))
        current=current[R]-M[R][:,I]@solved[:,:current.shape[1]]
        M=M[R][:,R]-M[R][:,I]@solved[:,current.shape[1]:]
        alive=[alive[j] for j in R];cand.remove(unit);removed.append(unit)
    return removed,current,alive


def select(G,G_reg,W,width,k,original,ages,maturity,cycle=0,policy='min',group_size=1):
    r=group_size
    groups={u:list(range(u*r,(u+1)*r)) for u in range(width)}
    cand=[u for u in range(width) if u>=original or ages[u]>=maturity]
    if len(cand)<k:raise ValueError('insufficient unprotected units')
    energies=[float(G[groups[u]][:,groups[u]].trace()) for u in range(width)]
    threshold=1e-12*sum(energies)/width
    dead=[u for u in cand if energies[u]==0 or energies[u]<threshold]
    dead.sort(key=lambda u:(u>=original,u))
    forced=dead[:k];remaining=k-len(forced)
    candidates=[u for u in cand if u not in forced]
    M=torch.cholesky_inverse(torch.linalg.cholesky(G_reg))
    def score(u):
        I=groups[u];return float((W[I]*(G[I][:,I]@W[I])).sum())
    D_score=forced+sorted(candidates,key=lambda u:(score(u),u))[:remaining]
    if remaining:
        if forced:
            removed,Wc,alive=backward_forced(G_reg,W,groups,forced)
            reduced_groups={u:[alive.index(c) for c in groups[u]] for u in candidates}
            reduced=G_reg[alive][:,alive]
            other,_,_=backward(reduced,Wc,reduced_groups,candidates,remaining)
            D_back=forced+other
        else:D_back=backward(G_reg,W,groups,candidates,k)[0]
    else:D_back=forced
    gs=g_exact(G_reg,W,columns(D_score,r),M);gb=g_exact(G_reg,W,columns(D_back,r),M)
    chosen=D_score if (policy=='score' or policy=='min' and gs<=gb) else D_back
    result=dict(dropped=chosen,g_score=gs,g_back=gb,g=min(gs,gb) if policy=='min' else gs if policy=='score' else gb,
        dead_units=len(dead),virtual_kept=k-sum(u>=original for u in chosen))
    if width<=20 and k<=3 and cycle%10==0:
        opt=min(g_exact(G_reg,W,columns(D,r),M) for D in combinations(cand,k))
        result.update(opt=opt,ratio=result['g']/opt if opt>0 else None)
    return result


def backward_forced(G,W,groups,forced):
    """Condition on the mandatory dead removals before the remaining backward search."""
    from ett.core.compensate import compensate
    drop=[c for u in forced for c in groups[u]];alive=[c for c in range(len(G)) if c not in drop]
    return forced,compensate(G,W,alive),alive
