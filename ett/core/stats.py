"""Fixed-network float64 Grams; no running-stat or RNG side effects."""
from contextlib import contextmanager
import torch
from torch.nn import functional as F


@contextmanager
def measurement(model,training=True):
    modes={m:m.training for m in model.modules()}
    buffers={n:b.detach().clone() for n,b in model.named_buffers()}
    rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        model.train(training);yield
    finally:
        with torch.no_grad():
            for n,b in model.named_buffers():b.copy_(buffers[n])
        for m,mode in modes.items():m.training=mode
        torch.set_rng_state(rng)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)


def design(site,features):
    if site.group_size==1:A=features.reshape(-1,features.shape[-1])
    else:
        last=site.last
        A=F.unfold(features,last.kernel_size,dilation=last.dilation,padding=last.padding,stride=last.stride).transpose(1,2).reshape(-1,site.width*9)
    A=A.double()
    if site.intercept:A=torch.cat((A,A.new_ones(len(A),1)),1)
    return A


def gram(site,batches):
    total=None
    def capture(module,args):
        nonlocal total
        A=design(site,args[0]);part=A.T@A
        total=part if total is None else total+part
    hook=site.last.register_forward_pre_hook(capture)
    try:
        with measurement(site.model),torch.no_grad():
            for x,y in batches:site.model(x)
    finally:hook.remove()
    if total is None:raise ValueError('empty statistics window')
    return total


def regularize(G,unit_columns,epsilon=1e-6):
    """At most three ridge increases; an all-zero feature block has no valid scale."""
    scale=G.diag()[:unit_columns].sum()/unit_columns
    attempts=[]
    for retry in range(4):
        eps=epsilon*10**retry;matrix=G.clone()
        matrix.diagonal()[:unit_columns].add_(eps*scale)
        L,info=torch.linalg.cholesky_ex(matrix)
        attempts.append(dict(epsilon=eps,cholesky_info=int(info)))
        if int(info)==0:return matrix,L,attempts
    return None,None,attempts
