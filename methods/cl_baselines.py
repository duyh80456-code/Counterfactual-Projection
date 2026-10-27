"""A-GEM/ER-ACE update laws, explicitly adapted to task-boundary PyCIL herding."""
import torch
import torch.nn.functional as F


def backward(model,optimizer,x,y,replay,method,seen,task):
    if method not in ('agem','er_ace'):raise ValueError(method)
    optimizer.zero_grad();logits=model(x)['logits']
    seen=torch.unique(torch.cat((seen.to(y.device),y.unique())))
    selected=logits
    if method=='er_ace' and task>0:
        allowed=torch.zeros(logits.shape[1],dtype=torch.bool,device=logits.device)
        allowed[y.unique()]=True;allowed[int(seen.max()):]=True
        selected=logits.masked_fill(~allowed,-1e9)
    loss=F.cross_entropy(selected,y)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite incoming loss')
    loss.backward()
    reference_loss=0.;projected=False;dot=None
    if replay is not None:
        parameters=[p for p in model.parameters() if p.requires_grad]
        incoming=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]
        optimizer.zero_grad();rx,ry=replay
        reference=F.cross_entropy(model(rx)['logits'],ry)
        if not bool(torch.isfinite(reference)):raise FloatingPointError('nonfinite replay loss')
        reference.backward();reference_loss=float(reference.detach())
        old=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]
        dot=sum((a.double()*b.double()).sum() for a,b in zip(incoming,old))
        square=sum(b.double().square().sum() for b in old)
        projected=method=='agem' and float(dot)<0 and float(square)>0
        coefficient=float(dot/square) if projected else 0.
        for p,a,b in zip(parameters,incoming,old):p.grad=a-coefficient*b if method=='agem' else a+b
    reported=loss.detach()+reference_loss if method=='er_ace' else loss.detach()
    return logits,reported,seen,dict(method=method,reference_loss=reference_loss,projected=projected,
        incoming_reference_dot=None if dot is None else float(dot),
        bn_behavior='sequential native train forwards, each updates running buffers',
        certificate_scope='A-GEM gradient only; momentum and weight decay are native SGD')
