"""Zero outgoing columns (C4a) and finite candidate-pool gradient ranking (C4b)."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ett.core.stats import measurement
from ett.core.rebuild import replace


def candidates(site,x,y,loss_fn,k,generator,init='gradmax_pool'):
    if init not in ('gradmax_pool','random'):raise ValueError('unknown ETT initializer')
    P=8*k if init=='gradmax_pool' else k
    first,last=site.first,site.last;captured={}
    def incoming(module,args):captured['x']=args[0].detach()
    def outgoing(module,args,output):captured['z']=output
    hooks=[first.register_forward_pre_hook(incoming),last.register_forward_hook(outgoing)]
    try:
        with measurement(site.model),torch.enable_grad():
            loss=loss_fn(site.model(x),y)
            delta=torch.autograd.grad(loss,captured['z'])[0].detach()
    finally:
        for hook in hooks:hook.remove()
    fan=first.weight[0].numel();bound=site.init_bound or 1/math.sqrt(fan)
    shape=(P,*first.weight.shape[1:])
    w=torch.empty(shape,device='cpu',dtype=first.weight.dtype).uniform_(-bound,bound,generator=generator).to(first.weight.device)
    if site.bn is None:w.mul_(bound*math.sqrt(fan/3)/w.flatten(1).norm(dim=1).clamp_min(torch.finfo(w.dtype).tiny).reshape(P,*([1]*(w.ndim-1))))
    bias=w.new_zeros(P)
    if first.bias is not None and site.bias_init=='uniform':bias.uniform_(-bound,bound,generator=generator)
    source=captured['x']
    pre=F.linear(source,w,bias if first.bias is not None else None) if isinstance(first,nn.Linear) else F.conv2d(source,w,bias if first.bias is not None else None,first.stride,first.padding,first.dilation)
    dims=(0,) if pre.ndim==2 else (0,2,3)
    mean=pre.mean(dims);var=pre.var(dims,unbiased=False)
    if site.bn is not None:
        view=(1,P) if pre.ndim==2 else (1,P,1,1)
        normalized=(pre-mean.reshape(view))/torch.sqrt(var.reshape(view)+site.bn.eps)
        feature=site.module(site.activation)(normalized)
    else:feature=site.module(site.activation)(pre)
    if feature.ndim==2:gradient=delta.T@feature
    else:
        patches=F.unfold(feature,last.kernel_size,last.dilation,last.padding,last.stride)
        gradient=torch.einsum('nol,nil->oi',delta.flatten(2),patches)
    energy=gradient.reshape(gradient.shape[0],P,-1).square().sum((0,2))
    indices=torch.argsort(energy,descending=True,stable=True)[:k] if init=='gradmax_pool' else torch.arange(k,device=w.device)
    count=pre.numel()//P
    return dict(weight=w[indices],bias=bias[indices],mean=mean[indices],var=var[indices]*(count/max(count-1,1)),
        gradient_energy=float(energy[indices].sum()),pool_size=P)


def expand(site,optimizer,proposal):
    k=len(proposal['weight']);n=site.width
    if k==0:return n
    first,last=site.first,site.last;indices=list(range(n))+[-1]*k
    replace(optimizer,first,'weight',torch.cat((first.weight,proposal['weight'].to(first.weight)),0),0,indices)
    if first.bias is not None:replace(optimizer,first,'bias',torch.cat((first.bias,proposal['bias'].to(first.bias)),0),0,indices)
    if isinstance(first,nn.Linear):first.out_features=n+k;last.in_features=n+k
    else:first.out_channels=n+k;last.in_channels=n+k
    if site.bn is not None:
        bn=site.bn
        for name,fill in (('weight',1),('bias',0)):
            v=getattr(bn,name);replace(optimizer,bn,name,torch.cat((v,v.new_full((k,),fill)),0),0,indices)
        for name,key in (('running_mean','mean'),('running_var','var')):
            setattr(bn,name,torch.cat((getattr(bn,name),proposal[key].to(getattr(bn,name))),0))
        bn.num_features=n+k
    shape=list(last.weight.shape);shape[1]=k
    replace(optimizer,last,'weight',torch.cat((last.weight,last.weight.new_zeros(shape)),1),1,indices)
    site.validate();return n
