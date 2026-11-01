"""Shape-changing parameters and optimizer state; no permanent masked slots."""
from copy import deepcopy
import torch
from torch import nn
from ett.optim.adam_rowstep import AdamRowStep


def replace(optimizer,module,name,value,axis,indices):
    old=getattr(module,name)
    if old is None:return
    new=nn.Parameter(value.detach().to(old).clone(),requires_grad=old.requires_grad)
    state=optimizer.state.pop(old,{})
    mapped={}
    for key,v in state.items():
        if isinstance(v,torch.Tensor) and v.shape==old.shape:
            target=v.new_zeros(new.shape)
            valid=[(j,int(i)) for j,i in enumerate(indices) if i>=0]
            if valid:
                dest=torch.tensor([j for j,_ in valid],device=v.device)
                src=torch.tensor([i for _,i in valid],device=v.device)
                target.index_copy_(axis,dest,v.index_select(axis,src))
            mapped[key]=target
        elif key=='step' and isinstance(optimizer,torch.optim.Adam):
            raise ValueError('ETT requires AdamRowStep, not a tensor-wide Adam clock')
        else:mapped[key]=deepcopy(v)
    for group in optimizer.param_groups:
        group['params']=[new if p is old else p for p in group['params']]
    setattr(module,name,new)
    if mapped:optimizer.state[new]=mapped


def rebuild(site,optimizer,indices,outgoing=None):
    """Lemma0: retained real slots first; virtual units fill vacated slots."""
    idx=torch.as_tensor(indices,device=site.first.weight.device,dtype=torch.long)
    first,last=site.first,site.last
    replace(optimizer,first,'weight',first.weight.index_select(0,idx),0,indices)
    if first.bias is not None:replace(optimizer,first,'bias',first.bias.index_select(0,idx),0,indices)
    if isinstance(first,nn.Linear):first.out_features=len(indices);last.in_features=len(indices)
    else:first.out_channels=len(indices);last.in_channels=len(indices)
    bn=site.bn
    if bn is not None:
        for name in ('weight','bias'):replace(optimizer,bn,name,getattr(bn,name).index_select(0,idx),0,indices)
        for name in ('running_mean','running_var'):setattr(bn,name,getattr(bn,name).index_select(0,idx).clone())
        bn.num_features=len(indices)
    value=last.weight.index_select(1,idx)
    if outgoing is not None:
        unit=outgoing[:len(indices)*site.group_size]
        value=unit.T.reshape_as(value)
    replace(optimizer,last,'weight',value,1,indices)
    if outgoing is not None and site.intercept:
        intercept=outgoing[-1].to(last.weight)
        if last.bias is not None:
            with torch.no_grad():last.bias.copy_(intercept)
        elif site.next_norm:
            # C3a: omitted additive intercept needs the opposite running-mean shift.
            with torch.no_grad():site.module(site.next_norm).running_mean.sub_(intercept)
    site.validate()


def slot_order(original_width,kept):
    kept=set(kept);virtual=iter(sorted(i for i in kept if i>=original_width))
    return [i if i in kept else next(virtual) for i in range(original_width)]
