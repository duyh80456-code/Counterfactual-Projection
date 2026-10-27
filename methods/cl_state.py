"""Transactional real updates and BN/RNG-safe counterfactual evaluation."""
from contextlib import contextmanager
from copy import deepcopy
import torch
from torch.func import functional_call
from experiments.vna_resume import rng_state,restore_rng


class State:
    def __init__(self,model,optimizer=None):
        self.model,self.optimizer=model,optimizer
        self.values=deepcopy(model.state_dict());self.modes={n:m.training for n,m in model.named_modules()}
        self.buffers={n:b.detach().clone() for n,b in model.named_buffers()}
        self.requires_grad={n:p.requires_grad for n,p in model.named_parameters()}
        self.registrations=[(m,dict(m._parameters),dict(m._buffers),dict(m._modules),set(m._non_persistent_buffers_set)) for m in model.modules()]
        self.gradients={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
        self.opt=deepcopy(optimizer.state_dict()) if optimizer is not None else None
        self.rng=rng_state()

    def restore(self):
        for module,params,buffers,children,nonpersistent in self.registrations:
            module._parameters.clear();module._parameters.update(params)
            module._buffers.clear();module._buffers.update(buffers)
            module._modules.clear();module._modules.update(children)
            module._non_persistent_buffers_set.clear();module._non_persistent_buffers_set.update(nonpersistent)
        with torch.no_grad():
            for n,p in self.model.named_parameters():
                ref=self.values[n]
                if p.shape!=ref.shape or p.dtype!=ref.dtype or p.device!=ref.device:p.data=ref.clone()
                p.requires_grad_(self.requires_grad[n])
            for n,b in self.model.named_buffers():
                ref=self.buffers[n]
                if b.shape!=ref.shape or b.dtype!=ref.dtype or b.device!=ref.device:b.data=ref.clone()
        self.model.load_state_dict(self.values)
        with torch.no_grad():
            for n,b in self.model.named_buffers():b.copy_(self.buffers[n])
        for n,m in self.model.named_modules():m.training=self.modes[n]
        for n,p in self.model.named_parameters():p.grad=None if self.gradients[n] is None else self.gradients[n].clone()
        if self.optimizer is not None:self.optimizer.load_state_dict(self.opt)
        restore_rng(self.rng)


@contextmanager
def isolated(model,optimizer=None):
    state=State(model,optimizer)
    try:yield state
    finally:state.restore()


def evaluate(model,inputs,components,mode='train'):
    """Fresh functional BN buffers and identical RNG on every trial."""
    if mode not in ('train','eval','frozen_bn'):raise ValueError(mode)
    with isolated(model),torch.no_grad():
        model.train(mode!='eval')
        if mode=='frozen_bn':
            for module in model.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        logits=functional_call(model,{n:b.clone() for n,b in model.named_buffers()},(inputs,))
        if isinstance(logits,dict):logits=logits['logits']
        return tuple(float(component(logits)) for component in components)


def predict(model,inputs,mode='eval'):
    with isolated(model),torch.no_grad():
        model.train(mode=='train')
        output=functional_call(model,{n:b.clone() for n,b in model.named_buffers()},(inputs,))
        return (output['logits'] if isinstance(output,dict) else output).detach().clone()


def parameters(model):return tuple((n,p) for n,p in model.named_parameters() if p.requires_grad)


def vector(model):return torch.cat([p.detach().flatten() for _,p in parameters(model)]).double()


def subtract(model,direction,step):
    refs=parameters(model)
    if direction.numel()!=sum(p.numel() for _,p in refs):raise ValueError('parameter-vector shape mismatch')
    with torch.no_grad():
        for (_,p),delta in zip(refs,direction.split([p.numel() for _,p in refs])):p.add_(delta.reshape_as(p).to(p),alpha=-step)
