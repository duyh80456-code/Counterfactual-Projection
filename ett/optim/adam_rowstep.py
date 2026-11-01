"""Adam with coordinate-shaped clocks; row/column births retain their own age."""
import torch
from torch.optim import Optimizer


class AdamRowStep(Optimizer):
    def __init__(self,params,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.):
        if lr<0 or eps<0 or not all(0<=b<1 for b in betas):raise ValueError('invalid Adam controls')
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        for group in self.param_groups:
            b1,b2=group['betas']
            for p in group['params']:
                if p.grad is None:continue
                if p.grad.is_sparse:raise ValueError('sparse Adam is unsupported')
                state=self.state[p]
                if not state:
                    state.update(step=torch.zeros_like(p,dtype=torch.int64),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
                g=p.grad.add(p,alpha=group['weight_decay']) if group['weight_decay'] else p.grad
                state['step'].add_(1);state['exp_avg'].lerp_(g,1-b1)
                state['exp_avg_sq'].mul_(b2).addcmul_(g,g,value=1-b2)
                age=state['step'].to(p.dtype)
                first=state['exp_avg']/(1-b1**age)
                second=state['exp_avg_sq']/(1-b2**age)
                p.addcdiv_(first,second.sqrt().add_(group['eps']),value=-group['lr'])
        return loss
