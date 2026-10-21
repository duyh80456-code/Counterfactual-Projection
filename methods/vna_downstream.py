"""Sample-block downstream products at a frozen, sample-separable noBN site."""
import time
import torch
from torch import nn
from torch.func import functional_call, vjp
from diagnostics.vna_align import _global_state
from projection.functional import _preserve_functional_state


def separable(model):
    return not any(isinstance(m,(nn.modules.batchnorm._BatchNorm,nn.modules.dropout._DropoutNd))
                   for m in model.modules())


class Downstream:
    block_samples = 8

    def __init__(self, model, site, inputs):
        if not separable(model): raise ValueError('BN/stochastic model requires full products')
        self.model,self.site,self.inputs=model,site,inputs
        self.seconds,self.bytes,self.vjp_calls,self.blocks = 0.,0,0,0
        self._check=self._block(inputs,check=True)
        self.classes=self._check[0].shape[1]
        self.dtype=inputs.dtype

    def _block(self,inputs,check=False):
        if inputs.is_cuda:torch.cuda.synchronize(inputs.device)
        start=time.perf_counter()
        model,site=self.model,self.site
        first,second=model.get_submodule(site.first),model.get_submodule(site.second)
        captured={}
        params={n:p.detach() for n,p in model.named_parameters()}
        buffers={n:b.detach().clone() for n,b in model.named_buffers()}
        def forward():
            value=functional_call(model,{**params,**{n:b.clone() for n,b in buffers.items()}},(inputs,))
            if isinstance(value,dict):value=value['logits']
            return value.reshape(len(inputs),-1)
        modes={m:m.training for m in model.modules()}
        with _global_state(),_preserve_functional_state(model),torch.enable_grad():
            try:
                model.train()
                h1=first.register_forward_pre_hook(lambda _m,args:captured.__setitem__('x',args[0].detach().clone()))
                h2=second.register_forward_hook(lambda _m,_args,out:captured.__setitem__('out',out.detach().clone()))
                try:
                    with torch.no_grad():logits=forward()
                finally:h1.remove();h2.remove()
                def suffix(value):
                    handle=second.register_forward_hook(lambda _m,_args,_out:value.clone())
                    try:return forward()
                    finally:handle.remove()
                _,pullback=vjp(suffix,captured['out'])
                n,c=logits.shape
                if check:
                    generator=torch.Generator(device=inputs.device).manual_seed(20261022)
                    seed=torch.randn(logits.shape,device=inputs.device,dtype=inputs.dtype,generator=generator)
                    result=seed,pullback(seed)[0].detach()
                    self.vjp_calls+=1
                else:
                    D=torch.empty((n,c,*captured['out'].shape[1:]),device=inputs.device,dtype=inputs.dtype)
                    for j in range(c):
                        seed=torch.zeros_like(logits);seed[:,j]=1.
                        D[:,j]=pullback(seed)[0].detach()
                    self.vjp_calls+=c
                    self.bytes=max(self.bytes,D.numel()*D.element_size())
                    self.blocks+=1
                    result=D,captured['x']
            finally:
                for m,mode in modes.items():m.training=mode
        if inputs.is_cuda:torch.cuda.synchronize(inputs.device)
        self.seconds+=time.perf_counter()-start
        return result

    def _blocks(self):
        for lo in range(0,len(self.inputs),self.block_samples):
            hi=min(lo+self.block_samples,len(self.inputs))
            D,x=self._block(self.inputs[lo:hi])
            if self._check is not None:
                seed,actual=self._check
                predicted=torch.einsum('nc,nc...->n...',seed[lo:hi],D)
                if not torch.allclose(actual[lo:hi],predicted,rtol=5e-10,atol=1e-12):
                    raise ValueError('cross-sample downstream coupling; use full products')
            yield lo,hi,D,x
            del D,x
        self._check=None

    def features(self,x,A):
        first=self.model.get_submodule(self.site.first)
        weight=A['weight'].to(x)
        bias=None if A.get('bias') is None else A['bias'].to(x)
        with torch.no_grad():
            if self.site.kind=='linear':h=nn.functional.linear(x,weight,bias)
            else:h=nn.functional.conv2d(x,weight,bias,first.stride,first.padding,first.dilation)
            for path in self.site.between:h=self.model.get_submodule(path)(h)
        return h

    def patches(self,h):
        second=self.model.get_submodule(self.site.second)
        if self.site.kind=='linear':return h.reshape(len(h),1,1)
        return nn.functional.unfold(h,second.kernel_size,dilation=second.dilation,
                                    padding=second.padding,stride=second.stride)

    def cotangent(self,D,residual):
        if self.site.kind=='linear':return torch.einsum('nc,nco->no',residual,D)
        return torch.einsum('nc,ncop->nop',residual,D.flatten(3))

    def gradient(self,patches,cotangent):
        if self.site.kind=='linear':return torch.einsum('nk,no->ok',patches[:,:,0],cotangent).flatten()
        return torch.einsum('nkp,nop->ok',patches,cotangent).flatten()

    def jacobian(self,D,patches):
        if self.site.kind=='linear':J=torch.einsum('nco,nk->ncok',D,patches[:,:,0])
        else:J=torch.einsum('ncop,nkp->ncok',D.flatten(3),patches)
        return J.reshape(D.shape[0]*D.shape[1],-1)

    def score(self,proposals,residual,metric):
        n,c=len(self.inputs),self.classes
        basis=torch.eye(c,device=self.inputs.device,dtype=self.inputs.dtype).repeat(1,n)
        # The frozen loss metric is block diagonal in samples; keep its full-window normalization.
        H=torch.vmap(metric)(basis).reshape(c,n,c).permute(1,2,0)
        residual=residual.reshape(n,c)
        totals=[None]*len(proposals)
        for lo,hi,D,x in self._blocks():
            cotangent=self.cotangent(D,residual[lo:hi])
            for i,p in enumerate(proposals):
                h=self.features(x,p['A']);patches=self.patches(h)
                s=self.gradient(patches,cotangent)
                J=self.jacobian(D,patches).reshape(hi-lo,c,-1)
                K=torch.einsum('nci,ncd,ndj->ij',J,H[lo:hi],J)
                active=h.ne(0).double().sum();count=h.numel()
                if totals[i] is None:totals[i]=[s,K,active,count]
                else:
                    totals[i][0]+=s;totals[i][1]+=K;totals[i][2]+=active;totals[i][3]+=count
            del D,x,J,patches,h,cotangent
        return totals

    def columns(self,A):
        result=None
        for lo,hi,D,x in self._blocks():
            J=self.jacobian(D,self.patches(self.features(x,A)))
            if result is None:result=J.new_empty((len(self.inputs)*self.classes,J.shape[1]))
            result[lo*self.classes:hi*self.classes]=J
            del D,x,J
        return result

    def measure(self):
        for _,_,D,x in self._blocks():del D,x
