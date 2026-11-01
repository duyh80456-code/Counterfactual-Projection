"""Explicit single-consumer unit sites (A1/A2)."""
from dataclasses import dataclass
import warnings
import torch
from torch import nn


@dataclass
class LinearUnitLayer:
    model: nn.Module
    incoming: str
    outgoing: str
    activation: str
    norm: str | None = None
    next_norm: str | None = None
    init_bound: float | None = None
    bias_init: str = 'zero'

    def __post_init__(self):
        assert isinstance(self.first, nn.Linear) and isinstance(self.last, nn.Linear), 'A2: linear single-consumer pair required'
        self.validate()

    def module(self, name):return self.model.get_submodule(name)
    @property
    def first(self):return self.module(self.incoming)
    @property
    def last(self):return self.module(self.outgoing)
    @property
    def bn(self):return self.module(self.norm) if self.norm else None
    @property
    def width(self):return self.first.weight.shape[0]
    @property
    def group_size(self):return 1
    @property
    def intercept(self):return self.last.bias is not None or self.next_norm is not None

    def validate(self):
        norm=self.bn
        if norm is not None and not isinstance(norm, (nn.BatchNorm1d, nn.BatchNorm2d)):
            raise ValueError('A1: only channelwise BatchNorm is eligible')
        assert self.first.weight.shape[0]==self.last.weight.shape[1], 'A2: mismatched channel dimensions'
        if norm is not None:assert norm.num_features==self.width and norm.affine and norm.track_running_stats
        assert isinstance(self.module(self.activation), nn.ReLU), 'only declared ReLU sites are supported'
        if self.next_norm:assert isinstance(self.module(self.next_norm), (nn.BatchNorm1d,nn.BatchNorm2d))

    def set_module(self, name, value):
        parent,_,key=name.rpartition('.')
        setattr(self.module(parent) if parent else self.model,key,value)

    def weights(self):
        w=self.last.weight.detach().T if self.group_size==1 else self.last.weight.detach().permute(1,2,3,0).reshape(-1,self.last.out_channels)
        if self.intercept:
            bias=self.last.bias.detach() if self.last.bias is not None else w.new_zeros(w.shape[1])
            w=torch.cat((w,bias[None]),0)
        return w.double()


@dataclass
class ConvUnitLayer(LinearUnitLayer):
    def __post_init__(self):
        assert isinstance(self.first,nn.Conv2d) and isinstance(self.last,nn.Conv2d), 'A2: convolution pair required'
        assert self.first.groups==self.last.groups==1 and self.last.kernel_size==(3,3)
        assert self.first.padding_mode==self.last.padding_mode=='zeros'
        self.validate()
    @property
    def group_size(self):return 9


def discover(model):
    """Only explicit sequential chains and basic-block conv1; never residual outputs."""
    sites=[];skipped=[]
    for path,parent in model.named_modules():
        prefix=path+'.' if path else ''
        if isinstance(parent,(nn.Sequential,nn.ModuleList)):
            children=list(parent.named_children())
            for i,(name,first) in enumerate(children):
                if not isinstance(first,(nn.Linear,nn.Conv2d)):continue
                between=[];last=None
                for key,child in children[i+1:]:
                    if isinstance(child,type(first)):last=key;break
                    between.append((key,child))
                    if not isinstance(child,(nn.ReLU,nn.BatchNorm1d,nn.BatchNorm2d,nn.LayerNorm,nn.GroupNorm)):break
                if last is None:continue
                if any(isinstance(m,(nn.LayerNorm,nn.GroupNorm)) for _,m in between):
                    skipped.append(dict(site=prefix+name,reason='A1: LayerNorm/GroupNorm'));continue
                acts=[key for key,m in between if isinstance(m,nn.ReLU)]
                norms=[key for key,m in between if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d))]
                if len(acts)!=1 or len(norms)>1:continue
                cls=LinearUnitLayer if isinstance(first,nn.Linear) else ConvUnitLayer
                sites.append(cls(model,prefix+name,prefix+last,prefix+acts[0],prefix+norms[0] if norms else None))
        elif all(hasattr(parent,k) for k in ('conv1','bn1','relu','conv2','bn2')):
            if isinstance(parent.bn1,(nn.GroupNorm,nn.LayerNorm)):
                skipped.append(dict(site=prefix+'conv1',reason='A1: LayerNorm/GroupNorm'));continue
            sites.append(ConvUnitLayer(model,prefix+'conv1',prefix+'conv2',prefix+'relu',prefix+'bn1',prefix+'bn2'))
    for item in skipped:warnings.warn(str(item),stacklevel=2)
    return sites,skipped
