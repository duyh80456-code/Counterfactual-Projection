"""Domain-IL data, replay and fixed-head controls, independent of CIL expansion."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import Dataset
from methods.cl_replay import ReplayLoss
from methods.vna import dedicated_generator
from methods.vna_sites import HookSite


class DomainMNIST(Dataset):
    def __init__(self,base,domain,kind,seed=0,angle_step=15.):
        if kind not in ('rotated','permuted'):raise ValueError(kind)
        self.base,self.domain,self.kind,self.angle_step=base,domain,kind,angle_step
        self.permutation=torch.randperm(784,generator=dedicated_generator(seed,f'mnist-domain:{domain}'))
    def identity(self):
        import hashlib
        return dict(domain=self.domain,kind=self.kind,angle_degrees=self.domain*self.angle_step if self.kind=='rotated' else None,
            permutation_sha256=hashlib.sha256(self.permutation.numpy().tobytes()).hexdigest() if self.kind=='permuted' else None,
            label_identity=True,domain_id_at_inference=False,rotation_interpolation='nearest',fill=0)
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        from torchvision.transforms import functional as TF
        image,label=self.base[index]
        if not torch.is_tensor(image):image=TF.to_tensor(image)
        if self.kind=='rotated':image=TF.rotate(image,self.domain*self.angle_step)
        else:image=image.flatten()[self.permutation].reshape(1,28,28)
        return image,label


class DomainNetwork(nn.Module):
    def __init__(self,normalization='bn',width=256):
        super().__init__()
        norm=nn.BatchNorm1d(width) if normalization=='bn' else nn.GroupNorm(16,width)
        if normalization not in ('bn','groupnorm'):raise ValueError(normalization)
        self.flatten=nn.Flatten();self.hidden=nn.Linear(784,width);self.norm=norm
        self.relu=nn.ReLU();self.head=nn.Linear(width,10)
    def forward(self,x):return self.head(self.relu(self.norm(self.hidden(self.flatten(x)))))
    def sites(self):
        if isinstance(self.norm,nn.GroupNorm):return []
        return [HookSite(self,'hidden','head','linear',between=('norm','relu'))]


class Reservoir:
    def __init__(self,capacity,seed):
        if capacity<0:raise ValueError('nonnegative capacity required')
        self.capacity=capacity;self.items=[];self.seen=0
        self.generator=dedicated_generator(seed,'domain-reservoir')
    def add(self,inputs,labels,domain):
        for image,label in zip(inputs.cpu(),labels.cpu()):
            self.seen+=1
            if len(self.items)<self.capacity:self.items.append((image.clone(),int(label),domain))
            else:
                index=int(torch.randint(self.seen,(1,),generator=self.generator))
                if index<self.capacity:self.items[index]=(image.clone(),int(label),domain)
    def sample(self,count,device):
        if not self.items:return None
        indices=torch.randperm(len(self.items),generator=self.generator)[:count].tolist()
        images,labels,domains=zip(*(self.items[i] for i in indices))
        return torch.stack(images).to(device),torch.tensor(labels,device=device),domains
    def state_dict(self):return dict(capacity=self.capacity,items=self.items,seen=self.seen,generator=self.generator.get_state())
    def load_state_dict(self,state):
        if state['capacity']!=self.capacity:raise ValueError('reservoir capacity mismatch')
        self.items,self.seen=state['items'],state['seen'];self.generator.set_state(state['generator'])


def domain_losses(targets,replay_mask,*,current_domain=None,replay_domains=None):
    if current_domain is not None:
        if replay_domains is None or len(replay_domains)!=int(replay_mask.sum()):raise ValueError('aligned replay domain ids required')
        replay_mask=replay_mask.clone()
        replay_mask[replay_mask.clone()]=torch.as_tensor(replay_domains,device=targets.device)!=current_domain
    result=ReplayLoss(targets,replay_mask,kind='er')
    if current_domain is not None:
        result.incoming.name='current_domain_ce';result.reference_loss.name='previous_domains_ce'
    return result
