"""F1/F2: only input/label samples from completed training steps, never features."""
from collections import deque
from copy import deepcopy
import torch


class SampleBuffer:
    def __init__(self,capacity):
        if capacity<5:raise ValueError('buffer needs a nonempty 80/20 split')
        self.capacity=capacity;self.samples=deque(maxlen=capacity);self.seen=0

    def __len__(self):return len(self.samples)

    def append(self,batch):
        x,y=batch
        if len(x)!=len(y):raise ValueError('unaligned samples/labels')
        for a,b in zip(x,y):
            self.samples.append((a.detach().cpu().clone(),b.detach().cpu().clone(),self.seen));self.seen+=1

    def split(self,device=None):
        if len(self)!=self.capacity:raise ValueError('wait for a full trained-sample buffer')
        cut=4*len(self)//5;items=list(self.samples)
        def pack(rows):
            return (torch.stack([r[0] for r in rows]).to(device),torch.stack([r[1] for r in rows]).to(device)),[r[2] for r in rows]
        g,gids=pack(items[:cut]);check,cids=pack(items[cut:])
        return g,check,dict(buffer_samples=len(self),gram_samples=cut,check_samples=len(self)-cut,
            gram_ids=gids,check_ids=cids,trained_samples_seen=self.seen)

    def state_dict(self):return deepcopy(dict(capacity=self.capacity,samples=list(self.samples),seen=self.seen))

    def load_state_dict(self,state):
        if state['capacity']!=self.capacity:raise ValueError('buffer capacity resume mismatch')
        self.samples=deque(deepcopy(state['samples']),maxlen=self.capacity);self.seen=state['seen']
