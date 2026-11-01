"""Online full-permutation MNIST; no class/task labels passed to the learner."""
from pathlib import Path
import hashlib
import torch


class PMNIST:
    def __init__(self,cfg,root,download=False):
        from torchvision.datasets import MNIST
        dataset=MNIST(root,train=True,download=download)
        self.x=dataset.data.flatten(1).float()/255;self.y=dataset.targets
        gen=torch.Generator().manual_seed(cfg['data_seed'])
        self.permutations=[torch.randperm(784,generator=gen) for _ in range(cfg['tasks'])]
        self.orders=[torch.randperm(len(self.x),generator=gen) for _ in range(cfg['tasks'])]
        self.task_steps=len(self.x)
        self.fingerprint=hashlib.sha256(dataset.data.numpy().tobytes()+dataset.targets.numpy().tobytes()).hexdigest()
    def batch(self,index):
        task,pos=divmod(index,self.task_steps);row=self.orders[task][pos]
        return self.x[row:row+1][:,self.permutations[task]],self.y[row:row+1]
