"""Unmodified official LPR callbacks through a PyCIL interface adapter."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[1]/'third_party/LPR'
PIN='76c44f8f2481f4e23881b2463eb58aaf94affabb'
SOURCE_SHA256='07d0aef6077bac8b77f4e52194b79e155696da956a14fa9cc55381d9ac28e7d6'


def official_module(root=ROOT):
    path=Path(root)/'src/strategies/lpr_plugin.py'
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=PIN:
        raise ValueError('official LPR commit mismatch')
    source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=SOURCE_SHA256:raise ValueError('official LPR source has changed')
    name='_standard_cl_official_lpr'
    if name in sys.modules:return sys.modules[name]
    tree=ast.parse(source,filename=str(path))
    # Only the unused framework base and disabled external logger imports are adapted.
    tree.body=[node for node in tree.body if not (isinstance(node,ast.ImportFrom) and node.module=='avalanche.training.plugins.strategy_plugin')
               and not (isinstance(node,ast.Import) and any(a.name=='wandb' for a in node.names))]
    module=types.ModuleType(name);module.SupervisedPlugin=object;sys.modules[name]=module
    try:exec(compile(tree,str(path),'exec'),module.__dict__)
    except BaseException:sys.modules.pop(name,None);raise
    return module


class ReplayPairs(Dataset):
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return 0 if self.dataset is None else len(self.dataset)
    def __getitem__(self,index):
        _,x,y=self.dataset[index];return x,y


class LinearHead(torch.nn.Linear):
    def __init__(self,head):
        torch.nn.Module.__init__(self)
        self.in_features,self.out_features=head.in_features,head.out_features
        self.weight,self.bias=head.weight,head.bias
    def forward(self,x):return {'logits':torch.nn.functional.linear(x,self.weight,self.bias)}


def prepare_head(model):
    # Same F.linear and Parameter objects; expose PyCIL's head to official isinstance checks.
    if not isinstance(model.fc,torch.nn.Linear):model.fc=LinearHead(model.fc)


class OfficialLPR:
    def __init__(self,config,device,state=None):
        if config['log']['grad_norms'] is not None:raise ValueError('external wandb logging is disabled by the adapter')
        self.module=official_module();self.device=device
        kwargs={k:SimpleNamespace(**deepcopy(v)) for k,v in config.items()}
        self.plugin=self.module.LPRPlugin(kwargs,SimpleNamespace(buffer=ReplayPairs(None)))
        if state:
            self.plugin.update_config.every_iter=state['every_iter']
            self.plugin._preconditioner_invs={name:self.module.PreconditionerInv(matrix=v['matrix'].to(device),has_bias=v['has_bias'],parallel=v['parallel'])
                                            for name,v in state['inverses'].items()}

    def strategy(self,model,dataset,step,task,task_step,epochs):
        self.plugin.storage_policy.buffer=ReplayPairs(dataset)
        return SimpleNamespace(model=model,device=self.device,is_training=True,train_epochs=epochs,
            clock=SimpleNamespace(train_iterations=step,train_exp_counter=task,train_exp_iterations=task_step))

    def state_dict(self):
        return dict(every_iter=self.plugin.update_config.every_iter,
            inverses={name:dict(matrix=v.matrix.detach().clone(),has_bias=v.has_bias,parallel=v.parallel)
                      for name,v in self.plugin._preconditioner_invs.items()})
