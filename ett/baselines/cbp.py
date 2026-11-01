"""Pinned upstream Generate-and-Test, including its original utility and reset logic."""
from copy import deepcopy
from pathlib import Path
import sys
import torch
from ett.core.expand import candidates
from ett.core.stats import measurement


def make(model,opt,cfg):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'third_party/loss-of-plasticity'))
    from lop.algos.gnt import GnT
    return GnT(model,hidden_activation='relu',opt=opt,decay_rate=.99,
        replacement_rate=cfg['rho'],maturity_threshold=cfg['cbp_maturity'],util_type='adaptable_contribution')


def observe(gnt,features):
    # Track the upstream EMA on the BP trajectory without performing resets.
    for i,h in enumerate(features):
        gnt.ages[i]+=1;gnt.update_utility(i,h)


def capture_features(model,x):
    values=[];out=x
    for i,layer in enumerate(model):
        out=layer(out)
        if isinstance(layer,torch.nn.ReLU):values.append(out.detach())
    return out,values


def single_event(gnt,sites,batch,loss_fn,generator,gradmax=False):
    with measurement(sites[0].model),torch.no_grad():_,features=capture_features(sites[0].model,batch[0])
    chosen,counts=gnt.test_features(features)
    proposals={}
    if gradmax:
        for site,units in zip(sites,chosen):
            if len(units):proposals[site.incoming]=candidates(site,*batch,loss_fn,len(units),generator)
    gnt.gen_new_features(chosen,counts);gnt.update_optim_params(chosen,counts)
    if gradmax:
        with torch.no_grad():
            for site,units in zip(sites,chosen):
                if not len(units):continue
                p=proposals[site.incoming];site.first.weight[units]=p['weight'];site.first.bias[units]=p['bias']
    return [len(c) for c in chosen]


def history(gnt):
    names=('util','bias_corrected_util','ages','mean_feature_act','accumulated_num_features_to_replace')
    return {name:deepcopy(getattr(gnt,name)) for name in names}


def restore(gnt,state):
    for key,value in state.items():setattr(gnt,key,deepcopy(value))
