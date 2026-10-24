"""Pilot GPU gate: native noBN ResNet32 and a real first-task CIFAR-100 batch."""
from copy import deepcopy
import os

import pytest
import torch
from torch import nn

from adapters.icarl_hooks import Prediction, sites_for
from adapters.cil_setup import class_order, resnet32_nobn, SkipScale
from experiments.runners.pycil import upstream_imports
from methods import vna_certswap as certswap
from methods.vna_losses import ICaRLPyCIL

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA GPU required')


@pytest.fixture
def native_window():
    upstream_imports()
    from utils.inc_net import IncrementalNet
    from torchvision.datasets import CIFAR100
    from torchvision import transforms
    root=os.environ.get('VNA_CERTSWAP_CIFAR_ROOT')
    if not root:pytest.fail('VNA_CERTSWAP_CIFAR_ROOT must point to real CIFAR-100')
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
        (0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
    dataset=CIFAR100(root,train=True,download=False,transform=transform)
    assert len(dataset)==50000
    order=class_order(0,data_dir=root);mapping={c:i for i,c in enumerate(order[:10])}
    ids=[i for i,label in enumerate(dataset.targets) if label in mapping][:8]
    batch=[dataset[i] for i in ids]
    x=torch.stack([sample[0] for sample in batch]).cuda()
    y=torch.tensor([mapping[sample[1]] for sample in batch],device='cuda')
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(20261024);torch.cuda.manual_seed_all(20261024)
        native=IncrementalNet({'convnet_type':'resnet32'},False)
        native.convnet=resnet32_nobn();del native.convnet.fc
        native.update_fc(10)
        model=Prediction(native).cuda().train()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module,SkipScale):module.alpha.fill_(.1)
        sites=sites_for(model);assert len(sites)==15
        site=sites[-1]
        with torch.no_grad():
            site.first_layer.layer.weight[0].zero_();site.first_layer.layer.bias[0]=-1.
        cache=certswap.PrefixCache(model,sites,x)
        A=dict(weight=site.first_layer.layer.weight[1:2].detach().clone(),
               bias=site.first_layer.layer.bias[1:2].detach().clone())
        spec=ICaRLPyCIL(T=2,known_classes=5)
        ctx=dict(targets=y,known_classes=5,teacher_logits=torch.zeros(len(x),5,device='cuda'))
        yield model,sites,(x,y),cache,A,spec,ctx


def candidate(cache,site,A):
    first,second=site.first_layer.layer,site.second_layer.layer
    index=torch.tensor([0],device=first.weight.device)
    overrides={'conv_a.weight':first.weight.detach().index_copy(0,index,A['weight']),
               'conv_a.bias':first.bias.detach().index_copy(0,index,A['bias'])}
    outgoing=second.weight.detach().clone();outgoing[:,0]=0
    def forward(vector):
        return cache.forward(site,{**overrides,'conv_b.weight':outgoing.index_copy(1,index,
            vector.reshape(outgoing.shape[0],1,*outgoing.shape[2:]))})
    return forward,outgoing


def test_real_cifar_dead_slot_has_exact_zero_ce_kd_jump(native_window):
    model,sites,batch,cache,A,spec,ctx=native_window
    site=sites[-1];forward,reference=candidate(cache,site,A)
    with torch.no_grad():
        before=cache.forward(site,None)
        swapped=forward(reference.new_zeros(reference[:,0].numel()))
    assert torch.equal(before,swapped)
    old=certswap.losses(before,spec,ctx);new=certswap.losses(swapped,spec,ctx)
    assert new[0]-old[0]==0. and new[1]-old[1]==0.


def test_real_cifar_suffix_matches_full_bitwise_and_shallow_cache_survives_deep_swap(native_window):
    model,sites,batch,cache,A,spec,ctx=native_window
    with torch.no_grad():
        full=model(batch[0]).double()
        for site in sites:assert torch.equal(cache.forward(site,None),full),site.module_name
        site=sites[-1]
        site.first_layer.layer.weight[0].copy_(A['weight'][0]);site.first_layer.layer.bias[0].copy_(A['bias'][0])
        site.second_layer.layer.weight[:,0].fill_(.001)
        full=model(batch[0]).double()
        for shallow in sites[:-1]:assert torch.equal(cache.forward(shallow,None),full),shallow.module_name


def test_real_cifar_same_candidate_repeats_j_and_rho_exactly(native_window):
    model,sites,batch,cache,A,spec,ctx=native_window
    site=sites[-1];forward,reference=candidate(cache,site,A)
    with torch.no_grad():before=certswap.losses(cache.forward(site,None),spec,ctx)
    state=deepcopy(model.state_dict());records=[]
    for _ in range(2):
        _,record=certswap.certify(forward,reference[:,0].numel(),reference,before,spec,ctx,mu0=1.,budget=4)
        assert record['jL']==record['jKD']==0.
        assert record['lm_trials'] and any(row['rho'] is not None for row in record['lm_trials'])
        records.append((record['jL'],record['jKD'],[(r['mu'],r['rho'],r['pred_B'],r['L'],r['KD']) for r in record['lm_trials']]))
    assert records[0]==records[1]
    assert all(torch.equal(value,state[name]) for name,value in model.state_dict().items())
