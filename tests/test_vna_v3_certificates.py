"""Certificates, native optimizer isolation and residual-network training products."""
from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from torch import nn

from methods.vna_losses import MSE, ICaRLPyCIL
from methods.vna_sites import HookSite
from methods.vna_ggn import GGNProducts
from methods.vna_v3 import V3Engine, seed_proposals, training_products, statistics


@pytest.fixture(autouse=True)
def isolated():
    threads = torch.get_num_threads()
    with torch.random.fork_rng():
        torch.manual_seed(71)
        torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def mlp(outputs=1, bn=False):
    model=nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4) if bn else nn.Identity(),
                        nn.ReLU(),nn.Linear(4,outputs)).double().train()
    site=HookSite(model,'0','3','linear',between=('1','2'))
    x=torch.randn(7,3,dtype=torch.float64)
    y=torch.randn(7,outputs,dtype=torch.float64)
    return model,site,(x,y),torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.005)


class AdversarialKernel(V3Engine):
    """Invert the extension kernel to exercise rejection of a genuine negative gain."""
    def _direction(self,*args,**kwargs):
        values=super()._direction(*args,**kwargs)
        if 'd' in values:
            for key in ('d','d_N','d_rep','uN'): values[key]=-values[key]
            kernel=values['kernel']
            values['kernel']=lambda vector:-kernel(vector)
        return values


def test_negative_total_certificate_skips_without_parameter_or_buffer_mutation(monkeypatch):
    model,site,batch,optimizer=mlp(bn=True)
    before=deepcopy(model.state_dict())
    engine=AdversarialKernel('vna');engine.begin_task(1);engine.tol=1e-12
    # Step sizing must only happen after the certificate has passed.
    def forbidden(*args,**kwargs): raise AssertionError('step size evaluated before certificate')
    monkeypatch.setattr('methods.vna_v3.correction_step',forbidden)
    row=engine.apply(model,[site],batch,MSE(),dict(targets=batch[1]),optimizer,1)
    assert row['g_dot_d'] < -row['tau_g']
    assert row['skipped'] and row['skip_reason']=='negative_total_certificate'
    assert row['s']==0 and row['K_next']==engine.K_base
    for name,value in before.items(): assert torch.equal(value,model.state_dict()[name])


def test_zero_old_residual_with_inexact_cg_does_not_false_skip():
    model,site,(x,_),optimizer=mlp(outputs=4)
    y=torch.tensor([0,1,2,3,0,2,3])
    with torch.no_grad(): teacher=model(x)[:,:2].clone()
    class OneIteration(V3Engine):
        @property
        def cfg(self): return replace(super().cfg,cg_iters=1)
    engine=OneIteration('vna');engine.begin_task(2)
    row=engine.apply(model,[site],(x,y),ICaRLPyCIL(T=2,known_classes=2),
        dict(targets=y,known_classes=2,teacher_logits=teacher),optimizer,1)
    assert row['V_O']==0 and row['h_dot_d']==0
    assert any(r['relative_residual']>1e-3 for r in row['cg'])
    assert row['tau_g']>0 and not row['skipped'] and not row['false_skip']


def test_nonconverged_exact_check_is_logged_but_not_counted():
    model, site, batch, optimizer = mlp()
    class IncompleteExact(V3Engine):
        def _direction(self, p, op, mu, **kwargs):
            values = super()._direction(p, op, mu, **kwargs)
            if kwargs.get('exact'):
                values['records'][0]['converged'] = False
            return values
    engine = IncompleteExact('vna'); engine.begin_task(1)
    row = engine.apply(model, [site], batch, MSE(), dict(targets=batch[1]), optimizer, 1)
    assert row['exact_cg_converged'] is False
    assert row['false_skip'] is False
    assert engine.exact_checks == engine.false_skips == 0
    assert engine.tol == 1e-3 and engine.power_iters == 5


def test_real_sgd_momentum_and_weight_decay_remain_native_after_correction():
    model,site,batch,optimizer=mlp(bn=True)
    x,y=batch
    optimizer.zero_grad();nn.functional.mse_loss(model(x),y).backward();optimizer.step()
    momentum={name:optimizer.state[p]['momentum_buffer'].clone() for name,p in model.named_parameters()}
    buffers={name:b.clone() for name,b in model.named_buffers()}
    before={name:p.clone() for name,p in model.named_parameters()}
    engine=V3Engine('vna');engine.begin_task(1)
    row=engine.apply(model,[site],batch,MSE(),dict(targets=y),optimizer,1)
    assert not row['skipped'] and row['s']>0
    assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())
    for name,p in model.named_parameters():
        assert torch.equal(momentum[name],optimizer.state[p]['momentum_buffer'])
    for name,b in model.named_buffers(): assert torch.equal(buffers[name],b)
    optimizer.zero_grad();nn.functional.mse_loss(model(x),y).backward()
    expected={name:p.detach()-optimizer.param_groups[0]['lr']*(.9*momentum[name]+p.grad+.005*p)
              for name,p in model.named_parameters()}
    optimizer.step()
    for name,p in model.named_parameters(): torch.testing.assert_close(p,expected[name],rtol=0,atol=1e-16)


class Block(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(cin,cout,3,stride,1,bias=False);self.bn1=nn.BatchNorm2d(cout)
        self.relu=nn.ReLU();self.conv2=nn.Conv2d(cout,cout,3,1,1,bias=False);self.bn2=nn.BatchNorm2d(cout)
        self.shortcut=nn.Identity() if cin==cout and stride==1 else nn.Conv2d(cin,cout,1,stride,bias=False)
    def forward(self,x): return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))+self.shortcut(x))


class ResNet8(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Conv2d(3,2,3,padding=1)
        self.blocks=nn.Sequential(Block(2,2),Block(2,4,2),Block(4,8,2));self.fc=nn.Linear(8,3)
    def forward(self,x): return self.fc(self.blocks(self.stem(x)).mean((2,3)))


def test_resnet8_cg_cosine_and_later_site_spatial_statistics():
    model=ResNet8().double().train()
    sites=[HookSite(model,f'blocks.{i}.conv1',f'blocks.{i}.conv2','conv',
                    between=(f'blocks.{i}.bn1',f'blocks.{i}.relu')) for i in range(3)]
    batch=torch.randn(2,3,8,8,dtype=torch.float64),torch.randn(2,3,dtype=torch.float64)
    engine=V3Engine('vna');engine.begin_task(1)
    with training_products(model,sites,seed_proposals(sites),batch[0]) as products:
        op=GGNProducts(products,batch[1],MSE(),dict(targets=batch[1]))
        approximate=engine._direction(products,op,.1)
        exact=engine._direction(products,op,.1,exact=True)
        assert nn.functional.cosine_similarity(approximate['d'],exact['d'],dim=0)>=.99
        for record in approximate['records']: assert record['relative_residual']<=1e-3
    for site in sites:
        S,Ns=statistics(model,site,batch,[torch.randn(6,dtype=torch.float64)])
        assert S.shape[0]==site.first_layer.layer.weight[0].numel()
        assert Ns[0].shape[0]==len(S)


def test_native_icarl_sites_exclude_channels_entering_residual_addition():
    from experiments.stage_b3 import upstream_imports
    upstream_imports()
    from adapters.icarl_hooks import sites_for, Prediction
    from convs.cifar_resnet import CifarResNet, ResNetBasicblock
    native=Prediction(CifarResNet(ResNetBasicblock,8))
    sites=sites_for(native)
    assert len(sites)==3
    assert all(s.first.endswith('.conv_a') and s.second.endswith('.conv_b') for s in sites)
    assert not any(s.first.endswith('.conv_b') for s in sites)


def test_gir_rechecks_actual_dead_slot_revived_by_upstream_correction(monkeypatch):
    model=nn.Sequential(nn.Linear(1,1),nn.ReLU(),nn.Linear(1,2),nn.ReLU(),nn.Linear(2,1)).double()
    with torch.no_grad():
        model[0].weight.zero_();model[0].bias.fill_(1)
        model[2].weight.fill_(1);model[2].bias.copy_(torch.tensor([-2.,0.]))
        model[4].weight.fill_(1);model[4].bias.zero_()
    site=HookSite(model,'2','4','linear',between=('3',))
    batch=torch.ones(4,1,dtype=torch.float64),torch.full((4,1),10.,dtype=torch.float64)
    optimizer=torch.optim.SGD(model.parameters(),lr=1.)
    class UpstreamDirection(V3Engine):
        def _direction(self,p,*args,**kwargs):
            values=super()._direction(p,*args,**kwargs)
            if 'd' in values:
                d=torch.zeros_like(values['d']);offset=0
                for name,parameter in p.parameter_refs:
                    if name=='0.bias': d[offset:offset+parameter.numel()]=-3
                    offset+=parameter.numel()
                values['d']=values['d_N']=d;values['d_rep']=torch.zeros_like(d)
            return values
    engine=UpstreamDirection('vna_gir');engine.begin_task(1)
    def forbidden(*args,**kwargs): raise AssertionError('revived slot must not be replaced')
    monkeypatch.setattr(engine,'prepare_gir',forbidden)
    assert engine.dead_slots(model,[site],batch)[site.module_name]==[0]
    row=engine.apply(model,[site],batch,MSE(),dict(targets=batch[1]),optimizer,1)
    assert not row['skipped'] and row['dead_before_step5']==1
    assert row['dead_after_step5']==0 and row['swaps']==0
    assert engine.dead_slots(model,[site],batch)[site.module_name]==[]
