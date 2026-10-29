"""Dense oracle and transaction tests for opt-in VNA/GIR v2 algorithmic changes."""
from types import SimpleNamespace
import math
import pytest
import torch
from torch import nn
from methods.cl_replay import CorrectionConfig,pareto_minimax_kernel_target
from methods.cl_h3 import choose_swap,SwapCandidate
from methods.cl_replay import LossComponent


def test_config_defaults_preserve_registered_algorithm():
    cfg=CorrectionConfig()
    assert cfg.candidate_strategy=='original' and cfg.gir_selection=='original'
    assert CorrectionConfig(candidate_strategy='pareto_minimax').candidate_strategy=='pareto_minimax'
    with pytest.raises(ValueError,match='candidate strategy'):
        CorrectionConfig(candidate_strategy='no_such_method')
    with pytest.raises(ValueError,match='GIR selection'):
        CorrectionConfig(gir_selection='no_such_method')


def test_pareto_minimax_matches_dense_convex_hull_oracle():
    torch.manual_seed(4)
    J_E=torch.tensor([[2.,.2],[-1.,1.],[.4,1.5]],dtype=torch.double)
    K=J_E@J_E.T
    gnew=torch.tensor([1.,-2.,1.],dtype=torch.double)
    gold=torch.tensor([-1.,1.,1.],dtype=torch.double)
    target,info=pareto_minimax_kernel_target(gnew,gold,K@gnew,K@gold)
    assert not info['candidate_zero']
    vnew=J_E.T@gnew;vold=J_E.T@gold
    t=info['weight_new'];p=t*vnew+(1-t)*vold
    torch.testing.assert_close(target,K@(t*gnew+(1-t)*gold)/p.norm(),rtol=1e-12,atol=1e-12)
    assert float(gnew@target)>=-1e-11 and float(gold@target)>=-1e-11
    achieved=min(float(gnew@target),float(gold@target))
    # 2D exhaustive unit-circle comparison to the exact virtual-feature max-min objective.
    for theta in torch.linspace(0,2*math.pi,501,dtype=torch.double):
        unit=torch.tensor([math.cos(theta),math.sin(theta)],dtype=torch.double)
        assert achieved+1e-6>=min(float(vnew@unit),float(vold@unit))


def test_opposed_virtual_features_yield_zero_direction():
    K=torch.eye(2,dtype=torch.double)
    zn=torch.tensor([1.,0.],dtype=torch.double);zr=-zn
    result,info=pareto_minimax_kernel_target(zn,zr,K@zn,K@zr)
    assert info['candidate_zero'] and bool((result==0).all())


def test_asymmetric_or_non_psd_virtual_gram_cannot_claim_pareto():
    from methods.cl_projection import Uncertified
    a=torch.tensor([1.,0.],dtype=torch.double)
    b=torch.tensor([0.,1.],dtype=torch.double)
    with pytest.raises(Uncertified,match='Gram'):
        pareto_minimax_kernel_target(a,b,torch.tensor([-1.,0.],dtype=torch.double),b)
    with pytest.raises(Uncertified,match='Gram'):
        pareto_minimax_kernel_target(a,b,torch.tensor([1.,1.],dtype=torch.double),b)


def test_gir_qualifier_is_an_actual_swap_selection_gate():
    net=nn.Linear(1,1,bias=False).double();net.weight.data.fill_(2)
    opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9)
    x=torch.ones(3,1,dtype=torch.double)
    losses=(LossComponent('square',lambda logits:logits.square().mean()),)
    candidate=SwapCandidate('better','live',lambda m:m.weight.data.fill_(1.),lambda *args:None)
    score=lambda m:float(m.weight.detach().square().sum())
    old=choose_swap(net,opt,x,losses,[candidate],surrogate=score)
    assert old['choice']=='better'
    assert net.weight.item()==1.
    net.weight.data.fill_(2.)
    guarded=choose_swap(net,opt,x,losses,[candidate],surrogate=score,
        candidate_qualifier=lambda c,obj:False)
    assert guarded['choice']=='identity' and guarded['trials'][0]['candidate_qualified'] is False
    assert net.weight.item()==2.
    accepted=choose_swap(net,opt,x,losses,[candidate],surrogate=score,
        candidate_qualifier=lambda c,obj:True)
    assert accepted['choice']=='better' and net.weight.item()==1.


def test_h4_pareto_strategy_is_reached_and_safely_falls_back_on_zero_margin():
    from methods.cl_h4 import domain_losses
    from methods.cl_replay import ReplayCorrection
    from methods.vna_sites import HookSite
    from methods.cl_state import vector
    torch.manual_seed(0)
    net=nn.Sequential(nn.Linear(2,2),nn.ReLU(),nn.Linear(2,2)).double()
    opt=torch.optim.SGD(net.parameters(),lr=.1)
    x=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],dtype=torch.double)
    loss=domain_losses(torch.tensor([0,1,0,1]),torch.tensor([False,False,True,True]),
        current_domain=1,replay_domains=(0,1))
    event=ReplayCorrection(CorrectionConfig(candidate_strategy='pareto_minimax',
        damping_relative=1.,cg_tolerance=1e-8,preconditioner_probes=0)).step(
        net,opt,x,loss,[HookSite(net,'0','2','linear',between=('1',))],.1)
    assert event['status']=='ACCEPTED' and event['step_norm']>0
    assert event['candidate_strategy_applied']=='original_fallback_zero_virtual_margin'
    assert event['candidate_strategy_diagnostics']['candidate_zero']
    assert event['trials'][-1]['train_gate']['passed']


def test_h2_actual_feature_drift_budget_is_a_real_physical_gate():
    from methods.cl_h2 import GeometryCorrection
    from methods.cl_replay import ReplayLoss
    from methods.cl_state import vector
    class Network(nn.Module):
        def __init__(self):
            super().__init__();self.hidden=nn.Linear(2,2);self.head=nn.Linear(2,2)
        def standard_features(self,x):return self.hidden(x)
        def forward(self,x):return self.head(self.standard_features(x))
    torch.manual_seed(12)
    net=Network().double();opt=torch.optim.SGD(net.parameters(),lr=.01)
    x=torch.randn(4,2,dtype=torch.double)
    losses=ReplayLoss(torch.zeros(4,dtype=torch.long),torch.tensor([False,False,True,True]))
    before=vector(net).clone()
    event=GeometryCorrection(CorrectionConfig(kernel='native',damping_relative=1.,max_trials=3)).step(
        net,opt,x,losses,[],.01,replay_inputs=x[2:],geometry_weight=10.,feature_drift_budget=0.)
    assert not event['accepted']
    assert all('additional' in row['failed_gates'] or row['physical_norm']==0
               for row in event.get('trials',[]))
    assert event['feature_drift_budget']==0.
    torch.testing.assert_close(before,vector(net),rtol=0,atol=0)

def test_h3_certified_plasticity_policy_is_used_for_real_candidate_selection():
    from methods.cl_h3 import TangentReconfiguration
    from methods.cl_replay import CurrentTaskLoss
    from methods.vna_sites import HookSite
    torch.manual_seed(0)
    net=nn.Sequential(nn.Linear(2,2),nn.ReLU(),nn.Linear(2,2)).double()
    with torch.no_grad():
        net[0].weight.fill_(1);net[0].bias.copy_(torch.tensor([-10.,1.]))
    opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9)
    x=torch.tensor([[0.,1.],[1.,0.],[1.,1.]],dtype=torch.double)
    loss=CurrentTaskLoss(torch.tensor([0,1,0]))
    result=TangentReconfiguration(CorrectionConfig(gir_selection='certified_plasticity',
        damping_relative=1.),max_candidates=4).step(
        net,opt,x,loss,[HookSite(net,'0','2','linear',between=('1',))])
    assert result['gir_selection_policy']=='certified_plasticity'
    assert result['status'] in ('ACCEPTED','IDENTITY')
    if result['status']=='ACCEPTED':
        winner=next(t for t in result['trials'] if t['candidate']==result['choice'])
        assert winner['candidate_qualified'] and result['surrogate_wins_identity_certified']
        identity=result['identity']
        assert (winner['predicted_gn_recovery']-winner['net_surrogate_error_bound']>
                identity['predicted_gn_recovery']+identity['net_surrogate_error_bound'])


def test_h2_calibration_trials_use_the_same_drift_gate(monkeypatch):
    from experiments.standard_online import GeometryPlugin
    from methods.cl_h2 import GeometryCorrection
    from torch.utils.data import TensorDataset
    class Dataset(TensorDataset):
        def eval(self):return self
    class Network(nn.Linear):
        def sites(self):return []
    net=Network(2,2).double();opt=torch.optim.SGD(net.parameters(),lr=.01)
    cfg=dict(correction={},geometry_weight=1.,lr=.01,feature_drift_budget=.1,
        geometry_selection=dict(weights=[0.,1.],first_task=2,max_windows=3,validation_per_task=4))
    plugin=GeometryPlugin(cfg,0);plugin.task=1;plugin.seen_classes={0,1}
    x=torch.ones(4,2,dtype=torch.double);y=torch.tensor([0,1,0,1])
    plugin.validation_stream=[SimpleNamespace(dataset=Dataset(x,y))]
    budgets=[]
    def trial(self,*args,**kwargs):
        budgets.append(kwargs['feature_drift_budget'])
        return dict(accepted=False,status='REJECTED',step_norm=0.)
    monkeypatch.setattr(GeometryCorrection,'step',trial)
    result=plugin.select_geometry(SimpleNamespace(model=net,optimizer=opt),x,None,torch.tensor([False,False,True,True]))
    assert budgets==[.1,.1] and result['selected_weight'] is None
