"""Physical-size pairing and conditional H3 certificate regressions."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from methods.cl_replay import CorrectionConfig,ReplayCorrection,CandidateUpdate,LossComponent,ReplayLoss
from methods.cl_h3 import certified_plasticity_diagnostics,choose_swap,SwapCandidate
from methods.cl_state import State,vector
from methods.vna_sites import HookSite


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_matched_first_trial_ignores_raw_direction_magnitude(monkeypatch,dtype):
    trials=[]
    for magnitude in (.01,100.):
        net=nn.Linear(1,1,bias=False).to(dtype);net.weight.data.fill_(1.)
        opt=torch.optim.SGD(net.parameters(),lr=.001)
        engine=ReplayCorrection(CorrectionConfig(physical_update_norm=.01,physical_budget_policy='matched'))
        direction=torch.tensor([magnitude],dtype=torch.double)
        monkeypatch.setattr(engine,'propose',lambda *a,**kw:CandidateUpdate(direction,torch.ones(1,1,dtype=torch.double),None,{}))
        loss=SimpleNamespace(components=(LossComponent('square',lambda z:z.square().mean()),))
        event=engine.step(net,opt,torch.ones(2,1,dtype=dtype),loss,[],.001)
        assert event['accepted'] and event['trials'][-1]['in_update_budget']
        assert event['step_norm']<=.01
        assert event['trials'][0]['nominal_update_norm']==pytest.approx(.01*(1-16*torch.finfo(dtype).eps))
        assert event['trials'][0]['physical_budget_fill_ratio']==pytest.approx(1.,abs=3e-6)
        trials.append(event['trials'][0]['physical_norm'])
    assert trials[0]==pytest.approx(trials[1],abs=1e-12)


def test_matched_pareto_and_original_share_real_checkpoint(tmp_path):
    torch.manual_seed(0)
    net=nn.Sequential(nn.Linear(2,4),nn.ReLU(),nn.Linear(4,2)).double()
    opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9)
    x=torch.randn(6,2,dtype=torch.double)
    loss=ReplayLoss(torch.tensor([0,1,0,1,0,1]),torch.tensor([False]*3+[True]*3))
    checkpoint=tmp_path/'paired.pt'
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),checkpoint)
    anchor=State(net,opt);events=[]
    for strategy in ('original','pareto_minimax'):
        anchor.restore();saved=torch.load(checkpoint,weights_only=False)
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        before=vector(net).clone()
        engine=ReplayCorrection(CorrectionConfig(candidate_strategy=strategy,
            physical_update_norm=.01,physical_budget_policy='matched',damping_relative=1.,preconditioner_probes=0))
        event=engine.step(net,opt,x,loss,[HookSite(net,'0','2','linear',between=('1',))],.01)
        assert event['trials'][0]['nominal_update_norm']==pytest.approx(.01)
        assert event['trials'][0]['physical_norm']==pytest.approx(.01,abs=1e-12)
        if strategy=='pareto_minimax':
            assert event['candidate_strategy_applied']=='pareto_minimax' and event['accepted']
            assert float((vector(net)-before).norm())>0
        events.append(event)
    assert events[0]['trials'][0]['physical_norm']==pytest.approx(events[1]['trials'][0]['physical_norm'],abs=1e-12)


def test_cap_default_and_backtracking_rollback(monkeypatch):
    net=nn.Linear(1,1,bias=False).double();net.weight.data.fill_(1.)
    opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9);snapshot=State(net,opt)
    loss=SimpleNamespace(components=(LossComponent('square',lambda z:z.square().mean()),))
    engine=ReplayCorrection(CorrectionConfig(physical_update_norm=.01,physical_budget_policy='matched',max_trials=3))
    monkeypatch.setattr(engine,'propose',lambda *a,**kw:CandidateUpdate(torch.ones(1,dtype=torch.double),torch.ones(1,1,dtype=torch.double),None,{}))
    result=engine.step(net,opt,torch.ones(2,1,dtype=torch.double),loss,[],.1,trial_guard=lambda m:dict(passed=False))
    assert result['status']=='REJECTED' and len(result['trials'])==3
    assert [t['nominal_update_norm'] for t in result['trials']]==pytest.approx([.01,.005,.0025])
    assert torch.equal(net.weight,snapshot.values['weight']) and opt.state_dict()==snapshot.opt
    cfg=CorrectionConfig();assert cfg.physical_update_norm is None and cfg.physical_budget_policy=='cap'
    with pytest.raises(ValueError,match='requires an update norm'):
        ReplayCorrection(CorrectionConfig(physical_budget_policy='matched')).step(net,opt,torch.ones(2,1),loss,[],.1)


def test_conditional_plasticity_eligibility_and_real_swap():
    identity=dict(net_surrogate_interval=[3.,3.2],predicted_gn_recovery=1.,net_surrogate_error_bound=.1)
    candidate=dict(net_surrogate_interval=[2.,2.2],predicted_gn_recovery=2.,net_surrogate_error_bound=.1)
    certificate=certified_plasticity_diagnostics(identity,candidate)
    assert certificate['eligible'] and not certificate['uncertain']
    uncertain=deepcopy(candidate);uncertain.update(net_surrogate_interval=[2.,3.1],net_surrogate_error_bound=1.)
    assert not certified_plasticity_diagnostics(identity,uncertain)['eligible']
    assert certified_plasticity_diagnostics(identity,uncertain)['uncertain']
    net=nn.Linear(1,1,bias=False).double();net.weight.data.fill_(2.)
    opt=torch.optim.SGD(net.parameters(),lr=.1)
    slot=SwapCandidate('better','live',lambda m:m.weight.data.fill_(1.),lambda *a:None)
    loss=(LossComponent('square',lambda z:z.square().mean()),)
    for score,expected in ((uncertain,'identity'),(candidate,'better')):
        net.weight.data.fill_(2.)
        result=choose_swap(net,opt,torch.ones(2,1,dtype=torch.double),loss,[slot],
            surrogate=lambda m:float(m.weight.detach().square().sum()),
            candidate_qualifier=lambda *a:certified_plasticity_diagnostics(identity,score)['eligible'])
        assert result['choice']==expected


def test_paired_configs_and_h4_propagation():
    from experiments.standard_domain import parse_args
    for branch in ('h1','h4'):
        pair=[json.loads(Path(f'configs/standard_cl/{branch}_physical_matched_{v}.json').read_text()) for v in ('v1','v2')]
        controls=[cfg['correction'] if branch=='h1' else cfg for cfg in pair]
        assert [c['candidate_strategy'] for c in controls]==['original','pareto_minimax']
        assert all(c['physical_update_norm']==.01 and c['physical_budget_policy']=='matched' for c in controls)
        controls[1]['candidate_strategy']='original';assert pair[0]==pair[1]
    args=parse_args(['--config','configs/standard_cl/h4_physical_matched_v2.json','--data-dir','/tmp/data','--out','/tmp/result'])
    assert args.physical_update_norm==.01 and args.physical_budget_policy=='matched' and args.candidate_strategy=='pareto_minimax'
