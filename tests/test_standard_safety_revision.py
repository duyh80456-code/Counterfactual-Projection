import json
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from methods.cl_projection import ProjectionOperator,ConstraintSolver,Uncertified
from methods.cl_replay import CorrectionConfig,ReplayCorrection,CurrentTaskLoss,CandidateUpdate,LossComponent
from methods.cl_state import State,vector


def test_operator_rejects_bad_adjoint_without_rng_changes():
    before=torch.get_rng_state().clone()
    A=ProjectionOperator(lambda x:x,lambda x:2*x,lambda x:x,1.)
    with pytest.raises(Uncertified,match='operator audit'):A.solve(torch.ones(3,dtype=torch.double))
    assert torch.equal(before,torch.get_rng_state())


def test_single_current_task_objective_has_no_dummy_replay():
    z=torch.randn(4,2,dtype=torch.double,requires_grad=True)
    loss=CurrentTaskLoss(torch.tensor([0,1,0,1]))
    assert len(loss.components)==1
    torch.testing.assert_close(loss.total(z),loss.components[0](z))
    A=ProjectionOperator(lambda x:x,lambda x:x,lambda x:x,1.)
    result=ConstraintSolver(A).solve(torch.tensor([-1.,1.],dtype=torch.double),torch.tensor([[1.,0.]],dtype=torch.double))
    assert result.slack.min()>=-1e-10


def test_domain_current_memory_is_not_old_reference():
    from methods.cl_h4 import domain_losses
    y=torch.tensor([0,1,0,1]);mask=torch.tensor([False,False,True,True])
    losses=domain_losses(y,mask,current_domain=2,replay_domains=(1,2))
    assert losses.replay.tolist()==[False,False,True,False]
    z=torch.randn(4,2,dtype=torch.double)
    torch.testing.assert_close(losses.total(z),domain_losses(y,mask).total(z))


def test_eval_increase_rejects_train_improvement(monkeypatch):
    from methods import cl_replay
    net=nn.Linear(1,1,bias=False).double();net.weight.data.zero_()
    opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9);state=State(net,opt)
    engine=ReplayCorrection(CorrectionConfig(max_trials=1))
    losses=SimpleNamespace(components=(LossComponent('loss',lambda z:z.mean()),))
    monkeypatch.setattr(engine,'propose',lambda *a,**kw:CandidateUpdate(torch.ones(1,dtype=torch.double),torch.ones(1,1,dtype=torch.double),None,{}))
    monkeypatch.setattr(cl_replay,'evaluate',lambda m,x,c,mode='train':(1.+float(m.weight.abs()) if mode=='eval' else 1.+float(m.weight),))
    event=engine.step(net,opt,torch.ones(2,1,dtype=torch.double),losses,[],.1)
    assert event['status']=='REJECTED' and event['rejection_reason']=='EVAL_LOSS_INCREASE'
    assert torch.equal(net.weight,state.values['weight'])


def test_gir_retries_whole_transaction_and_counts_one_event(monkeypatch):
    from methods.cl_h3 import TangentReconfiguration
    net=nn.Linear(1,1,bias=False);opt=torch.optim.SGD(net.parameters(),lr=.1)
    original=vector(net).clone();engine=TangentReconfiguration(CorrectionConfig(damping_retries=1))
    calls=[]
    def attempt(*args,damping=None):
        calls.append(damping);assert torch.equal(vector(net),original)
        if damping is None:
            net.weight.data.add_(3)
            return dict(status='UNCERTIFIED',reason='linear solve failed test',solver_failure=[dict(mu=.1)])
        return dict(status='IDENTITY',trials=[],solves=[])
    monkeypatch.setattr(engine,'_attempt',attempt)
    event=engine.step(net,opt,torch.ones(2,1),None,[])
    assert calls==[None,1.] and len(event['damping_attempts'])==2
    assert len(engine.events)==engine.applications==1
    assert torch.equal(vector(net),original)


def test_acceptance_report_reads_both_h3_streams(tmp_path):
    from scripts.collect_standard_cl import acceptance_report
    (tmp_path/'interventions.json').write_text(json.dumps([dict(task=1,status='UNCERTIFIED',reason='solver')]))
    (tmp_path/'vna_interventions.json').write_text(json.dumps([dict(task=1,status='ACCEPTED',acceptance_detail='ACCEPTED_NEUTRAL',step_norm=.1,certified=True,candidate_generated=True)]))
    rows=acceptance_report(tmp_path)
    assert len(rows)==2 and sum(r['attempts'] for r in rows)==2
    assert sum(r['neutral'] for r in rows)==1 and sum(r['accepted_effective'] for r in rows)==1


def test_bn_jvp_vjp_and_finite_difference_are_consistent():
    from methods.vna_v3 import training_products
    from methods.cl_state import predict,subtract
    torch.manual_seed(71)
    net=nn.Sequential(nn.Linear(2,3),nn.BatchNorm1d(3),nn.Tanh(),nn.Linear(3,2)).double()
    x=torch.randn(5,2,dtype=torch.double);initial=State(net)
    d=torch.randn_like(vector(net));d/=d.norm()
    with training_products(net,[],{},x) as products:
        jd=products.Jv(d).double();u=torch.randn_like(jd)
        torch.testing.assert_close(jd@u,d@products.JT(u).double(),rtol=1e-8,atol=1e-9)
    subtract(net,d,-1e-5);plus=predict(net,x,'train').flatten();initial.restore()
    subtract(net,d,1e-5);minus=predict(net,x,'train').flatten();initial.restore()
    torch.testing.assert_close(jd,(plus-minus)/2e-5,rtol=1e-5,atol=1e-7)
    assert torch.equal(vector(net),torch.cat([v.flatten() for k,v in initial.values.items() if k in dict(net.named_parameters())]).double())


def test_gir_unexpected_error_restores_optimizer_rng_and_parameters():
    from methods.cl_h3 import SwapCandidate,choose_swap
    from diagnostics.vna_align import _assert_bitwise
    net=nn.Linear(1,1,bias=False).double();opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9)
    x=torch.ones(3,1,dtype=torch.double);loss=(LossComponent('square',lambda z:z.square().mean()),)
    net(x).sum().backward();opt.step();state=State(net,opt)
    def fail(model):
        model.weight.data.add_(4);torch.rand(3)
        opt.state[model.weight]['momentum_buffer'].add_(5)
        raise RuntimeError('candidate error')
    with pytest.raises(RuntimeError,match='candidate error'):
        choose_swap(net,opt,x,loss,[SwapCandidate('bad','live',fail,lambda *a:None)])
    _assert_bitwise(net.state_dict(),state.values,'gir error model')
    _assert_bitwise(opt.state_dict(),state.opt,'gir error optimizer')
    _assert_bitwise(torch.get_rng_state(),state.rng['torch'],'gir error rng')


def test_ablations_keep_main_protocol_and_compile():
    import ast
    from scripts.build_kaggle_standard_compare import build_ablations,jobs
    assert len(jobs('H3')[0])==2 and len(jobs('H3')[1])==3
    for branch in ('H1','H2','H3','H4'):
        nb=build_ablations(branch,'a'*40)
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))


def test_repeated_restore_keeps_optimizer_snapshot_immutable():
    net=nn.Linear(1,1);opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9)
    net(torch.ones(2,1)).sum().backward();opt.step();state=State(net,opt)
    initial=state.opt['state'][0]['momentum_buffer'].clone()
    for _ in range(3):
        state.restore();opt.state[net.weight]['momentum_buffer'].add_(7)
        assert torch.equal(state.opt['state'][0]['momentum_buffer'],initial)
    state.restore();assert torch.equal(opt.state[net.weight]['momentum_buffer'],initial)


def test_acceptance_report_retains_tasks_without_interventions(tmp_path):
    from scripts.collect_standard_cl import acceptance_report
    (tmp_path/'summary.json').write_text(json.dumps(dict(tasks=[dict(task=1),dict(task=2)])))
    (tmp_path/'interventions.json').write_text('[]')
    rows=acceptance_report(tmp_path)
    assert len(rows)==2 and all(r['attempts']==r['accepted_effective']==0 for r in rows)
