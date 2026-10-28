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


def test_bad_gir_candidate_does_not_cancel_later_winner():
    from methods.cl_h3 import choose_swap,SwapCandidate
    net=nn.Linear(1,1,bias=False).double();net.weight.data.fill_(2)
    opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9)
    x=torch.ones(3,1,dtype=torch.double)
    def install(value):
        return lambda model:model.weight.data.fill_(value)
    def score(model):
        if model.weight.item()==3:raise Uncertified('linear solve failed candidate',[dict(mu=.1)])
        return model.weight.square().sum().item()
    result=choose_swap(net,opt,x,(LossComponent('square',lambda z:z.square().mean()),),[
        SwapCandidate('bad','live',install(3),lambda *a:None),
        SwapCandidate('good','live',install(1),lambda *a:None)],surrogate=score)
    assert result['choice']=='good' and net.weight.item()==1
    assert result['trials'][0]['rejection_reason']=='CANDIDATE_SOLVER_FAILURE'


def test_roundoff_gates_allow_float32_noise_but_not_real_increase():
    from methods.cl_replay import loss_gate
    assert loss_gate([1.],[1.+1e-7],torch.float32)['passed']
    assert not loss_gate([1.],[1.001],torch.float32)['passed']
    assert not loss_gate([1.],[float('nan')],torch.float32)['passed']


def test_h4_correction_applies_with_previous_domain_reference():
    from methods.cl_h4 import domain_losses
    from methods.vna_sites import HookSite
    torch.manual_seed(0)
    net=nn.Sequential(nn.Linear(2,2),nn.ReLU(),nn.Linear(2,2)).double()
    opt=torch.optim.SGD(net.parameters(),lr=.1)
    x=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],dtype=torch.double)
    loss=domain_losses(torch.tensor([0,1,0,1]),torch.tensor([False,False,True,True]),
        current_domain=1,replay_domains=(0,1))
    engine=ReplayCorrection(CorrectionConfig(damping_relative=1.,cg_tolerance=1e-8))
    before=vector(net).clone();event=engine.step(net,opt,x,loss,[HookSite(net,'0','2','linear',between=('1',))],.1)
    assert event['accepted'] and event['step_norm']>0 and not torch.equal(before,vector(net))
    assert event['trials'][-1]['train_gate']['passed'] and event['trials'][-1]['eval_gate']['passed']


def test_h3_current_task_real_correction_and_gir_swap():
    from methods.cl_h3 import TangentReconfiguration
    from methods.vna_sites import HookSite
    torch.manual_seed(0)
    net=nn.Sequential(nn.Linear(2,2),nn.ReLU(),nn.Linear(2,2)).double()
    with torch.no_grad():net[0].weight.fill_(1);net[0].bias.copy_(torch.tensor([-10.,1.]))
    opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9)
    x=torch.tensor([[0.,1.],[1.,0.],[1.,1.]],dtype=torch.double);loss=CurrentTaskLoss(torch.tensor([0,1,0]))
    event=ReplayCorrection(CorrectionConfig(damping_relative=1.)).step(net,opt,x,loss,[HookSite(net,'0','2','linear',between=('1',))],.01)
    assert event['accepted'] and event['step_norm']>0
    gir=TangentReconfiguration(CorrectionConfig(damping_relative=1.),max_candidates=4)
    result=gir.step(net,opt,x,loss,[HookSite(net,'0','2','linear',between=('1',))])
    assert result['certified'] and result['candidate_generated'] and result['trials']
    assert result['status']=='ACCEPTED' and result['actual_replacements']==1
    winner=next(t for t in result['trials'] if t['candidate']==result['choice'])
    assert winner['step_norm']>0 and winner['train_gate']['passed'] and winner['eval_gate']['passed']


def test_iterative_refinement_recovers_solution_without_damping_change(monkeypatch):
    from methods import cl_projection
    mu=1e-9;j=torch.tensor([0.,(1-mu)**.5],dtype=torch.double)
    A=ProjectionOperator(lambda x:j*x,lambda x:j*x,lambda x:x,mu,preconditioner_probes=4)
    b=torch.tensor([mu,1.],dtype=torch.double);calls=[]
    def solve(operator,rhs,**kwargs):
        calls.append(1)
        x=torch.tensor([0.,1.],dtype=torch.double) if len(calls)==1 else rhs/torch.tensor([mu,1.],dtype=torch.double)
        return SimpleNamespace(solution=x,iterations=1,breakdown=False)
    monkeypatch.setattr(cl_projection,'conjugate_gradient',solve)
    answer=A.solve(b)
    torch.testing.assert_close(answer,torch.ones(2,dtype=torch.double))
    assert A.solves[-1]['refinements'] and A.mu==mu
    assert A.solves[-1]['relative_solution_error_bound']<=.1


def test_geometry_curvature_contributions_are_measured():
    from methods.cl_h2 import GeometryCorrection
    from methods.cl_replay import ReplayLoss
    class Net(nn.Module):
        def __init__(self):super().__init__();self.hidden=nn.Linear(2,2);self.head=nn.Linear(2,2)
        def standard_features(self,x):return self.hidden(x)
        def forward(self,x):return self.head(self.standard_features(x))
    torch.manual_seed(12);net=Net().double();opt=torch.optim.SGD(net.parameters(),lr=.01)
    x=torch.randn(4,2,dtype=torch.double);loss=ReplayLoss(torch.zeros(4,dtype=torch.long),torch.tensor([False,False,True,True]))
    event=GeometryCorrection(CorrectionConfig(kernel='native',damping_relative=1.)).step(net,opt,x,loss,[],.001,replay_inputs=x[2:],geometry_weight=10.)
    assert event['geometry_curvature']>0 and event['ridge_curvature']>0
    assert 0<event['geometry_fraction']<1
    assert event['geometry_to_ggn']>0 and event['geometry_to_ridge']>0


def test_h4_runner_real_engine_commits_corrections(tmp_path):
    from experiments import standard_domain as runner
    args=runner.parse_args(['--data-dir',str(tmp_path),'--out',str(tmp_path/'run'),
        '--fake-data','--device','cpu','--domains','2','--epochs','1','--batch-size','4'])
    summary=runner.run(args)
    assert summary['status']=='COMPLETED'
    events=json.loads((tmp_path/'run/interventions.json').read_text())
    accepted=[e for e in events if e.get('accepted')]
    assert accepted and all(e['step_norm']>0 and e['trials'][-1]['train_gate']['passed']
        and e['trials'][-1]['eval_gate']['passed'] for e in accepted)
    assert any(e['previous_domain_samples']>0 for e in events)


def test_h3_runner_uses_real_correction_engines(tmp_path,monkeypatch):
    import numpy as np
    import pickle
    from experiments import standard_imagenet as runner
    from experiments.runners import lop
    lop.upstream_imports()
    from lop.nets import conv_net
    class Tiny(nn.Module):
        def __init__(self,num_classes=2):
            super().__init__();self.layers=nn.ModuleList([nn.Linear(2,2),nn.ReLU(),nn.Linear(2,num_classes)])
            self.act_type='relu';self.last_filter_output=1
            self.double()
        def predict(self,x):
            h=self.layers[1](self.layers[0](x));return self.layers[2](h),[h]
    monkeypatch.setattr(conv_net,'ConvNet',Tiny)
    original=lop.UPSTREAM;upstream=tmp_path/'native';(upstream/'lop/imagenet/cfg').mkdir(parents=True)
    params=json.loads((original/'lop/imagenet/cfg/cbp.json').read_text());params['mini_batch_size']=4
    (upstream/'lop/imagenet/cfg/cbp.json').write_text(json.dumps(params))
    with (upstream/'lop/imagenet/class_order').open('wb') as stream:pickle.dump([np.arange(1000)],stream)
    monkeypatch.setattr(lop,'UPSTREAM',upstream);monkeypatch.setattr(lop,'upstream_imports',lambda:None)
    monkeypatch.setattr(runner,'data_identity',lambda *a:{'synthetic':True})
    x=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],dtype=torch.double);y=torch.tensor([0,1,0,1])
    monkeypatch.setattr(runner,'load_task',lambda *a:(x.clone(),y.clone(),x.clone(),y.clone()))
    cfg=json.loads((__import__('pathlib').Path('configs/standard_cl/h3_imagenet_small.json')).read_text())
    cfg.update(num_tasks=1,num_showings=2,intervention_every=1,max_candidates=4)
    cfg['correction']['damping_relative']=1.
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg));out=tmp_path/'run'
    runner.run(runner.parser().parse_args(['--arm','vna_gir','--data-dir',str(tmp_path),'--device','cpu','--config',str(config),'--out',str(out)]))
    summary=json.loads((out/'summary.json').read_text());events=json.loads((out/'vna_interventions.json').read_text())
    assert summary['status']=='COMPLETED' and any(e.get('accepted') and e['step_norm']>0 for e in events)
    swaps=json.loads((out/'interventions.json').read_text())
    assert swaps and all('damping_attempts' in e for e in swaps)


def test_noisy_jacobi_shrinkage_stays_positive_and_certified():
    j=torch.ones(1,24,dtype=torch.double)
    A=ProjectionOperator(lambda x:j@x,lambda y:j.T@y,lambda x:x,.1,
        preconditioner_probes=16,tolerance=1e-10,max_iterations=100)
    rng=torch.get_rng_state().clone();rhs=torch.ones(24,dtype=torch.double)
    answer=A.solve(rhs)
    torch.testing.assert_close(answer,rhs/24.1,rtol=1e-7,atol=1e-9)
    assert A.preconditioner_quality['estimator']=='variance_shrunk_hutchinson'
    assert A.preconditioner_quality['shrinkage_fraction']>0
    assert bool((A._preconditioner>0).all()) and torch.equal(rng,torch.get_rng_state())


def test_jacobi_exact_zero_curvature_coordinates_keep_ridge_floor():
    values=torch.cat((torch.zeros(12),torch.ones(12))).double()
    A=ProjectionOperator(lambda x:values*x,lambda x:values*x,lambda x:x,1.,preconditioner_probes=4)
    A._jacobi(torch.ones(24,dtype=torch.double))
    torch.testing.assert_close(A._preconditioner,values.square()+1.,rtol=0,atol=0)
