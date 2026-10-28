"""Executable CPU gates for H1–H4; all stream fixtures are labelled synthetic."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from diagnostics.vna_align import _global_state,_assert_bitwise
from methods.cl_replay import ReplayLoss
from methods.cl_h2 import Features
from methods.vna_v3 import training_products
from methods.cl_projection import ProjectionOperator


@pytest.fixture(autouse=True)
def cpu_state():
    old=torch.get_num_threads();torch.set_num_threads(1)
    with _global_state():yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('kind',['er','icarl'])
@pytest.mark.parametrize('reduction',['pooled','separate_mean'])
def test_unequal_batch_reductions_match_gradients_and_hessian(kind,reduction):
    torch.manual_seed(44);net=nn.Sequential(nn.Linear(2,4),nn.BatchNorm1d(4),nn.Tanh(),nn.Linear(4,3)).double().train()
    x=torch.randn(5,2,dtype=torch.double);y=torch.tensor([0,0,1,2,2]);mask=torch.tensor([True,True,False,False,False])
    loss=ReplayLoss(y,mask,kind=kind,known=2,teacher=torch.randn(5,2,dtype=torch.double),reduction=reduction)
    z=net(x);components=[c(z) for c in loss.components]
    sum_grads=torch.autograd.grad(sum(components),tuple(net.parameters()),retain_graph=True)
    total_grads=torch.autograd.grad(loss.total(z),tuple(net.parameters()),retain_graph=True)
    for a,b in zip(sum_grads,total_grads):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
    v=torch.randn_like(z)
    hess=torch.autograd.functional.hvp(loss.total,z,v)[1]
    torch.testing.assert_close(loss.spec.metric_apply(z,v,loss.ctx),hess,atol=1e-12,rtol=1e-10)
    if kind=='er':
        ce=torch.nn.functional.cross_entropy
        expected=ce(z,y) if reduction=='pooled' else ce(z[mask],y[mask])+ce(z[~mask],y[~mask])
        torch.testing.assert_close(loss.total(z),expected)


def test_h2_feature_operator_dense_identity_and_small_step_drift():
    class FeatureNet(nn.Module):
        def __init__(self):super().__init__();self.hidden=nn.Linear(2,3);self.head=nn.Linear(3,2)
        def forward(self,x):
            h=self.hidden(x).tanh();return dict(features=h,fmaps=[h,h*2,h*3],logits=self.head(h))
    net=FeatureNet().double();features=Features(net,('features','stage_2'));x=torch.randn(4,2,dtype=torch.double)
    before=features(x).detach();params=tuple(features.parameters());size=sum(p.numel() for p in params)
    direction=torch.randn(size,dtype=torch.double);hooks=sum(len(m._forward_hooks) for m in net.modules())
    with training_products(features,[],{},x) as product:
        columns=torch.stack([product.Jv(e) for e in torch.eye(size,dtype=torch.double)],1).double()
        torch.testing.assert_close(product.JT(product.Jv(direction)),columns.T@columns@direction,atol=1e-11,rtol=1e-10)
        j=torch.randn(5,size,dtype=torch.double);h=torch.randn(5,5,dtype=torch.double);h=h@h.T
        ke=torch.randn(5,5,dtype=torch.double);ke=ke@ke.T;g=torch.randn(size,dtype=torch.double)
        A=ProjectionOperator(lambda d:j@d,lambda v:j.T@v,lambda v:h@v,2.,
            penalty=lambda d:.7*product.JT(product.Jv(d)),tolerance=1e-10)
        d,z=A.project(g,lambda v:ke@v);dense=j.T@h@j+.7*columns.T@columns+2*torch.eye(size,dtype=torch.double)
        expected=torch.linalg.solve(dense,j.T@h@ke@h@j@torch.linalg.solve(dense,g))
        torch.testing.assert_close(d,expected,atol=1e-9,rtol=1e-8)
        torch.testing.assert_close(g@d,z@ke@z,atol=1e-9,rtol=1e-8)
        predicted=columns@direction
    offset=0
    with torch.no_grad():
        for p in params:p.add_(direction[offset:offset+p.numel()].reshape_as(p),alpha=1e-6);offset+=p.numel()
    torch.testing.assert_close((features(x)-before).flatten()/1e-6,predicted,atol=1e-5,rtol=1e-4)
    assert sum(len(m._forward_hooks) for m in net.modules())==hooks


@pytest.mark.parametrize('branch,arm',[('H1','h1'),('H2','h2')])
def test_cil_two_tasks_native_memory_and_mid_epoch_resume(tmp_path,monkeypatch,branch,arm):
    from experiments import standard_cl as module
    cfg='configs/standard_cl/'+('h1' if branch=='H1' else 'h2')+'_cifar100_bn.json'
    def args(out):return module.parser().parse_args(['--config',cfg,'--data-dir',str(tmp_path),'--out',str(out),
        '--tasks','2','--device','cpu','--arm',arm,'--branch',branch,'--fake-data'])
    complete=tmp_path/'complete';paused=tmp_path/'paused'
    result=module.run(args(complete));assert result['status']=='COMPLETED' and len(result['tasks'])==2
    events=json.loads((complete/'interventions.json').read_text())
    assert events and any(e['status'] in ('ACCEPTED','REJECTED') for e in events)
    calls=0
    def due():
        nonlocal calls
        calls+=1;return calls==9
    monkeypatch.setattr(module,'deadline_due',due)
    with pytest.raises(SystemExit) as exc:module.run(args(paused))
    assert exc.value.code==75
    monkeypatch.setattr(module,'deadline_due',lambda:False)
    resumed=module.run(args(paused));assert resumed['status']=='COMPLETED'
    load=lambda p:torch.load(p/'latest.pt',weights_only=False)
    one,two=load(complete),load(paused)
    for key in ('_network','_old_network'):_assert_bitwise(one['learner'][key].state_dict(),two['learner'][key].state_dict(),key)
    for key in ('_data_memory','_targets_memory'):_assert_bitwise(one['learner'][key],two['learner'][key],key)
    _assert_bitwise(one['rng'],two['rng'],'resume_rng')
    assert one['steps']==two['steps'] and result['tasks']==resumed['tasks']


@pytest.mark.parametrize('method',['agem','er_ace'])
def test_strong_controls_execute_two_tasks(tmp_path,method):
    from experiments.standard_cl import parser,run
    args=parser().parse_args(['--data-dir',str(tmp_path),'--out',str(tmp_path/method),'--tasks','2',
        '--device','cpu','--arm','baseline','--baseline',method,'--fake-data'])
    result=run(args);assert result['status']=='COMPLETED' and len(result['tasks'])==2
    rows=json.loads((tmp_path/method/'baseline_updates.json').read_text())
    assert any(r['task']==2 and r['reference_loss']>0 for r in rows)


def test_agem_projection_and_er_ace_mask_are_real_update_laws():
    from methods.cl_baselines import backward
    class Net(nn.Module):
        def __init__(self):super().__init__();self.head=nn.Linear(1,3,bias=False)
        def forward(self,x):return {'logits':self.head(x)}
    net=Net().double();net.head.weight.data.zero_();opt=torch.optim.SGD(net.parameters(),lr=.1)
    x=torch.ones(1,1,dtype=torch.double);old=torch.tensor([0]);new=torch.tensor([1])
    _,_,_,event=backward(net,opt,x,new,(x,old),'agem',torch.tensor([0]),1)
    assert event['projected']
    reference=torch.autograd.grad(torch.nn.functional.cross_entropy(net(x)['logits'],old),tuple(net.parameters()))
    assert sum(float((p.grad*r).sum()) for p,r in zip(net.parameters(),reference))>=-1e-12
    backward(net,opt,x,new,None,'er_ace',torch.tensor([0]),1)
    assert net.head.weight.grad[0].eq(0).all()
    _,reported,_,event=backward(net,opt,x,new,(x,old),'er_ace',torch.tensor([0]),1)
    torch.testing.assert_close(reported,reported.new_tensor(2.).log()+event['reference_loss'])


@pytest.mark.parametrize('kind',['rotated','permuted'])
def test_domain_runner_two_domains_checkpoint_and_bn_diagnostics(tmp_path,monkeypatch,kind):
    from experiments import standard_domain as module
    def args(out):return module.parse_args(['--data-dir',str(tmp_path),'--out',str(out),'--kind',kind,'--domains','2',
        '--batch-size','4','--device','cpu','--arm','native','--fake-data'])
    complete=tmp_path/'complete';paused=tmp_path/'paused'
    result=module.run(args(complete));assert result['status']=='COMPLETED'
    assert len(result['tasks'])==2 and result['parameter_count']>0
    assert len(result['protocol']['transformations'])==2
    diagnostics=json.loads((complete/'normalization.json').read_text())
    assert any(d['eval_buffer_only_drift']>0 for d in diagnostics)
    calls=0
    def due():
        nonlocal calls
        calls+=1;return calls==5
    monkeypatch.setattr(module,'deadline_due',due)
    with pytest.raises(SystemExit):module.run(args(paused))
    raw=tmp_path/'MNIST/raw';raw.mkdir(parents=True,exist_ok=True)
    (raw/'optional-compressed-copy.gz').write_bytes(b'not part of canonical dataset identity')
    monkeypatch.setattr(module,'deadline_due',lambda:False)
    resumed=module.run(args(paused));assert result['tasks']==resumed['tasks']
    a=torch.load(complete/'latest.pt',weights_only=False);b=torch.load(paused/'latest.pt',weights_only=False)
    for key in ('model','optimizer','memory','rng'):_assert_bitwise(a[key],b[key],key)


@pytest.mark.parametrize('arm',['bp','cbp','redo','snp','h3'])
def test_plasticity_runner_authentic_adapters_and_generated_candidates(tmp_path,arm):
    from experiments.standard_plasticity import parser,run
    args=parser().parse_args(['--out',str(tmp_path/arm),'--arm',arm,'--fake-data'])
    result=run(args);assert result['status']=='COMPLETED' and len(result['tasks'])==2
    assert result['last_50_task_error']>=0
    if arm=='h3':
        events=json.loads((tmp_path/arm/'interventions.json').read_text())
        assert len(events)==2
        assert any(e.get('trials') for e in events),events
        assert all(e['status'] in ('ACCEPTED','IDENTITY') for e in events)


def test_real_resnet32_bn_forward_backward_correction():
    from experiments.standard_smoke import resnet_bn_smoke
    event=resnet_bn_smoke()['correction']
    assert event['status'] in ('ACCEPTED','REJECTED')
    assert event['solves'] and all(s['status']=='CERTIFIED' and s['relative_residual']<=1e-6
        and s['relative_solution_error_bound']<=.1 and s['operator_audit']['passed'] for s in event['solves'])


def test_generated_h3_live_dead_and_upstream_redo_reset():
    from methods.cl_h3 import TangentReconfiguration
    from methods.cl_replay import CorrectionConfig,LossComponent
    from methods.vna_losses import MSE
    from methods.vna_sites import HookSite
    net=nn.Sequential(nn.Linear(2,2),nn.ReLU(),nn.Linear(2,1)).double()
    with torch.no_grad():
        net[0].weight.fill_(1.);net[0].bias.copy_(torch.tensor([-10.,1.]));net[2].weight.fill_(1.)
    opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9)
    x=torch.tensor([[0.,1.],[1.,0.],[1.,1.]],dtype=torch.double);y=torch.zeros(3,1,dtype=torch.double)
    losses=SimpleNamespace(targets=y,spec=MSE(reduction='mean'),ctx=dict(targets=y),
        components=(LossComponent('mse',lambda z:(z-y).square().mean()),LossComponent('zero',lambda z:z.sum()*0)))
    event=TangentReconfiguration(CorrectionConfig(cg_tolerance=1e-8)).step(net,opt,x,losses,
        [HookSite(net,'0','2','linear',between=('1',))])
    assert {r['slot_kind'] for r in event['trials']}=={'dead','live'}
    assert all('predicted_gn_recovery' in r and 'vna_gain' in r for r in event['trials'])
    from experiments.runners.lop import upstream_imports
    upstream_imports()
    from lop.algos.gntRedo import GnTredo
    recycler=GnTredo(nn.ModuleList(list(net.children())),'relu',reset_period=1)
    original=net[0].weight.detach().clone()
    recycler.gen_and_test(torch.tensor([[[0.,1.]],[[0.,1.]]],dtype=torch.double))
    assert not torch.equal(net[0].weight[0],original[0]) and torch.equal(net[0].weight[1],original[1])
    assert net[2].weight[:,0].eq(0).all()


def test_h3_mid_candidate_deadline_resume(tmp_path,monkeypatch):
    from experiments import standard_plasticity as module
    from methods.cl_replay import ReplayCorrection
    def args(out):return module.parser().parse_args(['--out',str(out),'--arm','h3','--fake-data'])
    module.run(args(tmp_path/'complete'))
    original=ReplayCorrection.boundary;calls=0
    def stop():
        nonlocal calls
        calls+=1
        if calls==7:raise SystemExit(75)
        original()
    monkeypatch.setattr(ReplayCorrection,'boundary',staticmethod(stop))
    with pytest.raises(SystemExit):module.run(args(tmp_path/'paused'))
    monkeypatch.setattr(ReplayCorrection,'boundary',staticmethod(original))
    module.run(args(tmp_path/'paused'))
    a=torch.load(tmp_path/'complete/latest.pt',weights_only=False);b=torch.load(tmp_path/'paused/latest.pt',weights_only=False)
    for key in ('model','optimizer','rng','rows','curves'):_assert_bitwise(a[key],b[key],key)


@pytest.mark.parametrize('arm',['cbp','redo','snp'])
def test_plasticity_upstream_auxiliary_state_resume(tmp_path,monkeypatch,arm):
    from experiments import standard_plasticity as module
    def args(out):return module.parser().parse_args(['--out',str(out),'--arm',arm,'--fake-data'])
    module.run(args(tmp_path/'complete'));calls=0
    def stop():
        nonlocal calls
        calls+=1;return calls==15
    monkeypatch.setattr(module,'deadline_due',stop)
    with pytest.raises(SystemExit):module.run(args(tmp_path/'paused'))
    monkeypatch.setattr(module,'deadline_due',lambda:False)
    module.run(args(tmp_path/'paused'))
    a=torch.load(tmp_path/'complete/latest.pt',weights_only=False);b=torch.load(tmp_path/'paused/latest.pt',weights_only=False)
    for key in ('model','optimizer','rng','rows','curves','recycler'):_assert_bitwise(a[key],b[key],key)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_nearly_parallel_halfspaces(dtype):
    from methods.cl_projection import ConstraintSolver
    operator=ProjectionOperator(lambda d:d,lambda d:d,lambda d:d,1.,tolerance=1e-10)
    gradients=torch.tensor([[1.,0.],[1.,1e-6]],dtype=dtype)
    solution=ConstraintSolver(operator,tolerance=1e-9).solve(torch.tensor([-1.,1.],dtype=dtype),gradients)
    torch.testing.assert_close(solution.direction,torch.tensor([0.,1.],dtype=torch.double),atol=1e-9,rtol=1e-9)
    assert solution.slack.min()>=-1e-9 and solution.multipliers.min()>=0


def test_collector_groups_h3_arms_but_rejects_paired_data_mismatch(tmp_path):
    from experiments.standard_plasticity import parser,run
    from scripts.collect_standard_cl import collect
    for arm in ('bp','cbp','snp'):
        run(parser().parse_args(['--out',str(tmp_path/arm),'--arm',arm,'--fake-data']))
    result=collect([tmp_path]);assert len({json.dumps(r['protocol'],sort_keys=True) for r in result})==1
    assert all(r['metrics']['last_50_task_error']['paired_seeds']==1 for r in result)
    for name in ('protocol.json','summary.json'):
        path=tmp_path/'snp'/name;value=json.loads(path.read_text())
        (value if name=='protocol.json' else value['protocol'])['data_sha256']='different'
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='paired data'):collect([tmp_path])


def test_unified_cil_config_drives_real_optimizer_and_correction_loop(tmp_path):
    from experiments.standard_cl import parser,run
    args=parser().parse_args(['--config','configs/standard_cl/h1_cifar100_bn_unified.json','--data-dir',str(tmp_path),
        '--out',str(tmp_path/'unified'),'--arm','native','--tasks','2','--device','cpu','--fake-data'])
    result=run(args);assert result['status']=='COMPLETED'
    assert result['protocol']['mode']=='unified' and result['protocol']['config']['intervention_every']==1
    events=json.loads((tmp_path/'unified/interventions.json').read_text())
    assert len(events)==result['native_steps'] and all(e['mode']=='unified' for e in events)


def test_data_generation_does_not_select_plasticity_initialization(tmp_path,monkeypatch):
    from experiments import standard_plasticity as module
    def args(out):return module.parser().parse_args(['--out',str(out),'--arm','bp','--fake-data'])
    module.run(args(tmp_path/'original'));original=module.lop.prepare_data
    def consumes_rng(*a,**kw):
        torch.rand(17)
        return original(*a,**kw)
    monkeypatch.setattr(module.lop,'prepare_data',consumes_rng)
    module.run(args(tmp_path/'extra_rng'))
    a=torch.load(tmp_path/'original/latest.pt',weights_only=False);b=torch.load(tmp_path/'extra_rng/latest.pt',weights_only=False)
    for key in ('model','optimizer','rng','curves'):_assert_bitwise(a[key],b[key],key)


def test_h2_zero_penalty_has_bitwise_identical_physical_update():
    from methods.cl_h2 import GeometryCorrection
    from methods.cl_replay import ReplayCorrection,CorrectionConfig
    from methods.cl_state import State
    from experiments.vna_resume import rng_state
    class Net(nn.Module):
        def __init__(self):super().__init__();self.hidden=nn.Linear(2,3);self.norm=nn.BatchNorm1d(3);self.head=nn.Linear(3,2)
        def forward(self,x):
            f=self.norm(self.hidden(x)).tanh();return dict(features=f,logits=self.head(f))
    class Logits(nn.Module):
        def __init__(self,net):super().__init__();self.native=net
        def forward(self,x):return self.native(x)['logits']
    model=Logits(Net().double()).train();opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.01)
    x=torch.randn(6,2,dtype=torch.double);y=torch.tensor([0,1,0,1,0,1]);losses=ReplayLoss(y,torch.tensor([True,True,True,False,False,False]))
    opt.zero_grad();losses.total(model(x)).backward();opt.step();state=State(model,opt)
    cfg=CorrectionConfig(kernel='native',constrained=False,damping_relative=.5,cg_tolerance=1e-8)
    ReplayCorrection(cfg).step(model,opt,x,losses,[],.01)
    expected=deepcopy(model.state_dict());optimizer=deepcopy(opt.state_dict());rng=rng_state()
    state.restore();GeometryCorrection(cfg).step(model,opt,x,losses,[],.01,replay_inputs=x[:3],geometry_weight=0.)
    _assert_bitwise(expected,model.state_dict(),'lambda0.model');_assert_bitwise(optimizer,opt.state_dict(),'lambda0.optimizer')
    _assert_bitwise(rng,rng_state(),'lambda0.rng')
