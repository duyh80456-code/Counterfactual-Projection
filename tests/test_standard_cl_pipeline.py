"""Focused CPU integration/resume contracts; no real-data training."""
from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from diagnostics.vna_align import _global_state,_assert_bitwise
from experiments.vna_resume import rng_state,restore_rng
from methods.cl_replay import ReplayLoss,ReplayCorrection,CorrectionConfig,_ChunkedSelector
from methods.cl_state import State,evaluate
from methods.cl_h3 import SwapCandidate,choose_swap,reset_sgd_coordinates
from methods.cl_h4 import Reservoir,DomainNetwork,DomainMNIST


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('mode',['intervention','unified'])
def test_native_h1_toy_finite_acceptance_and_resume(dtype,mode):
    with _global_state():
        torch.manual_seed(33)
        net=nn.Linear(2,2).to(dtype);opt=torch.optim.SGD(net.parameters(),lr=.01,momentum=.9)
        inputs=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[-1.,0.]],dtype=dtype);targets=torch.tensor([0,1,0,1])
        losses=ReplayLoss(targets,torch.tensor([True,False,True,False]))
        engine=ReplayCorrection(CorrectionConfig(kernel='native',mode=mode,cg_tolerance=1e-5 if dtype==torch.float32 else 1e-9))
        def step():
            opt.zero_grad();losses.total(net(inputs)).backward()
            if mode=='intervention':opt.step()
            before=evaluate(net,inputs,losses.components)
            event=engine.step(net,opt,inputs,losses,[],.01)
            assert event['status'] in ('ACCEPTED','REJECTED')
            assert all(a<=b for a,b in zip(evaluate(net,inputs,losses.components),before))
            return event
        step();checkpoint=State(net,opt);saved_engine=engine.state_dict()
        expected=step();expected_net=deepcopy(net.state_dict());expected_opt=deepcopy(opt.state_dict());expected_rng=rng_state()
        checkpoint.restore();engine.load_state_dict(saved_engine);actual=step()
        _assert_bitwise(expected_net,net.state_dict(),'resume.model')
        _assert_bitwise(expected_opt,opt.state_dict(),'resume.optimizer')
        _assert_bitwise(expected_rng,rng_state(),'resume.rng')
        for key in ('accepted','step','step_norm','before','after'):assert actual[key]==expected[key]


def test_er_native_optimizer_identity_without_correction():
    with _global_state():
        torch.manual_seed(8);first=nn.Sequential(nn.Linear(3,4),nn.BatchNorm1d(4),nn.Linear(4,3)).train()
        second=deepcopy(first);x=torch.randn(6,3);y=torch.tensor([0,1,2,0,1,2])
        one=torch.optim.SGD(first.parameters(),lr=.1,momentum=.9,weight_decay=.0002)
        two=torch.optim.SGD(second.parameters(),lr=.1,momentum=.9,weight_decay=.0002)
        one.zero_grad();torch.nn.functional.cross_entropy(first(x),y).backward();one.step()
        losses=ReplayLoss(y,y<2);two.zero_grad();losses.total(second(x)).backward();two.step()
        _assert_bitwise(first.state_dict(),second.state_dict(),'native.baseline')
        _assert_bitwise(one.state_dict(),two.state_dict(),'native.optimizer')


def test_serial_trace_matches_full_toy_extension():
    matrix=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],dtype=torch.double)
    p=SimpleNamespace(_call=lambda _:torch.zeros(3),extension_refs=[('b',torch.zeros(2))],
        inputs=torch.zeros(1,dtype=torch.double),JEv=lambda v:matrix@v,JET=lambda v:matrix.T@v)
    selector=_ChunkedSelector('vna',seed=0)
    assert selector._trace_extension(p)==float(matrix.square().sum())


def test_h3_dead_slot_preservation_live_jump_and_optimizer_reset():
    net=nn.Sequential(nn.Linear(1,2),nn.ReLU(),nn.Linear(2,1,bias=False)).double()
    with torch.no_grad():
        net[0].weight.fill_(1);net[0].bias.fill_(-2);net[2].weight.fill_(1)
    x=torch.tensor([[0.],[1.]],dtype=torch.double);old=net(x).detach().clone()
    opt=torch.optim.SGD(net.parameters(),lr=.1,momentum=.9)
    for p in net.parameters():opt.state[p]['momentum_buffer']=torch.ones_like(p)
    snapshot=State(net,opt)
    with torch.no_grad():net[0].weight[0].fill_(3);net[0].bias[0].zero_();net[2].weight[:,0].zero_()
    torch.testing.assert_close(net(x),old,rtol=0,atol=0)
    mask=torch.zeros_like(net[0].weight,dtype=torch.bool);mask[0]=True
    reset_sgd_coordinates(opt,[(net[0].weight,mask)])
    assert opt.state[net[0].weight]['momentum_buffer'][0].eq(0).all()
    snapshot.restore()
    with torch.no_grad():net[0].bias.fill_(1)
    live=net(x).detach().clone()
    with torch.no_grad():net[2].weight[:,0].zero_()
    assert not torch.equal(live,net(x))


def test_h3_identity_is_valid_and_structure_failures_restore():
    net=nn.Linear(1,1,bias=False).double();net.weight.data.zero_();opt=torch.optim.SGD(net.parameters(),lr=.1)
    x=torch.ones(2,1,dtype=torch.double)
    c=lambda z:z.square().mean();before=deepcopy(net.state_dict())
    def worsen(model):
        with torch.no_grad():model.weight.fill_(10)
    result=choose_swap(net,opt,x,(c,c),[SwapCandidate('worse','live',worsen,lambda *a:None)])
    assert result['choice']=='identity';_assert_bitwise(net.state_dict(),before,'swap.identity')
    def resize(model):model.weight=nn.Parameter(torch.zeros(2,1,dtype=torch.double))
    with pytest.raises(ValueError,match='fixed capacity'):
        choose_swap(net,opt,x,(c,c),[SwapCandidate('invalid','live',resize,lambda *a:None)])
    _assert_bitwise(net.state_dict(),before,'swap.structure.rollback')


def test_domain_fixed_head_and_reservoir_resume():
    with _global_state():
        memory=Reservoir(4,17);x=torch.arange(12*4).reshape(12,1,2,2).float();y=torch.arange(12)%10
        memory.add(x,y,0);state=deepcopy(memory.state_dict());expected=memory.sample(3,'cpu')
        memory.load_state_dict(state);actual=memory.sample(3,'cpu')
        torch.testing.assert_close(expected[0],actual[0]);torch.testing.assert_close(expected[1],actual[1])
        net=DomainNetwork(width=16);net.eval()
        assert net(torch.zeros(3,1,28,28)).shape==(3,10) and net.head.out_features==10


def test_result_collector_keeps_failures_and_rejects_protocol_mismatch(tmp_path):
    from scripts.collect_standard_cl import collect
    for arm,status in (('baseline','COMPLETED'),('h1','FAILED_RESOURCE_OOM')):
        root=tmp_path/arm;root.mkdir()
        p=dict(branch='H1',code_commit='same',seed=0,arm=arm,tasks=1)
        summary=dict(protocol=p,status=status,tasks=[{'task':1}] if status=='COMPLETED' else [],
            average_incremental_accuracy=10.,final_accuracy=10.,forgetting=0.,backward_transfer=0.,gpu_hours=1.)
        (root/'protocol.json').write_text(json.dumps(p));(root/'summary.json').write_text(json.dumps(summary))
    rows=collect([tmp_path]);failure=next(r for r in rows if r['arm']=='h1')
    assert failure['outcomes']=={'FAILED_RESOURCE_OOM':1} and failure['metrics']['final_accuracy']['mean'] is None
    (tmp_path/'h1/protocol.json').write_text('{}')
    with pytest.raises(ValueError,match='protocol'):collect([tmp_path])
