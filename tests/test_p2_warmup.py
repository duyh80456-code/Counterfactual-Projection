"""Opt-in P2 warmup and Inc10; written without executing tests or training."""
import copy
from contextlib import nullcontext
import pytest
import torch
from diagnostics.vna_align import _assert_bitwise
from experiments.lr_warmup import task_warmup
from experiments import main_cil as main,v4_competitors as competitors


def test_old_defaults_and_inc10_keep_the_native_recipe():
    main_args=main.parser().parse_args(['--out','unused'])
    other_args=competitors.parser().parse_args(['--method','icarl_cbp','--out','unused'])
    assert (main_args.increment,main_args.warmup_epochs,main_args.gir_rule)==(5,0,'schur_dead')
    assert (other_args.increment,other_args.warmup_epochs)==(5,0)
    main.cil.upstream_imports()
    for increment in (5,10):
        cfg=main.recipe(0,'cpu','resnet32_nobn',increment)
        cbp,_,_=competitors.recipe('icarl_cbp','nobn',0,'cpu',increment=increment)
        assert cfg['init_cls']==cfg['increment']==cbp['init_cls']==cbp['increment']==increment
        assert cfg['memory_size']==cbp['memory_size']==2000
    assert main.training_settings()==dict(init_epoch=100,epochs=50,
        init_milestones=[30,60,85],milestones=[24,35],init_lr=.1,lrate=.1)


@pytest.mark.parametrize('task',[1,2])
def test_warmup_first_S_and_Splus1_follow_current_scheduled_lr(task,monkeypatch):
    messages=[];monkeypatch.setattr('experiments.lr_warmup.log',messages.append)
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64))
    opt=torch.optim.SGD([p],lr=.05,momentum=.9)
    observed=[]
    for step in range(1,6):
        scheduled=.005 if step==5 else .05
        opt.param_groups[0]['lr']=scheduled
        with task_warmup(opt,1,4,step,task):observed.append(opt.param_groups[0]['lr'])
        assert opt.param_groups[0]['lr']==scheduled
    assert observed==[.05*(1/4),.05*(2/4),.05*(3/4),.05,.005]
    assert len(messages)==3
    assert all(f'task={task}' in line for line in messages)


def test_disabled_warmup_leaves_native_sgd_loss_parameters_and_state_bitwise():
    torch.manual_seed(7)
    initial=torch.nn.Linear(3,2).double()
    x=torch.randn(4,3,dtype=torch.float64);target=torch.randn(4,2,dtype=torch.float64)
    outputs=[]
    for enabled in (False,True):
        net=copy.deepcopy(initial);opt=torch.optim.SGD(net.parameters(),lr=.05,momentum=.9,weight_decay=.001)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[1],gamma=.1)
        losses=[];lrs=[]
        for epoch in range(2):
            for batch in range(3):
                opt.zero_grad();loss=(net(x)-target).square().mean();loss.backward()
                before=opt.param_groups[0]['lr']
                with task_warmup(opt,0,3,epoch*3+batch+1,1) if enabled else nullcontext():
                    opt.step();lrs.append(opt.param_groups[0]['lr'])
                assert opt.param_groups[0]['lr']==before
                losses.append(loss.detach().clone())
            scheduler.step()
        outputs.append(dict(losses=losses,lrs=lrs,model=net.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict()))
    _assert_bitwise(*outputs)


def test_warmup_restores_scheduled_lr_when_update_raises():
    opt=torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],lr=.05)
    with pytest.raises(RuntimeError,match='interrupted'):
        with task_warmup(opt,1,4,1,2):
            assert opt.param_groups[0]['lr']==.05/4
            raise RuntimeError('interrupted')
    assert opt.param_groups[0]['lr']==.05


def test_competitor_loader_warms_each_task_and_ignores_nested_eval():
    from torch.utils.data import TensorDataset
    main.cil.upstream_imports()
    args=competitors.parser().parse_args(['--method','icarl_cbp','--out','unused','--fake-data',
        '--increment','10','--warmup-epochs','1','--device','cpu'])
    cfg,_,_=competitors.recipe('icarl_cbp','nobn',0,'cpu',increment=10)
    with competitors.runtime(args,'icarl_cbp',cfg,lambda:None,lambda:None) as (module,_):
        train_loader=module.DataLoader(TensorDataset(torch.arange(3)),batch_size=1)
        test_loader=module.DataLoader(TensorDataset(torch.arange(1)),batch_size=1)
        train_loader._v4_training=True
        class Learner:
            _cur_task=0
            def train(self,train_loader,optimizer):
                rows=[]
                for epoch in range(2):
                    for batch in train_loader:
                        rows.append(optimizer.param_groups[0]['lr'])
                        for item in test_loader:pass
                return rows
        learner=Learner()
        for task in range(2):
            learner._cur_task=task
            optimizer=torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],lr=.05)
            assert learner.train(train_loader,optimizer)==[.05/3,.05*(2/3),.05,.05,.05,.05]
            assert optimizer.param_groups[0]['lr']==.05


def test_main_resume_warmup_uses_completed_epoch_and_current_batch_index():
    # Main checkpoints contain completed epochs; no mutable warmup counter is needed.
    opt=torch.optim.SGD([torch.nn.Parameter(torch.ones(1))],lr=.05)
    records=[]
    for completed_epoch,current_batch in ((0,1),(0,4),(1,1)):
        step=completed_epoch*4+current_batch
        with task_warmup(opt,1,4,step,2):records.append(opt.param_groups[0]['lr'])
    assert records==[.05/4,.05,.05]
