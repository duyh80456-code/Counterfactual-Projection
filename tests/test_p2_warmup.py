"""Opt-in P2 warmup, Inc10, and shared native-loop integration."""
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


def test_actual_icarl_hook_and_competitor_loader_share_warmup_at_task_boundaries(monkeypatch):
    from types import SimpleNamespace
    from torch.utils.data import TensorDataset
    from adapters.icarl_hooks import ICaRLHook
    main.cil.upstream_imports()
    from models import icarl
    data=TensorDataset(torch.arange(3),torch.ones(3,2),torch.zeros(3,1))
    main_trace=[];intervention_lrs=[]
    class Recorder(ICaRLHook):
        warmup_epochs=1
        def before_optimizer_step(self,frame):
            main_trace.append((self.learner._cur_task+1,self.current_epoch*3+self.epoch_step,
                               self.optimizer.param_groups[0]['lr']))
        def after_epoch(self,learner,optimizer,frame):self.current_epoch=int(frame.f_locals['epoch'])+1
    intervention=SimpleNamespace(engine=SimpleNamespace(events=[]),iteration_times=[],
        start_task=lambda learner:None,
        after_step=lambda hook,learner,opt:intervention_lrs.append(opt.param_groups[0]['lr']))
    def incremental(self,manager):
        self._cur_task+=1
        self.train_loader=icarl.DataLoader(data,batch_size=1,shuffle=False,num_workers=0)
        self.test_loader=icarl.DataLoader(data,batch_size=1,shuffle=False,num_workers=0)
        self._train(self.train_loader,self.test_loader)
    def train(self,train_loader,test_loader):
        optimizer=torch.optim.SGD(self._network.parameters(),lr=.05)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1],gamma=.1)
        for epoch in icarl.tqdm(range(2)):
            for _,x,y in train_loader:
                optimizer.zero_grad();(self._network(x)-y).square().mean().backward();optimizer.step()
            scheduler.step()
    monkeypatch.setattr(icarl.iCaRL,'incremental_train',incremental)
    monkeypatch.setattr(icarl.iCaRL,'_train',train)
    learner=object.__new__(icarl.iCaRL)
    learner._network=torch.nn.Linear(2,1)
    learner._old_network=None;learner._cur_task=-1;learner._device=torch.device('cpu')
    with Recorder(intervention=intervention).install():
        learner.incremental_train(None);learner.incremental_train(None)
    assert intervention_lrs==[.05]*3+[.05*.1]*3+[.05]*3+[.05*.1]*3
    competitor_trace=[]
    args=competitors.parser().parse_args(['--method','icarl_cbp','--out','unused','--fake-data',
        '--increment','10','--warmup-epochs','1','--device','cpu'])
    cfg,_,_=competitors.recipe('icarl_cbp','nobn',0,'cpu',increment=10)
    with competitors.runtime(args,'icarl_cbp',cfg,lambda:None,lambda:None) as (module,_):
        loader=module.DataLoader(data,batch_size=1,shuffle=False,num_workers=0);loader._v4_training=True
        class Native:
            _cur_task=0
            def train(self,train_loader):
                model=torch.nn.Linear(2,1)
                optimizer=torch.optim.SGD(model.parameters(),lr=.05)
                scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[1],gamma=.1)
                optimizer.register_step_pre_hook(lambda opt,a,kw:competitor_trace.append(
                    (self._cur_task+1,len([r for r in competitor_trace if r[0]==self._cur_task+1])+1,opt.param_groups[0]['lr'])) and None)
                for epoch in range(2):
                    for _,x,y in train_loader:
                        optimizer.zero_grad();(model(x.to(model.weight))-y).square().mean().backward();optimizer.step()
                    scheduler.step()
        native=Native()
        for task in range(2):native._cur_task=task;native.train(loader)
    assert competitor_trace==main_trace
    assert [row for row in main_trace if row[1] in (1,3,4)]==[
        (task,step,lr) for task in (1,2) for step,lr in ((1,.05/3),(3,.05),(4,.05*.1))]
