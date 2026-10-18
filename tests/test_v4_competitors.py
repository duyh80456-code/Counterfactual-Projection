import copy
import importlib.util
import json
import time
import numpy as np
import pytest
import torch
from torch import nn
from adapters.v4_competitors import dormant_score, recycle, CompetitorHook, shrink_perturb
from experiments import v4_competitors as v4
from experiments.v4_competitor_setup import SUPERCLASSES, class_order, skipinit
from diagnostics.v4_competitors import diagnose


def args(out, method='icarl', *extra):
    return v4.parser().parse_args(['--method',method,'--out',str(out),'--fake-data',*extra])


def state(out):
    return torch.load(out/'latest.pt',weights_only=False)['model']._network.state_dict()


@pytest.fixture(scope='module')
def baseline(tmp_path_factory):
    out = tmp_path_factory.mktemp('v4_baseline')
    result = v4.run(args(out))
    return result,state(out)


@pytest.mark.parametrize('method',v4.ROUND2)
def test_77_native_wrapper_and_disabled_identity(tmp_path,method,baseline):
    started = time.monotonic()
    disabled = args(tmp_path/'off',method,'--disabled')
    result = v4.run(disabled)
    assert result['tasks'] == baseline[0]['tasks']
    assert all(torch.equal(value,state(tmp_path/'off')[key]) for key,value in baseline[1].items())
    if method == 'gem' and importlib.util.find_spec('quadprog') is None:
        with pytest.raises(RuntimeError,match='GEM requires quadprog'): v4.run(args(tmp_path/'on',method))
        pytest.skip('GEM native two-task run blocked: quadprog absent; disabled identity passed')
    result = v4.run(args(tmp_path/'on',method))
    assert result['status'] == 'COMPLETED'
    assert result['tasks_run'] == 2
    for name in ('summary.json','tasks.json','continual_eval.csv','diagnostics.json','protocol.json','status.json'):
        assert (tmp_path/'on'/name).is_file()
    assert len(json.loads((tmp_path/'on/diagnostics.json').read_text())) == 2
    assert result['parameter_count'] > 0
    assert time.monotonic()-started < 60


def test_order_and_skipinit():
    assert sorted(sum((list(g) for g in SUPERCLASSES),[])) == list(range(100))
    order = class_order(0,'superclass')
    assert order != class_order(1,'superclass')
    assert all(set(order[i:i+5]) in [set(g) for g in SUPERCLASSES] for i in range(0,100,5))
    v4.cil.upstream_imports()
    from convs.cifar_resnet import resnet32
    net = skipinit(resnet32()).double()
    x = torch.randn(2,3,32,32,dtype=torch.float64)
    net(x)['features'].square().mean().backward()
    assert not any(isinstance(m,nn.BatchNorm2d) for m in net.modules())
    assert all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in net.parameters())


def test_redo_score_reset_and_momentum():
    v4.cil.upstream_imports()
    from convs.cifar_resnet import ResNetBasicblock
    block = ResNetBasicblock(2,2).double()
    optimizer = torch.optim.SGD(block.parameters(),lr=.1,momentum=.9)
    for p in block.parameters(): optimizer.state[p]['momentum_buffer'] = torch.ones_like(p)
    activation = torch.tensor([[[[0.,0.]],[[1.,3.]]]],dtype=torch.float64)
    torch.testing.assert_close(dormant_score(activation),torch.tensor([0.,2.],dtype=torch.float64))
    first, second = block.conv_a.weight.clone(),block.conv_b.weight.clone()
    assert recycle(block,activation,optimizer,.025) == 1
    assert not torch.equal(first[0],block.conv_a.weight[0])
    assert torch.equal(first[1],block.conv_a.weight[1])
    assert torch.equal(second[:,1],block.conv_b.weight[:,1])
    assert not torch.count_nonzero(block.conv_b.weight[:,0])
    assert not torch.count_nonzero(optimizer.state[block.conv_a.weight]['momentum_buffer'][0])
    assert not torch.count_nonzero(optimizer.state[block.conv_b.weight]['momentum_buffer'][:,0])


def test_cbp_calls_original_and_preserves_ages(tmp_path):
    v4.cil.upstream_imports()
    cfg,_,_ = v4.recipe('icarl_cbp','nobn',0,'cpu')
    cfg['device'] = [torch.device('cpu')]
    from types import SimpleNamespace
    from models import icarl
    from convs.cifar_resnet import ResNetBasicblock
    from experiments.v4_competitor_setup import FixtureBackbone
    net = nn.Sequential(FixtureBackbone(ResNetBasicblock))
    learner = SimpleNamespace(_network=skipinit(net).double(),args=cfg)
    hook = CompetitorHook('icarl_cbp',dict(replacement_rate=1e-3))
    with hook.install(learner,icarl):
        learner._network(torch.randn(2,3,32,32,dtype=torch.float64))
        for _,gnt in hook.generators:
            assert type(gnt).__module__ == 'lop.algos.res_gnt'
            gnt.ages[0].fill_(101)
            gnt.accumulated_num_features_to_replace[0] = .999
        opt = icarl.optim.SGD(learner._network.parameters(),lr=.01)
        opt.step()
        assert all((gnt.ages[0] == 0).any() for _,gnt in hook.generators)
    assert len(learner._v4_cbp) == 1
    next_hook = CompetitorHook('icarl_cbp',dict(replacement_rate=1e-3))
    with next_hook.install(learner,icarl):
        assert all(torch.equal(gnt.ages[0],saved['ages'][0]) for (_,gnt),saved in zip(next_hook.generators,learner._v4_cbp))


def test_diagnose_preserves_parameters_buffers_gradients_and_rng(tmp_path):
    v4.cil.upstream_imports()
    from convs.cifar_resnet import ResNetBasicblock
    from experiments.v4_competitor_setup import FixtureBackbone
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.convnet = FixtureBackbone(ResNetBasicblock)
            self.fc = nn.Linear(2,5)
        def forward(self,x): return dict(logits=self.fc(self.convnet(x)['features']))
    net = Model().double().train()
    batch = torch.randn(3,3,4,4,dtype=torch.float64),torch.tensor([0,1,2])
    for p in net.parameters(): p.grad = torch.ones_like(p)
    before = copy.deepcopy(net.state_dict())
    rng = torch.get_rng_state().clone()
    row = diagnose(net,batch)
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in before.items())
    assert torch.equal(rng,torch.get_rng_state())
    assert net.training and all(torch.equal(p.grad,torch.ones_like(p)) for p in net.parameters())
    assert 0 <= row['reach'] <= 1.001


def test_task_restart_exit75_is_bitwise_exact(tmp_path,monkeypatch):
    full = v4.run(args(tmp_path/'full','icarl_cbp'))
    calls = 0
    def pause(checkpoint):
        nonlocal calls
        calls += 1
        if calls == 4:
            checkpoint()
            raise SystemExit(75)
    with monkeypatch.context() as patch:
        patch.setattr(v4,'pause_if_due',pause)
        with pytest.raises(SystemExit) as exc: v4.run(args(tmp_path/'resume','icarl_cbp'))
        assert exc.value.code == 75
    result = v4.run(args(tmp_path/'resume','icarl_cbp','--resume'))
    assert result['tasks'] == full['tasks']
    assert all(torch.equal(v,state(tmp_path/'resume')[k]) for k,v in state(tmp_path/'full').items())


def test_grid_and_selection_provenance(tmp_path,monkeypatch):
    assert [len(v4.grid(m)) for m in ('icarl_cbp','icarl_redo','icarl_snp','icarl_upgd')] == [4,6,4,9]
    seen = []
    def run(trial):
        seen.append(trial)
        return dict(status='COMPLETED',average_incremental_accuracy=10.)
    monkeypatch.setattr(v4,'run',run)
    trial = args(tmp_path,'icarl_cbp')
    best = v4.sweep(trial)
    assert best['config'] == dict(replacement_rate=1e-6)
    assert all((t.seed,t.backbone,t.order) == (0,'nobn','superclass') for t in seen)
    trial.best_config = str(tmp_path/'best_config.json')
    trial.fake_data = False
    with pytest.raises(ValueError,match='provenance'): v4.matched(trial)


def test_v4_collector_validates_real_provenance_and_rejects_fixtures():
    from experiments.parallel_validity import validate_record
    from scripts.collect_parallel import validate_result
    cfg,source,overrides = v4.recipe('wa','bn',0,'cpu')
    protocol = dict(benchmark='v4_competitor',arm='wa',effective_method='wa',backbone='bn',
        order='superclass',seed=0,code_commit='a'*40,cfg=cfg,cfg_source=source,
        config_overrides=overrides,upstream_commit=v4.cil.UPSTREAM_PIN,competitor_pins=v4.PINS,
        method_config={},implementation=dict(source='PyCIL/models/wa.py',reimplemented=False),
        fake_data=False,disabled=False,epochs='100/50',expected_tasks=20,dataset_fingerprint={n:'b'*64 for n in ('train','test','meta')})
    tasks = [dict(task=i,cnn_per_class=[10.]*(5*i),cnn_top1=10.,cnn_new_accuracy=10.,
                  cnn_average_accuracy=10.,cnn_forgetting=0.,learning_accuracy=10.) for i in range(1,21)]
    summary = dict(status='COMPLETED',identity=protocol,tasks=tasks,parameter_count=100,gpu_hours=1.)
    job = 'wa__bn__superclass__s0'
    assert validate_result(job,summary,protocol,tasks)[1] == ('v4_competitor','wa','bn','superclass')
    bad = copy.deepcopy(protocol); bad['fake_data'] = True
    with pytest.raises(ValueError,match='synthetic'): validate_record(job,summary,bad,tasks)
    bad = copy.deepcopy(protocol); bad['cfg']['memory_size'] = 40
    with pytest.raises(ValueError,match='recipe'): validate_record(job,summary,bad,tasks)


def test_b5_width_grid_and_native_recipe(tmp_path):
    from experiments.v4_b5_width import jobs
    from experiments.pilot_v3 import lop_recipe
    trials = jobs(tmp_path,tmp_path/'schedules',fake=True)
    assert len(trials) == 36
    assert {(t.width,t.arm,t.seed) for t in trials} == {
        (width,'b5_'+arm,seed) for width in (5,20,100)
        for arm in ('bp','bp_vna','bp_gir','bp_vna_gir') for seed in (0,1,2)}
    for width in (5,20,100):
        trial = next(t for t in trials if t.width == width)
        _,cfg,_ = lop_recipe(trial)
        assert cfg['num_features'] == width


def test_intervention_nonfinite_marks_run_diverged(tmp_path,monkeypatch):
    from adapters import v4_competitors as adapter
    def bad_reset(net,*a,**kw):
        with torch.no_grad(): next(net.parameters()).fill_(float('nan'))
    monkeypatch.setattr(adapter,'shrink_perturb',bad_reset)
    result = v4.run(args(tmp_path,'icarl_snp'))
    assert result['status'] == 'DIVERGED'
    assert result['nonfinite_params'] and result['step'] == 0
    assert json.loads((tmp_path/'status.json').read_text())['phase'] == 'DIVERGED'


def test_l2_init_keeps_old_classifier_initialization(tmp_path):
    result = v4.run(args(tmp_path,'icarl_l2init','--config','{"coefficient":0.01}'))
    assert result['status'] == 'COMPLETED'
    first = torch.load(tmp_path/'latest.previous.pt',weights_only=False)['model']._v4_initial
    last = torch.load(tmp_path/'latest.pt',weights_only=False)['model']._v4_initial
    assert first['fc.weight'].shape[0] == 5 and last['fc.weight'].shape[0] == 10
    assert torch.equal(first['fc.weight'],last['fc.weight'][:5])


def test_superclass_reads_and_verifies_official_pickle(tmp_path):
    import pickle
    from experiments.v4_competitor_setup import COARSE_NAMES
    directory = tmp_path/'cifar-100-python'; directory.mkdir()
    coarse = [next(i for i,g in enumerate(SUPERCLASSES) if fine in g) for fine in range(100)]
    (directory/'train').write_bytes(pickle.dumps(dict(fine_labels=list(range(100)),coarse_labels=coarse)))
    (directory/'meta').write_bytes(pickle.dumps(dict(coarse_label_names=COARSE_NAMES)))
    assert class_order(0,'superclass',tmp_path) == class_order(0,'superclass')
    (directory/'meta').write_bytes(pickle.dumps(dict(coarse_label_names=list(reversed(COARSE_NAMES)))))
    with pytest.raises(ValueError,match='coarse label names'): class_order(0,'superclass',tmp_path)


def test_cflat_original_protocol_fixture(tmp_path):
    result = v4.run(args(tmp_path,'icarl_cflat','--reproduction','--backbone','bn','--order','random','--seed','1993'))
    assert result['status'] == 'COMPLETED'
    assert result['identity']['cfg']['shuffle'] is True
    assert result['identity']['initialization_seed'] == 1
    assert result['reproduction']['value'] == 59.13
    assert result['reproduction']['status'] == 'INVALID'
    assert 'delta_cnn_pp' in result['reproduction'] and 'delta_nme_pp' in result['reproduction']
