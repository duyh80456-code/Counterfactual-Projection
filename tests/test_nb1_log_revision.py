from copy import deepcopy
import json
import pickle
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from diagnostics.vna_align import _assert_bitwise
from methods.vna_losses import ICaRLPyCIL
from methods.vna_sites import HookSite
from methods.vna_v3 import V3Engine, bounded_scale, FULL_POLICY_VERSION
from methods import vna_certswap


@pytest.fixture(autouse=True)
def isolated():
    threads = torch.get_num_threads()
    with torch.random.fork_rng():
        torch.manual_seed(29); torch.set_num_threads(1)
        yield
    torch.set_num_threads(threads)


def problem(bn=False, dead=False):
    model = nn.Sequential(nn.Linear(2,2), nn.BatchNorm1d(2) if bn else nn.Identity(),
                          nn.ReLU(), nn.Linear(2,3)).double().train()
    with torch.no_grad():
        model[0].weight.fill_(0. if dead else .2); model[0].bias.fill_(-1. if dead else 1.)
        model[3].weight.fill_(0.); model[3].bias.zero_()
        if bn: model[1].bias.fill_(-1. if dead else 1.)
    x = torch.tensor([[.1,.2],[.2,.4],[.3,.6],[.4,.8]],dtype=torch.float64)
    y = torch.ones(4,dtype=torch.long)
    batch = (x,y)
    ctx = dict(targets=y,known_classes=1,teacher_logits=torch.zeros(4,1,dtype=x.dtype))
    spec = ICaRLPyCIL(known_classes=1)
    site = HookSite(model,'0','3','linear',between=('1','2'))
    opt = torch.optim.SGD(model.parameters(),lr=.05,momentum=.9)
    for p in model.parameters():
        p.grad = torch.ones_like(p); opt.state[p]['momentum_buffer'] = torch.ones_like(p)
    engine = V3Engine('vna_gir',solver='gram',curvature='power10',full_revision=True); engine.begin_task(2)
    return model,site,batch,spec,ctx,opt,engine


@pytest.mark.parametrize('reason',['no_positive_directions','chol_fail','negative_certificate'])
@pytest.mark.parametrize('bn',[False,True])
def test_skipped_vna_does_not_run_gir(monkeypatch,reason,bn):
    model,site,batch,spec,ctx,opt,engine = problem(bn,dead=True)
    monkeypatch.setattr(engine,'_apply',lambda *a:dict(skipped=True,skip_reason=reason,swaps=0))
    before=engine._training_state(model,opt)
    monkeypatch.setattr(vna_certswap,'run',lambda *a,**k:pytest.fail('skipped VNA entered CertSwap'))
    monkeypatch.setattr(engine,'_run_gir',lambda *a,**k:pytest.fail('skipped VNA entered GIR'))
    row = engine.apply(model,[site],batch,spec,ctx,opt,1,force=True)
    assert engine.policy_version == FULL_POLICY_VERSION
    assert row['skip_reason'] == reason and row['swaps'] == 0 and 'gir_independent' not in row
    _assert_bitwise(before,engine._training_state(model,opt))
    assert not engine.install_ages


@pytest.mark.parametrize('after,expected',[(39.,False),(40.,False),(118.,True),(None,True)])
def test_vna_sharpness_gate_restores_bitwise(monkeypatch,after,expected):
    model,site,batch,spec,ctx,opt,engine = problem()
    before = engine._training_state(model,opt)
    values = iter([11.,after,11.])
    def stability(*a):
        return dict(lambda_max_ggn=next(values),method='power10_estimate')
    monkeypatch.setattr(engine,'_stability',stability)
    monkeypatch.setattr(engine,'_run_gir',lambda *a,**k:[])
    row = engine.apply(model,[site],batch,spec,ctx,opt,1,force=True)
    assert row['sharpness_rejected'] == expected
    assert row['sharpness_limit'] == 40. and row['lambda_max_ggn_vna_after'] == after
    assert row['step_norm'] <= row['Delta']
    if expected:
        assert row['rejected'] and row['s'] == 0.
        _assert_bitwise(before,engine._training_state(model,opt))
    else:
        assert not row['rejected'] and row['s'] > 0


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('magnitude',[1e-20,1e-12,1e-4,1.])
def test_large_scale_uses_actual_direction_norm(dtype,magnitude):
    d = torch.tensor([1.,2.,-3.],dtype=dtype)*magnitude
    scale,norm,step = bounded_scale(2.4e10,d,1e-3)
    assert 0 <= scale <= 2.4e10 and norm == float(d.double().norm())
    assert step <= 1e-3 and float((scale*d.double()).norm()) <= 1e-3


def test_full_revision_resume_and_zero_coefficient_identity(monkeypatch):
    model,site,batch,spec,ctx,opt,engine = problem(dead=True)
    before = engine._training_state(model,opt)
    zero = V3Engine('vna_gir',coefficient=0.,full_revision=True)
    assert zero.apply(model,[site],batch,spec,ctx,opt,1,force=True) is None
    _assert_bitwise(before,engine._training_state(model,opt))
    with pytest.raises(ValueError,match='policy'):
        engine.load_state_dict(V3Engine('vna_gir').state_dict())
    monkeypatch.setattr('time.perf_counter',lambda:0.)
    monkeypatch.setattr(engine,'_apply',lambda *a:dict(skipped=True,skip_reason='no_positive_directions',swaps=0))
    engine.apply(model,[site],batch,spec,ctx,opt,1,force=True)
    checkpoint=(engine._training_state(model,opt),engine.state_dict())
    engine.apply(model,[site],batch,spec,ctx,opt,2,force=True)
    expected=(engine._training_state(model,opt),engine.state_dict())
    engine._restore_training_state(model,opt,checkpoint[0]);engine.load_state_dict(checkpoint[1])
    engine.apply(model,[site],batch,spec,ctx,opt,2,force=True)
    _assert_bitwise(expected,(engine._training_state(model,opt),engine.state_dict()))


def test_accepted_vna_swap_keeps_post_intervention_followup(monkeypatch):
    from experiments import main_cil as main
    model,site,batch,spec,ctx,opt,engine = problem(dead=True)
    event=dict(skipped=False,skip_reason=None,s=.1,swaps=1)
    monkeypatch.setattr(engine,'apply',lambda *a,**k:event)
    monkeypatch.setattr(main,'loss_context',lambda *a:(spec,ctx))
    monkeypatch.setattr(main,'Prediction',lambda model:model)
    monkeypatch.setattr(main,'sites_for',lambda model:[site])
    learner=SimpleNamespace(_device=torch.device('cpu'),_total_classes=3,_network=model,_cur_task=1)
    hook=SimpleNamespace(steps=1,current_batch=batch)
    intervention=main.Intervention(engine,1)
    intervention.start_task(learner)
    intervention.after_step(hook,learner,opt)
    assert intervention.events==[event]
    assert len(intervention.followups)==1 and intervention.followups[0]['status']=='RUNNING'


@pytest.mark.parametrize('corruption',[None,'native_summary_only','source_digest','epochs','cfg','dataset','order','measurement','tasks','status',
                                      'protocol_identity','parameter_count','mu0'])
def test_reused_nb1_baseline_is_validated_without_rewriting(tmp_path,monkeypatch,corruption):
    from experiments import p1_compare as study, main_cil as main
    from experiments.parallel_validity import dataset_fingerprint
    from adapters.cil_setup import COARSE_NAMES,FINE_TO_COARSE
    data=tmp_path/'data'/'cifar-100-python';data.mkdir(parents=True)
    (data/'train').write_bytes(pickle.dumps(dict(fine_labels=list(range(100)),coarse_labels=FINE_TO_COARSE)))
    (data/'test').write_bytes(b'test fixture')
    (data/'meta').write_bytes(pickle.dumps(dict(coarse_label_names=COARSE_NAMES)))
    root=tmp_path/'baseline';(root/'run').mkdir(parents=True)
    cfg=main.recipe(0,'cuda:0','resnet32_nobn',20);cfg['shuffle']=False
    p=dict(arm='icarl',seed=0,backbone='resnet32_nobn',class_order_kind='superclass',
        code_commit='ccf60af0909afa8853357c660f8760c98cc1b828',
        source_digest='5e78fd3c7d62f86ef1d0cd3df67b2e0085feb75934bd739652305d361fc6e6f7',
        upstream_commit=main.cil.UPSTREAM_PIN,policy_version='schur_gamma_bound_rho25_v2',algorithm_version='v4',
        fake_data=False,expected_tasks=5,preflight_only=False,nobn_lr_scale=.5,coefficient=1.,
        solver='gram',curvature='power10',gir_live=False,init_cls=20,increment=20,cfg=cfg,
        dataset_fingerprint=dataset_fingerprint(data.parent),class_order=main.class_order(0,'superclass',data.parent),
        training=dict(first_epochs=100,later_epochs=50,first_milestones=[30,60,85],later_milestones=[24,35]))
    measurement={k:p[k] for k in ('code_commit','source_digest','arm','seed','backbone','increment')}
    measurement.update(selected_model='final_epoch',ggn=dict(definition='engine_stability_current_minibatch_v2'))
    tasks=[dict(task=t,cnn_top1=50.,nme_top1=50.,parameter_count=460000+20*t) for t in range(1,6)]
    summary=dict(status='COMPLETED',identity=p,tasks=tasks)
    if corruption=='source_digest':p['source_digest']='a'*64
    if corruption=='epochs':p['training']['later_epochs']=49
    if corruption=='cfg':p['cfg']['memory_size']=1999
    if corruption=='dataset':p['dataset_fingerprint']['test']='a'*64
    if corruption=='order':p['class_order'].reverse()
    if corruption=='measurement':measurement['selected_model']='best_epoch'
    if corruption=='tasks':tasks.pop()
    if corruption=='status':summary['status']='FAILED_DIVERGED'
    saved_protocol={**p,'parameter_count':460100,'mu0':None}
    if corruption=='protocol_identity':saved_protocol['seed']=1
    if corruption=='parameter_count':saved_protocol['parameter_count']=460099
    if corruption=='mu0':saved_protocol['mu0']=.1
    for name,value in (('summary.json',summary),('run/protocol.json',saved_protocol),('run/tasks.json',tasks),
                       ('measurement_protocol.json',measurement)):
        (root/name).write_text(json.dumps(value))
    (root/'run/latest.pt').write_bytes(b'original checkpoint stays immutable')
    if corruption=='native_summary_only':
        (root/'summary.json').replace(root/'run/summary.json')
    before={str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()}
    monkeypatch.setattr(main,'run',lambda *a:pytest.fail('baseline must not train'))
    if corruption not in (None,'native_summary_only'):
        with pytest.raises(ValueError):study.validate_nb1_baseline(root,0,data.parent)
    else:
        result=study.validate_nb1_baseline(root,0,data.parent)
        assert result['status']=='VALIDATED_ORIGINAL_BASELINE' and result['source_digest']==p['source_digest']
    assert before=={str(f):f.read_bytes() for f in root.rglob('*') if f.is_file()}


def test_missing_baseline_reports_the_restore_path(tmp_path):
    from experiments.p1_compare import validate_nb1_baseline
    with pytest.raises(FileNotFoundError,match='summary not restored: .*icarl__s0/summary.json'):
        validate_nb1_baseline(tmp_path/'icarl__s0',0,tmp_path/'data')
