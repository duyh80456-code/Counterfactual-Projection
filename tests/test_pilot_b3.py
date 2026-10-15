"""B3 native identities and quick as a subset of the full measurement path."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from adapters.icarl_hooks import ICaRLHook, Prediction, measurement_epoch
from adapters.pilot_icarl import ICaRLPilot
from diagnostics.icarl_measure import B3Measurement
from diagnostics.vna_align import _global_state, _assert_bitwise, MeasurementHook
from experiments.pilot_gir_vna import PilotEngine
from experiments.stage_b3 import upstream_imports
from experiments.vna_resume import rng_state, hook_state, restore_hook
from methods.vna_losses import CE
from methods.vna_sites import HookSite


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    with _global_state(): yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('options', [dict(coefficient=0.),dict(gir=True)])
def test_icarl_zero_lambda_or_no_dormant_bitwise(monkeypatch, options):
    upstream_imports()
    from tests import test_vna_stage_b3 as native
    import adapters.pilot_icarl as pilot
    actual_stats = pilot.activity_stats
    def no_dormant(*args, **kw):
        stats = actual_stats(*args, **kw)
        for row in stats.values():
            row['dormant_indices'] = row['dormant_indices'][:0]; row['dormant_fraction'] = 0.
        return stats
    monkeypatch.setattr(pilot,'activity_stats',no_dormant)
    results = []
    for enabled in (False, True):
        torch.manual_seed(0); np.random.seed(0)
        with monkeypatch.context() as patch:
            if enabled:
                patch.setattr(native,'ICaRLHook',lambda *a, **kw: ICaRLHook(*a, **kw,
                    intervention=ICaRLPilot(PilotEngine(**options), window=4)))
            result, dm = native.tiny_run(patch, True)
            results.append(dict(state=result, rng=rng_state()))
    _assert_bitwise(*results)


def tiny_measure(profile, first=True):
    class Native(nn.Sequential):
        def forward(self,x): return {'logits':super().forward(x).mean((2,3))}
    torch.manual_seed(31)
    model = Prediction(Native(nn.Conv2d(2,3,1),nn.ReLU(),nn.Conv2d(3,2,1)).double())
    batch = torch.randn(6,2,2,2,dtype=torch.float64), torch.arange(6)%2
    def sites(m): return [HookSite(m,'native.0','native.2','conv',between=('native.1',))]
    measure = B3Measurement(dict(seed=11), measure_profile=profile, sites_fn=sites,
        context_fn=lambda ctx,b: (CE(),dict(targets=b[1],known_classes=0)), null_repeats=1, groups=False)
    # Explicitly equal small configuration: no separate numerical quick path.
    measure.mus=[.01]; measure.ranks=(1,); measure.site_names=None
    ctx = dict(task=2, epoch=1, step=2, end_of_task=False,train_acc=.5,known_classes=0,
               targets=batch[1],first_measurement_point=first)
    rows = MeasurementHook(measure,seed=11)(model,batch,CE(),ctx)
    return [{k:v for k,v in r.items() if k not in
        ('measure_profile','wall_seconds','M_build_seconds','factor_seconds','tiny_V_peak_cuda_bytes')} for r in rows]


def test_quick_equals_full_given_identical_small_configuration():
    _assert_bitwise(tiny_measure('full'),tiny_measure('quick'))


def test_quick_configuration_and_schedule():
    full,quick = B3Measurement(dict(seed=0)),B3Measurement(dict(seed=0),measure_profile='quick')
    assert full.ranks==(4,8,16) and full.mus==[.001,.01]
    assert quick.ranks==(16,) and quick.mus==[.01] and quick.null_repeats==8
    assert quick.site_names==('stage_1.4','stage_2.4','stage_3.4')
    assert [(t,e) for t in range(1,11) for e in (85,170) if measurement_epoch(t,e,170,'quick')]==[(2,85),(5,85),(10,85)]
    assert ICaRLHook(measure_profile='quick').buffer.m==128
    assert ICaRLHook().buffer.m==512


def test_quick_gate_never_overwrites_official(tmp_path):
    from experiments.pilot_b3 import parser, run
    official = b'{"official":"preserve exact bytes"}\n'
    (tmp_path/'gate.json').write_bytes(official)
    args = parser().parse_args(['--arm','icarl','--smoke','--fake-data','--no-measure',
                               '--device','cpu','--workers','0','--out',str(tmp_path)])
    run(args)
    assert (tmp_path/'gate.json').read_bytes()==official
    quick = json.loads((tmp_path/'gate_quick.json').read_text())
    assert quick['status_label']=='exploratory'
    assert quick['by_mu']['0.01']['reproduction']['status']=='INVALID'
    assert not list(tmp_path.glob('gate_mu_*.json'))
    summary_before = (tmp_path/'summary.json').read_bytes()
    run(args)
    assert (tmp_path/'summary.json').read_bytes() == summary_before
    assert (tmp_path/'gate.json').read_bytes() == official


def test_intervention_checkpoint_roundtrip_private_rng():
    engine = PilotEngine(gir=True,random=True)
    engine.generator(torch.device('cpu'))
    pilot = ICaRLPilot(engine,window=4)
    pilot.batch = (torch.randn(4,2),torch.zeros(4,dtype=torch.long))
    pilot.measured_tasks.add(1)
    hook = ICaRLHook(intervention=pilot,measure_profile='quick')
    hook.buffer.update(pilot.batch)
    state = hook_state(hook)
    restored = ICaRLHook(intervention=ICaRLPilot(PilotEngine()),measure_profile='quick')
    restore_hook(restored,state)
    _assert_bitwise(hook.intervention.state_dict(),restored.intervention.state_dict())


def test_quick_verifies_d_and_cg_only_at_first_point(monkeypatch):
    import diagnostics.icarl_measure as measurement
    import diagnostics.measure_efficiency as efficiency
    calls=[]
    original_d, original_cg = measurement.joint_correction, efficiency.exact_cg_check
    def d(*a, **kw):
        calls.append('d'); return original_d(*a, **kw)
    def cg(*a, **kw):
        calls.append('cg'); return original_cg(*a, **kw)
    monkeypatch.setattr(measurement,'joint_correction',d)
    monkeypatch.setattr(efficiency,'exact_cg_check',cg)
    tiny_measure('quick',True)
    assert sorted(calls)==['cg','d']
    calls.clear()
    tiny_measure('quick',False)
    assert calls==[]


def test_gir_resets_selected_conv_and_bn_channels(monkeypatch):
    import adapters.pilot_icarl as pilot
    class Native(nn.Sequential):
        def forward(self,x): return {'logits':super().forward(x).mean((2,3))}
    native = Native(nn.Conv2d(2,2,1,bias=False),nn.BatchNorm2d(2),nn.ReLU(),nn.Conv2d(2,2,1)).double()
    def sites(model): return [HookSite(model,'native.0','native.3','conv',between=('native.1','native.2'))]
    monkeypatch.setattr(pilot,'sites_for',sites)
    def stats(model,batch,selected):
        return {selected[0].module_name:dict(dormant_indices=torch.tensor([0]),dormant_fraction=.5,effective_rank=1.)}
    monkeypatch.setattr(pilot,'activity_stats',stats)
    monkeypatch.setattr(pilot,'loss_context',lambda learner,batch:(CE(),dict(targets=batch[1])))
    dataset=[(i,torch.randn(2,2,2,dtype=torch.float64),i%2) for i in range(4)]
    learner=SimpleNamespace(_network=native,_cur_task=1,train_loader=SimpleNamespace(dataset=dataset))
    class Loader:
        def __len__(self): return 1
    learner.train_loader=Loader(); learner.train_loader.dataset=dataset
    native[1].running_mean.fill_(3.); native[1].running_var.fill_(4.)
    other=native[0].weight[1].detach().clone()
    engine=PilotEngine(gir=True,pool=2)
    intervention=ICaRLPilot(engine,window=4)
    intervention.start_task(learner)
    assert torch.count_nonzero(native[3].weight[:,0])==0
    torch.testing.assert_close(native[0].weight[1],other,rtol=0,atol=0)
    assert native[1].running_mean.tolist()==[0.,3.]
    assert native[1].running_var.tolist()==[1.,4.]
    assert float(native[0].weight[0].norm())==pytest.approx(2**.5,rel=1e-12)
    assert next(r for r in engine.events if r['kind']=='gir_replace')['replaced']==1


@pytest.mark.parametrize('schedule', [dict(K=100,synthetic=True,authoritative=True),
                                      dict(K=100,synthetic=False,authoritative=False)])
def test_real_schedule_rejects_synthetic_or_nonshared_calibration(tmp_path,schedule):
    from experiments.pilot_b3 import parser, run
    path=tmp_path/'schedule.json'; path.write_text(json.dumps(schedule))
    args=parser().parse_args(['--arm','icarl_ngd','--schedule',str(path)])
    with pytest.raises(ValueError): run(args)


def test_forgetting_includes_current_accuracy_like_native_pycl():
    hook=ICaRLHook()
    hook.epochs=[dict(train_loss=0.,train_acc=.5)]
    learner=SimpleNamespace(_cur_task=0,_data_memory=[],samples_per_class=200)
    first=dict(top1=50.,grouped={'00-09':50.})
    hook.record_task(learner,(first,first))
    learner._cur_task=1
    cnn=dict(top1=45.,grouped={'00-09':40.,'10-19':50.})
    nme=dict(top1=50.,grouped={'00-09':60.,'10-19':40.})
    hook.record_task(learner,(cnn,nme))
    assert hook.tasks[-1]['cnn_forgetting']==10.
    assert hook.tasks[-1]['nme_forgetting']==0.
    assert hook.tasks[-1]['cnn_average_accuracy']==47.5
