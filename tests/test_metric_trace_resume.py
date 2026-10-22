"""Only metadata may change when transferring the known memory-failure run."""
from copy import deepcopy
import hashlib
import json
import pytest
import torch
from diagnostics.vna_align import _assert_bitwise
from experiments.vna_resume import rng_state,Progress
from experiments.main_cil_metric_resume import SOURCE,MEMORY_SOURCE,restore_metric_trace_run


def fixture(stage,source_commit=SOURCE):
    root=stage/'run'/source_commit[:12];directory=root/'icarl_full';directory.mkdir(parents=True)
    schedule=dict(code_commit=source_commit,source_digest=None,policy_version='schur_gamma_bound_rho25_v2',
        gir_live=False,solver='gram',curvature='power10',backbone='resnet32_nobn',nobn_lr_scale=.5,
        authoritative=True,synthetic=False,K0=40,calibration=[dict(task=2,vna_seconds=1.)]*3)
    if source_commit==MEMORY_SOURCE:schedule['calibration_code_commit']=SOURCE
    (root/'_schedules').mkdir();(root/'_schedules'/'gram_power10.json').write_text(json.dumps(schedule))
    identity=dict(code_commit=source_commit,source_digest=None,policy_version=schedule['policy_version'],
        setting='icarl',algorithm_version='v4',arm='icarl_full',seed=0,solver='gram',curvature='power10',
        backbone='resnet32_nobn',class_order_kind='superclass',nobn_lr_scale=.5,coefficient=1.,
        fake_data=False,preflight_only=False,expected_tasks=20,gir_live=False,schedule=schedule,K0=40,
        dataset_fingerprint=dict(train='sha'),class_order=list(range(100)))
    state=dict(identity=identity,completed=19,epoch=37,mid=True,rng=rng_state(),
        learner=dict(weights=torch.arange(7.),teacher=torch.arange(4.),memory=torch.arange(9.)),
        optimizer=dict(momentum=torch.arange(7.)*.2),scheduler=dict(last_epoch=37),
        hook=dict(steps=1990,intervention_state=dict(mu=.7,A=torch.arange(3.),followups=[dict(samples=[1.,2.])])),
        artifacts={'protocol.json':json.dumps({**identity,'parameter_count':7,'mu0':.03}).encode(),
                   'epochs.json':b'[{"task": 19}]'})
    for name in ('latest.pt','latest.previous.pt'):torch.save(state,directory/name)
    (directory/'protocol.json').write_bytes(state['artifacts']['protocol.json'])
    (directory/'summary.json').write_text(json.dumps(dict(status='PAUSED',identity=identity)))
    return directory,state


@pytest.mark.parametrize('source_commit',[SOURCE,MEMORY_SOURCE])
def test_known_memory_run_transfer_preserves_all_native_training_state(tmp_path,source_commit):
    source,state=fixture(tmp_path,source_commit);commit='b'*40
    digest=hashlib.sha256((source/'latest.pt').read_bytes()).hexdigest();before=rng_state()
    manifest=restore_metric_trace_run(tmp_path,'run',commit)
    target=tmp_path/'run'/commit[:12]/'icarl_full'
    result=torch.load(target/'latest.pt',weights_only=False)
    for key in state.keys()-{'identity','artifacts'}:_assert_bitwise(state[key],result[key])
    _assert_bitwise(before,rng_state())
    expected=deepcopy(state['identity']);expected['code_commit']=commit
    expected['schedule']={**expected['schedule'],'code_commit':commit,'calibration_code_commit':SOURCE}
    assert result['identity']==expected and manifest['source_commit']==source_commit
    assert Progress(target,expected).state['completed']==19
    assert not (target/'summary.json').exists()
    assert next(r for r in manifest['checkpoints'] if r['name']=='latest.pt')['source_sha256']==digest
    assert hashlib.sha256((source/'latest.pt').read_bytes()).hexdigest()==digest
    assert result['artifacts']['epochs.json']==state['artifacts']['epochs.json']
    assert json.loads(result['artifacts']['protocol.json'])['code_commit']==commit
    assert json.loads((target.parent/'_schedules'/'gram_power10.json').read_text())==expected['schedule']
    assert restore_metric_trace_run(tmp_path,'run',commit) is None


@pytest.mark.parametrize('field,value',[('solver','cg_ir'),('seed',1),('policy_version','older'),('K0',41)])
def test_memory_transfer_rejects_wrong_identity_and_keeps_source(tmp_path,field,value):
    source,state=fixture(tmp_path);state['identity'][field]=value;torch.save(state,source/'latest.pt')
    before=(source/'latest.pt').read_bytes()
    with pytest.raises(ValueError,match='mismatch'):restore_metric_trace_run(tmp_path,'run','b'*40)
    assert (source/'latest.pt').read_bytes()==before
    assert not (tmp_path/'run'/('b'*12)/'icarl_full').exists()


@pytest.mark.parametrize('advanced',[SOURCE,MEMORY_SOURCE])
def test_repair_transfer_selects_most_advanced_registered_checkpoint(tmp_path,advanced):
    for source in (SOURCE,MEMORY_SOURCE):
        directory,state=fixture(tmp_path,source)
        if source==advanced:
            state['hook']['steps']+=10;torch.save(state,directory/'latest.pt')
    manifest=restore_metric_trace_run(tmp_path,'run','b'*40)
    assert manifest['source_commit']==advanced
    result=torch.load(tmp_path/'run'/('b'*12)/'icarl_full'/'latest.pt',weights_only=False)
    assert result['hook']['steps']==2000
