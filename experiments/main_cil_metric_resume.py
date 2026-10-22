"""Narrow checkpoint transfer for the 29cc735 metric-trace memory repair."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import torch
from diagnostics.vna_align import _global_state
from experiments.io_utils import atomic_json_save,atomic_torch_save
from methods.vna_v3 import POLICY_VERSION

SOURCE='29cc735ab8770b3602c54c3c1f157649d02f807a'


def _identity(value,commit,schedule):
    expected=dict(code_commit=SOURCE,source_digest=None,policy_version='schur_gamma_bound_rho25_v2',
        setting='icarl',algorithm_version='v4',arm='icarl_full',seed=0,solver='gram',curvature='power10',
        backbone='resnet32_nobn',class_order_kind='superclass',nobn_lr_scale=.5,coefficient=1.,
        fake_data=False,preflight_only=False,expected_tasks=20,gir_live=False)
    if POLICY_VERSION!=expected['policy_version'] or any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('metric-trace repair checkpoint identity mismatch')
    if value.get('schedule')!=schedule or value.get('K0')!=schedule['K0']:
        raise ValueError('metric-trace repair checkpoint/schedule mismatch')
    result=deepcopy(value);result.update(code_commit=commit,schedule=_schedule(schedule,commit))
    return result


def _schedule(value,commit):
    expected=dict(code_commit=SOURCE,source_digest=None,policy_version=POLICY_VERSION,gir_live=False,
        solver='gram',curvature='power10',backbone='resnet32_nobn',nobn_lr_scale=.5,authoritative=True)
    if any(value.get(k)!=v for k,v in expected.items()) or value.get('synthetic') or not isinstance(value.get('K0'),int) or value['K0']<1:
        raise ValueError('metric-trace repair schedule identity mismatch')
    return {**deepcopy(value),'code_commit':commit,'calibration_code_commit':SOURCE}


@_global_state()
def restore_metric_trace_run(stage,run_tag,commit):
    source=Path(stage)/run_tag/SOURCE[:12]
    target=Path(stage)/run_tag/commit[:12]
    old=source/'icarl_full';new=target/'icarl_full'
    if commit==SOURCE or not (old/'latest.pt').exists() or new.exists():return None
    schedule=json.loads((source/'_schedules'/'gram_power10.json').read_text())
    converted=_schedule(schedule,commit)
    staging=target/'icarl_full.metric-trace-import'
    if staging.exists():shutil.rmtree(staging)
    staging.parent.mkdir(parents=True,exist_ok=True)
    manifest=dict(source_commit=SOURCE,target_commit=commit,reason='bounded metric-trace temporary; unchanged frozen GN formula',
        source_schedule_sha256=hashlib.sha256((source/'_schedules'/'gram_power10.json').read_bytes()).hexdigest(),checkpoints=[])
    try:
        shutil.copytree(old,staging)
        for path in sorted(staging.glob('latest*.pt')):
            state=torch.load(path,map_location='cpu',weights_only=False)
            if state.get('completed',20)>=20:raise ValueError('Only unfinished metric-trace runs can transfer')
            state['identity']=_identity(state['identity'],commit,schedule)
            manifest['checkpoints'].append(dict(name=path.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                completed=state['completed'],epoch=state.get('epoch'),mid=state.get('mid')))
            artifacts=state.get('artifacts',{})
            if 'protocol.json' in artifacts:
                value=json.loads(artifacts['protocol.json'])
                artifacts['protocol.json']=json.dumps(_identity(value,commit,schedule),default=str).encode()
            state['metric_trace_repair']=deepcopy(manifest['checkpoints'][-1])|dict(source_commit=SOURCE,target_commit=commit)
            atomic_torch_save(state,path)
        protocol=staging/'protocol.json'
        if protocol.exists():atomic_json_save(_identity(json.loads(protocol.read_text()),commit,schedule),protocol)
        # Rebuild progress artifacts from the safe checkpoint, not the failed in-memory application.
        (staging/'summary.json').unlink(missing_ok=True)
        atomic_json_save(manifest,staging/'metric_trace_repair.json')
        destination=target/'_schedules'/'gram_power10.json'
        if destination.exists() and json.loads(destination.read_text())!=converted:
            raise ValueError('metric-trace repair destination schedule conflict')
        atomic_json_save(converted,destination)
        staging.replace(new)
    except BaseException:
        if staging.exists():shutil.rmtree(staging)
        raise
    print('METRIC TRACE REPAIR: restored native checkpoint',json.dumps(manifest),flush=True)
    return manifest
