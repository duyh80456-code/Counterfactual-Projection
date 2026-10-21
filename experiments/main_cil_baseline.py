"""Reuse the registered seed-0 baseline only after an exact epoch-loss replay."""
import hashlib
import json
import math
from pathlib import Path
import torch
from experiments.io_utils import atomic_json_save

OLD_BASELINE_COMMIT='d88abc496eb1fe21d1b6df66d90b47cf93d7ded1'


def prefix(epochs):
    selected=[r for r in epochs if r['task'] in (1,2)]
    rows={(r['task'],r['epoch']):r['train_loss'] for r in selected}
    keys=[(t,e) for t,n in ((1,100),(2,50)) for e in range(1,n+1)]
    if len(selected)!=150 or set(rows)!=set(keys) or any(not math.isfinite(rows[k]) for k in keys):
        raise ValueError('Need all 100+50 finite epoch losses')
    return [dict(task=t,epoch=e,train_loss=rows[t,e]) for t,e in keys]


def validate(reference,protocol):
    old=reference['protocol']
    for key in ('arm','seed','backbone','nobn_lr_scale','class_order_kind','class_order',
                'dataset_fingerprint','training','comparison_protocol','coefficient','upstream_commit'):
        if old.get(key)!=protocol.get(key):raise ValueError('baseline reference mismatch: '+key)
    if (old.get('code_commit')!=OLD_BASELINE_COMMIT or old.get('fake_data') or old.get('preflight_only')
            or reference['summary'].get('status')!='COMPLETED' or reference['summary'].get('tasks_run')!=20):
        raise ValueError('Need the completed registered d88abc4 baseline')
    from scripts.collect_parallel import validate_result
    validate_result('icarl__nobn__superclass__s0',reference['summary'],old,
                    reference['summary'].get('tasks'),OLD_BASELINE_COMMIT)
    prefix(reference['epochs'])


def prepare_reference(roots,destination):
    destination=Path(destination)
    if destination.exists():return destination
    from experiments.runners.pycil import upstream_imports
    upstream_imports()
    matches={}
    for root in roots:
        for path in Path(root).rglob('summary.json'):
            try:
                summary=json.loads(path.read_text());protocol=summary.get('identity',{})
                if (protocol.get('arm'),protocol.get('seed'),protocol.get('code_commit'))!=('icarl',0,OLD_BASELINE_COMMIT):continue
                if summary.get('status')!='COMPLETED' or summary.get('tasks_run')!=20:continue
                checkpoint=path.with_name('latest.pt')
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                if state.get('identity')!=protocol:raise ValueError('baseline checkpoint identity mismatch')
                reference=dict(protocol=protocol,summary=summary,epochs=prefix(state['hook']['epochs']),
                    source=str(path),summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                validate(reference,protocol)
                key=(reference['summary_sha256'],json.dumps(reference['epochs'],sort_keys=True))
                matches[key]=reference
            except (OSError,KeyError,ValueError,AttributeError,RuntimeError,ImportError) as error:
                print(f'Baseline reference rejected: {path}: {error}',flush=True)
    if len(matches)>1:raise ValueError('Conflicting completed old seed-0 baseline summaries')
    if not matches:return None
    atomic_json_save(next(iter(matches.values())),destination)
    return destination


def compare(reference,epochs):
    old,new=prefix(reference['epochs']),prefix(epochs)
    differences=[dict(task=a['task'],epoch=a['epoch'],reference=a['train_loss'],current=b['train_loss'])
                 for a,b in zip(old,new) if a['train_loss']!=b['train_loss']]
    return dict(status='PASS' if not differences else 'FAIL',epochs_compared=150,
        absolute_tolerance=0.,relative_tolerance=0.,differences=differences,
        reference_commit=reference['protocol']['code_commit'],
        reference_summary_sha256=reference['summary_sha256'])
