"""Stage 2a arm restore and immutable hourly archives."""
from pathlib import Path
import os
import shutil
import tarfile
import tempfile
from experiments.kaggle_checkpoint_discovery import discover_checkpoints
from experiments.shared_protocol import atomic_json_save, validate_resume_protocol


def restore_training(input_root, output, protocols):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    cache=output/'_resume_cache'
    def identify(payload):
        for name,protocol in protocols.items():
            if payload.get('protocol')==protocol: return name
        raise ValueError('no compatible Stage 2a arm protocol')
    try:
        found,rejected=discover_checkpoints(input_root,cache,kind='vna_training',
            payload_filter=lambda p: bool(identify(p)),
            payload_transform=lambda p: {'name':identify(p),'epoch':p['epoch']})
        import torch
        for name,protocol in protocols.items():
            target=output/name/'latest.pt'; candidates=[i for i in found if i['payload']['name']==name]
            if target.exists():
                cp=torch.load(target,map_location='cpu',weights_only=False)
                validate_resume_protocol(cp['protocol'],protocol)
                local_epoch=cp['epoch']
            else: local_epoch=-1
            if candidates:
                latest=max(candidates,key=lambda i:i['payload']['epoch'])
                if latest['payload']['epoch']>local_epoch:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    temporary=target.with_suffix('.pt.tmp'); shutil.copy2(latest['path'],temporary); temporary.replace(target)
            if target.exists():
                cp=torch.load(target,map_location='cpu',weights_only=False)
                atomic_json_save(cp['history'],target.parent/'history.json')
                if cp['final'] is not None: atomic_json_save(cp['final'],target.parent/'final.json')
                else: (target.parent/'final.json').unlink(missing_ok=True)
        print('Restored candidates:',len(found),'excluded:',len(rejected))
    finally: shutil.rmtree(cache,ignore_errors=True)


def archive_training(output,budget=5_000_000_000):
    output=Path(output); target=output.with_suffix('.tar.gz')
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='vna_training_snapshot_') as folder:
        snapshot=Path(folder)
        for path in output.rglob('*'):
            if path.is_file() and not path.name.endswith('.tmp') and '_resume_cache' not in path.parts:
                dest=snapshot/path.relative_to(output); dest.parent.mkdir(parents=True,exist_ok=True)
                if path.suffix=='.log': shutil.copy2(path,dest)
                else: os.link(path,dest)
        raw=sum(p.stat().st_size for p in snapshot.rglob('*') if p.is_file())
        if 3*raw>budget: raise RuntimeError(f'Archive peak estimate {3*raw} exceeds {budget}')
        pending=target.with_suffix('.gz.tmp')
        try:
            with tarfile.open(pending,'w:gz',compresslevel=1) as archive: archive.add(snapshot,arcname='.')
            pending.replace(target)
        finally: pending.unlink(missing_ok=True)
    print('Output bytes:',raw,'archive bytes:',target.stat().st_size)
    return target


def choose_solver(rows,override=None,*,damping_rel=1e-2,vna_every=10,n_sites=None):
    import json
    import math
    train=[r for r in rows if math.isclose(r['relative_mu'],damping_rel,rel_tol=1e-9)]
    checks=[r['cos_disjoint_batches'] for r in train if 'cos_disjoint_batches' in r]
    if checks and min(checks)<.3:
        print('WARNING: disjoint-batch cosine < 0.3; recommend VNA_BATCH=64 and larger VNA_EVERY.')
    if override is not None:
        if override not in ('exact','kfac','lite'): raise ValueError('invalid SOLVER_OVERRIDE')
        return override
    if not rows: raise RuntimeError('No Phase-0 validation checkpoint; set SOLVER_OVERRIDE explicitly.')
    exact=[r for r in train if r['variant']=='exact']
    if not exact or not all(r['cg_converged'] for r in exact): raise ValueError('unconverged exact reference at training damping')
    kfac=[r for r in train if r['variant']=='kfac']
    if kfac and len(kfac)==len(exact) and min(r['cosine'] for r in kfac)>=.9 and all(r['realized_gain']>0 for r in kfac): return 'kfac'
    table=[]
    for name in ('exact','kfac','lite'):
        selected=[r for r in train if r['variant']==name]
        if not selected or len(selected)!=len(exact): continue
        counts={epoch:len({r.get('site','site') for r in exact if r.get('epoch')==epoch}) for epoch in {r.get('epoch') for r in exact}}
        sites=n_sites if n_sites is not None else max(counts.values())
        cost=max(r['sgd_units'] for r in selected)
        table.append({'solver':name,'n_sites':sites,'per_correction_sgd_units':cost,'total_overhead':sites*cost/vna_every})
    print('Solver overhead at training damping:',json.dumps(table))
    # Within the affordable set, fidelity order is exact > kfac > lite.
    for row in table:
        if row['total_overhead']<=1.: return row['solver']
    raise RuntimeError('No solver fits overhead <= 1.0; set SOLVER_OVERRIDE. Table: '+json.dumps(table))


def check_training_budget(root, target, payload, budget):
    """Conservative serialized-size bound without allocating a checkpoint copy.

    Full-width EKFAC factors can be large. Fail before writing over the disk cap,
    preserving the previous latest rather than silently dropping resume state.
    """
    import pickle
    import torch
    storages={}
    def metadata(value):
        if torch.is_tensor(value):
            storage=value.untyped_storage()
            storages[(str(value.device),storage.data_ptr())]=storage.nbytes()
            return ('tensor',tuple(value.shape),str(value.dtype))
        if isinstance(value,dict): return {k:metadata(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [metadata(v) for v in value]
        return value
    meta=metadata(payload)
    estimate=sum(storages.values())+len(pickle.dumps(meta))+1_048_576
    current=sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and '_resume_cache' not in p.parts)
    old=target.stat().st_size if target.exists() else 0
    projected=current-old+estimate
    peak=max(current+estimate,3*projected)
    if peak>budget:
        raise RuntimeError(f'Full-state output/archive peak estimate {peak} exceeds disk budget {budget}; previous latest preserved')


def choose_batch(rows, batch_override=None, every_override=None, *, damping_rel=1e-2):
    """Pre-registered worst-site/epoch batch agreement rule, before solver choice."""
    import math
    checks = [r for r in rows if r.get('variant') == 'exact'
              and math.isclose(r['relative_mu'], damping_rel, rel_tol=1e-9)]
    cos16 = min((r['cos16'] for r in checks), default=None)
    cos64 = min((r['cos64'] for r in checks), default=None)
    batch, every = ((16, 10) if cos16 is not None and cos16 >= .5 else
                    (64, 20) if cos64 is not None and cos64 >= .5 else (128, 20))
    if batch_override is not None: batch = batch_override
    if every_override is not None: every = every_override
    if batch < 1 or every < 1: raise ValueError('batch/every overrides must be positive')
    print('Pre-registered batch rule: cos16 >= 0.5 -> (16,10); else cos64 >= 0.5 -> (64,20); else (128,20).',
          'Minimum cos16:', cos16, 'cos64:', cos64,
          'overrides:', batch_override, every_override, 'chosen:', batch, every)
    return batch, every


def apply_budget_policy(arms, projections, budget=30, allow_over_budget=False):
    """The budget is total GPU-hours across all arms of one seed."""
    import json
    selected = list(arms); dropped = []
    def total(): return sum(projections[name] for name, _, _ in selected)
    for name in ('sgd_cm', 'vna_tiny_U'):
        if total() > budget and any(arm[0] == name for arm in selected):
            selected = [arm for arm in selected if arm[0] != name]; dropped.append(name)
    print('Dropped arms:', dropped, 'Projected total GPU-hours:', total(), 'budget:', budget)
    table = [{'arm': name, 'GPU-hours': projections[name], 'dropped': name in dropped}
             for name, _, _ in arms]
    print('Smoke projection table:', json.dumps(table))
    if total() > budget and not allow_over_budget:
        raise RuntimeError(f'Projected {total():.2f} GPU-hours exceeds {budget}; '
                           'set ALLOW_OVER_BUDGET=True to proceed. Table: ' + json.dumps(table))
    return selected, total()


def smoke_diagnostics(output):
    """Report-only pre-registered thresholds; applied ratios include stale steps."""
    import json
    import numpy as np
    output = Path(output)
    def records(name):
        path = output / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    corrections = records('corrections.jsonl'); steps = records('steps.jsonl')
    ratios = [a['applied_delta_ratio'] for row in steps for a in row['applications']]
    median, p95 = (float(np.median(ratios)), float(np.percentile(ratios, 95))) if ratios else (None, None)
    jumps = {label: max((r['loss_jump'] for r in steps
                        if r['correction_step'] == flag and r['loss_jump'] is not None), default=None)
             for label, flag in [('correction_steps', True), ('non_correction_steps', False)]}
    flags = []
    if p95 is not None and p95 > 3: flags.append('IMPULSE_RISK')
    if median is not None and median < .05: flags.append('LIKELY_NULL')
    return {'median_applied_delta_ratio': median, 'p95_applied_delta_ratio': p95,
            'fraction_corrections_skipped': sum(r['skipped'] for r in corrections)/len(corrections) if corrections else None,
            'max_train_loss_jump': jumps, 'flags': flags,
            'thresholds': {'IMPULSE_RISK_p95_gt': 3, 'LIKELY_NULL_median_lt': .05},
            'loss_jump_note': 'Loss at step t precedes its update; jumps are grouped by scheduled correction steps.'}
