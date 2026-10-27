"""Strict source/protocol grouping; failures remain visible, no fabricated claims."""
import argparse
from collections import defaultdict,Counter
import json
import math
from pathlib import Path
import statistics


def signature(protocol):
    p=dict(protocol)
    for key in ('seed','arm','evidence_status','test_gates'):p.pop(key,None)
    if p.get('branch')=='H4':
        # Transform hashes depend on seed; the generation rule remains in config.
        p.pop('transformations',None)
    if p.get('branch')=='H3':
        p.pop('data_sha256',None);p.pop('upstream_recipe',None)
        params=dict(p['params'])
        for key in ('agent','num_runs','replacement_rate','decay_rate','weight_decay','perturb_scale','util_type',
                    'maturity_threshold','gir_maturity','data_dir','gir_replacement_rates'):params.pop(key,None)
        p['params']=params
    correction=dict(p.get('correction',{}))
    for key in ('kernel','constrained','seed'):correction.pop(key,None)
    p['correction']=correction
    if 'config' in p:
        p['config']=dict(p['config'])
        for key in ('arm','seed'):p['config'].pop(key,None)
    return json.dumps(p,sort_keys=True)


def collect(roots):
    records={};identities={}
    for root in map(Path,roots):
        for path in root.rglob('summary.json'):
            value=json.loads(path.read_text());p=value.get('protocol',{})
            if p.get('branch') not in ('H1','H2','H3','H4'):continue
            manifest=path.parent/'protocol.json'
            if not manifest.exists() or json.loads(manifest.read_text())!=p:raise ValueError(f'summary/protocol mismatch: {path}')
            arm=p.get('arm',p.get('config',{}).get('arm'));seed=p.get('seed',p.get('config',{}).get('seed'))
            key=(signature(p),arm,seed)
            identity=p.get('data_sha256') if p['branch']=='H3' else p.get('transformations') if p['branch']=='H4' else p.get('dataset_fingerprint')
            cohort=(key[0],seed)
            if cohort in identities and identities[cohort]!=identity:raise ValueError(f'paired data/transform mismatch: {path}')
            identities[cohort]=identity
            if key in records and records[key]!=value:raise ValueError(f'conflicting duplicate result: {path}')
            if value.get('status')=='COMPLETED':
                expected=p.get('tasks',p.get('config',{}).get('domains'))
                if len(value.get('tasks',[]))!=expected:raise ValueError(f'incomplete completed result: {path}')
                required=('last_50_task_error','plasticity_decay','gpu_hours') if p['branch']=='H3' else ('average_incremental_accuracy','final_accuracy','forgetting','backward_transfer','gpu_hours')
                if not all(isinstance(value.get(k),(int,float)) and math.isfinite(value[k]) for k in required):
                    raise ValueError(f'nonfinite/missing completed metrics: {path}')
            records[key]=value
    grouped=defaultdict(list)
    for (group,arm,seed),value in records.items():grouped[(group,arm)].append((seed,value))
    result=[]
    for (group,arm),runs in grouped.items():
        completed=[(s,r) for s,r in runs if r.get('status')=='COMPLETED']
        metrics={}
        fields=('last_50_task_error','plasticity_decay','gpu_hours') if json.loads(group)['branch']=='H3' else ('average_incremental_accuracy','final_accuracy','forgetting','backward_transfer','gpu_hours')
        baseline='bp' if json.loads(group)['branch']=='H3' else 'baseline'
        for metric in fields:
            values=[r[metric] for _,r in completed]
            delta=[r[metric]-records[(group,baseline,s)][metric] for s,r in completed
                   if (group,baseline,s) in records and records[(group,baseline,s)].get('status')=='COMPLETED']
            metrics[metric]=dict(mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values)>1 else None,
                paired_delta=statistics.mean(delta) if delta else None,paired_seeds=len(delta))
        result.append(dict(protocol=json.loads(group),arm=arm,seeds=[s for s,_ in runs],
            outcomes=dict(Counter(r['status'] for _,r in runs)),metrics=metrics,evidence_status='INCONCLUSIVE',
            costs=[dict(seed=s,parameter_count=r.get('parameter_count'),wall_seconds=r.get('wall_seconds'),
                correction_seconds=r.get('correction_seconds'),peak_vram_bytes=r.get('peak_vram_bytes')) for s,r in runs],
            failures=[dict(seed=s,status=r['status'],failure=r.get('failure')) for s,r in runs if r['status']!='COMPLETED']))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('roots',nargs='+')
    args=parser.parse_args();print(json.dumps(collect(args.roots),indent=2))
