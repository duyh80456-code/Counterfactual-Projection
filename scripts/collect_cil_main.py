"""Collect completed CIL runs across notebooks into tables and task curves.

python -m scripts.collect_cil_main outputs1.tar.gz outputs2/ --out results/cil
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import re
from pathlib import Path,PurePosixPath
import statistics
from scripts.collect_parallel import files,validate_result
from experiments.main_cil_results import configuration_signature
from experiments.io_utils import atomic_json_save

ARTIFACTS={'summary.json','protocol.json','tasks.json','diagnostics.json','interventions.json',
           'status.json','continual_eval.csv','best_config.json','reused_baseline.json','reused_candidate.json'}


def collect(inputs):
    bundles={};skipped=[];selections={}
    sources=[]
    for value in inputs:
        p=Path(value);sources.append(p)
        if p.is_dir():sources.extend(q for q in p.rglob('*') if q.name.endswith(('.tar.gz','.tgz','.tar','.zip')))
    for source in sources:
        for path,data in files(source,ARTIFACTS):
            bundles.setdefault((str(source),path.parent),{})[path.name]=data
    winners={}
    for (source,parent),payload in bundles.items():
        if 'best_config.json' not in payload:continue
        best=json.loads(payload['best_config.json'])
        if best.get('synthetic'):continue
        relative=PurePosixPath(best.get('selected_trial',''))
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0]!='scan':
            continue
        from experiments.v4_competitors import grid,ROUND2,NOBN_LR_SCALES
        if best.get('method') not in ROUND2 or best.get('config') not in grid(best['method']) or best.get('lr_scales')!=list(NOBN_LR_SCALES):
            raise ValueError('Unregistered sweep selection')
        if ((best.get('seed'),best.get('backbone'),best.get('order'),best.get('metric'))!=(0,'nobn','superclass','cnn_average_accuracy')
            or best.get('nobn_lr_scale') not in NOBN_LR_SCALES or not math.isfinite(best.get('score',float('nan')))
            or not re.fullmatch('[0-9a-f]{40}',str(best.get('code_commit')))):
            raise ValueError('Invalid sweep selection provenance')
        key=best['method'];choice=(best['config'],best['nobn_lr_scale'])
        if key in selections and selections[key]!=choice:raise ValueError('Conflicting selected configuration: '+key)
        selections[key]=choice;winners[(source,parent/relative)]=best
    records={}
    for (source,parent),payload in bundles.items():
        if 'summary.json' not in payload:continue
        if 'scan' in parent.parts and (source,parent) not in winners:continue
        summary=json.loads(payload['summary.json']);p=json.loads(payload.get('protocol.json',b'{}'))
        reused=None
        if summary.get('status')=='REUSED_BASELINE':
            reused=json.loads(payload['reused_baseline.json'])
            verification=reused['verification']
            if verification.get('status')!='PASS' or verification.get('epochs_compared')!=150 or verification.get('differences'):
                raise ValueError('Invalid historical baseline verification')
            from experiments.main_cil_baseline import validate
            validate(reused['reference'],p)
            summary,p=reused['reference']['summary'],reused['reference']['protocol']
        if not (p.get('setting')=='icarl' or p.get('benchmark')=='v4_competitor'):continue
        if summary.get('status')!='COMPLETED' or p.get('expected_tasks')!=20:
            skipped.append(dict(source=source,path=str(parent),status=summary.get('status'),reason='not a completed 20-task result'));continue
        tasks=summary.get('tasks') if reused else json.loads(payload.get('tasks.json',b'null'))
        if tasks is None:raise ValueError('Missing tasks: '+str(parent))
        backbone={'bn':'resnet32','nobn':'resnet32_nobn'}.get(p['backbone'],p['backbone'])
        order=p.get('class_order_kind',p.get('order'));seed=p['seed'];arm=p['arm']
        if (source,parent) in winners:
            best=winners[(source,parent)]
            if (arm,seed,p.get('method_config'),p.get('nobn_lr_scale'),p.get('dataset_fingerprint'),summary.get('average_incremental_accuracy'))!=(
                best['method'],0,best['config'],best['nobn_lr_scale'],best.get('dataset_fingerprint'),best['score']):
                raise ValueError('Selected sweep winner artifact mismatch')
            if best.get('source_digest')!=p.get('source_digest'):
                raise ValueError('Selected sweep source digest mismatch')
            if best['code_commit']!=p['code_commit']:
                reuse=json.loads(payload.get('reused_candidate.json',b'{}'))
                if p['arm']!='gem' or reuse.get('source_commit')!=p['code_commit'] or reuse.get('consumer_commit')!=best['code_commit'] or reuse.get('summary_sha256')!=hashlib.sha256(payload['summary.json']).hexdigest():
                    raise ValueError('Undeclared cross-commit sweep candidate reuse')
        if p.get('reproduction'):
            skipped.append(dict(source=source,path=str(parent),reason='BN reproduction excluded from main table'));continue
        if seed not in (0,1,2):raise ValueError('Main table requires registered seeds 0,1,2')
        if arm in selections and backbone=='resnet32_nobn':
            cfg,scale=selections[arm]
            if (p.get('method_config'),p.get('nobn_lr_scale'))!=(cfg,scale):
                if seed!=0:raise ValueError(arm+': matched seed uses an unselected LR/config')
                skipped.append(dict(source=source,path=str(parent),reason='non-winning seed0 LR/config'));continue
        if backbone=='resnet32_nobn' and p.get('nobn_lr_scale')!=.5:raise ValueError('Main noBN table requires nobn_lr_scale=0.5')
        job=f'{arm}__{"nobn" if backbone=="resnet32_nobn" else "bn"}__{order}__s{seed}'
        artifacts={name:json.loads(data) if name.endswith('.json') else data for name,data in payload.items()}
        validate_result(job,summary,p,tasks,artifacts=None if reused else artifacts)
        if p.get('benchmark')=='v4_competitor':
            if len(artifacts.get('diagnostics.json',[]))!=20 or artifacts.get('status.json',{}).get('phase')!='COMPLETED':
                raise ValueError('Incomplete competitor diagnostics/status: '+job)
            if len(list(csv.DictReader(artifacts.get('continual_eval.csv',b'').decode().splitlines())))!=20:
                raise ValueError('Incomplete competitor evaluation: '+job)
        for head in ('cnn','nme'):
            for metric in ('average_accuracy','last_accuracy'):
                name=head+'_'+metric
                if summary.get(name)!=tasks[-1].get(name):raise ValueError('Summary/final task metric mismatch: '+job+'/'+name)
        if p.get('setting')=='icarl' and arm!='icarl' and not p.get('policy_version'):
            raise ValueError('Missing intervention policy identity: '+job)
        key=(arm,backbone,order,seed)
        record=dict(job=job,summary=summary,protocol=p,tasks=tasks,source=source,path=str(parent))
        previous=records.get(key)
        if previous and any(previous[k]!=record[k] for k in ('summary','protocol','tasks')):
            raise ValueError('Conflicting duplicate completed seed: '+job)
        records[key]=record
    groups=defaultdict(list)
    for (arm,backbone,order,seed),record in records.items():groups[(arm,backbone,order)].append(record)
    for key,runs in groups.items():
        first=configuration_signature(runs[0]['protocol'])
        if any(configuration_signature(r['protocol'])!=first for r in runs):
            raise ValueError(str(key)+': mixed solver/curvature/LR/configuration across seeds')
    fingerprints=[r['protocol']['dataset_fingerprint'] for r in records.values()]
    if fingerprints and any(f!=fingerprints[0] for f in fingerprints):raise ValueError('CIL dataset mismatch')
    return groups,skipped


def estimate(values):
    return statistics.mean(values),statistics.stdev(values) if len(values)>1 else None


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def export(inputs,out):
    groups,skipped=collect(inputs);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[];curves=[];manifest=[]
    for (arm,backbone,order),runs in sorted(groups.items()):
        seeds=sorted(r['protocol']['seed'] for r in runs)
        base=dict(arm=arm,backbone=backbone,order=order,seeds=','.join(map(str,seeds)),n=len(seeds),
                  coverage='COMPLETE_3_SEEDS' if seeds==[0,1,2] else 'PARTIAL_SEEDS')
        row=dict(base)
        for head in ('cnn','nme'):
            for label,metric in (('A','average_accuracy'),('last','last_accuracy')):
                values=[r['summary'].get(head+'_'+metric) for r in runs]
                if any(v is None or not math.isfinite(v) for v in values):mean,std=None,None
                else:mean,std=estimate(values)
                row[head+'_'+label+'_mean'],row[head+'_'+label+'_std']=mean,std
            for task in range(1,21):
                values=[r['tasks'][task-1].get(head+'_average_accuracy') for r in runs]
                if any(v is None or not math.isfinite(v) for v in values):continue
                mean,std=estimate(values)
                curves.append(dict(**base,head=head,task=task,mean=mean,std=std))
        rows.append(row)
        manifest.extend(dict(job=r['job'],source=r['source'],path=r['path'],code_commit=r['protocol']['code_commit'],
                             configuration=configuration_signature(r['protocol'])) for r in runs)
    write_csv(out/'main_cil_table.csv',rows);write_csv(out/'average_accuracy_by_task.csv',curves)
    def text(mean,std):return 'NA' if mean is None else f'{mean:.2f} ± {std:.2f}' if std is not None else f'{mean:.2f} (n=1; std unavailable)'
    lines=['| Method | Backbone | Order | Seeds | Status | CNN Ā | CNN last | NME Ā | NME last |',
           '|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        metrics=[text(row[h+'_'+m+'_mean'],row[h+'_'+m+'_std']) for h in ('cnn','nme') for m in ('A','last')]
        lines.append('| '+' | '.join([row['arm'],row['backbone'],row['order'],row['seeds'],row['coverage'],*metrics])+' |')
    (out/'main_cil_table.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for backbone,order,head in sorted({(c['backbone'],c['order'],c['head']) for c in curves}):
        fig,ax=plt.subplots(figsize=(8,5))
        for arm in sorted({c['arm'] for c in curves if (c['backbone'],c['order'],c['head'])==(backbone,order,head)}):
            data=[c for c in curves if (c['arm'],c['backbone'],c['order'],c['head'])==(arm,backbone,order,head)]
            x=[c['task'] for c in data];y=[c['mean'] for c in data];sd=[c['std'] or 0. for c in data]
            ax.plot(x,y,label=f'{arm} (n={data[0]["n"]})')
            if data[0]['n']>1:ax.fill_between(x,[v-s for v,s in zip(y,sd)],[v+s for v,s in zip(y,sd)],alpha=.15)
        ax.set(xlabel='Completed task',ylabel='Average incremental accuracy (%)',title=f'{backbone} / {order} / {head.upper()}')
        ax.set_xticks([1,5,10,15,20]);ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        for extension in ('png','svg'):fig.savefig(out/f'average_accuracy_{backbone}_{order}_{head}.{extension}',dpi=160)
        plt.close(fig)
    atomic_json_save(dict(records=manifest,skipped=skipped),out/'provenance.json')
    print('\n'.join(lines));return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+');parser.add_argument('--out',required=True)
    args=parser.parse_args();export(args.inputs,args.out)
