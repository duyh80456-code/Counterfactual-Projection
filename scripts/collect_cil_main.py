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
           'status.json','continual_eval.csv','best_config.json','reused_baseline.json','reused_candidate.json',
           'clip_reuse.json','study.json'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def clipping_reuse(package):
    """Validate the approved replay scope while retaining the native producer."""
    from experiments.clip_study import GRID,A_CASES,C_CASES
    selection=package['selection'];tau=selection.get('tau');consumer=package.get('consumer_commit')
    if (selection.get('status')!='SELECTED' or selection.get('grid')!=list(GRID) or tau not in GRID or
            not re.fullmatch('[0-9a-f]{40}',str(consumer)) or selection.get('code_commit')!=consumer or
            canonical_hash(selection)!=package.get('selection_sha256')):
        raise ValueError('Invalid clipping reuse selection')
    checks=selection.get('checks',[]);G=selection.get('G_max')
    if (not isinstance(G,(int,float)) or not math.isfinite(G) or G<0 or
            [c.get('tau') for c in checks]!=list(GRID) or
            any(c.get('baseline_bound') is not (c['tau']>=2*G) or
                not isinstance(c.get('all_four_survive_60'),bool) for c in checks)):
        raise ValueError('Invalid clipping reuse threshold rule')
    eligible=[c['tau'] for c in checks if c['baseline_bound'] and c['all_four_survive_60']]
    if not eligible or min(eligible)!=tau:raise ValueError('Invalid clipping reuse chosen threshold')
    reports=selection.get('source_reports',{})
    expected_reports={f'A/{arm}__s{seed}/{label}/study.json' for arm,seed in A_CASES
                      for label in ('none',*(f'tau{t:g}' for t in GRID))}|{f'B/icarl__s{s}/study.json' for s in (0,1)}
    if expected_reports!=reports.keys() or any(not re.fullmatch('[0-9a-f]{64}',str(v)) for v in reports.values()):
        raise ValueError('Missing clipping selection evidence hashes')
    reference=package['reference'];proof=package['verification'];study=package['study']
    artifacts=package['source_artifacts'];hashes=package.get('source_artifact_sha256',{})
    if set(hashes)!=set(artifacts) or any(canonical_hash(v)!=hashes[k] for k,v in artifacts.items()):
        raise ValueError('Clipping reuse source artifact hash mismatch')
    raw=package.get('source_summary_text','');summary=artifacts['summary.json'];p=artifacts['protocol.json']
    digest=hashlib.sha256(raw.encode()).hexdigest()
    producer={k:v for k,v in p.items() if k not in ('parameter_count','mu0')}
    expected_identity=reference.get('identity')
    if (not raw or json.loads(raw)!=summary or summary!=reference.get('summary') or
            producer!={k:v for k,v in (expected_identity or {}).items() if k not in ('parameter_count','mu0')} or
            expected_identity!=summary.get('identity') or
            digest!=package.get('source_summary_sha256') or digest!=reference.get('summary_sha256') or
            digest!=study.get('reference_sha256') or proof.get('reference')!=reference):
        raise ValueError('Clipping reuse native producer identity mismatch')
    if ('parameter_count' in p and p['parameter_count']!=summary.get('parameter_count') or
            p.get('mu0') is not None and (not isinstance(p['mu0'],(int,float)) or not math.isfinite(p['mu0']) or p['mu0']<=0)):
        raise ValueError('Clipping reuse runtime metadata mismatch')
    arm,seed=p.get('arm'),p.get('seed')
    if ((arm,seed) not in C_CASES or arm in ('wa','podnet','icarl_cflat') or
            arm=='gem' and seed in (1,2) or 'consumer_source_digest' not in package or
            package['consumer_source_digest'] is not None and
            not re.fullmatch('[0-9a-f]{64}',str(package['consumer_source_digest']))):
        raise ValueError('Clipping reuse is not registered for this run')
    if (p.get('grad_clip') is not None or p.get('backbone') not in ('nobn','resnet32_nobn') or
            p.get('nobn_lr_scale')!=.5 or p.get('class_order_kind',p.get('order'))!='superclass'):
        raise ValueError('Clipping reuse requires the native no-clip noBN setup')
    baseline=arm=='icarl' and seed in (0,1)
    scope='all 20 tasks; full norm replay in B' if baseline else dict(task=4,epoch=None)
    epochs=1050 if baseline else 250
    if (proof.get('status')!='EQUIVALENT_PREFIX' or proof.get('tau')!=tau or proof.get('scope')!=scope or
            proof.get('n_clipped')!=0 or study.get('mode')!=('B' if baseline else 'prefix') or
            study.get('tasks')!=(20 if baseline else 4) or study.get('epoch') is not None or
            study.get('status') not in ('COMPLETED','PREFIX_COMPLETED') or
            study.get('diagnostic_only') is not True or study.get('equivalent_under_clip') is not True or
            (study.get('arm'),study.get('seed'),study.get('code_commit'))!=(arm,seed,consumer) or
            study.get('tau')!=(None if baseline else tau)):
        raise ValueError('Invalid clipping reuse replay scope')
    loss=study.get('loss_check',{});norms=(study.get('grad_clip') or {}).get('run',{})
    if (loss!=proof.get('loss_check') or loss.get('status')!='PASS' or
            loss.get('epochs_compared')!=epochs or loss.get('expected_epochs')!=epochs or
            loss.get('absolute_tolerance')!=0 or loss.get('relative_tolerance')!=0 or loss.get('differences')!=[] or
            norms.get('n_clipped')!=0 or norms.get('nonfinite_steps')!=0 or
            not isinstance(norms.get('n_steps'),int) or norms['n_steps']<epochs or
            proof.get('n_steps_checked')!=norms['n_steps'] or
            not isinstance(norms.get('max_preclip_norm'),(int,float)) or
            not math.isfinite(norms['max_preclip_norm']) or not 0<=norms['max_preclip_norm']<=tau):
        raise ValueError('Invalid clipping reuse loss/norm evidence')
    if baseline and norms['max_preclip_norm']>G:raise ValueError('Clipping baseline exceeds selected G_max')
    if not baseline and (study.get('checkpoint_sha256') is not None or
            not re.fullmatch('[0-9a-f]{64}',str(proof.get('checkpoint_sha256'))) or
            proof.get('checkpoint_sha256')!=reference.get('safe',{}).get('sha256')):
        raise ValueError('Clipping reuse checkpoint mismatch')
    comparison={**p,'grad_clip':tau,'code_commit':consumer,'source_digest':package.get('consumer_source_digest')}
    metadata=dict(tau=tau,consumer_commit=consumer,consumer_source_digest=package.get('consumer_source_digest'),
        scope=scope,epochs_compared=epochs,source_summary_sha256=digest,
        claim='full loss/norm replay' if baseline else 'approved four-task prefix replay; no full-run bitwise identity claim')
    return artifacts,comparison,metadata


def collect(inputs):
    bundles={};skipped=[];selections={}
    sources=[]
    for value in inputs:
        p=Path(value);sources.append(p)
        if p.is_dir():sources.extend(q for q in p.rglob('*') if '_sources' not in q.relative_to(p).parts and q.name.endswith(('.tar.gz','.tgz','.tar','.zip')))
    for source in sources:
        for path,data in files(source,ARTIFACTS):
            if '_sources' in path.parts:continue
            bundles.setdefault((str(source),path.parent),{})[path.name]=data
    for key,payload in list(bundles.items()):
        study=json.loads(payload.get('study.json',b'{}'))
        if study.get('diagnostic_only'):
            skipped.append(dict(source=key[0],path=str(key[1]),reason='diagnostic-only clipping study'))
            del bundles[key]
    clip_packages={}
    for key,payload in bundles.items():
        if 'clip_reuse.json' not in payload:continue
        artifacts,comparison,metadata=clipping_reuse(json.loads(payload['clip_reuse.json']))
        if 'summary.json' in payload:raise ValueError('Clipping reuse package conflicts with native run files')
        payload.update({name:json.dumps(value).encode() if name.endswith('.json') else value.encode()
                        for name,value in artifacts.items()})
        clip_packages[key]=(comparison,metadata)
    winners={}
    for (source,parent),payload in bundles.items():
        if 'best_config.json' not in payload:continue
        best=json.loads(payload['best_config.json'])
        if best.get('synthetic'):continue
        relative=PurePosixPath(best.get('selected_trial',''))
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0]!='scan':
            continue
        from experiments.v4_competitors import grid,ROUND2,NOBN_LR_SCALES,DEFAULT_CONFIGS
        if best.get('method') in DEFAULT_CONFIGS:continue
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
        from experiments.v4_competitors import DEFAULT_CONFIGS
        if arm_config:=DEFAULT_CONFIGS.get(p.get('arm')):
            if p.get('method_config')!=arm_config or p.get('parameter_policy')!='original_defaults_no_search_v1':
                skipped.append(dict(source=source,path=str(parent),reason='retired C1/tuned plasticity configuration'));continue
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
        comparison,clip_metadata=clip_packages.get((source,parent),(p,None))
        record=dict(job=job,summary=summary,protocol=p,tasks=tasks,source=source,path=str(parent),comparison_protocol=comparison)
        if clip_metadata:record['clip_reuse']=clip_metadata
        previous=records.get(key)
        if previous:
            if any(previous[k]!=record[k] for k in ('summary','protocol','tasks')):
                raise ValueError('Conflicting duplicate completed seed: '+job)
            if previous['comparison_protocol']!=record['comparison_protocol']:
                if bool(previous.get('clip_reuse'))==bool(record.get('clip_reuse')):
                    raise ValueError('Conflicting duplicate completed seed: '+job)
                if previous.get('clip_reuse'):record=previous
        records[key]=record
    groups=defaultdict(list)
    for (arm,backbone,order,seed),record in records.items():groups[(arm,backbone,order)].append(record)
    for key,runs in groups.items():
        first=configuration_signature(runs[0]['comparison_protocol'])
        if any(configuration_signature(r['comparison_protocol'])!=first for r in runs):
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
                             configuration=configuration_signature(r['comparison_protocol']),
                             **({'clip_reuse':r['clip_reuse'],'producer_protocol':r['protocol']} if r.get('clip_reuse') else {})) for r in runs)
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
