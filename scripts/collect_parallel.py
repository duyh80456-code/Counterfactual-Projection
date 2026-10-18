"""Validate and collect pilot archives, paired metrics and registered decisions."""
import argparse
import csv
import copy
import io
import json
import statistics
import math
import re
import tarfile
import zipfile
import sys
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from experiments.parallel_validity import validate_record


def files(source):
    source = Path(source)
    if source.is_dir():
        for path in source.rglob('*'):
            if path.is_file(): yield PurePosixPath(path.relative_to(source).as_posix()), path.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                if not name.endswith('/'): yield PurePosixPath(name), archive.read(name)
    elif tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            for member in archive.getmembers():
                if member.isfile(): yield PurePosixPath(member.name), archive.extractfile(member).read()
    else:
        raise ValueError(f'unsupported input: {source}')


def validate_result(job, summary, protocol, tasks, expected_commit=None, source_digest=None):
    if source_digest is not None and protocol.get('source_digest') != source_digest:
        raise ValueError(f'{job}: source_digest mismatch')
    if protocol.get('setting') not in ('regression','mnist','icarl','online') and not protocol.get('competitor'):
        return validate_record(job, summary, protocol, tasks, expected_commit)
    commit = protocol.get('code_commit')
    if not re.fullmatch(r'[0-9a-f]{40}', str(commit)):
        raise ValueError(f'{job}: missing immutable code_commit')
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(f'{job}: code_commit mismatch')
    if source_digest is not None and protocol.get('source_digest') != source_digest:
        raise ValueError(f'{job}: source_digest mismatch')
    if summary.get('status') != 'COMPLETED': raise ValueError(f'{job}: incomplete result')
    identity = summary.get('identity', summary)
    if protocol.get('algorithm_version') == 'v3':
        if protocol.get('arm') != job.split('__')[0]: raise ValueError(f'{job}: arm identity mismatch')
        match=re.search(r'__s(\d+)$',job)
        if match and protocol.get('seed') != int(match[1]): raise ValueError(f'{job}: seed identity mismatch')
    for key in ('code_commit','source_digest'):
        if identity.get(key, protocol.get(key)) != protocol.get(key):
            raise ValueError(f'{job}: summary/protocol {key} mismatch')
    if protocol.get('algorithm_version') == 'v3' and protocol.get('synthetic') is not False:
        raise ValueError(f'{job}: synthetic must explicitly be false')
    if any(protocol.get(k) for k in ('fake_data','smoke','preflight_only')):
        raise ValueError(f'{job}: synthetic/preflight result')
    if not isinstance(tasks, (list,dict)) or not tasks:
        raise ValueError(f'{job}: missing completed tasks')
    if summary.get('tasks') is not None and summary['tasks'] != tasks:
        raise ValueError(f'{job}: task artifacts disagree')
    cfg=protocol.get('cfg',{})
    setting=protocol.get('setting')
    expected = protocol.get('expected_tasks')
    if expected is None and protocol.get('algorithm_version') == 'v3':
        expected = (cfg.get('num_data_points',0)//cfg.get('flip_after',1) if setting=='regression' else cfg.get('num_tasks') if setting=='mnist' else 10 if setting=='icarl' else protocol.get('model_config',{}).get('n_tasks',10))
    if protocol.get('algorithm_version') == 'v3' and not expected: raise ValueError(f'{job}: missing full horizon')
    if expected is not None and len(tasks) != expected:
        raise ValueError(f'{job}: incomplete trajectory')
    if not protocol.get('upstream_commit') and not protocol.get('upstream_commits'):
        raise ValueError(f'{job}: missing upstream provenance')
    # Algorithms retain their upstream hyperparameters; compare the common setup.
    setting = protocol.get('setting', 'competitor')
    fingerprint = protocol.get('dataset_fingerprint')
    common = copy.deepcopy(protocol.get('comparison_protocol',cfg))
    for key in list(common):
        if key.startswith('_comment') or key in ('data_file','env_file','data_dir','env_data_dir','run_idx','num_runs','agent','use_gpu','replacement_rate','util_type','decay_rate','maturity_threshold','seed','device','run_id','data_root_dir'):
            common.pop(key)
    if setting in ('icarl','online') and (not isinstance(fingerprint,dict) or set(fingerprint)!={'train','test','meta'} or any(not re.fullmatch('[0-9a-f]{64}',str(v)) for v in fingerprint.values())):
        raise ValueError(f'{job}: missing dataset fingerprint')
    return commit, (setting,json.dumps(protocol.get('model_config'),sort_keys=True),cfg.get('learner') if setting=='online' else None), dict(dataset=fingerprint, common=common,
                                    source_digest=protocol.get('source_digest'))


def collect(inputs, run_tag):
    jobs = {}
    commit = None
    protocols = {}
    for source in inputs:
        grouped = {}
        for path, data in files(source):
            parts = path.parts
            if run_tag not in parts: continue
            index = parts.index(run_tag)
            # Only the job's own files, never a nested rates/.../summary.json.
            if len(parts) != index+3: continue
            job, name = parts[index+1], parts[-1]
            if name in {'summary.json','tasks.json','protocol.json','reproduction_gate.json','vna_log.csv','continual_eval.csv','diagnostics.json','status.json'}:
                grouped.setdefault(job, {})[name] = data
        for job, payload in grouped.items():
            if 'summary.json' not in payload: continue
            record = {'job': job, 'source': str(source),
                      'summary': json.loads(payload['summary.json'])}
            if record['summary'].get('status') != 'COMPLETED': continue
            for name in ('tasks.json','protocol.json','reproduction_gate.json','diagnostics.json','status.json'):
                if name in payload: record[name[:-5]] = json.loads(payload[name])
            if 'vna_log.csv' in payload:
                record['vna'] = list(csv.DictReader(io.StringIO(payload['vna_log.csv'].decode())))
            if 'protocol' not in record or 'tasks' not in record:
                raise ValueError(f'{job}: missing protocol/tasks artifact')
            if 'continual_eval.csv' in payload:
                record['continual_eval'] = list(csv.DictReader(io.StringIO(payload['continual_eval.csv'].decode())))
            if record['protocol'].get('benchmark') == 'v4_competitor':
                if len(record.get('diagnostics',[])) != 20 or len(record.get('continual_eval',[])) != 20 or record.get('status',{}).get('phase') != 'COMPLETED':
                    raise ValueError(f'{job}: missing v4 diagnostics/evaluation/status artifacts')
            current, group, signature = validate_result(job, record['summary'],
                record['protocol'], record['tasks'], commit)
            commit = current
            if group in protocols and protocols[group] != signature:
                raise ValueError(f'{job}: inconsistent comparison protocol')
            protocols[group] = signature
            previous = jobs.get(job)
            if previous is not None and any(previous.get(k) != record.get(k)
                    for k in ('summary','protocol','tasks','reproduction_gate','vna','continual_eval')):
                raise ValueError(f'{job}: conflicting duplicate completed results')
            jobs[job] = record
    digests={r['protocol'].get('source_digest') for r in jobs.values()}
    if len(digests)>1: raise ValueError('mixed source_digest artifacts')
    offline=[r['protocol'].get('dataset_fingerprint') for r in jobs.values() if r['protocol'].get('setting')=='icarl' or r['protocol'].get('benchmark') in ('cil_competitor','v4_competitor')]
    if offline and any(f != offline[0] for f in offline): raise ValueError('offline CIL dataset mismatch')
    validate_shared_schedules(jobs)
    return jobs


def validate_shared_schedules(jobs):
    shared = {}
    for job, record in jobs.items():
        protocol=record['protocol']; schedule=protocol.get('schedule',protocol.get('preflight'))
        if not schedule: continue
        if not isinstance(schedule,dict): raise ValueError(f'{job}: schedule must contain authority artifact')
        setting=(protocol.get('setting'),json.dumps(protocol.get('model_config'),sort_keys=True))
        signature={key:schedule.get(key) for key in ('K_base','mu_rel','cg_tol','tol','power_iters','code_commit','source_digest')}
        if signature['K_base'] is None: raise ValueError(f'{job}: missing shared K_base')
        if setting in shared and signature != shared[setting]:
            raise ValueError(f'{job}: inconsistent shared authority schedule')
        shared[setting]=signature


def metrics(record):
    tasks=record.get('tasks',[])
    if not isinstance(tasks,list): return record['summary']
    result=dict(tasks[-1] if tasks else {},**record['summary'])
    for key in ('anytime_accuracy_auc','anytime_loss_auc','learning_accuracy','iterations_to_half_loss',
                'dead_fraction','stability_gap_depth','stability_recovery_iterations','min_old_accuracy'):
        values=[float(t[key]) for t in tasks if t.get(key) is not None]
        if values and result.get(key) is None: result[key]=statistics.mean(values)
        elif values and key not in record['summary']: result[key]=statistics.mean(values)
    return result


def paired(jobs, treatment, control, metric, lower=False, seeds=(0,1,2)):
    def values(arm):
        result={}
        for name,record in jobs.items():
            summary=metrics(record)
            if name.split('__')[0] != arm or '__rate' in name: continue
            value=summary.get(metric)
            if value is not None: result[summary.get('seed',record['protocol'].get('seed'))]=float(value)
        return result
    treated,base=values(treatment),values(control)
    shared=sorted(set(treated)&set(base)&set(seeds))
    deltas=[treated[s]-base[s] for s in shared]
    wins=sum(d < 0 if lower else d > 0 for d in deltas)
    nonzero=sum(d != 0 for d in deltas)
    p=sum(math.comb(nonzero,k) for k in range(wins,nonzero+1))/(2**nonzero) if nonzero else 1.
    return dict(treatment=treatment,control=control,metric=metric,seeds=shared,
                delta_mean=statistics.mean(deltas) if deltas else None,
                delta_sd=statistics.stdev(deltas) if len(deltas)>1 else None,
                wins=wins,required_seeds=list(seeds),complete=shared==list(seeds),
                sign_test_p_one_sided=p)


def decisions(jobs):
    result=[]
    for setting,prefix,metric,lower,required in (
            ('regression','b5','mean_last50',True,3),
            ('mnist','pm','mean_last_half_accuracy',False,2)):
        for question,control in (('Q-A','bp_kself'),('Q-B','bp_vna_rand')):
            row=paired(jobs,prefix+'_bp_vna',prefix+'_'+control,metric,lower)
            row.update(setting=setting,question=question,signal=row['complete'] and row['wins']>=required)
            result.append(row)
        checks=[paired(jobs,prefix+'_cbp_gir',prefix+'_'+arm,metric,lower) for arm in ('cbp','cbp_gradmax')]
        result.append(dict(setting=setting,question='Q-C',comparisons=checks,
            signal=all(r['complete'] and r['wins']>=2 for r in checks)))
    def summary(arm,model=None):
        return next((metrics(r) for j,r in jobs.items() if j.split('__')[0]==arm and (model is None or r['protocol'].get('model_config')==model)),{})
    vna=summary('icarl_vna')
    for question,arm in (('Q-A','icarl_kself'),('Q-B','icarl_vna_rand')):
        control=summary(arm); metric='cnn_average_accuracy'
        delta=vna.get(metric,{}); delta=(delta-control[metric]) if isinstance(delta,(int,float)) and metric in control else None
        result.append(dict(setting='icarl',question=question,delta=delta,signal=delta is not None and delta>=.5,pilot_only=True))
    b0=summary('icarl_vna_b0')
    keys=('cnn_forgetting','stability_gap_depth')
    complete=all(k in vna and k in b0 for k in keys)
    result.append(dict(setting='icarl',question='Q-D',signal=complete and all(vna[k]<b0[k] for k in keys),complete=complete,pilot_only=True))
    online_configs={json.dumps(r['protocol'].get('model_config'),sort_keys=True) for r in jobs.values() if r['protocol'].get('setting')=='online'} or {'null'}
    for encoded in sorted(online_configs):
        model=json.loads(encoded)
        for question,arm in (('Q-A','er_kself'),('Q-B','er_vna_rand')):
            treated,control=summary('er_vna',model),summary(arm,model)
            keys=('final_accuracy','learning_accuracy')
            complete=all(treated.get(k) is not None and control.get(k) is not None for k in keys)
            result.append(dict(setting='online',model_config=model,question=question,signal=complete and all(treated[k]-control[k]>=.5 for k in keys),complete=complete,pilot_only=True))
    return result


def mean_sd(values):
    values = [float(v) for v in values if v is not None]
    if not values: return None
    return f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}' if len(values)>1 else f'{values[0]:.4f}'


def markdown(title, rows, columns):
    print('\n### '+title)
    print('| '+' | '.join(columns)+' |')
    print('| '+' | '.join('---' for _ in columns)+' |')
    for row in rows: print('| '+' | '.join(str(row.get(c,'')) for c in columns)+' |')


def cil_row(record):
    summary = record['summary']; tasks = record.get('tasks', summary.get('tasks', []))
    last = tasks[-1] if tasks else {}
    protocol = record.get('protocol', {})
    return dict(job=record['job'], arm=record['job'].split('__')[0],
        seed=summary.get('seed', protocol.get('seed')), init_cls=summary.get('init_cls', protocol.get('init_cls', protocol.get('args',{}).get('init_cls',10))),
        increment=summary.get('increment', protocol.get('increment', protocol.get('args',{}).get('increment',10))),
        cnn_average=last.get('cnn_average_accuracy'), nme_average=last.get('nme_average_accuracy'),
        cnn_last=last.get('cnn_top1'), nme_last=last.get('nme_top1'),
        cnn_forgetting=last.get('cnn_forgetting'), nme_forgetting=last.get('nme_forgetting'),
        reference=record.get('reproduction_gate', summary.get('reproduction',{})).get('reference','none'),
        reproduction_gate=record.get('reproduction_gate', summary.get('reproduction',{})).get('status','none'),
        parameters=summary.get('parameter_count'),evaluated_parameters=summary.get('evaluated_parameter_count',summary.get('parameter_count')),parameter_count_source=summary.get('parameter_count_source',''),gpu_hours=summary.get('gpu_hours',float(summary.get('seconds',0))/3600),
        model_group=summary.get('model_group','fixed'))


def vna_cost(record):
    rows = record.get('vna', [])
    rho = [float(r['rho']) for r in rows if r.get('rho') not in (None,'')]
    seconds = sum(float(r.get('seconds') or 0) for r in rows)
    ks = sorted({r.get('K') for r in rows if r.get('K') not in (None,'')})
    train = record['summary'].get('training_seconds')
    if train is None: train = max(0., float(record['summary'].get('seconds',0))-seconds)
    skip_rate=sum(str(r.get('skipped','')).lower() in ('1','true') for r in rows)/max(1,len(rows))
    return dict(job=record['job'], K=','.join(ks), rho_mean=mean_sd(rho),skip_rate=skip_rate,
                vna_seconds=f'{seconds:.2f}', train_seconds=f'{train:.2f}',
                vna_over_train=f'{seconds/train:.4f}' if train else '')


def report_v4(jobs):
    groups = {}
    records = [r for r in jobs.values() if r['protocol'].get('benchmark') == 'v4_competitor']
    controls = {(r['protocol']['backbone'],r['protocol']['order'],r['protocol']['seed']):r for r in records if r['protocol']['arm'] == 'icarl'}
    for record in records:
        p = record['protocol']
        groups.setdefault((p['arm'],p['backbone'],p['order']),[]).append(record)
    rows = []
    for (arm,backbone,order), group in sorted(groups.items()):
        deltas = []
        for r in group:
            p = r['protocol']
            control = controls.get((backbone,order,p['seed']))
            if control:
                deltas.append(r['tasks'][-1]['cnn_average_accuracy']-control['tasks'][-1]['cnn_average_accuracy'])
        rows.append(dict(arm=arm,backbone=backbone,order=order,seeds=len(group),
            average_accuracy=mean_sd([r['tasks'][-1]['cnn_average_accuracy'] for r in group]),
            final_forgetting=mean_sd([r['tasks'][-1]['cnn_forgetting'] for r in group]),
            learning_accuracy=mean_sd([statistics.mean(t['learning_accuracy'] for t in r['tasks']) for r in group]),
            paired_delta=mean_sd(deltas),seeds_won=sum(d>0 for d in deltas)))
    if rows: markdown('V4 matched competitors',rows,['arm','backbone','order','seeds','average_accuracy','final_forgetting','learning_accuracy','paired_delta','seeds_won'])


def report(jobs):
    report_v4(jobs)
    report_v3(jobs)
    cil = [cil_row(r) for name,r in sorted(jobs.items()) if name.startswith('icarl') or r['protocol'].get('benchmark') in ('cil_competitor','v4_competitor')]
    controls={(r['init_cls'],r['increment'],r['seed']):r for r in cil if r['arm']=='icarl'}
    for row in cil:
        control=controls.get((row['init_cls'],row['increment'],row['seed']))
        row['gpu_ratio_icarl']=row['gpu_hours']/control['gpu_hours'] if control and control['gpu_hours'] else None
    inc10 = [r for r in cil if r['init_cls']==10 and r['increment']==10]
    display = [r for r in inc10 if r['seed']==1993]
    for arm in ('icarl','icarl_vna'):
        rows = [r for r in inc10 if r['arm']==arm]
        if rows:
            display.append(dict(arm=arm+' (3-seed)', seed='all',
                cnn_average=mean_sd([r['cnn_average'] for r in rows]),
                nme_average=mean_sd([r['nme_average'] for r in rows]),
                cnn_last=mean_sd([r['cnn_last'] for r in rows]), nme_last=mean_sd([r['nme_last'] for r in rows]),
                cnn_forgetting=mean_sd([r['cnn_forgetting'] for r in rows]),
                nme_forgetting=mean_sd([r['nme_forgetting'] for r in rows])))
    controls = {r['seed']:r for r in inc10 if r['arm']=='icarl'}
    treated = {r['seed']:r for r in inc10 if r['arm']=='icarl_vna'}
    paired = [treated[s]['nme_average']-controls[s]['nme_average'] for s in controls.keys() & treated.keys()]
    if paired: display.append(dict(arm='paired icarl_vna-icarl', seed='all', nme_average=mean_sd(paired)))
    columns = ['arm','seed','cnn_average','nme_average','cnn_last','nme_last','cnn_forgetting','nme_forgetting','parameters','evaluated_parameters','parameter_count_source','gpu_hours','gpu_ratio_icarl','model_group','reproduction_gate']
    markdown('CIL Inc10', display, columns)
    def protocol_rows(selected):
        result = list(selected)
        controls = {r['seed']:r for r in selected if r['arm']=='icarl'}
        treated = {r['seed']:r for r in selected if r['arm']=='icarl_vna'}
        shared = controls.keys() & treated.keys()
        if shared:
            result.append(dict(arm='paired icarl_vna-icarl', seed='all',
                cnn_average=mean_sd([treated[s]['cnn_average']-controls[s]['cnn_average'] for s in shared]),
                nme_average=mean_sd([treated[s]['nme_average']-controls[s]['nme_average'] for s in shared]),
                cnn_last=mean_sd([treated[s]['cnn_last']-controls[s]['cnn_last'] for s in shared]),
                nme_last=mean_sd([treated[s]['nme_last']-controls[s]['nme_last'] for s in shared])))
        return result
    inc5 = [r for r in cil if r['init_cls']==10 and r['increment']==5]
    b50 = [r for r in cil if r['init_cls']==50 and r['increment']==10]
    markdown('CIL Inc5', protocol_rows(inc5), columns)
    markdown('CIL B50 Inc10', protocol_rows(b50), columns+['reference'])

    b5 = []
    for name, record in sorted(jobs.items()):
        if not name.startswith('b5_'): continue
        summary = record['summary']
        variants = summary.get('variants') or [summary]
        for variant in variants:
            b5.append(dict(job=name, arm=name.split('__')[0][3:], seed=variant.get('seed'),
                           replacement_rate=variant.get('replacement_rate',record['protocol'].get('replacement_rate')),
                           mean_last50=variant.get('mean_last50')))
    controls = {r['seed']:r for r in b5 if r['arm']=='cbp' and '__rate' not in r['job']}
    b5_table = []
    for arm, rate in sorted({(r['arm'],r['replacement_rate']) for r in b5}, key=str):
        rows = [r for r in b5 if r['arm']==arm and r['replacement_rate']==rate]
        delta = [r['mean_last50']-controls[r['seed']]['mean_last50'] for r in rows
                 if r['seed'] in controls and r['mean_last50'] is not None]
        b5_table.append(dict(arm=arm, replacement_rate=rate, seeds=len(rows), mean_last50=mean_sd([r['mean_last50'] for r in rows]),
                             paired_delta_cbp=mean_sd(delta)))
    markdown('B5 regression', b5_table, ['arm','replacement_rate','seeds','mean_last50','paired_delta_cbp'])
    costs = [vna_cost(r) for r in jobs.values() if r.get('vna')]
    markdown('VNA cost', costs, ['job','K','rho_mean','skip_rate','vna_seconds','train_seconds','vna_over_train'])


def report_v3(jobs):
    columns=['job','metric','value','anytime_auc','learning_accuracy','iterations_to_half_loss','dead_fraction','swaps','gpu_hours','parameters']
    rows=[]
    for job,record in sorted(jobs.items()):
        summary=record['summary']; protocol=record['protocol']
        setting=protocol.get('setting')
        if setting not in ('regression','mnist','online','icarl') and not protocol.get('benchmark','').endswith('_competitor'): continue
        metric={'regression':'mean_last50','mnist':'mean_last_half_accuracy','online':'final_accuracy','icarl':'cnn_average_accuracy'}.get(setting,'mean_last50')
        summary=metrics(record)
        row=dict(job=job,metric=metric,value=summary.get(metric))
        for key in columns[3:]: row[key]=summary.get(key)
        row['parameters']=summary.get('parameter_count',summary.get('parameters'))
        if setting in ('icarl','online'): row['gpu_hours']=summary.get('gpu_hours',float(summary.get('seconds',0))/3600)
        row['anytime_auc']=summary.get('anytime_accuracy_auc',summary.get('anytime_loss_auc',summary.get('anytime_auc')))
        row['model_group']='expanding' if job.split('__')[0] in ('der','memo','tagfex') else 'fixed'
        row['reproduction_gate']=record.get('reproduction_gate',{}).get('status','none')
        rows.append(row)
    markdown('Pilot primary and learning dynamics',rows,columns+['model_group','reproduction_gate'])
    pm=[]
    for arm in sorted({j.split('__')[0] for j in jobs if j.startswith('pm_')}):
        selected=[metrics(r) for j,r in jobs.items() if j.split('__')[0]==arm]
        pm.append(dict(arm=arm,seeds=len(selected),
            accuracy=mean_sd([r.get('mean_last_half_accuracy') for r in selected]),
            anytime_auc=mean_sd([r.get('anytime_accuracy_auc') for r in selected]),
            learning_accuracy=mean_sd([r.get('learning_accuracy') for r in selected]),
            iterations_to_half_loss=mean_sd([r.get('iterations_to_half_loss') for r in selected]),
            dead_fraction=mean_sd([r.get('dead_fraction') for r in selected]),
            swaps=mean_sd([r.get('swaps') for r in selected])))
    markdown('Permuted MNIST paired seeds',pm,['arm','seeds','accuracy','anytime_auc','learning_accuracy','iterations_to_half_loss','dead_fraction','swaps'])
    markdown('Preregistered decisions (missing comparisons remain incomplete)',[
        dict(setting=r['setting'],model_config=r.get('model_config'),question=r['question'],signal=r['signal'],
             paired_delta=r.get('delta_mean',r.get('delta')),
             wins=r.get('wins'),sign_test_p=r.get('sign_test_p_one_sided')) for r in decisions(jobs)],
        ['setting','model_config','question','signal','paired_delta','wins','sign_test_p'])
    baseline=next((r['summary'].get('mean_last50') for j,r in jobs.items() if j=='b5_cbp__s0'),None)
    if baseline is not None:
        better=[j for j,r in jobs.items() if j.startswith('b5_cbp__rate') and r['summary'].get('mean_last50',float('inf'))<baseline]
        if better: print('CBP baseline rate check: better seed-0 rates found; keep registered 1e-3 unchanged:',', '.join(sorted(better)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+')
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--json-out')
    args = parser.parse_args()
    jobs = collect(args.archives, args.run_tag)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(dict(jobs={k:v['summary'] for k,v in jobs.items()},decisions=decisions(jobs)), indent=2)+'\n')
    report(jobs)


if __name__ == '__main__': main()
