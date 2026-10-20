"""Validate and collect v4 run archives and print the main paired table."""
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


def validate_result(job, summary, protocol, tasks, expected_commit=None, source_digest=None, artifacts=None):
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
    if protocol.get('algorithm_version') == 'v4' and protocol.get('setting') == 'icarl':
        from experiments.main_cil import ARMS, RESULT_FILES
        if protocol.get('arm') not in ARMS or protocol['arm'] != job.split('__')[0]:
            raise ValueError(f'{job}: arm identity mismatch')
        match = re.search(r'__s(\d+)$', job)
        if match and protocol.get('seed') != int(match[1]): raise ValueError(f'{job}: seed identity mismatch')
        if len(tasks) != 20 or protocol.get('expected_tasks') != 20:
            raise ValueError(f'{job}: incomplete main CIL trajectory')
        cfg = protocol.get('cfg',{})
        if any(cfg.get(k) != v for k,v in dict(init_cls=5,increment=5,memory_size=2000,fixed_memory=False).items()):
            raise ValueError(f'{job}: main CIL requires B0 Inc5 and dynamic memory 2000')
        if protocol.get('backbone') not in ('resnet32','resnet32_nobn') or protocol.get('r') != 16:
            raise ValueError(f'{job}: invalid main CIL backbone/rank')
        training = protocol.get('training',{})
        if any(training.get(k) != v for k,v in dict(first_epochs=100,later_epochs=50,
                first_milestones=[30,60,85],later_milestones=[24,35]).items()):
            raise ValueError(f'{job}: incorrect main CIL training schedule')
        if protocol.get('class_order_kind') not in ('superclass','random') or sorted(protocol.get('class_order',[])) != list(range(100)):
            raise ValueError(f'{job}: invalid class order')
        for i, row in enumerate(tasks, 1):
            if row.get('task') != i: raise ValueError(f'{job}: unordered tasks')
            for label in ('cnn','nme'):
                for metric in ('top1','new_top1','learning_accuracy','forgetting','average_accuracy','last_accuracy'):
                    if not isinstance(row.get(label+'_'+metric),(int,float)) or not math.isfinite(row[label+'_'+metric]):
                        raise ValueError(f'{job}: missing/nonfinite {label}_{metric}')
                if i>1 and (not isinstance(row.get(label+'_old_top1'),(int,float)) or not math.isfinite(row[label+'_old_top1'])):
                    raise ValueError(f'{job}: missing old-class accuracy')
                if set(row.get(label+'_per_class',{})) != {str(c) for c in range(i*5)}:
                    raise ValueError(f'{job}: missing per-class metrics')
            if i in (5,10,15,20) and (not isinstance(row.get('feature_accuracy'), (int,float))
                    or not math.isfinite(row['feature_accuracy'])):
                raise ValueError(f'{job}: missing feature accuracy')
        if artifacts is not None:
            required = set(RESULT_FILES)
            if protocol['arm'] != 'icarl': required.add('interventions.json')
            if not required <= set(artifacts): raise ValueError(f'{job}: missing main CIL files')
            if artifacts['status.json']['status'] != 'COMPLETED': raise ValueError(f'{job}: incomplete status')
            diagnostics = artifacts['diagnostics.json']
            if len(diagnostics) != 20 or [r.get('task') for r in diagnostics] != list(range(1,21)):
                raise ValueError(f'{job}: incomplete diagnostics')
            for row in diagnostics:
                if row.get('probe_size',256) != 256 or any(not isinstance(row.get(k), (int,float)) or not math.isfinite(row[k]) for k in ('B','mu_diag','norm_w2')):
                    raise ValueError(f'{job}: invalid diagnostics')
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
    return commit, (setting,json.dumps(protocol.get('model_config'),sort_keys=True),cfg.get('learner') if setting=='online' else cfg.get('num_features') if setting=='regression' else None), dict(dataset=fingerprint, common=common,
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
            if name in {'summary.json','tasks.json','protocol.json','reproduction_gate.json','vna_log.csv','continual_eval.csv','diagnostics.json','interventions.json','status.json'}:
                grouped.setdefault(job, {})[name] = data
        for job, payload in grouped.items():
            if 'summary.json' not in payload: continue
            record = {'job': job, 'source': str(source),
                      'summary': json.loads(payload['summary.json'])}
            if record['summary'].get('status') != 'COMPLETED': continue
            for name in ('tasks.json','protocol.json','reproduction_gate.json','diagnostics.json','interventions.json','status.json'):
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
                record['protocol'], record['tasks'], commit, artifacts={
                    **{name:record.get(name[:-5]) for name in payload if name.endswith('.json')},
                    **{name:payload[name] for name in payload if not name.endswith('.json')}})
            commit = current
            if group in protocols and protocols[group] != signature:
                raise ValueError(f'{job}: inconsistent comparison protocol')
            protocols[group] = signature
            previous = jobs.get(job)
            if previous is not None and any(previous.get(k) != record.get(k)
                    for k in ('summary','protocol','tasks','reproduction_gate','vna','continual_eval','diagnostics','interventions','status')):
                raise ValueError(f'{job}: conflicting duplicate completed results')
            jobs[job] = record
    digests={r['protocol'].get('source_digest') for r in jobs.values()}
    if len(digests)>1: raise ValueError('mixed source_digest artifacts')
    offline=[r['protocol'].get('dataset_fingerprint') for r in jobs.values() if r['protocol'].get('setting')=='icarl' or r['protocol'].get('benchmark') in ('cil_competitor','v4_competitor')]
    if offline and any(f != offline[0] for f in offline): raise ValueError('offline CIL dataset mismatch')
    scales = {r['protocol'].get('nobn_lr_scale',1.) for r in jobs.values()
              if (r['protocol'].get('algorithm_version') == 'v4' or r['protocol'].get('benchmark') == 'v4_competitor')
              and r['protocol'].get('backbone') in ('nobn','resnet32_nobn')}
    if len(scales)>1: raise ValueError('mixed noBN learning-rate scales')
    validate_shared_schedules(jobs)
    return jobs


def validate_shared_schedules(jobs):
    shared = {}
    for job, record in jobs.items():
        protocol=record['protocol']; schedule=protocol.get('schedule',protocol.get('preflight'))
        if not schedule: continue
        if not isinstance(schedule,dict): raise ValueError(f'{job}: schedule must contain authority artifact')
        setting=(protocol.get('setting'),json.dumps(protocol.get('model_config'),sort_keys=True))
        setting = (protocol.get('setting'), protocol.get('backbone'), protocol.get('cfg',{}).get('num_features'))
        signature = {key:schedule.get(key) for key in ('K0','seconds_per_application','code_commit','source_digest')}
        if not isinstance(signature['K0'],int) or signature['K0'] < 1: raise ValueError(f'{job}: missing shared K0')
        if setting in shared and signature != shared[setting]:
            raise ValueError(f'{job}: inconsistent shared authority schedule')
        shared[setting]=signature


def mean_sd(values):
    values = [float(v) for v in values if v is not None]
    if not values: return None
    return f'{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}' if len(values)>1 else f'{values[0]:.4f}'


def markdown(title, rows, columns):
    print('\n### '+title)
    print('| '+' | '.join(columns)+' |')
    print('| '+' | '.join('---' for _ in columns)+' |')
    for row in rows: print('| '+' | '.join(str(row.get(c,'')) for c in columns)+' |')


def report(jobs):
    from experiments.main_cil_results import table
    main_rows = table(jobs.values())
    for row in main_rows:
        row['estimate'] = f"{row['mean']:.2f} ± {row['std']:.2f}"
        delta = row['paired_delta']
        row['delta'] = f"{delta['mean']:.2f} ± {delta['std']:.2f}" if delta else None
    markdown('Main CIL B0 Inc5', main_rows, ['arm','backbone','class_order','metric','seeds',
             'estimate','delta','paired_seeds','seeds_won','ratio_vs_icarl'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+')
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--json-out')
    parser.add_argument('--stability-figure')
    args = parser.parse_args()
    jobs = collect(args.archives, args.run_tag)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(dict(jobs={k:v['summary'] for k,v in jobs.items()}), indent=2)+'\n')
    report(jobs)
    if args.stability_figure:
        from experiments.main_cil_results import stability_plasticity
        stability_plasticity(jobs.values(), args.stability_figure)


if __name__ == '__main__': main()
