"""Collect the six parallel Kaggle archives into preregistered result tables."""
import argparse
import csv
import io
import json
import statistics
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


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


def collect(inputs, run_tag):
    jobs = {}
    for source in inputs:
        grouped = {}
        for path, data in files(source):
            parts = path.parts
            if run_tag not in parts: continue
            index = parts.index(run_tag)
            if len(parts) <= index+2: continue
            job, name = parts[index+1], parts[-1]
            if name in {'summary.json','tasks.json','protocol.json','reproduction_gate.json','vna_log.csv'}:
                grouped.setdefault(job, {})[name] = data
        for job, payload in grouped.items():
            if 'summary.json' not in payload: continue
            record = {'job': job, 'source': str(source),
                      'summary': json.loads(payload['summary.json'])}
            for name in ('tasks.json','protocol.json','reproduction_gate.json'):
                if name in payload: record[name[:-5]] = json.loads(payload[name])
            if 'vna_log.csv' in payload:
                record['vna'] = list(csv.DictReader(io.StringIO(payload['vna_log.csv'].decode())))
            previous = jobs.get(job)
            if previous is None or record['summary'].get('status') == 'COMPLETED': jobs[job] = record
    return jobs


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
        seed=summary.get('seed', protocol.get('seed')), init_cls=summary.get('init_cls', protocol.get('init_cls', 10)),
        increment=summary.get('increment', protocol.get('increment', 10)),
        cnn_average=last.get('cnn_average_accuracy'), nme_average=last.get('nme_average_accuracy'),
        cnn_last=last.get('cnn_top1'), nme_last=last.get('nme_top1'),
        cnn_forgetting=last.get('cnn_forgetting'), nme_forgetting=last.get('nme_forgetting'),
        reference=record.get('reproduction_gate', {}).get('reference', ''))


def vna_cost(record):
    rows = record.get('vna', [])
    rho = [float(r['rho']) for r in rows if r.get('rho') not in (None,'')]
    seconds = sum(float(r.get('seconds') or 0) for r in rows)
    ks = sorted({r.get('K') for r in rows if r.get('K') not in (None,'')})
    train = record['summary'].get('training_seconds')
    if train is None: train = max(0., float(record['summary'].get('seconds',0))-seconds)
    return dict(job=record['job'], K=','.join(ks), rho_mean=mean_sd(rho),
                vna_seconds=f'{seconds:.2f}', train_seconds=f'{train:.2f}',
                vna_over_train=f'{seconds/train:.4f}' if train else '')


def report(jobs):
    cil = [cil_row(r) for name,r in sorted(jobs.items()) if name.startswith('icarl')]
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
    columns = ['arm','seed','cnn_average','nme_average','cnn_last','nme_last','cnn_forgetting','nme_forgetting']
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
                           replacement_rate=variant.get('replacement_rate'),
                           mean_last50=variant.get('mean_last50')))
    controls = {(r['seed'],r['replacement_rate']):r for r in b5 if r['arm']=='cbp'}
    b5_table = []
    for arm, rate in sorted({(r['arm'],r['replacement_rate']) for r in b5}, key=str):
        rows = [r for r in b5 if r['arm']==arm and r['replacement_rate']==rate]
        delta = [r['mean_last50']-controls[(r['seed'],rate)]['mean_last50'] for r in rows
                 if (r['seed'],rate) in controls]
        b5_table.append(dict(arm=arm, replacement_rate=rate, seeds=len(rows), mean_last50=mean_sd([r['mean_last50'] for r in rows]),
                             paired_delta_cbp=mean_sd(delta)))
    markdown('B5 regression', b5_table, ['arm','replacement_rate','seeds','mean_last50','paired_delta_cbp'])
    costs = [vna_cost(r) for r in jobs.values() if r.get('vna')]
    markdown('VNA cost', costs, ['job','K','rho_mean','vna_seconds','train_seconds','vna_over_train'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+')
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--json-out')
    args = parser.parse_args()
    jobs = collect(args.archives, args.run_tag)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({k:v['summary'] for k,v in jobs.items()}, indent=2)+'\n')
    report(jobs)


if __name__ == '__main__': main()
