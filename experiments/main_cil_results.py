"""Seed-paired main CIL tables and an exportable stability–plasticity figure."""
from collections import defaultdict
import statistics


def mean_std(values):
    values = list(values)
    return dict(mean=statistics.mean(values), std=statistics.stdev(values) if len(values)>1 else 0.)


def configuration_signature(p):
    backbone={'bn':'resnet32','nobn':'resnet32_nobn'}.get(p.get('backbone'),p.get('backbone'))
    result={k:p.get(k) for k in ('dataset_fingerprint','upstream_commit','training','expected_tasks','reproduction','source_digest')}
    result.update(backbone=backbone,order=p.get('class_order_kind',p.get('order')),
                  lr_scale=p.get('nobn_lr_scale',1.) if backbone=='resnet32_nobn' else 1.)
    if p.get('benchmark')=='v4_competitor':
        cfg={k:v for k,v in p.get('cfg',{}).items() if k not in ('seed','device')}
        result.update(cfg=cfg,method_config=p.get('method_config'),parameter_policy=p.get('parameter_policy'),pins=p.get('competitor_pins'),implementation=p.get('implementation'))
    else:
        result.update(comparison=p.get('comparison_protocol'),cfg={k:v for k,v in p.get('cfg',{}).items() if k not in ('seed','device')})
        if p.get('arm')!='icarl':
            result.update({k:p.get(k) for k in ('solver','curvature','policy_version','gir_live','coefficient','r','compute_budget')})
            authority=p.get('schedule') or {}
            result['K0']=authority.get('K0',p.get('K0'))
            result['calibration']=authority.get('calibration_code_commit',authority.get('code_commit'))
    return result


def table(records):
    records = [r for r in records if (r['protocol'].get('algorithm_version') == 'v4'
               and r['protocol'].get('setting') == 'icarl') or
               r['protocol'].get('benchmark') == 'v4_competitor']
    records = [{**r, 'protocol': {**r['protocol'],
                'backbone': {'bn':'resnet32','nobn':'resnet32_nobn'}.get(r['protocol']['backbone'],r['protocol']['backbone']),
                'class_order_kind': r['protocol'].get('class_order_kind',r['protocol'].get('order'))+
                    ('/reproduction' if r['protocol'].get('reproduction') else '')}}
               for r in records]
    grouped = defaultdict(list)
    controls = {}
    for r in records:
        p = r['protocol']; group = (p['backbone'], p['class_order_kind'])
        grouped[(p['arm'], *group)].append(r)
        if p['arm'] == 'icarl': controls[(*group, p['seed'])] = r['summary']
    rows = []
    for (arm, backbone, order), runs in sorted(grouped.items()):
        signature=configuration_signature(runs[0]['protocol'])
        if any(configuration_signature(r['protocol'])!=signature for r in runs):
            raise ValueError(f'{arm}: mixed solver/curvature/LR/configuration across seeds')
        if len({r['protocol']['seed'] for r in runs})!=len(runs):raise ValueError(f'{arm}: duplicate seed results')
        for metric in [f'{label}_{name}' for label in ('cnn','nme') for name in
                       ('average_accuracy','last_accuracy','learning_accuracy','forgetting')]+['gpu_hours']:
            values, deltas, ratios = [], [], []
            for r in runs:
                summary, p = r['summary'], r['protocol']
                values.append(summary[metric])
                baseline = controls.get((backbone, order, p['seed']))
                if baseline:
                    deltas.append(summary[metric]-baseline[metric])
                    if metric == 'gpu_hours' and baseline[metric]>0: ratios.append(summary[metric]/baseline[metric])
            lower = metric.endswith('forgetting') or metric=='gpu_hours'
            rows.append(dict(arm=arm, backbone=backbone, class_order=order, metric=metric,
                seeds=len(values), **mean_std(values), paired_seeds=len(deltas),
                paired_delta=mean_std(deltas) if deltas else None,
                seeds_won=sum(d<0 if lower else d>0 for d in deltas),
                ratio_vs_icarl=mean_std(ratios) if ratios else None))
    return rows


def stability_plasticity(records, path, label='nme'):
    from html import escape
    from pathlib import Path
    if Path(path).suffix.lower() != '.svg': raise ValueError('stability figure must use .svg')
    rows = table(records)
    points = []
    for row in rows:
        if row['metric'] != label+'_forgetting': continue
        y = next(r for r in rows if all(r[k] == row[k] for k in ('arm','backbone','class_order'))
                 and r['metric'] == label+'_learning_accuracy')
        points.append((row,y))
    xmin = min([r['mean']-r['std'] for r,y in points]+[0.])
    xmax = max([r['mean']+r['std'] for r,y in points]+[1.])*1.1
    ymax = max([y['mean']+y['std'] for r,y in points]+[1.])*1.1
    x = lambda v: 80+640*(v-xmin)/(xmax-xmin)
    y = lambda v: 350-300*v/ymax
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{440+20*len(points)}" viewBox="0 0 800 {440+20*len(points)}">',
             '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="13">',
             '<path d="M80 50 V350 H720" fill="none" stroke="black"/>',
             '<text x="400" y="395" text-anchor="middle">Final per-class forgetting (pp)</text>',
             '<text transform="translate(20 200) rotate(-90)" text-anchor="middle">Mean learning accuracy (%)</text>']
    for i in range(6):
        tick = xmin+(xmax-xmin)*i/5
        parts.extend([f'<text x="{x(tick)}" y="375" text-anchor="middle">{tick:.1f}</text>',
                      f'<text x="70" y="{y(ymax*i/5)+4}" text-anchor="end">{ymax*i/5:.1f}</text>'])
    colors = ('#2563eb','#dc2626','#059669','#9333ea','#d97706','#0891b2')
    for i,(r,yp) in enumerate(points):
        color = colors[i%len(colors)]; cx,cy=x(r['mean']),y(yp['mean'])
        left,right=x(r['mean']-r['std']),x(r['mean']+r['std'])
        top,bottom=y(yp['mean']+yp['std']),y(yp['mean']-yp['std'])
        name = escape(f"{r['arm']} / {r['backbone']} / {r['class_order']}")
        parts.extend([f'<path d="M{left} {cy} H{right} M{left} {cy-4} V{cy+4} M{right} {cy-4} V{cy+4} M{cx} {top} V{bottom} M{cx-4} {top} H{cx+4} M{cx-4} {bottom} H{cx+4}" stroke="{color}"/>',
                      f'<circle cx="{cx}" cy="{cy}" r="4" fill="{color}"><title>{name}</title></circle>',
                      f'<text x="80" y="{430+20*i}" fill="{color}">{name} (± sample std)</text>'])
    parts.append('</g></svg>')
    Path(path).write_text('\n'.join(parts)+'\n')
