"""Collect completed real Phase0 artifacts, plot diagnostics and enforce the stop gate."""
import argparse
import json
from pathlib import Path
import zipfile
from experiments.io_utils import atomic_json_save,write_csv
from ett.phase0.measure_ge import summarize


def report(roots,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=[];commits=set()
    for root in map(Path,roots):
        if (root/'VALIDITY.json').exists() and json.loads((root/'VALIDITY.json').read_text()).get('status')=='INVALID_WINDOW':
            raise ValueError(f'INVALID_WINDOW: {root}')
        state=json.loads((root/'status.json').read_text())
        if state['status']!='COMPLETED':raise ValueError(f'incomplete Phase0 setting: {root}')
        commits.add(state['commit'])
        rows.extend(json.loads(line) for line in (root/'measurements.jsonl').read_text().splitlines())
    if len(commits)!=1:raise ValueError('mixed source commits in Phase0')
    tables=summarize(rows)
    if {r['setting'] for r in tables}!={'scr','pmnist'}:raise ValueError('Phase0 needs both registered settings')
    result=dict(commit=next(iter(commits)),tables=tables,gate_pass=any(r['gate'] for r in tables),
        next_action='STOP: report and await instruction; never launch Phase1 automatically')
    atomic_json_save(result,out/'phase0_report.json');write_csv(out/'summary.csv',tables)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for setting in ('scr','pmnist'):
        fig,axes=plt.subplots(1,3,figsize=(12,3))
        for ax,metric in zip(axes,('GE','delta','net')):
            for h in (1,10,100):
                points=sorted([r for r in rows if r['setting']==setting and r['h']==h],key=lambda r:r['step'])
                ax.plot([r['step'] for r in points],[r[metric] for r in points],label=f'h={h}')
            ax.axhline(0,color='black',linewidth=.5);ax.set_title(metric);ax.set_xlabel('Native BP step');ax.legend()
        fig.tight_layout();fig.savefig(out/f'{setting}.png',dpi=160);plt.close(fig)
    with zipfile.ZipFile(out/'phase0_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for root in map(Path,roots):
            for path in root.rglob('*'):
                if path.suffix in ('.json','.jsonl','.csv','.log'):archive.write(path,f'{root.name}/{path.relative_to(root)}')
        for path in out.iterdir():
            if path.is_file() and path.suffix in ('.json','.csv','.png'):archive.write(path,path.name)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(report(a.roots,a.out),indent=2))
