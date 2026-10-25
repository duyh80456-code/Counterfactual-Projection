"""Build the offline VNA-chain notebook (GN pilot s0/s1 and noBN icarl_full s0; no training)."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_kaggle_pilot_common import notebook,cell

RUNNER=r'''from concurrent.futures import ThreadPoolExecutor
from experiments.vna_resume import launch_logged

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,ValueError):return default

def job_dir(name):
    found=[p.parents[3] for p in Path('/kaggle/input').rglob('latest.pt')
           if p.parent.name=='end_task1' and p.parents[3].name==name and (p.parents[3]/'run'/'protocol.json').exists()]
    if not found:raise FileNotFoundError(f'Attach the output containing {name}/stability_gap/checkpoints/end_task1')
    return found[0]

JOBS={'gn_vna__s0':'icarl_vna__gn__s0','gn_vna__s1':'icarl_vna__gn__s1','nobn_full__s0':'icarl_full__s0'}
DIRS={k:job_dir(v) for k,v in JOBS.items()}
COMMITS={k:read(d/'run'/'protocol.json')['code_commit'] for k,d in DIRS.items()}
for k in JOBS:print(f'CHAIN {k}: commit {COMMITS[k]}  source {DIRS[k]}',flush=True)
SOURCES={}
for commit in sorted(set(COMMITS.values())):
    source=Path('/kaggle/working')/f'job_source_{commit[:12]}'
    if not (source/'.git').exists():
        private_clone(MAIN_URL,source,commit)
        subprocess.run(['bash','scripts/fetch_third_party.sh'],cwd=source,check=True)
    SOURCES[commit]=source

def chain(gpu,name):
    out=ROOT/name
    if (out/'vna_chain.json').exists():return True
    source=SOURCES[COMMITS[name]]
    cmd=[sys.executable,'-u',str(REPO/'diagnostics'/'vna_chain.py'),'--repo',str(source),'--job-dir',str(DIRS[name]),
         '--name',name,'--data-dir',str(DATA_ROOT),'--out',str(out)]
    return launch_logged(cmd,cwd=source,env={**gpu_env(gpu),'PYTHONPATH':str(source)},log_path=RUN_ROOT/f'{name}.log',
                         lane=gpu,job=name,lanes=lane_state,paused=session_paused,deadline=VNA_DEADLINE)

def lane(gpu,names):
    for name in names:
        if pause_due():return
        if not chain(gpu,name):print('CHAIN FAILED',name,'see log',flush=True)

try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(lane,0,['gn_vna__s0','nobn_full__s0']),pool.submit(lane,1,['gn_vna__s1'])]:future.result()
finally:
    archive_stop.set();archiver.join(timeout=10);archive_outputs()
'''

REPORT=r'''COLUMNS=('k','s','s_times_d','L_window','KD_window','ggn_lambda_max','V_O','h','old_acc','new_acc','mu','skip_reason','rejected','stop')
for name in ('gn_vna__s0','gn_vna__s1','nobn_full__s0'):
    r=read(ROOT/name/'vna_chain.json',{})
    if not r:print(name,'MISSING; see log');continue
    print(f'\n{name} ({r["backbone"]}, commit {r["code_commit"][:7]}); k=0 is the state after the first task-2 SGD step')
    print(' | '.join(f'{c:>12s}' for c in COLUMNS))
    for row in r['rows']:
        print(' | '.join(f'{(round(row.get(c),5) if isinstance(row.get(c),float) else row.get(c))!s:>12s}' for c in COLUMNS))
'''


def build(commit):
    result=notebook(commit)
    cells=result['cells']
    config=f"""REPO_COMMIT = {commit!r}
SOURCE_DIGEST = None
RUN_TAG = 'vna_chain'
import time
SESSION_HOURS = 2
SESSION_START = time.time()
VNA_DEADLINE = SESSION_START + SESSION_HOURS*3600
DISK_CAP_BYTES = 10_000_000_000
"""
    markdown=cell('markdown','''# Offline chain of VNA applications at the first task-2 intervention (no training)
Enable Internet, T4x2 and the `github_token` secret. Attach CIFAR-100, the GN pilot output (`icarl_vna__gn__s0/s1`) and the stability-gap replay output (`icarl_full__s0`, noBN, for comparison); each needs `run/protocol.json` and `stability_gap/checkpoints/end_task1/latest.pt`.

Each job resumes its start-of-task-2 checkpoint on its own commit; the native code takes the first task-2 SGD step, then up to 8 successive VNA applications (registered definition, VNA step only, GIR off on the copied engine) are applied to a copy on the same window, each fully recomputed. Stop when the GGN lambda_max < 2/eta = 40 or a step is rejected/skipped. After each: s, |s d|, window L/KD, GGN lambda_max, V_O, h, old/new test accuracy. No run is modified.
''')

    bootstrap=''.join(cells[2]['source']).replace('/kaggle/working/vna4_main','/kaggle/working/vna_chain')
    runtime=''.join(cells[4]['source'])
    runtime=runtime.replace("restore_stage_archives(Path('/kaggle/input'), STAGE_B, 'outputs')","ROOT=STAGE_B/RUN_TAG\nROOT.mkdir(parents=True,exist_ok=True)")
    runtime=runtime.replace('folders=("outputs",)','folders=("outputs/vna_chain",)')
    result['cells']=[markdown,cell('code',config+'\n'+bootstrap+'\n'+''.join(cells[3]['source'])),
                     cell('code',runtime),cell('code',RUNNER),cell('code',REPORT)]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--output',default=str(ROOT/'notebooks/vna5_vna_chain.ipynb'))
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(build(args.commit),indent=1)+'\n')


if __name__=='__main__':main()
