"""Build the offline GroupNorm virtual-channel definition A/B notebook (no training)."""
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

def pilot_dir(seed):
    found=[p.parent.parent.parent.parent for p in Path('/kaggle/input').rglob('latest.pt')
           if p.parent.name=='end_task1' and p.parents[3].name==f'icarl_vna__gn__s{seed}']
    found=[d for d in found if (d/'run'/'protocol.json').exists() and (d/'run'/'interventions.json').exists()]
    if not found:raise FileNotFoundError(f'Attach the GN pilot output with icarl_vna__gn__s{seed}')
    return found[0]

PILOTS={seed:pilot_dir(seed) for seed in (0,1)}
SCHEDULE=next(Path('/kaggle/input').rglob('gn_vna.json'))
COMMITS={read(PILOTS[s]/'run'/'protocol.json')['code_commit'] for s in (0,1)}
if len(COMMITS)!=1:raise ValueError(f'GN pilot seeds ran on different commits: {COMMITS}')
PILOT_COMMIT=COMMITS.pop()
print('GN pilot commit',PILOT_COMMIT,'schedule',SCHEDULE,{s:str(d) for s,d in PILOTS.items()},flush=True)
PILOT_SRC=Path('/kaggle/working')/f'pilot_source_{PILOT_COMMIT[:12]}'
if not (PILOT_SRC/'.git').exists():
    private_clone(MAIN_URL,PILOT_SRC,PILOT_COMMIT)
    subprocess.run(['bash','scripts/fetch_third_party.sh'],cwd=PILOT_SRC,check=True)

def job(gpu,seed):
    out=ROOT/f'gn_ab__s{seed}'
    if (out/'gn_definition_ab.json').exists():return True
    cmd=[sys.executable,'-u',str(REPO/'diagnostics'/'gn_definition_ab.py'),'--repo',str(PILOT_SRC),'--pilot',str(PILOTS[seed]),
         '--schedule',str(SCHEDULE),'--data-dir',str(DATA_ROOT),'--out',str(out)]
    return launch_logged(cmd,cwd=PILOT_SRC,env={**gpu_env(gpu),'PYTHONPATH':str(PILOT_SRC)},log_path=RUN_ROOT/f'gn_ab__s{seed}.log',
                         lane=gpu,job=f'gn_ab__s{seed}',lanes=lane_state,paused=session_paused,deadline=VNA_DEADLINE)

try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(job,0,0),pool.submit(job,1,1)]:future.result()
finally:
    archive_stop.set();archiver.join(timeout=10);archive_outputs()
'''

REPORT=r'''FIELDS=('lambda_E','V_O','h','g_dot_d','dFd','mu','s','s_F_star','skip_reason','rejected','L_before','L_after',
        'KD_before','KD_after','ggn_lambda_max_before','ggn_lambda_max_after')
for seed in (0,1):
    r=read(ROOT/f'gn_ab__s{seed}'/'gn_definition_ab.json',{})
    if not r:print('seed',seed,'MISSING; see log');continue
    print(f'\nseed {seed}: definition A matches the pilot record: {r["A_matches_pilot"]}')
    print(f'{"quantity":24s} {"A (bypass GN)":>22s} {"B (own-channel norm)":>22s}')
    for k in FIELDS:print(f'{k:24s} {str(r["A"].get(k)):>22s} {str(r["B"].get(k)):>22s}')
    for name in ('A','B'):
        layers=r[name]['d_norm_by_layer'] or r[name]['step_norm_by_layer']
        label='|d| by layer' if r[name]['d_norm_by_layer'] else '|s d| by layer (step rejected or skipped)'
        print(f'{name} {label}:',json.dumps({k:round(v,6) for k,v in layers.items()}))
'''


def build(commit):
    result=notebook(commit)
    cells=result['cells']
    config=f"""REPO_COMMIT = {commit!r}
SOURCE_DIGEST = None
RUN_TAG = 'vna_gn_definition_ab'
import time
SESSION_HOURS = 2
SESSION_START = time.time()
VNA_DEADLINE = SESSION_START + SESSION_HOURS*3600
DISK_CAP_BYTES = 10_000_000_000
"""
    markdown=cell('markdown','''# GroupNorm virtual-channel definition: offline A/B (no training)
Enable Internet, T4x2 and the `github_token` secret. Attach CIFAR-100 and the GN pilot output (`icarl_vna__gn__s0/s1` with `stability_gap/checkpoints/end_task1/latest.pt`, `run/protocol.json`, `run/interventions.json`, and `_schedules/gn_vna.json`).

For each seed the pilot's start-of-task-2 checkpoint is resumed on the pilot's own commit; the native code takes the first task-2 SGD step and, at iteration 2001, the VNA application is computed on copies under (A) the registered bypass of GroupNorm and (B) per-sample spatial normalization of the virtual channel alone (gamma 1, beta 0; runtime patch, not committed). Printed side by side: lambda_E, V_O, h, g.d, d^T F d, mu, s, L/KD before/after, GGN lambda_max before/after and |d| per layer. A is checked against the pilot's recorded intervention. No run is modified.
''')
    bootstrap=''.join(cells[2]['source']).replace('/kaggle/working/vna4_main','/kaggle/working/vna_gn_definition_ab')
    runtime=''.join(cells[4]['source'])
    runtime=runtime.replace("restore_stage_archives(Path('/kaggle/input'), STAGE_B, 'outputs')","ROOT=STAGE_B/RUN_TAG\nROOT.mkdir(parents=True,exist_ok=True)")
    runtime=runtime.replace('folders=("outputs",)','folders=("outputs/vna_gn_definition_ab",)')
    result['cells']=[markdown,cell('code',config+'\n'+bootstrap+'\n'+''.join(cells[3]['source'])),
                     cell('code',runtime),cell('code',RUNNER),cell('code',REPORT)]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--output',default=str(ROOT/'notebooks/vna5_gn_definition_ab.ipynb'))
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(build(args.commit),indent=1)+'\n')


if __name__=='__main__':main()
