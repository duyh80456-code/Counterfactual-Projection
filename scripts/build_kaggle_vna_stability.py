"""Build the stability-gap replay (P1, tasks 1-5) + GroupNorm V_O/h notebook."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_kaggle_pilot_common import notebook,cell
from scripts.build_kaggle_vna_parallel import OVERLAY

RUNNER=r'''from concurrent.futures import ThreadPoolExecutor

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,ValueError):return default

errors=[]

GN_SCHEDULE=ROOT/'_schedules'/'gn_vna.json'
schedule_lock=threading.Lock()

def gn_schedule():
    # GroupNorm K0 authority for the VNA arm (P1 protocol, Gram/power10), built once.
    with schedule_lock:
        if GN_SCHEDULE.exists():return True
        cmd=[sys.executable,'-u','-m','experiments.main_cil','--arm','icarl_vna','--backbone','resnet32_gn',
             '--class-order','superclass','--seed','0','--nobn-lr-scale','0.5','--solver','gram','--curvature','power10',
             '--device','cuda:0','--data-dir',str(DATA_ROOT),'--out',str(ROOT/'_preflight'/'gn'),'--preflight-only',
             '--schedule-output',str(GN_SCHEDULE),'--schedule-source','self']
        ok=launch(cmd,0,'gn_preflight',RUN_ROOT/'gn_preflight.log')
        if GN_SCHEDULE.exists():print('GN PREFLIGHT:',GN_SCHEDULE.read_text(),flush=True)
        return ok and GN_SCHEDULE.exists()

def gn_pilot(gpu,arm,seed):
    out=ROOT/f'{arm}__gn__s{seed}'
    if read(out/'stability_gap'/'report.json',{}).get('status')=='OK':return True
    cmd=[sys.executable,'-u','-m','diagnostics.gn_pilot','--arm',arm,'--seed',str(seed),'--out',str(out),'--data-dir',str(DATA_ROOT)]
    if arm=='icarl_vna':
        if not gn_schedule():return False
        cmd+=['--schedule',str(GN_SCHEDULE)]
    return launch(cmd,gpu,f'{arm}__gn__s{seed}',RUN_ROOT/f'{arm}__gn__s{seed}.log')

# P1 replays are complete (attach their output). One seed per GPU: GN baseline, then GN VNA.
def lane(gpu,seed):
    for name,work in ((f'icarl__gn__s{seed}',lambda:gn_pilot(gpu,'icarl',seed)),
                      (f'icarl_vna__gn__s{seed}',lambda:gn_pilot(gpu,'icarl_vna',seed))):
        if pause_due():return
        try:
            if not work():
                if pause_due():return
                errors.append(dict(job=name,error='non-zero exit; see log'))
        except Exception as error:errors.append(dict(job=name,error=f'{type(error).__name__}: {error}'))
        archive_outputs()

try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        for job in [pool.submit(lane,0,0),pool.submit(lane,1,1)]:job.result()
finally:
    archive_stop.set();archiver.join(timeout=10);archive_outputs()
if session_paused.is_set():print('PAUSED: attach this output and rerun to continue.')
if errors:print('JOB ERRORS:',json.dumps(errors,indent=1),flush=True)
'''

REPORT=r'''import matplotlib.pyplot as plt
# GroupNorm pilot first (independent of the P1 replays).
for seed in (0,1):
    for arm in ('icarl','icarl_vna'):
        r=read(ROOT/f'{arm}__gn__s{seed}'/'stability_gap'/'report.json',{})
        if r.get('status')!='OK':print('GN',arm,seed,'MISSING');continue
        print('GN',arm,seed,'task1 acc',r['task1_end_cnn_top1'],'run',r['run_status'])
        for t in r['tasks']:
            print(json.dumps(dict(job=f'{arm}__gn__s{seed}',task=t['task'],pre_switch_old=t['pre_switch_old'],min_old=t['min_old'],
                min_step=t['min_old_step'],gap_depth=t['gap_depth'],recovery_to_pre=t['recovery_step_to_pre_switch'],
                end_old=t['end_old'],end_new=t['end_new'])))
            for e in t['interventions']:print(json.dumps(dict(job=f'{arm}__gn__s{seed}',task=t['task'],**e)))
reports={job:read(ROOT/job/'stability_gap'/'report.json',{}) for job in ('icarl__s0','icarl_full__s0','icarl__s1','icarl_full__s1')}
ok={job:r for job,r in reports.items() if r.get('status')=='OK'}
for job,r in reports.items():
    if job not in ok:
        status='MISMATCH' if r.get('status')=='MISMATCH_NO_REPORT' else (r.get('status') or 'MISSING')
        print(job,status,json.dumps(read(ROOT/job/'stability_gap'/'match.json',{}))[:1500])
rows=[]
for job,r in ok.items():
    for t in r['tasks']:
        rows.append(dict(job=job,task=t['task'],pre_switch_old=t['pre_switch_old'],r0_old=t['r0_old'],min_old=t['min_old'],
            min_step=t['min_old_step'],gap_depth=t['gap_depth'],recovery_to_pre=t['recovery_step_to_pre_switch'],
            recovery_to_half=t['recovery_step_to_half_gap'],end_old=t['end_old'],end_new=t['end_new']))
print('STABILITY GAP, PASS replays only (old-class CNN accuracy, %; steps = SGD updates since the task switch):')
for row in rows:print(json.dumps(row))
for job,r in ok.items():
    if job.startswith('icarl_full'):
        for t in r['tasks']:
            for e in t['interventions']:print(json.dumps(dict(job=job,task=t['task'],**e)))
fig,axes=plt.subplots(2,4,figsize=(20,8),sharey=True)
for i,seed in enumerate((0,1)):
    for j,task in enumerate((2,3,4,5)):
        ax=axes[i][j]
        for arm,style in (('icarl','-'),('icarl_full','--')):
            job=f'{arm}__s{seed}'
            if job not in ok:
                ax.plot([],[],style,label=f'{arm} ({"MISMATCH" if reports[job].get("status")=="MISMATCH_NO_REPORT" else "MISSING"})');continue
            t=next(x for x in ok[job]['tasks'] if x['task']==task)
            pts=[p for p in t['curve'] if p['r']<=400]
            ax.plot([p['r'] for p in pts],[p['old'] for p in pts],style,label=arm)
            if arm=='icarl_full':
                for e in t['interventions']:
                    if e['step']<=400:ax.axvline(e['step'],color='grey',alpha=.3)
            ax.axhline(t['pre_switch_old'],color='k',lw=.5)
        ax.set(title=f'seed {seed} task {task}',xlabel='step in task',ylabel='old-class acc')
        ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(ROOT/'stability_gap.png',dpi=120);plt.show()
'''


def build(commit,source_digest=None):
    result=notebook(commit)
    cells=result['cells']
    config=f'''REPO_COMMIT = {commit!r}
SOURCE_DIGEST = {source_digest!r}
RUN_TAG = 'vna_stability_gap'
import time
SESSION_HOURS = 11.5
SESSION_START = time.time()
VNA_DEADLINE = SESSION_START + SESSION_HOURS*3600
DISK_CAP_BYTES = 20_000_000_000
'''
    markdown=cell('markdown','''# Stability gap (P1 replay, tasks 1-5) and GroupNorm V_O/h
Enable Internet, T4x2 and the `github_token` secret. Attach CIFAR-100 (`cifar-100-python/{train,test,meta}`) and this notebook's previous output (it holds the finished P1 replays). This session runs the GroupNorm pilot (P1 protocol, tasks 1-5): per GPU one seed, baseline icarl then VNA (Gram/power10, GIR off) after a GN K0 preflight; the summary cell reports PASS replays and marks the others MISMATCH.

Each P1 job is replayed through task 5 with the same K0 on the commit that ran its tasks 1-5 (VNA: metric_trace_repair.json source_commit if present, else schedule.code_commit; baseline: its seed's VNA commit), one checkout per commit; the commit of every job is printed before launch. Old/new-class test accuracy is read at every SGD step for r = 0-10, every 5 steps to 200 in tasks 2-5, then every epoch, under no_grad with all RNG and module modes restored. Task-boundary checkpoints (end of tasks 1-4) are kept. Epoch losses and tasks.json for tasks 1-5 must equal P1 exactly; otherwise no measurement is reported.

GroupNorm pilot: ResNet-32 with GroupNorm(8) (zero-init gamma of each block's last GroupNorm), P1 protocol (Inc5, superclass, lr 0.05, no warmup), tasks 1-5, seeds 0/1. Arms: baseline icarl and icarl_vna (VNA Gram/power10, GIR off because T29e does not hold under GroupNorm). Old/new test accuracy per step with the replay's reader; every VNA intervention is logged (V_O, h, L/KD before/after, accepted or rejected and why). Task-1 accuracy is recorded, not gated.
''')
    bootstrap=''.join(cells[2]['source']).replace('/kaggle/working/vna4_main','/kaggle/working/vna_stability_gap')+OVERLAY
    runtime=''.join(cells[4]['source'])
    runtime=runtime.replace("restore_stage_archives(Path('/kaggle/input'), STAGE_B, 'outputs')","ROOT=STAGE_B/RUN_TAG\nROOT.mkdir(parents=True,exist_ok=True)\nrestore_stage_archives(Path('/kaggle/input'), ROOT, RUN_TAG)")
    runtime=runtime.replace('folders=("outputs",)','folders=("outputs/vna_stability_gap",)')
    result['cells']=[markdown,cell('code',config+'\n'+bootstrap+'\n'+''.join(cells[3]['source'])),
                     cell('code',runtime),cell('code',RUNNER),cell('code',REPORT)]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--output',default=str(ROOT/'notebooks/vna5_stability_gap.ipynb'))
    args=parser.parse_args()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(build(args.commit),indent=1)+'\n')


if __name__=='__main__':main()
