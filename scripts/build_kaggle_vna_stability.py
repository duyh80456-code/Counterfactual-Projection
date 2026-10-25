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

def groupnorm(gpu,seed):
    out=ROOT/f'groupnorm__s{seed}'
    if read(out/'gn_vo_h_summary.json',{}).get('status')=='OK':return True
    if out.exists():shutil.rmtree(out)  # earlier gated attempt: restart from task 1 on this commit
    cmd=[sys.executable,'-u','-m','diagnostics.gn_vo_h','--seed',str(seed),'--out',str(out),'--data-dir',str(DATA_ROOT)]
    return launch(cmd,gpu,f'groupnorm__s{seed}',RUN_ROOT/f'groupnorm__s{seed}.log')

# The P1 replays are complete (attach their output); this session runs only GroupNorm s0/s1.
def lane(gpu,seed):
    for name,work in ((f'groupnorm__s{seed}',lambda:groupnorm(gpu,seed)),):
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
# GroupNorm first: it does not depend on the P1 replays and is printed even if a replay mismatches.
for seed in (0,1):
    summary=read(ROOT/f'groupnorm__s{seed}'/'gn_vo_h_summary.json',{})
    print('GROUPNORM',seed,json.dumps(summary,indent=1) if summary else 'MISSING')
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
Enable Internet, T4x2 and the `github_token` secret. Attach CIFAR-100 (`cifar-100-python/{train,test,meta}`) and this notebook's previous output (it holds the finished P1 replays). This session runs only the two GroupNorm jobs (s0 on GPU0, s1 on GPU1); the summary cell reports PASS replays and marks the others MISMATCH.

Each P1 job is replayed through task 5 with the same K0 on the commit that ran its tasks 1-5 (VNA: metric_trace_repair.json source_commit if present, else schedule.code_commit; baseline: its seed's VNA commit), one checkout per commit; the commit of every job is printed before launch. Old/new-class test accuracy is read at every SGD step for r = 0-10, every 5 steps to 200 in tasks 2-5, then every epoch, under no_grad with all RNG and module modes restored. Task-boundary checkpoints (end of tasks 1-4) are kept. Epoch losses and tasks.json for tasks 1-5 must equal P1 exactly; otherwise no measurement is reported.

GroupNorm (run last in each lane): baseline iCaRL on ResNet-32 with GroupNorm(8) (lr 0.05, zero-init gamma of each block's last GroupNorm) trains tasks 1-5 (task-1 accuracy is recorded, not gated); in tasks 2-5 (every step for r = 0-10, every 5 steps to 200) old/new test accuracy is read as in the replay and a VNA engine (Gram/power10) applies one full T26 application to a discarded copy of the network and logs V_O, h, L/KD before/after, s, Delta and would_accept (T26f); the real model is checked bitwise unchanged. The virtual neuron bypasses GroupNorm (a definition choice, see preregistration). Output: `outputs/vna_stability_gap/`.
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
