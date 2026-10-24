"""Build the stability-gap replay (P1, tasks 1-5) + GroupNorm V_O/h notebook."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_kaggle_pilot_common import notebook,cell
from scripts.build_kaggle_vna_parallel import OVERLAY

RUNNER=r'''import queue
from concurrent.futures import ThreadPoolExecutor
from experiments.vna_resume import launch_logged
POLICY='schur_gamma_bound_rho25_v2'

def read(path,default=None):
    try:return json.loads(Path(path).read_text())
    except (FileNotFoundError,ValueError):return default

def p1_references():
    """Attached registered P1 jobs: noBN, superclass, lr scale 0.5, 20-task protocol, >= 5 tasks."""
    found={}
    for path in Path('/kaggle/input').rglob('protocol.json'):
        p=read(path,{})
        arm,seed=p.get('arm'),p.get('seed')
        if (p.get('setting')!='icarl' or p.get('algorithm_version')!='v4' or arm not in ('icarl','icarl_full')
                or seed not in (0,1) or p.get('backbone')!='resnet32_nobn' or p.get('class_order_kind')!='superclass'
                or p.get('nobn_lr_scale')!=.5 or 'max_tasks' in p.get('comparison_protocol',{})
                or p.get('fake_data') or p.get('preflight_only')):continue
        if arm=='icarl_full' and (p.get('solver'),p.get('curvature'),p.get('policy_version'))!=('gram','power10',POLICY):continue
        tasks=read(path.parent/'tasks.json',[]);epochs=read(path.parent/'epochs.json',[])
        if len([t for t in tasks if t.get('task',99)<=5])<5 or len([e for e in epochs if e.get('task',99)<=5])<300:continue
        prefix=json.dumps(dict(e=[e['train_loss'] for e in epochs if e['task']<=5],t=[t for t in tasks if t['task']<=5]),sort_keys=True)
        found.setdefault(f'{arm}__s{seed}',{}).setdefault(prefix,path.parent)
    jobs={}
    for job in ('icarl_full__s0','icarl_full__s1','icarl__s0','icarl__s1'):
        candidates=list(found.get(job,{}).values())
        if not candidates:raise FileNotFoundError(f'Attach the P1 output containing {job} (>=5 tasks, epochs.json)')
        if len(candidates)>1:raise ValueError(f'{job}: attached P1 copies disagree on tasks 1-5: {candidates}')
        jobs[job]=candidates[0]
    commits={read(jobs[j]/'protocol.json')['schedule']['code_commit'] for j in ('icarl_full__s0','icarl_full__s1')}
    if len(commits)!=1:raise ValueError(f'P1 VNA seeds use different schedule commits: {commits}')
    return jobs,commits.pop()

REFERENCES,P1_COMMIT=p1_references()
print('P1 REFERENCES:',json.dumps({k:str(v) for k,v in REFERENCES.items()},indent=1),'\nP1 COMMIT:',P1_COMMIT,flush=True)
P1_SRC=Path('/kaggle/working/p1_source')
if not (P1_SRC/'.git').exists():
    private_clone(MAIN_URL,P1_SRC,P1_COMMIT)
    subprocess.run(['bash','scripts/fetch_third_party.sh'],cwd=P1_SRC,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P1_SRC,text=True).strip()
if head!=P1_COMMIT:raise RuntimeError(f'P1 checkout is {head}, expected {P1_COMMIT}')
errors=[]

def replay(gpu,job):
    out=ROOT/job
    if read(out/'stability_gap'/'report.json',{}).get('status')=='OK':return True
    env={**gpu_env(gpu),'PYTHONPATH':str(P1_SRC)}
    cmd=[sys.executable,'-u',str(REPO/'diagnostics'/'stability_gap.py'),'--repo',str(P1_SRC),
         '--reference',str(REFERENCES[job]),'--job',job,'--out',str(out),'--data-dir',str(DATA_ROOT)]
    return launch_logged(cmd,cwd=P1_SRC,env=env,log_path=RUN_ROOT/f'{job}.log',lane=gpu,job=job,
                         lanes=lane_state,paused=session_paused,deadline=VNA_DEADLINE)

def groupnorm(gpu,seed):
    out=ROOT/f'groupnorm__s{seed}'
    if (out/'gn_vo_h_summary.json').exists():return True
    cmd=[sys.executable,'-u','-m','diagnostics.gn_vo_h','--seed',str(seed),'--out',str(out),'--data-dir',str(DATA_ROOT)]
    return launch(cmd,gpu,f'groupnorm__s{seed}',RUN_ROOT/f'groupnorm__s{seed}.log')

def lane(gpu,seed):
    for name,work in ((f'icarl_full__s{seed}',lambda:replay(gpu,f'icarl_full__s{seed}')),
                      (f'icarl__s{seed}',lambda:replay(gpu,f'icarl__s{seed}')),
                      (f'groupnorm__s{seed}',lambda:groupnorm(gpu,seed))):
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
reports={job:read(ROOT/job/'stability_gap'/'report.json',{}) for job in ('icarl__s0','icarl_full__s0','icarl__s1','icarl_full__s1')}
bad={job:(r.get('status') or 'MISSING') for job,r in reports.items() if r.get('status')!='OK'}
if bad:
    for job in bad:print(job,'MATCH:',json.dumps(read(ROOT/job/'stability_gap'/'match.json',{}),indent=1)[:3000])
    raise RuntimeError(f'Stability-gap measurements withheld until every replay matches P1 exactly: {bad}')
rows=[]
for job,r in reports.items():
    for t in r['tasks']:
        rows.append(dict(job=job,task=t['task'],pre_switch_old=t['pre_switch_old'],r0_old=t['r0_old'],min_old=t['min_old'],
            min_step=t['min_old_step'],gap_depth=t['gap_depth'],recovery_to_pre=t['recovery_step_to_pre_switch'],
            recovery_to_half=t['recovery_step_to_half_gap'],end_old=t['end_old'],end_new=t['end_new']))
print('STABILITY GAP (old-class CNN accuracy, %; steps = SGD updates since the task switch):')
for row in rows:print(json.dumps(row))
for seed in (0,1):
    for t in reports[f'icarl_full__s{seed}']['tasks']:
        for e in t['interventions']:print(json.dumps(dict(seed=seed,task=t['task'],**e)))
fig,axes=plt.subplots(2,4,figsize=(20,8),sharey=True)
for i,seed in enumerate((0,1)):
    for j,task in enumerate((2,3,4,5)):
        ax=axes[i][j]
        for arm,style in (('icarl','-'),('icarl_full','--')):
            t=next(x for x in reports[f'{arm}__s{seed}']['tasks'] if x['task']==task)
            pts=[p for p in t['curve'] if p['r']<=400]
            ax.plot([p['r'] for p in pts],[p['old'] for p in pts],style,label=arm)
            if arm=='icarl_full':
                for e in t['interventions']:
                    if e['step']<=400:ax.axvline(e['step'],color='grey',alpha=.3)
            ax.axhline(t['pre_switch_old'],color='k',lw=.5)
        ax.set(title=f'seed {seed} task {task}',xlabel='step in task',ylabel='old-class acc')
        ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(ROOT/'stability_gap.png',dpi=120);plt.show()
for seed in (0,1):print('GROUPNORM',seed,json.dumps(read(ROOT/f'groupnorm__s{seed}'/'gn_vo_h_summary.json',{})))
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
Enable Internet, T4x2 and the `github_token` secret. Attach CIFAR-100 (`cifar-100-python/{train,test,meta}`) and the P1 output(s) containing `icarl` and `icarl_full` (Gram/power10) seeds 0 and 1, noBN, superclass. Attach this notebook's own output to resume.

Each P1 job is replayed on its own source commit (taken from the attached P1 VNA schedule) through task 5 with the same K0. Old/new-class test accuracy is read at every SGD step for r = 0-10, every 5 steps to 200 in tasks 2-5, then every epoch, under no_grad with all RNG and module modes restored. Task-boundary checkpoints (end of tasks 1-4) are kept. Epoch losses and tasks.json for tasks 1-5 must equal P1 exactly; otherwise no measurement is reported.

GroupNorm: baseline iCaRL on ResNet-32 with GroupNorm(8) trains task 1; in task 2 (every step for r = 0-10, every 5 steps to 200) a VNA engine (Gram/power10) applies one full T26 application to a discarded copy of the network and logs V_O, h, L/KD before/after, s, Delta and would_accept (T26f); the real model is checked bitwise unchanged. The virtual neuron bypasses GroupNorm (a definition choice, see preregistration). Output: `outputs/vna_stability_gap/`.
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
