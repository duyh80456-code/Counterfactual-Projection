"""Fork the M1 runtime for the separately registered task-1 live GIR pilot."""
import argparse
import json
import subprocess
from pathlib import Path
from scripts.build_kaggle_vna_parallel import build as build_m1, config
from scripts.build_kaggle_pilot_common import cell

ROOT = Path(__file__).resolve().parents[1]
TAG = 'vna4_m1_gir_live_task1_s01'

RUN = r'''from scripts.collect_parallel import validate_result
ROOT = STAGE_B / REPO_COMMIT[:12] / RUN_TAG
ROOT.mkdir(parents=True, exist_ok=True)
ENV.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

def terminal(job):
    folder = ROOT/job
    if not (folder/'summary.json').exists(): return False
    summary = json.loads((folder/'summary.json').read_text())
    if summary.get('status') not in ('COMPLETED','FAILED_DIVERGED'): return False
    protocol = json.loads((folder/'protocol.json').read_text())
    identity = summary['identity']
    seed = int(job.rsplit('__s',1)[1])
    expected = dict(code_commit=REPO_COMMIT, source_digest=None, arm='icarl_full',
        seed=seed, gir_live_task1=True, backbone='resnet32_nobn',
        class_order_kind='superclass', K0=512, solver='gram', curvature='power10',
        intervention_start_task=1, nobn_lr_scale=.5, expected_tasks=20)
    if (any(identity.get(k)!=v for k,v in expected.items())
            or any(protocol.get(k)!=v for k,v in identity.items())):
        raise ValueError('Pilot result identity mismatch: '+job)
    if summary['status']=='COMPLETED':
        validate_result(job, summary, protocol, summary['tasks'], REPO_COMMIT)
    elif not summary.get('diverged'):
        raise ValueError('Divergence coordinates missing: '+job)
    print('TERMINAL',job,summary['status'],summary.get('diverged'),flush=True)
    return True

def lane(gpu,jobs):
    for job in jobs:
        if pause_due(): return
        if terminal(job): continue
        seed = job.rsplit('__s',1)[1]
        cmd = [sys.executable,'-u','-m','experiments.main_cil','--arm','icarl_full',
            '--backbone','resnet32_nobn','--class-order','superclass','--seed',seed,
            '--nobn-lr-scale','0.5','--solver','gram','--curvature','power10',
            '--gir-live-task1','--max-tasks','20','--workers','2',
            '--device','cuda:0','--data-dir',str(DATA_ROOT),'--out',str(ROOT/job)]
        print('K0=512 (fixed); GPU',gpu,'seed',seed,'task1 GIR=d88abc4 live/dead; task2+=6be6543',flush=True)
        if not launch(cmd,gpu,job,RUN_ROOT/f'{RUN_TAG}_{job}.log'): return
        terminal(job)
        archive_outputs()

try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(lane,0,JOBS_GPU0),pool.submit(lane,1,JOBS_GPU1)]
        for future in jobs: future.result()
finally:
    archive_stop.set();archiver.join(timeout=10);archive_outputs()
'''

REPORT = r'''rows=[]
for job in JOBS_GPU0+JOBS_GPU1:
    folder=ROOT/job
    summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
    events=json.loads((folder/'interventions.json').read_text()) if (folder/'interventions.json').exists() else []
    swaps=[s for e in events if e.get('task')==1 for s in e.get('swap_records',[])]
    rows.append(dict(job=job,status=summary.get('status','PAUSED_OR_PENDING'),
        gpu_hours=summary.get('gpu_hours'),diverged=summary.get('diverged'),
        task1_live=sum(s['slot_kind']=='live' for s in swaps),
        task1_dead=sum(s['slot_kind']=='dead' for s in swaps),
        task1_interventions=sum(e.get('task')==1 for e in events)))
print(json.dumps(rows,indent=2))
(ROOT/'pilot_report.json').write_text(json.dumps(rows,indent=2))
archive_outputs()
if session_paused.is_set(): print('PAUSED: attach this notebook Output to resume exactly; keep the same pin and tag.')
if archive_errors: raise RuntimeError(archive_errors)
'''


def build(commit):
    subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=ROOT,check=True)
    source=subprocess.check_output(['git','show',commit+':experiments/main_cil.py'],cwd=ROOT,text=True)
    if '--gir-live-task1' not in source: raise ValueError('Pinned commit lacks the task-1 live GIR flag')
    result=build_m1(commit,'M1',TAG,None)
    jobs=([f'icarl_full__nobn__superclass__s0'],[f'icarl_full__nobn__superclass__s1'],[])
    result['cells'][0]=cell('code',config(commit,jobs,TAG,'M1_GIR_LIVE_TASK1_S01',None)+
        '\nGRAD_CLIP = None\nK0 = 512\nPLANNED_SEEDS = list(range(8))\n')
    result['cells'][1]=cell('markdown','# M1 task-1 live GIR: seeds 0 and 1\n'
        'Fork of vna4_M1 on base 6be6543. T4×2: GPU0=s0, GPU1=s1. '
        'Same CIFAR-100 input and github_token Secret; Internet on. '
        'No source ZIP. K0 is the preregistered constant 512; no Gram authority or preflight is needed. '
        'Attach this pilot Output only when resuming it. Baselines and competitors are reused externally; '
        'this notebook trains only icarl_full with --gir-live-task1. Full 20 tasks, no clipping. '
        'GPU-hours and task-1 live/dead swap counts are printed at the end. '
        'Task-1 candidate Gram solves may be expensive; T4 runtime has not been measured. '
        'Deadline pauses preserve the complete task chain for a later session.')
    source=''.join(result['cells'][2]['source'])
    result['cells'][2]=cell('code',source.replace('/kaggle/working/vna4_main','/kaggle/working/'+TAG))
    result['cells'][-2]=cell('code',RUN)
    result['cells'][-1]=cell('code',REPORT)
    for item in result['cells']:
        if item['cell_type']=='code': compile(''.join(item['source']),'<notebook>','exec')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    args=parser.parse_args()
    path=ROOT/'notebooks/vna4_M1_gir_live_task1_s01.ipynb'
    path.write_text(json.dumps(build(args.commit),ensure_ascii=False,indent=1)+'\n')
    print(path)
