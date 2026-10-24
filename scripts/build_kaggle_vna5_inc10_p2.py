"""Build the separate seed-0 Inc10/P2 CertSwap pilot; leave v4 notebooks intact."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_kaggle_pilot_common import notebook,cell
from scripts.build_kaggle_vna_parallel import OVERLAY,source_bundle

ARMS=('icarl_full','icarl','icarl_cbp','icarl_redo')

GPU_GATE=r'''
print('CERTSWAP GPU GATE: real CIFAR / native ResNet32 / exact jumps and reproducibility',flush=True)
gate_xml=RUN_ROOT/'certswap_gpu_gate.xml'
gate_log=RUN_ROOT/'certswap_gpu_gate.log'
gate_env={**test_env,'CUDA_VISIBLE_DEVICES':'0','VNA_CERTSWAP_CIFAR_ROOT':str(DATA_ROOT)}
gate=subprocess.run([sys.executable,'-m','pytest','tests/test_certswap_cuda.py','-ra',
    '--junitxml='+str(gate_xml)],cwd=REPO,env=gate_env,capture_output=True,text=True)
gate_log.write_text(gate.stdout+gate.stderr)
print(gate.stdout+gate.stderr,flush=True)
if gate.returncode:raise RuntimeError('CERTSWAP GPU GATE FAIL: preflight and training are blocked; see '+str(gate_log))
import xml.etree.ElementTree as ET
cases=list(ET.parse(gate_xml).getroot().iter('testcase'))
if len(cases)!=3 or any(c.find(tag) is not None for c in cases for tag in ('skipped','failure','error')):
    raise RuntimeError('CERTSWAP GPU GATE FAIL: all three CUDA tests must PASS, with no skips')
GPU_GATE_PASSED=True
print('CERTSWAP GPU GATE PASS: all 3 checks passed; preflight/training may start.',flush=True)
'''

RUNNER=r'''import queue
from concurrent.futures import ThreadPoolExecutor
from experiments.io_utils import atomic_json_save
if not globals().get('GPU_GATE_PASSED',False):
    raise RuntimeError('Run the first code cell and PASS the CertSwap GPU gate before launching jobs')
SCHEDULE=ROOT/'_schedules'/'inc10_p2_certswap.json'
errors=[]
report_stop=threading.Event()
report_lock=threading.Lock()

def read(path,default):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return default

def identity(value,arm):
    value=value.get('identity',value)
    cfg=value.get('cfg',{})
    if (value.get('code_commit')!=REPO_COMMIT or value.get('source_digest')!=SOURCE_DIGEST
            or value.get('arm')!=arm or value.get('seed')!=0
            or cfg.get('init_cls')!=10 or cfg.get('increment')!=10
            or value.get('warmup_epochs')!=1 or value.get('grad_clip') is not None
            or value.get('class_order_kind',value.get('order'))!='superclass'
            or value.get('backbone') not in ('nobn','resnet32_nobn')
            or not value.get('class_order_sha256')
            or value.get('fake_data') or value.get('preflight_only')):
        raise ValueError('P2 completed identity mismatch: '+arm)
    if arm=='icarl_full' and (value.get('policy_version'),value.get('solver'),value.get('curvature'))!=('certswap_t32_v1','gram','power10'):
        raise ValueError('P2 CertSwap solver/policy mismatch')

def job_name(arm):return arm+'__nobn__superclass__inc10__p2__s0'

def command(arm,out):
    common=['--increment','10','--warmup-epochs','1','--seed','0',
            '--nobn-lr-scale','.5','--device','cuda:0','--data-dir',str(DATA_ROOT),'--out',str(out)]
    if arm in ('icarl','icarl_full'):
        cmd=[sys.executable,'-u','-m','experiments.main_cil','--arm',arm,
             '--backbone','resnet32_nobn','--class-order','superclass','--solver','gram',
             '--curvature','power10','--max-tasks','10',*common]
        if arm=='icarl_full':cmd+=['--gir-rule','certswap','--certswap-topk','8','--certswap-lm-budget','20','--schedule',str(SCHEDULE)]
        return cmd
    return [sys.executable,'-u','-m','experiments.v4_competitors','--method',arm,
            '--backbone','nobn','--order','superclass','--resume',*common]

def prepare_schedule():
    if SCHEDULE.exists():
        value=read(SCHEDULE,{})
        expected=dict(authoritative=True,code_commit=REPO_COMMIT,source_digest=SOURCE_DIGEST,
            init_cls=10,increment=10,warmup_epochs=1,gir_rule='certswap',
            policy_version='certswap_t32_v1',solver='gram',curvature='power10')
        if any(value.get(k)!=v for k,v in expected.items()):raise ValueError('Inc10/P2 authority schedule mismatch')
        return True
    cmd=command('icarl_full',ROOT/'_preflight')
    position=cmd.index('--schedule');del cmd[position:position+2]
    cmd+=['--preflight-only','--schedule-output',str(SCHEDULE),'--schedule-source','self']
    if not launch(cmd,0,'inc10_p2_preflight',RUN_ROOT/'inc10_p2_preflight.log'):return False
    archive_outputs()
    if not SCHEDULE.exists():raise RuntimeError('Inc10/P2 preflight did not produce its own K0')
    value=read(SCHEDULE,{})
    print('INC10/P2 PREFLIGHT:',json.dumps(value,indent=2),flush=True)
    prepare_schedule()
    return True

def report():
    checkpoint2=[];final=[];activity=[];timings=[]
    for arm in ARMS:
        directory=ROOT/job_name(arm)
        summary=read(directory/'summary.json',{})
        tasks=read(directory/'tasks.json',summary.get('tasks',[]))
        events=read(directory/'interventions.json',[])
        diagnostics=read(directory/'diagnostics.json',[])
        if isinstance(diagnostics,dict):diagnostics=list(diagnostics.values())
        for diagnostic in diagnostics:
            activity.append(dict(arm=arm,task=diagnostic.get('task'),sites=diagnostic.get('sites',{})))
        for event in events:
            block=event.get('certswap',{})
            timings.append(dict(arm=arm,task=event.get('task'),iteration=event.get('iteration'),
                vna_seconds=event.get('vna_seconds'),certswap_seconds=event.get('certswap_seconds',block.get('seconds')),
                accepted=len(block.get('accepted',[])),rejections=block.get('rejections',{})))
        task2=next((row for row in tasks if row.get('task')==2),None)
        diverged=summary.get('diverged')
        task2_events=[r for r in timings if r['arm']==arm and r['task']==2]
        reasons={}
        for event in task2_events:
            for reason,count in event['rejections'].items():reasons[reason]=reasons.get(reason,0)+count
        if task2 or diverged:
            checkpoint2.append(dict(arm=arm,status=summary.get('status','RUNNING'),diverged=diverged,
                learning_accuracy=None if task2 is None else task2.get('cnn_learning_accuracy'),
                certswap_accepted=sum(r['accepted'] for r in task2_events),certswap_rejections=reasons))
        if len(tasks)==10 or summary.get('status') in ('DIVERGED','FAILED_DIVERGED'):
            last=tasks[-1] if tasks else {}
            final.append(dict(arm=arm,status=summary.get('status','RUNNING'),tasks=len(tasks),
                cnn_average=summary.get('cnn_average_accuracy',last.get('cnn_average_accuracy')),
                nme_average=summary.get('nme_average_accuracy',last.get('nme_average_accuracy')),
                cnn_last=summary.get('cnn_last_accuracy',last.get('cnn_top1')),
                nme_last=summary.get('nme_last_accuracy',last.get('nme_top1')),
                learning_accuracy=summary.get('cnn_learning_accuracy',last.get('cnn_learning_accuracy')),
                nme_learning_accuracy=summary.get('nme_learning_accuracy',last.get('nme_learning_accuracy')),
                forgetting=summary.get('cnn_forgetting',last.get('cnn_forgetting')),
                nme_forgetting=summary.get('nme_forgetting',last.get('nme_forgetting'))))
    swaps_by_task=[dict(task=task,accepted=sum(row['accepted'] for row in timings if row['task']==task)) for task in range(1,11)]
    result=dict(task2=checkpoint2,task10=final,activity_by_task=activity,applications=timings,swaps_by_task=swaps_by_task,errors=list(errors))
    with report_lock:atomic_json_save(result,ROOT/'P2_PILOT_REPORT.json')
    return result

def print_report(result):
    print('P2 TASK2:',json.dumps(result['task2'],indent=2),flush=True)
    print('P2 TASK10:',json.dumps(result['task10'],indent=2),flush=True)
    if result['task10']:
        print('P2 CERTSWAP BY TASK:',json.dumps(result['swaps_by_task'],indent=2),flush=True)
        print('P2 CERTSWAP BY APPLICATION:',json.dumps(result['applications'],indent=2),flush=True)
        print('P2 DEAD/DORMANT BY TASK:',json.dumps(result['activity_by_task'],indent=2),flush=True)

def watch():
    previous=None
    while not report_stop.wait(30):
        value=report();key=json.dumps((value['task2'],value['task10']),sort_keys=True)
        if key!=previous and (value['task2'] or value['task10']):print_report(value);previous=key

pending=queue.Queue()
for arm in ('icarl_cbp','icarl_redo'):pending.put(arm)

def lane(gpu,first):
    initial=iter(first)
    while not pause_due():
        try:arm=next(initial)
        except StopIteration:
            try:arm=pending.get_nowait()
            except queue.Empty:return
        directory=ROOT/job_name(arm)
        summary=read(directory/'summary.json',{})
        if summary:
            identity(summary,arm)
            if summary.get('status')=='COMPLETED':
                if len(summary.get('tasks',[]))!=10:raise ValueError('Completed P2 run must contain all ten tasks')
                continue
            if summary.get('status') in ('FAILED_DIVERGED','DIVERGED'):
                print('FAILED_DIVERGED',summary.get('diverged'),arm,flush=True);continue
        try:
            if arm=='icarl_full' and not prepare_schedule():return
            if not launch(command(arm,directory),gpu,job_name(arm),RUN_ROOT/(job_name(arm)+'.log')):return
            outcome=read(directory/'summary.json',{})
            if outcome.get('status') in ('FAILED_DIVERGED','DIVERGED'):
                print('FAILED_DIVERGED',outcome.get('diverged'),arm,flush=True)
        except Exception as error:
            outcome=read(directory/'summary.json',{})
            if outcome.get('status') in ('FAILED_DIVERGED','DIVERGED'):
                print('FAILED_DIVERGED',outcome.get('diverged'),arm,flush=True)
            else:
                errors.append(dict(arm=arm,error=f'{type(error).__name__}: {error}'))
                print('P2 JOB ERROR:',errors[-1],flush=True)
        archive_outputs()
        print_report(report())

reporter=threading.Thread(target=watch,daemon=True);reporter.start()
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(lane,0,['icarl_full']),pool.submit(lane,1,['icarl'])]
        for job in jobs:job.result()
finally:
    report_stop.set();reporter.join(timeout=10)
    print_report(report())
    archive_stop.set();archiver.join(timeout=10);archive_outputs()
if session_paused.is_set():print('PAUSED: attach this pilot output and rerun to continue P2 only.')
if errors:raise RuntimeError('P2 jobs with errors: '+json.dumps(errors))
if archive_errors:raise RuntimeError(archive_errors)
'''


def build(commit,source_digest=None):
    result=notebook(commit)
    cells=result['cells']
    cells[0]=cell('code',f'''REPO_COMMIT = {commit!r}
SOURCE_DIGEST = {source_digest!r}
GPU_GATE_PASSED = False
RUN_TAG = 'vna_v5_inc10_p2'
ARMS = {ARMS!r}
JOBS_GPU0 = ['icarl_full__nobn__superclass__inc10__p2__s0']
JOBS_GPU1 = ['icarl__nobn__superclass__inc10__p2__s0','icarl_cbp__nobn__superclass__inc10__p2__s0','icarl_redo__nobn__superclass__inc10__p2__s0']
import time
SESSION_HOURS = 11.5
SESSION_START = time.time()
VNA_DEADLINE = SESSION_START + SESSION_HOURS*3600
DISK_CAP_BYTES = 20_000_000_000
''')
    cells[1]=cell('markdown','''# VNA v5 — B0 Inc10 / P2 / seed-0 CertSwap pilot
Enable Internet, T4×2 and the `github_token` secret. Attach CIFAR-100 with `cifar-100-python/{train,test,meta}` and this pilot's own prior output archive to resume. When SOURCE_DIGEST is set, attach the exact source ZIP matching that digest; the base Git pin alone does not contain the local changes.

All four arms use ResNet32 noBN/SkipInit, superclass order, memory 2000, native 100/50 epochs, lr scale 0.5 and one epoch of linear warmup at EVERY task; clipping stays off. GPU0 runs a fresh Inc10/P2 CertSwap authority preflight and then full seed 0 (Gram/power10). GPU1 starts baseline seed 0; available GPUs then draw CBP/ReDo from the shared queue, using their registered original defaults. No seeds 1–2 or other competitors are launched.

The first code cell loads the pinned source and runs the mandatory CUDA gate on real CIFAR images: exact zero dead-slot CE/KD jumps, bitwise suffix/full equality, and identical j/rho for repeated candidates. FAIL or SKIP blocks preflight and every training job. GPU gate XML and logs remain in Output.

Task-2 and task-10 reports appear as soon as each arm reaches them and do not stop training. The report retains CertSwap acceptance/rejection counts, separate VNA/CertSwap timing and per-site dead/dormant fractions on the native fixed 256-sample probe. Output: `outputs/vna_v5_inc10_p2/`. P1, clipping studies and all old notebook outputs are kept separate.
''')
    bootstrap=''.join(cells[2]['source']).replace('/kaggle/working/vna4_main','/kaggle/working/vna5_inc10_p2_pilot')+OVERLAY
    cells[2]=cell('code',bootstrap)
    runtime=''.join(cells[4]['source'])
    runtime=runtime.replace("restore_stage_archives(Path('/kaggle/input'), STAGE_B, 'outputs')", "ROOT=STAGE_B/RUN_TAG\nROOT.mkdir(parents=True,exist_ok=True)\nrestore_stage_archives(Path('/kaggle/input'), ROOT, RUN_TAG)")
    runtime=runtime.replace('folders=("outputs",)', 'folders=("outputs/vna_v5_inc10_p2",)')
    cells[4]=cell('code',runtime)
    cells[5]=cell('code',RUNNER)
    result['cells']=[cells[1],cell('code',''.join(cells[0]['source'])+'\n'+bootstrap+'\n'+
        ''.join(cells[3]['source'])+'\n'+GPU_GATE),cells[4],cells[5]]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--output',default=str(ROOT/'notebooks/vna5_inc10_p2_pilot.ipynb'))
    parser.add_argument('--source-bundle')
    args=parser.parse_args()
    digest=source_bundle(args.source_bundle) if args.source_bundle else None
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(build(args.commit,digest),indent=1)+'\n')


if __name__=='__main__':main()
