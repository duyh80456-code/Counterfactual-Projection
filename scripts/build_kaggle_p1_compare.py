"""Build P1 comparison notebooks and the amended Inc5 full-only seed-2 run."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_kaggle_pilot_common import notebook,cell
from scripts.build_kaggle_vna_parallel import OVERLAY,source_bundle

CASES={'NB1':('resnet32_nobn',20),'NB2':('resnet32_nobn',10),
       'NB3':('resnet32',20),'NB4':('resnet32',10),'M1s2':('resnet32_nobn',5)}

RUNNER=r'''from concurrent.futures import ThreadPoolExecutor
from experiments.io_utils import atomic_json_save
import fcntl
import hashlib
import math
from experiments.p1_compare import estimate_lane
from methods.vna_v3 import FULL_POLICY_VERSION, POLICY_VERSION

def read(path,default=None):
    try:return json.loads(path.read_text())
    except FileNotFoundError:return default

BN_PROTOCOL=None
if BACKBONE=='resnet32':
    candidates=[]
    for path in Path('/kaggle/input').rglob('protocol.json'):
        try:p=read(path,{})
        except (ValueError,OSError):continue
        if p.get('arm')=='icarl_cflat' and p.get('backbone')=='bn' and not p.get('fake_data'):
            rates=tuple(p.get('training',{}).get(k) for k in ('init_lr','lrate'))
            if all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in rates):candidates.append((path,rates))
    if not candidates:raise FileNotFoundError('Attach the existing C-Flat BN protocol.json; LR will not be guessed')
    if len({rates for _,rates in candidates})!=1:raise ValueError('Conflicting C-Flat BN learning rates')
    BN_PROTOCOL=candidates[0][0]
    print('C-FLAT BN LR SOURCE:',BN_PROTOCOL,'LR:',candidates[0][1],
          'SHA256:',hashlib.sha256(BN_PROTOCOL.read_bytes()).hexdigest(),flush=True)

SCHEDULE=ROOT/'_schedules'/f'{BACKBONE}_inc{INCREMENT}.json'
SCHEDULE.parent.mkdir(parents=True,exist_ok=True)
schedule_lock=threading.Lock()
estimate_lock=threading.Lock()
errors=[]

def command(arm,seed,out):
    cmd=[sys.executable,'-u','-m','experiments.p1_compare','--arm',arm,
        '--seed',str(seed),'--backbone',BACKBONE,'--class-order','superclass',
        '--increment',str(INCREMENT),'--max-tasks',str(100//INCREMENT),
        '--warmup-epochs','0','--gir-rule','schur_dead','--solver','gram','--curvature','power10',
        '--nobn-lr-scale','.5','--device','cuda:0','--workers','2',
        '--data-dir',str(DATA_ROOT),'--out',str(out)]
    if BN_PROTOCOL:cmd+=['--bn-lr-protocol',str(BN_PROTOCOL)]
    return cmd

def schedule(gpu):
    with schedule_lock,SCHEDULE.with_suffix('.lock').open('a') as file:
        fcntl.flock(file,fcntl.LOCK_EX)
        if not SCHEDULE.exists():
            cmd=command('icarl_full',0,ROOT/'_preflight')
            cmd+=['--preflight-only','--schedule-output',str(SCHEDULE),'--schedule-source','self']
            if not launch(cmd,gpu,'preflight',RUN_ROOT/'preflight.log'):return False
            archive_outputs()
        value=read(SCHEDULE,{})
        expected=dict(authoritative=True,backbone=BACKBONE,init_cls=INCREMENT,increment=INCREMENT,
            code_commit=REPO_COMMIT,source_digest=SOURCE_DIGEST,solver='gram',curvature='power10',
            policy_version=FULL_POLICY_VERSION,gir_live=False)
        if any(value.get(k)!=v for k,v in expected.items()) or value.get('synthetic'):
            raise ValueError('K0 authority mismatch')
        print(NOTEBOOK,'K0 =',value['K0'],json.dumps(value),flush=True)
        estimate()
        return True

def estimate():
    authority=read(SCHEDULE,{})
    preflight=read(ROOT/'_preflight'/'summary.json',{})
    result=dict(notebook=NOTEBOOK,lanes=[])
    for seed in (0,1):
        base=ROOT/f'icarl__s{seed}'
        baseline=read(base/'summary.json',None)
        if baseline is None:baseline=read(base/'run'/'summary.json',{})
        if baseline.get('status')=='FAILED_DIVERGED':
            result['lanes'].append(dict(seed=seed,status='NO_FULL_RUNTIME_ESTIMATE_BASELINE_DIVERGED',
                baseline_outcome='FAILED_DIVERGED',diverged=baseline.get('diverged')))
            continue
        if baseline.get('status')!='COMPLETED' or not authority.get('calibration') or 'seconds' not in preflight:
            result['lanes'].append(dict(seed=seed,status='PENDING_MEASURED_BASELINE_OR_PREFLIGHT'));continue
        lane=estimate_lane(INCREMENT,baseline['seconds'],authority,preflight['seconds'])
        lane.update(seed=seed,status='ESTIMATED',
            measured_baseline_ggn_seconds=sum(p['seconds'] for p in read(base/'ggn_power10.json',[])))
        result['lanes'].append(lane)
        if lane['total_hours_range'][1]>11:print('WARNING:',NOTEBOOK,'seed',seed,
            'projects above 11 hours:',lane['total_hours_range'],'sessions:',lane['sessions_11h_range'],
            '; keep all tasks and resume next session',flush=True)
    ready=[lane for lane in result['lanes'] if lane['status']=='ESTIMATED']
    if len(ready)==2:
        result['notebook_hours_range']=[max(lane['total_hours_range'][i] for lane in ready) for i in (0,1)]
        result['notebook_sessions_11h_range']=[max(lane['sessions_11h_range'][i] for lane in ready) for i in (0,1)]
    with estimate_lock:atomic_json_save(result,ROOT/'time_estimate.json')
    print('TIME ESTIMATE:',json.dumps(result),flush=True)

def lane(gpu,seed):
    for arm in ('icarl','icarl_full'):
        if pause_due():return
        name=f'{arm}__s{seed}';out=ROOT/name
        existing=read(out/'summary.json',{})
        if existing.get('status') in ('COMPLETED','FAILED_DIVERGED'):
            p=existing.get('identity',{})
            expected=dict(code_commit=REPO_COMMIT,source_digest=SOURCE_DIGEST,arm=arm,seed=seed,
                backbone=BACKBONE,solver='gram',curvature='power10',fake_data=False,
                expected_tasks=100//INCREMENT,policy_version=FULL_POLICY_VERSION if arm=='icarl_full' else POLICY_VERSION,gir_live=False)
            actual={k:p.get(k) for k in expected}
            expected.update(warmup_epochs=0,grad_clip=None,increment=INCREMENT,memory_size=2000)
            actual.update(warmup_epochs=p.get('warmup_epochs',0),grad_clip=p.get('grad_clip'),
                increment=p.get('cfg',{}).get('increment'),memory_size=p.get('cfg',{}).get('memory_size'))
            mismatches={k:dict(saved=actual[k],required=v) for k,v in expected.items() if actual[k]!=v}
            if mismatches:raise ValueError('Terminal result identity mismatch: '+name+' '+json.dumps(mismatches,sort_keys=True))
            if existing['status']=='COMPLETED' and len(existing.get('tasks',[]))!=100//INCREMENT:
                raise ValueError('Completed result is missing tasks: '+name)
            print(name,'existing terminal status:',existing['status'],flush=True);continue
        try:
            if arm=='icarl_full' and not schedule(gpu):return
            cmd=command(arm,seed,out)
            if arm=='icarl_full':cmd+=['--schedule',str(SCHEDULE)]
            if not launch(cmd,gpu,name,RUN_ROOT/(name+'.log')):
                if pause_due():return
                errors.append(dict(job=name,error='nonzero exit'));continue
            result=read(out/'summary.json',{})
            print(name,'STATUS:',result.get('status'),result.get('diverged'),flush=True)
        except Exception as error:errors.append(dict(job=name,error=str(error)))
        archive_outputs();estimate()

try:
    estimate()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(lane,0,0),pool.submit(lane,1,1)]
        for job in jobs:job.result()
finally:
    archive_stop.set();archiver.join(timeout=10);estimate();archive_outputs()
if errors:print('JOB ERRORS',json.dumps(errors),flush=True)
'''


def build(commit,case,source_digest=None,source_file_hashes=None):
    overlay = source_digest is not None
    if not overlay:source_digest=hashlib.sha256(('git:'+commit).encode()).hexdigest()
    backbone,increment=CASES[case]
    result=notebook(commit);cells=result['cells'];tag='p1_compare_'+case.lower()
    if case=='NB1':tag+='_logfix_v3'
    if case=='M1s2':tag='p1_inc5_full_v3_s2'
    config=f'''REPO_COMMIT={commit!r}
SOURCE_DIGEST={source_digest!r}
SOURCE_FILE_HASHES={source_file_hashes or {}!r}
NOTEBOOK={case!r}
BACKBONE={backbone!r}
INCREMENT={increment}
RUN_TAG={tag!r}
import time
SESSION_HOURS=11.5
SESSION_START=time.time()
VNA_DEADLINE=SESSION_START+SESSION_HOURS*3600
DISK_CAP_BYTES=20_000_000_000
'''
    bootstrap=''.join(cells[2]['source']).replace('/kaggle/working/vna4_main','/kaggle/working/p1_compare')
    if overlay:bootstrap+=OVERLAY
    else:
        bootstrap+='''
os.environ['VNA_SOURCE_DIGEST']=SOURCE_DIGEST
(RUN_ROOT/'source_manifest.json').write_text(json.dumps(dict(code_commit=REPO_COMMIT,
    source_digest=SOURCE_DIGEST,verification='git_commit_sha256'),indent=2)+'\\n')
print('Source: verified pinned Git commit; no ZIP overlay',flush=True)
'''
    runtime=''.join(cells[4]['source']).replace("restore_stage_archives(Path('/kaggle/input'), STAGE_B, 'outputs')",
        "ROOT=STAGE_B/RUN_TAG\nROOT.mkdir(parents=True,exist_ok=True)\nrestore_stage_archives(Path('/kaggle/input'), ROOT, RUN_TAG)")
    runtime=runtime.replace('folders=("outputs",)',f'folders=("outputs/{tag}",)')
    runner=RUNNER
    if case=='NB1':
        runtime += "\nBASELINE_ROOT=STAGE_B/'p1_compare_nb1'\nBASELINE_ROOT.mkdir(parents=True,exist_ok=True)\nrestore_stage_archives(Path('/kaggle/input'), BASELINE_ROOT, 'p1_compare_nb1', exclude_parts=('icarl_full__s0','icarl_full__s1','_preflight','_schedules','data'))\n"
        marker='BN_PROTOCOL=None'
        validation="""from experiments.p1_compare import validate_nb1_baseline
for seed in (0,1):
    reference=validate_nb1_baseline(BASELINE_ROOT/f'icarl__s{seed}',seed,DATA_ROOT)
    atomic_json_save(reference,ROOT/f'baseline_reuse__s{seed}.json')
    print('REUSING BASELINE (no training):',json.dumps(reference),flush=True)

"""
        runner=runner.replace(marker,validation+marker)
        runner=runner.replace("for arm in ('icarl','icarl_full'):","for arm in ('icarl_full',):")
        runner=runner.replace("base=ROOT/f'icarl__s{seed}'","base=BASELINE_ROOT/f'icarl__s{seed}'")
    if case=='M1s2':
        runtime += """
BASELINE_ROOT=STAGE_B/'_p1_inc5_baseline'
for stage in ('icarl__nobn__superclass__s2','icarl__s2'):
    restore_stage_archives(Path('/kaggle/input'),BASELINE_ROOT/'icarl__s2',stage)
AUTHORITY_ROOT=STAGE_B/'p1_inc5_full_v3_authority'
AUTHORITY_ROOT.mkdir(parents=True,exist_ok=True)
restore_stage_archives(Path('/kaggle/input'),AUTHORITY_ROOT,'p1_inc5_full_v3_authority')
"""
        runtime=runtime.replace(f'folders=("outputs/{tag}",)',
            f'folders=("outputs/{tag}","outputs/p1_inc5_full_v3_authority")')
        validation="""from experiments.p1_compare import validate_p1_baseline
reference=validate_p1_baseline(BASELINE_ROOT/'icarl__s2',2,DATA_ROOT,5)
atomic_json_save(reference,ROOT/'baseline_reuse__s2.json')
print('REUSING BASELINE (no training):',json.dumps(reference),flush=True)

"""
        runner=runner.replace('BN_PROTOCOL=None',validation+'BN_PROTOCOL=None')
        runner=runner.replace("SCHEDULE=ROOT/'_schedules'/f'{BACKBONE}_inc{INCREMENT}.json'",
            "SCHEDULE=AUTHORITY_ROOT/'resnet32_nobn_inc5_seed0.json'")
        runner=runner.replace("ROOT/'_preflight'","AUTHORITY_ROOT/'_preflight_seed0'")
        runner=runner.replace('policy_version=FULL_POLICY_VERSION,gir_live=False)',
            "policy_version=FULL_POLICY_VERSION,gir_live=False,calibration_seed=0,nobn_lr_scale=.5,class_order_kind='superclass',dataset_fingerprint=reference['dataset_fingerprint'])")
        runner=runner.replace('for seed in (0,1):','for seed in (2,):')
        runner=runner.replace("base=ROOT/f'icarl__s{seed}'","base=BASELINE_ROOT/f'icarl__s{seed}'")
        runner=runner.replace("lane.update(seed=seed,status='ESTIMATED',", """lane['baseline_reused']=True
        lane['total_hours_range']=[v-baseline['seconds']/3600 for v in lane['total_hours_range']]
        lane['sessions_11h_range']=[math.ceil(v/11) for v in lane['total_hours_range']]
        lane['assumptions']+=' Legacy baseline may lack Tracker/GGN timings; their unmeasured full-run overhead is additional, so this is not an upper bound.'
        lane.update(seed=seed,status='ESTIMATED',""")
        runner=runner.replace('if len(ready)==2:','if len(ready)==1:')
        runner=runner.replace("for arm in ('icarl','icarl_full'):","for arm in ('icarl_full',):")
        runner=runner.replace('jobs=[pool.submit(lane,0,0),pool.submit(lane,1,1)]',
            'jobs=[pool.submit(lane,0,2)]')
    description=f'''# {case}: P1 full vs baseline — {backbone}, Inc{increment}
Seeds 0 and 1 on separate T4 lanes: baseline → locked pair-specific preflight → full.
All {100//increment} tasks, final epochs, no warmup or clipping. Attach CIFAR-100, matching source ZIP,
and prior notebook Output for resume. NB3/NB4 also require the existing C-Flat BN protocol.json.
Runtime is unverified on T4: the notebook prints measured projections and warns over 11 hours;
it never shortens the task sequence. Source changes are bundled because this task forbids commits.
'''
    if case=='NB1':
        description=description.replace('baseline → locked pair-specific preflight → full.',
            'validate original baseline → fresh locked pair-specific preflight → new full.')
        description += '\nNB1 log amendment: only icarl_full s0/s1 train in a new output namespace. Attach original NB1 Output: both completed baselines are strictly validated and retained with original provenance. Fresh locked preflight; no old full checkpoint is imported.\n'
    if case=='M1s2':
        description='''# P1 Inc5 — amended icarl_full seed 2 only
20 tasks, noBN/SkipInit ResNet-32, LR 0.05, superclass order, memory 2000,
100/50 epochs, final-epoch model, no warmup/clipping. Policy:
`schur_t32_sharpness_trust_v3`, Gram/power10, verified SOURCE_DIGEST.

Attach CIFAR-100, the matching new source ZIP/dataset, and original P1/M1 Output
containing the seed-2 icarl baseline: completed, or the validated recorded
FAILED_DIVERGED outcome at task 2, epoch 1, step 8 with tensor identified.
GPU0 runs fresh locked seed-0
preflight, then only icarl_full seed 2; GPU1 has no training job. The new shared
authority is `outputs/p1_inc5_full_v3_authority/resnet32_nobn_inc5_seed0.json`.
It is archived for all future Inc5 seeds using this source/policy. Old
gram_power10.json is never imported. Attach this notebook's previous Output
to resume. New full output: `outputs/p1_inc5_full_v3_s2/icarl_full__s2/`.

Tracker covers all tasks; copy-only bitwise-checked full-loss GGN power10 at
r=0,1,2,5,10,20,50,100,200. Intervention logs retain all VNA/GIR decisions.
Legacy baseline measurements are not fabricated. Actual T4 runtime is unknown;
class-dependent estimates warn above 11 hours without shortening any task.
'''
    if not overlay:
        description=description.replace('matching source ZIP','pinned repository source').replace(
            'the matching new source ZIP/dataset','the pinned repository source (no ZIP input)')
        description=description.replace('Source changes are bundled because this task forbids commits.',
            'Source comes directly from the verified pinned Git commit; no ZIP overlay is used.')
        description=description.replace('same source\nbundle','same pinned Git source')
    result['cells']=[cell('markdown',description),cell('code',config+bootstrap+''.join(cells[3]['source'])),
                     cell('code',runtime),cell('code',runner)]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit',required=True);p.add_argument('--source-bundle')
    args=p.parse_args();digest=None;manifest=None
    if args.source_bundle:
        digest=source_bundle(args.source_bundle)
        with zipfile.ZipFile(args.source_bundle) as archive:
            manifest={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
    for case in CASES:
        filename='vna4_M1s2.ipynb' if case=='M1s2' else f'vna5_p1_compare_{case}.ipynb'
        (ROOT/'notebooks'/filename).write_text(json.dumps(build(args.commit,case,digest,manifest),indent=1)+'\n')
    print('Source SHA256:',digest)


if __name__=='__main__':main()
