"""Build one resumable pilot notebook with first-cell job variants."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.build_kaggle_pilot_common import notebook
from scripts.build_kaggle_vna_signal_curve_notebook import cell

B5_ARMS = ('bp','bp_kself','bp_vna','bp_vna_rand','bp_gir','bp_vna_gir',
           'cbp','cbp_gir','cbp_gradmax','cbp_vna_gir')
PM_ARMS = ('bp','bp_kself','bp_vna','bp_vna_rand','cbp','cbp_gir','cbp_gradmax','cbp_vna_gir')
PRESETS = {
 'P-B5': ([], [], [f'b5_{arm}__s{s}' for s in range(3) for arm in B5_ARMS]+
          [f'b5_cbp__rate{r}__s0' for r in ('1e-6','1e-5','1e-4','1e-2')]),
 'P-PM': ([], [], [f'pm_{arm}__s{s}' for s in range(3) for arm in PM_ARMS]),
 'Q1': (['icarl__inc10__s1993'], ['icarl_kself__inc10__s1993'], []),
 'Q2': (['icarl_vna_b0__inc10__s1993'], ['icarl_vna__inc10__s1993'], []),
 'Q3': (['icarl_vna_rand__inc10__s1993'], [], []),
 'P7': (['finetune__inc10__s1993','joint__s1993','finetune__b50inc10__s1993'],
        ['wa__inc10__s1993','wa__b50inc10__s1993'], ['b5_l2']),
 'P8': (['podnet__inc10__s1993','podnet__b50inc10__s1993'],
        ['bic__inc10__s1993','bic__b50inc10__s1993'], ['b5_snp']),
 'P9': (['foster__inc10__s1993','foster__b50inc10__s1993'],
        ['der__inc10__s1993','der__b50inc10__s1993'], ['b5_redo']),
 'P10': (['memo__inc10__s1993','memo__b50inc10__s1993'],
         ['tagfex__inc10__s1993','tagfex__b50inc10__s1993'], []),
}
def source_bundle(path):
    """Capture source bytes including uncommitted changes, excluding outputs/secrets."""
    paths = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(paths)):
            file = ROOT/name
            source = name.startswith(('adapters/','probe/','projection/','training/','methods/','diagnostics/','experiments/','scripts/','tests/','third_party/patches/'))
            document = file.parent == ROOT/'docs/vna'
            if file.is_file() and (source or document or name in ('pyproject.toml','requirements-upstream.txt')) and file.suffix in ('.py','.md','.toml','.txt','.sh','.json','.patch'):
                info = zipfile.ZipInfo(name, (2026,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,file.read_bytes())
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(commit, jobs, run_tag, preset=None, source_digest=None):
    gpu0,gpu1,cpu=jobs
    return f'''JOBS_GPU0 = {gpu0!r}
JOBS_GPU1 = {gpu1!r}
JOBS_CPU = {cpu!r}
RUN_TAG = {run_tag!r}
PRESET = {preset!r}
REPO_COMMIT = {commit!r}
SOURCE_DIGEST = {source_digest!r}
import time
SESSION_HOURS = 11.5
SESSION_START = time.time()
VNA_DEADLINE = SESSION_START + SESSION_HOURS*3600
ALLOW_OVER_BUDGET = True
DISK_CAP_BYTES = 5_000_000_000
'''


OVERLAY = r'''
if SOURCE_DIGEST:
    bundles = [p for p in Path('/kaggle/input').rglob('vna_v3_source.zip')
               if hashlib.sha256(p.read_bytes()).hexdigest() == SOURCE_DIGEST]
    if not bundles: raise FileNotFoundError('Attach vna_v3_source.zip matching SOURCE_DIGEST; this notebook uses uncommitted source')
    with zipfile.ZipFile(bundles[0]) as archive:
        for name in archive.namelist():
            target = (REPO/name).resolve()
            if not target.is_relative_to(REPO.resolve()): raise ValueError('unsafe source bundle member')
        archive.extractall(REPO)
    os.environ['VNA_SOURCE_DIGEST'] = SOURCE_DIGEST
    shutil.copy2(bundles[0], RUN_ROOT/'vna_v3_source.zip')
    (RUN_ROOT/'source_manifest.json').write_text(json.dumps(dict(code_commit=REPO_COMMIT, source_digest=SOURCE_DIGEST)))
'''

RUNNER = r'''import re
from concurrent.futures import ThreadPoolExecutor
from scripts.collect_parallel import validate_result
ROOT = STAGE_B / RUN_TAG
ROOT.mkdir(parents=True, exist_ok=True)
ENV.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

def completed(job):
    directory = ROOT/job
    path = directory/'summary.json'
    if not path.exists(): return False
    summary = json.loads(path.read_text())
    if summary.get('status') != 'COMPLETED': return False
    protocol = json.loads((directory/'protocol.json').read_text())
    tasks = json.loads((directory/'tasks.json').read_text())
    validate_result(job, summary, protocol, tasks, REPO_COMMIT, SOURCE_DIGEST)
    return True

def setting_for(job):
    if job.startswith('b5_'): return 'regression'
    if job.startswith('pm_'): return 'mnist'
    return 'icarl'

def command(job):
    if job in ('b5_l2','b5_snp','b5_redo'):
        return [sys.executable,'-u','-m','experiments.competitors','b5','--arm',job[3:],'--out',str(ROOT/job),'--resume']
    label = job.split('__')[0]
    if label in ('finetune','joint','wa','podnet','bic','foster','der','memo','tagfex'):
        setting = 'b50inc10' if '__b50inc10' in job else 'inc10'
        return [sys.executable,'-u','-m','experiments.competitors','cil','--method',label,
                '--setting',setting,'--seed','1993','--out',str(ROOT/job),
                '--data-dir',str(DATA_ROOT),'--device','cuda:0','--resume']
    seed = re.search(r'__s(\d+)$',job).group(1)
    setting = setting_for(job)
    cmd = [sys.executable,'-u','-m','experiments.pilot_v3','--setting',setting,
           '--arm',label,'--seed',seed,'--out',str(ROOT/job),'--resume',
           '--device','cpu' if setting in ('regression','mnist') else 'cuda:0',
           '--workers','2']
    data_root = globals().get('MNIST_ROOT') if setting == 'mnist' else DATA_ROOT
    if data_root is not None: cmd += ['--data-dir',str(data_root)]
    if label.startswith(('b5_cbp','pm_cbp')):
        rate = re.search(r'__rate([^_]+)',job)
        cmd += ['--replacement-rate',rate.group(1) if rate else '1e-3']
    if label not in ('b5_bp','b5_cbp','pm_bp','pm_cbp','icarl','er','erace'):
        schedule = ROOT/'_schedules'/f"{setting}.json"
        if not schedule.exists():
            print(f'SKIP {job}: family schedule {schedule.name} missing; attach the authority archive and rerun', flush=True)
            return None
        cmd += ['--schedule',str(schedule)]
    return cmd

def prepare_schedule(setting, gpu=None):
    filename = setting+'.json'
    destination = ROOT/'_schedules'/filename
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists(): return
    attached = []
    for path in Path('/kaggle/input').rglob(filename):
        try: value=json.loads(path.read_text())
        except (ValueError,OSError): continue
        if value.get('code_commit') == REPO_COMMIT and value.get('source_digest') == SOURCE_DIGEST and value.get('setting') == setting:
            attached.append((path,value))
    if attached:
        if any(value != attached[0][1] for _,value in attached): raise ValueError('conflicting authoritative schedules')
        shutil.copy2(attached[0][0],destination); return
    authority = {'icarl':'Q2'}
    authority_preset=authority.get(setting,'')
    if setting in authority and PRESET != authority_preset:
        # Baselines need no schedule; scheduled arms are skipped until the archive is attached.
        print(f'NOTE: attach {authority_preset} archive with {setting}.json to run scheduled {PRESET} arms', flush=True)
        return
    arm = dict(regression='b5_bp_vna_gir',mnist='pm_bp_vna',icarl='icarl_vna')[setting]
    seed = '1993' if setting == 'icarl' else '0'
    cmd = [sys.executable,'-u','-m','experiments.pilot_v3','--setting',setting,'--arm',arm,
           '--seed',seed,'--out',str(ROOT/'_preflight'/setting),'--device','cpu' if gpu is None else 'cuda:0',
           '--resume','--preflight-only','--schedule-output',str(destination)]
    data_root = globals().get('MNIST_ROOT') if setting == 'mnist' else DATA_ROOT
    if data_root is not None: cmd += ['--data-dir',str(data_root)]
    if gpu is None:
        from experiments.vna_resume import launch_logged
        return launch_logged(cmd,cwd=REPO,env={**ENV,'CUDA_VISIBLE_DEVICES':''},log_path=RUN_ROOT/f'{RUN_TAG}_{setting}_preflight.log',
            lane='CPU',job=setting+'_preflight',lanes=lane_state,paused=session_paused,deadline=VNA_DEADLINE)
    return launch(cmd,gpu,setting+'_preflight',RUN_ROOT/f'{RUN_TAG}_{setting}_preflight.log')

def run_gpu(gpu,jobs):
    for job in jobs:
        if pause_due(): break
        if completed(job): continue
        cmd = command(job)
        if cmd is None: continue
        if not launch(cmd,gpu,job,RUN_ROOT/f'{RUN_TAG}_{job}.log'): break
        archive_outputs()

def run_cpu(job):
    if pause_due() or completed(job): return
    cmd = command(job)
    if cmd is None: return
    from experiments.vna_resume import launch_logged
    return launch_logged(cmd,cwd=REPO,env={**ENV,'CUDA_VISIBLE_DEVICES':''},
        log_path=RUN_ROOT/f'{RUN_TAG}_{job}.log',lane=job,job=job,lanes=lane_state,
        paused=session_paused,deadline=VNA_DEADLINE)

try:
    v3jobs = [j for j in JOBS_GPU0+JOBS_GPU1+JOBS_CPU if j not in ('b5_l2','b5_snp','b5_redo')
              and j.split('__')[0].startswith(('b5_','pm_','icarl','er'))]
    for setting in sorted({setting_for(j) for j in v3jobs}):
        prepare_schedule(setting,0 if setting == 'icarl' else None)
        if pause_due(): break
    archive_outputs()
    if not pause_due():
        with ThreadPoolExecutor(max_workers=2) as gpu_pool, ThreadPoolExecutor(max_workers=4) as cpu_pool:
            futures=[gpu_pool.submit(run_gpu,0,JOBS_GPU0),gpu_pool.submit(run_gpu,1,JOBS_GPU1)]
            futures += [cpu_pool.submit(run_cpu,job) for job in JOBS_CPU]
            for future in futures: future.result()
finally:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs()
from scripts.collect_parallel import collect, report
report(collect([ROOT.parent],RUN_TAG))
if session_paused.is_set(): print('PAUSED: attach output archive and rerun to resume')
if archive_errors: raise RuntimeError(archive_errors)
'''


def build(commit, preset=None, run_tag='vna_v3_pilot', source_digest=None):
    result=notebook(commit)
    cells=result['cells']
    cells[0]=cell('markdown','# VNA v4 pilot\nAttach the source bundle and previous archives to resume. Q2 provides the iCaRL authority schedule; preflight records calibration and continues training.')
    cells[1]=cell('code',config(commit,PRESETS[preset] if preset else ([],[],[]),run_tag,preset,source_digest))
    cells[2]=cell('code',''.join(cells[2]['source'])+OVERLAY)
    setup=''.join(cells[3]['source'])
    start=setup.index('import torch\n'); end=setup.index('if not (REPO / "experiments/vna_kaggle.py").exists():')
    dataset_setup=setup[start:end]
    setup=setup[:start]+"DATA_ROOT = Path('/kaggle/input')\nMNIST_ROOT = next((p.parent for p in Path('/kaggle/input').rglob('mnist_') if p.is_file()),None)\nif JOBS_GPU0 or JOBS_GPU1:\n"+'\n'.join('    '+line for line in dataset_setup.splitlines())+'\n'+setup[end:]
    setup=setup.replace('print("Official TEST verified:", DATA_ROOT / "cifar-100-python/test")','')
    # The local affected tests run before publication; avoid the full suite on Kaggle.
    import re
    setup=re.sub(r'subprocess.run\(\[sys.executable, "-m", "pytest".*?check=True\)\n','',setup,flags=re.S)
    cells[3]=cell('code',setup)
    runtime=''.join(cells[4]['source']).replace('from experiments.stage_b5 import restore_archives','from experiments.vna_kaggle import restore_stage_archives').replace('RUN_ROOT / "outputs" / "stageB"','RUN_ROOT / "outputs"').replace('restore_archives(Path("/kaggle/input"), STAGE_B)',"restore_stage_archives(Path('/kaggle/input'), STAGE_B, 'outputs')").replace('folders=("outputs/stageB",)','folders=("outputs",)')
    cells[4]=cell('code',runtime)
    cells[-1]=cell('code',RUNNER)
    result['cells']=[cells[1],cells[0],*cells[2:]]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--run-tag',default='vna_v3_pilot')
    parser.add_argument('--output',default=str(ROOT/'notebooks/kaggle_vna_parallel.ipynb'))
    parser.add_argument('--variants',action='store_true')
    parser.add_argument('--source-bundle',help='Write source overlay ZIP here and pin its SHA256 in every notebook')
    args=parser.parse_args(); output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    digest=source_bundle(args.source_bundle) if args.source_bundle else None
    output.write_text(json.dumps(build(args.commit,None,args.run_tag,digest),indent=1)+'\n')
    if args.variants:
        for name in PRESETS:
            path=output.with_name(output.stem+'_'+name+output.suffix)
            path.write_text(json.dumps(build(args.commit,name,args.run_tag,digest),indent=1)+'\n')


if __name__=='__main__': main()
