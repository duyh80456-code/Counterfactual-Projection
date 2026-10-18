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
from scripts.build_kaggle_pilot_common import cell

TUNED = ('icarl_cbp','icarl_redo','icarl_snp','icarl_upgd')

def cil_jobs(arms, backbones=('nobn',), order='superclass'):
    return [f'{arm}__{backbone}__{order}__s{seed}' for backbone in backbones
            for seed in range(3) for arm in arms]


def lanes(jobs): return (jobs[::2], jobs[1::2], [])


PRESETS = {
    'M1': lanes(cil_jobs(('icarl','icarl_full'))),
    'M2': lanes(cil_jobs(('icarl','icarl_full'), ('bn',))),
    'M3': lanes(cil_jobs(('icarl','icarl_full'), order='random')),
    'C1': lanes([method+'__sweep' for method in TUNED]),
    'C2': lanes(cil_jobs(('podnet','wa','gem','icarl_cflat'), ('nobn','bn'))),
    'C3': lanes(cil_jobs(TUNED, ('nobn',))),
    'C4': lanes(cil_jobs(TUNED, ('bn',))),
    'C5': (['icarl_cflat__bn__random__s1993'], [], []),
    'B5w': ([], [], [f'b5_{arm}__width{width}__s{seed}' for width in (5,20,100)
                    for arm in ('bp','bp_vna','bp_gir','bp_vna_gir') for seed in range(3)]),
}
def source_bundle(path):
    """Capture source bytes including uncommitted changes, excluding outputs/secrets."""
    paths = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(paths)):
            file = ROOT/name
            source = name.startswith(('adapters/','probe/','projection/','methods/','diagnostics/','experiments/','scripts/','tests/','third_party/patches/'))
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

def attached_file(filename, destination, predicate):
    if destination.exists(): return destination
    candidates = []
    for path in Path('/kaggle/input').rglob(filename):
        try: value = json.loads(path.read_text())
        except (ValueError, OSError): continue
        if predicate(value): candidates.append((path,value))
    if not candidates: return None
    if any(value != candidates[0][1] for _,value in candidates): raise ValueError(f'conflicting {filename}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0][0], destination)
    return destination

def completed(job):
    directory = ROOT/job
    if job.endswith('__sweep'):
        path = directory/'best_config.json'
        if not path.exists(): return False
        value = json.loads(path.read_text())
        if value.get('code_commit') != REPO_COMMIT or value.get('source_digest') != SOURCE_DIGEST or value.get('synthetic'):
            raise ValueError('best_config identity mismatch')
        return True
    path = directory/'summary.json'
    if not path.exists(): return False
    summary = json.loads(path.read_text())
    if summary.get('status') != 'COMPLETED': return False
    protocol = json.loads((directory/'protocol.json').read_text())
    tasks = json.loads((directory/'tasks.json').read_text())
    validate_result(job, summary, protocol, tasks, REPO_COMMIT, SOURCE_DIGEST)
    return True

def schedule_name(backbone): return 'icarl_BN.json' if backbone == 'bn' else 'icarl_noBN.json'

def prepare_schedule(backbone):
    filename = schedule_name(backbone)
    destination = ROOT/'_schedules'/filename
    attached_file(filename, destination, lambda v: v.get('code_commit') == REPO_COMMIT
                  and v.get('source_digest') == SOURCE_DIGEST and v.get('authoritative')
                  and v.get('backbone') == ('resnet32' if backbone == 'bn' else 'resnet32_nobn'))
    if destination.exists(): return
    cmd = [sys.executable,'-u','-m','experiments.main_cil','--arm','icarl_full',
           '--backbone','resnet32' if backbone == 'bn' else 'resnet32_nobn',
           '--class-order','superclass','--seed','0','--out',str(ROOT/'_preflight'/backbone),
           '--device','cuda:0','--data-dir',str(DATA_ROOT),'--preflight-only',
           '--schedule-output',str(destination),'--schedule-source','self' if PRESET == 'M3' else 'authority']
    launch(cmd,0,backbone+'_preflight',RUN_ROOT/f'{RUN_TAG}_{backbone}_preflight.log')
    archive_outputs()
    if not destination.exists() and not pause_due(): raise RuntimeError('preflight did not produce K0')

def command(job):
    if job.startswith('b5_'):
        arm,width,seed = re.fullmatch(r'b5_(.+)__width(\d+)__s(\d+)',job).groups()
        cmd = [sys.executable,'-u','-m','experiments.pilot_v3','--setting','regression',
               '--arm','b5_'+arm,'--width',width,'--seed',seed,'--device','cpu',
               '--out',str(ROOT/job),'--resume']
        if arm != 'bp': cmd += ['--schedule',str(ROOT/'_schedules'/f'b5_width{width}.json')]
        return cmd
    if job.endswith('__sweep'):
        return [sys.executable,'-u','-m','experiments.v4_competitors','--method',job.split('__')[0],
                '--mode','sweep','--device','cuda:0','--data-dir',str(DATA_ROOT),
                '--out',str(ROOT/job),'--resume']
    arm,backbone,order,seed = job.split('__'); seed = seed[1:]
    if arm in ('icarl','icarl_full'):
        cmd = [sys.executable,'-u','-m','experiments.main_cil','--arm',arm,
               '--backbone','resnet32' if backbone == 'bn' else 'resnet32_nobn',
               '--class-order',order,'--seed',seed,'--out',str(ROOT/job),
               '--device','cuda:0','--data-dir',str(DATA_ROOT)]
        if arm == 'icarl_full': cmd += ['--schedule',str(ROOT/'_schedules'/schedule_name(backbone))]
        return cmd
    cmd = [sys.executable,'-u','-m','experiments.v4_competitors','--method',arm,
           '--backbone',backbone,'--order',order,'--seed',seed,'--out',str(ROOT/job),
           '--device','cuda:0','--data-dir',str(DATA_ROOT),'--resume']
    if PRESET in ('C3','C4'):
        destination = ROOT/'_configs'/arm/'best_config.json'
        path = attached_file('best_config.json', destination, lambda v: v.get('method') == arm
                             and v.get('code_commit') == REPO_COMMIT and v.get('source_digest') == SOURCE_DIGEST
                             and v.get('synthetic') is False)
        if path is None: raise FileNotFoundError(f'Attach C1 best_config.json for {arm}')
        cmd += ['--best-config',str(path)]
    if PRESET == 'C5': cmd += ['--reproduction']
    return cmd

def run_gpu(gpu,jobs):
    for job in jobs:
        if pause_due(): break
        if completed(job): continue
        if not launch(command(job),gpu,job,RUN_ROOT/f'{RUN_TAG}_{job}.log'): break
        archive_outputs()

def run_cpu(job):
    if pause_due() or completed(job): return
    from experiments.vna_resume import launch_logged
    result = launch_logged(command(job),cwd=REPO,env={**ENV,'CUDA_VISIBLE_DEVICES':''},
        log_path=RUN_ROOT/f'{RUN_TAG}_{job}.log',lane=job,job=job,lanes=lane_state,
        paused=session_paused,deadline=VNA_DEADLINE)
    archive_outputs()
    return result

try:
    if PRESET in ('M1','M2','M3'): prepare_schedule('bn' if PRESET == 'M2' else 'nobn')
    if PRESET == 'B5w':
        for width in (5,20,100):
            destination = ROOT/'_schedules'/f'b5_width{width}.json'
            attached_file(destination.name,destination,lambda v: v.get('code_commit') == REPO_COMMIT
                          and v.get('source_digest') == SOURCE_DIGEST and v.get('authoritative') and v.get('width') == width)
            if not destination.exists():
                from experiments.vna_resume import launch_logged
                cmd = [sys.executable,'-u','-m','experiments.pilot_v3','--setting','regression',
                       '--arm','b5_bp_vna_gir','--width',str(width),'--seed','0','--device','cpu',
                       '--out',str(ROOT/'_preflight'/f'width{width}'),'--resume','--preflight-only',
                       '--schedule-output',str(destination)]
                launch_logged(cmd,cwd=REPO,env={**ENV,'CUDA_VISIBLE_DEVICES':''},
                    log_path=RUN_ROOT/f'{RUN_TAG}_width{width}_preflight.log',lane='CPU',job='preflight',
                    lanes=lane_state,paused=session_paused,deadline=VNA_DEADLINE)
            if destination.exists():
                schedule = json.loads(destination.read_text())
                schedule['width'] = width
                destination.write_text(json.dumps(schedule,indent=2))
            if pause_due(): break
    archive_outputs()
    if not pause_due():
        with ThreadPoolExecutor(max_workers=2) as gpu_pool, ThreadPoolExecutor(max_workers=4) as cpu_pool:
            futures=[gpu_pool.submit(run_gpu,0,JOBS_GPU0),gpu_pool.submit(run_gpu,1,JOBS_GPU1)]
            futures += [cpu_pool.submit(run_cpu,job) for job in JOBS_CPU]
            for future in futures: future.result()
finally:
    archive_stop.set(); archiver.join(timeout=10); archive_outputs()
'''

REPORT = r'''from scripts.collect_parallel import collect, report
from experiments.main_cil_results import stability_plasticity
from collections import Counter
records = collect([ROOT.parent],RUN_TAG)
if PRESET == 'C1':
    from experiments.v4_competitors import grid
    for method in ('icarl_cbp','icarl_redo','icarl_snp','icarl_upgd'):
        selection = ROOT/(method+'__sweep')/'best_config.json'
        if not selection.exists(): continue
        best = json.loads(selection.read_text())
        directory = selection.parent/'scan'/str(grid(method).index(best['config']))
        record = {key:json.loads((directory/(key+'.json')).read_text())
                  for key in ('summary','protocol','tasks','diagnostics')}
        job = method+'__nobn__superclass__s0'
        validate_result(job,record['summary'],record['protocol'],record['tasks'],REPO_COMMIT,SOURCE_DIGEST)
        records[job] = dict(record,job=job)
if PRESET != 'B5w': report(records)
flags = []
for record in records.values():
    events = record.get('interventions',record.get('vna',[]))
    counts = Counter()
    for event in events:
        for flag in ('gir_fallback','warm_reset','rejected'):
            value = event.get(flag,False)
            if flag == 'warm_reset':
                value = value or any(r.get('warm_reset',False) for r in event.get('cg',[]) if isinstance(r,dict))
            counts[flag] += str(value).lower() in ('true','1')
        counts['skip_reason='+str(event.get('skip_reason') or 'none')] += 1
    flags.append(dict(job=record['job'],events=len(events),
                      rates={key:value/max(1,len(events)) for key,value in counts.items()}))
print(json.dumps(flags,indent=2))
(ROOT/'flag_rates.json').write_text(json.dumps(flags,indent=2))
figure = ROOT/'stability_plasticity.svg'
if PRESET == 'B5w':
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import statistics
    groups = defaultdict(list)
    for record in records.values():
        width = re.search(r'__width(\d+)__',record['job']).group(1)
        groups[(record['protocol']['arm'],width)].append(record['summary'])
    from scripts.collect_parallel import markdown, mean_sd
    rows = []
    for (arm,width),summaries in sorted(groups.items()):
        controls = {v['seed']:v for v in groups.get(('b5_bp',width),[])}
        deltas = [v['mean_last50']-controls[v['seed']]['mean_last50'] for v in summaries if v['seed'] in controls]
        rows.append(dict(width=width,arm=arm,seeds=len(summaries),
            online_error=mean_sd([v['mean_last50'] for v in summaries]),
            paired_delta_bp=mean_sd(deltas),seeds_won=sum(v<0 for v in deltas),
            cpu_hours=mean_sd([v['seconds']/3600 for v in summaries])))
    markdown('B5 widths',rows,['width','arm','seeds','online_error','paired_delta_bp','seeds_won','cpu_hours'])
    fig,ax = plt.subplots()
    for (arm,width),summaries in sorted(groups.items()):
        xs = [v['mean_last50'] for v in summaries]
        ys = [-v['anytime_loss_auc'] for v in summaries]
        std = lambda vs: statistics.stdev(vs) if len(vs)>1 else 0.
        ax.errorbar(statistics.mean(xs),statistics.mean(ys),xerr=std(xs),yerr=std(ys),fmt='o',label=arm+'/'+width)
    ax.set(xlabel='Mean last-50 online error (regression stability proxy)',
           ylabel='Negative task loss AUC (regression plasticity proxy)')
    if groups: ax.legend(fontsize=6)
    fig.savefig(figure,bbox_inches='tight'); plt.close(fig)
else:
    stability_plasticity(records.values(),figure)
from IPython.display import SVG, display
display(SVG(filename=str(figure)))
for path in ROOT.rglob('best_config.json'): print(path, path.read_text())
for path in ROOT.rglob('reproduction_gate.json'): print(path, path.read_text())
archive_outputs()
if session_paused.is_set(): print('PAUSED: attach output archive and rerun to resume')
if archive_errors: raise RuntimeError(archive_errors)
'''



def build(commit, preset=None, run_tag='vna_v4_main', source_digest=None):
    result=notebook('0'*40 if commit == 'PLACEHOLDER' else commit)
    if commit == 'PLACEHOLDER':
        for item in result['cells']:
            item['source'] = [line.replace('0'*40,commit) for line in item['source']]
    cells=result['cells']
    cells[0]=cell('markdown','# VNA v4 main run\nAttach previous output archives to resume. M1/M2 own backbone schedules; C3/C4 require C1 selections.')
    cells[1]=cell('code',config(commit,PRESETS[preset] if preset else ([],[],[]),run_tag,preset,source_digest))
    cells[2]=cell('code',''.join(cells[2]['source'])+OVERLAY)
    setup=''.join(cells[3]['source'])
    start=setup.index('import torch\n'); end=setup.index('if not (REPO / "experiments/vna_kaggle.py").exists():')
    dataset_setup=setup[start:end]
    setup=setup[:start]+"DATA_ROOT = Path('/kaggle/input')\nMNIST_ROOT = next((p.parent for p in Path('/kaggle/input').rglob('mnist_') if p.is_file()),None)\nif JOBS_GPU0 or JOBS_GPU1:\n"+'\n'.join('    '+line for line in dataset_setup.splitlines())+'\n'+setup[end:]
    setup += '\nsubprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"quadprog\"], check=True)\n'
    setup=setup.replace('print("Official TEST verified:", DATA_ROOT / "cifar-100-python/test")','')
    # The local affected tests run before publication; avoid the full suite on Kaggle.
    import re
    setup=re.sub(r'subprocess.run\(\[sys.executable, "-m", "pytest".*?check=True\)\n','',setup,flags=re.S)
    cells[3]=cell('code',setup)
    runtime=''.join(cells[4]['source'])
    cells[4]=cell('code',runtime)
    cells[-1]=cell('code',RUNNER)
    cells.append(cell('code',REPORT))
    result['cells']=[cells[1],cells[0],*cells[2:]]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit',required=True)
    parser.add_argument('--run-tag',default='vna_v4_main')
    parser.add_argument('--output',default=str(ROOT/'notebooks/vna4_M1.ipynb'))
    parser.add_argument('--variants',action='store_true')
    parser.add_argument('--source-bundle',help='Write source overlay ZIP here and pin its SHA256 in every notebook')
    args=parser.parse_args(); output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    digest=source_bundle(args.source_bundle) if args.source_bundle else None
    output.write_text(json.dumps(build(args.commit,'M1',args.run_tag,digest),indent=1)+'\n')
    if args.variants:
        for name in PRESETS:
            stem = output.stem[:-3] if output.stem.endswith('_M1') else output.stem
            path=output.with_name(stem+'_'+name+output.suffix)
            path.write_text(json.dumps(build(args.commit,name,args.run_tag,digest),indent=1)+'\n')


if __name__=='__main__': main()
