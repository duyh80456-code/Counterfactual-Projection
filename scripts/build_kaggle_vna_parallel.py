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

def cil_jobs(arms, backbones=('nobn',), order='superclass', seeds=range(3)):
    return [f'{arm}__{backbone}__{order}__s{seed}' for backbone in backbones
            for seed in seeds for arm in arms]


# Rough relative GPU cost per job (icarl = 1): icarl_full may spend up to 1.0x extra on the
# algorithm (§2.5 budget), C-Flat reports ~2.9x, sweeps scale with their grid size.
COST = {'icarl_full': 2., 'icarl_cflat': 2.9, 'gem': 1.5, 'podnet': 1.3,
        'icarl_cbp__sweep': 16., 'icarl_redo__sweep': 24., 'icarl_snp__sweep': 16., 'icarl_upgd__sweep': 36.,
        'podnet__sweep': 5.2, 'wa__sweep': 4., 'gem__sweep': 6., 'icarl_cflat__sweep': 11.6}


def lanes(jobs):
    """Balance the two T4 lanes: longest job first onto the less loaded GPU."""
    load, gpus = [0., 0.], ([], [])
    for job in sorted(jobs, key=lambda j: -COST.get(j if j.endswith('__sweep') else j.split('__')[0], 1.)):
        gpu = load.index(min(load))
        gpus[gpu].append(job); load[gpu] += COST.get(job if job.endswith('__sweep') else job.split('__')[0], 1.)
    return (gpus[0], gpus[1], [])


PRESETS = {
    'M1': lanes(cil_jobs(('icarl','icarl_full'),seeds=(1,2))),
    'M2': lanes(cil_jobs(('icarl','icarl_full'), ('bn',))),
    'M3': lanes(cil_jobs(('icarl','icarl_full'), order='random')),
    # Seed 0 only: icarl_full (GPU0) next to icarl (GPU1), same setup and K0 authority as M1.
    'M1s0': (['icarl_full__nobn__superclass__s0'], ['icarl__nobn__superclass__s0'], []),
    'C1': lanes([method+'__sweep' for method in TUNED]),
    'C2': tuple(a+b for a,b in zip(
        lanes([m+'__sweep' for m in ('podnet','wa','gem','icarl_cflat')]),
        lanes(cil_jobs(('podnet','wa','gem','icarl_cflat'),seeds=(1,2))))),
    'C3': lanes(cil_jobs(TUNED, ('nobn',),seeds=(1,2))),
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

RUNNER = r'''if PRESET == 'M1s0':
    raise RuntimeError('MAIN RUN HELD: review exact-solver T4 profiling and A–D NaN report before enabling M1')
import re
NOBN_LR_SCALE = .5  # registered noBN lr = 0.05, chosen by the 3-task baseline probe (PREREGISTRATION_CHANGES)
from concurrent.futures import ThreadPoolExecutor
from scripts.collect_parallel import validate_result
ROOT = STAGE_B / REPO_COMMIT[:12] / RUN_TAG
ROOT.mkdir(parents=True, exist_ok=True)
ENV.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

def attached_file(filename, destination, predicate):
    if destination.exists(): return destination
    candidates = []
    for root in (STAGE_B,Path('/kaggle/input')):
        for path in root.rglob(filename):
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

def schedule_name(backbone):
    if PRESET=='M1':return 'gram_power10.json'
    name='icarl_BN' if backbone=='bn' else 'icarl_noBN'
    return name+('_random' if PRESET=='M3' else '')+'.json'

def prepare_schedule(backbone):
    filename = schedule_name(backbone)
    destination = ROOT/'_schedules'/filename
    if PRESET=='M1':
        from experiments.main_cil_metric_resume import seed0_gram_schedule
        seed0_gram_schedule(STAGE_B,destination,REPO_COMMIT)
        return
    from methods.vna_v3 import POLICY_VERSION
    attached_file(filename, destination, lambda v: v.get('code_commit') == REPO_COMMIT
                  and v.get('source_digest') == SOURCE_DIGEST and v.get('authoritative')
                  and v.get('backbone') == ('resnet32' if backbone == 'bn' else 'resnet32_nobn')
                  and v.get('policy_version')==POLICY_VERSION and v.get('gir_live') is False
                  and v.get('solver')=='cg_ir' and v.get('curvature')=='exact')
    if destination.exists(): return
    cmd = [sys.executable,'-u','-m','experiments.main_cil','--arm','icarl_full',
           '--backbone','resnet32' if backbone == 'bn' else 'resnet32_nobn',
           '--class-order','random' if PRESET=='M3' else 'superclass','--seed','0',
           '--out',str(ROOT/'_preflight'/(backbone+('_random' if PRESET=='M3' else ''))),
           '--device','cuda:0','--data-dir',str(DATA_ROOT),'--solver','cg_ir','--curvature','exact','--preflight-only',
           '--schedule-output',str(destination),'--schedule-source','self' if PRESET == 'M3' else 'authority']
    if backbone == 'nobn': cmd += ['--nobn-lr-scale',str(NOBN_LR_SCALE)]
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
        cmd=[sys.executable,'-u','-m','experiments.v4_competitors','--method',job.split('__')[0],
            '--mode','sweep','--device','cuda:0','--data-dir',str(DATA_ROOT),
            '--out',str(ROOT/job),'--resume','--nobn-lr-scale',str(NOBN_LR_SCALE)]
        if PRESET=='C2' and job.startswith('gem__'):cmd+=['--reuse-root',str(STAGE_B)]
        return cmd
    arm,backbone,order,seed = job.split('__'); seed = seed[1:]
    if arm in ('icarl','icarl_full'):
        cmd = [sys.executable,'-u','-m','experiments.main_cil','--arm',arm,
               '--backbone','resnet32' if backbone == 'bn' else 'resnet32_nobn',
               '--class-order',order,'--seed',seed,'--out',str(ROOT/job),
               '--solver','gram' if PRESET=='M1' else 'cg_ir','--curvature','power10' if PRESET=='M1' else 'exact',
               '--device','cuda:0','--data-dir',str(DATA_ROOT)]
        if arm == 'icarl_full': cmd += ['--schedule',str(ROOT/'_schedules'/schedule_name(backbone))]
        if backbone == 'nobn': cmd += ['--nobn-lr-scale',str(NOBN_LR_SCALE)]
        return cmd
    cmd = [sys.executable,'-u','-m','experiments.v4_competitors','--method',arm,
           '--backbone',backbone,'--order',order,'--seed',seed,'--out',str(ROOT/job),
           '--device','cuda:0','--data-dir',str(DATA_ROOT),'--resume']
    if PRESET in ('C2','C3','C4'):
        destination = ROOT/'_configs'/arm/'best_config.json'
        path = ROOT/(arm+'__sweep')/'best_config.json' if PRESET=='C2' else None
        if path is None:path = attached_file('best_config.json', destination, lambda v: v.get('method') == arm
                             and v.get('code_commit') == REPO_COMMIT and v.get('source_digest') == SOURCE_DIGEST
                             and v.get('synthetic') is False)
        if path is None: raise FileNotFoundError(f'Attach the current C1 best_config.json for {arm}')
        cmd += ['--best-config',str(path)]
    if PRESET == 'C5': cmd += ['--reproduction']
    if backbone == 'nobn': cmd += ['--nobn-lr-scale',str(NOBN_LR_SCALE)]
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
    if PRESET in ('M1','M2','M3','M1s0') and not pause_due(): prepare_schedule('bn' if PRESET == 'M2' else 'nobn')
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
    if PRESET=='C2' and not pause_due():
        with ThreadPoolExecutor(max_workers=2) as sweep_pool:
            sweeps=[sweep_pool.submit(run_gpu,gpu,[j for j in jobs if j.endswith('__sweep')])
                    for gpu,jobs in enumerate((JOBS_GPU0,JOBS_GPU1))]
            for future in sweeps:future.result()
    if not pause_due():
        if PRESET=='C2':
            JOBS_GPU0=[j for j in JOBS_GPU0 if not j.endswith('__sweep')]
            JOBS_GPU1=[j for j in JOBS_GPU1 if not j.endswith('__sweep')]
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
if PRESET in ('C1','C2'):
    methods=('icarl_cbp','icarl_redo','icarl_snp','icarl_upgd') if PRESET=='C1' else ('podnet','wa','gem','icarl_cflat')
    for method in methods:
        selection = ROOT/(method+'__sweep')/'best_config.json'
        if not selection.exists(): continue
        best = json.loads(selection.read_text())
        directory = selection.parent/best['selected_trial']
        record = {key:json.loads((directory/(key+'.json')).read_text())
                  for key in ('summary','protocol','tasks','diagnostics')}
        job = method+'__nobn__superclass__s0'
        validate_result(job,record['summary'],record['protocol'],record['tasks'])
        records[job] = dict(record,job=job)
if PRESET != 'B5w': report(records)
flags = []
for record in records.values():
    events = record.get('interventions',record.get('vna',[]))
    counts = Counter()
    for event in events:
        for flag in ('gir_fallback','rejected','nonfinite_after_apply','chol_fail','leak_monotonicity_violation'):
            value = event.get(flag,False)
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
    cells[0]=cell('markdown','# VNA v4 main run\nAttach previous output archives to resume. M1: Gram/power10, seeds1/2 only, attach seed0 Gram output for frozen K0. C1/C2: joint method-config x four-LR seed0 sweep. C2: noBN only, winner supplies seed0. C3: matched seeds1/2 plus C1 winner seed0. C3/C4 require current C1 selections.')
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
    runner=RUNNER[RUNNER.index('import re\n'):] if preset=='M1' else RUNNER
    cells[-1]=cell('code',runner)
    cells.append(cell('code',REPORT))
    result['cells']=[cells[1],cells[0],*cells[2:]]
    return result


def build_exact_diagnostics(commit, source_digest):
    result = build(commit, 'M1s0', 'vna_v4_exact_diagnostics', source_digest)
    result['cells'][0] = cell('code', config(commit, (['diagnostics'], ['diagnostics'], []),
        'vna_v4_exact_diagnostics', 'EXACT_DIAGNOSTICS', source_digest)+"\nMODE = 'profile'  # GPU0: current CG-IR exact + capacity; GPU1: current Gram + fixed reference, then CG-IR power10; M1 held\n")
    setup = ''.join(result['cells'][3]['source'])
    setup = setup.replace('torch.cuda.device_count() != 2', 'torch.cuda.device_count() < 1').replace('Select T4 x2', 'Select at least one T4')
    result['cells'][3] = cell('code', setup)
    inputs = 'Attach CIFAR-100 and the old task-1 checkpoint or an output archive; checkpoint discovery scans all inputs automatically.'
    inputs += ' Attach the matching source ZIP.' if source_digest else ' Code loads directly from the pinned Git commit; no source ZIP is needed.'
    result['cells'][1] = cell('markdown', '# T29–T31: CG versus frozen Gram reference\n'+inputs+' GPU0 runs current D-IR (shared D_site, exact noBN curvature, CG-IR) with --capacity-profile at m=12800 and m=25600. GPU1 runs current Gram/power10 with a fixed-system 8c56d02 check, then current CG-IR/power10 to isolate the solver change. All full applications use the same current policy; the old reference checks fixed-system math only. Both start from the same task-1 checkpoint and profile the first task-2 application. Compare strictly at rel tol 1e-5; failures remain explicit. With one GPU run CG first. M1 stays held. After reviewing profile results, MODE=smoke3 runs baseline/full seed 0 for three full training tasks, with a fresh CG authority preflight. It does not run M1.')
    import textwrap
    result['cells'][-2] = cell('code', "if MODE == 'smoke3':\n"+textwrap.indent(CG_SMOKE3,'    ')+"\nelse:\n"+textwrap.indent(EXACT_DIAGNOSTICS,'    '))
    result['cells'][-1] = cell('code', r'''summary_root = STAGE_B/'exact_diagnostics'/MODE/REPO_COMMIT[:12]
for path in sorted(summary_root.rglob('diagnostic.json')):
    report = json.loads(path.read_text())
    event = (report.get('applications') or [{}])[0]
    print(path)
    print(json.dumps({**{k:report.get(k) for k in ('status','code_commit','application_seconds',
        'application_peak_cuda_bytes','final_L_KD','post_application_sgd','fixed_gram_reference','capacity_profile','ir_summary','gamma_precision')},
        **{k:event.get(k) for k in ('mu_star','rho_tr','swaps','skip_reason','downstream')},
        'cg_systems':len(report.get('cg',[]))},indent=2))
for path in sorted(summary_root.glob('*/summary.json')):
    report = json.loads(path.read_text())
    print(path, {k:report.get(k) for k in ('status','tasks_run','seconds','diverged')})
print('Full reports, candidate rankings and comparisons are retained in the output archive.')
archive_outputs()
''')
    return result


CG_SMOKE3 = r'''from concurrent.futures import ThreadPoolExecutor
root = STAGE_B/'exact_diagnostics'/'smoke3'/REPO_COMMIT[:12]
root.mkdir(parents=True,exist_ok=True)
schedule = root/'schedule.json'
common = [sys.executable,'-u','-m','experiments.main_cil','--backbone','resnet32_nobn',
    '--class-order','superclass','--nobn-lr-scale','.5','--seed','0','--device','cuda:0',
    '--data-dir',str(DATA_ROOT),'--solver','cg_ir','--curvature','exact']
try:
    if not schedule.exists():
        launch(common+['--arm','icarl_full','--preflight-only','--schedule-output',str(schedule),
            '--out',str(root/'_preflight')],0,'smoke3_preflight',RUN_ROOT/'smoke3_preflight.log')
    if schedule.exists() and not pause_due():
        gpu_count = min(2,torch.cuda.device_count())
        if gpu_count < 1: raise RuntimeError('smoke3 requires a GPU')
        def run_smoke_lane(gpu,jobs):
            for arm in jobs:
                command = common+['--arm',arm,'--max-tasks','3','--out',str(root/arm)]
                if arm=='icarl_full': command+=['--schedule',str(schedule)]
                if not launch(command,gpu,'smoke3_'+arm,RUN_ROOT/f'smoke3_{arm}.log'): return
                archive_outputs()
        with ThreadPoolExecutor(max_workers=gpu_count) as pool:
            pending=[pool.submit(run_smoke_lane,gpu,('icarl_full','icarl')[gpu::gpu_count]) for gpu in range(gpu_count)]
            for future in pending: future.result()
finally:
    archive_outputs()
print('Three-task rehearsal only. M1 remains held.')
'''


EXACT_DIAGNOSTICS = r'''OLD_REPO = REPO.with_name('counterfactual-projection-d88abc4')
OLD_COMMIT = 'd88abc496eb1fe21d1b6df66d90b47cf93d7ded1'
REFERENCE_REPO = REPO.with_name('counterfactual-projection-profile-reference')
REFERENCE_COMMIT = '8c56d02dab1ee16c47eb9138bf4aa3a64b9b45e2'
if MODE == 'profile':
    if REFERENCE_REPO.exists():
        ref_head = subprocess.check_output(['git','-C',str(REFERENCE_REPO),'rev-parse','HEAD'],text=True).strip()
        ref_status = subprocess.check_output(['git','-C',str(REFERENCE_REPO),'status','--porcelain','--untracked-files=no'],text=True).strip()
        if ref_head != REFERENCE_COMMIT or ref_status: shutil.rmtree(REFERENCE_REPO)
    if not REFERENCE_REPO.exists(): private_clone(MAIN_URL, REFERENCE_REPO, REFERENCE_COMMIT)
    ref_head = subprocess.check_output(['git','-C',str(REFERENCE_REPO),'rev-parse','HEAD'],text=True).strip()
    ref_status = subprocess.check_output(['git','-C',str(REFERENCE_REPO),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if ref_head != REFERENCE_COMMIT or ref_status: raise RuntimeError('Reference D must be a clean 8c56d02 tree')
    for upstream in (REPO/'third_party').iterdir():
        destination = REFERENCE_REPO/'third_party'/upstream.name
        if upstream.is_dir() and not destination.exists(): destination.symlink_to(upstream, target_is_directory=True)
if OLD_REPO.exists():
    head = subprocess.run(['git','-C',str(OLD_REPO),'rev-parse','HEAD'],capture_output=True,text=True)
    status = subprocess.run(['git','-C',str(OLD_REPO),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True)
    shared = (OLD_REPO/'.git/objects/info/alternates').exists()
    if shared or head.returncode or head.stdout.strip() != OLD_COMMIT or status.returncode or status.stdout.strip():
        print('Rebuilding incomplete or shared historical source checkout:', OLD_REPO)
        shutil.rmtree(OLD_REPO)
if not OLD_REPO.exists():
    private_clone(MAIN_URL, OLD_REPO, OLD_COMMIT)
head = subprocess.check_output(['git','-C',str(OLD_REPO),'rev-parse','HEAD'],text=True).strip()
status = subprocess.check_output(['git','-C',str(OLD_REPO),'status','--porcelain','--untracked-files=no'],text=True).strip()
if head != OLD_COMMIT or status: raise RuntimeError('Historical source checkout is not a clean d88abc4 tree')
for upstream in (REPO/'third_party').iterdir():
    destination = OLD_REPO/'third_party'/upstream.name
    if upstream.is_dir() and not destination.exists(): destination.symlink_to(upstream, target_is_directory=True)
script = REPO/'diagnostics/exact_solver.py'
from experiments.runners.pycil import upstream_imports
from diagnostics.exact_solver import find_task1_checkpoint
upstream_imports()
checkpoint = find_task1_checkpoint(Path('/kaggle/input'), STAGE_B/'exact_diagnostics'/'task1_source.pt')
from concurrent.futures import ThreadPoolExecutor, as_completed
diagnostic_root = STAGE_B/'exact_diagnostics'/MODE/REPO_COMMIT[:12]
branches = ('D-IR','D-Gram','D-IR-power10') if MODE == 'profile' else ('A','B','C','D') if MODE == 'nan' else ()
if not branches: raise ValueError('MODE must be profile or nan')
gpu_count = min(2, torch.cuda.device_count())
if gpu_count < 1: raise RuntimeError('Diagnostics require at least one T4 GPU')
ENV['VNA_DIAGNOSTIC_LANES'] = str(gpu_count)
queues = [('D-IR',),('D-Gram','D-IR-power10')] if MODE == 'profile' and gpu_count == 2 else [branches[gpu::gpu_count] for gpu in range(gpu_count)]
outcomes = {}
print('Diagnostic GPU queues:', {gpu: list(jobs) for gpu, jobs in enumerate(queues)}, flush=True)
def run_diagnostic_lane(gpu, jobs):
    for branch in jobs:
        tree = OLD_REPO if branch in ('B','C') else REPO
        directory = diagnostic_root/branch
        directory.mkdir(parents=True,exist_ok=True)
        command = [sys.executable,'-u',str(script),'--repo',str(tree),'--branch','D' if branch.startswith('D-') else branch,
            '--checkpoint',str(checkpoint),'--data-dir',str(DATA_ROOT),'--out',str(directory),
            '--curvature','power10' if branch in ('D-Gram','D-IR-power10','B','C') else 'exact']
        if MODE == 'profile':
            command+=['--profile','--solver','gram' if branch=='D-Gram' else 'cg_ir']
        if MODE == 'profile' and branch=='D-Gram':command+=['--gram-reference',str(REFERENCE_REPO)]
        if MODE == 'profile' and branch == 'D-IR': command.append('--capacity-profile')
        try:
            if not launch(command,gpu,f'{MODE}_{branch}',RUN_ROOT/f'{MODE}_{branch}.log'):
                outcomes[branch] = {'status':'PAUSED'}
                return False
            outcomes[branch] = {'status':'COMPLETED'}
        except Exception as error:
            outcomes[branch] = {'status':'FAILED','error':str(error)}
            if MODE != 'profile' or branch == 'D-IR': raise
            print(f'Advisory Gram reference check {branch} failed: {error}',flush=True)
        archive_outputs()
    return True
try:
    with ThreadPoolExecutor(max_workers=gpu_count) as pool:
        pending = [pool.submit(run_diagnostic_lane, gpu, jobs) for gpu, jobs in enumerate(queues)]
        failures = []
        for future in as_completed(pending):
            try: future.result()
            except Exception as error: failures.append(error)
    if MODE == 'profile':
        from diagnostics.exact_solver import compare_profiles
        from experiments.io_utils import atomic_json_save
        def profile_result(branch):
            outcome = outcomes.get(branch,{'status':'NOT_RUN'})
            path = diagnostic_root/branch/'diagnostic.json'
            if outcome['status'] != 'COMPLETED': return outcome
            try: return json.loads(path.read_text())
            except (OSError,ValueError) as error: return {'status':'MISSING_REPORT','error':str(error)}
        reference,power,current = (profile_result(branch) for branch in ('D-Gram','D-IR-power10','D-IR'))
        atomic_json_save(reference.get('fixed_gram_reference',dict(status='INCOMPLETE')),diagnostic_root/'fixed_gram_reference.json')
        for name,left,right,strict in (('ir_power10_vs_gram',reference,power,True),
                                      ('ir_exact_vs_power10',power,current,False)):
            comparison=compare_profiles(left,right,strict=strict)
            comparison.update(advisory=True,execution=outcomes)
            atomic_json_save(comparison,diagnostic_root/(name+'.json'))
            print(json.dumps({k:comparison[k] for k in ('status','checks','application_seconds','lm_numeric_audit') if k in comparison},indent=2),flush=True)
    if failures: raise failures[0]
finally:
    archive_outputs()
print('M1 remains held: review the actual profiles and A–D diagnostic reports.')
'''


def build_gram_m1_ir(commit,source_digest=None):
    result=build(commit,'M1s0','vna_v4_gram_m1s0_ir',source_digest)
    result['cells'][0]=cell('code',config(commit,(['icarl_full Gram seed0'],['Gram verification; CG-IR profile'],[]),
        'vna_v4_gram_m1s0_ir','GRAM_M1S0_IR_PROFILE',source_digest))
    result['cells'][1]=cell('markdown', '''# Gram M1s0 + CG-IR profile — current policy, T4 × 2
Attach CIFAR-100, the d88abc4 full task-1 checkpoint, and old baseline/output archives.
Both active lanes use the same immutable source commit and dead-only Schur/rho>=1/4 policy.
The 8c56d02 checkout checks fixed-system Gram direction math only; changed
LM/GIR decisions are not compared to that old policy.

GPU0 calibrates Gram/power10 K0 from three task-2 points while GPU1 profiles
current Gram on the shared checkpoint and checks the fixed system against 8c56d02.
Full Gram M1s0 starts only after the current profile completes and that check passes. GPU1 then
profiles CG-IR/exact with capacity measurements and stops; it never trains M1.
After full completes, GPU0 checks the old baseline against a fresh 150-epoch
baseline prefix; exact loss equality permits historical result reuse, otherwise
training continues to task 20. Missing old baseline runs a fresh full baseline.
Outputs/checkpoints resume from attached archives; archives update every 15 min.
Dead/dormant probe=256; B uses CG-IR on the first 128 of those samples.
''')
    bootstrap=''.join(result['cells'][2]['source']).replace('/kaggle/working/vna4_main','/kaggle/working/vna4_gram_m1s0_ir')
    result['cells'][2]=cell('code',bootstrap)
    runtime=''.join(result['cells'][4]['source']).replace('archive_stop.wait(3600)','archive_stop.wait(900)')
    result['cells'][4]=cell('code',runtime)
    result['cells'][-2]=cell('code',GRAM_M1_IR_SETUP+GRAM_M1_IR_RUNNER)
    result['cells'][-1]=cell('code',r'''for name in ('gram_compatibility.json','cg_ir_vs_gram.json'):
    path=ROOT/'profiles'/name
    if path.exists():print(name,path.read_text())
for name in ('icarl_full','icarl'):
    path=ROOT/name/'summary.json'
    if path.exists():
        value=json.loads(path.read_text())
        if value.get('status')=='REUSED_BASELINE':
            reused=json.loads((path.parent/'reused_baseline.json').read_text())
            value=reused['reference']['summary']
            print('Historical baseline reused after exact 150-epoch loss verification:',reused['verification'])
        print(name,{k:value.get(k) for k in ('status','tasks_run','seconds','cnn_average_accuracy','nme_average_accuracy')})
path=ROOT/'profiles'/'cg_ir'/'diagnostic.json'
if path.exists():
    value=json.loads(path.read_text());event=(value.get('applications') or [{}])[0]
    print(json.dumps({**{k:value.get(k) for k in ('status','application_seconds','application_peak_cuda_bytes','post_application_sgd',
        'capacity_profile','gamma_precision','ir_summary','timers','final_L_KD')},
        'downstream':event.get('downstream'),'solves':value.get('cg')},indent=2))
archive_outputs()
print('GPU1 has no further jobs. CG-IR M1 remains unapproved and was not launched.')
''')
    return result


GRAM_M1_IR_SETUP=r'''from concurrent.futures import ThreadPoolExecutor, as_completed
from experiments.io_utils import atomic_json_save
from diagnostics.exact_solver import find_task1_checkpoint, compare_profiles
from experiments.main_cil_baseline import prepare_reference
from experiments.runners.pycil import upstream_imports
ROOT=STAGE_B/RUN_TAG/REPO_COMMIT[:12]
ROOT.mkdir(parents=True,exist_ok=True)
if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
    raise RuntimeError('This notebook requires T4 x2')
ENV['VNA_DIAGNOSTIC_LANES']='2'
REFERENCE_COMMIT='8c56d02dab1ee16c47eb9138bf4aa3a64b9b45e2'
REFERENCE_REPO=REPO.with_name('counterfactual-projection-gram-8c56d02')
if REFERENCE_REPO.exists():
    head=subprocess.check_output(['git','-C',str(REFERENCE_REPO),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(REFERENCE_REPO),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if head!=REFERENCE_COMMIT or dirty:shutil.rmtree(REFERENCE_REPO)
if not REFERENCE_REPO.exists():private_clone(MAIN_URL,REFERENCE_REPO,REFERENCE_COMMIT)
for upstream in (REPO/'third_party').iterdir():
    destination=REFERENCE_REPO/'third_party'/upstream.name
    if upstream.is_dir() and not destination.exists():destination.symlink_to(upstream,target_is_directory=True)
upstream_imports()
from experiments.main_cil_metric_resume import restore_metric_trace_run
restore_metric_trace_run(STAGE_B,RUN_TAG,REPO_COMMIT)
checkpoint=find_task1_checkpoint(Path('/kaggle/input'),ROOT/'profiles'/'task1_source.pt')
script=REPO/'diagnostics/exact_solver.py'
'''

GRAM_M1_IR_RUNNER=r'''gate=threading.Event()
gram_check={}
main_common=[sys.executable,'-u','-m','experiments.main_cil','--solver','gram','--curvature','power10',
    '--seed','0','--backbone','resnet32_nobn','--class-order','superclass','--nobn-lr-scale','.5',
    '--device','cuda:0','--max-tasks','20','--data-dir',str(DATA_ROOT)]

def profile(name,tree,solver,curvature,capacity=False,reference=False):
    out=ROOT/'profiles'/name;out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-u',str(script),'--repo',str(tree),'--branch','D','--profile',
        '--solver',solver,'--curvature',curvature,'--checkpoint',str(checkpoint),
        '--data-dir',str(DATA_ROOT),'--out',str(out)]
    if capacity:cmd+=['--capacity-profile']
    if reference:cmd+=['--gram-reference',str(REFERENCE_REPO)]
    if not launch(cmd,1,name,RUN_ROOT/(name+'.log')):return None
    archive_outputs()
    return json.loads((out/'diagnostic.json').read_text())

def run_gpu1():
    try:
        current=profile('gram_current',REPO,'gram','power10',reference=True)
        if current is None:return
        result=dict(current.get('fixed_gram_reference',dict(status='INCOMPLETE')))
        if current.get('status')!='PROFILE_COMPLETED':result.update(status='FAIL',reason='Current Gram profile incomplete')
        result['profile_identity']=current.get('identity')
        atomic_json_save(result,ROOT/'profiles'/'gram_compatibility.json')
        gram_check.update(result)
        if result['status']!='PASS':raise RuntimeError('Fixed-system Gram check must PASS before full training')
    finally:
        gate.set()
        archive_outputs()
    try:
        cheap=profile('cg_ir',REPO,'cg_ir','exact',capacity=True)
        if cheap is None:return
    except Exception as error:
        path=ROOT/'profiles'/'cg_ir'/'diagnostic.json'
        cheap=json.loads(path.read_text()) if path.exists() else dict(status='FAILED_DIAGNOSTIC',error=str(error))
        print('CG-IR profile failed; the approved Gram training lane continues:',error,flush=True)
    result=compare_profiles(current,cheap,strict=True,allow_curvature_change=True)
    if result['status']=='INCOMPLETE':
        result.update(status='FAIL',comparison_status='INCOMPLETE',reason='Profile did not complete')
    result.setdefault('lm_numeric_audit',dict(status='UNAVAILABLE',reason='Profile incomplete'))
    atomic_json_save(result,ROOT/'profiles'/'cg_ir_vs_gram.json')
    print('CG-IR versus same-commit Gram:',json.dumps(result,indent=2),flush=True)
    lane_state[1]={'job':'idle','last':'CG-IR profile finished; no cheap M1 authorized'}
    archive_outputs()

def run_gpu0():
    schedule=ROOT/'_schedules'/'gram_power10.json'
    if not schedule.exists():
        cmd=main_common+['--arm','icarl_full','--preflight-only','--schedule-output',str(schedule),
            '--out',str(ROOT/'_preflight'/'gram')]
        if not launch(cmd,0,'gram_preflight',RUN_ROOT/'gram_preflight.log'):return
        archive_outputs()
    while not gate.wait(1):
        if pause_due():return
    if pause_due():return
    if gram_check.get('status')!='PASS':raise RuntimeError('Gram reference check failed; full training held')
    cmd=main_common+['--arm','icarl_full','--schedule',str(schedule),'--out',str(ROOT/'icarl_full')]
    if not launch(cmd,0,'gram_full_s0',RUN_ROOT/'gram_full_s0.log'):return
    archive_outputs()
    full=json.loads((ROOT/'icarl_full'/'summary.json').read_text())
    if full.get('status')!='COMPLETED':raise RuntimeError('Gram full did not finish; baseline remains queued')
    baseline=ROOT/'icarl';reference=ROOT/'baseline_reference.json'
    if reference.exists() or not (baseline/'latest.pt').exists():
        try:prepare_reference((STAGE_B,Path('/kaggle/input')),reference)
        except ValueError as error:print('Old baseline cannot be reused; run fresh:',error,flush=True)
    cmd=main_common+['--arm','icarl','--out',str(baseline)]
    if reference.exists():cmd+=['--baseline-reference',str(reference)]
    if not pause_due():launch(cmd,0,'gram_baseline_s0',RUN_ROOT/'gram_baseline_s0.log')
    archive_outputs()

failures=[]
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_gpu0),pool.submit(run_gpu1)]
        for future in as_completed(futures):
            try:future.result()
            except Exception as error:failures.append(error)
finally:
    archive_stop.set();archiver.join(timeout=10);archive_outputs()
if failures:raise failures[0]
if archive_errors:raise RuntimeError(archive_errors)
'''

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
        output.with_name('vna4_gram_m1s0_ir.ipynb').write_text(json.dumps(
            build_gram_m1_ir(args.commit,digest),indent=1)+'\n')
        output.with_name('vna4_exact_diagnostics.ipynb').write_text(json.dumps(
            build_exact_diagnostics(args.commit, digest), indent=1)+'\n')
        for name in PRESETS:
            stem = output.stem[:-3] if output.stem.endswith('_M1') else output.stem
            path=output.with_name(stem+'_'+name+output.suffix)
            path.write_text(json.dumps(build(args.commit,name,args.run_tag,digest),indent=1)+'\n')


if __name__=='__main__': main()
