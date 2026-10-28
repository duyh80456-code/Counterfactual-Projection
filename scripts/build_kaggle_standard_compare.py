"""Seed-zero H1-H4 paired full runs; immutable source, no ZIP overlay."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_kaggle_pilot_common import cell,BOOTSTRAP


def jobs(branch):
    cil=['--seed','0','--tasks','20','--baseline','icarl']
    def job(name,module,args):return dict(name=name,module=module,args=args)
    if branch=='H1':return ([job('icarl_s0','experiments.standard_cl',cil+['--arm','baseline'])],
        [job('icarl_replay_vna_s0','experiments.standard_cl',cil+['--arm','h1'])])
    if branch=='H2':
        args=['--seed','0','--config','configs/standard_cl/h2_online_cifar100.json']
        return ([job('lpr_s0','experiments.standard_online',args+['--arm','lpr'])],
            [job('geometry_vna_s0','experiments.standard_online',args+['--arm','h2'])])
    if branch=='H3':
        args=['--seed','0','--config','configs/standard_cl/h3_imagenet_small.json']
        return ([job('cbp_s0','experiments.standard_imagenet',args+['--arm','cbp']),
                 job('bp_s0','experiments.standard_imagenet',args+['--arm','bp'])],
            [job('plasticity_vna_gir_s0','experiments.standard_imagenet',args+['--arm','vna_gir']),
             job('plasticity_vna_s0','experiments.standard_imagenet',args+['--arm','vna']),
             job('plasticity_gir_s0','experiments.standard_imagenet',args+['--arm','gir'])])
    if branch=='H4':
        return tuple([job(f'{"er" if arm=="baseline" else "domain_vna"}_{kind}_s0','experiments.standard_domain',['--seed','0','--arm',arm,
            '--config',f'configs/standard_cl/h4_{kind}_mnist.json']) for kind in ('rotated',)]
            for arm in ('baseline','h1'))
    raise ValueError(branch)


def lock_geometry_config(config,authority,data_dir,source_commit):
    import hashlib,math
    authority=Path(authority);report=json.loads(authority.read_text())
    weight=report.get('selected_weight');source=report.get('benchmark_config',{})
    if report.get('status')!='SELECTED' or isinstance(weight,bool) or not isinstance(weight,(int,float)) or not math.isfinite(weight) or weight<0:
        raise ValueError('Geometry authority must contain a selected finite weight; fallback is not an authority')
    if report.get('seed')!=0 or report.get('source_commit')!=source_commit:raise ValueError('Geometry authority seed/source mismatch')
    if weight not in source.get('geometry_selection',{}).get('weights',[]):raise ValueError('Selected weight outside registered grid')
    def native(cfg):return {k:v for k,v in cfg.items() if k not in ('geometry_weight','geometry_selection','correction','require_geometry_authority')}
    if native(source)!=native(config):raise ValueError('Geometry authority native protocol mismatch')
    allowed={'preconditioner_probes','damping_retries','eval_acceptance'}
    if {k:v for k,v in source['correction'].items() if k not in allowed}!={k:v for k,v in config['correction'].items() if k not in allowed}:
        raise ValueError('Geometry authority differs beyond numerical ablation controls')
    fingerprints=report.get('dataset_fingerprint',{})
    for name in ('train','test','meta'):
        path=Path(data_dir)/'cifar-100-python'/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=fingerprints.get(name):raise ValueError('Geometry authority dataset mismatch: '+name)
    result=json.loads(json.dumps(config));result.pop('geometry_selection',None);result.pop('require_geometry_authority',None);result['geometry_weight']=weight
    result['geometry_authority']=dict(status='LOCKED',selected_weight=weight,seed=0,source_commit=source_commit,
        sha256=hashlib.sha256(authority.read_bytes()).hexdigest(),validation_sha256=report.get('validation_sha256'))
    return result


SETUP=r'''os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
print('KAGGLE HOST',sys.version,flush=True)
print(subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,flush=True)
import importlib.util
for package in ('pandas','matplotlib'):
    if importlib.util.find_spec(package) is None:
        subprocess.run([sys.executable,'-m','pip','install','-q',package],check=True)
def fetch(name,url,pin):
    path=REPO/'third_party'/name
    if not path.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',url,str(path)],check=True)
        subprocess.run(['git','-C',str(path),'fetch','origin',pin],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',pin],check=True)
    if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=pin:
        raise ValueError('Upstream pin mismatch: '+name)
if BRANCH=='H3':fetch('loss-of-plasticity','https://github.com/shibhansh/loss-of-plasticity.git','a6b79580d85f3025bdb601566d3627c5f489f13b')
else:fetch('PyCIL','https://github.com/LAMDA-CL/PyCIL.git','f3509b8ca3f20660ce4aa13f19d5283de81b4b35')
if BRANCH=='H2':fetch('LPR','https://github.com/plai-group/LPR.git','76c44f8f2481f4e23881b2463eb58aaf94affabb')
PYTHON=sys.executable
if BRANCH=='H2':
    if importlib.util.find_spec('uv') is None:subprocess.run([sys.executable,'-m','pip','install','-q','uv'],check=True)
    environment=RUN_ROOT/'lpr_env'
    if not (environment/'bin/python').exists():
        subprocess.run([sys.executable,'-m','uv','venv','--python','3.10',str(environment)],check=True)
    PYTHON=str(environment/'bin/python')
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PYTHON,'torch==2.2.2','torchvision==0.17.2',
        '-r',str(REPO/'requirements-h2.txt'),'-r',str(REPO/'requirements-upstream.txt'),'-e',str(REPO)],check=True)
    # Only this tested torch typing metadata exception is allowed by the API/dependency gate.
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PYTHON,'--no-deps',
        'avalanche-lib==0.4.0','typing-extensions==4.4.0'],check=True)
else:
    subprocess.run([PYTHON,'-m','pip','install','-q','-e',str(REPO)+'[vision,diagnostics]',
        '-r',str(REPO/'requirements-upstream.txt')],check=True)
if BRANCH=='H3':subprocess.run([PYTHON,'-m','pip','install','-q','gdown==5.2.0'],check=True)
sys.path.insert(0,str(REPO))
CONFIG={'H1':'configs/standard_cl/h1_cifar100_bn.json','H2':'configs/standard_cl/h2_online_cifar100.json',
        'H3':'configs/standard_cl/h3_imagenet_small.json','H4':'configs/standard_cl/h4_rotated_mnist.json'}[BRANCH]
PREFLIGHT_REPORT=RUN_ROOT/'preflight'/RUN_TAG/'preflight.json'
preflight=[PYTHON,'-m','diagnostics.standard_preflight','--branch',BRANCH,'--config',CONFIG,
    '--stage',str(RUN_ROOT/'input_data'),'--out',str(PREFLIGHT_REPORT),'--device','cuda:0']
if DATA_DIR:preflight+=['--data-dir',DATA_DIR]
env={**os.environ,'PYTHONPATH':str(REPO),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLBACKEND':'Agg'}
subprocess.run(preflight,cwd=REPO,env=env,check=True)
report=json.loads(PREFLIGHT_REPORT.read_text())
if report['status']!='PASSED':raise RuntimeError('Preflight did not pass; training blocked')
RESOLVED_DATA=report['data_dir']
print('PREFLIGHT PASSED; seed0; input=',RESOLVED_DATA,'source=',REPO_COMMIT,flush=True)
'''


RUNNER=r'''from concurrent.futures import ThreadPoolExecutor
from experiments.vna_kaggle import restore_stage_archives,archive_outputs
from experiments.vna_resume import launch_logged,print_heartbeat
from experiments.io_utils import atomic_json_save
from scripts.collect_standard_cl import collect,export_comparison
ROOT_OUT=OUTPUT/RUN_TAG
if any(ROOT_OUT.rglob('latest.pt')) or any(ROOT_OUT.rglob('summary.json')):
    print('Keeping local working checkpoints; attached older output will not overwrite them',flush=True)
else:restore_stage_archives(Path('/kaggle/input'),OUTPUT,'standard_outputs',prefix=RUN_TAG)
ROOT_OUT.mkdir(parents=True,exist_ok=True)
VNA_DEADLINE=SESSION_START+11*3600
base_env={**os.environ,'MPLBACKEND':'Agg','PYTHONPATH':str(REPO),'PYTHONUNBUFFERED':'1','TQDM_DISABLE':'1',
    'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION':'python','OMP_NUM_THREADS':'1',
    'MKL_NUM_THREADS':'1','VNA_DEADLINE':str(VNA_DEADLINE)}
configs=RUN_ROOT/'effective_configs';configs.mkdir(exist_ok=True)
def command(job):
    args=list(job['args'])
    if job.get('geometry_lock'):
        from scripts.build_kaggle_standard_compare import lock_geometry_config
        authority=globals().get('GEOMETRY_AUTHORITY')
        if authority is None:
            import hashlib
            choices={}
            for path in Path('/kaggle/input').rglob('geometry_selection.json'):
                report=json.loads(path.read_text())
                if report.get('status')=='SELECTED' and report.get('seed')==0 and report.get('source_commit')==REPO_COMMIT:
                    choices[hashlib.sha256(path.read_bytes()).hexdigest()]=path
            if len(choices)!=1:raise ValueError('Attach one matching main H2 geometry_selection.json or set GEOMETRY_AUTHORITY explicitly')
            authority=next(iter(choices.values()))
        index=args.index('--config');config=json.loads((REPO/args[index+1]).read_text())
        config=lock_geometry_config(config,authority,RESOLVED_DATA,REPO_COMMIT)
        target=configs/(job['name']+'.json');target.write_text(json.dumps(config,indent=2)+'\n');args[index+1]=str(target)
    if BRANCH=='H1':
        index=args.index('--config') if '--config' in args else None
        source=args[index+1] if index is not None else 'configs/standard_cl/h1_cifar100_bn.json'
        config=json.loads((REPO/source).read_text());config['workers']=1
        target=configs/Path(source).name;target.write_text(json.dumps(config,indent=2)+'\n')
        if index is not None:args[index+1]=str(target)
        else:args+=['--config',str(target)]
    args+=['--data-dir',RESOLVED_DATA,'--device','cuda:0']
    return [PYTHON,'-m',job['module'],*args,'--out',str(ROOT_OUT/job['name'])]
commands=[[command(job) for job in lane] for lane in JOBS]
manifest=dict(branch=BRANCH,seed=0,source_commit=REPO_COMMIT,jobs=JOBS,commands=commands,
    paper_role='baseline choice rationale only; no paper-number reproduction gate',
    runtime_overrides={'workers':1 if BRANCH=='H1' else 0,'OMP_NUM_THREADS':1,'MKL_NUM_THREADS':1},
    preflight=str(PREFLIGHT_REPORT),horizons='registered H3-small pilot; full H1/H2/H4; exit75 resumes without changing identity')
atomic_json_save(manifest,ROOT_OUT/'notebook_protocol.json')
stop=threading.Event();paused=threading.Event();lanes={};archive_lock=threading.Lock();archive_errors=[]
def archive():
    with archive_lock:return archive_outputs(RUN_ROOT,folders=('standard_outputs','preflight'),exclude_parts=('data',))
def periodic():
    while not stop.wait(300):
        try:archive()
        except Exception as exc:archive_errors.append(repr(exc));print('ARCHIVE ERROR',repr(exc),flush=True)
def heartbeat():
    while not stop.wait(30):print_heartbeat(RUN_ROOT,SESSION_START,VNA_DEADLINE,lanes)
def lane(index):
    env={**base_env,'CUDA_VISIBLE_DEVICES':str(index)}
    for job,cmd in zip(JOBS[index],commands[index]):
        print('START',job['name'],cmd,flush=True)
        if not launch_logged(cmd,cwd=REPO,env=env,log_path=RUN_ROOT/(RUN_TAG+'_'+job['name']+'.log'),
            lane=index,job=job['name'],lanes=lanes,paused=paused,deadline=VNA_DEADLINE):break
        archive()
threads=[threading.Thread(target=periodic,daemon=True),threading.Thread(target=heartbeat,daemon=True)]
for thread in threads:thread.start()
try:
    for index in range(2):
        probe=f"from experiments.vna_resume import reset_cuda_peak_memory_stats; import torch; reset_cuda_peak_memory_stats('cuda:0'); print('FRESH LANE {index} CUDA:', torch.cuda.device_count(), torch.cuda.get_device_name(0), flush=True)"
        subprocess.run([PYTHON,'-c',probe],cwd=REPO,env={**base_env,'CUDA_VISIBLE_DEVICES':str(index)},check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,index) for index in range(2)]
        for future in futures:future.result()
finally:
    stop.set()
    for thread in threads:thread.join(timeout=10)
    atomic_json_save(collect([ROOT_OUT]),ROOT_OUT/'collected.json')
    export_comparison(ROOT_OUT)
    atomic_json_save(dict(paused=paused.is_set(),archive_errors=archive_errors),ROOT_OUT/'session.json')
    archive()
'''

REPORT=r'''import pandas as pd
from IPython.display import Image,display
comparison=ROOT_OUT/'comparison.csv'
if comparison.exists():display(pd.read_csv(comparison))
if (ROOT_OUT/'accuracy_curves.png').exists():display(Image(filename=str(ROOT_OUT/'accuracy_curves.png')))
print('Archive:',RUN_ROOT.with_suffix('.tar.gz'))
print('If PAUSED: attach this Output and the same dataset, rerun the same source pin/run tag.')
print('No source ZIP, Gram authority or old VNA checkpoint is needed.')
'''



def build(branch,commit):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('immutable full commit required')
    lanes=jobs(branch)
    tag=branch.lower()+('_small' if branch=='H3' else '')
    first=f"BRANCH = {branch!r}\nREPO_COMMIT = {commit!r}\nRUN_TAG = 'standard_{tag}_s0_{commit[:12]}'\nDATA_DIR = None  # optional explicit root under /kaggle/input\nJOBS = {lanes!r}\nimport time\nSESSION_START = time.time()\n"
    bootstrap=BOOTSTRAP.replace('/kaggle/working/vna4_main','/kaggle/working/standard_cl').replace('OUTPUT = RUN_ROOT\n','OUTPUT = RUN_ROOT / "standard_outputs"\n')
    note=f'''# {branch}: official baseline vs full algorithm — seed 0

Paper references motivate baseline selection only. No comparison with paper accuracy numbers.
Pinned source commit; no ZIP overlay. Full registered horizon, final-epoch models, lightweight API/dataset preflight; no benchmark training in preflight; no truncation.
H1: CIFAR-100, BN ResNet32 PyCIL, 20 tasks ×5 classes, memory2000, 200/170 epochs.
H2: validation-only geometry grid selected from task2 within three eligible windows, then frozen; official LPR online Split-CIFAR100, SlimResNet18, 20 experiences, minibatch10 ×3 repeats, class-balanced replay2000, validation5%, LPR omega0=4/beta=2/update30 from official README, SGD0.1 without momentum/weight decay/scheduler.
H3-small: original Continual ImageNet32, first100 seed0 binary pairs, 50 epochs/task, batch100, ConvNet, SGD0.01 momentum0.9, native head reset. VNA at first full window of each task, then every200 updates/window128; about300 opportunities over100 tasks; at most16 exact swap candidates plus identity, selected by deterministic first-order LM ranking. Shortlist-only guarantee, not global optimization. Single current-task CE objective, no replay. CBP remains main baseline; BP, VNA-only and GIR-only are queued ablations. This pilot does not reproduce full-horizon CBP results.
H4: Rotated MNIST, 20 fixed angles 0,15,...,285 degrees; fixed 10-class head, ER vs ER + prior-domain constrained VNA; training-only domain IDs partition loss constraints, shared fixed backbone/head.

Inputs: {'optional original CBP ImageNet32 files; otherwise automatically download the official archive to Kaggle disk and extract only the 200 required class files' if branch=='H3' else 'optional CIFAR-100 Python train/test/meta; auto-download from official torchvision source if absent' if branch in ('H1','H2') else 'optional MNIST raw IDX files (plain/gzip); auto-download from torchvision source if absent' }.
Enable Internet and Kaggle Secret `github_token`. Select T4 ×2; baseline GPU0, full GPU1, one BLAS thread per lane.
H2 builds an isolated Python3.10 environment for the pinned original Avalanche0.4 stack.
Optional previous Output is only for exact resume. H3-small queues five arms (GPU0 CBP then BP; GPU1 full then VNA-only then GIR-only),60000 native updates per arm plus bounded interventions; the full5000×250 config remains separate and is not queued. Exact T4 timings remain unmeasured. Other branches may also need multiple 11-hour sessions. The horizon is never shortened.
'''
    cells=[cell('markdown',note),cell('code',first),cell('code',bootstrap),cell('code',SETUP),cell('code',RUNNER),cell('code',REPORT)]
    for index,c in enumerate(cells):
        c['id']=f'{branch.lower()}-{index}'
        if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')
    return dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(accelerator='gpu',dataSources=[])))


def build_ablations(branch,commit):
    notebook=build(branch,commit)
    lanes=[[],[]]
    base=jobs(branch)[1][0]
    variants={'H1':(('solver_ablation',0),('train_gate_ablation',1)),
        'H2':(('solver_ablation',0),('train_gate_ablation',1),('no_geometry',0)),
        'H3':(('cadence4000',0),('cadence1000',1),('cadence8000',0)),
        'H4':(('mixed',0),('cg',1),('train_gate',0))}[branch]
    for variant,lane in variants:
        args=list(base['args'])
        if branch in ('H1','H2'):
            config='h1_cifar100_bn' if branch=='H1' else 'h2_online_cifar100'
            if '--config' in args:args[args.index('--config')+1]=f'configs/standard_cl/{config}_{variant}.json'
            else:args+=['--config',f'configs/standard_cl/{config}_{variant}.json']
        elif branch=='H3':args[args.index('--config')+1]=f'configs/standard_cl/h3_imagenet_{variant}.json'
        elif variant=='mixed':args+=['--domain-reference','mixed']
        elif variant=='cg':args+=['--solver','cg']
        else:args+=['--no-eval-acceptance']
        lanes[lane].append(dict(name=branch.lower()+'_'+variant+'_s0',module=base['module'],args=args,geometry_lock=branch=='H2' and variant!='no_geometry'))
    if branch=='H2':
        lanes[0].insert(0,dict(name='h2_reference_locked_s0',module=base['module'],args=list(base['args']),geometry_lock=True))
        args=list(base['args']);args[args.index('--arm')+1]='er'
        lanes[1].append(dict(name='er_online_s0',module=base['module'],args=args))
    for cell in notebook['cells']:
        if cell['cell_type']=='code' and ''.join(cell['source']).startswith('BRANCH ='):
            source=''.join(cell['source']).replace("RUN_TAG = 'standard_", "RUN_TAG = 'ablation_")
            lines=source.splitlines();lines=[f'JOBS = {tuple(lanes)!r}' if line.startswith('JOBS = ') else line for line in lines]
            if branch=='H2':lines.append('GEOMETRY_AUTHORITY = None  # attach main geometry_selection.json; matched-lambda numerical ablations')
            cell['source']=[line+'\n' for line in lines]
    notebook['cells'][0]['source']=['# '+branch+' separate ablations — seed0\n',
        'Separate from the main run; unchanged horizons and certificates. Jobs queue across sessions. GPU timings unmeasured.\n']
    return notebook


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True);args=parser.parse_args()
    for branch in ('H1','H2','H3','H4'):
        path=ROOT/f'notebooks/standard_cl_{branch}_s0.ipynb';path.write_text(json.dumps(build(branch,args.commit),ensure_ascii=False,indent=1)+'\n');print(path)
        path=ROOT/f'notebooks/standard_cl_{branch}_ablations_s0.ipynb';path.write_text(json.dumps(build_ablations(branch,args.commit),ensure_ascii=False,indent=1)+'\n');print(path)
