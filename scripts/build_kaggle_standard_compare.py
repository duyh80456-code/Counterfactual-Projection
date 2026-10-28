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
    if branch=='H3':return ([job('cbp_s0','experiments.standard_imagenet',['--seed','0','--arm','cbp'])],
        [job('plasticity_vna_gir_s0','experiments.standard_imagenet',['--seed','0','--arm','vna_gir'])])
    if branch=='H4':
        return tuple([job(f'{"er" if arm=="baseline" else "domain_vna"}_{kind}_s0','experiments.standard_domain',['--seed','0','--arm',arm,
            '--config',f'configs/standard_cl/h4_{kind}_mnist.json']) for kind in ('rotated',)]
            for arm in ('baseline','h1'))
    raise ValueError(branch)


SETUP=r'''os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
PYTHON=sys.executable
if BRANCH=='H2':
    # Avalanche 0.4 predates Kaggle's Python 3.13; give the original stack its supported interpreter.
    subprocess.run([sys.executable,'-m','pip','install','-q','uv'],check=True)
    environment=RUN_ROOT/'lpr_env'
    if not (environment/'bin/python').exists():
        subprocess.run([sys.executable,'-m','uv','venv','--python','3.10',str(environment)],check=True)
    PYTHON=str(environment/'bin/python')
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PYTHON,'torch==2.2.2','torchvision==0.17.2',
        'numpy==1.26.4','torchmetrics==0.11.4','pytorchcv==0.0.67',
        'psutil','gputil','scikit-learn==1.6.1','matplotlib','wandb==0.15.12','protobuf==4.25.8',
        'tensorboard==2.19.0','tqdm','gdown==5.2.0','quadprog','dill==0.3.8','packaging','setuptools==75.8.0'],check=True)
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PYTHON,'-e',str(REPO)+'[vision,diagnostics]',
        '-r',str(REPO/'requirements-upstream.txt')],check=True)
    # Avalanche requires old Protocol constructor semantics; override torch's typing metadata, not its algorithm.
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PYTHON,'--no-deps',
        'avalanche-lib==0.4.0','typing-extensions==4.4.0'],check=True)
else:
    subprocess.run([PYTHON,'-m','pip','install','-q','-e',str(REPO)+'[vision,diagnostics]'],check=True)
    subprocess.run([PYTHON,'-m','pip','install','-q','-r',str(REPO/'requirements-upstream.txt')],check=True)
def fetch(name,url,pin):
    path=REPO/'third_party'/name
    if not path.exists():
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',url,str(path)],check=True)
        subprocess.run(['git','-C',str(path),'fetch','origin',pin],check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',pin],check=True)
    if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=pin:
        raise ValueError('Upstream pin mismatch: '+name)
if BRANCH=='H3':
    fetch('loss-of-plasticity','https://github.com/shibhansh/loss-of-plasticity.git','a6b79580d85f3025bdb601566d3627c5f489f13b')
else:
    fetch('PyCIL','https://github.com/LAMDA-CL/PyCIL.git','f3509b8ca3f20660ce4aa13f19d5283de81b4b35')
if BRANCH=='H2':fetch('LPR','https://github.com/plai-group/LPR.git','76c44f8f2481f4e23881b2463eb58aaf94affabb')
sys.path.insert(0,str(REPO))
subprocess.run([PYTHON,'-c',"import torch; assert torch.cuda.device_count()==2, 'Select GPU T4 x2'"],check=True)
def input_root():
    candidates=[]
    for directory,_,files in os.walk(DATA_DIR or '/kaggle/input',followlinks=True):
        names=set(files);path=Path(directory)
        if BRANCH in ('H1','H2') and {'train','test','meta'}<=names:candidates.append(path)
        if BRANCH=='H3' and all(str(i)+'.npy' in names for i in range(1000)):candidates.append(path)
        if BRANCH=='H4':
            needed={'train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte'}
            if all(name in names or name+'.gz' in names for name in needed):candidates.append(path)
    if len(candidates)!=1:raise ValueError(f'Set DATA_DIR to exactly one dataset root; found {candidates}')
    source=candidates[0];stage=RUN_ROOT/'input_data';stage.mkdir(exist_ok=True)
    if BRANCH=='H3':return str(source)
    if BRANCH in ('H1','H2'):
        target=stage/'cifar-100-python'
        if not target.exists():target.symlink_to(source,target_is_directory=True)
    else:
        import gzip
        raw=stage/'MNIST/raw';raw.mkdir(parents=True,exist_ok=True)
        for name in sorted(needed):
            target=raw/name
            if target.exists():continue
            if (source/name).exists():target.symlink_to(source/name)
            else:
                with gzip.open(source/(name+'.gz'),'rb') as a,target.with_suffix('.tmp').open('wb') as b:shutil.copyfileobj(a,b)
                target.with_suffix('.tmp').replace(target)
    return str(stage)
RESOLVED_DATA=input_root()
print('SEED=0; source=',REPO_COMMIT,'input=',RESOLVED_DATA,'full horizons; no smoke or pytest',flush=True)
'''

RUNNER=r'''from concurrent.futures import ThreadPoolExecutor
from experiments.vna_kaggle import restore_stage_archives,archive_outputs
from experiments.vna_resume import launch_logged,print_heartbeat
from experiments.io_utils import atomic_json_save
from scripts.collect_standard_cl import collect
ROOT_OUT=OUTPUT/RUN_TAG
if any(ROOT_OUT.rglob('latest.pt')) or any(ROOT_OUT.rglob('summary.json')):
    print('Keeping local working checkpoints; attached older output will not overwrite them',flush=True)
else:restore_stage_archives(Path('/kaggle/input'),OUTPUT,'standard_outputs',prefix=RUN_TAG)
ROOT_OUT.mkdir(parents=True,exist_ok=True)
VNA_DEADLINE=SESSION_START+11*3600
base_env={**os.environ,'PYTHONPATH':str(REPO),'PYTHONUNBUFFERED':'1','TQDM_DISABLE':'1',
    'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION':'python','OMP_NUM_THREADS':'1',
    'MKL_NUM_THREADS':'1','VNA_DEADLINE':str(VNA_DEADLINE)}
configs=RUN_ROOT/'effective_configs';configs.mkdir(exist_ok=True)
def command(job):
    args=list(job['args'])
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
    horizons='full registered configs; exit75 resumes, never truncates or changes seed')
atomic_json_save(manifest,ROOT_OUT/'notebook_protocol.json')
stop=threading.Event();paused=threading.Event();lanes={};archive_lock=threading.Lock();archive_errors=[]
def archive():
    with archive_lock:return archive_outputs(RUN_ROOT,folders=('standard_outputs',),exclude_parts=('data',))
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
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,index) for index in range(2)]
        for future in futures:future.result()
finally:
    stop.set()
    for thread in threads:thread.join(timeout=10)
    atomic_json_save(collect([ROOT_OUT]),ROOT_OUT/'collected.json')
    atomic_json_save(dict(paused=paused.is_set(),archive_errors=archive_errors),ROOT_OUT/'session.json')
    archive()
'''

REPORT=r'''import pandas as pd
rows=[]
for path in sorted(ROOT_OUT.rglob('summary.json')):
    summary=json.loads(path.read_text());done=summary['status']=='COMPLETED'
    row=dict(job=path.parent.name,status=summary['status'])
    for key in ('average_incremental_accuracy','final_accuracy','nme_average_incremental_accuracy','forgetting',
                'backward_transfer','last_50_task_error','plasticity_decay','gpu_hours','wall_seconds'):
        if key in summary:row[key]=summary[key] if done else None
    rows.append(row)
display(pd.DataFrame(rows))
print('Archive:',RUN_ROOT.with_suffix('.tar.gz'))
print('If PAUSED: attach this Output, keep source pin and datasets, rerun. No source ZIP needed.')
'''


def build(branch,commit):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('immutable full commit required')
    lanes=jobs(branch)
    first=f"BRANCH = {branch!r}\nREPO_COMMIT = {commit!r}\nRUN_TAG = 'standard_{branch.lower()}_s0_{commit[:12]}'\nDATA_DIR = None  # optional explicit root under /kaggle/input\nJOBS = {lanes!r}\nimport time\nSESSION_START = time.time()\n"
    bootstrap=BOOTSTRAP.replace('/kaggle/working/vna4_main','/kaggle/working/standard_cl').replace('OUTPUT = RUN_ROOT\n','OUTPUT = RUN_ROOT / "standard_outputs"\n')
    note=f'''# {branch}: official baseline vs full algorithm — seed 0

Paper references motivate baseline selection only. No comparison with paper accuracy numbers.
Pinned source commit; no ZIP overlay. Full registered horizon, final-epoch models, no automatic smoke/test/truncation.
H1: CIFAR-100, BN ResNet32 PyCIL, 20 tasks ×5 classes, memory2000, 200/170 epochs.
H2: official LPR online Split-CIFAR100, SlimResNet18, 20 experiences, minibatch10 ×3 repeats, class-balanced replay2000, validation5%, SGD0.1 without momentum/weight decay/scheduler.
H3: upstream Continual ImageNet32, 5000 binary tasks, 250 epochs/task, ConvNet, SGD0.01 momentum0.9, native head reset; CBP vs BP + VNA → GIR.
H4: Rotated MNIST, 20 fixed angles 0,15,...,285 degrees; fixed 10-class head, ER vs ER + Domain-Adaptive VNA.

Inputs: {'CBP ImageNet32 original class files 0.npy through 999.npy (700 NCHW images/class)' if branch=='H3' else 'CIFAR-100 Python train/test/meta' if branch in ('H1','H2') else 'MNIST raw IDX train/test images and labels (plain or gzip)' }.
Enable Internet and Kaggle Secret `github_token`. Select T4 ×2; baseline GPU0, full GPU1, one BLAS thread per lane.
H2 builds an isolated Python3.10 environment for the pinned original Avalanche0.4 stack.
Optional previous Output is only for exact resume. H3 has 15 million native optimizer updates per arm before interventions and requires many sessions; T4 timings are unmeasured. Other branches may also need multiple 11-hour sessions. The horizon is never shortened.
'''
    cells=[cell('markdown',note),cell('code',first),cell('code',bootstrap),cell('code',SETUP),cell('code',RUNNER),cell('code',REPORT)]
    for index,c in enumerate(cells):
        c['id']=f'{branch.lower()}-{index}'
        if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')
    return dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(accelerator='gpu',dataSources=[])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--commit',required=True);args=parser.parse_args()
    for branch in ('H1','H2','H3','H4'):
        path=ROOT/f'notebooks/standard_cl_{branch}_s0.ipynb';path.write_text(json.dumps(build(branch,args.commit),ensure_ascii=False,indent=1)+'\n');print(path)
