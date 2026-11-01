"""Build a CPU-only, source-pinned Phase0 notebook; no VNA runtime imports."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.build_kaggle_pilot_common import cell

SETUP=r'''import os,sys,json,subprocess,hashlib,time,threading,tarfile,shutil,gzip
from pathlib import Path
SESSION_START=time.monotonic()
RUN_ROOT=Path('/kaggle/working/ett_phase0')
REPO=RUN_ROOT/'source'/REPO_COMMIT[:12]
ROOT_OUT=RUN_ROOT/'outputs'/RUN_TAG
RUN_ROOT.mkdir(parents=True,exist_ok=True)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    PYTHONUNBUFFERED='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(RUN_ROOT/'mpl_cache'))

def checkout(url,path,pin,private=False):
    if path.exists():
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if actual!=pin:raise ValueError('Existing checkout differs from pin: '+str(path))
        if subprocess.run(['git','-C',str(path),'diff','--quiet','HEAD']).returncode:
            raise ValueError('Tracked source changed after checkout: '+str(path))
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'GIT_TERMINAL_PROMPT':'0'};askpass=RUN_ROOT/'.git_askpass.py'
    if private:
        try:
            from kaggle_secrets import UserSecretsClient
            token=UserSecretsClient().get_secret('github_token').strip()
        except Exception:token=''
        if token:
            askpass.write_text("#!/usr/bin/env python3\nimport os,sys\nprint('x-access-token' if 'Username' in sys.argv[1] else os.environ['ETT_GIT_TOKEN'])\n")
            askpass.chmod(0o700);env.update(GIT_ASKPASS=str(askpass),ETT_GIT_TOKEN=token)
    try:
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',url,str(path)],env=env,check=True)
        subprocess.run(['git','-C',str(path),'fetch','origin',pin],env=env,check=True)
        subprocess.run(['git','-C',str(path),'checkout','--detach',pin],env=env,check=True)
    finally:askpass.unlink(missing_ok=True)

checkout('https://github.com/duyh80456-code/Counterfactual-Projection.git',REPO,REPO_COMMIT,True)
checkout('https://github.com/shibhansh/loss-of-plasticity.git',REPO/'third_party/loss-of-plasticity',LOP_COMMIT)
import importlib.util
if importlib.util.find_spec('uv') is None:
    subprocess.run([sys.executable,'-m','pip','install','-q','uv'],check=True)
ENV_DIR=RUN_ROOT/'python310';PYTHON=str(ENV_DIR/'bin/python')
if not Path(PYTHON).exists():
    subprocess.run([sys.executable,'-m','uv','venv','--python','3.10',str(ENV_DIR)],check=True)
ready=ENV_DIR/'ett_packages.json'
packages={'torch':'2.2.2+cpu','torchvision':'0.17.2+cpu','numpy':'1.26.4','matplotlib':'3.8.4'}
if not ready.exists() or json.loads(ready.read_text())!=packages:
    # uv prioritizes extra-index-url; CPU wheels must be searched before PyPI.
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PYTHON,
        '--index-url','https://pypi.org/simple','--extra-index-url','https://download.pytorch.org/whl/cpu',
        'torch==2.2.2+cpu','torchvision==0.17.2+cpu'],check=True)
    subprocess.run([sys.executable,'-m','uv','pip','install','--python',PYTHON,
        'numpy==1.26.4','matplotlib==3.8.4'],check=True)
    ready.write_text(json.dumps(packages))
ENV={**os.environ,'PYTHONPATH':str(REPO)}
metadata="import torch,torchvision,numpy,matplotlib,sys,json; print(json.dumps(dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,numpy=numpy.__version__,matplotlib=matplotlib.__version__,cpu_capability=torch.backends.cpu.get_cpu_capability())))"
RUNTIME=json.loads(subprocess.check_output([PYTHON,'-c',metadata],env=ENV,text=True))
print('CPU runtime:',json.dumps(RUNTIME),flush=True)
print('Pinned source:',REPO_COMMIT,'LoP:',LOP_COMMIT,flush=True)
'''

RESTORE=r'''# Restore only this source/run tag. Distinct attached snapshots require an explicit choice.
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_manifest(data):
    manifest=json.loads(data)
    if manifest.get('source_commit')!=REPO_COMMIT or manifest.get('run_tag')!=RUN_TAG:
        raise ValueError('Resume source/run tag mismatch')
    if manifest.get('runtime')!=RUNTIME:raise ValueError('Resume runtime differs; keep the original environment')

if any(ROOT_OUT.rglob('latest.pt')):
    verify_manifest((ROOT_OUT/'notebook_protocol.json').read_bytes())
    print('Keeping local working checkpoint',flush=True)
else:
    source=Path(RESUME_FROM) if RESUME_FROM else None
    if source is None:
        candidates=list(Path('/kaggle/input').rglob(RUN_TAG+'.tar.gz'))
        unique={digest(p):p for p in candidates}
        if len(unique)>1:raise ValueError('Several resume archives: set RESUME_FROM to the desired snapshot')
        if unique:source=next(iter(unique.values()))
        else:
            folders=[]
            for p in Path('/kaggle/input').rglob('notebook_protocol.json'):
                try:info=json.loads(p.read_text())
                except (ValueError,OSError):continue
                if info.get('run_tag')==RUN_TAG and info.get('source_commit')==REPO_COMMIT:folders.append(p.parent)
            if len(folders)>1:raise ValueError('Several extracted Outputs: set RESUME_FROM explicitly')
            if folders:source=folders[0]
    if source is not None:
        if source.is_dir():
            verify_manifest((source/'notebook_protocol.json').read_bytes())
            shutil.copytree(source,ROOT_OUT,dirs_exist_ok=True)
        else:
            prefix='outputs/'+RUN_TAG+'/'
            with tarfile.open(source,'r:gz') as archive:
                verify_manifest(archive.extractfile(prefix+'notebook_protocol.json').read())
                for item in archive.getmembers():
                    if not item.isfile():
                        if not item.isdir():raise ValueError('Resume archive cannot contain links/devices')
                        continue
                    if not item.name.startswith(prefix):raise ValueError('Unexpected archive scope: '+item.name)
                    relative=Path(item.name[len(prefix):])
                    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe archive path')
                    target=ROOT_OUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    with archive.extractfile(item) as src,target.open('wb') as dest:shutil.copyfileobj(src,dest)
        print('Restored:',source,flush=True)
ROOT_OUT.mkdir(parents=True,exist_ok=True)
'''

DATA=r'''# Optional original MNIST raw IDX input; otherwise the real runner downloads it.
DATA_ROOT=RUN_ROOT/'input_data';DATA_ROOT.mkdir(exist_ok=True)
search=Path(DATA_DIR) if DATA_DIR else Path('/kaggle/input')
names=('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')
if not all((DATA_ROOT/'MNIST/raw'/name).exists() for name in names):
    locations=[]
    if search.exists():
        locations=sorted({p.parent for pattern in (names[0],names[0]+'.gz') for p in search.rglob(pattern)},key=str)
    valid=[p for p in locations if all((p/n).is_file() or (p/(n+'.gz')).is_file() for n in names)]
    if valid:
        raw=DATA_ROOT/'MNIST/raw';raw.mkdir(parents=True,exist_ok=True)
        for name in names:
            target=raw/name
            if target.exists():continue
            plain=valid[0]/name
            if plain.exists():target.symlink_to(plain.resolve())
            else:
                with gzip.open(valid[0]/(name+'.gz'),'rb') as src,target.open('wb') as dest:shutil.copyfileobj(src,dest)
        print('MNIST input:',valid[0],flush=True)
    else:print('No attached MNIST raw files; torchvision will download the official data',flush=True)
configs=ROOT_OUT/'configs';configs.mkdir(exist_ok=True)
for setting in ('scr','pmnist'):
    config=json.loads((REPO/f'ett/configs/phase0_{setting}.yaml').read_text())
    # Fresh BP is the portable default: no local81a7d04 Output is required.
    config.pop('reuse_native_from',None);config.pop('reuse_source',None)
    target=configs/f'phase0_{setting}.yaml';text=json.dumps(config,indent=2)+'\n'
    if target.exists() and json.loads(target.read_text())!=config:raise ValueError('Effective config changed on resume')
    target.write_text(text)
manifest=dict(source_commit=REPO_COMMIT,upstream_commit=LOP_COMMIT,run_tag=RUN_TAG,runtime=RUNTIME,
    seed=0,device='cpu',settings=['scr','pmnist'],branches=['native','ett','cbp','gradmax_reset'],
    native_reuse='disabled: fresh BP; no previous Output needed',
    anchor_schedule='20 fresh evenly-spaced anchors, shifted for current-task reserves',
    runtime_overrides=['remove local reuse_native_from/reuse_source only'],
    protocol_note='4 paired branches run serially to isolate global RNG; settings run in separate CPU processes; stop after Phase0')
manifest_path=ROOT_OUT/'notebook_protocol.json'
if manifest_path.exists():verify_manifest(manifest_path.read_bytes())
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
'''

RUN=r'''from concurrent.futures import ThreadPoolExecutor
ARCHIVE=RUN_ROOT/(RUN_TAG+'.tar.gz')
stop=threading.Event();lock=threading.Lock();state={};errors=[]
DEADLINE=SESSION_START+11*3600

def archive():
    with lock:
        temporary=ARCHIVE.with_suffix('.tmp')
        with tarfile.open(temporary,'w:gz') as bundle:
            def include(item):return None if item.name.endswith('.tmp') or item.issym() or item.islnk() else item
            bundle.add(ROOT_OUT,arcname='outputs/'+RUN_TAG,filter=include)
        temporary.replace(ARCHIVE)
    return ARCHIVE

def periodic():
    last=0.
    while not stop.wait(30):
        statuses={}
        for setting in ('scr','pmnist'):
            p=ROOT_OUT/setting/'status.json'
            if p.exists():statuses[setting]=json.loads(p.read_text())
        print('HEARTBEAT',int(time.monotonic()-SESSION_START),'s',json.dumps(statuses),flush=True)
        if time.monotonic()-last>=300:
            try:archive();last=time.monotonic()
            except Exception as exc:errors.append(repr(exc));print('ARCHIVE ERROR',repr(exc),flush=True)

def launch(setting):
    remaining=DEADLINE-time.monotonic()
    if remaining<=0:return 75
    command=[PYTHON,'-m','ett.phase0.measure_ge','--config',str(configs/f'phase0_{setting}.yaml'),
        '--out',str(ROOT_OUT/setting),'--data-dir',str(DATA_ROOT),'--deadline-seconds',str(remaining)]
    if setting=='pmnist':command.append('--download')
    print('START',setting,command,flush=True)
    logfile=ROOT_OUT/(setting+'.log')
    with logfile.open('a') as log:
        process=subprocess.Popen(command,cwd=REPO,env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for line in process.stdout:
                log.write(line);log.flush();print('['+setting+']',line,end='',flush=True)
            code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
    state[setting]=code
    if code not in (0,75):errors.append(setting+' exited '+str(code))
    archive();return code

worker=threading.Thread(target=periodic,daemon=True);worker.start()
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(launch,setting) for setting in ('scr','pmnist')]
        for future in futures:future.result()
    if all(state.get(s)==0 for s in ('scr','pmnist')):
        command=[PYTHON,'-m','ett.phase0.report',str(ROOT_OUT/'scr'),str(ROOT_OUT/'pmnist'),'--out',str(ROOT_OUT/'report')]
        subprocess.run(command,cwd=REPO,env=ENV,check=True)
    else:print('Not complete: keep same source/run tag and resume; no gate conclusion',flush=True)
finally:
    stop.set();worker.join(timeout=10)
    (ROOT_OUT/'session.json').write_text(json.dumps(dict(exit_codes=state,errors=errors,stop_after='Phase0'),indent=2))
    archive()
print('Output archive:',ARCHIVE,flush=True)
if errors:raise RuntimeError('Session errors: '+repr(errors))
'''

REPORT=r'''from IPython.display import display,FileLink
p=ROOT_OUT/'report/phase0_report.json'
if p.exists():
    report=json.loads(p.read_text())
    columns=('setting','h','checkpoints','net_median','net_iqr','sign_p','cbp_minus_compact_median','net_post_median','gate')
    print(' | '.join(columns))
    for row in report['tables']:print(' | '.join(str(row.get(k)) for k in columns))
    print('GATE:', 'PASS' if report['gate_pass'] else 'NOT PASSED')
    print('STOP after Phase0. Phase1 is never launched by this notebook.')
    display(FileLink(str(ROOT_OUT/'report/phase0_results.zip')))
display(FileLink(str(ARCHIVE)))
print('For resume: attach this notebook Output; keep REPO_COMMIT and RUN_TAG unchanged.')
'''


def build(commit):
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('full immutable commit required')
    note='''# ETT Phase 0 — SCR + PM-small, seed0, CPU only

Select **CPU / accelerator None**, enable Internet. No GPU, source ZIP, Gram file,
old baseline Output or local81a7d04 checkpoint is needed. Optional MNIST raw IDX
input is scanned recursively; otherwise torchvision downloads official MNIST.
A private source repo needs Kaggle Secret `github_token` (public access needs none).

Full fresh BP trajectories: SCR1,000,000 updates, PM20 permutations ×60,000,
MLP3×10.20 anchors per setting; h1/10/100; buffer1000 trained samples,800/200;
primary1000 future samples plus100 post steps and a disjoint1000-sample post set.
Four isolated paired branches: native, ETT, upstream-GnT reset, GradMax-pool reset.
Branches run serially to preserve global RNG isolation; SCR/PM run concurrently
in two CPU processes. This does not change the paired scientific comparison.

This is the registered PM-small/SCR diagnostic, **not** a Nature-paper BP/CBP
score reproduction. The previous local Phase0 did not pass its gate; this notebook
is a fresh rerun with evenly spaced anchors and no native-checkpoint reuse.
No Phase1 baseline audit or ReDo/SNR/SWR port is performed here.

No pytest, fake data or smoke. Source/LoP pins and CPU package versions are checked.
Fresh CPU timings on Kaggle are unmeasured; prior local timings12/23 minutes used
native reuse. Full fresh runs can take longer. No horizon/anchor truncation.
512-MiB storage gate,11-hour deadline/exit75, periodic checkpoint/output archive.
A hard stop replays only work since the last saved native checkpoint/case.
Attach this notebook's Output to resume with the same pin/tag; RESUME_FROM can
select an archive or extracted run folder when several snapshots are attached.
'''
    first=f"REPO_COMMIT = {commit!r}\nLOP_COMMIT = 'a6b79580d85f3025bdb601566d3627c5f489f13b'\nRUN_TAG = 'ett_phase0_s0_{commit[:12]}'\nDATA_DIR = None  # optional MNIST raw IDX input tree; auto-download when absent\nRESUME_FROM = None  # optional exact archive or extracted run folder\n"
    cells=[cell('markdown',note),cell('code',first)]
    for source in (SETUP,RESTORE,DATA,RUN,REPORT):cells.append(cell('code',source))
    for index,c in enumerate(cells):
        c['id']=f'ett-phase0-{index}'
        if c['cell_type']=='code':compile(''.join(c['source']),'<ett_notebook>','exec')
    return dict(cells=cells,nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),kaggle=dict(accelerator='none',isInternetEnabled=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
    target=ROOT/'notebooks/ett_phase0_scr_pmnist_s0.ipynb';target.write_text(json.dumps(build(a.commit),ensure_ascii=False,indent=1)+'\n');print(target)
