"""Build the T4x2 P1 fixed-checkpoint kernel notebook, seeds 0/1 only."""
import argparse
import json
import subprocess
from pathlib import Path
from scripts.build_kaggle_vna_parallel import build as base
from scripts.build_kaggle_pilot_common import cell

ROOT=Path(__file__).resolve().parents[1]
RUNNER=r'''from diagnostics.kernel_ablation import discover_inputs
from experiments.runners.pycil import upstream_imports
upstream_imports()
ROOT=STAGE_B/REPO_COMMIT[:12]/RUN_TAG
ROOT.mkdir(parents=True,exist_ok=True)
ENV.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
RESOLVED={seed:discover_inputs((Path('/kaggle/input'),STAGE_B),seed,INPUTS[seed]) for seed in (0,1)}
print('INPUTS',json.dumps(RESOLVED,indent=2),flush=True)

def command(seed,arm,mode,out):
    cmd=[sys.executable,'-u','-m','diagnostics.kernel_ablation','--seed',str(seed),'--arm',arm,
        '--mode',mode,'--checkpoint',RESOLVED[seed]['checkpoint'],'--original-run',RESOLVED[seed]['original_run'],
        '--anchor-dir',str(ROOT/'anchors'),'--data-dir',str(DATA_ROOT),'--out',str(out),'--device','cuda:0']
    if RESOLVED[seed].get('baseline_checkpoint'):cmd+=['--baseline-checkpoint',RESOLVED[seed]['baseline_checkpoint']]
    return cmd

def lane(gpu,seed):
    if seed==1:
        if pause_due():return
        folder=ROOT/'s1_replay'
        if not launch(command(1,'vna','replay',folder),gpu,'s1_replay',RUN_ROOT/'s1_replay.log'):return
        audit=json.loads((folder/'replay.json').read_text())
        print('S1 REPLAY AUDIT',json.dumps(audit),flush=True)
        print('S1 conclusions gated:',not audit.get('conclusion_allowed',False),flush=True)
        archive_outputs()
    for arm in ('vna','kernel_self','vna_rand'):
        if pause_due():return
        name=f'{arm}__s{seed}';folder=ROOT/name
        if not launch(command(seed,arm,'ablation',folder),gpu,name,RUN_ROOT/(name+'.log')):return
        archive_outputs()

try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in (pool.submit(lane,0,0),pool.submit(lane,1,1)):future.result()
finally:
    archive_stop.set();archiver.join(timeout=10);archive_outputs()
'''

REPORT=r'''for seed in (0,1):
    if seed==1:
        path=ROOT/'s1_replay'/'replay.json'
        print('S1 AUDIT:',path.read_text() if path.exists() else 'MISSING: no conclusions from s1')
    for arm in ('vna','kernel_self','vna_rand'):
        folder=ROOT/f'{arm}__s{seed}'
        print('\nSEED',seed,'KERNEL',arm)
        if (folder/'chain.json').exists():
            for row in json.loads((folder/'chain.json').read_text()):
                print(json.dumps({k:v for k,v in row.items() if k!='event'}))
        path=folder/'tasks.json'
        if path.exists():
            for task in json.loads(path.read_text()):
                print({k:v for k,v in task.items() if k=='task' or k.endswith(('_old_top1','_new_top1','_average_accuracy'))})
        path=folder/'summary.json'
        print('SUMMARY',path.read_text() if path.exists() else 'PAUSED_OR_PENDING')
archive_outputs()
if archive_errors:raise RuntimeError(archive_errors)
'''


def build(commit):
    source=subprocess.check_output(['git','show',commit+':diagnostics/kernel_ablation.py'],cwd=ROOT,text=True)
    if 'class KernelEngine' not in source:raise ValueError('source pin lacks the kernel ablation')
    nb=base(commit,'M1','p1_kernel_ablation_s01',None)
    nb['cells'][0]=cell('code',f'''REPO_COMMIT = {commit!r}
SOURCE_DIGEST = None
RUN_TAG = 'p1_kernel_ablation_s01'
JOBS_GPU0 = ['vna__s0','kernel_self__s0','vna_rand__s0']
JOBS_GPU1 = ['s1_replay','vna__s1','kernel_self__s1','vna_rand__s1']
JOBS_CPU = []
INPUTS = {{0: {{}}, 1: {{}}}}  # optional checkpoint, original_run, baseline_checkpoint paths
import time
SESSION_HOURS = 11.5
SESSION_START = time.time()
VNA_DEADLINE = SESSION_START + SESSION_HOURS*3600
ALLOW_OVER_BUDGET = True
DISK_CAP_BYTES = 20_000_000_000
''')
    nb['cells'][1]=cell('markdown','''# P1 fixed-checkpoint kernel ablation: s0/s1
Enable T4x2, Internet and github_token. Attach CIFAR-100 and original P1 noBN Inc5
outputs with baseline end-task1/native-task2-epoch0 checkpoints and original full
protocol.json, epochs.json and interventions.json. The notebook locates compatible
expanded files or restored output archives; use INPUTS paths to resolve ambiguity.
No baseline training, task-1 GIR amendments or new preflight. Preserve the original
6be6543 certificate/Gram trust region; actual update norms are observations only.
Radius violations block interpretation. Native K is an untruncated control, not
a rank-matched arm; r=16 counts incoming directions per site for virtual kernels.
Random-A has one seeded trajectory per seed, with no statistical-superiority claim.
Each arm attempts all 8 copied VNA-only steps, then resumes training from step 1
through task 5 with VNA–GIR and the same projection kernel. Independent losses use
a fixed balanced official-test batch; they never select/accept steps. GPU1 audits
the original s1 replay through task2 epoch43 before computing its ablations.
Missing flags or mismatches block conclusions from s1. Seeds 2–7 are held out.
The currently supplied p1_ablation_bundle.zip has no checkpoints and lacks primary
s1 protocol/epoch records; add those before launching. See KERNEL_ABLATION_INPUT_AUDIT.md.
Runtime is unmeasured; multiple sessions may be necessary. Resume with this Output,
the same source pin and original inputs. No source ZIP is required.
''')
    source=''.join(nb['cells'][2]['source']).replace('/kaggle/working/vna4_main','/kaggle/working/p1_kernel_ablation')
    nb['cells'][2]=cell('code',source+"\nos.environ.pop('VNA_SOURCE_DIGEST',None)\n")
    nb['cells'][-2]=cell('code',RUNNER);nb['cells'][-1]=cell('code',REPORT)
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source']),'<notebook>','exec')
    return nb


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--commit',required=True)
    args=p.parse_args();path=ROOT/'notebooks/vna5_kernel_ablation_s01.ipynb'
    path.write_text(json.dumps(build(args.commit),ensure_ascii=False,indent=1)+'\n');print(path)
