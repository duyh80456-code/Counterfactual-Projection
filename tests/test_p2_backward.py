"""Compare default P1 numerics with an independent 3584b04 worktree process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from diagnostics.vna_align import _assert_bitwise


ROOT=Path(__file__).resolve().parents[1]
BASELINE='3584b04'
DRIVER=r'''
import json,random,sys,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
repo,out=map(Path,sys.argv[1:])
sys.path.insert(0,str(repo))
from experiments import main_cil as main
from experiments.vna_resume import rng_state
from methods.vna_v3 import V3Engine
from methods.vna_sites import HookSite
from methods.vna_losses import MSE
torch.set_num_threads(1)
time.perf_counter=lambda:0.
time.monotonic=lambda:0.
parser=main.parser()
cli=['--arm','icarl','--fake-data','--device','cpu','--workers','0','--out',str(out/'native')]
if '--gir-rule' in parser._option_string_actions:cli+=['--gir-rule','schur_dead','--warmup-epochs','0']
summary=main.run(parser.parse_args(cli))
checkpoint=torch.load(out/'native/latest.pt',map_location='cpu',weights_only=False)
native=dict(model=checkpoint['learner']['_network'].state_dict(),optimizer=checkpoint['optimizer'],
            rng=checkpoint['rng'],epochs=json.loads((out/'native/epochs.json').read_text()),
            tasks=summary['tasks'],diagnostics=json.loads((out/'native/diagnostics.json').read_text()),
            protocol=summary['identity'])
random.seed(312);np.random.seed(312);torch.manual_seed(312)
model=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.ReLU(),torch.nn.Linear(3,1)).double()
with torch.no_grad():model[0].weight[0]=0;model[0].bias[0]=-1;model[2].weight[:,0]=0
batch=torch.randn(4,2,dtype=torch.float64),torch.randn(4,1,dtype=torch.float64)
optimizer=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=.001)
losses=[]
for _ in range(3):
    optimizer.zero_grad();loss=torch.nn.functional.mse_loss(model(batch[0]),batch[1])
    losses.append(loss.detach().clone());loss.backward();optimizer.step()
engine=V3Engine('vna_gir',solver='gram',curvature='exact');engine.begin_task(2)
site=HookSite(model,'0','2','linear',between=('1',))
record=engine.apply(model,[site],batch,MSE(),dict(targets=batch[1]),optimizer,3,force=True)
assert record is not None and engine.applications==1
def scientific(value):
    if isinstance(value,dict):return {k:scientific(v) for k,v in value.items()
        if not isinstance(k,str) or k not in ('code_commit','source_digest') and 'seconds' not in k}
    if isinstance(value,list):return [scientific(v) for v in value]
    if isinstance(value,tuple):return tuple(scientific(v) for v in value)
    return value
torch.save(dict(native=scientific(native),engine=scientific(dict(model=model.state_dict(),
    optimizer=optimizer.state_dict(),losses=losses,log=record,state=engine.state_dict(),rng=rng_state()))),out/'snapshot.pt')
'''


def historical_worktree():
    expected=subprocess.check_output(['git','rev-parse',BASELINE],cwd=ROOT,text=True).strip()
    listing=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True)
    for record in listing.strip().split('\n\n'):
        fields=dict(line.split(' ',1) for line in record.splitlines() if ' ' in line)
        if fields.get('HEAD')!=expected:continue
        path=Path(fields['worktree'])
        subprocess.run(['git','diff','--exit-code',expected,'--'],cwd=path,check=True)
        for dependency in (ROOT/'third_party').iterdir():
            target=path/'third_party'/dependency.name
            if dependency.is_dir() and not target.exists():target.symlink_to(dependency.resolve(),target_is_directory=True)
        return path
    raise AssertionError('Create the baseline with git worktree add --detach /tmp/vna-p2-before-3584b04 3584b04')


def test_old_default_native_training_and_gir_match_3584b04_bitwise(tmp_path):
    baseline=historical_worktree()
    env={**os.environ,'REQUIRE_UPSTREAM':'1','PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION':'python',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','VNA_SOURCE_DIGEST':''}
    for key in ('VNA_DEADLINE','PYTHONPATH'):env.pop(key,None)
    snapshots=[]
    for name,repo in (('before',baseline),('after',ROOT)):
        output=tmp_path/name;output.mkdir()
        result=subprocess.run([sys.executable,'-c',DRIVER,str(repo),str(output)],cwd=repo,env=env,
                              capture_output=True,text=True,timeout=600)
        (output/'process.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
        snapshots.append(torch.load(output/'snapshot.pt',map_location='cpu',weights_only=False))
    _assert_bitwise(*snapshots)
