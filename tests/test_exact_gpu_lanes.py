"""Exercise notebook GPU dispatch with subprocess stand-ins, without GPU training."""
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
import threading
import json
from types import SimpleNamespace
import pytest
from scripts.build_kaggle_vna_parallel import EXACT_DIAGNOSTICS


def run_schedule(tmp_path, mode, gpus, launch, archives):
    scope=dict(MODE=mode,torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:gpus)),
        ENV={},REPO_COMMIT='fixture',OLD_REPO=tmp_path/'old',REFERENCE_REPO=tmp_path/'reference',REPO=tmp_path/'new',STAGE_B=tmp_path/'outputs',
        RUN_ROOT=tmp_path,checkpoint=tmp_path/'shared-task1.pt',script=tmp_path/'diagnostic.py',
        DATA_ROOT=tmp_path/'cifar',sys=sys,json=json,launch=launch,archive_outputs=lambda:archives.append(True))
    source=EXACT_DIAGNOSTICS[EXACT_DIAGNOSTICS.index('from concurrent.futures import'):]
    exec(source,scope)
    return scope


def write_report(command,mu=1.):
    report=dict(status='PROFILE_COMPLETED',application_seconds=10.,final_L_KD=[2.,1.],
        curvature=command[command.index('--curvature')+1],identity=dict(checkpoint_sha256='fixture'),
        applied_swaps=[dict(site='s',unit=0,candidate_id='a',A=dict(weight=[1.]))],
        candidate_rankings=[dict(site='s',candidates=[dict(id='a',score=1.,curvature=1.,power_generator_seed=12,power_iterations=10)])],
        applications=[dict(mu_star=mu,rho_tr=.5,L_before=3.,L_after=2.,KD_before=2.,KD_after=1.,
            swap_records=[dict(site='s',unit=0,Gamma=.2)],gir_trials=[dict(site='s',unit=0,Gamma_woodbury=.2)])])
    path=Path(command[command.index('--out')+1])/'diagnostic.json'
    path.write_text(json.dumps(report))


@pytest.mark.parametrize('mode,gpus,expected',[
    ('profile',2,{0:['R'],1:['D-power10','D-exact']}),
    ('nan',2,{0:['A','C'],1:['B','D']}),
    ('profile',1,{0:['R','D-power10','D-exact']}),
    ('nan',1,{0:['A','B','C','D']}),
])
def test_gpu_dispatch_parallel_and_single_gpu_fallback(tmp_path,mode,gpus,expected):
    calls=defaultdict(list);active=defaultdict(int);peak=[0]
    archives=[];lock=threading.Lock();barrier=threading.Barrier(gpus)
    def launch(command,gpu,name,log):
        branch=name.split('_')[-1]
        assert command[command.index('--branch')+1]==('D' if mode=='profile' else branch)
        assert command[command.index('--checkpoint')+1]==str(tmp_path/'shared-task1.pt')
        assert command[command.index('--repo')+1]==str(tmp_path/('old' if branch in ('B','C') else 'reference' if branch=='R' else 'new'))
        assert ('--profile' in command)==(mode=='profile')
        assert command[command.index('--curvature')+1]==('power10' if branch in ('R','D-power10','B','C') else 'exact')
        assert ('--capacity-profile' in command)==(mode=='profile' and branch=='D-exact')
        with lock:
            calls[gpu].append(branch);active[gpu]+=1
            assert active[gpu]==1
            first=len(calls[gpu])==1
            peak[0]=max(peak[0],sum(active.values()))
        if first:barrier.wait(timeout=5)
        if mode=='profile':write_report(command)
        with lock:active[gpu]-=1
        return True
    scope=run_schedule(tmp_path,mode,gpus,launch,archives)
    assert dict(calls)==expected and peak[0]==gpus
    assert scope['ENV']['VNA_DIAGNOSTIC_LANES']==str(1 if mode=='profile' else gpus)
    assert len(archives)==sum(map(len,expected.values()))+1


def test_paused_lane_does_not_launch_its_remaining_jobs(tmp_path):
    calls=[];archives=[]
    def launch(command,*args):
        calls.append(command[command.index('--branch')+1])
        return False
    run_schedule(tmp_path,'nan',1,launch,archives)
    assert calls==['A'] and len(archives)==1


def test_child_failure_propagates_and_archives_outputs(tmp_path):
    archives=[];barrier=threading.Barrier(2)
    def launch(command,gpu,name,*args):
        barrier.wait(timeout=5)
        if name=='profile_D-power10':
            raise subprocess.CalledProcessError(1,command)
        return True
    with pytest.raises(subprocess.CalledProcessError):
        run_schedule(tmp_path,'profile',2,launch,archives)
    assert archives


def test_power10_mismatch_holds_exact_stage(tmp_path):
    calls=[];archives=[]
    def launch(command,gpu,name,*a):
        calls.append(name)
        write_report(command,mu=2. if name.endswith('D-power10') else 1.)
        return True
    with pytest.raises(RuntimeError,match='D-power10 differs'):
        run_schedule(tmp_path,'profile',1,launch,archives)
    assert calls==['profile_R','profile_D-power10']
    path=tmp_path/'outputs/exact_diagnostics/profile/fixture/stage1_power10_vs_b97d25c.json'
    assert json.loads(path.read_text())['status']=='FAIL'
