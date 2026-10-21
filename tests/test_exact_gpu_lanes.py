"""Exercise notebook GPU dispatch with subprocess stand-ins, without GPU training."""
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import pytest
from scripts.build_kaggle_vna_parallel import EXACT_DIAGNOSTICS


def run_schedule(tmp_path, mode, gpus, launch, archives):
    scope=dict(MODE=mode,torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:gpus)),
        ENV={},REPO_COMMIT='fixture',OLD_REPO=tmp_path/'old',REFERENCE_REPO=tmp_path/'reference',REPO=tmp_path/'new',STAGE_B=tmp_path/'outputs',
        RUN_ROOT=tmp_path,checkpoint=tmp_path/'shared-task1.pt',script=tmp_path/'diagnostic.py',
        DATA_ROOT=tmp_path/'cifar',sys=sys,launch=launch,archive_outputs=lambda:archives.append(True))
    source=EXACT_DIAGNOSTICS[EXACT_DIAGNOSTICS.index('from concurrent.futures import'):]
    exec(source,scope)
    return scope


@pytest.mark.parametrize('mode,gpus,expected',[
    ('profile',2,{0:['R'],1:['D']}),
    ('nan',2,{0:['A','C'],1:['B','D']}),
    ('profile',1,{0:['R','D']}),
    ('nan',1,{0:['A','B','C','D']}),
])
def test_gpu_dispatch_parallel_and_single_gpu_fallback(tmp_path,mode,gpus,expected):
    calls=defaultdict(list);active=defaultdict(int);peak=[0]
    archives=[];lock=threading.Lock();barrier=threading.Barrier(gpus)
    def launch(command,gpu,name,log):
        branch=name.split('_')[-1]
        assert command[command.index('--branch')+1]==('D' if branch=='R' else branch)
        assert command[command.index('--checkpoint')+1]==str(tmp_path/'shared-task1.pt')
        assert command[command.index('--repo')+1]==str(tmp_path/('old' if branch in ('B','C') else 'reference' if branch=='R' else 'new'))
        assert ('--profile' in command)==(mode=='profile')
        assert ('--capacity-profile' in command)==(mode=='profile' and branch=='D')
        with lock:
            calls[gpu].append(branch);active[gpu]+=1
            assert active[gpu]==1
            first=len(calls[gpu])==1
            peak[0]=max(peak[0],sum(active.values()))
        if first:barrier.wait(timeout=5)
        with lock:active[gpu]-=1
        return True
    scope=run_schedule(tmp_path,mode,gpus,launch,archives)
    assert dict(calls)==expected and peak[0]==gpus
    assert scope['ENV']['VNA_DIAGNOSTIC_LANES']==str(gpus)
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
        if name=='profile_D':
            raise subprocess.CalledProcessError(1,command)
        return True
    with pytest.raises(subprocess.CalledProcessError):
        run_schedule(tmp_path,'profile',2,launch,archives)
    assert archives
