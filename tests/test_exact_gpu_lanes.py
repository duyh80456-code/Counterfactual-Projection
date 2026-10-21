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
    ('profile',2,{0:['D-IR'],1:['R-Gram','D-IR-power10']}),
    ('nan',2,{0:['A','C'],1:['B','D']}),
    ('profile',1,{0:['D-IR','R-Gram','D-IR-power10']}),
    ('nan',1,{0:['A','B','C','D']}),
])
def test_gpu_dispatch_parallel_and_single_gpu_fallback(tmp_path,mode,gpus,expected):
    calls=defaultdict(list);active=defaultdict(int);peak=[0]
    archives=[];lock=threading.Lock();barrier=threading.Barrier(gpus)
    def launch(command,gpu,name,log):
        branch=name.split('_')[-1]
        assert command[command.index('--branch')+1]==('D' if mode=='profile' else branch)
        assert command[command.index('--checkpoint')+1]==str(tmp_path/'shared-task1.pt')
        assert command[command.index('--repo')+1]==str(tmp_path/('old' if branch in ('B','C') else 'reference' if branch=='R-Gram' else 'new'))
        assert ('--profile' in command)==(mode=='profile')
        assert command[command.index('--curvature')+1]==('power10' if branch in ('R-Gram','D-IR-power10','B','C') else 'exact')
        assert ('--capacity-profile' in command)==(mode=='profile' and branch=='D-IR')
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
    assert scope['ENV']['VNA_DIAGNOSTIC_LANES']==str(gpus)
    assert len(archives)==sum(map(len,expected.values()))+1


def test_paused_lane_does_not_launch_its_remaining_jobs(tmp_path):
    calls=[];archives=[]
    def launch(command,*args):
        calls.append(command[command.index('--branch')+1])
        return False
    run_schedule(tmp_path,'nan',1,launch,archives)
    assert calls==['A'] and len(archives)==1


def test_cg_failure_propagates_and_archives_outputs(tmp_path):
    archives=[];barrier=threading.Barrier(2)
    def launch(command,gpu,name,*args):
        if name in ('profile_D-IR','profile_R-Gram'): barrier.wait(timeout=5)
        if name=='profile_D-IR':
            raise subprocess.CalledProcessError(1,command)
        write_report(command)
        return True
    with pytest.raises(subprocess.CalledProcessError):
        run_schedule(tmp_path,'profile',2,launch,archives)
    assert archives


def test_gram_mismatch_does_not_gate_cg(tmp_path):
    calls=[];archives=[]
    def launch(command,gpu,name,*a):
        calls.append(name)
        write_report(command,mu=2. if name.endswith('R-Gram') else 1.)
        return True
    run_schedule(tmp_path,'profile',1,launch,archives)
    assert calls==['profile_D-IR','profile_R-Gram','profile_D-IR-power10']
    root=tmp_path/'outputs/exact_diagnostics/profile/fixture'
    report=json.loads((root/'ir_power10_vs_gram.json').read_text())
    assert report['status']=='FAIL' and report['advisory']
    assert json.loads((root/'D-IR/diagnostic.json').read_text())['status']=='PROFILE_COMPLETED'


@pytest.mark.parametrize('failed',['R-Gram'])
def test_gram_failure_keeps_cg(tmp_path,failed):
    calls=defaultdict(list);archives=[];barrier=threading.Barrier(2)
    def launch(command,gpu,name,*args):
        branch=name.removeprefix('profile_')
        calls[gpu].append(branch)
        if len(calls[gpu])==1: barrier.wait(timeout=5)
        if branch==failed: raise subprocess.CalledProcessError(1,command)
        write_report(command)
        return True
    run_schedule(tmp_path,'profile',2,launch,archives)
    assert dict(calls)=={0:['D-IR'],1:['R-Gram','D-IR-power10']}
    root=tmp_path/'outputs/exact_diagnostics/profile/fixture'
    report=json.loads((root/'ir_power10_vs_gram.json').read_text())
    assert report['status']=='INCOMPLETE' and report['execution'][failed]['status']=='FAILED'
    assert json.loads((root/'D-IR/diagnostic.json').read_text())['status']=='PROFILE_COMPLETED'


@pytest.mark.parametrize('gpus',[1,2])
def test_three_task_rehearsal_uses_fresh_schedule_and_full_epoch_recipe(tmp_path,gpus):
    from scripts.build_kaggle_vna_parallel import CG_SMOKE3
    calls=[];archives=[]
    def launch(command,gpu,name,*args):
        calls.append((gpu,name,command))
        if '--preflight-only' in command:
            Path(command[command.index('--schedule-output')+1]).write_text('{}')
        return True
    scope=dict(STAGE_B=tmp_path/'outputs',REPO_COMMIT='fixture',sys=sys,DATA_ROOT=tmp_path/'cifar',
        RUN_ROOT=tmp_path,torch=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:gpus)),
        launch=launch,archive_outputs=lambda:archives.append(True),pause_due=lambda:False)
    exec(CG_SMOKE3,scope)
    assert calls[0][1]=='smoke3_preflight'
    jobs=calls[1:]
    assert {name for _,name,_ in jobs}=={'smoke3_icarl','smoke3_icarl_full'}
    for gpu,name,command in jobs:
        assert command[command.index('--max-tasks')+1]=='3'
        assert command[command.index('--curvature')+1]=='exact'
        assert '--checkpoint' not in command and '--fake-data' not in command
        assert ('--schedule' in command)==(name=='smoke3_icarl_full')
        assert gpu==(1 if gpus==2 and name=='smoke3_icarl' else 0)
    assert len(archives)==3
