"""Fresh-child CUDA setup ordering, without GPU training."""
import pytest
import torch
from experiments.vna_resume import reset_cuda_peak_memory_stats


@pytest.mark.parametrize('device',['cuda:0','cuda'])
def test_fresh_masked_lane_initializes_before_reset(monkeypatch,device):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    state=dict(initialized=False);calls=[];rng=torch.get_rng_state().clone()
    def initialize():state['initialized']=True;calls.append('init')
    def select(value):
        assert state['initialized'] and value==torch.device('cuda:0');calls.append('select')
    def reset(value):
        if not state['initialized']:raise RuntimeError('Invalid device argument')
        assert value==torch.device('cuda:0');calls.append('reset')
    monkeypatch.setattr(torch.cuda,'init',initialize)
    monkeypatch.setattr(torch.cuda,'current_device',lambda:0)
    monkeypatch.setattr(torch.cuda,'set_device',select)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',reset)
    reset_cuda_peak_memory_stats(device)
    assert calls==['init','select','reset'] and torch.equal(rng,torch.get_rng_state())


def test_cpu_startup_does_not_touch_cuda(monkeypatch):
    def unexpected(*args):raise AssertionError('CPU startup touched CUDA')
    for name in ('init','set_device','reset_peak_memory_stats'):monkeypatch.setattr(torch.cuda,name,unexpected)
    reset_cuda_peak_memory_stats('cpu')


def test_fresh_probe_failure_blocks_both_training_lanes_and_archives(tmp_path,monkeypatch):
    import json,os,subprocess,sys,threading,time
    from pathlib import Path
    from scripts.build_kaggle_standard_compare import RUNNER,jobs
    from experiments import vna_kaggle,vna_resume
    probes=[];training=[];archives=[]
    def probe(command,**kwargs):
        visible=kwargs['env']['CUDA_VISIBLE_DEVICES'];probes.append(visible)
        assert "reset_cuda_peak_memory_stats('cuda:0')" in command[2]
        if visible=='1':raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(subprocess,'run',probe)
    monkeypatch.setattr(vna_resume,'launch_logged',lambda *a,**kw:training.append(kw))
    monkeypatch.setattr(vna_kaggle,'archive_outputs',lambda *a,**kw:archives.append(True))
    monkeypatch.setattr(vna_kaggle,'restore_stage_archives',lambda *a,**kw:None)
    scope=dict(RUN_ROOT=tmp_path,OUTPUT=tmp_path/'standard_outputs',RUN_TAG='fixture',BRANCH='H1',
        REPO=Path(__file__).resolve().parents[1],REPO_COMMIT='fixture',JOBS=jobs('H1'),PYTHON=sys.executable,
        RESOLVED_DATA=str(tmp_path),PREFLIGHT_REPORT=tmp_path/'preflight.json',SESSION_START=time.time(),
        os=os,sys=sys,json=json,threading=threading,subprocess=subprocess,Path=Path)
    with pytest.raises(subprocess.CalledProcessError):exec(RUNNER,scope)
    assert probes==['0','1'] and training==[] and archives==[True]
