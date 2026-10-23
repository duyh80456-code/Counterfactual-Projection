"""Dual solver source identity, archival Gram oracle and T4 lane dispatch."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import inspect
import json
import subprocess
import sys
import threading
import pytest
import torch
from torch import nn
from methods.vna_v3 import V3Engine,diagnose
from methods.vna_sites import HookSite
from methods.vna_losses import MSE
from methods.vna_exact import gram
from experiments.vna_resume import rng_state,restore_rng
from experiments.main_cil_baseline import prefix,compare
from diagnostics.exact_solver import compare_profiles
from scripts.build_kaggle_vna_parallel import build_gram_m1_ir,GRAM_M1_IR_RUNNER
from tests.test_exact_gpu_lanes import write_report


@pytest.fixture(autouse=True)
def isolated():
    dtype,threads=torch.get_default_dtype(),torch.get_num_threads()
    with torch.random.fork_rng():
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
        yield
    torch.set_default_dtype(dtype);torch.set_num_threads(threads)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_fixed_gram_direction_matches_8c56d02_exactly(dtype):
    from methods.vna_v3 import training_products,seed_proposals
    from methods.vna_ggn import GGNProducts
    source=subprocess.check_output(['git','show','8c56d02:methods/vna_v3.py'],text=True)
    namespace={'__name__':'reference_gram'};exec(compile(source,'8c56d02/vna_v3.py','exec'),namespace)
    torch.manual_seed(29)
    model=nn.Sequential(nn.Linear(3,4),nn.ReLU(),nn.Linear(4,1)).to(dtype)
    site=HookSite(model,'0','2','linear',between=('1',))
    batch=torch.randn(6,3,dtype=dtype),torch.randn(6,1,dtype=dtype)
    results=[]
    with training_products(model,[site],seed_proposals([site]),batch[0]) as p:
        op=GGNProducts(p,batch[1],MSE(),dict(targets=batch[1]));G,_=gram(p)
        for cls in (namespace['V3Engine'],V3Engine):
            engine=cls('vna_gir',curvature='power10',**({'solver':'gram'} if cls is V3Engine else {}))
            engine.begin_task(2)
            results.append(engine._direction(p,op,.3,G=G))
    a,b=results
    for name in ('y','yO','z','zO','zN','dy','d_N','d_rep','uN'):
        torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    for name in ('V','V_O','beta','beta_branch','leak'):
        assert a[name]==b[name]
    assert 'GRAM_LIMIT' not in inspect.getsource(gram)


def test_solver_state_and_schedule_cannot_cross_backends():
    a=V3Engine(solver='gram')
    with pytest.raises(ValueError,match='solver'):V3Engine().load_state_dict(a.state_dict())
    with pytest.raises(ValueError,match='solver'):a.configure_schedule(dict(K0=1,solver='cg_ir',policy_version=a.policy_version))


def test_B_uses_prefix_while_dead_dormant_use_all_samples():
    model=nn.Sequential(nn.Linear(1,1),nn.ReLU(),nn.Linear(1,1)).double()
    with torch.no_grad():
        model[0].weight.fill_(1);model[0].bias.zero_();model[2].weight.fill_(1)
    site=HookSite(model,'0','2','linear',between=('1',))
    batch=torch.tensor([[-1.],[-2.],[1.],[2.]]),torch.zeros(4,1)
    full=diagnose(model,batch,sites=[site],spec=MSE(),mu=.3,b_probe_size=2)
    small=diagnose(model,tuple(t[:2] for t in batch),spec=MSE(),mu=.3)
    assert full['B']==small['B'] and full['b_probe_size']==2 and full['activity_probe_size']==4
    assert full['sites'][site.module_name]['dead']==0.


def test_baseline_requires_all_150_losses_with_exact_equality():
    rows=[dict(task=t,epoch=e,train_loss=.1*t+e) for t,n in ((1,100),(2,50)) for e in range(1,n+1)]
    ref=dict(epochs=rows,protocol=dict(code_commit='old'),summary_sha256='fixture')
    assert compare(ref,deepcopy(rows))['status']=='PASS'
    changed=deepcopy(rows);changed[-1]['train_loss']+=1e-12
    assert compare(ref,changed)['status']=='FAIL'
    with pytest.raises(ValueError):prefix(rows[:-1])
    with pytest.raises(ValueError):prefix(rows+[rows[0]])


def test_new_notebook_has_one_source_pin_and_two_solvers():
    nb=build_gram_m1_ir('a'*40)
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    source='\n'.join(''.join(c['source']) for c in nb['cells'])
    assert source.count("REPO_COMMIT = '"+'a'*40+"'")==1
    assert "'--solver','gram'" in source and "'cg_ir','exact',capacity=True" in source
    assert 'archive_stop.wait(900)' in source and 'MAIN RUN HELD' not in source
    assert 'CG-IR M1 remains unapproved' in source


@pytest.mark.parametrize('failure',[None,'compatibility','cheap'])
def test_lanes_overlap_and_only_Gram_full_is_authorized(tmp_path,failure):
    root=tmp_path/'outputs';root.mkdir();calls=[];archives=[];lock=threading.Lock()
    barrier=threading.Barrier(2);compat_done=threading.Event();paused=threading.Event()
    def launch(cmd,gpu,name,log):
        with lock:calls.append((gpu,name,cmd))
        if name in ('gram_preflight','gram_current'):barrier.wait(timeout=5)
        if name=='gram_preflight':
            p=Path(cmd[cmd.index('--schedule-output')+1]);p.parent.mkdir(parents=True);p.write_text('{}')
        elif '--profile' in cmd:
            if name=='cg_ir' and failure=='cheap':raise RuntimeError('fixture OOM')
            write_report(cmd)
            if name=='gram_current' and failure=='compatibility':
                path=Path(cmd[cmd.index('--out')+1])/'diagnostic.json'
                value=json.loads(path.read_text());value['fixed_gram_reference']['status']='FAIL';path.write_text(json.dumps(value))
            if name=='gram_current':compat_done.set()
        else:
            assert gpu==0 and compat_done.is_set()
            p=Path(cmd[cmd.index('--out')+1]);p.mkdir(parents=True,exist_ok=True)
            (p/'summary.json').write_text(json.dumps(dict(status='COMPLETED',tasks_run=20)))
        return True
    scope=dict(ROOT=root,sys=sys,DATA_ROOT=tmp_path/'cifar',REPO=tmp_path/'current',
        REFERENCE_REPO=tmp_path/'old',script=tmp_path/'diagnostics.py',checkpoint=tmp_path/'task1.pt',
        RUN_ROOT=tmp_path,STAGE_B=root,threading=threading,json=json,
        launch=launch,archive_outputs=lambda:archives.append(True),
        atomic_json_save=lambda v,p:Path(p).write_text(json.dumps(v)),compare_profiles=compare_profiles,
        prepare_reference=lambda *a:None,pause_due=paused.is_set,lane_state={},
        archive_stop=threading.Event(),archiver=SimpleNamespace(join=lambda **k:None),archive_errors=[],
        ThreadPoolExecutor=__import__('concurrent.futures').futures.ThreadPoolExecutor,
        as_completed=__import__('concurrent.futures').futures.as_completed,Path=Path)
    if failure=='compatibility':
        with pytest.raises(RuntimeError):exec(GRAM_M1_IR_RUNNER,scope)
        assert not any(name in ('gram_full_s0','cg_ir') for _,name,_ in calls)
    else:
        exec(GRAM_M1_IR_RUNNER,scope)
        assert [n for g,n,_ in calls if g==0]==['gram_preflight','gram_full_s0','gram_baseline_s0']
        assert [n for g,n,_ in calls if g==1]==['gram_current','cg_ir']
        cheap=next(c for g,n,c in calls if n=='cg_ir')
        assert '--capacity-profile' in cheap and '--arm' not in cheap
        result=json.loads((root/'profiles'/'cg_ir_vs_gram.json').read_text())
        assert result['status']==('FAIL' if failure=='cheap' else 'PASS')
    assert archives


@pytest.mark.parametrize('matches',[False,True])
def test_baseline_gate_only_reuses_after_complete_prefix(tmp_path,monkeypatch,matches):
    from experiments import main_cil as main
    rows=[dict(task=t,epoch=e,train_loss=float(e)) for t,n in ((1,100),(2,50)) for e in range(1,n+1)]
    reference=dict(epochs=deepcopy(rows),protocol=dict(code_commit='old'),summary_sha256='sha')
    if not matches:rows[0]['train_loss']+=1e-12
    monkeypatch.setattr(main,'measure_task',lambda *a,**kw:(dict(task=2),{}))
    hook=main.MainHook(manager=None,status=None,progress=SimpleNamespace(state=None,out=tmp_path))
    hook.baseline_reference=reference;hook.epochs=rows
    learner=SimpleNamespace(_cur_task=1)
    if matches:
        with pytest.raises(main.BaselineReused):hook.record_task(learner,None)
    else:hook.record_task(learner,None)
    value=json.loads((tmp_path/'baseline_loss_check.json').read_text())
    assert value['status']==('PASS' if matches else 'FAIL') and value['epochs_compared']==150


def test_strict_cross_curvature_still_checks_numeric_values(tmp_path):
    report=tmp_path/'diagnostic.json'
    write_report(['--curvature','power10','--out',str(tmp_path)])
    old=json.loads(report.read_text());new=deepcopy(old);new['curvature']='exact'
    candidate=new['candidate_rankings'][0]['candidates'][0]
    candidate.update(curvature_method='eigvalsh_downstream',power_generator_seed=None,power_iterations=None)
    assert compare_profiles(old,new,strict=True)['status']=='FAIL'
    assert compare_profiles(old,new,strict=True,allow_curvature_change=True)['status']=='PASS'
    candidate['curvature']*=1.01
    assert compare_profiles(old,new,strict=True,allow_curvature_change=True)['status']=='FAIL'


def test_fixed_reference_loads_historical_solver_and_detects_numeric_change(tmp_path,monkeypatch):
    from diagnostics.exact_solver import load_gram_reference,compare_fixed_gram
    from methods.vna_v3 import training_products,seed_proposals
    from methods.vna_ggn import GGNProducts
    from diagnostics.vna_align import _assert_bitwise
    reference=tmp_path/'reference';(reference/'methods').mkdir(parents=True)
    for name in ('vna_v3','vna_exact'):
        source=subprocess.check_output(['git','show','8c56d02:methods/'+name+'.py'],text=True)
        (reference/'methods'/(name+'.py')).write_text(source)
    monkeypatch.setattr('diagnostics.exact_solver.subprocess.check_output',lambda cmd,**kw:
        '8c56d02dab1ee16c47eb9138bf4aa3a64b9b45e2\n' if cmd[-1]=='HEAD' else '')
    cls=load_gram_reference(reference)
    assert 'q' in cls._output_solver.__globals__['OutputSolve'].__dict__
    model=nn.Sequential(nn.Linear(2,3),nn.ReLU(),nn.Linear(3,1)).double()
    site=HookSite(model,'0','2','linear',between=('1',))
    batch=torch.randn(5,2),torch.randn(5,1)
    engine=V3Engine(solver='gram',curvature='power10');engine.begin_task(2)
    with training_products(model,[site],seed_proposals([site]),batch[0]) as p:
        op=GGNProducts(p,batch[1],MSE(),dict(targets=batch[1]));G,_=gram(p)
        current=engine._direction(p,op,.3,G=G)
        before=deepcopy(model.state_dict()),rng_state(),engine.state_dict()
        report=compare_fixed_gram(cls,engine,p,op,.3,G,current)
        assert report['status']=='PASS' and 'excludes changed LM/GIR' in report['scope']
        _assert_bitwise(before,(model.state_dict(),rng_state(),engine.state_dict()))
        changed={**current,'dy':current['dy']+.01}
        assert compare_fixed_gram(cls,engine,p,op,.3,G,changed)['status']=='FAIL'
    monkeypatch.setattr('diagnostics.exact_solver.subprocess.check_output',lambda *a,**kw:'wrong')
    with pytest.raises(ValueError,match='clean 8c56d02'):load_gram_reference(reference)


@pytest.mark.parametrize('preset,order,filename',[('M2','superclass','icarl_BN.json'),('M3','random','icarl_noBN_random.json')])
def test_generic_authority_uses_current_policy_and_own_class_order(tmp_path,preset,order,filename):
    from scripts.build_kaggle_vna_parallel import RUNNER
    from methods.vna_v3 import POLICY_VERSION
    nodes=[n for n in ast.parse(RUNNER).body if isinstance(n,ast.FunctionDef) and n.name in ('schedule_name','prepare_schedule')]
    calls=[]
    def launch(command,*args):
        calls.append(command)
        path=Path(command[command.index('--schedule-output')+1]);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{}');return True
    scope=dict(ROOT=tmp_path,PRESET=preset,Path=Path,sys=sys,DATA_ROOT=tmp_path/'cifar',RUN_ROOT=tmp_path,RUN_TAG='fixture',
        REPO_COMMIT='a'*40,SOURCE_DIGEST=None,NOBN_LR_SCALE=.5,GRAD_CLIP=None,launch=launch,
        attached_file=lambda *a:None,archive_outputs=lambda:None,pause_due=lambda:False)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'authority','exec'),scope)
    scope['prepare_schedule']('bn' if preset=='M2' else 'nobn')
    cmd=calls[0]
    assert cmd[cmd.index('--class-order')+1]==order
    assert cmd[cmd.index('--solver')+1]=='cg_ir' and cmd[cmd.index('--curvature')+1]=='exact'
    assert Path(cmd[cmd.index('--schedule-output')+1]).name==filename
