"""Exact prefix evidence and the closed-grid clipping decision gate."""
import copy
import json
import math
from pathlib import Path
import pytest
from diagnostics.grad_clip_study import compare_epochs,mathematical_identity
from experiments.clip_study import A_CASES,GRID,C_CASES,choose_tau,key,load_selection

COMMIT='a'*40


def epochs(tasks=20,last_epoch=None):
    return [dict(task=task,epoch=epoch,train_loss=1./(task+epoch))
            for task in range(1,tasks+1)
            for epoch in range(1,(last_epoch if task==tasks and last_epoch is not None else 100 if task==1 else 50)+1)]


@pytest.mark.parametrize('tasks,last,count',[(1,None,100),(2,None,150),(4,None,250),(3,7,157),(20,None,1050)])
def test_exact_epoch_gate_complete_and_interrupted_prefix(tasks,last,count):
    rows=epochs(tasks,last)
    report=compare_epochs(rows,copy.deepcopy(rows),tasks,last)
    assert report['status']=='PASS' and report['epochs_compared']==count
    changed=copy.deepcopy(rows);changed[-1]['train_loss']=math.nextafter(changed[-1]['train_loss'],math.inf)
    assert compare_epochs(rows,changed,tasks,last)['status']=='FAIL'


@pytest.mark.parametrize('damage',['duplicate','missing','wrong_epoch','infinite'])
def test_exact_epoch_gate_rejects_invalid_reference_coverage(damage):
    reference=epochs(2)
    if damage=='duplicate':reference[-1]=copy.deepcopy(reference[-2])
    elif damage=='missing':reference.pop()
    elif damage=='wrong_epoch':reference[-1]['epoch']=51
    else:reference[-1]['train_loss']=float('inf')
    # Comparing the same corrupt data must not certify the requested prefix.
    assert compare_epochs(reference,copy.deepcopy(reference),2)['status']=='FAIL'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def norm_summary(rows):
    norms=[r['preclip_norm'] for r in rows if r['preclip_norm'] is not None]
    return dict(n_steps=len(rows),n_clipped=sum(r['clipped'] for r in rows),
        nonfinite_steps=sum(r['nonfinite_norm'] for r in rows),max_preclip_norm=max(norms,default=None))


def control_report(arm,seed):
    failure={'icarl':2008,'icarl_full':2011,'icarl_cbp':6,'icarl_redo':2006}[arm]
    within={'icarl':8,'icarl_full':11,'icarl_cbp':6,'icarl_redo':6}[arm]
    rows=[dict(task=2,step=failure-within+i,task_step=i,preclip_norm=3.,loss=4.,
               nonfinite_norm=False,clipped=False) for i in range(1,within+1)]
    rows[-1].update(preclip_norm=None,nonfinite_norm=True)
    return dict(mode='A',arm=arm,seed=seed,tau=None,tasks=20,epoch=None,code_commit=COMMIT,
        status='FAILED_DIVERGED',rows=rows,
        outcome=dict(status='FAILED_DIVERGED',diverged=dict(task=2,epoch=1,step=failure,tensor='gradient')),
        frozen_K0=512 if arm=='icarl_full' else None,
        intervention_trace=[dict(kind='correction',task=2,step=2001,K=512,K0=512)] if arm=='icarl_full' else [],
        grad_clip=dict(tau=None,run=norm_summary(rows)))


def studies(root):
    for arm,seed in A_CASES:
        for tau in (None,*GRID):
            survive=tau is not None and tau>=5
            rows=[dict(task=2,step=100+i,task_step=i,preclip_norm=3.,loss=4.,nonfinite_norm=False,clipped=tau is not None and tau<3)
                  for i in range(1,61 if survive else 6)]
            report=dict(mode='A',arm=arm,seed=seed,tau=tau,tasks=20,epoch=None,code_commit=COMMIT,
                status='REPLAY_COMPLETED' if survive else 'FAILED_DIVERGED',rows=rows,
                grad_clip=dict(tau=tau,run=norm_summary(rows)))
            if arm=='icarl_full':
                report.update(frozen_K0=512,intervention_trace=control_report(arm,seed)['intervention_trace'])
            if tau is None:report=control_report(arm,seed)
            write(root/'A'/key(arm,seed)/('none' if tau is None else f'tau{tau:g}')/'study.json',report)
    for seed in (0,1):
        report=dict(mode='B',arm='icarl',seed=seed,tau=None,tasks=20,epoch=None,
            code_commit=COMMIT,status='COMPLETED',equivalent_under_clip=True,
            loss_check=dict(status='PASS',epochs_compared=1050,expected_epochs=1050,
                            absolute_tolerance=0.,relative_tolerance=0.,differences=[]),
            grad_clip=dict(tau=None,run=dict(n_steps=1000,n_clipped=0,max_preclip_norm=2.,nonfinite_steps=0)))
        write(root/'B'/key('icarl',seed)/'study.json',report)


def test_selection_requires_all28_A_and_both_B_and_uses_smallest_grid(tmp_path):
    studies(tmp_path)
    assert len(list((tmp_path/'A').rglob('study.json')))==28
    selection=choose_tau(tmp_path,COMMIT)
    assert selection['status']=='SELECTED' and selection['tau']==5 and selection['G_max']==2
    assert len(selection['source_reports'])==30
    proof=tmp_path/'grad_clip_selection.json';write(proof,selection)
    assert load_selection(proof)==selection
    missing=tmp_path/'A'/key(*A_CASES[-1])/'tau50'/'study.json'
    missing.unlink()
    assert choose_tau(tmp_path,COMMIT)['status']=='INCOMPLETE'
    with pytest.raises(ValueError):load_selection(proof)


@pytest.mark.parametrize('damage',['missing_control','paused_B','missing_B'])
def test_selection_waits_for_all_diagnostic_evidence(tmp_path,damage):
    studies(tmp_path)
    if damage=='missing_control':
        (tmp_path/'A'/key(*A_CASES[0])/'none'/'study.json').unlink()
    elif damage=='missing_B':
        (tmp_path/'B'/key('icarl',1)/'study.json').unlink()
    else:
        path=tmp_path/'B'/key('icarl',0)/'study.json';r=json.loads(path.read_text());r['status']='PAUSED';write(path,r)
    assert choose_tau(tmp_path,COMMIT)['status']=='INCOMPLETE'


@pytest.mark.parametrize('damage',['none_norm','nan_norm','inf_norm','missing_norm','nonfinite_flag','duplicate_step'])
def test_A_survival_requires60_unique_finite_observed_optimizer_steps(tmp_path,damage):
    studies(tmp_path)
    for tau in GRID:
        path=tmp_path/'A'/key(*A_CASES[0])/f'tau{tau:g}'/'study.json'
        report=json.loads(path.read_text());report['status']='REPLAY_COMPLETED'
        report['rows']=[dict(task=2,step=100+i,task_step=i,preclip_norm=3.,loss=4.,nonfinite_norm=False,clipped=False) for i in range(1,61)]
        bad=report['rows'][-1]
        if damage=='none_norm':bad['preclip_norm']=None
        elif damage=='nan_norm':bad['preclip_norm']=float('nan')
        elif damage=='inf_norm':bad['preclip_norm']=float('inf')
        elif damage=='missing_norm':bad.pop('preclip_norm')
        elif damage=='nonfinite_flag':bad['nonfinite_norm']=True
        else:bad['step']=report['rows'][-2]['step']
        write(path,report)
    result=choose_tau(tmp_path,COMMIT)
    assert result['status']=='NO_FEASIBLE_TAU' and result['tau'] is None


@pytest.mark.parametrize('damage',['zero_steps','none_max','nan_max','infinite_max','false_loss_gate','wrong_epoch_count','clipped','wrong_commit'])
def test_B_cannot_certify_baseline_from_inconsistent_or_empty_norm_replay(tmp_path,damage):
    studies(tmp_path)
    path=tmp_path/'B'/key('icarl',0)/'study.json';report=json.loads(path.read_text())
    stats=report['grad_clip']['run']
    if damage=='zero_steps':stats['n_steps']=0
    elif damage=='none_max':stats['max_preclip_norm']=None
    elif damage=='nan_max':stats['max_preclip_norm']=float('nan')
    elif damage=='infinite_max':stats['max_preclip_norm']=float('inf')
    elif damage=='false_loss_gate':report['loss_check']['status']='FAIL'
    elif damage=='wrong_epoch_count':report['loss_check']['epochs_compared']=250
    elif damage=='clipped':stats['n_clipped']=1
    else:report['code_commit']='b'*40
    write(path,report)
    # Bad evidence may be rejected explicitly or treated as incomplete, never selected.
    try:result=choose_tau(tmp_path,COMMIT)
    except ValueError:return
    assert result['status']!='SELECTED'


def test_checkpoint_math_identity_ignores_execution_metadata_only():
    identity=dict(arm='icarl_full',seed=2,code_commit='a'*40,source_digest='old',data_dir='/old',
        resume=False,grad_clip=None,cfg={'lr':.05,'device':['0']},
        schedule={'K0':100,'code_commit':'a'*40,'source_digest':'old'},solver='gram',curvature='power10')
    current=copy.deepcopy(identity)
    current.update(code_commit='b'*40,source_digest='new',data_dir='/new',resume=True,grad_clip=5.)
    current['cfg']['device']=['cpu'];current['schedule'].update(code_commit='b'*40,source_digest='new')
    assert mathematical_identity(identity)==mathematical_identity(current)
    for key,value in [('solver','cg'),('curvature','exact'),('seed',1)]:
        changed=copy.deepcopy(current);changed[key]=value
        assert mathematical_identity(identity)!=mathematical_identity(changed)
    changed=copy.deepcopy(current);changed['schedule']['K0']=200
    assert mathematical_identity(identity)!=mathematical_identity(changed)


def test_registered_C_covers_ten_methods_three_seeds():
    assert len(C_CASES)==30 and len(set(C_CASES))==30
    assert {seed for arm,seed in C_CASES}=={0,1,2}


def test_paused_worker_writes_report_and_preserves_exit75(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import diagnostics.grad_clip_study as worker
    from experiments import main_cil, v4_competitors
    from experiments.runners import pycil
    from experiments import vna_provenance
    repo=Path(__file__).resolve().parents[1]
    args=worker.parser().parse_args(['--repo',str(repo),'--support-repo',str(repo),
        '--mode','A','--arm','icarl','--seed','2','--out',str(tmp_path), '--data-dir',str(tmp_path)])
    class Parser:
        def parse_args(self,cli):return SimpleNamespace(out=str(tmp_path))
    def pause(parsed):raise SystemExit(75)
    monkeypatch.setattr(pycil,'upstream_imports',lambda:None)
    monkeypatch.setattr(main_cil,'parser',lambda:Parser())
    monkeypatch.setattr(main_cil,'run',pause)
    monkeypatch.setattr(vna_provenance,'code_commit',lambda:COMMIT)
    original_main=main_cil.MainHook.__init__;original_progress=main_cil.Progress
    with pytest.raises(SystemExit) as caught:worker.run(args)
    assert caught.value.code==75
    report=json.loads((tmp_path/'study.json').read_text())
    assert report['status']=='PAUSED' and report['equivalent_under_clip'] is None
    assert main_cil.MainHook.__init__ is original_main and main_cil.Progress is original_progress


@pytest.mark.parametrize('changed_math',[False,True])
def test_imported_task1_checkpoint_is_transactional_and_checks_math(monkeypatch,tmp_path,changed_math):
    import hashlib
    import torch
    from types import SimpleNamespace
    import diagnostics.grad_clip_study as worker
    from experiments import main_cil,vna_provenance
    from experiments.runners import pycil
    repo=Path(__file__).resolve().parents[1]
    source=dict(arm='icarl',seed=2,code_commit='b'*40,cfg=dict(lr=.05,device=['0']))
    current=dict(source,code_commit=COMMIT,grad_clip=5.)
    current['cfg']=dict(lr=.1 if changed_math else .05,device=['0'])
    checkpoint=tmp_path/'source.pt'
    torch.save(dict(identity=source,completed=1,mid=False,
        hook=dict(grad_clip_state={'old':'must not survive diagnostic reset'}),
        artifacts={'protocol.json':b'old','grad_clip.json':b'old','tasks.json':b'[]'}),checkpoint)
    source_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reference=tmp_path/'reference.json';write(reference,dict(identity=source,summary_sha256='c'*64))
    out=tmp_path/'run'
    args=worker.parser().parse_args(['--repo',str(repo),'--support-repo',str(repo),
        '--mode','A','--arm','icarl','--seed','2','--tau','5','--out',str(out),
        '--data-dir',str(tmp_path),'--reference',str(reference),'--checkpoint',str(checkpoint)])
    class Parser:
        def parse_args(self,cli):return SimpleNamespace(out=str(out))
    seen=[]
    def paused(parsed):
        seen.append(main_cil.Progress(out,current).state)
        raise SystemExit(75)
    monkeypatch.setattr(pycil,'upstream_imports',lambda:None)
    monkeypatch.setattr(main_cil,'parser',lambda:Parser())
    monkeypatch.setattr(main_cil,'run',paused)
    monkeypatch.setattr(vna_provenance,'code_commit',lambda:COMMIT)
    if changed_math:
        with pytest.raises(RuntimeError,match='mathematical identity mismatch'):worker.run(args)
        assert not (out/'latest.pt').exists()
    else:
        with pytest.raises(SystemExit) as caught:worker.run(args)
        assert caught.value.code==75 and seen[0]['identity']==current
        assert 'grad_clip_state' not in seen[0]['hook']
        assert set(seen[0]['artifacts'])=={'tasks.json'}
        imported=json.loads((out/'checkpoint_import.json').read_text())
        assert imported['source_sha256']==source_hash
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==source_hash


@pytest.mark.parametrize('arm,seed',A_CASES)
def test_unclipped_controls_match_registered_failure_coordinates(arm,seed):
    from experiments.clip_study import control_check
    report=control_report(arm,seed)
    checked=control_check(report)
    assert checked['status']=='PASS'
    assert 'expected' in checked and 'observed' in checked


@pytest.mark.parametrize('damage',['survived','wrong_step','wrong_task','wrong_epoch','missing_outcome'])
@pytest.mark.parametrize('arm,seed',A_CASES)
def test_unclipped_controls_reject_unreproduced_failures(arm,seed,damage):
    from experiments.clip_study import control_check
    report=control_report(arm,seed)
    if damage=='survived':report['status']='REPLAY_COMPLETED';report['outcome']=None
    elif damage=='wrong_step':report['outcome']['diverged']['step']+=1
    elif damage=='wrong_task':report['outcome']['diverged']['task']=3
    elif damage=='wrong_epoch':report['outcome']['diverged']['epoch']=2
    else:report.pop('outcome')
    assert control_check(report)['status']=='REPLAY_MISMATCH'


def test_tau_selection_stops_when_an_unclipped_control_does_not_reproduce(tmp_path):
    studies(tmp_path)
    path=tmp_path/'A'/key('icarl',2)/'none'/'study.json'
    report=json.loads(path.read_text());report['outcome']['diverged']['step']=2009;write(path,report)
    result=choose_tau(tmp_path,COMMIT)
    assert result['status']=='REPLAY_MISMATCH' and result.get('tau') is None
    assert result['mismatches']


@pytest.mark.parametrize('damage',['missing','wrong_step','wrong_K0'])
def test_full_control_needs_actual_correction_at2001_with_frozen_K0(damage):
    from experiments.clip_study import control_check
    report=control_report('icarl_full',2)
    if damage=='missing':report['intervention_trace']=[]
    elif damage=='wrong_step':report['intervention_trace'][0]['step']=2002
    else:
        report['intervention_trace'][0]['K0']=100
        report['intervention_trace'][0]['K']=100
    assert control_check(report)['status']=='REPLAY_MISMATCH'


@pytest.mark.parametrize('mode,tau,expected',[
    ('train',5.,'DIVERGED_WITH_CLIP'),('train',None,'FAILED_DIVERGED'),('A',5.,'FAILED_DIVERGED')])
def test_clipped_phaseC_divergence_has_distinct_terminal_status(monkeypatch,tmp_path,mode,tau,expected):
    from types import SimpleNamespace
    import diagnostics.grad_clip_study as worker
    from experiments import main_cil,vna_provenance
    from experiments.runners import pycil
    repo=Path(__file__).resolve().parents[1]
    cli=['--repo',str(repo),'--support-repo',str(repo),'--mode',mode,'--arm','icarl',
         '--seed','2','--out',str(tmp_path),'--data-dir',str(tmp_path)]
    if tau is not None:cli+=['--tau',str(tau)]
    args=worker.parser().parse_args(cli)
    class Parser:
        def parse_args(self,cli):return SimpleNamespace(out=str(tmp_path))
    outcome=dict(status='FAILED_DIVERGED',diverged=dict(task=2,epoch=1,step=2008,tensor='weight'))
    monkeypatch.setattr(pycil,'upstream_imports',lambda:None)
    monkeypatch.setattr(main_cil,'parser',lambda:Parser())
    monkeypatch.setattr(main_cil,'run',lambda parsed:outcome)
    monkeypatch.setattr(vna_provenance,'code_commit',lambda:COMMIT)
    report=worker.run(args)
    assert report['status']==expected and report['outcome']['diverged']==outcome['diverged']
    assert json.loads((tmp_path/'study.json').read_text())['status']==expected


def test_phaseB_stop_signal_is_graceful_before_another_optimizer_step(monkeypatch,tmp_path):
    import hashlib
    from types import SimpleNamespace
    import diagnostics.grad_clip_study as worker
    from experiments import main_cil,vna_provenance
    from experiments.runners import pycil
    repo=Path(__file__).resolve().parents[1]
    stop=tmp_path/'replay_mismatch.json';write(stop,dict(status='REPLAY_MISMATCH'))
    checkpoint=tmp_path/'latest.pt';checkpoint.write_bytes(b'last native safe checkpoint')
    before_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    reference_epochs=tmp_path/'original_epochs.json';write(reference_epochs,epochs())
    args=worker.parser().parse_args(['--repo',str(repo),'--support-repo',str(repo),
        '--mode','B','--arm','icarl','--seed','0','--out',str(tmp_path),'--data-dir',str(tmp_path),
        '--reference-epochs',str(reference_epochs),'--stop-file',str(stop)])
    calls=[]
    class Hook:
        def __init__(self):self.clipper=None;self.intervention=None;self.epochs=[]
        def before_optimizer_step(self,frame):calls.append('must not run')
    class Parser:
        def parse_args(self,cli):return SimpleNamespace(out=str(tmp_path))
    def native(parsed):
        hook=main_cil.MainHook();hook.before_optimizer_step(None)
        raise AssertionError('stop signal was ignored')
    monkeypatch.setattr(pycil,'upstream_imports',lambda:None)
    monkeypatch.setattr(main_cil,'MainHook',Hook)
    monkeypatch.setattr(main_cil,'parser',lambda:Parser())
    monkeypatch.setattr(main_cil,'run',native)
    monkeypatch.setattr(vna_provenance,'code_commit',lambda:COMMIT)
    report=worker.run(args)
    assert report['status']=='STOPPED_REPLAY_MISMATCH' and not calls
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before_hash
    assert json.loads((tmp_path/'study.json').read_text())['equivalent_under_clip'] is False


def test_worker_full_trace_records_actual_correction_and_frozen_K0(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import diagnostics.grad_clip_study as worker
    from experiments import main_cil,vna_provenance
    from experiments.runners import pycil
    repo=Path(__file__).resolve().parents[1]
    source=dict(arm='icarl_full',seed=2,code_commit=COMMIT,
        solver='gram',curvature='power10',gir_live=False,schedule=dict(K0=512,code_commit=COMMIT))
    reference=tmp_path/'reference.json';write(reference,dict(identity=source,summary_sha256='c'*64))
    args=worker.parser().parse_args(['--repo',str(repo),'--support-repo',str(repo),
        '--mode','A','--arm','icarl_full','--seed','2','--out',str(tmp_path),
        '--data-dir',str(tmp_path),'--reference',str(reference)])
    actual=dict(kind='correction',task=2,step=2001,iteration=2001,K=512,K_next=1024,skipped=False)
    class Hook:
        def __init__(self):
            self.clipper=None;self.epochs=[]
            self.intervention=SimpleNamespace(K0=512,events=[dict(kind='correction',task=1,step=1),actual])
    class Parser:
        def parse_args(self,cli):return SimpleNamespace(out=str(tmp_path))
    def native(parsed):
        main_cil.MainHook()
        return control_report('icarl_full',2)['outcome']
    monkeypatch.setattr(pycil,'upstream_imports',lambda:None)
    monkeypatch.setattr(main_cil,'MainHook',Hook)
    monkeypatch.setattr(main_cil,'parser',lambda:Parser())
    monkeypatch.setattr(main_cil,'run',native)
    monkeypatch.setattr(vna_provenance,'code_commit',lambda:COMMIT)
    report=worker.run(args)
    assert report['frozen_K0']==512 and len(report['intervention_trace'])==1
    trace=report['intervention_trace'][0]
    assert (trace['kind'],trace['task'],trace['step'],trace['K'],trace['K0'])==('correction',2,2001,512,512)


def test_tau_selection_requires_full_trace_even_at_largest_grid_threshold(tmp_path):
    studies(tmp_path)
    path=tmp_path/'A'/key('icarl_full',2)/'tau50'/'study.json'
    report=json.loads(path.read_text());report['intervention_trace']=[];write(path,report)
    result=choose_tau(tmp_path,COMMIT)
    assert result['status']=='REPLAY_MISMATCH' and result.get('tau') is None
